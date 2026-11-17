"""The native step executor of a Qwen3.5/3.8 model (``resin._resin.QwenExecutor``), built at startup.

Python's part ends here. This module gives every step mode a table image layout (``resin._resin.QwenFill``,
the one source of the offsets), one device table image and two pinned host copies of it, the pinned staging
sets of the outputs, and one step program per mode (``resin._resin.GaudiProgram``) whose launches bind their
tables inside the image. With a drafter (DFlash 2 or DSpine) it also gives the block steps an image of their own
(``resin._resin.QwenDraftTables``) and one program per (rows, verify width). With the MTP layer it gives the
commit and draft chain after each verify step an image of their own (``QwenFill.mtp_chain``). The executor then
runs each step in Rust: it fills the host copies, and the programs upload them in one copy each, make the
launches and copy the outputs out behind events.

A mode's program has these sections, run in the order the step needs:

- ``target``: the image upload (and an event when the host copy is free again), the embedding, every layer
  (a full-attention layer's slot launches in a ``take`` of the step's launch count; the drafter's taps after
  its tap layers), the head and sampler;
- ``inject`` (a drafter): the step's columns into the drafter's cache;
- ``fix`` (a verify mode): ``spec_fix``, the acceptance on the device and the tables after it;
- ``out``: the sampled tokens (and a verify step's accepted counts) to the staging set; ``scores``: the logprobs
  and top lists; ``record``: the staging set's event 0;
- ``commit`` (a verify mode): the accepted tokens' GDN updates;
- ``next`` and ``record1`` (a drafter): the block step's tokens to the staging set, and its event 1;
- with the MTP layer: ``mkeys`` (the MTP layer's keys at a decode step's or a prefill chunk's columns),
  ``mhidden`` (a prefill chunk's last residual into its STATE block), ``mdraft`` (the first draft after a
  prompt), ``mfull`` (the MTP layer at every column of a verify step), ``mout`` (its drafts to the staging set),
  ``mcommit`` (the chain image, the accepted tokens' GDN updates and the next step's first tokens), ``mchain``
  (the draft chain: drafts 2 .. K) and ``record1`` (the staging set's event 1).

A block step's program has ``upload`` (its image) and ``run`` (embedding, per drafter layer pre / slot
launches / post, head). Prompt scoring has a program of its own (``_Programs.scoring``). The runner builds this
executor unless the ``oldqwen`` feature is on (``resin_models_gaudi.features``); with it the steps run on the
Python executor.
"""

from __future__ import annotations

import numpy as np

from resin_models_gaudi.common.layers import SLOT_KEYS
from resin_models_gaudi.common.native import bind, block_program, block_tables
from resin_models_gaudi.common.sampling import TOP_MAX
from resin_models_gaudi.qwen3_5.cache import (
    DRAFT_GROUP,
    KV_GROUP,
    MTP_GROUP,
    ROW_BYTES,
    STATE_GROUP,
    STATE_TOKENS,
)
from resin_models_gaudi.qwen3_5.layers import SCORE_PAIRS, SCORE_ROWS
from resin_models_gaudi.qwen3_5.model import SCORE_TOPS
from resin_models_gaudi.synapse import runtime

SMALL = 256  # bytes of each small table in the image (qwen3_5.model.SMALL)
FIX = (
    "fix_cb0",
    "fix_cb1",
    "fix_rows",
    "commit_bases",
    "commit_counts",
    "hidden_table",
    "fix_patch",
    "fix_meta",
)  # spec_fix's tables, in resin_gaudi::qwen_draft::FixLayout's order


def _round(nbytes: int) -> int:
    return max(SMALL, -(-nbytes // SMALL) * SMALL)


def build_executor(
    model,
    *,
    max_batch_size: int,
    max_seq_len: int,
    lookahead: int,
    policy: str,
):
    """The native executor of ``model`` (a ``Qwen35Gaudi`` with its cache attached): it runs every step."""
    if model.device.glue is None:
        raise RuntimeError(
            "Qwen3.5/3.8 steps run in the native executor: it needs the runtime library's glue (Device.glue)"
        )
    try:
        from resin._resin import GaudiProgram, QwenDraftTables, QwenExecutor, QwenFill
    except ImportError as error:
        raise RuntimeError(
            "Qwen3.5/3.8 steps run in the native executor: resin must be built with the gaudi feature"
        ) from error
    dev, d, g = model.device, model.dims, model.geometry
    drafter, cfg = model.drafter, model.dfl_cfg
    fill = QwenFill(
        gdn_layers=len(model.gdn),
        kv_groups=d.kv_groups,
        hpg=d.hpg,
        state_layer_rows=g.state_layer_rows,
        ssm_rows=g.ssm_rows,
        max_positions=model.max_positions,
        kv8=g.kv8,
    )
    tables = None
    if drafter is not None:
        tables = QwenDraftTables(
            layers=cfg.layers,
            pairs=cfg.pairs,
            layer_rows=cfg.layer_rows,
            block=cfg.block,
            window=cfg.window,
            causal=cfg.causal,
            mask_token=cfg.mask_token,
            scratch_rows=int(drafter.scratch[0, 0, 0, 0]),
            max_rows=model.spec_buckets[-1],
        )
    chunks = -(-model.max_positions // SLOT_KEYS)

    def max_launches(m, groups: int = d.kv_groups) -> int:
        """the slot launches of a step of mode ``m`` at the longest context, of ``groups`` KV groups"""
        return -(
            -(m.columns // m.per_row)
            * -(-m.per_row // m.query_block)
            * groups
            * chunks
            // m.slots
        )

    def mtp_launches(m) -> int:
        """the slot launches of an MTP pass of mode ``m``: the whole MTP layer's KV groups"""
        return max_launches(m, model.mtp_dims.kv_groups)

    layouts, extras = {}, {}
    one = model.modes[f"d{model.decode_buckets[0]}"]  # a prompt's first MTP draft runs one row's decode mode
    for m in model.modes.values():
        if m.full:  # the slots without a mask, then at most two key chunks a query block with one
            blocks = (m.columns // m.per_row) * -(-m.per_row // m.query_block)
            launches = -(-2 * blocks * d.kv_groups // m.slots)
            wide = -(-max_launches(m) * m.slots // m.wide)
        elif m.wide:  # wide launches, then fewer narrow ones than make one wide launch
            launches = m.wide // m.slots - 1
            wide = -(-max_launches(m) * m.slots // m.wide)
        else:
            launches, wide = max_launches(m), 0
        sizes = [] if tables is None else tables.inject_sizes(m.columns)
        verify = m.name[0] == "v"
        if verify and drafter is not None:
            sizes += QwenDraftTables.fix_sizes(
                m.columns // m.per_row, len(model.gdn), model.fix_patch
            )
        mtp_tables: list[tuple[str, int]] = []  # (name, tables) of the MTP layer's tables, after the drafter's
        if model.mtp:
            mlaunches = mtp_launches(m)
            if verify:  # the MTP layer at every column, with its attention and head
                mtp_tables = [("mfull", 6)]
                sizes += QwenFill.mtp_pass_sizes(m.columns, m.slots, mlaunches, True)
            else:  # its keys; a prefill chunk's hidden and the prompt's first draft
                mtp_tables = [("mkeys", 6)]
                sizes += QwenFill.mtp_pass_sizes(m.columns, m.slots, mlaunches, False)
                if m.name[0] == "p":
                    mtp_tables += [("mhidden", 1), ("mdraft", 6)]
                    sizes += [8]
                    sizes += QwenFill.mtp_pass_sizes(
                        one.columns, one.slots, mtp_launches(one), True
                    )
        layout = fill.add_mode(
            m.name,
            columns=m.columns,
            per_row=m.per_row,
            query_block=m.query_block,
            slots=m.slots,
            prefill=m.name[0] == "p",
            max_launches=launches,
            extras=sizes,
            wide_slots=m.wide,
            max_wide=wide,
            wide_full=m.full,
        )
        layouts[m.name] = layout
        if tables is not None:
            pos, kv = layout["extras"][:2]
            extras[m.name] = {"inject": (pos, kv, _round(4 * 8 * m.columns))}
            if verify:
                extras[m.name]["fix"] = (model.fix_patch, layout["extras"][2:])
        if mtp_tables:
            offsets, at = layout["extras"], 0
            extras[m.name] = {"mlaunches": mtp_launches(m)}
            for key, count in mtp_tables:
                extras[m.name][key] = offsets[at] if count == 1 else offsets[at : at + count]
                at += count
    stride = _round(max(layout["nbytes"] for layout in layouts.values()))
    image = dev.malloc(stride)
    host = dev.host_buffer(2 * stride, np.uint8)
    # events: host image 0 / 1 free again, block-step image 0 / 1 free again, staging set k's event e at 4 + k
    # + 2 e
    events = dev.event_slots(8)
    T = model.columns
    rows = model.spec_buckets[-1] if model.K else 1
    block = cfg.block if cfg is not None else 1
    staging = {
        "tokens": dev.host_buffer(2 * T * 4, np.uint8),
        "lp": dev.host_buffer(2 * T * 4, np.uint8),
        "top_idx": dev.host_buffer(2 * T * TOP_MAX * 4, np.uint8),
        "top_lp": dev.host_buffer(2 * T * TOP_MAX * 4, np.uint8),
        "accepted": None,
        "next": None,
        "columns": T,
        "rows": rows,
        "block": block,
        "events": events + 4,
    }
    blocks = None
    if drafter is not None:
        staging["accepted"] = dev.host_buffer(2 * rows * 4, np.uint8)
        staging["next"] = dev.host_buffer(2 * rows * block * 4, np.uint8)
        block_layouts = {B: tables.layout(B) for B in model.spec_buckets}
        block_stride = _round(
            max(layout["nbytes"] for layout in block_layouts.values())
        )
        blocks = {
            "layouts": block_layouts,
            "stride": block_stride,
            "image": dev.malloc(block_stride),
            "host": dev.host_buffer(2 * block_stride, np.uint8),
            "events": events + 2,
        }
    chain = None
    if model.mtp:  # the commit and draft chain after each verify step: an image of their own
        width = model.K + 1
        chain_layouts = {}
        for B in model.spec_buckets:
            m = model.modes[f"d{B}"]
            chain_layouts[B] = QwenFill.mtp_chain(
                B, m.slots, mtp_launches(m), width, len(model.gdn)
            )
        chain_stride = _round(max(c["nbytes"] for c in chain_layouts.values()))
        chain = {
            "layouts": chain_layouts,
            "stride": chain_stride,
            "image": dev.malloc(chain_stride),
            "host": dev.host_buffer(2 * chain_stride, np.uint8),
            "events": events + 2,
            "tokens": dev.host_buffer(2 * T * 4, np.uint8),
            "drafts": dev.host_buffer(2 * model.K * rows * 4, np.uint8),
        }
    builder = _Programs(
        model,
        QwenExecutor,
        GaudiProgram,
        image,
        host,
        stride,
        events,
        staging,
        blocks,
        chain,
        {m.name: max_launches(m) for m in model.modes.values()},
        {m.name: mtp_launches(m) for m in model.modes.values()} if model.mtp else {},
    )
    programs = {
        m.name: builder.program(m, layouts[m.name], extras.get(m.name))
        for m in model.modes.values()
    }
    draft = None
    if drafter is not None:
        assert blocks is not None
        widths = tuple(model.verify_widths)
        draft = {
            "layers": cfg.layers,
            "pairs": cfg.pairs,
            "layer_rows": cfg.layer_rows,
            "block": cfg.block,
            "window": cfg.window,
            "causal": cfg.causal,
            "mask_token": cfg.mask_token,
            "scratch_rows": int(drafter.scratch[0, 0, 0, 0]),
            "max_rows": model.spec_buckets[-1],
            "k": model.K,
            "widths": list(widths),
            "programs": {
                (B, w): builder.block_program(B, w)
                for B in model.spec_buckets
                for w in widths
            },
            "image": blocks["image"],
            "host": blocks["host"],
            "host_stride": blocks["stride"],
            "image_events": blocks["events"],
            "next": drafter.next,
            "group": DRAFT_GROUP,
            "draft_block_rows": g.draft_block_rows,
            "state": (
                g.state_layer_rows,
                g.ssm_rows,
                g.linear_layers * g.state_layer_rows,
            ),
            "policy": policy,
        }
    scoring = builder.scoring()
    mtp = None
    if chain is not None:
        mtp = {
            "k": model.K,
            # the MTP layer's attention: all its KV groups on every rank
            "attention": (
                model.mtp_dims.kv_groups,
                model.mtp_dims.hpg,
                model.max_positions,
            ),
            "kv8": g.kv8,
            "state": (
                g.state_layer_rows,
                g.ssm_rows,
                g.linear_layers * g.state_layer_rows,
            ),
            "group": MTP_GROUP,
            "block_rows": g.mtp_layer_rows,
            "zero": model.mtp_buf["mtp_zero"],
            "arena": model.arena,
            "row_bytes": ROW_BYTES,
            "next_tok": model.spec_buf["next_tok"],
            "image": chain["image"],
            "host": chain["host"],
            "host_stride": chain["stride"],
            "image_events": chain["events"],
            "buckets": list(model.spec_buckets),
            "tokens": chain["tokens"],
            "drafts": chain["drafts"],
            "policy": policy,
        }
    return QwenExecutor(
        glue=dev.glue,
        fill=fill,
        programs=programs,
        extras=extras,
        prefill_buckets=list(model.buckets),
        decode_buckets=list(model.decode_buckets),
        image=image,
        host=host,
        host_stride=stride,
        image_events=events,
        staging=staging,
        tok_out=model.buf["tok_out"],
        spin_ns=int(runtime._SPIN_WAIT * 1e9),
        vocab=d.vocab,
        cache=(
            g.slab_rows,
            g.kv_block_rows,
            g.state_block_rows,
            STATE_TOKENS,
            KV_GROUP,
            STATE_GROUP,
        ),
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        lookahead=lookahead,
        draft=draft,
        mtp=mtp,
        scoring=scoring,
    )


class _Programs:
    """The step programs of a model's modes and block steps, over its compiled recipes and bindings."""

    def __init__(
        self,
        model,
        executor,
        program_type,
        image,
        host,
        stride,
        events,
        staging,
        blocks,
        chain,
        launches: dict[str, int],
        mtp_launches: dict[str, int],
    ):
        self.m, self.E, self.Program = model, executor, program_type
        self.image, self.host, self.stride = image, host.ctypes.data, stride
        self.events, self.staging, self.blocks = events, staging, blocks
        self.chain, self.launches, self.mtp_launches = chain, launches, mtp_launches

    _bind = staticmethod(bind)

    def _launch(
        self,
        program,
        kind: str,
        mode: str,
        layer: int,
        tables: dict,
        dynamic=None,
        variant: str = "",
    ) -> None:
        """One launch of the model's ``(kind, mode)`` recipe as ``layer`` (``variant``: ``Qwen35Gaudi._bind_table``'s),
        with the step's tables in the image."""
        recipe = self.m.recipe[(kind, mode)]
        b = self.m.binds[(kind, mode, layer, variant)]
        if "tok" in recipe.persistent and kind != "embed":
            raise ValueError(f"{kind} reads the step's tokens; only the embedding may")
        dynamic = dict(dynamic or {})
        if (
            kind == "embed"
        ):  # the columns' tokens: the image's, or another buffer's (a register)
            dynamic["tok"] = (0, self.E.R_TOK, 1)
        self._bind(program, recipe, b.slots, b.addresses, tables, dynamic)

    def _block_tables(self, bucket: int) -> dict:
        """The device addresses of a block step's tables of ``bucket`` rows."""
        return block_tables(self.blocks, bucket)

    def program(self, mode, layout: dict, extra: dict | None):
        m, E, image = self.m, self.E, self.image
        name, prefill, verify = mode.name, mode.name[0] == "p", mode.name[0] == "v"
        drafter = m.drafter
        program = self.Program(E.REGISTERS)
        columns = {
            key: image + layout[key]
            for key in ("tok", "pos", "meta", "last", "samp_params", "kv_rows")
        }
        # ---- target
        program.section("target")
        program.upload(image, (self.host, E.R_IMAGE, self.stride), (0, E.R_BYTES, 1))
        program.select([(E.R_IMAGE, 2)], [0, 1])
        for k in range(2):
            program.arm()
            program.record(self.events + k)
        program.end()
        self._launch(program, "embed", name, -1, columns)
        taps = drafter.tap_steps(name) if drafter is not None else {}
        launches, wide = layout["launches"], layout["wide_launches"]
        split, reduce = m.split, m.nranks > 1
        count = m.dims.hidden * mode.columns  # f32 elements of a partial
        gathered = m.gathered(name) > 1

        def collect(part: int) -> None:
            """A partial over the ranks: all-gathered (the next graph sums it), or all-reduced in place."""
            if gathered:
                program.allgather(part, m.gathered_part(part), count)
            else:
                program.allreduce(part, count)


        def tap(layer: int) -> None:  # the drafter's context projection of the layer's output
            if layer in taps:
                recipe, slots, addresses = taps[layer]
                self._bind(program, recipe, slots, addresses, {}, {})

        for i in range(m.n_layers):
            if i in m.gdn:
                j = m.gdn.index(i)
                base = image + layout["state"] + j * layout["state_stride"]
                if prefill:
                    state = {
                        key: base + k * SMALL
                        for k, key in enumerate(
                            ("conv_src", "ssm_src", "conv_dst", "ssm_dst")
                        )
                    }
                else:
                    state = {"gdn_bases": base, "conv_dst": base + layout["conv_dst"]}
                kind = "gdn0" if split and i == 0 else "gdn"
                self._launch(program, kind, name, i, {**columns, **state})
            else:
                self._launch(program, "pre", name, i, columns)
                if wide:  # the step's first slots, in wide launches
                    program.take(E.R_WLAUNCHES)
                    for n in range(wide):
                        at = image + layout["wide"] + n * layout["wide_stride"]
                        slot_tables = {
                            "slots": at,
                            "k_pages": at + layout["wide_k_pages"],
                            "v_pages": at + layout["wide_v_pages"],
                        }
                        self._launch(
                            program, "wslots", name, i, {**columns, **slot_tables}
                        )
                    program.end()
                program.take(E.R_LAUNCHES)
                for n in range(launches):
                    at = image + layout["slots"] + n * layout["launch_stride"]
                    slot_tables = {
                        "slots": at,
                        "k_pages": at + layout["k_pages"],
                        "v_pages": at + layout["v_pages"],
                    }
                    self._launch(program, "slots", name, i, {**columns, **slot_tables})
                program.end()
                self._launch(program, "post", name, i, columns)
            if not split:
                tap(i)
                continue
            # a split layer: the mixer's partial summed over the ranks, then the MLP and its partial; layer
            # i - 1's output is complete in the mixer's carry out
            dump = getattr(m, "dump", None) if not prefill and mode.columns <= 8 else None
            slot = m.dims.hidden * 8 * 4
            if dump is not None:  # this rank's partial, before the all-reduce
                program.copy(dump + (6 * i + 4) * slot, m.part(i, False), count * 4)
            if reduce:
                collect(m.part(i, False))
            if dump is not None:
                program.copy(dump + (6 * i) * slot, m.carry[1]["x"], count * 4)
                program.copy(dump + (6 * i + 1) * slot, m.part(i, False), count * 4)
            tap(i - 1)
            self._launch(program, "mlp", name, i, columns)
            if dump is not None:
                program.copy(dump + (6 * i + 5) * slot, m.part(i, True), count * 4)
            if reduce:
                collect(m.part(i, True))
            if dump is not None:
                program.copy(dump + (6 * i + 2) * slot, m.carry[0]["x"], count * 4)
                program.copy(dump + (6 * i + 3) * slot, m.part(i, True), count * 4)
        self._launch(program, "head", name, -1, columns)
        if m.mtp:
            assert extra is not None
            self._mtp_sections(program, mode, layout, extra)
        if drafter is not None:
            assert extra is not None
            # ---- inject: the step's columns into the drafter's cache
            program.section("inject")
            pos, kv, kv_stride = extra["inject"]
            recipe, slots, addresses = drafter._step(("dfl_inject", name))
            inject = {"pos": image + pos}
            inject.update(
                {
                    f"kv_rows{layer}": image + kv + layer * kv_stride
                    for layer in range(m.dfl_cfg.layers)
                }
            )
            self._bind(program, recipe, slots, addresses, inject, {})
        fix = {}
        if verify and drafter is not None:
            # ---- fix: spec_fix's acceptance and the tables after it
            assert extra is not None
            program.section("fix")
            fix = {
                key: image + offset
                for key, offset in zip(FIX, extra["fix"][1], strict=True)
            }
            bucket = mode.columns // mode.per_row
            block = self._block_tables(bucket)
            fix.update(
                btok=block["tok"],
                bpos=block["pos"],
                banc=block["anchors"],
                dtables=block["tables"],
            )
            recipe = m.recipe[("spec_fix", name)]
            b = m.binds[("spec_fix", name, -1, "")]
            self._bind(
                program,
                recipe,
                b.slots,
                b.addresses,
                fix,
                {"fix_tin": (0, E.R_FIXTIN, 1)},
            )
        # ---- outputs
        T, buf, s = m.columns, m.buf, self.staging
        program.section("out")
        program.download(
            (s["tokens"].ctypes.data, E.R_STAGE, 4 * T),
            buf["tok_out"],
            (0, E.R_ROWS4, 1),
        )
        if verify and drafter is not None:
            program.download(
                (s["accepted"].ctypes.data, E.R_STAGE, 4 * s["rows"]),
                m.spec_buf["fix_acc"],
                (0, E.R_ACC, 1),
            )
        if m.mtp and mode.name[0] in "pv":  # the MTP layer's drafts: a verify step's columns, a prompt's first
            program.section("mout")
            program.download(
                (self.chain["tokens"].ctypes.data, E.R_STAGE, 4 * T),
                m.mtp_buf["mtp_tok_out"],
                (0, E.R_ROWS4, 1),
            )
        program.section("scores")
        program.download(
            (s["lp"].ctypes.data, E.R_STAGE, 4 * T), buf["lp_out"], (0, E.R_ROWS4, 1)
        )
        for key in ("top_idx", "top_lp"):
            program.download(
                (s[key].ctypes.data, E.R_STAGE, 4 * T * TOP_MAX),
                buf[key],
                (0, E.R_TOP, 1),
            )
        self._record(program, "record", 0)
        if verify and m.mtp:
            self._chain_sections(program, mode)
            self._record(program, "record1", 1)
        if verify and drafter is not None:
            program.section("commit")
            recipe = m.recipe[("commit", name)]
            b = m.binds[("commit", name, -1, "")]
            self._bind(program, recipe, b.slots, b.addresses, fix, {})
        if drafter is not None:
            program.section("next")
            program.download(
                (s["next"].ctypes.data, E.R_STAGE, 4 * s["rows"] * s["block"]),
                drafter.next,
                (0, E.R_NEXT, 1),
            )
            self._record(program, "record1", 1)
        program.finish()
        return program

    def _mtp_pass(
        self,
        program,
        mode: str,
        variant: str,
        at: dict,
        *,
        full: bool,
        prev,
        tok,
        meta: int,
        launches: int = 0,
    ) -> None:
        """An MTP pass of ``mode`` (``Qwen35Gaudi._mtp_pass``): the MTP layer's input (``variant`` "mtp": from the
        target's residual, "mtpc": from its own), its attention's pre launch (it stores the keys), and for a
        ``full`` pass the slot launches (a ``take`` of register ``launches``), post and the MTP head. ``at``: the
        addresses of the pass's tables; ``prev`` and ``tok``: the hidden before the first position and the tokens
        (an address, or an operand of a register); ``meta``: the attention's metadata (the step's)."""
        m, M = self.m, self.m.mtp_layer
        tables, dynamic = {"mtp_cols": at["cols"]}, {}
        for name, operand in (("mtp_prev", prev), ("mtp_tok", tok)):
            if isinstance(operand, tuple):
                dynamic[name] = operand
            else:
                tables[name] = operand
        self._launch(program, "mtp_in", mode, M, tables, dynamic, variant=variant)
        rows = {"pos": at["pos"], "kv_rows": at["kv_rows"], "meta": meta}
        self._launch(program, m.mtp_kind("pre"), mode, M, rows, variant="mtp")
        if not full:
            return
        program.take(launches)
        for n in range(self.mtp_launches[mode]):
            base = at["slots"] + n * at["stride"]
            slot_tables = {
                "slots": base,
                "k_pages": base + at["k_pages"],
                "v_pages": base + at["v_pages"],
            }
            self._launch(
                program,
                m.mtp_kind("slots"),
                mode,
                M,
                {**rows, **slot_tables},
                variant="mtp",
            )
        program.end()
        self._launch(program, m.mtp_kind("post"), mode, M, rows, variant="mtp")
        self._launch(
            program,
            m.mtp_kind("head"),
            mode,
            -1,
            {"samp_params": at["params"]},
            variant="mtp",
        )

    def _pass_tables(self, base: int, offsets, slots: int) -> dict:
        """The addresses of an MTP pass's tables at ``offsets`` (``QwenFill.mtp_pass_sizes`` order) from ``base``."""
        at = {
            key: base + offset
            for key, offset in zip(
                ("cols", "tok", "pos", "kv_rows", "params", "slots"), offsets, strict=True
            )
        }
        from resin._resin import QwenFill

        at["k_pages"], at["v_pages"], at["stride"] = QwenFill.launch_tables(slots)
        return at

    def _mtp_sections(self, program, mode, layout: dict, extra: dict) -> None:
        """A mode's MTP layer sections: ``mkeys`` (a decode step's or a prefill chunk's keys), ``mhidden`` (a prefill
        chunk's last hidden into its STATE block), ``mdraft`` (the first draft after a prompt, a pass of one
        row's decode mode), ``mfull`` (a verify step's MTP layer at every column)."""
        m, E, image = self.m, self.E, self.image
        name = mode.name
        meta = image + layout["meta"]
        zero, tok_out = m.mtp_buf["mtp_zero"], m.buf["tok_out"]
        if name[0] == "v":
            program.section("mfull")
            at = self._pass_tables(image, extra["mfull"], mode.slots)
            self._mtp_pass(
                program,
                name,
                "mtp",
                at,
                full=True,
                prev=zero,
                tok=tok_out,
                meta=meta,
                launches=E.R_MLAUNCHES,
            )
            return
        program.section("mkeys")
        at = self._pass_tables(image, extra["mkeys"], mode.slots)
        if name[0] == "p":  # the chunk's tokens from the image, the hidden before it from a register
            prev, tok = (0, E.R_MPREV, 1), at["tok"]
        else:  # each row's sampled token and the target's hidden
            prev, tok = zero, tok_out
        self._mtp_pass(program, name, "mtp", at, full=False, prev=prev, tok=tok, meta=meta)
        if name[0] != "p":
            return
        program.section("mhidden")
        recipe = m.recipe[("hidden", "x")]
        b = m.binds[("hidden", "x", -1, "")]
        hidden = {"hidden_table": image + extra["mhidden"]}
        self._bind(program, recipe, b.slots, b.addresses, hidden, {})
        program.section("mdraft")
        one = m.modes[f"d{m.decode_buckets[0]}"]
        at = self._pass_tables(image, extra["mdraft"], one.slots)
        self._mtp_pass(
            program,
            one.name,
            "mtp",
            at,
            full=True,
            prev=zero,
            tok=tok_out,
            meta=meta,
            launches=E.R_MLAUNCHES,
        )

    def _chain_sections(self, program, mode) -> None:
        """A verify mode's ``mcommit`` (the chain image's upload, the commit, and the next step's [anchor, first
        draft] tokens) and ``mchain`` (``Qwen35Gaudi._chain``: links 1 .. K - 1 of one row's MTP layer each,
        their drafts into the staging set and the next step's tokens)."""
        m, E, chain = self.m, self.E, self.chain
        B, width = mode.columns // mode.per_row, mode.per_row
        layout, cimage = chain["layouts"][B], chain["image"]
        program.section("mcommit")
        program.upload(
            cimage,
            (chain["host"].ctypes.data, E.R_CIMAGE, chain["stride"]),
            (0, E.R_CBYTES, 1),
        )
        program.select([(E.R_CIMAGE, 2)], [0, 1])
        for k in range(2):
            program.arm()
            program.record(chain["events"] + k)
        program.end()
        recipe = m.recipe[("commit", mode.name)]
        b = m.binds[("commit", mode.name, -1, "")]
        commit = {
            key: cimage + layout[key]
            for key in ("commit_bases", "commit_counts", "hidden_table")
        }
        self._bind(program, recipe, b.slots, b.addresses, commit, {})
        next_tok = m.spec_buf["next_tok"]
        program.copy(next_tok, cimage + layout["following"], 4 * B * width)
        program.section("mchain")
        decode = m.modes[f"d{B}"]
        tok_out = m.mtp_buf["mtp_tok_out"]
        rows, K = self.staging["rows"], m.K
        for j, link in enumerate(layout["links"], start=1):
            at = {
                key: cimage + link[key]
                for key in ("cols", "tok", "pos", "kv_rows", "params", "slots")
            }
            at.update(stride=link["stride"], k_pages=link["k_pages"], v_pages=link["v_pages"])
            self._mtp_pass(
                program,
                decode.name,
                "mtpc",
                at,
                full=True,
                prev=m.mtp_buf["mtp_zero"],
                tok=at["tok"] if j == 1 else tok_out,
                meta=cimage + layout["meta"],
                launches=E.R_LINK + j - 1,
            )
            program.download(
                (chain["drafts"].ctypes.data + 4 * j * rows, E.R_STAGE, 4 * K * rows),
                tok_out,
                4 * B,
            )
            # each real row's draft into column j + 1 of its next tokens
            program.select([(E.R_ROWS, B + 1)], [-1, *range(B)])
            for n in range(1, B + 1):
                program.arm()
                program.copy_multi(
                    [tok_out + 4 * r for r in range(n)],
                    [4] * n,
                    [next_tok + 4 * (r * width + j + 1) for r in range(n)],
                )
            program.end()

    def scoring(self) -> dict:
        """The prompt scoring program (``Qwen35Gaudi._score``): a section ``k{K}`` per top-list length of the score
        recipes, each uploading the tables (a chunk's positions, then its (row, token) pairs), making the score
        launch and copying the outputs to pinned buffers behind one event."""
        m, dev = self.m, self.m.device
        rows_bytes = _round(4 * SCORE_ROWS)
        nbytes = rows_bytes + _round(8 * SCORE_PAIRS)
        image = dev.malloc(nbytes)
        host = dev.host_buffer(nbytes, np.uint8)
        top = 4 * SCORE_TOPS[-1] * SCORE_ROWS
        out = {
            "lp": dev.host_buffer(4 * SCORE_PAIRS, np.uint8),
            "stats": dev.host_buffer(4 * 16 * SCORE_ROWS, np.uint8),
            "top_v": dev.host_buffer(top, np.uint8),
            "top_i": dev.host_buffer(top, np.uint8),
        }
        event = dev.event_slots(1)
        program = self.Program(self.E.REGISTERS)
        buf = m.buf
        for K in SCORE_TOPS:
            program.section(f"k{K}")
            program.upload(image, host.ctypes.data, nbytes)
            recipe = m.recipe[("score", f"k{K}")]
            b = m.binds[("score", f"k{K}", -1, "")]
            tables = {"score_rows": image, "score_pairs": image + rows_bytes}
            self._bind(program, recipe, b.slots, b.addresses, tables, {})
            program.download(out["lp"].ctypes.data, buf["score_lp"], 4 * SCORE_PAIRS)
            if K:
                program.download(out["stats"].ctypes.data, buf["score_stats"], 4 * 16 * SCORE_ROWS)
                for key in ("top_v", "top_i"):
                    program.download(out[key].ctypes.data, buf[f"score_{key}"], 4 * K * SCORE_ROWS)
            program.record(event)
        program.finish()
        return {"program": program, "host": host, "pairs_at": rows_bytes, "event": event, **out}

    def _record(self, program, section: str, e: int) -> None:
        """A section recording staging set k's event ``e`` (k from R_STAGE)."""
        program.section(section)
        program.select([(self.E.R_STAGE, 2)], [0, 1])
        for k in range(2):
            program.arm()
            program.record(self.staging["events"] + k + 2 * e)
        program.end()

    def block_program(self, bucket: int, width: int):
        """The block step of ``bucket`` rows for a verify of ``width`` columns."""
        m = self.m
        return block_program(self.Program, self.E, m.drafter, m.dfl_cfg, self.blocks, bucket, width)
