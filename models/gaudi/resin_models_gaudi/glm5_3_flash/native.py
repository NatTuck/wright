"""GLM-5.3-Flash's native step executor (``resin._resin.GlmExecutor``), built at startup.

The model's step programs (``Glm53Gaudi._native_program``: the embedding, each layer's launches and collectives,
the head and the logits all-gather) are built for every (bucket, indexer, sparse) shape. Here each bucket also
gets an I/O program whose sections the executor runs around them, with the executor's registers
(``GlmExecutor.R_*``):

- ``upload``: the header, row and KDA tables of host copy ``R_IMAGE`` of the table image; ``upload_blocks`` (decode
  buckets) its sparse block table; ``upload_slots`` / ``upload_index`` its first ``R_SLOT_BYTES`` / ``R_INDEX_BYTES``
  of slot and index records (``_upload``); ``uploaded``: that copy's event (the copy is free again when it is done);
- ``sample`` (decode buckets): the sampler's table (ring upload) and the sampler over every column; ``sample_last``
  (prefill buckets): over the chunk's last column, the head's first (``sample_launched``);
- ``out``: the sampler's outputs to staging set ``R_STAGE``, then its event;
- ``head_scored`` (prefill buckets): the head columns' table (ring upload), the chunk's head at those columns and the
  logits all-gather; ``scores``: every rank's logits of them to the score buffer, then its event (``head_logits``).

The executor owns its pinned host copies of the table image, its staging sets, the sampler table and the score
buffers; the device's table buffer, sampler buffers and event slots are the model's.
"""

from __future__ import annotations

import numpy as np

from resin_models_gaudi.common.native import block_program
from resin_models_gaudi.glm5_3_flash.cache import ROW_BYTES, STATE_TOKENS
from resin_models_gaudi.glm5_3_flash.model import (
    _SAMPLE_LAST,
    _SAMPLE_TABLE,
    _SAMPLED,
    _SAMPLED_BYTES,
    ALL_BUCKETS,
    DECODE_BUCKETS,
    HEAD_COLS,
    MAX_T,
    MTP_T,
    PREFILL_BUCKETS,
    TMAX,
)
from resin_models_gaudi.synapse import runtime


def _round(nbytes: int, unit: int = 4096) -> int:
    return -(-nbytes // unit) * unit


def available(runner) -> bool:
    """Whether ``runner``'s model can run on the native executor: the extension has it, the device has the glue,
    and the model has the native fill."""
    try:
        from resin._resin import GlmExecutor  # noqa: F401
    except ImportError:
        return False
    m = runner.model
    return m.device.glue is not None and m._native is not None


def build_executor(runner):
    """The native executor of ``runner`` (a ``Glm53FlashGaudiRunner`` with its cache attached): it runs every
    step."""
    from resin._resin import GaudiProgram, GlmExecutor
    from resin_models_gaudi.glm5_3_flash.runner import step_cost_terms

    E, m = GlmExecutor, runner.model
    GlmExecutor.trace_rank(m.rank)
    dev = m.device
    stride = _round(max(tb.nbytes for tb in m.tables.values()))
    host = dev.host_buffer(2 * stride, np.uint8)
    sstride = _round(_SAMPLED_BYTES, 256)
    staging = dev.host_buffer(2 * sstride, np.uint8)
    sample = dev.host_buffer(_SAMPLE_TABLE, np.uint8)
    host_events, stage_events = m._host_event, m._stage_event
    # prompt logprobs: the head columns' table, and every rank's logits of them
    V = m.d.vocab
    sc_stride = _round(4 * HEAD_COLS * V)
    scoring = {
        "columns": dev.host_buffer(256, np.uint8),
        "host": dev.host_buffer(m.nranks * sc_stride, np.uint8),
        "stride": sc_stride,
        "vocab": V,
        "ranks": m.nranks,
        "event": dev.event_slots(1),
    }
    steps, io = {}, {}
    for T in ALL_BUCKETS:
        prefill = T > MAX_T
        # a verify step of Tq columns a row (drafts): its target pass records the KDA updates
        verifies = (0, m.Tq) if m.K and m.Tq <= T <= MTP_T else (0,)
        for verify in verifies:
            for index in (False, True):
                for sparse in (False, True) if index and not prefill else (False,):
                    prog = m._program(T, index, sparse, verify)
                    steps[T, index, sparse, verify] = prog.native or m._native_program(prog, T)
    dflash = _dflash_side(E, GaudiProgram, runner) if m.drafter is not None else None
    commit = _commit_side(m) if m.K else None
    for T in ALL_BUCKETS:
        io[T] = _io(E, GaudiProgram, m, T, host, stride, staging, sstride, sample, scoring, dflash, commit)
    g = runner.geometry
    return GlmExecutor(
        glue=dev.glue,
        fill=m._native,
        layout=runner.cache_layout,
        steps=steps,
        io=io,
        image=(m.table, host, stride, host_events),
        staging=(
            staging,
            sstride,
            stage_events,
            (_SAMPLED["tok_out"], _SAMPLED["lp_out"], _SAMPLED["top_idx"], _SAMPLED["top_lp"], _SAMPLED_BYTES),
        ),
        sample=(sample, _SAMPLE_LAST),
        geometry=(g.slab_rows, g.kv_block_rows, g.state_block_rows, STATE_TOKENS),
        model={
            "vocab": runner.vocab_size,
            "prefill_buckets": list(PREFILL_BUCKETS),
            "decode_buckets": list(DECODE_BUCKETS),
            "max_t": MAX_T,
            "head_cols": HEAD_COLS,
            "max_batch_size": runner.max_batch_size,
            "max_seq_len": runner.max_seq_len,
            "spin_ns": int(runtime._SPIN_WAIT * 1e9),
            "draft_tokens": runner.K,
            "speculative_rows": m.speculative_rows if runner.K else 0,
            "spec_rows": runner.spec_rows,
            "lookahead": runner.lookahead,
            "policy": runner.policy,
            "prior": 0.4 * runner.K,
            "costs": step_cost_terms(
                m.nranks, runner.K, m.drafter is not None, m.speculative_rows if runner.K else 1
            ),
            **({"mtp_block_rows": g.mtp_block_rows} if runner.mtp else {}),
            **({"draft_block_rows": g.draft_block_rows} if m.drafter is not None else {}),
        },
        scoring=scoring,
        mtp=_mtp_side(E, GaudiProgram, runner) if runner.mtp else None,
        dflash=None if dflash is None else {k: v for k, v in dflash.items() if k != "inject"},
        commit=commit,
        share=_share(runner),
    )


def _commit_side(m) -> dict:
    """A verify step's commit (``_commit``): the pinned tables the executor fills (``qwen_mtp::fill_commit``: bases,
    counts and the hidden table of at most MTP_T // Tq rows) and the STATE geometry they follow."""
    g = m.geo
    L, B = max(len(m.kda), 1), MTP_T // m.Tq
    counts_at = _round(16 * B * L, 256)
    table_at = counts_at + _round(4 * B * L, 256)
    return {
        "host": m.device.host_buffer(table_at + _round(8 * B, 256), np.uint8),
        "at": (counts_at, table_at),
        "geometry": {
            "state_layer_rows": g.state_layer_rows,
            "ssm_rows": g.ssm_rows,
            "hidden_offset": len(m.kda) * g.state_layer_rows,
            "kda_layers": L,
        },
    }


def _share(runner):
    """Without a native rank group, the ranks' draft policies decide with the first rank's step times: this sends
    them (``ranks.from_first_rank``) at the same steps on every rank. None on one card, and with a native rank
    group: its driver decides for every rank with the step times it measures, and sends the decision with the
    step."""
    from resin_models_gaudi import ranks

    if runner.nranks == 1 or not runner.K or runner.native_ranks is not None:
        return None
    groups = runner.groups
    return lambda table: ranks.from_first_rank(table, groups)


def _mtp_side(E, GaudiProgram, runner) -> dict:
    """The MTP layer's programs, image and buffers: per MTP table bucket an I/O program (``_mtp_io``), the MTP pass
    programs of every (columns, hidden columns, indexer, sparse) shape a step makes, and the pinned buffers the
    executor fills (the image's host copies, the hidden columns, tokens, sampler and commit tables, staging)."""
    m = runner.model
    dev, H = m.device, m.d.hidden
    stride = _round(max(tb.nbytes for tb in m._mtables.values()))
    host = dev.host_buffer(2 * stride, np.uint8)
    sstride = _round(_SAMPLED_BYTES, 256)
    buffers = {
        "cols": dev.host_buffer(4 * TMAX, np.uint8),
        "tok": dev.host_buffer(4 * TMAX, np.uint8),
        "sample": dev.host_buffer(_SAMPLE_TABLE, np.uint8),
        "hidden": dev.host_buffer(256, np.uint8),
        "staging": dev.host_buffer(2 * sstride, np.uint8),
    }
    io = {T: _mtp_io(E, GaudiProgram, m, T, host, stride, sstride, buffers) for T in ALL_BUCKETS}
    programs = {}

    def add(T: int, columns: int, index: bool, sparse: bool) -> None:
        prog = m._mtp_program(T, columns, index, sparse)
        programs[T, columns, index, sparse] = prog.native or m._mtp_native(prog, T)

    for T in DECODE_BUCKETS:
        add(T, T, False, False)  # a plain step's keys; with the layer, a verify step's or a chain link's
        if T <= MTP_T:
            add(T, T, True, True)
    for T in PREFILL_BUCKETS:
        add(T, T, False, False)  # a chunk's keys
        add(1, T, False, False)  # the draft after a prompt
        add(1, T, True, True)
    g = m.geo
    return {
        "image": (m.mtable, host, stride, m._mhost_event),
        "io": io,
        "programs": programs,
        **buffers,
        "staging_stride": sstride,
        "stage_events": m._mtp_event,
        "addresses": {
            "hf": list(m.m["hf"]),
            "io_hf": m.io["hf"],
            "zero": m.m["mtp_zero"],
            "tok_out": m.sampled["tok_out"],
            "mtp_tok": m.m["mtp_tok"],
            "arena": m.arena.address,
        },
        "geometry": {
            "state_layer_rows": g.state_layer_rows,
            "ssm_rows": g.ssm_rows,
            "hidden_offset": len(m.kda) * g.state_layer_rows,
            "kda_layers": max(len(m.kda), 1),
            "mtp_block_rows": g.mtp_block_rows,
            "row_bytes": ROW_BYTES,
            "hidden": H,
        },
    }


def _mtp_io(E, GaudiProgram, m, T: int, host, stride: int, sstride: int, buffers: dict):
    """The MTP side's I/O program of table bucket T, with the executor's registers:

    - ``upload`` .. ``uploaded``: the MTP tables of host copy ``R_MIMAGE`` (``R_MSLOT_BYTES`` / ``R_MINDEX_BYTES`` of
      slot and index records), then that copy's event;
    - ``cols``, ``tok``: the pass's hidden columns and token ids (ring uploads of T of each);
    - ``sample`` (decode buckets): the MTP sampler's table and the sampler over every column; ``out``: its outputs
      to MTP staging set ``R_MSTAGE``, then its event;
    - ``final``, ``hidden`` (prefill buckets): a chunk's final hidden into ``io["hf"]``, and its last real column into
      its STATE block (``hidden_table`` ring-uploaded);
    - ``chain_hf`` (verify buckets): the first ``R_CHAIN_ROWS`` rows' accepted MTP hiddens (column b Tq + ``R_ACC`` + b
      of ``hf[0]``) to ``hf[1]``."""
    tb, prefill, mm = m._mtables[T], T > MAX_T, m.m
    image = host.ctypes.data
    io = GaudiProgram(E.REGISTERS)
    io.section("upload")
    io.upload(m.mtable, (image, E.R_MIMAGE, stride), tb.BLOCKS)
    if not prefill:
        io.section("upload_blocks")
        io.upload(m.mtable + tb.BLOCKS, (image + tb.BLOCKS, E.R_MIMAGE, stride), 4 * m.pages * T)
    for name, at, register in (
        ("upload_slots", tb.slot(0, 0), E.R_MSLOT_BYTES),
        ("upload_index", tb.index(0, 0), E.R_MINDEX_BYTES),
    ):
        io.section(name)
        io.upload(m.mtable + at, (image + at, E.R_MIMAGE, stride), (0, register, 1))
    io.section("uploaded")
    io.select([(E.R_MIMAGE, 2)], [0, 1])
    for k in range(2):
        io.arm()
        io.record(m._mhost_event + k)
    io.end()
    io.section("cols")
    io.upload_ring(mm["mtp_cols"], buffers["cols"].ctypes.data, 4 * T)
    io.section("tok")
    io.upload_ring(mm["mtp_tok"], buffers["tok"].ctypes.data, 4 * T)
    if prefill:
        io.section("final")
        op = m._final_op(T)
        io.launch(op.recipe.handle, op.slots.tolist(), op.addresses.tolist())
        io.section("hidden")
        io.upload_ring(mm["hidden_table"], buffers["hidden"].ctypes.data, 8)
        op = m._hidden_op(T)
        io.launch(op.recipe.handle, op.slots.tolist(), op.addresses.tolist())
    else:
        io.section("sample")
        io.upload_ring(mm["sample_table"], buffers["sample"].ctypes.data, _SAMPLE_TABLE)
        op = m._sample_op(T, T, True)
        io.launch(op.recipe.handle, op.slots.tolist(), op.addresses.tolist())
        io.section("out")
        io.download((buffers["staging"].ctypes.data, E.R_MSTAGE, sstride), mm["sampled"], _SAMPLED_BYTES)
        io.select([(E.R_MSTAGE, 2)], [0, 1])
        for k in range(2):
            io.arm()
            io.record(m._mtp_event + k)
        io.end()
    Tq = m.Tq
    if not prefill and Tq <= T <= MTP_T:
        Bc = T // Tq
        io.section("chain_hf")
        H4 = 4 * m.d.hidden
        hf0, hf1 = mm["hf"]
        io.take(E.R_CHAIN_ROWS)
        for b in range(Bc):
            io.copy(hf1 + H4 * b, (hf0 + H4 * b * Tq, E.R_ACC + b, H4), H4)
        io.end()
    io.finish()
    return io


def _dflash_side(E, GaudiProgram, runner) -> dict:
    """The DFlash 2 drafter's side: the injection image (each target pass's columns into the drafter's cache:
    positions, then every drafter layer's kv rows, at the drafter's own offsets), the block image of every row
    bucket's block step (``QwenDraftTables`` layouts), the block step programs (``common.native.block_program``) and
    the pinned buffer the next drafts come back to."""
    from resin._resin import QwenDraftTables

    m = runner.model
    dev, drafter, cfg = m.device, m.drafter, m.dfl_cfg
    host_rows = drafter.m.decode_buckets
    tables = QwenDraftTables(
        layers=cfg.layers, pairs=cfg.pairs, layer_rows=cfg.layer_rows, block=cfg.block, window=cfg.window,
        causal=cfg.causal, mask_token=cfg.mask_token, scratch_rows=int(drafter.scratch[0, 0, 0, 0]),
        max_rows=host_rows[-1],
    )  # fmt: skip
    layouts = {B: tables.layout(B) for B in host_rows}
    bstride = _round(max(layout["nbytes"] for layout in layouts.values()))
    events = dev.event_slots(5)  # block image 0 / 1, injection image 0 / 1, the next drafts
    blocks = {
        "layouts": layouts,
        "stride": bstride,
        "image": dev.malloc(bstride),
        "host": dev.host_buffer(2 * bstride, np.uint8),
        "events": events,
    }
    width = m.K + 1
    programs = {B: block_program(GaudiProgram, E, drafter, cfg, blocks, B, width) for B in host_rows}
    kv_at = _round(4 * TMAX, 256)
    istride = _round(kv_at + cfg.layers * drafter.kv_stride)
    inject = {
        "host": dev.host_buffer(2 * istride, np.uint8),
        "stride": istride,
        "kv_rows": kv_at,
        "kv_stride": drafter.kv_stride,
        "events": events + 2,
    }
    nbytes = 4 * cfg.block * host_rows[-1]
    next_host = dev.host_buffer(_round(nbytes, 256), np.uint8)
    out = GaudiProgram(E.REGISTERS)
    out.section("next")
    out.download(next_host.ctypes.data, drafter.next, nbytes)
    out.record(events + 4)
    out.finish()
    return {
        "inject": inject,
        "drafter": {
            "layers": cfg.layers, "pairs": cfg.pairs, "layer_rows": cfg.layer_rows, "block": cfg.block,
            "window": cfg.window, "causal": cfg.causal, "mask_token": cfg.mask_token,
            "scratch_rows": int(drafter.scratch[0, 0, 0, 0]), "max_rows": host_rows[-1],
        },  # fmt: skip
        "row_buckets": list(host_rows),
        "blocks": (blocks["image"], blocks["host"], bstride, events),
        "programs": programs,
        "inject_image": (inject["host"], istride, events + 2, kv_at, drafter.kv_stride),
        "next": (next_host, out, events + 4),
        "max_positions": drafter.m.max_positions,
        "draft_block_rows": runner.geometry.draft_block_rows,
    }


def _io(E, GaudiProgram, m, T: int, host, stride: int, staging, sstride: int, sample, scoring, dflash=None,
        commit=None):  # fmt: skip
    """Bucket T's I/O program (the module's docstring); with the drafter, ``inject``: the injection image's host
    copy ``R_DIMAGE`` (T columns) to the drafter's buffers, that copy's event, and the injection launch; with drafts,
    ``commit`` (verify buckets): the commit's tables (ring uploads) and the commit of T // Tq rows."""
    tb, prefill = m.tables[T], T > MAX_T
    image = host.ctypes.data
    io = GaudiProgram(E.REGISTERS)
    io.section("upload")
    io.upload(m.table, (image, E.R_IMAGE, stride), tb.BLOCKS)
    if not prefill:
        io.section("upload_blocks")
        io.upload(m.table + tb.BLOCKS, (image + tb.BLOCKS, E.R_IMAGE, stride), 4 * m.pages * T)
    for name, at, register in (
        ("upload_slots", tb.slot(0, 0), E.R_SLOT_BYTES),
        ("upload_index", tb.index(0, 0), E.R_INDEX_BYTES),
    ):
        io.section(name)
        io.upload(m.table + at, (image + at, E.R_IMAGE, stride), (0, register, 1))
    io.section("uploaded")
    io.select([(E.R_IMAGE, 2)], [0, 1])
    for k in range(2):
        io.arm()
        io.record(m._host_event + k)
    io.end()

    def sampler(t: int, R: int) -> None:
        io.upload_ring(m.sample_table, sample.ctypes.data, _SAMPLE_TABLE)
        op = m._sample_op(t, R)
        io.launch(op.recipe.handle, op.slots.tolist(), op.addresses.tolist())

    if prefill:
        io.section("sample_last")  # the head ran at the chunk's last column only (HEAD_COLS copies of it)
        sampler(HEAD_COLS, 1)
        io.section("head_scored")
        io.upload_ring(m.head_cols, scoring["columns"].ctypes.data, 4 * HEAD_COLS)
        op = m._program(T, False, False, 0).head_scored
        io.launch(op.recipe.handle, op.slots.tolist(), op.addresses.tolist())
        if m.nranks > 1:
            io.allgather(m.logits, m.logits_all, m.d.vocab * HEAD_COLS)
        io.section("scores")
        part = 4 * m.d.vocab * HEAD_COLS
        for r in range(m.nranks):
            io.download(scoring["host"].ctypes.data + r * scoring["stride"], m.logits_all + r * part, part)
        io.record(scoring["event"])
    else:
        io.section("sample")
        sampler(T, T)
    if dflash is not None:
        inj, drafter = dflash["inject"], m.drafter
        io.section("inject")
        base = inj["host"].ctypes.data
        io.upload(drafter.pos_in, (base, E.R_DIMAGE, inj["stride"]), 4 * T)
        for layer in range(m.dfl_cfg.layers):
            at = inj["kv_rows"] + layer * inj["kv_stride"]
            io.upload(drafter.kv_rows + layer * inj["kv_stride"], (base + at, E.R_DIMAGE, inj["stride"]), 32 * T)
        io.select([(E.R_DIMAGE, 2)], [0, 1])
        for k in range(2):
            io.arm()
            io.record(inj["events"] + k)
        io.end()
        recipe, slots, addresses = drafter._step(("dfl_inject", f"T{T}"))
        io.launch(recipe.handle, np.asarray(slots).tolist(), np.asarray(addresses).tolist())
    if commit is not None and T <= MAX_T and m.Tq <= T <= MTP_T:
        Bc, L = T // m.Tq, max(len(m.kda), 1)
        counts_at, table_at = commit["at"]
        mm, base = m.m, commit["host"].ctypes.data
        io.section("commit")
        io.upload_ring(mm["commit_bases"], base, 16 * Bc * L)
        io.upload_ring(mm["commit_counts"], base + counts_at, 4 * Bc * L)
        io.upload_ring(mm["hidden_table"], base + table_at, 8 * Bc)
        op = m._commit_op(Bc)
        io.launch(op.recipe.handle, op.slots.tolist(), op.addresses.tolist())
    io.section("out")
    io.download((staging.ctypes.data, E.R_STAGE, sstride), m.sampled_buf, _SAMPLED_BYTES)
    io.select([(E.R_STAGE, 2)], [0, 1])
    for k in range(2):
        io.arm()
        io.record(m._stage_event + k)
    io.end()
    io.finish()
    return io
