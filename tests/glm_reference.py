"""GLM-5.3-Flash's Python step pieces that the native code replaced, kept as references for the CPU checks:
the numpy table fill (``_fill_numpy``, ``_attention_tables``, ``_store_rows``, ``_index_launches``, ``_kda_tables``:
``resin_gaudi::glm``) and the Python launch loops (``_launches``, ``_mtp_launches``: ``Glm53Gaudi._native_program``
and ``_mtp_native``). ``Reference(model)`` runs them on a model's attributes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from resin_models_gaudi.common.attention import SlotRow, plan_slots
from resin_models_gaudi.common.sampling import TOP_MAX
from resin_models_gaudi.glm5_3_flash.cache import (
    INDEX_PAGE_ROWS,
    LATENT_PAGE_ROWS,
    PAGE_TOKENS,
    POOL,
    TAIL_POOLS,
    tail_rows,
)
from resin_models_gaudi.glm5_3_flash.layers import PAGE_POOLS, ROWS_FIELDS, SLOT_POOLS
from resin_models_gaudi.glm5_3_flash.model import (
    _SAMPLED,
    HEAD_COLS,
    MAX_T,
    SLOTS,
    Filled,
    Row,
    _head_columns,
    _IndexLaunch,
    _MtpProgram,
    _Program,
)

TAIL_OFFSET = LATENT_PAGE_ROWS + INDEX_PAGE_ROWS


class Reference:
    """The reference step pieces over ``model``'s attributes (a ``Glm53Gaudi``, or an object with what they read)."""

    def __init__(self, model) -> None:
        self.__dict__["model"] = model

    def __getattr__(self, name: str):
        return getattr(self.model, name)

    def _index_launches(self, rows: Sequence[Row], query_block: int, S: int) -> list[_IndexLaunch]:
        """Index slots of a step: per row and query block, every 2048-pool chunk of the pools it may see,
        packed S per launch. Keys are pools; qpos0 past every pool, since visibility is applied per column
        after the scores (ix_thr, gx_mask, ix_list use ``vis``)."""
        per_chunk = SLOT_POOLS // PAGE_POOLS
        out: list[tuple[list[int], np.ndarray]] = []
        NI = self.d.index_heads
        for r in rows:
            total = (r.pos0 + r.real) // POOL
            for t0 in range(0, r.real, query_block):
                seen = min((r.pos0 + t0 + query_block) // POOL, total)
                for key0 in range(0, seen, SLOT_POOLS):
                    first = key0 // PAGE_POOLS
                    live = min(per_chunk, -(-(seen - key0) // PAGE_POOLS))
                    pages = np.full(per_chunk, -1, np.int64)
                    pages[:live] = r.blocks[first : first + live]
                    rec = [0, NI * (r.column + t0), 1 << 30, key0, seen, r.column + t0, 0, 0]
                    out.append((rec, pages))
        launches = []
        for a in range(0, len(out), S):
            part = out[a : a + S]
            slots = np.zeros((S, 8), np.int32)
            slots[:, 0] = -1
            pages = np.full((S, per_chunk), -1, np.int64)
            for s, (rec, pg) in enumerate(part):
                slots[s] = rec
                pages[s] = pg
            launches.append(_IndexLaunch(slots, pages))
        return launches

    def _attention_tables(self, h, tb, rows, T, qb, S, offsets, keep_tails=False):
        """Fill a step's attention tables for layers at page ``offsets``: visibility, key and pool stores, slot and
        index records, the sparse block table. -> (key launches, index launches, indexer on, sparse)"""
        d = self.d
        vis = np.zeros(T, np.int32)
        klen = np.zeros(T, np.int32)
        store, pool = _store_rows(T, rows, keep_tails, self.geo.tail_pools)
        for r in rows:
            p = r.pos0 + np.arange(r.real)
            vis[r.column : r.column + r.real] = (p + 1) // POOL
            klen[r.column : r.column + r.real] = p + 1
        index = bool(vis.max() > d.top_pools)
        sparse = index and qb == 1
        slot_rows = [
            SlotRow(r.column, r.pos0, r.real, r.blocks[None, :], r.blocks[None, :]) for r in rows
        ]
        launches = [] if sparse else plan_slots(slot_rows, qb, d.mla_heads, S)
        ix_launches = self._index_launches(rows, qb, S) if index else []
        if len(launches) > tb.key_launches or len(ix_launches) > tb.index_launches:
            raise ValueError("a step's slot launches exceed the table capacity")
        h[tb.VIS : tb.VIS + 4 * T].view(np.int32)[:] = vis
        h[tb.KLEN : tb.KLEN + 4 * T].view(np.int32)[:] = klen
        M = len(offsets)
        if M:
            off = offsets[:, None, None]
            so = h[tb.STORE : tb.STORE + M * tb.ROWS_BYTES].view(np.int32)
            so = so.reshape(M, tb.ROWS_BYTES // (4 * ROWS_FIELDS), ROWS_FIELDS)
            so[:] = -1
            live = store[:, 0] >= 0
            so[:, :T, 0] = np.where(live, store[:, 0][None] + off[:, :, 0], -1)
            so[:, :T, 3] = np.where(live, store[:, 3][None] + off[:, :, 0], -1)
            so[:, :T, 4] = np.where(live, store[:, 4][None], -1)
            po = h[tb.POOL : tb.POOL + M * tb.ROWS_BYTES].view(np.int32)
            po = po.reshape(M, tb.ROWS_BYTES // (4 * ROWS_FIELDS), ROWS_FIELDS)
            po[:] = -1
            live = pool[:, 0] >= 0
            po[:, :T, 0] = np.where(live, pool[:, 0][None] + 2 * off[:, :, 0], -1)
            for s in (1, 2, 3):
                # a column >= 0, or -(arena row + 1): tails shift with the layer
                po[:, :T, s] = np.where(
                    pool[:, s] < 0, pool[:, s][None] - off[:, :, 0], pool[:, s][None]
                )
            tail = pool[:, 4] >= 0
            po[:, :T, 4] = np.where(tail, pool[:, 4][None] + off[:, :, 0], -1)
        for j, launch in enumerate(launches):
            rec = h[tb.slot(j, 0) : tb.slot(j, 0) + M * tb.SLOT_RECORD].view(np.int32)
            rec = rec.reshape(M, -1)
            pages = launch.k_pages[None]
            rec[:, : 8 * S] = launch.slots.ravel()
            lat = np.where(pages >= 0, pages + offsets[:, None, None], -1)
            o = tb.SLOTS_BYTES // 4
            rec[:, o : o + pages[0].size] = lat.reshape(M, -1)
        for j, launch in enumerate(ix_launches):
            rec = h[tb.index(j, 0) : tb.index(j, 0) + M * tb.INDEX_RECORD].view(np.int32)
            rec = rec.reshape(M, -1)
            pages = launch.pages[None]
            rec[:, : 8 * S] = launch.slots.ravel()
            ixp = np.where(pages >= 0, pages + offsets[:, None, None] + LATENT_PAGE_ROWS, -1)
            o = tb.SLOTS_BYTES // 4
            rec[:, o : o + pages[0].size] = ixp.reshape(M, -1)
        if sparse:
            bt = h[tb.BLOCKS : tb.BLOCKS + 4 * self.pages * T].view(np.int32)
            bt = bt.reshape(T, self.pages)
            bt[:] = -1
            for r in rows:
                bt[r.column : r.column + r.real, : len(r.blocks)] = r.blocks
        return launches, ix_launches, index, sparse

    def _fill_numpy(self, h, tb, mtp, rows, n, first, tokens, head, verify, keep_tails, meta_columns) -> Filled:
        T = tb.T
        qb = T if T > MAX_T else 1
        h[: tb.HEADER].view(np.int32)[: tb.HEADER // 4] = 0
        if tokens is not None:
            h[tb.TOK : tb.TOK + 4 * n].view(np.int32)[:] = tokens
        # real columns: the router sends padding nowhere and the chunked KDA passes it through
        meta = n if meta_columns is None else meta_columns
        h[tb.META : tb.META + 16].view(np.int32)[:] = (first, meta, 0, 0)
        if head:
            h[tb.HEAD : tb.HEAD + 4 * HEAD_COLS].view(np.int32)[:] = n - 1
        offsets = self._mtp_off if mtp else self._dsa_off
        launches, ix_launches, index, sparse = self._attention_tables(
            h, tb, rows, T, qb, SLOTS[qb], offsets, keep_tails=keep_tails
        )
        if not mtp and (K := len(self.kda)):
            self._kda_tables(h, tb, rows, T, K, T > MAX_T, verify)
        uploads = [(0, tb.BLOCKS), (0, 0), (0, 0), (0, 0)]
        if sparse:
            uploads[1] = (tb.BLOCKS, 4 * self.pages * T)
        if launches:
            uploads[2] = (tb.slot(0, 0), tb.slot(len(launches), 0) - tb.slot(0, 0))
        if ix_launches:
            uploads[3] = (tb.index(0, 0), tb.index(len(ix_launches), 0) - tb.index(0, 0))
        nsl = 1 if sparse else len(launches)
        return Filled(index, sparse, nsl, len(ix_launches), tuple(uploads))

    def _kda_tables(self, h, tb, rows, T, K, prefill, verify=0) -> None:
        g = self.geo
        if prefill:
            r = rows[0]
            a = np.full((K, tb.KDA_RECORD // 4), -1, np.int32)
            # conv_src, ssm_src, ssm_dst, conv_dst 256 bytes apart
            for nth in range(K):
                ss, cs = g.state_rows(r.state_src, nth)
                sd, cd = g.state_rows(r.state_dst, nth)
                a[nth, 0], a[nth, 64], a[nth, 128], a[nth, 192] = cs, ss, sd, cd
        else:
            a = np.full((K, tb.KDA_RECORD // 4), -1, np.int32)
            bases = a[:, : tb.BASES_BYTES // 4].reshape(K, -1, 4)
            cdst = a[:, tb.BASES_BYTES // 4 :]
            for r in rows:
                c = r.column // verify if verify else r.column
                for nth in range(K):
                    ss, cs = g.state_rows(r.state_src, nth)
                    sd, cd = g.state_rows(r.state_dst, nth)
                    bases[nth, c] = cs, ss, -1, sd
                    cdst[nth, c] = cd
        h[tb.KDA : tb.KDA + a.nbytes].view(np.int32)[:] = a.ravel()

    def _launches(self, prog: _Program, T: int, device_tokens: bool, nix: int, nsl: int) -> None:
        """A step's launches and collectives, embed through the logits all-gather: the first ``nix`` index slot
        launches and ``nsl`` slot launches of each attention layer; the embedding reads the previous step's
        sampled tokens when ``device_tokens``: the Python loop that ``_native_program`` replaced."""
        dev, d = self.device, self.d
        lib = dev.lib
        gather = d.gathered(T)
        ar = None
        if self.nranks > 1:
            ar = lib.rg_ccl_allgather if gather else lib.rg_ccl_allreduce
        dst_a, dst_b = (self.pa_all, self.pb_all) if gather else (self.pa, self.pb)
        count = d.hidden * T
        embed = prog.embed_device if device_tokens else prog.embed
        assert embed is not None and prog.head is not None
        embed.run(lib)
        probe = getattr(self.model, "probe", None)
        for i, (attn, ix, ix_end, slots, post, ffn) in enumerate(prog.layers):
            attn.run(lib)
            if (tap := prog.taps.get(i)) is not None:
                tap.run(lib)
            for op in ix[:nix]:
                op.run(lib)
            if ix_end is not None:
                ix_end.run(lib)
            for op in slots[:nsl]:
                op.run(lib)
            if post is not None:
                post.run(lib)
            if ar is not None and ar(self.pa, dst_a, count, 0):
                dev.check(-1, "residual collective")
            if probe is not None:
                probe(i, "attn", T, dst_a)
            ffn.run(lib)
            if ar is not None and ar(self.pb, dst_b, count, 0):
                dev.check(-1, "residual collective")
            if probe is not None:
                probe(i, "mlp", T, dst_b)
        prog.head.run(lib)
        if self.nranks > 1:
            dev.allgather(self.logits, self.logits_all, d.vocab * _head_columns(T))

    def _mtp_launches(
        self, prog: _MtpProgram, T: int, layer: bool, prev: int, hf: int, tokens: int, hf_out: int, nix: int, nsl: int
    ) -> None:
        """The MTP pass's launches: its input (hidden columns of [``prev`` | ``hf``], token ids at ``tokens``) and
        pre launch, then with ``layer`` the rest of the layer and the head (its hidden into ``hf_out``) and the
        logits all-gather, with ``nix`` index slot launches and ``nsl`` slot launches. One native call
        (``_mtp_native``) when the device has the glue, else the Python loop."""
        dev, d = self.device, self.d
        mi, pre, post, ffn, head = prog.mtp_in, prog.pre, prog.post, prog.ffn, prog.head
        assert mi is not None and pre is not None
        mi.addresses[prog.slot_prev] = prev
        mi.addresses[prog.slot_hf] = hf
        mi.addresses[prog.slot_tok] = tokens
        lib = dev.lib
        mi.run(lib)
        pre.run(lib)
        if not layer:
            return
        assert post is not None and ffn is not None and head is not None
        gather = d.gathered(T)
        ar = None
        if self.nranks > 1:
            ar = lib.rg_ccl_allgather if gather else lib.rg_ccl_allreduce
        dst_a, dst_b = (self.pa_all, self.pb_all) if gather else (self.pa, self.pb)
        count = d.hidden * T
        for op in prog.ix[:nix]:
            op.run(lib)
        if prog.ix_end is not None:
            prog.ix_end.run(lib)
        for op in prog.slots[:nsl]:
            op.run(lib)
        post.run(lib)
        if ar is not None and ar(self.pa, dst_a, count, 0):
            dev.check(-1, "residual collective")
        ffn.run(lib)
        if ar is not None and ar(self.pb, dst_b, count, 0):
            dev.check(-1, "residual collective")
        head.addresses[prog.slot_hf_out] = hf_out
        head.run(lib)
        if self.nranks > 1:
            dev.allgather(self.logits, self.logits_all, d.vocab * T)


def _store_rows(
    T: int, rows: Sequence[Row], keep_tails: bool = False, tail_pools: int = TAIL_POOLS
) -> tuple[np.ndarray, np.ndarray]:
    """Per column, relative to the layer's pages: ``store`` (T, 5) [latent row, -, -, scale row, scale index] for
    ``mla_store``, ``pool`` (T, 5) [pool unit, src0, src1, src2, tail row] for ``gx_pool``. Rows are block rows
    plus the fixed page offsets; the caller adds the layer offset (a tail source -(row + 1) subtracts it).
    ``tail_pools``: how many consecutive pools keep their tails apart (``cache.tail_rows``)."""
    store = np.full((T, 5), -1, np.int64)
    pool = np.full((T, 5), -1, np.int64)
    # sources default to column 0: a column completing no pool never reads them
    pool[:, 1:4] = 0
    for r in rows:
        pos = r.pos0 + np.arange(r.real)
        cols = r.column + np.arange(r.real)
        b = np.asarray(r.blocks, np.int64)[pos // PAGE_TOKENS]
        slot = pos % PAGE_TOKENS
        store[cols, 0] = b + slot
        store[cols, 3] = b + PAGE_TOKENS
        store[cols, 4] = slot
        last = r.pos0 + r.real - 1
        done = pos % POOL == POOL - 1
        P = pos // POOL
        pool[cols[done], 0] = 2 * (b[done] + LATENT_PAGE_ROWS) + P[done] % PAGE_POOLS
        for s in range(3):
            q = pos[done] - 3 + s
            inside = q >= r.pos0
            pool[cols[done], 1 + s] = np.where(
                inside,
                cols[done] - 3 + s,
                -(b[done] + TAIL_OFFSET + tail_rows(q, tail_pools) + 1),
            )
        # the pool of an unfinished token: its last token is past this launch, so a later launch reads it. A verify
        # step keeps every tail (``keep_tails``): a rejected draft restarts the next step inside any of its pools
        open_ = pos % POOL != POOL - 1
        if not keep_tails:
            open_ &= P * POOL + POOL - 1 > last
        pool[cols[open_], 4] = b[open_] + TAIL_OFFSET + tail_rows(pos[open_], tail_pools)
    return store, pool


def _sampled_fields(out: np.ndarray, R: int, n: int, layout: dict[str, int] | None = None):
    """(tokens, logprobs, top ids, top logprobs) of the first n of R sampled columns; ``layout``: the fields'
    byte offsets (default ``_SAMPLED``, this module's MAX_T rows)"""
    layout = _SAMPLED if layout is None else layout

    def field(name, dtype, shape):
        off = layout[name]
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        return out[off : off + size].view(dtype).reshape(shape)[:n].copy()

    return (
        field("tok_out", np.int32, (R,)),
        field("lp_out", np.float32, (R,)),
        field("top_idx", np.int32, (R, TOP_MAX)),
        field("top_lp", np.float32, (R, TOP_MAX)),
    )
