"""Reference copies of DeepSeek-V4.1's Python step code, which the native executor replaced (tests only): the
numpy table fills (``_fill_positions``, ``_fill_tokens``, ``_fill_draft``, the Engram hash), the Python launch
loops of a step and a draft (``_step_launches``, ``_run_layer``, ``_head``, ``_draft_launches``) and the Engram
context. The native code must equal them: byte for byte (tables), call for call (programs), record for record
(context). A test model is a ``ReferenceModel``: ``DeepSeekV41Gaudi`` with these methods."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from resin_models_gaudi.deepseek_v41 import layers as L
from resin_models_gaudi.deepseek_v41.cache import (
    HISTORY_TOKENS,
    INDEX_ROWS,
    PAGE_ROWS,
    PAGE_TOKENS,
    PAIR_ROWS,
)
from resin_models_gaudi.deepseek_v41.model import (
    _R_CAND,
    _R_DENSE,
    _R_IX,
    _R_IXC,
    _R_PF,
    _R_RADIX,
    _R_SPARSE,
    DENSE_KEYS,
    INDEX_KEYS,
    PREFILTER_BLOCKS,
    PREFILTER_MIN,
    RADIX_MIN,
    DeepSeekV41Gaudi,
    J,
    Row,
    Shape,
    _Program,
    _Tables,
)
from resin_models_gaudi.deepseek_v41.rope import cs_table
from resin_models_gaudi.deepseek_v41.weights import engram_rows, split64

_NONE = np.zeros(0, np.int64)
logger = logging.getLogger("wright")


if TYPE_CHECKING:
    _Model = DeepSeekV41Gaudi  # the methods' ``self`` (ReferenceModel)
else:
    _Model = object


class Reference(_Model):
    """The Python step code's methods (``self``: a DeepSeekV41Gaudi)."""

    host: list[np.ndarray]  # the host copies of the table image (the tests set them)
    _registers: np.ndarray  # a step program's registers (the tests set them)

    def _step_launches(
        self, prog: _Program, s: Shape, counts: dict, logits: str | int | None
    ) -> None:
        """A step's launches in Python: embed, the layers, the DSpark injection, the head (``_head``)."""
        lib, ccl = self.device.lib, self.nranks > 1
        assert prog.embed is not None
        prog.embed.run(lib)
        for layer, ops in enumerate(prog.layers):
            self._run_layer(layer, ops, counts, s, ccl)
        if prog.inject is not None:  # the DSpark stages' window keys at the step's positions
            prog.inject.run(lib)
        self._head(prog, logits, s.T, ccl)

    def _run_layer(
        self, layer: int, ops: dict, counts: dict, s: Shape, ccl: bool
    ) -> None:
        """One layer's launches and collectives in the step's order."""
        dev, lib, T = self.device, self.device.lib, s.T
        if "eg" in ops:
            if "eg_gather" in ops:
                ops["eg_gather"].run(lib)
                if ccl:
                    dev.allreduce(self.io["eg_emb"], 256 * J * T)
            ops["eg"].run(lib)
            if "eg_gate" in ops:
                if ccl:
                    dev.allgather(
                        self.io["eg_kv"], self.io["eg_kv_all"], self.eg_rows_r * T
                    )
                ops["eg_gate"].run(lib)
        ratio = self.cfg.compress_ratios[layer]
        dense = counts.get(("dense", ratio), False)
        (ops["attn_dense"] if dense and "attn_dense" in ops else ops["attn"]).run(lib)
        if "ixp" in ops:
            ratio = self.cfg.compress_ratios[layer]
            plain = ops["ixp"]
            if counts["sparse"] and "ixsp" in ops:
                # only the candidate entries
                for op in ops["ixsp"][: counts["ixc"]]:
                    op.run(lib)
                for op in ops.get("ixsp_tail", []):
                    op.run(lib)
            elif counts["cand"]:
                for op in ops["ixc2"][: counts["pf", ratio]]:
                    op.run(lib)
                for op in ops.get("ixc_tail", []):
                    op.run(lib)
            else:
                for op in plain[: counts[ratio]]:
                    op.run(lib)
        if "out" in ops:
            (ops["out_dense"] if dense and "out_dense" in ops else ops["out"]).run(lib)
        if ccl:
            self._reduce(ops["p_attn"], s)
        ops["ffn"].run(lib)
        if ccl:
            self._reduce(ops["p_ffn"], s)

    def _head(
        self, prog: _Program, logits: str | int | None, T: int, ccl: bool
    ) -> None:
        dev, d = self.device, self.d
        self._column = False
        if logits is None:
            return
        assert prog.head is not None
        prog.head.run(dev.lib)
        if ccl and logits == "all":
            dev.allgather(self.logits, self.logits_all, d.vocab * T)
        elif ccl:  # one column, gathered as a one-column step's logits
            dev.allgather(
                self.logits + 4 * d.vocab * int(logits), self.logits_all, d.vocab
            )
            self._column = True

    def _reduce(self, partial: int, s: Shape) -> None:
        """Sum a sub-layer's partial over the ranks, or all-gather it for the next graph to sum."""
        count = self.d.hidden * s.T
        if self.d.gathered(s.T):
            self.device.allgather(partial, self.pall[self.pbuf.index(partial)], count)
        else:
            self.device.allreduce(partial, count)

    def _draft_launches(self, prog: dict, s: Shape, B: int) -> None:
        """A DSpark draft's launches in Python: embed, the stages, the head, the Markov steps of ``B`` rows."""
        dev, lib, ccl = self.device, self.device.lib, self.nranks > 1
        prog["embed"].run(lib)
        for ops in prog["stages"]:
            ops["attn"].run(lib)
            if ccl:
                self._reduce(ops["p_attn"], s)
            ops["ffn"].run(lib)
            if ccl:
                self._reduce(ops["p_ffn"], s)
        prog["head"].run(lib)
        if ccl:
            dev.allgather(self.logits, self.logits_all, self.d.vocab * s.T)
        prog["markov"][B].run(lib)

    def _fill_draft(
        self, k: int, tb: _Tables, anchors: np.ndarray, rows: Sequence[Row]
    ) -> None:
        """A draft step's tables: row b's block in columns b block .. b block + block - 1 at positions pos0 ..
        (their RoPE), each column's window slot over the stages' pages of the target's last positions before
        pos0 (the block's own keys are not stored), and the block keys' columns and slots."""
        cfg, s, v = self.cfg, tb.shape, tb.v[k]
        T, K, nh, B = s.T, self.block, self.d.heads, len(rows)
        n = B * K
        self.host[k][: tb.nbytes] = 0
        row_of, i_of = np.repeat(np.arange(B), K), np.tile(np.arange(K), B)
        start = np.array([r.pos0 for r in rows], np.int64)
        tok = np.full(n, cfg.dspark_noise_token_id, np.int64)
        tok[i_of == 0] = anchors
        v["tok"][:n] = tok
        v["meta"][:] = (int(start[0]), n, 0, 0)
        pos = np.zeros(T, np.int64)
        pos[:n] = start[row_of] + i_of
        for r in (0, 1):
            cs = self._cs(r, pos)
            v[f"cs{r}"][:] = cs
            v[f"cs{r}_inv"][:, :64] = cs[:, :64]
            v[f"cs{r}_inv"][:, 64:] = -cs[:, 64:]
        q = (
            start[row_of] - 1
        )  # every column's window ends at the position before its anchor
        key0 = np.maximum(q - (cfg.sliding_window - 1), 0) // PAGE_TOKENS * PAGE_TOKENS
        cols, zeros = np.arange(n), np.zeros(n, np.int64)
        slots = v["slots"]
        slots[:, 0] = -1
        slots[:n] = np.stack(
            [zeros, nh * cols, q, key0, start[row_of], zeros, zeros, np.full(n, cfg.sliding_window)], 1
        )  # fmt: skip
        pages = s.win_keys // PAGE_TOKENS
        lat0 = np.full((s.win_slots, pages), -1, np.int64)
        for col in range(n):
            r = rows[row_of[col]]
            idx = key0[col] // PAGE_TOKENS + np.arange(pages) - r.window_first
            ok = (
                (idx >= 0)
                & (idx < len(r.window))
                & (idx + r.window_first <= q[col] // PAGE_TOKENS)
            )
            ok &= q[col] >= 0
            lat0[col] = np.where(ok, np.asarray(r.window)[np.where(ok, idx, 0)], -1)
        v["lat"][:] = -1
        v["rows"][:] = -1  # nothing stored
        for st in range(self.stages):
            layer = self.n_layers + st
            v["lat"][layer] = np.where(lat0 >= 0, lat0 + self._layer_rows[layer], -1)
        v["dsp_bidx"][: n * K] = (row_of[:, None] * K + np.arange(K)).ravel()
        ls = v["dsp_lslots"]
        ls[:, 0] = -1
        ls[:n] = np.stack(
            [zeros, nh * cols, np.full(n, K), zeros, np.full(n, K), cols, zeros, zeros],
            1,
        )
        v["dsp_anchor"][:B] = anchors

    def _cs(self, r: int, positions: np.ndarray) -> np.ndarray:
        """cs_table(positions, freqs[r]); for many positions, rows of the precomputed table"""
        positions = np.asarray(positions, np.int64)
        if len(positions) <= 64:
            return cs_table(positions, self.freqs[r])
        return self._cs_all[r][positions]

    def _sections(self, first: int, last: int, logits: str | int | None) -> tuple[int, ...]:
        """The sections (``_step_program``) of layers ``first`` .. ``last`` - 1, with the embedding before them
        when ``first`` is 0 and the injection and the head for ``logits`` after them when ``last`` is the end."""
        n = self.n_layers
        head = () if logits is None else (n + 3,) if self._one_column(logits) else (n + 2,)
        embed = (0,) if first == 0 else ()
        after = (n + 1, *head) if last == n else ()
        return (*embed, *range(1 + first, 1 + last), *after)

    def _one_column(self, logits: str | int | None) -> bool:
        """whether the head gathers one column's logits (``_head``)"""
        return self.nranks > 1 and logits is not None and logits != "all"

    def _step_registers(self, counts: dict) -> np.ndarray:
        """A decode step's counts as the registers of its native program (``_R_*``)."""
        r = self._registers
        r[_R_SPARSE] = counts["sparse"]
        r[_R_CAND] = counts["cand"]
        r[_R_IXC] = counts.get("ixc", 0)
        for ratio in (1, 2):
            r[_R_RADIX[ratio]] = counts["radix", ratio]
            r[_R_IX[ratio]] = counts.get(ratio, 0)
            r[_R_PF[ratio]] = counts.get(("pf", ratio), False)
            r[_R_DENSE[ratio]] = counts.get(("dense", ratio), False)
        return r

    def _fill_tokens(
        self, k: int, tb: _Tables, tokens, n: int, rows: Sequence[Row]
    ) -> None:
        s, v = tb.shape, tb.v[k]
        if tokens is not None:
            v["tok"][:n] = tokens
        if self.engram_layers and self.hashed(s):
            xor = self._engram_older(rows, s.T, tokens)
            for layer in self.engram_layers:
                e = self.cfg.engram_layer_ids.index(layer)
                v[f"eg_xh.{layer}"][:, :J] = xor[:, e, :, 0]
                v[f"eg_xl.{layer}"][:, :J] = xor[:, e, :, 1]
        elif self.engram_layers:
            ids = self._engram_ids(rows, tokens, s.T)
            for layer in self.engram_layers:
                e = self.cfg.engram_layer_ids.index(layer)
                if self.engram == "card":
                    v[f"eg.{layer}"][:] = ids[:, e]
                else:
                    emb = v[f"eg.{layer}"]
                    for r in rows:
                        c = slice(r.column, r.column + r.real)
                        emb[c] = engram_rows(
                            self.src, layer, ids[c, e].reshape(-1)
                        ).reshape(r.real, -1)

    def _fill_positions(
        self, k: int, tb: _Tables, positions, rows: Sequence[Row]
    ) -> dict:
        """Write the tables of host copy ``k`` that follow from the step's positions and rows (all but
        ``_fill_tokens``'); returns the index launches per ratio."""
        cfg, geo, s, v = self.cfg, self.geo, tb.shape, tb.v[k]
        T, n, nh, R = s.T, len(positions), self.d.heads, len(rows)
        for a, b in tb.zero:
            self.host[k][a:b] = 0
        v["meta"][:] = (int(positions[0]), n, 0, 0)
        pos = np.zeros(T, np.int64)
        pos[:n] = positions
        for r in (0, 1):
            cs = self._cs(r, pos)
            v[f"cs{r}"][:] = cs
            v[f"cs{r}_inv"][:, :64] = cs[:, :64]
            v[f"cs{r}_inv"][:, 64:] = -cs[:, 64:]
        # each row's blocks, padded: WINDOW rows and ids from logical block wfirst, HISTORY rows from block 0
        wlen = max(len(r.window) for r in rows)
        wrow, wid = np.zeros((R, wlen), np.int64), np.zeros((R, wlen), np.int64)
        hist = np.full(
            (R, max(self.pages, max(len(r.history) for r in rows))), -1, np.int64
        )
        wfirst, pos0 = np.empty(R, np.int64), np.empty(R, np.int64)
        prev_id = np.zeros(R, np.int64)  # the WINDOW block id of position pos0 - 1
        col_row = np.full(T, -1, np.int64)
        slot_row, slot_b0 = [], []
        for k, r in enumerate(rows):
            wrow[k, : len(r.window)], wid[k, : len(r.window)] = r.window, r.window_ids
            hist[k, : len(r.history)] = r.history
            wfirst[k], pos0[k] = r.window_first, r.pos0
            if r.pos0:
                prev_id[k] = r.window_ids[(r.pos0 - 1) // PAGE_TOKENS - r.window_first]
            col_row[r.column : r.column + r.real] = k
            for b0 in range(0, r.real, s.qb):
                slot_row.append(k)
                slot_b0.append(b0)
        cols = np.flatnonzero(col_row >= 0)
        live = col_row >= 0
        cr, cp = col_row[cols], pos[cols]
        block = cp // PAGE_TOKENS - wfirst[cr]
        win = np.zeros(T, np.int64)
        win_id = np.zeros(T, np.int64)
        win[cols], win_id[cols] = wrow[cr, block], wid[cr, block]
        last = np.array([r.pos0 + r.real - 1 for r in rows], np.int64)
        last_col = np.array([r.column + r.real - 1 for r in rows], np.int64)
        # window slots, and every layer's store rows and window pages
        sr, sb = np.array(slot_row, np.int64), np.array(slot_b0, np.int64)
        S, pages = len(sr), s.win_keys // PAGE_TOKENS
        q0 = pos0[sr] + sb
        key0 = np.maximum(q0 - (cfg.sliding_window - 1), 0) // PAGE_TOKENS * PAGE_TOKENS
        col0 = np.array([rows[k].column for k in slot_row], np.int64) + sb
        slots = v["slots"]
        slots[:, 0] = -1
        slots[:S] = np.stack(
            [np.zeros(S, np.int64), nh * col0, q0, key0, last[sr] + 1, np.zeros(S, np.int64),
             np.zeros(S, np.int64), np.full(S, cfg.sliding_window)], 1,
        )  # fmt: skip
        pg = key0[:, None] // PAGE_TOKENS + np.arange(pages)
        ok = pg <= last[sr, None] // PAGE_TOKENS
        lat0 = np.full((s.win_slots, pages), -1, np.int64)
        lat0[:S] = np.where(
            ok, wrow[sr[:, None], np.where(ok, pg - wfirst[sr, None], 0)], -1
        )
        off = self._layer_rows[:, None, None]
        v["lat"][:] = np.where(lat0 >= 0, lat0[None] + off, -1)
        slot_pos = cp % PAGE_TOKENS
        rows0 = np.full((T, 8), -1, np.int64)
        rows0[cols, 0] = win[cols] + slot_pos
        rows0[cols, 3] = win[cols] + PAGE_TOKENS
        rows0[cols, 4] = slot_pos
        shift = np.zeros((T, 8), np.int64)
        shift[cols, 0] = shift[cols, 3] = 1
        np.add(rows0[None], shift[None] * off, out=v["rows"], casting="unsafe")
        # compressor: ratio-2 groups (completed at odd positions) and pending tokens, ratio-1 entries
        odd = cols[cp % 2 == 1]
        gr = col_row[odd]
        inside = pos[odd] - 1 >= pos0[gr]
        G2 = len(odd)
        entries2 = np.full(s.groups2, -1, np.int64)
        entries2[:G2] = pos[odd] // 2
        if geo.pending_owners:
            v["cs_g2"][:] = self._cs(1, np.where(entries2 >= 0, 2 * entries2, 0))
        # the columns whose pending state is stored: a prefill row's last column when its position is even (the
        # next chunk or step completes the group); every even column of a decode or verify row, since a rejected
        # draft can end the committed range at any of them, each in the slot of its position
        if s.prefill:
            stored = last_col[last % 2 == 0]
        else:
            stored = cols[cp % 2 == 0]
        for i, o in enumerate(geo.pending_owners):
            groups = v[f"cmp_groups.{o}"]
            groups[:G2, 0] = np.where(inside, odd - 1, -1)
            groups[:G2, 1] = odd
            groups[:G2, 2] = np.where(
                inside, 0, geo.pending_row(prev_id[gr], i, pos[odd] - 1)
            )
            pend = v[f"cmp_rows.{o}"]
            pend[:] = -1
            pend[stored] = geo.pending_row(win_id[stored], i, pos[stored])
        if len(geo.pending_owners) < len(geo.owners):
            v["cs_g1"][:] = self._cs(1, np.where(live, pos, 0))
        for o in geo.owners:
            ratio, op, ooff = (
                cfg.compress_ratios[o],
                geo.owner_pages(o),
                geo.owner_offset(o),
            )
            store = v[f"cmp_store.{o}"]
            store[:] = -1
            g, j, er = (
                (np.arange(G2), entries2[:G2], gr) if ratio > 1 else (cols, cp, cr)
            )
            P = j // PAGE_TOKENS
            page = hist[er, P // op] + ooff + PAIR_ROWS * (P % op)
            e = j % PAGE_TOKENS
            store[g, 0] = page + e
            store[g, 2] = (
                4 * (page + PAGE_ROWS) + e
            )  # FP8 index key: unit e of the index page
            store[g, 5] = page + PAGE_ROWS + INDEX_ROWS - 1  # its scale: row 32, f32 e
            store[g, 6] = e
            store[g, 3] = page + PAGE_TOKENS
            store[g, 4] = e
        # indexer: the tiled chain's groups, its per-group tables and its launches
        counts = {"sparse": False, "cand": False}
        v["ix_vis1"][:] = self.ix_rows(s) + 1
        lanes = np.arange(INDEX_KEYS // PAGE_TOKENS)
        tp, lpg = self.ix_tp(s), self.ix_lpg(s)
        for ratio in (1, 2):
            v[f"ix_vis{ratio}"][:] = np.where(live, (pos + 1) // ratio, 0)
            vis = v[f"ix_vis{ratio}"]
            counts["radix", ratio] = False
            # each query block: its row, first column, position and visible keys (its last column's)
            nblk = s.T // s.qb
            blk: list = [None] * nblk
            for k, r in enumerate(rows):
                seen_all = (r.pos0 + r.real) // ratio
                for b0 in range(0, r.real, s.qb):
                    q = r.pos0 + b0
                    seen = min((q + s.qb) // ratio, seen_all)
                    blk[b0 // s.qb] = (k, r.column + b0, q, seen)
            nb = -(-vis // 8)
            v[self._visb(ratio)][:] = np.minimum(-(-nb // 64) * 64, self.kmax // 8)
            counts["cand"] = bool(
                not s.prefill
                and cfg.candidate_source_layer >= 0
                and int(vis.max()) > cfg.candidate_topk_blocks * 8
            )
            counts["dense", ratio] = self.dense(s) and bool(
                np.all(vis[live] <= DENSE_KEYS)
            )
            if counts["dense", ratio]:
                # a slot per query block over entries 0 .. klen - 1 (what its last column sees), and the
                # owners' latent pages of those entries
                kl = np.minimum(q0 + s.qb, last[sr] + 1) // ratio
                cslots = v[f"cslots{ratio}"]
                cslots[:] = 0
                cslots[:, 0] = -1
                z = np.zeros(S, np.int64)
                cslots[:S] = np.stack(
                    [z, nh * col0, q0, z, kl, col0, np.full(S, ratio), z], 1
                )
                pg = np.arange(DENSE_KEYS // PAGE_TOKENS)
                ok = pg[None, :] * PAGE_TOKENS < kl[:, None]
                for o in geo.owners:
                    if cfg.compress_ratios[o] != ratio:
                        continue
                    op, ooff = geo.owner_pages(o), geo.owner_offset(o)
                    lat = v[f"cmp_lat.{o}"]
                    lat[:] = -1
                    lat[:S] = np.where(
                        ok,
                        hist[sr[:, None], np.where(ok, pg // op, 0)]
                        + ooff
                        + PAIR_ROWS * (pg % op),
                        -1,
                    )
            # every live column sees more than kc entries: index sources after the candidate source score
            # only their columns' candidate entries
            sparse = (
                self.ixc_launches(s) > 0
                and cfg.candidate_source_layer >= 0
                and bool(np.all(vis[live] > self.kc))
            )
            counts["sparse"] = sparse
            if sparse:
                v["ix_visc"][:] = np.where(live, self.kc, 0)
                lc = np.flatnonzero(live)
                chunks = self.kc // INDEX_KEYS
                rec = np.zeros((len(lc) * chunks, 8), np.int64)
                rec[:, 1] = L.INDEX_HEADS * np.repeat(lc, chunks)
                # query position kc - 1: every compact entry is visible to the slot
                rec[:, 2] = self.kc - 1
                rec[:, 3] = INDEX_KEYS * np.tile(np.arange(chunks), len(lc))
                rec[:, 4] = self.kc
                rec[:, 5] = np.repeat(lc, chunks)
                rec[:, 6] = 1
                n = self.ixc_launches(s)
                table = np.zeros((n * s.ix_slots, 8), np.int64)
                table[:, 0] = -1
                table[: len(rec)] = rec
                counts["ixc"] = -(-len(rec) // s.ix_slots)
                for j in range(n):
                    v[f"ixc.{j}"][:] = table[j * s.ix_slots : (j + 1) * s.ix_slots]
            per = HISTORY_TOKENS // ratio // PAGE_TOKENS  # pages per HISTORY block
            pr = np.where(
                hist[:, :, None] >= 0, hist[:, :, None] + PAIR_ROWS * np.arange(per), -1
            )
            blocks = v[f"cmp_blocks{ratio}"]
            blocks[~live] = -1
            if R == 1:  # every column is the one row's (a broadcast, not a gather)
                blocks[cols] = pr.reshape(R, -1)[0, : self.pages]
            else:
                blocks[cols] = pr.reshape(R, -1)[cr, : self.pages]
            # the tiled chain: how many groups this step needs, and each group's tables
            chunks = 0
            for b in blk:
                if b is not None:
                    chunks = max(chunks, -(-b[3] // INDEX_KEYS))
            used = -(-chunks // tp)
            counts[ratio] = used * (lpg + 1)
            counts["pf", ratio] = used * (lpg + 2)
            owners = [o for o in geo.owners if cfg.compress_ratios[o] == ratio]
            for g in range(self.ix_groups(s, ratio)):
                first = g * tp * INDEX_KEYS
                keys = np.clip(vis - first, 0, tp * INDEX_KEYS)
                v[f"ixs{ratio}"][g] = first
                v[f"ixg{ratio}"][g] = keys + (0 if g == 0 else L.INDEX_RUN)
                v[f"ixn{ratio}"][g] = np.minimum(vis, L.INDEX_RUN)
                if not s.prefill:
                    v[f"ixt{ratio}"][g] = keys
                    newest = np.where(vis > 0, (vis - 1) // 8 - first // 8, -1)
                    v[f"ixp{ratio}"][g] = np.where(
                        (newest >= 0) & (newest < tp * (INDEX_KEYS // 8)), newest, -1
                    )
            for g in range(self.ix_groups(s, ratio)):
                for l in range(lpg):
                    table = np.zeros((s.ix_slots, 8), np.int64)
                    table[:, 0] = -1
                    pages = {o: np.full((s.ix_slots, len(lanes)), -1, np.int64) for o in owners}
                    for si in range(s.ix_slots):
                        rg = l * s.ix_slots + si
                        tile = g * tp + rg // nblk
                        b = rg % nblk
                        first = tile * INDEX_KEYS
                        if blk[b] is None or blk[b][3] <= first:
                            continue
                        k, col, q, seen = blk[b]
                        ln = min(seen - first, INDEX_KEYS)
                        key0 = 0 if tile == 0 else L.INDEX_RUN + (tile - g * tp) * INDEX_KEYS
                        table[si] = [
                            0,
                            L.INDEX_HEADS * col,
                            q,
                            key0,
                            key0 + ln,
                            col,
                            ratio,
                            0,
                        ]
                        P = first // PAGE_TOKENS + lanes
                        ok = lanes < min(len(lanes), -(-ln // PAGE_TOKENS))
                        for o in owners:
                            op, ooff = geo.owner_pages(o), geo.owner_offset(o)
                            pages[o][si] = np.where(
                                ok,
                                hist[k, np.where(ok, P // op, 0)]
                                + ooff
                                + PAIR_ROWS * (P % op)
                                + PAGE_ROWS,
                                -1,
                            )
                    j = g * lpg + l
                    v[f"ix{ratio}.{j}"][:] = table
                    for o in owners:
                        v[f"ix{ratio}.{j}.{o}"][:] = pages[o]
        return counts

    def _engram_ids(
        self, rows: Sequence[Row], tokens: np.ndarray, T: int
    ) -> np.ndarray:
        """(T, Engram layers, J) i32 hashed rows of every column (padding: row 0)"""
        h = self._engram_hash()
        n = self.cfg.engram_max_ngram_size
        out = np.zeros((T, len(self.cfg.engram_layer_ids), J), np.int32)
        windows = np.zeros(
            (T, n), np.int64
        )  # each column's token and those before it, newest first
        blocked = np.zeros((T, n), bool)
        back = np.arange(n)
        for r in rows:
            c = slice(r.column, r.column + r.real)
            seq = np.concatenate(
                [np.asarray(r.before, np.int64), np.asarray(tokens[c], np.int64)]
            )
            at = len(r.before) + np.arange(r.real)[:, None] - back
            windows[c] = seq[np.maximum(at, 0)]
            blocked[c] = (r.pos0 + np.arange(r.real))[:, None] < back
        cols = np.concatenate([np.arange(r.column, r.column + r.real) for r in rows])
        out[cols] = h.hash(
            np.where(blocked[cols], h.pad_id, h.token_map[windows[cols]])
        )
        return out

    def _engram_older(self, rows: Sequence[Row], T: int, tokens=None) -> np.ndarray:
        """(T, Engram layers, J, 2) i32: per decode column and lane 8 (n - 2) + head, the XOR of the products of
        the n - 1 tokens before the column's (``Row.before``, then the row's earlier columns of the step:
        ``tokens``, needed when a row has more than one column), as (hi, lo); ``eg_hash`` adds the column's
        own."""
        h = self._engram_hash()
        n = self.cfg.engram_max_ngram_size
        older = np.full((T, n - 1), h.pad_id, np.int64)
        for r in rows:
            if r.real != 1 and tokens is None:
                raise ValueError(
                    "a step with several columns per row needs its tokens on the host"
                )
            seen = np.asarray(r.before, np.int64)
            for j in range(r.real):
                if j:
                    assert tokens is not None
                    seen = np.append(seen, tokens[r.column + j - 1])
                before = seen[::-1][: n - 1]  # newest first
                older[r.column + j, : len(before)] = h.token_map[before]
        xor = h.older(older)  # (T, layers, n - 1)
        heads = self.cfg.engram_n_heads
        return split64(np.repeat(xor, heads, axis=2))


class ReferenceModel(Reference, DeepSeekV41Gaudi):
    """A test model: the Python reference's methods on DeepSeekV41Gaudi."""


class EngramContext:
    """The ``n`` tokens before each position that Engram hashes, for every sequence of the runner.

    ``seen`` records the tokens of a sequence at their positions as they are committed, from its first token:
    tokens at a position it already holds replace those at that position and after. It also records the last ``n``
    tokens of each WINDOW block that they complete, by window block id. A sequence that the Engine starts from
    cached blocks (a prefix hit, at a block boundary) has no record of its own. ``before`` then takes its tokens
    from the record of the block before its first position, which is in its WINDOW table.

    A sequence's record can end one position past the next step's first position: a plain decode step records its
    sample at the position after its own (the input of the step that continues it on the device), and when the
    Engine instead starts the row's next step afresh, that step's first position is the sample's, with the sample
    as its host token. So a record keeps ``n + 1`` tokens, and ``before`` and ``seen`` go by position, not by the
    record's end.

    The Engine does not change a shared block, and the new owner of a free block writes it again before the
    block can be shared. Thus the record of a block that a plan names is the record of that block's tokens."""

    def __init__(self, n: int, window_blocks: int) -> None:
        if n < 1:
            raise ValueError("Engram hashes at least one token before a position")
        self.n = n
        self.last: dict[int, tuple[int, np.ndarray]] = {}
        """sequence id -> (end, its last ``n + 1`` recorded tokens, at positions ``end - len`` .. ``end - 1``)"""
        self._windows: dict[int, tuple[int, np.ndarray]] = {}
        """sequence id -> its WINDOW table of the latest plan: first logical block and the window block ids"""
        self._tails = np.full((window_blocks, n), -1, np.int64)
        """window block id -> the last ``n`` tokens of the block (-1: not recorded)"""

    def has(self, sequence: int) -> bool:
        """Whether ``sequence`` has a record (it is not released)."""
        return sequence in self.last

    def before(
        self, sequence: int, pos0: int, first: int, ids: np.ndarray
    ) -> np.ndarray:
        """The tokens at positions ``pos0 - n`` .. ``pos0 - 1`` that Engram hashes (those that exist), for a row
        whose WINDOW table starts at logical block ``first`` with window block ids ``ids``. They come from the
        sequence's record when it holds them, else from the record of the block before ``pos0``."""
        ids = np.asarray(ids, np.int64)
        self._windows[sequence] = (first, ids)
        if pos0 == 0:
            return _NONE
        start = pos0 - min(self.n, pos0)
        record = self.last.get(sequence)
        if record is not None:
            end, tokens = record
            if end - len(tokens) <= start and pos0 <= end:
                at = start - (end - len(tokens))
                return tokens[at : at + pos0 - start]
        return self._restore(sequence, pos0, first, ids)

    def _restore(
        self, sequence: int, pos0: int, first: int, ids: np.ndarray
    ) -> np.ndarray:
        block = pos0 // PAGE_TOKENS - 1 - first
        if pos0 % PAGE_TOKENS or not 0 <= block < len(ids):
            raise RuntimeError(
                f"sequence {sequence} starts at position {pos0}, which is not after a block in its WINDOW table"
            )
        tail = self._tails[ids[block]]
        if (tail < 0).any():
            raise RuntimeError(
                f"sequence {sequence} starts at position {pos0} after a block with no recorded tokens"
            )
        tail = tail[-min(self.n, pos0) :].copy()
        self.last[sequence] = (pos0, tail)
        logger.debug(
            f"sequence {sequence} starts from cached blocks at position {pos0} after tokens {tail.tolist()}"
        )
        return tail

    def seen(self, sequence: int, pos0: int, tokens) -> None:
        """Record ``tokens``, committed at positions ``pos0`` and after, in place of any tokens the record holds at
        ``pos0`` and after. Call after ``before`` for a row of this sequence whose WINDOW table covers those
        positions."""
        kept = _NONE  # the recorded tokens before pos0, when the record reaches pos0
        record = self.last.get(sequence)
        if record is not None and pos0 > 0:
            last_end, last = record
            if last_end - len(last) <= pos0 <= last_end:
                kept = last[: pos0 - (last_end - len(last))]
        full = np.concatenate([kept, np.asarray(tokens, np.int64)])
        start = pos0 - len(kept)  # the position of full[0]
        end = start + len(full)
        window = self._windows.get(sequence)
        if window is not None:
            first, ids = window
            for e in range(
                -(-(pos0 + 1) // PAGE_TOKENS) * PAGE_TOKENS, end + 1, PAGE_TOKENS
            ):
                block = e // PAGE_TOKENS - 1 - first
                if e - self.n >= start and 0 <= block < len(ids):
                    self._tails[ids[block]] = full[e - self.n - start : e - start]
        self.last[sequence] = (end, full[-(self.n + 1) :])

    def release(self, sequences) -> None:
        """Forget released sequences. The records of their blocks stay: cached blocks outlive their sequence."""
        for sequence in sequences:
            self.last.pop(int(sequence), None)
            self._windows.pop(int(sequence), None)
