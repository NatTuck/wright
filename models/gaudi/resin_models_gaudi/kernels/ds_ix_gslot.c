// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// DeepSeek-V4.1 tiled index chain, its group tables on the device: every slot record of every merge group,
// derived from the step's positions rather than built by the host fill (which built groups x T entries per
// layer). The host's per-step work stays O(T) (``blk``); everything per group is here.
//
// A merge group g covers tiles [g*tp, (g+1)*tp) of ``keys`` (2048) compressed entries. Its launch j (of lpg)
// holds ``slots`` slots; slot rg = j*slots + s of the group is query block b = rg % nblk of tile
// g*tp + rg / nblk, whose columns start at c0 = b*qb (a prefill step's blocks are qb columns each, a decode
// step's one column each, so c0 = b*qb in both). The record's lanes are what index_score_graph reads:
//   [0] group (0), [1] index_heads*c0 (qcol0), [2] q (qpos0), [3] key0, [4] key0 + len, [5] c0 (qtok0),
//   [6] ratio, [7] the tile's first index page (ds_ix_gather's page table base)
// with len = clamp(seen - first, 0, keys) the keys of the tile the block's last column can see (blk[1, b]).
// A slot with no key of its tile visible gets lane 0 = -1 and the rest 0 (the launches' "no keys" padding).
//
// It also publishes each group's first key (gbase), which the merge and the block kernels read, so the host
// writes no per-group value at all.
// tensors: 0 blk (2, T) i32 (per query block: q, seen) | 1 rec (8, groups*S2) i32 | 2 gbase (groups,) i32
// params: S2 (lpg*slots), nblk, qb, heads, ratio, keys, run, tp, page.  index space: [groups, S2]
#include "common.h"

void main(tensor blk, tensor rec, tensor gbase, int S2, int nblk, int qb, int heads, int ratio,
          int keys, int run, int tp, int page)
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const int64 lane = (int64)read_lane_id_4b_b();
    scalar_cache_reset();
    for (int g = off[0]; g < end[0]; g++) {
        if (off[1] == 0) {
            // gbase is (32, groups) col-major: gbase[0, g] lands at element g*32 (byte 128*g), the
            // 128-aligned per-group section the merge/block kernels bind (table + ix.gb + 128*g).
            // The old (groups, 32) + {g, 0} wrote gbase[g, 0] at element g (byte 4*g), so every group
            // g>0 read unwritten padding (=0) as its base and the merge listed only group 0's keys.
            int5 gc = {0, g, 0, 0, 0};
            s_i32_st_g(gen_addr(gc, gbase), (int)(g * tp * keys));
        }
        for (int rg = off[1]; rg < end[1]; rg++) {
            const int b = rg % nblk;
            const int tile = g * tp + rg / nblk;
            const int first = tile * keys;
            const int c0 = b * qb;
            // ``blk`` is one row per QUERY BLOCK, not per column: block b's row is b (the fill writes
            // (rows.column + b0) / qb, which is b).  Reading row b*qb -- the block's first column -- is the
            // same row only when qb == 1, i.e. in a decode step; in a prefill step it reads a row the fill
            // never wrote, so every block past the first gets len = 0 and the whole chain lists nothing.
            int5 bc = {0, b, 0, 0, 0};
            const int q = s_i32_ld_g(gen_addr(bc, blk));
            bc[0] = 1;
            const int seen = s_i32_ld_g(gen_addr(bc, blk));
            int len = seen - first;
            len = len < 0 ? 0 : (len > keys ? keys : len);
            // the tile-LOCAL score row of the tile's first key. The score launches write ix_tile
            // (a (rows, T) buffer, column stride rows) at rows 0 .. rows for EVERY group; the merge
            // op's copy then places that tile at the group's merge-buffer rows (row0 = 0 for group 0,
            // `run` for a later group's running-candidate prefix). So key0 carries NO `run` offset —
            // the copy's destination adds it. The old `tile==0 ? 0 : run + ...` added `run` to every
            // tile>0: group 0's tiles>0 landed +run rows high (the merge listed each key `run`
            // positions too high, dropping the decode needle), and a later group's acc wrote rows
            // run .. run+rows — overlapping group 0's 0 .. rows and overflowing the rows-sized tile —
            // so its copy (reading 0 .. rows) re-listed group 0's keys and corrupted prefill merge.
            const int key0 = (tile - g * tp) * keys;
            const int live = len > 0;
            // a padding slot is lane 0 = -1 and lanes 1 .. 7 = 0, the launches' "no keys" record
            int64 v = 0;
            v = v_i32_sel_eq_i32_b(lane, (int64)0, (int64)(live ? 0 : -1), v);
            v = v_i32_sel_eq_i32_b(lane, (int64)1, (int64)(live ? heads * c0 : 0), v);
            v = v_i32_sel_eq_i32_b(lane, (int64)2, (int64)(live ? q : 0), v);
            v = v_i32_sel_eq_i32_b(lane, (int64)3, (int64)(live ? key0 : 0), v);
            v = v_i32_sel_eq_i32_b(lane, (int64)4, (int64)(live ? key0 + len : 0), v);
            v = v_i32_sel_eq_i32_b(lane, (int64)5, (int64)(live ? c0 : 0), v);
            v = v_i32_sel_eq_i32_b(lane, (int64)6, (int64)(live ? ratio : 0), v);
            v = v_i32_sel_eq_i32_b(lane, (int64)7, (int64)(live ? first / page : 0), v);
            int5 rc = {0, g * S2 + rg, 0, 0, 0};
            v_i32_st_tnsr_partial(rc, rec, v, 7, 0);
        }
    }
}
