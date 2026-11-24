// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// DeepSeek-V4.1 unrolled index chain: assemble one tile's merge buffer on the device — the running set's
// scores (``run_scores``, the previous merge's top-``run``) in rows 0 .. run, then the tile's own scores
// (``tile_scores``, written by the acc) in rows run .. run + rows. This replaces the host copy_multi that
// placed a group's tile into its merge buffer: the chain unrolls many tiles in one recipe, so the placement
// must be a graph node, and keeping the running set in its own buffer (the merge's sout) means the merge
// buffer has a single producer (this kernel) — no read-modify-write section. Rows past run + rows (the first
// tile has run = 0 and a shorter visible prefix) are NINF so a merge ranks them last.
// tensors: 0 run_scores (RUN, T) f32  1 tile_scores (rows, T) f32 | 2 out (K, T) f32
// params: run (0 for the first tile, else RUN), rows.  index space: [K / 64, T] (one member per 64-row block).
#include "common.h"

#define NINF (-3.0e38f)

void main(tensor run_scores, tensor tile_scores, tensor out, int run, int rows)
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const float64 ninf = NINF;
    scalar_cache_reset();
    for (int c = off[1]; c < end[1]; c++) {
        for (int b = off[0]; b < end[0]; b++) {
            const int row = b * 64;
            int5 oc = {row, c, 0, 0, 0};
            float64 v;
            if (row < run) {
                int5 rc = {row, c, 0, 0, 0};
                v = v_f32_ld_tnsr_b(rc, run_scores);
            } else if (row - run < rows) {
                int5 tc = {row - run, c, 0, 0, 0};
                v = v_f32_ld_tnsr_b(tc, tile_scores);
            } else {
                v = ninf;
            }
            v_f32_st_tnsr(oc, out, v);
        }
    }
}