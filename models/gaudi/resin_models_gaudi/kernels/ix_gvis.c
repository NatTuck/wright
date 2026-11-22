// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// DeepSeek-V4.1 tiled index chain: one merge group's visible score rows and the column's exact count, derived
// from the column's visible entries (``ix_vis``) and the group's first key (``base``) instead of from a per-group
// table the host built (which cost groups x T entries per ratio per layer).
//   vis   = clamp(seen - base, 0, rows) + (base > 0 ? run : 0)   the rows of the group's merge buffer
//   exact = min(seen, KS)                                        the column's count in its list
// tensors: 0 ixvis (T,) i32  1 base (1,) i32 | 2 vis (T,) i32  3 exact (T,) i32
// params: run, rows, KS.  index space: [T]
#include "common.h"

void main(tensor ixvis, tensor base, tensor vis, tensor exact, int run, int rows, int KS)
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    int5 zc = {0, 0, 0, 0, 0};
    const int b = s_i32_ld_g(gen_addr(zc, base));
    const int row0 = b > 0 ? run : 0;
    scalar_cache_reset();
    for (int c = off[0]; c < end[0]; c++) {
        int5 vc = {c, 0, 0, 0, 0};
        const int seen = s_i32_ld_g(gen_addr(vc, ixvis));
        int keys = seen - b;
        keys = keys < 0 ? 0 : (keys > rows ? rows : keys);
        int5 oc = {c, 0, 0, 0, 0};
        s_i32_st_g(gen_addr(oc, vis), keys + row0);
        s_i32_st_g(gen_addr(oc, exact), seen < KS ? seen : KS);
    }
}
