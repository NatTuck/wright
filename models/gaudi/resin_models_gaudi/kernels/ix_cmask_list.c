// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// DeepSeek-V4.1 candidate blocks, level two, LIST membership: a column's scores with every visible entry outside
// its candidate blocks set to -3e38. The candidate blocks are the running top-2048 block list (cand_list) the
// source layer selected — no full (kb, T) block-maxima array is published. Entry i is kept when its block i/8 is
// one of the column's cand_list[:n] blocks (n = cand_slots[4]). Like ix_cmask, vectors past the visible entries are
// not written; the tile-local visible keys come from ixvis and gbase. Each lane holds one block of the group and
// scans the candidate list (broadcast each listed block, compare), so this is the rare mixed-batch fallback path.
// tensors: 0 score (K, T) f32  1 ixvis (T) i32  2 cand_list (top, T) i32  3 cand_slots (8, T) i32
//          4 gbase (1,) i32 | 5 out (K, T) f32
// index space: [K / 512, T]
#include "common.h"

#define NINF (-3.0e38f)

// lane l <- element e (0..15) of its dual group
static inline uchar256 ctl_of(int64 e)
{
    const int64 c = (0x80 | ((e >> 3) << 5) | (e & 7)) * 0x01010101;
    return __builtin_bit_cast(uchar256, c);
}

void main(tensor score, tensor ixvis, tensor cand_list, tensor cand_slots, tensor gbase, tensor out)
{
    int5 zc = {0, 0, 0, 0, 0};
    const int base = s_i32_ld_g(gen_addr(zc, gbase));
    const int K = (int)get_dim_size(score, 0);
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const int64 lane = (int64)read_lane_id_4b_b();
    const int64 e8 = lane & 8;
    const uint64 one = as_uint64((float64)1.0f);
    uchar256 S[8];  // same entry-mapping as ix_cmask: keep is in the permuted block order
    #pragma unroll
    for (int j = 0; j < 8; j++)
        S[j] = ctl_of(e8 | j);
    scalar_cache_reset();
    for (int c = off[1]; c < end[1]; c++) {
        int5 vc = {c, 0, 0, 0, 0};
        const int seen = s_i32_ld_g(gen_addr(vc, ixvis));
        int keys = seen - base;
        keys = keys < 0 ? 0 : (keys > K ? K : keys);
        const int lim = keys;
        int5 nc = {4, c, 0, 0, 0};
        const int n = s_i32_ld_g(gen_addr(nc, cand_slots));
        for (int g = off[0]; g < end[0]; g++) {
            if (g * 512 >= lim)
                break;
            // keep[lane] in ix_cmask's permuted order: block 64g + 8*(lane%8) + lane/8 (matches S[j] above)
            const int64 b = 64 * g + 8 * (lane & 7) + (lane >> 3);
            uint64 keep = 0;
            for (int j0 = 0; j0 < n; j0 += 64) {
                int5 lc = {j0, c, 0, 0, 0};
                const int64 cl = v_i32_ld_tnsr_b(lc, cand_list);
                const int rem = n - j0 < 64 ? n - j0 : 64;
                for (int k = 0; k < rem; k++) {
                    const int64 v = as_int64(lane_bcast(as_float64(cl), k));
                    keep = v_u32_or_b(keep, v_u32_sel_eq_u32_b(as_uint64(v), as_uint64(b), (uint64)0, one));
                }
            }
            const float64 keepf = as_float64(keep);
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int5 sc = {g * 512 + 64 * j, c, 0, 0, 0};
                const float64 k = v_f32_shuffle_b(keepf, S[j], 0, keepf);
                v_f32_st_tnsr(sc, out, v_f32_sel_eq_f32_b(k, 1.0f, v_f32_ld_tnsr_b(sc, score), NINF));
            }
        }
    }
}