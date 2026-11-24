// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// DeepSeek-V4.1 candidate blocks, level two: a column's scores with every visible entry outside its candidate
// blocks set to -3e38. Entry i is in a candidate block when mkey(bmax of block i / 8) >= cthr[c] (ix_bmax's
// block order; cthr from ix_thr over the blocks). Vectors past the visible entries are not written.
// The group's tile-local visible keys are derived here from the column's visible entries (``ixvis``, the step's
// O(T) table) and the group's first key (``gbase``): vis = clamp(seen - base, 0, K). No per-group host table.
// tensors: 0 score (K, T) f32  1 ixvis (T) i32  2 bmax (K / 8, T) f32  3 cthr (64, T) i32  4 gbase (1,) i32
//        | 5 out (K, T) f32   (inputs first, then the output: the graph binds tensors by position)
// index space: [K / 512, T]
#include "common.h"

#define NINF (-3.0e38f)

static inline uint64 mkey(float64 v)
{
    const uint64 b = as_uint64(v);
    return as_uint64(v_f32_sel_less_f32_b(v, 0.0f, as_float64(~b), as_float64(b | (uint64)0x80000000u)));
}

// lane l <- element e (0..15) of its dual group
static inline uchar256 ctl_of(int64 e)
{
    const int64 c = (0x80 | ((e >> 3) << 5) | (e & 7)) * 0x01010101;
    return __builtin_bit_cast(uchar256, c);
}

void main(tensor score, tensor ixvis, tensor bmax, tensor cthr, tensor gbase, tensor out)
{
    int5 zc = {0, 0, 0, 0, 0};
    const int base = s_i32_ld_g(gen_addr(zc, gbase));
    const int K = (int)get_dim_size(score, 0);
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const int64 lane = (int64)read_lane_id_4b_b();
    const int64 e8 = lane & 8;
    const uint64 one = as_uint64((float64)1.0f);
    uchar256 S[8];  // lane 8 k + i <- lane 8 k + j: block 8 j + k of the group
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
        int5 tc = {0, c, 0, 0, 0};
        const uint64 th = v_u32_ld_tnsr_b(tc, cthr);
        for (int g = off[0]; g < end[0]; g++) {
            if (g * 512 >= lim)
                break;
            int5 bc = {64 * g, c, 0, 0, 0};
            const float64 keep = as_float64(v_u32_sel_less_u32_b(mkey(v_f32_ld_tnsr_b(bc, bmax)), th, (uint64)0, one));
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int5 sc = {g * 512 + 64 * j, c, 0, 0, 0};
                const float64 k = v_f32_shuffle_b(keep, S[j], 0, keep);
                v_f32_st_tnsr(sc, out, v_f32_sel_eq_f32_b(k, 1.0f, v_f32_ld_tnsr_b(sc, score), NINF));
            }
        }
    }
}
