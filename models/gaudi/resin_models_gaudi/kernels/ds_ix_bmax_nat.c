// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// DeepSeek-V4.1 candidate blocks, level one, NATURAL block order: bmax[b] = block b's max (row == block), feeding
// the running block merge (ds_ix_merge), whose row-based vis/position arithmetic requires row == block. It reads
// the tile's (K, T) scores directly (lane == key), butterfly-reduces each contiguous block of 8 lanes into the
// permuted lane order (exactly ds_ix_bmax: lane 8k+j holds block 8j+k), then transposes that 8x8 to natural
// (nat lane k = out lane perm(k), perm an involution, so nat lane k holds block k). The column's pin block holds
// +3e38; blocks with no visible entry hold -3e38.
// tensors: 0 score (K, T) f32  1 ixvis (T) i32  2 gbase (1,) i32 | 3 bmax (K/8, T) f32
// index space: [K / 512, T]
#include "common.h"

#define NINF (-3.0e38f)
#define PIN 3.0e38f

static inline uchar256 ctl_of(int64 e)
{
    const int64 c = (0x80 | ((e >> 3) << 5) | (e & 7)) * 0x01010101;
    return __builtin_bit_cast(uchar256, c);
}

void main(tensor score, tensor ixvis, tensor gbase, tensor bmax)
{
    int5 zc = {0, 0, 0, 0, 0};
    const int base = s_i32_ld_g(gen_addr(zc, gbase));
    const int K = (int)get_dim_size(score, 0);
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const int64 lane = (int64)read_lane_id_4b_b();
    const float64 lanef = v_convert_i32_to_f32_b(lane, SW_RHNE, 0, 1, 0);
    const float64 l8f = v_convert_i32_to_f32_b(lane & 7, SW_RHNE, 0, 1, 0);
    const int64 e = lane & 15;
    const uchar256 SH1 = ctl_of(e ^ 1), SH2 = ctl_of(e ^ 2), SH4 = ctl_of(e ^ 4);
    scalar_cache_reset();
    for (int c = off[1]; c < end[1]; c++) {
        int5 vc = {c, 0, 0, 0, 0};
        const int seen = s_i32_ld_g(gen_addr(vc, ixvis));
        int keys = seen - base;
        keys = keys < 0 ? 0 : (keys > K ? K : keys);
        const int lim = keys;
        const float limf = (float)lim;
        const int pinned = seen > 0 ? (seen - 1) / 8 - base / 8 : -1;
        const int last = pinned >= 0 && pinned < (int)get_dim_size(bmax, 0) ? pinned : -1;
        for (int g = off[0]; g < end[0]; g++) {
            float64 out = NINF;
            if (g * 512 < lim) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    const int e0 = g * 512 + 64 * j;
                    int5 sc = {e0, c, 0, 0, 0};
                    float64 v = v_f32_sel_less_f32_b(lanef + (float)e0, limf, v_f32_ld_tnsr_b(sc, score), NINF);
                    v = v_f32_max_b(v, v_f32_shuffle_b(v, SH1, 0, v));
                    v = v_f32_max_b(v, v_f32_shuffle_b(v, SH2, 0, v));
                    v = v_f32_max_b(v, v_f32_shuffle_b(v, SH4, 0, v));
                    out = v_f32_sel_eq_f32_b(l8f, (float)j, v, out);  // permuted: lane 8k+j holds block 8j+k
                }
                const int b = last - 64 * g;
                if (b >= 0 && b < 64)
                    out = v_f32_sel_eq_f32_b(lanef, (float)(8 * (b & 7) + (b >> 3)), PIN, out);
                // transpose permuted -> natural: nat lane k = out lane perm(k); perm(perm(k)) = k, so nat lane k
                // holds block perm(perm(k)) = block k.
                float64 nat = NINF;
                #pragma unroll
                for (int k = 0; k < 64; k++) {
                    const int pk = 8 * (k & 7) + (k >> 3);
                    nat = v_f32_sel_eq_f32_b(lanef, (float)k, lane_bcast(out, pk), nat);
                }
                out = nat;
            }
            int5 oc = {64 * g, c, 0, 0, 0};
            v_f32_st_tnsr(oc, bmax, out);
        }
    }
}