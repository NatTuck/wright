// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// DeepSeek-V4.1 online-attention finalize after the query tile's last key block: add the per-head
// sink to the denominator ONLY (never to the max), normalize, inverse-RoPE, output f32:
//   den = sum[h, c] + exp(sink[h] - max[h, c])
//   out[:, h, c] = invRoPE( acc_o[:, h, c] / den )    (f32, NO bf16 cast — see below)
// A column with no valid key (sum = 0, max = -1e30) outputs exactly 0 (the ds_attn_fused guard:
// inv = 0 when the keys-only sum < 1e-30; the official 0 / exp(sink + 1e30) = 0 agrees).
// Inverse RoPE MATCHES ds_attn_fused.c / ds_rope.c exactly: adjacent pairs (2i, 2i+1) of the LAST 64
// rows of the D = 512 latent, out[l] = v[l] * co[l] + v[l^1] * si[l] with the pair-partner shuffle.
// cs_inv (128, Q) column c holds query position c's INVERSE table: co lane l = cos_(l>>1), si lane
// l = (l odd ? -1 : +1) * sin_(l>>1) — the sign-flipped si half of ds_rope's forward cs table; the
// rotation direction lives in the table, the kernel formula is identical to the forward one.
// The divide, the rotation AND the output are f32 (ds_attn_fused convention): the o_inv fidelity gate
// reads this tensor BEFORE wo (norm_rel <= 1e-6 vs the f64 oracle, no bf16 cast), and _attention_out
// quantizes o_inv to fp8 for wo_a downstream (coarser than bf16) — a bf16 cast here would be
// precision-redundant and fail the gate. o stays f32; the persistent o_out buffer is f32 (D, nh, T).
// tensors: 0 acc_o (D, nh, Q) f32  1 max (nh, Q) f32  2 sum (nh, Q) f32  3 sink (nh,) f32
//          4 cs_inv (128, Q) f32 [inverse RoPE] | 5 o (D, nh, Q) f32
// params: none. index space: [nh, Q]. D a multiple of 64, at most 512 (design D = 512).
#include "common.h"

#define MAXV 8

// exp accurate to ~1e-7 (fp32 limit); see ds_slot_fused.c.
static inline float64 exp8(float64 x)
{
    x = v_f32_max_b(v_f32_min_b(x, 88.0f), -87.0f);
    const float64 t = x * 1.4426950408889634f;
    const int64 n = v_convert_f32_to_i32_b(t, SW_RD, 0, 1, 0);
    const float64 nf = v_convert_i32_to_f32_b(n, SW_RHNE, 0, 1, 0);
    const float64 r = v_f32_mac_b(nf, -0.693145751953125f, x);
    const float64 r2 = v_f32_mac_b(nf, -1.42860682030941723212e-6f, r);
    float64 pl = 1.0f / 40320.0f;
    pl = v_f32_mac_b(pl, r2, 1.0f / 5040.0f);
    pl = v_f32_mac_b(pl, r2, 1.0f / 720.0f);
    pl = v_f32_mac_b(pl, r2, 1.0f / 120.0f);
    pl = v_f32_mac_b(pl, r2, 1.0f / 24.0f);
    pl = v_f32_mac_b(pl, r2, 1.0f / 6.0f);
    pl = v_f32_mac_b(pl, r2, 0.5f);
    pl = v_f32_mac_b(pl, r2, 1.0f);
    pl = v_f32_mac_b(pl, r2, 1.0f);
    const int64 e = v_i32_shl_b(n + 127, (int64)23);
    return pl * as_float64(e);
}

void main(tensor acc_o, tensor mx, tensor sm, tensor sink, tensor cs_inv, tensor o)
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const int D = get_dim_size(acc_o, 0), nv = D / 64;
    const int64 lane = (int64)read_lane_id_4b_b();
    const int64 e = (lane ^ 1) & 15;                                   // pair partner within the dual group
    const int64 ctl = (0x80 | ((e >> 3) << 5) | (e & 7)) * 0x01010101;
    const uchar256 SW = __builtin_bit_cast(uchar256, ctl);
    scalar_cache_reset();
    for (int c = off[1]; c < end[1]; c++) {
        for (int h = off[0]; h < end[0]; h++) {
            int5 stc = {h, c, 0, 0, 0};
            const float M = s_f32_ld_g(gen_addr(stc, mx));
            const float L = s_f32_ld_g(gen_addr(stc, sm));
            int5 hc = {h, 0, 0, 0, 0};
            const float sk = s_f32_ld_g(gen_addr(hc, sink));
            const float64 den = (float64)L + exp8((float64)sk - (float64)M);
            const float64 inv = v_f32_sel_less_f32_b((float64)L, 1.0e-30f, 0.0f, nl_recip(den));
            int5 tc = {0, c, 0, 0, 0};
            const float64 co = v_f32_ld_tnsr_b(tc, cs_inv);
            tc[0] = 64;
            const float64 si = v_f32_ld_tnsr_b(tc, cs_inv);
            float64 v[MAXV];
            #pragma unroll
            for (int j = 0; j < MAXV; j++) {
                int5 oc = {64 * j, h, c, 0, 0};
                v[j] = 64 * j < D ? v_f32_ld_tnsr_b(oc, acc_o) * inv : 0.0f;
            }
            #pragma unroll
            for (int j = 0; j < MAXV; j++)
                if (j == nv - 1)
                    v[j] = v_f32_mac_b(v_f32_shuffle_b(v[j], SW, 0, v[j]), si, v[j] * co);
            #pragma unroll
            for (int j = 0; j < MAXV; j++) {
                if (64 * j < D) {
                    int5 oc = {64 * j, h, c, 0, 0};
                    v_f32_st_tnsr(oc, o, v[j]);
                }
            }
        }
    }
}
