// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// DeepSeek-V4.1 fused attention merge: the window and compressed listed parts were softmaxed together
// at one global max M (ds_slot_fused), so their unnormalized P.V results pv1 / pv2 and FP32 sums
// lsw / lsc simply add. Per column c (head h = c % hpg, token t = c / hpg):
//   out = (pv1 + pv2) / (lsw + lsc + exp(sink[h] - M))
// then the inverse RoPE on each head's last 64 rows (cs (128, T), as ds_attn_comb_r). M is the keys-only
// global max from ds_slot_fused; the sink enters only the denominator. No normalize-then-denormalize.
// tensors: 0 pv1 (D, W, G) f32  1 pv2 (D, W, G) f32  2 mx (Mp, T) f32  3 lsw (Mp, T) f32
//          4 lsc (Mp, T) f32  5 sink (hpg) f32  6 cs (128, T) f32 | 7 out (D, W, G) f32
// params: hpg. index space: [W, G]
#include "common.h"

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

// lane k of v broadcast to all lanes
#define GET_LANE(v, k, out)                                                                  \
    { const float lo_ = (float)(k) - 0.5f, hi_ = (float)(k) + 0.5f;                          \
      float64 t_ = v_f32_sel_less_f32_b(lanef, lo_, 0.0f, v_f32_sel_less_f32_b(lanef, hi_, v, 0.0f)); \
      NL_REDUCE_ADD(t_, IDX, out) }

void main(tensor pv1, tensor pv2, tensor mx, tensor lsw, tensor lsc, tensor sink, tensor cs,
          tensor out, int hpg)
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const int D = get_dim_size(out, 0);
    const float64 lanef = v_convert_i32_to_f32_b((int64)read_lane_id_4b_b(), SW_RHNE, 0, 1, 0);
    DECL_IDX(IDX)
    const int64 lane = (int64)read_lane_id_4b_b();
    const int64 e = (lane ^ 1) & 15;                                   // pair partner within the dual group
    const int64 ctl = (0x80 | ((e >> 3) << 5) | (e & 7)) * 0x01010101;
    const uchar256 SW = __builtin_bit_cast(uchar256, ctl);
    scalar_cache_reset();
    for (int g = off[1]; g < end[1]; g++) {
        for (int c = off[0]; c < end[0]; c++) {
            const int h = c % hpg, t = c / hpg;
            int5 sc = {h & ~63, t, 0, 0, 0};
            float64 M, lw, lc;
            GET_LANE(v_f32_ld_tnsr_b(sc, mx), h & 63, M)
            GET_LANE(v_f32_ld_tnsr_b(sc, lsw), h & 63, lw)
            GET_LANE(v_f32_ld_tnsr_b(sc, lsc), h & 63, lc)
            int5 hc = {h, 0, 0, 0, 0};
            const float sk = s_f32_ld_g(gen_addr(hc, sink));
            const float64 den = lw + lc + exp8((float64)sk - M);
            const float64 inv = v_f32_sel_less_f32_b(lw + lc, 1.0e-30f, 0.0f, nl_recip(den));
            int5 tc = {0, t, 0, 0, 0};
            const float64 co = v_f32_ld_tnsr_b(tc, cs);
            tc[0] = 64;
            const float64 si = v_f32_ld_tnsr_b(tc, cs);
            for (int d = 0; d < D; d += 64) {
                int5 oc = {d, c, g, 0, 0};
                float64 v = (v_f32_ld_tnsr_b(oc, pv1) + v_f32_ld_tnsr_b(oc, pv2)) * inv;
                if (d == D - 64)
                    v = v_f32_mac_b(v_f32_shuffle_b(v, SW, 0, v), si, v * co);
                v_f32_st_tnsr(oc, out, v);
            }
        }
    }
}