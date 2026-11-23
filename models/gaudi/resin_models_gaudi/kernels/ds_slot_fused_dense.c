// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// DeepSeek-V4.1 DENSE prefill fused single-pass global-max softmax over the window keys and every visible
// compressed entry of a query block (each column's top-512 list as a mask), matching the official sparse_attn
// for <= 64 visible keys. The window scores scw (C, M, S) are masked causally from wslots (8, S); the dense
// compressed scores scc (K, M, S) by ds_slot_pk's KEPT rule (visible AND index score above the column's
// threshold, or equal at a position <= the tie cut). The global max M is over BOTH kept key sets (keys only;
// the sink is added to the denominator in ds_attn_fused_dense). P = bf16(exp8(sc*scale-M)) natural order for
// the numerator; ls = FP32 sum for the denominator. mx/lsw/lsc are per-column (M, S) (lane r = column r).
// slots wslots (8, S): [group, qcol0, qpos0, key0, klen, 0, kmin, win]; cslots (8, S):
//   [group, qcol0, qpos0, key0, klen, qtok0, ratio, 0].
// tensors: 0 scw (C, M, S) f32  1 scc (K, M, S) f32  2 wslots (8, S) i32  3 cslots (8, S) i32
//          4 score (scores, T) f32  5 thr (64, T) i32  6 lslots (8, T) i32
//          | 7 Pw (C, M, S) bf16  8 Pc (K, M, S) bf16  9 mx (M, S) f32  10 lsw (M, S) f32  11 lsc (M, S) f32
// params: scale, hpg. index space: [ceil(M / 64), S]
#include "common.h"

#define NINF (-3.0e38f)

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

static inline uint64 mkey(float64 v)
{
    const uint64 b = as_uint64(v);
    return as_uint64(v_f32_sel_less_f32_b(v, 0.0f, as_float64(~b), as_float64(b | (uint64)0x80000000u)));
}

// 1.0f where dense entry i of the column is visible (i < lim) and listed, else 0 (ds_slot_pk's KEPT)
#define KEPT(li, i)                                                                                  \
    ({ int5 kc_ = {ckey0 + (i), col, 0, 0, 0};                                                        \
       const uint64 k_ = mkey(v_f32_ld_tnsr_b(kc_, score));                                         \
       const float64 tie_ = v_f32_sel_grt_f32_b(li + (float)ckey0, cutf, 0.0f, 1.0f);              \
       const float64 eq_ = as_float64(v_u32_sel_eq_u32_b(k_, th, as_uint64(tie_), (uint64)0));     \
       const float64 gt_ = as_float64(v_u32_sel_less_u32_b(th, k_, one, (uint64)0));              \
       v_f32_sel_less_f32_b(li, climf, v_f32_max_b(gt_, eq_), 0.0f); })

void main(tensor scw, tensor scc, tensor wslots, tensor cslots, tensor score, tensor thr, tensor lslots,
          tensor Pw, tensor Pc, tensor mx, tensor lsw, tensor lsc, float scale, int hpg)
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const int C = get_dim_size(scw, 0), K = get_dim_size(scc, 0), M = get_dim_size(scw, 1);
    const float64 lanef = v_convert_i32_to_f32_b((int64)read_lane_id_4b_b(), SW_RHNE, 0, 1, 0);
    const uint64 one = as_uint64((float64)1.0f);
    DECL_IDX(IDX)
    scalar_cache_reset();
    for (int s = off[1]; s < end[1]; s++) {
        int5 wc = {2, s, 0, 0, 0};
        const int wqpos0 = s_i32_ld_g(gen_addr(wc, wslots));
        wc[0] = 3;
        const int wkey0 = s_i32_ld_g(gen_addr(wc, wslots));
        wc[0] = 4;
        const int wklen = s_i32_ld_g(gen_addr(wc, wslots));
        wc[0] = 6;
        const int wkmin = s_i32_ld_g(gen_addr(wc, wslots));
        wc[0] = 7;
        const int win = s_i32_ld_g(gen_addr(wc, wslots));
        int5 fc = {2, s, 0, 0, 0};
        const int cqpos0 = s_i32_ld_g(gen_addr(fc, cslots));
        fc[0] = 3;
        const int ckey0 = s_i32_ld_g(gen_addr(fc, cslots));
        fc[0] = 4;
        const int cklen = s_i32_ld_g(gen_addr(fc, cslots));
        fc[0] = 5;
        const int tok0 = s_i32_ld_g(gen_addr(fc, cslots));
        fc[0] = 6;
        int ratio = s_i32_ld_g(gen_addr(fc, cslots));
        ratio = ratio > 0 ? ratio : 1;
        for (int cb = off[0]; cb < end[0]; cb++) {
            float64 mv = NINF * scale, lwv = 0.0f, lcv = 0.0f;
            int t = 0, h = cb * 64;
            while (h >= hpg) { h -= hpg; t++; }
            for (int r = 0; r < 64 && cb * 64 + r < M; r++) {
                const int m = cb * 64 + r;
                // window causal mask for this column's token t
                const int wpos = wqpos0 + t;
                int wlim = wpos + 1 < wklen ? wpos + 1 : wklen;
                wlim -= wkey0;
                wlim = wlim < 0 ? 0 : (wlim > C ? C : wlim);
                const int wlof0 = wkmin > wkey0 ? wkmin - wkey0 : 0;
                const int wfirst = wpos - win + 1 - wkey0;
                int wlof = win > 0 && wfirst > wlof0 ? wfirst : wlof0;
                wlof = wlof < 0 ? 0 : wlof;
                const float wlimf = (float)wlim, wloff = (float)wlof;
                // dense compressed mask
                const int seen = (cqpos0 + t + 1) / ratio;
                int clim = seen < cklen ? seen : cklen;
                clim -= ckey0;
                clim = clim < 0 ? 0 : (clim > K ? K : clim);
                const float climf = (float)clim;
                const int col = tok0 + t;
                int5 tc = {0, col, 0, 0, 0};
                const uint64 th = __builtin_bit_cast(uint64, v_i32_ld_tnsr_b(tc, thr));
                int5 cc = {7, col, 0, 0, 0};
                const int cut = s_i32_ld_g(gen_addr(cc, lslots));
                const float cutf = cut > 16777216 ? 3.0e38f : (float)cut;
                // pass 1: global max over the visible window keys and the kept dense entries
                float64 mw = NINF;
                int5 cw = {0, m, s, 0, 0};
                for (int i = 0; i < C; i += 64) {
                    cw[0] = i;
                    const float64 li = lanef + (float)i;
                    const float64 v = v_f32_ld_tnsr_b(cw, scw);
                    mw = v_f32_max_b(mw, v_f32_sel_less_f32_b(li, wloff, NINF,
                                     v_f32_sel_less_f32_b(li, wlimf, v, NINF)));
                }
                float64 mc = NINF;
                int5 cd = {0, m, s, 0, 0};
                for (int i = 0; i < clim; i += 64) {
                    cd[0] = i;
                    const float64 li = lanef + (float)i;
                    mc = v_f32_max_b(mc, v_f32_sel_eq_f32_b(KEPT(li, i), 1.0f, v_f32_ld_tnsr_b(cd, scc), NINF));
                }
                float64 mm = v_f32_max_b(mw, mc);
                NL_REDUCE_MAX(mm, IDX, mm);
                const float64 ms = mm * scale;
                // pass 2: bf16 probabilities (numerator) and FP32 sums (denominator)
                float64 sw = 0.0f, sc = 0.0f;
                int5 pw = {0, m, s, 0, 0};
                for (int i = 0; i < C; i += 128) {
                    float64 ea = 0.0f, eb = 0.0f;
                    if (i < wlim) {
                        const float64 li = lanef + (float)i;
                        cw[0] = i;
                        const float64 va = v_f32_ld_tnsr_b(cw, scw);
                        cw[0] = i + 64;
                        const float64 vb = v_f32_ld_tnsr_b(cw, scw);
                        ea = v_f32_sel_less_f32_b(li, wlimf, exp8(v_f32_mac_b(va, scale, -ms)), 0.0f);
                        eb = v_f32_sel_less_f32_b(li + 64.0f, wlimf, exp8(v_f32_mac_b(vb, scale, -ms)), 0.0f);
                        ea = v_f32_sel_less_f32_b(li, wloff, 0.0f, ea);
                        eb = v_f32_sel_less_f32_b(li + 64.0f, wloff, 0.0f, eb);
                        sw = sw + ea + eb;
                    }
                    pw[0] = i;
                    v_bf16_st_tnsr(pw, Pw, cvt_bf16_lin(ea, eb));
                }
                int5 pc = {0, m, s, 0, 0};
                for (int i = 0; i < K; i += 128) {
                    float64 ea = 0.0f, eb = 0.0f;
                    if (i < clim) {
                        const float64 li = lanef + (float)i;
                        cd[0] = i;
                        const float64 va = v_f32_ld_tnsr_b(cd, scc);
                        cd[0] = i + 64;
                        const float64 vb = v_f32_ld_tnsr_b(cd, scc);
                        ea = v_f32_sel_eq_f32_b(KEPT(li, i), 1.0f, exp8(v_f32_mac_b(va, scale, -ms)), 0.0f);
                        eb = v_f32_sel_eq_f32_b(KEPT(li + 64.0f, i + 64), 1.0f, exp8(v_f32_mac_b(vb, scale, -ms)), 0.0f);
                        sc = sc + ea + eb;
                    }
                    pc[0] = i;
                    v_bf16_st_tnsr(pc, Pc, cvt_bf16_lin(ea, eb));
                }
                float64 tw, tc2;
                NL_REDUCE_ADD(sw, IDX, tw);
                NL_REDUCE_ADD(sc, IDX, tc2);
                mv = v_f32_sel_less_f32_b(lanef, (float)r - 0.5f, mv, ms);   // lanes >= r
                lwv = v_f32_sel_less_f32_b(lanef, (float)r - 0.5f, lwv, tw);
                lcv = v_f32_sel_less_f32_b(lanef, (float)r - 0.5f, lcv, tc2);
                if (++h == hpg) { h = 0; t++; }
            }
            int5 stc = {cb * 64, s, 0, 0, 0};
            v_f32_st_tnsr(stc, mx, mv);
            v_f32_st_tnsr(stc, lsw, lwv);
            v_f32_st_tnsr(stc, lsc, lcv);
        }
    }
}