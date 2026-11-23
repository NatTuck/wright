// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// DeepSeek-V4.1 fused single-pass global-max softmax over the window keys and the compressed listed
// keys of one column, matching the official sparse_attn for <= 64 visible keys (one block). One member
// per (head m, token t) (PER_COLUMN, index space [nh, T]). The window scores scw (C, nh, T) are masked
// causally from the window slot table wslots (8, S): slot t / qb, position qpos0 + t % qb, key0, klen,
// kmin, win (field 7). The compressed scores scc (K, nh, T) are masked by the count lslots[4, t]. The
// global max M is over BOTH valid key sets (keys only; the per-head sink is added to the denominator
// later, in ds_attn_fused, never to M). P = bf16(exp(sc * scale - M)) in natural key order (P_LINEAR)
// for the numerator; ls = FP32 sum(exp(sc * scale - M)) for the denominator (NOT the bf16 P sum).
// tensors: 0 scw (C, nh, T) f32  1 scc (K, nh, T) f32  2 wslots (8, S) i32  3 lslots (8, T) i32
//          | 4 Pw (C, nh, T) bf16  5 Pc (K, nh, T) bf16  6 mx (Mp, T) f32  7 lsw (Mp, T) f32
//            8 lsc (Mp, T) f32   (Mp = nh rounded up to 64; lane r of stats column b is head 64 b + r)
// params: scale, nh, qb. index space: [nh, T]
#include "common.h"

#define NINF (-3.0e38f)

// exp accurate to ~1e-7 (fp32 limit): degree-8 Taylor on r in [0, ln2) with a Cody-Waite ln2 split
// (ln2_hi has 9 trailing zero mantissa bits, so n*ln2_hi is exact for n <= 127). The bf16 P cast must
// match the official exp(sc-M) at rounding boundaries, so the exp error must sit well below bf16's ulp.
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

void main(tensor scw, tensor scc, tensor wslots, tensor lslots, tensor Pw, tensor Pc, tensor mx,
          tensor lsw, tensor lsc, float scale, int nh, int qb)
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const int C = get_dim_size(scw, 0), K = get_dim_size(scc, 0);
    const float64 lanef = v_convert_i32_to_f32_b((int64)read_lane_id_4b_b(), SW_RHNE, 0, 1, 0);
    DECL_IDX(IDX)
    scalar_cache_reset();
    for (int t = off[1]; t < end[1]; t++) {
        const int ws = t / qb;
        int5 wc = {2, ws, 0, 0, 0};
        const int qpos0 = s_i32_ld_g(gen_addr(wc, wslots));
        wc[0] = 3;
        const int key0 = s_i32_ld_g(gen_addr(wc, wslots));
        wc[0] = 4;
        const int klen = s_i32_ld_g(gen_addr(wc, wslots));
        wc[0] = 6;
        const int kmin = s_i32_ld_g(gen_addr(wc, wslots));
        wc[0] = 7;
        const int win = s_i32_ld_g(gen_addr(wc, wslots));
        const int position = qpos0 + (t % qb);
        int limw = position + 1 < klen ? position + 1 : klen;
        limw -= key0;
        limw = limw < 0 ? 0 : (limw > C ? C : limw);
        const int lof0 = kmin > key0 ? kmin - key0 : 0;
        const int wfirst = position - win + 1 - key0;
        int lofw = win > 0 && wfirst > lof0 ? wfirst : lof0;
        lofw = lofw < 0 ? 0 : lofw;
        int5 lc = {4, t, 0, 0, 0};
        int cnt = s_i32_ld_g(gen_addr(lc, lslots));
        cnt = cnt < 0 ? 0 : (cnt > K ? K : cnt);
        const float limwf = (float)limw, lofwf = (float)lofw, cntf = (float)cnt;
        for (int m = off[0]; m < end[0]; m++) {
            // pass 1: the global max over the visible window keys and the listed compressed keys
            float64 mw = NINF;
            int5 cw = {0, m, t, 0, 0};
            for (int i = 0; i < C; i += 64) {
                cw[0] = i;
                const float64 li = lanef + (float)i;
                const float64 v = v_f32_ld_tnsr_b(cw, scw);
                mw = v_f32_max_b(mw, v_f32_sel_less_f32_b(li, lofwf, NINF,
                                 v_f32_sel_less_f32_b(li, limwf, v, NINF)));
            }
            float64 mc = NINF;
            int5 cc = {0, m, t, 0, 0};
            for (int i = 0; i < K; i += 64) {
                cc[0] = i;
                const float64 li = lanef + (float)i;
                const float64 v = v_f32_ld_tnsr_b(cc, scc);
                mc = v_f32_max_b(mc, v_f32_sel_less_f32_b(li, cntf, v, NINF));
            }
            float64 mm = v_f32_max_b(mw, mc);
            NL_REDUCE_MAX(mm, IDX, mm);
            const float64 ms = mm * scale;
            // pass 2: bf16 probabilities (numerator) and their FP32 sums (denominator)
            float64 sw = 0.0f, sc = 0.0f;
            int5 pw = {0, m, t, 0, 0};
            for (int i = 0; i < C; i += 128) {
                float64 ea = 0.0f, eb = 0.0f;
                if (i < limw) {
                    const float64 li = lanef + (float)i;
                    cw[0] = i;
                    const float64 va = v_f32_ld_tnsr_b(cw, scw);
                    cw[0] = i + 64;
                    const float64 vb = v_f32_ld_tnsr_b(cw, scw);
                    ea = v_f32_sel_less_f32_b(li, limwf, exp8(v_f32_mac_b(va, scale, -ms)), 0.0f);
                    eb = v_f32_sel_less_f32_b(li + 64.0f, limwf, exp8(v_f32_mac_b(vb, scale, -ms)), 0.0f);
                    ea = v_f32_sel_less_f32_b(li, lofwf, 0.0f, ea);
                    eb = v_f32_sel_less_f32_b(li + 64.0f, lofwf, 0.0f, eb);
                    sw = sw + ea + eb;
                }
                pw[0] = i;
                v_bf16_st_tnsr(pw, Pw, cvt_bf16_lin(ea, eb));
            }
            int5 pc = {0, m, t, 0, 0};
            for (int i = 0; i < K; i += 128) {
                float64 ea = 0.0f, eb = 0.0f;
                if (i < cnt) {
                    const float64 li = lanef + (float)i;
                    cc[0] = i;
                    const float64 va = v_f32_ld_tnsr_b(cc, scc);
                    cc[0] = i + 64;
                    const float64 vb = v_f32_ld_tnsr_b(cc, scc);
                    ea = v_f32_sel_less_f32_b(li, cntf, exp8(v_f32_mac_b(va, scale, -ms)), 0.0f);
                    eb = v_f32_sel_less_f32_b(li + 64.0f, cntf, exp8(v_f32_mac_b(vb, scale, -ms)), 0.0f);
                    sc = sc + ea + eb;
                }
                pc[0] = i;
                v_bf16_st_tnsr(pc, Pc, cvt_bf16_lin(ea, eb));
            }
            float64 tw, tc;
            NL_REDUCE_ADD(sw, IDX, tw);
            NL_REDUCE_ADD(sc, IDX, tc);
            int5 stc = {m, t, 0, 0, 0};
            v_f32_st_tnsr_partial(stc, mx, ms, 0, 0);
            v_f32_st_tnsr_partial(stc, lsw, tw, 0, 0);
            v_f32_st_tnsr_partial(stc, lsc, tc, 0, 0);
        }
    }
}