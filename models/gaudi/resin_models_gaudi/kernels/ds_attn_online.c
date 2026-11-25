// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// DeepSeek-V4.1 ONLINE-softmax over one key block of the unified 640-key stream (window 128 + listed
// 512, streamed as 10 blocks of B = 64), matching the official sparse_attn_kernel recurrence exactly
// (bf16 compute, f32 accumulate; the global-max ds_slot_fused matches it only for <= 64 keys). One
// member per (head h, query-tile column c) (index space [nh, Q]); vector lanes are the block's keys.
// Per block, with the RAW (unscaled) QK scores sc and the persisted running max / sum:
//   s        = sc * scale                       (f32, scale BEFORE the max, as the official kernel)
//   new_max  = max(prev_max, max_valid_keys s)  (prev_max init -1e30; invalid keys masked to NINF)
//   sscale   = exp(prev_max - new_max)
//   P        = bf16(exp(s - new_max))           (natural key order, P_LINEAR; cast BEFORE the PV GEMM)
//   new_sum  = prev_sum * sscale + sum_valid_keys exp(s - new_max)   (f32 sum of the UNCAST exps)
// A key is invalid when its gather index is -1 (the gather wrote kv = 0, so QK gives 0 there): the
// idx table (B, Q) i32 holds the block's per-column key indices, head-independent like lslots; lanes
// with idx < 0 (and lanes >= B) are masked to NINF for the max and their exp is forced to exactly 0
// for P and the sum. A block with no valid key leaves max / sum untouched (sscale = 1, P = 0).
// The acc_o rescale + add of the block's fresh PV output is ds_attn_acc (MME batch_gemm has no beta).
// tensors: 0 sc (B, nh, Q) f32 [raw QK, unscaled]  1 max (nh, Q) f32 [in-place persist, input-only]
//          2 sum (nh, Q) f32 [in-place persist, input-only]  3 idx (B, Q) i32 [key indices, -1 = invalid]
//          | 4 P (B, nh, Q) bf16  5 sscale (nh, Q) f32  6 sink (64, 1) i32 [fresh dataflow marker,
//          member (0,0) writes 1 — mla_slot_acc in-place convention; max/sum are input-only, sink chains]
// params: scale, nh, B. index space: [nh, Q]
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

void main(tensor sc, tensor mx, tensor sm, tensor idx, tensor P, tensor sscale, tensor sink,
          float scale, int nh, int B)
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const float64 lanef = v_convert_i32_to_f32_b((int64)read_lane_id_4b_b(), SW_RHNE, 0, 1, 0);
    const float64 Bf = (float)B;
    DECL_IDX(IDX)
    scalar_cache_reset();
    for (int c = off[1]; c < end[1]; c++) {
        for (int h = off[0]; h < end[0]; h++) {
            int5 stc = {h, c, 0, 0, 0};
            const float pm = s_f32_ld_g(gen_addr(stc, mx));
            const float ps = s_f32_ld_g(gen_addr(stc, sm));
            // pass 1: the block max over the valid scaled scores, then the running max
            float64 mb = NINF;
            int5 cc = {0, h, c, 0, 0};
            int5 ic = {0, c, 0, 0, 0};
            for (int i = 0; i < B; i += 64) {
                cc[0] = i;
                ic[0] = i;
                const float64 li = lanef + (float)i;
                const float64 idxf = v_convert_i32_to_f32_b(v_i32_ld_tnsr_b(ic, idx), SW_RHNE, 0, 1, 0);
                const float64 s = v_f32_ld_tnsr_b(cc, sc) * scale;
                const float64 v = v_f32_sel_less_f32_b(idxf, 0.0f, NINF, s);
                mb = v_f32_max_b(mb, v_f32_sel_less_f32_b(li, Bf, v, NINF));
            }
            NL_REDUCE_MAX(mb, IDX, mb);
            const float64 nm = v_f32_max_b((float64)pm, mb);
            const float64 ss = exp8((float64)pm - nm);
            // pass 2: bf16 probabilities (natural key order) and the f32 sum of the UNCAST exps
            float64 sv = 0.0f;
            int5 pc = {0, h, c, 0, 0};
            for (int i = 0; i < B; i += 128) {
                cc[0] = i;
                ic[0] = i;
                const float64 li = lanef + (float)i;
                float64 idxf = v_convert_i32_to_f32_b(v_i32_ld_tnsr_b(ic, idx), SW_RHNE, 0, 1, 0);
                float64 s = v_f32_ld_tnsr_b(cc, sc) * scale;
                float64 ea = v_f32_sel_less_f32_b(idxf, 0.0f, 0.0f, exp8(s - nm));
                ea = v_f32_sel_less_f32_b(li, Bf, ea, 0.0f);
                float64 eb = 0.0f;
                if (i + 64 < B) {
                    cc[0] = i + 64;
                    ic[0] = i + 64;
                    idxf = v_convert_i32_to_f32_b(v_i32_ld_tnsr_b(ic, idx), SW_RHNE, 0, 1, 0);
                    s = v_f32_ld_tnsr_b(cc, sc) * scale;
                    eb = v_f32_sel_less_f32_b(idxf, 0.0f, 0.0f, exp8(s - nm));
                    eb = v_f32_sel_less_f32_b(li + 64.0f, Bf, eb, 0.0f);
                }
                sv = sv + ea + eb;
                pc[0] = i;
                const int rem = B - i;
                if (rem >= 128)
                    v_bf16_st_tnsr(pc, P, cvt_bf16_lin(ea, eb));
                else
                    v_bf16_st_tnsr_partial(pc, P, cvt_bf16_lin(ea, eb), rem - 1, 0);
            }
            float64 ts;
            NL_REDUCE_ADD(sv, IDX, ts);
            const float64 ns = v_f32_mac_b((float64)ps, ss, ts);
            v_f32_st_tnsr_partial(stc, sscale, ss, 0, 0);
            v_f32_st_tnsr_partial(stc, mx, nm, 0, 0);
            v_f32_st_tnsr_partial(stc, sm, ns, 0, 0);
        }
    }
    if (off[0] == 0 && off[1] == 0) {
        int5 zc = {0, 0, 0, 0, 0};
        v_i32_st_tnsr(zc, sink, (int64)1);
    }
}
