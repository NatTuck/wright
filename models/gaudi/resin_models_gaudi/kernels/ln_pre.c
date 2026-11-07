// SPDX-License-Identifier: AGPL-3.0-or-later OR PolyForm-Noncommercial-1.0.0
// Copyright Aaron Botsis and Nat Tuck
// Required Notice: Copyright Aaron Botsis
// LayerNorm over dim 0 of (x + off), with weight and bias:
//   y[c, t] = (s - mean_t) * rsqrt(var_t + eps) * w[c] + b[c],  s = x[c, t] + off[c, t]
//
// The offset exists so the vision tower's patch-embed can fold its position embedding into
// the norm instead of adding it in a separate node, which saves a full pass over
// (hidden, patches). An earlier comment here claimed that an elementwise add between an MME
// GEMM and this norm left part of the next GEMM's output unwritten; that was a bug in the
// test harness, not on the card (docs/vision.md keeps the diagnosis).
//
// tensors: 0 x (n, T) | 1 off (n, T) | 2 w (n,) | 3 b (n,) | 4 y (n, T). params: eps (f32).
// index space: [T].
#include "common.h"

void main(tensor x, tensor off, tensor w, tensor b, tensor y, float eps)
{
    const int5 is = get_index_space_offset();
    const int5 ie = is + get_index_space_size();
    const int n = get_dim_size(y, 0);
    const float inv_n = 1.0f / (float)n;
    DECL_IDX(IDX)
    for (int t = is[0]; t < ie[0]; t++) {
        float64 s0 = 0.0f, s1 = 0.0f;
        for (int i = 0; i < n; i += 128) {
            int5 c = {i, t, 0, 0, 0};
            s0 += v_f32_ld_tnsr_b(c, x) + v_f32_ld_tnsr_b(c, off);
            if (i + 64 < n) {
                c[0] = i + 64;
                s1 += v_f32_ld_tnsr_b(c, x) + v_f32_ld_tnsr_b(c, off);
            }
        }
        float64 tot;
        NL_REDUCE_ADD(s0 + s1, IDX, tot);
        const float64 mean = tot * inv_n;
        float64 q0 = 0.0f, q1 = 0.0f;
        for (int i = 0; i < n; i += 128) {
            int5 c = {i, t, 0, 0, 0};
            const float64 a = v_f32_ld_tnsr_b(c, x) + v_f32_ld_tnsr_b(c, off) - mean;
            q0 = v_f32_mac_b(a, a, q0);
            if (i + 64 < n) {
                c[0] = i + 64;
                const float64 d = v_f32_ld_tnsr_b(c, x) + v_f32_ld_tnsr_b(c, off) - mean;
                q1 = v_f32_mac_b(d, d, q1);
            }
        }
        float64 qt;
        NL_REDUCE_ADD(q0 + q1, IDX, qt);
        const float64 r = nl_rsqrt(qt * inv_n + eps);
        for (int i = 0; i < n; i += 64) {
            int5 c = {i, t, 0, 0, 0}, wc = {i, 0, 0, 0, 0};
            const float64 v =
                (v_f32_ld_tnsr_b(c, x) + v_f32_ld_tnsr_b(c, off) - mean) * r *
                v_f32_ld_tnsr_b(wc, w);
            v_f32_st_tnsr(c, y, v + v_f32_ld_tnsr_b(wc, b));
        }
    }
}
