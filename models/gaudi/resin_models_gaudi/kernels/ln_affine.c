// SPDX-License-Identifier: AGPL-3.0-or-later OR PolyForm-Noncommercial-1.0.0
// Copyright Aaron Botsis and Nat Tuck
// Required Notice: Copyright Aaron Botsis
// LayerNorm over dim 0 with weight and bias, for the Qwen3.8 vision tower.
// y[c, t] = (x[c, t] - mean_t) * rsqrt(var_t + eps) * w[c] + b[c]
//
// tensors: 0 x (C, T) f32 | 1 w (C,) f32 | 2 b (C,) f32 | 3 y (C, T) f32
// params: eps (f32). index space: [T], one program per column.
//
// Validated on module 6: relative error 5.6e-07 against numpy at (1152, 64).
// The early 2x-sum confusion was a harness bug, not this kernel: Synapse tensors keep
// dim 0 contiguous, so the host array must be bound feature-major (x.T). At T=1 the two
// layouts coincide, which is why single-column tests looked exact and multi-column did not.
//
// Structure follows ml4_norm's RMSNorm path (which is validated in production): the default
// geometry gives [T], so the loop runs over off[0], and each pass loads two 64-lane vectors
// per 128 rows. The stock layer_norm_fwd_affine_* kernels reject every shape we can give
// them, and ml4_norm's own mode-1 bias path is numerically wrong on this box, hence this.
#include "common.h"

void main(tensor x, tensor w, tensor b, tensor y, float eps)
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const int n = get_dim_size(y, 0);
    const float inv_n = 1.0f / (float)n;
    DECL_IDX(IDX)
    for (int t = off[0]; t < end[0]; t++) {
        // mean
        float64 s0 = 0.0f, s1 = 0.0f;
        for (int i = 0; i < n; i += 128) {
            int5 c = {i, t, 0, 0, 0};
            s0 += v_f32_ld_tnsr_b(c, x);
            if (i + 64 < n) {
                c[0] = i + 64;
                s1 += v_f32_ld_tnsr_b(c, x);
            }
        }
        float64 tot;
        NL_REDUCE_ADD(s0 + s1, IDX, tot);
        const float64 mean = tot * inv_n;
        // variance about that mean
        float64 q0 = 0.0f, q1 = 0.0f;
        for (int i = 0; i < n; i += 128) {
            int5 c = {i, t, 0, 0, 0};
            const float64 a = v_f32_ld_tnsr_b(c, x) - mean;
            q0 = v_f32_mac_b(a, a, q0);
            if (i + 64 < n) {
                c[0] = i + 64;
                const float64 d = v_f32_ld_tnsr_b(c, x) - mean;
                q1 = v_f32_mac_b(d, d, q1);
            }
        }
        float64 qt;
        NL_REDUCE_ADD(q0 + q1, IDX, qt);
        const float64 r = nl_rsqrt(qt * inv_n + eps);
        for (int i = 0; i < n; i += 64) {
            int5 c = {i, t, 0, 0, 0}, wc = {i, 0, 0, 0, 0};
            const float64 v = (v_f32_ld_tnsr_b(c, x) - mean) * r * v_f32_ld_tnsr_b(wc, w);
            v_f32_st_tnsr(c, y, v + v_f32_ld_tnsr_b(wc, b));
        }
    }
}
