// SPDX-License-Identifier: AGPL-3.0-or-later OR PolyForm-Noncommercial-1.0.0
// Copyright Aaron Botsis and Nat Tuck
// Required Notice: Copyright Aaron Botsis
// Softmax over dim 0 of each column, for the Qwen3.8 vision tower's attention.
// y[c, t] = exp(x[c, t] - max_t) / sum_c exp(x[c, t] - max_t)
//
// tensors: 0 x (n, T) f32 | 1 y (n, T) f32
// index space: [T] (the default geom_out_tokens), one program per column.
//
// Same pass structure as ln_affine: two 64-lane loads per 128 rows, a horizontal reduce
// per pass, and a final scaled store. Dim 0 is the feature dim, so with feature-major
// tensors the reduction runs over contiguous memory.
#include "common.h"

void main(tensor x, tensor y)
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const int n = get_dim_size(y, 0);
    DECL_IDX(IDX)
    for (int t = off[0]; t < end[0]; t++) {
        // column max, for stability
        float64 m0 = -3.4e38f, m1 = -3.4e38f;
        for (int i = 0; i < n; i += 128) {
            int5 c = {i, t, 0, 0, 0};
            m0 = v_f32_max_b(m0, v_f32_ld_tnsr_b(c, x));
            if (i + 64 < n) {
                c[0] = i + 64;
                m1 = v_f32_max_b(m1, v_f32_ld_tnsr_b(c, x));
            }
        }
        float64 mx;
        NL_REDUCE_MAX(v_f32_max_b(m0, m1), IDX, mx);
        // denominator
        float64 s0 = 0.0f, s1 = 0.0f;
        for (int i = 0; i < n; i += 128) {
            int5 c = {i, t, 0, 0, 0};
            s0 += nl_exp(v_f32_ld_tnsr_b(c, x) - mx);
            if (i + 64 < n) {
                c[0] = i + 64;
                s1 += nl_exp(v_f32_ld_tnsr_b(c, x) - mx);
            }
        }
        float64 sum;
        NL_REDUCE_ADD(s0 + s1, IDX, sum);
        const float64 r = nl_recip(sum);
        for (int i = 0; i < n; i += 64) {
            int5 c = {i, t, 0, 0, 0};
            v_f32_st_tnsr(c, y, nl_exp(v_f32_ld_tnsr_b(c, x) - mx) * r);
        }
    }
}
