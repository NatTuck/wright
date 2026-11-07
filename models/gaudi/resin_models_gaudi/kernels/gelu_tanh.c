// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright 2026 Nat Tuck
// GELU, tanh approximation, elementwise f32. The vision tower's MLP uses it
// (gelu_pytorch_tanh), and the stock gelu_fwd_f32 will not compile in our graphs.
// y = x * sigmoid(k * (x + 0.044715 x^3)),  k = 2 * sqrt(2/pi)   [= tanh form]
// tensors: 0 x (n, T) f32 | 1 y (n, T) f32. index space: [T].
#include "common.h"

#define GELU_K 1.5957691216057308f

void main(tensor x, tensor y)
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const int n = get_dim_size(y, 0);
    for (int t = off[0]; t < end[0]; t++) {
        for (int i = 0; i < n; i += 64) {
            int5 c = {i, t, 0, 0, 0};
            const float64 v = v_f32_ld_tnsr_b(c, x);
            const float64 a = GELU_K * (v + 0.044715f * v * v * v);
            // nl_sigmoid's reciprocal breaks down when its argument is huge, so saturate a at
            // +-30: beyond that sigmoid is 1 or 0 to f32 precision anyway.
            const float64 ac = v_f32_max_b(v_f32_min_b(a, 30.0f), -30.0f);
            v_f32_st_tnsr(c, y, v * nl_sigmoid(ac));
        }
    }
}
