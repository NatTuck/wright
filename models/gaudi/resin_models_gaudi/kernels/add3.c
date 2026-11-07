// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright 2026 Nat Tuck
// Elementwise sum of three f32 tensors of the same shape: y = a + b + c.
// Fuses two adds into one node, saving a pass over the tensor. Not used by the vision tower
// today (its residuals go through rg_resid_bias_f32 and its head sums through rg_add_f32)
// and not validated on the card. The "two TPC nodes in a row misbehave" note that used to
// sit here was a harness artifact, not a property of the stack.
// tensors: 0 a (n, T) | 1 b (n, T) | 2 c (n, T) | 3 y (n, T). index space: [T].
#include "common.h"

void main(tensor a, tensor b, tensor c, tensor y)
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const int n = get_dim_size(y, 0);
    for (int t = off[0]; t < end[0]; t++) {
        for (int i = 0; i < n; i += 64) {
            int5 p = {i, t, 0, 0, 0};
            v_f32_st_tnsr(p, y,
                          v_f32_ld_tnsr_b(p, a) + v_f32_ld_tnsr_b(p, b) + v_f32_ld_tnsr_b(p, c));
        }
    }
}
