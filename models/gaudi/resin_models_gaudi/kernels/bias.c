// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright 2026 Nat Tuck
// Add a per-row bias to every column: y[c, t] = x[c, t] + b[c].
// The GEMM node does not carry a bias, and the stock broadcast add will not create in our
// graphs, so the tower's linear layers end with this.
// tensors: 0 x (n, T) f32 | 1 b (n,) f32 | 2 y (n, T) f32. index space: [T].
#include "common.h"

void main(tensor x, tensor b, tensor y)
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const int n = get_dim_size(y, 0);
    for (int t = off[0]; t < end[0]; t++) {
        for (int i = 0; i < n; i += 64) {
            int5 c = {i, t, 0, 0, 0}, bc = {i, 0, 0, 0, 0};
            v_f32_st_tnsr(c, y, v_f32_ld_tnsr_b(c, x) + v_f32_ld_tnsr_b(bc, b));
        }
    }
}
