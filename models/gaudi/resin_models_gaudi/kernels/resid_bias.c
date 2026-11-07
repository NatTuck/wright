// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright 2026 Nat Tuck
// Residual plus a per-row bias: y[c, t] = x[c, t] + m[c, t] + b[c].
// Fuses what would otherwise be a bias node followed by a residual add, so a residual costs
// one TPC pass instead of two.
// tensors: 0 x (n, T) | 1 m (n, T) | 2 b (n,) | 3 y (n, T). index space: [T].
#include "common.h"

void main(tensor x, tensor m, tensor b, tensor y)
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const int n = get_dim_size(y, 0);
    for (int t = off[0]; t < end[0]; t++) {
        for (int i = 0; i < n; i += 64) {
            int5 p = {i, t, 0, 0, 0}, bc = {i, 0, 0, 0, 0};
            v_f32_st_tnsr(p, y,
                          v_f32_ld_tnsr_b(p, x) + v_f32_ld_tnsr_b(p, m) + v_f32_ld_tnsr_b(bc, b));
        }
    }
}
