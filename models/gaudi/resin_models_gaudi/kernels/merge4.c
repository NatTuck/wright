// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright 2026 Nat Tuck
// Merge each group of four consecutive patch columns into one row: y[f + H*j, k] = x[f, 4k+j].
//
// The tower's patches already arrive in spatial-merge-block order, so on the host this is a
// plain `view(-1, 4*H)`. On the card it is not: tensors are feature-major, so memory index is
// `feature + H*token`, and the destination of column 4k+j starts at `k*4*H + j*H` - a
// different place. Hence a strided copy rather than a reshape. The four moves per output
// column are each H contiguous floats, so this is four block copies per column.
//
// tensors: 0 x (H, 4M) f32 | 1 y (4H, M) f32. index space: [M].
#include "common.h"

void main(tensor x, tensor y)
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const int H = get_dim_size(x, 0);
    for (int k = off[0]; k < end[0]; k++) {
        for (int j = 0; j < 4; j++) {
            for (int i = 0; i < H; i += 64) {
                int5 sc = {i, 4 * k + j, 0, 0, 0};
                int5 dc = {i + H * j, k, 0, 0, 0};
                v_f32_st_tnsr(dc, y, v_f32_ld_tnsr_b(sc, x));
            }
        }
    }
}
