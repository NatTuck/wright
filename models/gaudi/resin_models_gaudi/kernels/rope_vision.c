// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright 2026 Nat Tuck
// Axial 2D rope for the vision tower's attention: y = x*cos + rotate_half(x)*sin, applied to
// q and k only, with a per-token cos/sin table.
//
// The head dim arrives permuted so the rotation is a 64-lane-aligned half swap: the real
// head_dim (72) is laid out as [0..36) in lanes 0..35 and [36..72) in lanes 64..99, with
// lanes 36..63 and 100..127 zero. Then rotate_half is "the other 64-lane block", and a zero
// lane pairs with a zero lane, so the padding stays zero through the rotation. The
// permutation is harmless for the scores: q and k carry the same one, so q.k is unchanged,
// and v (which never sees rope) supplies the values.
//
// tensors: 0 x (n, T) f32 | 1 cos (n, T) f32 | 2 sin (n, T) f32 | 3 y (n, T) f32.
// index space: [T].
#include "common.h"

void main(tensor x, tensor cos, tensor sin, tensor y)
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const int n = get_dim_size(y, 0);
    for (int t = off[0]; t < end[0]; t++) {
        for (int i = 0; i < n; i += 64) {
            int5 c = {i, t, 0, 0, 0};
            int5 pc = {i ^ 64, t, 0, 0, 0};  // the other half of the head dim
            const float64 xv = v_f32_ld_tnsr_b(c, x);
            const float64 pv = v_f32_ld_tnsr_b(pc, x);
            // the first half subtracts its partner, the second half adds it
            const float64 rot = i == 0 ? -pv : pv;
            v_f32_st_tnsr(c, y, xv * v_f32_ld_tnsr_b(c, cos) + rot * v_f32_ld_tnsr_b(c, sin));
        }
    }
}
