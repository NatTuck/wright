// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright 2026 Nat Tuck
// 2x2 spatial merge for the Qwen3.8 vision tower's merger: (hidden, patches) ->
// (4*hidden, merged), where merged token m = i * (w/2) + j collects the four patches
// (2i+di, 2j+dj) into rows (2*di + dj) * hidden + c.
//
// tensors: 0 x (hidden, patches) f32 | 1 y (4*hidden, merged) f32
// params: patch grid width w (i32). index space: [merged] (the default geom_out_tokens).
#include "common.h"

void main(tensor x, tensor y, int w)
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const int hidden = get_dim_size(x, 0);
    const int w2 = w / 2;
    for (int m = off[0]; m < end[0]; m++) {
        const int i = m / w2;
        const int j = m - i * w2;
        for (int c = 0; c < hidden; c += 64) {
            for (int di = 0; di < 2; di++) {
                for (int dj = 0; dj < 2; dj++) {
                    int5 src = {c, (2 * i + di) * w + 2 * j + dj, 0, 0, 0};
                    int5 dst = {(2 * di + dj) * hidden + c, m, 0, 0, 0};
                    v_f32_st_tnsr(dst, y, v_f32_ld_tnsr_b(src, x));
                }
            }
        }
    }
}
