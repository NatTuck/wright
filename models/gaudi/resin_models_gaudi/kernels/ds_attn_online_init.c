// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// DeepSeek-V4.1 online-attention accumulator reset before a query tile's first key block:
// acc_o = 0, max = -1e30 (the official finite init: a column with NO valid key keeps max = -1e30,
// sum = 0 and outputs 0, never NaN), sum = 0. One member per (head h, query-tile column c).
// tensors: 0 acc_o (D, nh, Q) f32  1 max (nh, Q) f32  2 sum (nh, Q) f32  (FRESH outputs, the SSA chain's
//          block-0 state; online/acc each produce the next block's state as a fresh output — Synapse
//          treats input-only tensors as immutable within one recipe, so loop-carried state must be SSA)
// params: none. index space: [nh, Q]. D a multiple of 64, at most 512 (design D = 512).
#include "common.h"

#define MAXV 8

void main(tensor acc_o, tensor mx, tensor sm)
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const int D = get_dim_size(acc_o, 0);
    const float64 zero = 0.0f, minit = -1.0e30f;
    for (int c = off[1]; c < end[1]; c++) {
        for (int h = off[0]; h < end[0]; h++) {
            int5 stc = {h, c, 0, 0, 0};
            v_f32_st_tnsr_partial(stc, mx, minit, 0, 0);
            v_f32_st_tnsr_partial(stc, sm, zero, 0, 0);
            int5 oc = {0, h, c, 0, 0};
            #pragma unroll
            for (int j = 0; j < MAXV; j++) {
                if (64 * j < D) {
                    oc[0] = 64 * j;
                    v_f32_st_tnsr(oc, acc_o, zero);
                }
            }
        }
    }
}
