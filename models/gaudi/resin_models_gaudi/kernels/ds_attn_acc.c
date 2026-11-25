// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// DeepSeek-V4.1 online-attention accumulator step: rescale the persisted acc_o by the block's
// sscale and add the block's FRESH MME PV output (batch_gemm has no beta/accumulate, so the
// official acc_o = acc_o * sscale + P @ ks_blk is split into the PV GEMM writing pv and this add):
//   acc_o[d, h, c] = acc_o[d, h, c] * sscale[h, c] + pv[d, h, c]
// One member per (head h, query-tile column c) (index space [nh, Q]); the D loop is unrolled so the
// 64-dim blocks are independent work. The loop-carried acc_o is SSA across the 10 key-blocks of ONE
// recipe: this node READS acc_o (block i's state, input-only) and WRITES a FRESH acc_o_out (block i+1's
// state). Synapse treats input-only tensors as immutable within a recipe, so the recurrence must produce
// a fresh output per block (single producer); the compiler reuses the dead acc_o[i] memory across the
// control-dep-serialized blocks, bounding the working set.
// tensors: 0 acc_o (D, nh, Q) f32 [block i state, read-only input]  1 sscale (nh, Q) f32  2 pv (D, nh, Q) f32
//          (pv = the block's fresh P @ ks_blk output)  | 3 acc_o_out (D, nh, Q) f32 [block i+1 state, fresh]
// params: none. index space: [nh, Q]. D a multiple of 64, at most 512 (design D = 512).
#include "common.h"

#define MAXV 8

void main(tensor acc_o, tensor sscale, tensor pv, tensor acc_o_out)
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const int D = get_dim_size(acc_o, 0);
    scalar_cache_reset();
    for (int c = off[1]; c < end[1]; c++) {
        for (int h = off[0]; h < end[0]; h++) {
            int5 stc = {h, c, 0, 0, 0};
            const float ss = s_f32_ld_g(gen_addr(stc, sscale));
            int5 oc = {0, h, c, 0, 0};
            #pragma unroll
            for (int j = 0; j < MAXV; j++) {
                if (64 * j < D) {
                    oc[0] = 64 * j;
                    const float64 a = v_f32_ld_tnsr_b(oc, acc_o);
                    const float64 p = v_f32_ld_tnsr_b(oc, pv);
                    v_f32_st_tnsr(oc, acc_o_out, v_f32_mac_b(a, (float64)ss, p));
                }
            }
        }
    }
}
