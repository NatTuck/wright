// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// DeepSeek-V4.1 online-attention accumulator step: rescale the persisted acc_o by the block's
// sscale and add the block's FRESH MME PV output (batch_gemm has no beta/accumulate, so the
// official acc_o = acc_o * sscale + P @ ks_blk is split into the PV GEMM writing pv and this add):
//   acc_o[d, h, c] = acc_o[d, h, c] * sscale[h, c] + pv[d, h, c]
// One member per (head h, query-tile column c) (index space [nh, Q]); the D loop is unrolled so the
// 64-dim blocks are independent work. acc_o persists across the 10 key-block launches (one block per
// launch, MME PV in between), so there is no within-launch reuse to VLM-cache (unlike attn_sdpa_acc,
// which merges many slots in one launch); acc_o lives in HBM / compiler-placed SRAM.
// tensors: 0 acc_o (D, nh, Q) f32 [in-place persist, input-only]  1 sscale (nh, Q) f32  2 pv (D, nh, Q) f32
//          (pv = the block's fresh P @ ks_blk output; writes acc_o in place)  | 3 sink (64, 1) i32
//          (fresh dataflow marker, member (0,0) writes 1 — the mla_slot_acc in-place convention; Synapse
//          rejects acc_o as input0 AND output0, so acc_o is input-only and the sink chains the recurrence)
// params: none. index space: [nh, Q]. D a multiple of 64, at most 512 (design D = 512).
#include "common.h"

#define MAXV 8

void main(tensor acc_o, tensor sscale, tensor pv, tensor sink)
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
                    v_f32_st_tnsr(oc, acc_o, v_f32_mac_b(a, (float64)ss, p));
                }
            }
        }
    }
    if (off[0] == 0 && off[1] == 0) {
        int5 zc = {0, 0, 0, 0, 0};
        v_i32_st_tnsr(zc, sink, (int64)1);
    }
}