// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// fp4 compressed-latent dequant producer (DeepSeek-V4.1, D=512): decode a CONTIGUOUS staging slab of
// fp4 e2m1 codes + e4m3 group scales into a dense bf16 slab, EXACTLY (fp4 code * e4m3 scale -> bf16,
// bit-for-bit the checkpoint fp4_act_quant value).  Separate from the gather so the graph compiler can
// SRAM-slice this compute producer feeding batch_gemm and overlap TPC-dequant with MME-GEMM (the MoE
// mx4_decs pattern).  Staging layout per column: 256 packed code bytes (low nibble first) + 32 e4m3
// scale bytes (one per 16-element group).
// tensors: 0 codes (256, M) u8  1 scales (32, M) u8 | 2 out (512, M) bf16.  index space: [1, M]
#include "common.h"
#include "fp4_common.h"

void main(tensor codes, tensor scales, tensor out)
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    for (int m = off[1]; m < end[1]; m++) {
        int5 cc = {0, m, 0, 0, 0};
        const uchar512 nib = v_convert_u4_to_u8_all_b(v_u8_ld_tnsr_b(cc, codes, SW_UNPACK | SW_UNPCK_4_TO_8));
        int5 sc = {0, m, 0, 0, 0};
        const uchar256 sb0 = v_u8_ld_tnsr_b(sc, scales);
        sc[0] = 16;
        const uchar256 sb1 = v_u8_ld_tnsr_b(sc, scales);
        const bfloat256 a = fp4_to_bf16_256(nib.v1, sb0);
        const bfloat256 b = fp4_to_bf16_256(nib.v2, sb1);
        int5 oc = {0, m, 0, 0, 0};
        v_bf16_st_tnsr(oc, out, a.v1);
        oc[0] = 128;
        v_bf16_st_tnsr(oc, out, a.v2);
        oc[0] = 256;
        v_bf16_st_tnsr(oc, out, b.v1);
        oc[0] = 384;
        v_bf16_st_tnsr(oc, out, b.v2);
    }
}