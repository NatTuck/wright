// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// mla_gather_list for DeepSeek-V4.1 fp4 KV (D = 512): each column's listed compressed latents into a dense
// bf16 slab, dequantized EXACTLY (fp4 e2m1 code * e4m3 group scale -> bf16, bit-for-bit the checkpoint's
// fp4_act_quant inplace value).  Key k of slot s is position list[k, s] of column s's sequence: block row
// blocks[p >> 7, s] plus the latent offset, entry p & 127.  A fp4 latent page (see ds_store_f4.c) holds
// 128 entries: 256 B of codes each (2 entries/row, rows 0..63) and 32 B of e4m3 scales each (8 entries/row,
// rows 64..79).  Keys at or past the slot's count (slots[4, s]) are zeros.
// tensors: 0 arena (512, PR, NB) u8  1 list (C, T) i32  2 blocks (P, T) i32  3 loff (64,) i32
//          [latent offset, ...]  4 slots (8, T) i32 (ix_lslots)  5 after (unread) | 6 out (512, C, T) bf16
// index space: [1, C / 16, T]
#include "common.h"
#include "fp4_common.h"

#define KEYS 16
#define ROW(r) (r) & pm, (r) >> psh

void main(tensor arena, tensor list, tensor blocks, tensor loff, tensor slots, tensor after, tensor out)
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const int PR = get_dim_size(arena, 1);
    int psh = 0;
    while ((1 << psh) < PR)
        psh++;
    const int pm = PR - 1;
    const bfloat128 zero = 0.0f;
    scalar_cache_reset();
    int5 lc = {0, 0, 0, 0, 0};
    const int lat_off = s_i32_ld_g(gen_addr(lc, loff));
    for (int s = off[2]; s < end[2]; s++) {
        int5 kc = {4, s, 0, 0, 0};
        const int cnt = s_i32_ld_g(gen_addr(kc, slots));
        for (int j = off[1]; j < end[1]; j++) {
            const int k0 = j * KEYS;
            int p[KEYS], lat[KEYS];
            #pragma unroll
            for (int u = 0; u < KEYS; u++) {
                int5 pc = {k0 + u, s, 0, 0, 0};
                p[u] = s_i32_ld_g(gen_addr(pc, list));
                p[u] = k0 + u < cnt ? p[u] : 0;
            }
            #pragma unroll
            for (int u = 0; u < KEYS; u++) {
                int5 bc = {p[u] >> 7, s, 0, 0, 0};
                lat[u] = s_i32_ld_g(gen_addr(bc, blocks)) + lat_off;
            }
            #pragma unroll
            for (int u = 0; u < KEYS; u++) {
                const int r = p[u] & 127;
                const int live = k0 + u < cnt;
                int5 ac = {(r & 1) * 256, ROW(lat[u] + (r >> 1)), 0, 0};
                const uchar512 nib = v_convert_u4_to_u8_all_b(v_u8_ld_tnsr_b(ac, arena, SW_UNPACK | SW_UNPCK_4_TO_8));
                int5 sc = {(r & 7) * 32, ROW(lat[u] + 64 + (r >> 3)), 0, 0};
                const uchar256 sb0 = v_u8_ld_tnsr_b(sc, arena);
                sc[0] += 16;
                const uchar256 sb1 = v_u8_ld_tnsr_b(sc, arena);
                const bfloat256 a = fp4_to_bf16_256(nib.v1, sb0);
                const bfloat256 b = fp4_to_bf16_256(nib.v2, sb1);
                int5 oc = {0, k0 + u, s, 0, 0};
                v_bf16_st_tnsr(oc, out, live ? a.v1 : zero);
                oc[0] = 128;
                v_bf16_st_tnsr(oc, out, live ? a.v2 : zero);
                oc[0] = 256;
                v_bf16_st_tnsr(oc, out, live ? b.v1 : zero);
                oc[0] = 384;
                v_bf16_st_tnsr(oc, out, live ? b.v2 : zero);
            }
        }
    }
}