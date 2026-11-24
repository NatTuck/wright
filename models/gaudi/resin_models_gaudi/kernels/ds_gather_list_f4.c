// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// mla_gather_list for DeepSeek-V4.1 fp4 KV (D = 512): each column's listed compressed latents into a
// CONTIGUOUS staging slab of packed fp4 e2m1 codes + e4m3 group scales (NO dequant here).  The dequant
// is a separate producer (ds_ldec_f4) so the graph compiler can SRAM-slice it feeding batch_gemm and
// overlap TPC-dequant with MME-GEMM (the MoE mx4_decs pattern).  Key k of slot s is position list[k, s]
// of column s's sequence: block row blocks[p >> 7, s] plus the latent offset, entry p & 127.  A fp4
// latent page (ds_store_f4.c) holds 128 entries: 256 B codes each (2/row, rows 0..63) and 32 B e4m3
// scales each (8/row, rows 64..79).  Keys at or past the slot's count (slots[4, s]) are zeros.
// tensors: 0 arena (512, PR, NB) u8  1 list (C, T) i32  2 blocks (P, T) i32  3 loff (64,) i32
//          4 slots (8, T) i32 (ix_lslots)  5 after (unread) | 6 codes (256, C, T) u8  7 scales (32, C, T) u8
// index space: [1, C / 16, T]
#include "common.h"

#define KEYS 16
#define ROW(r) (r) & pm, (r) >> psh

void main(tensor arena, tensor list, tensor blocks, tensor loff, tensor slots, tensor after, tensor codes,
          tensor scales)
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const int PR = get_dim_size(arena, 1);
    int psh = 0;
    while ((1 << psh) < PR)
        psh++;
    const int pm = PR - 1;
    const uchar256 zero8 = 0;
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
                const uchar256 c256 = live ? v_u8_ld_tnsr_b(ac, arena) : zero8;
                int5 oc = {0, k0 + u, s, 0, 0};
                v_u8_st_tnsr_partial(oc, codes, c256, 255, 0);
                int5 sc = {(r & 7) * 32, ROW(lat[u] + 64 + (r >> 3)), 0, 0};
                const uchar256 s32 = live ? v_u8_ld_tnsr_b(sc, arena) : zero8;
                v_u8_st_tnsr_partial(oc, scales, s32, 31, 0);
            }
        }
    }
}