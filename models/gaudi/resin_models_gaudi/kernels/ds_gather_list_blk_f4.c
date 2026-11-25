// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// DeepSeek-V4.1 fp4 listed-latent BLOCK gather (D=512): one key-block of B=64 listed compressed latents
// for a query-tile of Q columns, into a contiguous fp4 staging slab (codes+scales, NO dequant). The
// dequant is ds_ldec_f4 (separate producer) so the compiler SRAM-slices it feeding batch_gemm and overlaps
// TPC-dequant with MME-GEMM. This is ds_gather_list_f4 bounded to a single 64-key block (param b0) of a
// single query-tile (runtime tile offset tile_off), so the attention streams 640 keys as 10 reused blocks
// instead of materializing the full (512, T) slab. Key k of column s is position list[b0*64+k, s] of
// column s's sequence: block row blocks[p>>7, s] + latent offset, entry p&127. Keys at/past the column's
// count (slots[4, s]) are zeros. A fp4 latent page holds 128 entries: 256B codes (2/row, rows 0..63) +
// 32B e4m3 scales (8/row, rows 64..79).
// tensors: 0 arena (512, PR, NB) u8  1 list (C, T) i32  2 blocks (P, T) i32  3 loff (64,) i32
//          4 slots (8, T) i32 (ix_lslots)  5 tile_off (1,) i32  6 after (unread)
//        | 7 codes (256, B, Q) u8  8 scales (32, B, Q) u8  9 idx (B, Q) i32 (source key index, -1=invalid)
// params: b0 (key-block index), B (keys/block=64), Q (tile columns). index space: [1, B/16, Q]
#include "common.h"

#define KEYS 16
#define ROW(r) (r) & pm, (r) >> psh

void main(tensor arena, tensor list, tensor blocks, tensor loff, tensor slots, tensor tile_off,
          tensor after, tensor codes, tensor scales, tensor idx, int b0, int B, int Q)
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
    int5 tc = {0, 0, 0, 0, 0};
    const int s0 = s_i32_ld_g(gen_addr(tc, tile_off));   // the tile's first column
    int5 lc = {0, 0, 0, 0, 0};
    const int lat_off = s_i32_ld_g(gen_addr(lc, loff));
    const int kbase = b0 * B;                            // first global key of this block
    for (int q = off[2]; q < end[2]; q++) {
        const int s = s0 + q;                            // the global column
        int5 kc = {4, s, 0, 0, 0};
        const int cnt = s_i32_ld_g(gen_addr(kc, slots));
        for (int j = off[1]; j < end[1]; j++) {
            const int k0 = j * KEYS;                     // local key within the block
            int p[KEYS], lat[KEYS];
            #pragma unroll
            for (int u = 0; u < KEYS; u++) {
                const int gk = kbase + k0 + u;           // global key index
                int5 pc = {gk, s, 0, 0, 0};
                p[u] = s_i32_ld_g(gen_addr(pc, list));
                p[u] = gk < cnt ? p[u] : 0;
            }
            #pragma unroll
            for (int u = 0; u < KEYS; u++) {
                int5 bc = {p[u] >> 7, s, 0, 0, 0};
                lat[u] = s_i32_ld_g(gen_addr(bc, blocks)) + lat_off;
            }
            #pragma unroll
            for (int u = 0; u < KEYS; u++) {
                const int gk = kbase + k0 + u;
                const int r = p[u] & 127;
                const int live = gk < cnt;
                int5 ac = {(r & 1) * 256, ROW(lat[u] + (r >> 1)), 0, 0};
                const uchar256 c256 = live ? v_u8_ld_tnsr_b(ac, arena) : zero8;
                int5 oc = {0, k0 + u, q, 0, 0};
                v_u8_st_tnsr_partial(oc, codes, c256, 255, 0);
                int5 sc = {(r & 7) * 32, ROW(lat[u] + 64 + (r >> 3)), 0, 0};
                const uchar256 s32 = live ? v_u8_ld_tnsr_b(sc, arena) : zero8;
                v_u8_st_tnsr_partial(oc, scales, s32, 31, 0);
                int5 ic = {k0 + u, q, 0, 0, 0};
                v_i32_st_tnsr_partial(ic, idx, (int64)(live ? gk : -1), 0, 0);
            }
        }
    }
}