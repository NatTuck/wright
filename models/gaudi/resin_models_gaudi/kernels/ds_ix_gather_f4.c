// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// DSA index keys of one slot, fp4 KV (DeepSeek-V4.1): gather the slot's C fp4 index keys (packed e2m1
// codes + e8m0 group scales) into CONTIGUOUS staging for mx4_dec (which dequants fp4 -> fp8 with the
// e8m0 scale folded into the fp8 exponent, k=0).  An fp4 index page (see ds_store_f4.c) holds 128 keys:
// 64 B of codes each (8 keys/row, rows 0..15) and 4 B of e8m0 scales each (128 keys/row, row 16).
// The page rows come from cmp_blocks (per column) plus the owner's latent offset and CMP_PAGE_ROWS.
// Keys at or past the slot's klen and pages of -1 get zero codes and scale 127 (2^0).
// tensors: 0 arena (512, PR, NB) u8  1 cmp_blocks (npages, T) i32  2 slots (8, S) i32  3 after (unread)
//          4 loff (64,) i32 | 5 codes (64, C, S) u8 (packed, low nibble first)  6 scales (4, C, S) u8 (e8m0)
// params: page_rows (rows of a fp4 latent page = CMP_PAGE_ROWS).  index space: [1, C / 32, S]
#include "common.h"

#define KEYS 32
#define ROW(r) (r) & pm, (r) >> psh

void main(tensor arena, tensor pages, tensor slots, tensor after, tensor loff, tensor codes, tensor scales,
          int page_rows)
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
    const int ioff = s_i32_ld_g(gen_addr(lc, loff)) + page_rows;
    for (int s = off[2]; s < end[2]; s++) {
        int5 kc = {3, s, 0, 0, 0};
        const int key0 = s_i32_ld_g(gen_addr(kc, slots));
        kc[0] = 4;
        const int klen = s_i32_ld_g(gen_addr(kc, slots));
        kc[0] = 5;
        const int col = s_i32_ld_g(gen_addr(kc, slots));
        kc[0] = 7;
        const int pbase = s_i32_ld_g(gen_addr(kc, slots));
        for (int j = off[1]; j < end[1]; j++) {
            const int page = j / (128 / KEYS);
            int5 pc = {pbase + page, col, 0, 0, 0};
            const int row = s_i32_ld_g(gen_addr(pc, pages));
            const int row0 = row >= 0 ? row + ioff : -1;
            const int cnt = row0 >= 0 ? klen - key0 - page * 128 : 0;
            const int q0 = (j % (128 / KEYS)) * KEYS;
            #pragma unroll 4
            for (int i = 0; i < KEYS; i++) {
                const int key = q0 + i;
                const int live = key < cnt;
                // 64 packed code bytes of key (row row0 + key/8, byte (key%8)*64)
                int5 ic = {(key & 7) * 64, ROW(row0 + (key >> 3)), 0, 0};
                const uchar256 c64 = live ? v_u8_ld_tnsr_b(ic, arena) : zero8;
                int5 oc = {0, j * KEYS + i, s, 0, 0};
                v_u8_st_tnsr_partial(oc, codes, c64, 63, 0);
                // 4 e8m0 scale bytes of key (row row0 + 16, byte 4*key); dead keys -> 127
                int5 sc = {4 * key, ROW(row0 + 16), 0, 0};
                uchar256 s4 = live ? v_u8_ld_tnsr_b(sc, arena) : zero8;
                s4 = live ? s4 : v_u8_or_b(s4, (uchar256)127);
                int5 os = {0, j * KEYS + i, s, 0, 0};
                v_u8_st_tnsr_partial(os, scales, s4, 3, 0);
            }
        }
    }
}