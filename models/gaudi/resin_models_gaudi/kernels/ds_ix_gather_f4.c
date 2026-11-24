// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// DSA index keys of one slot, fp4 KV (DeepSeek-V4.1): gather the slot's C fp4 index keys (packed e2m1
// codes + e8m0 group scales) into CONTIGUOUS staging for mx4_decs, which dequants fp4 -> fp8 with a
// PER-KEY tensor scale k (out = FP8(code * 2^(S-127-k)), sk = 2^k restored by ds_ix_qacc).  Folding the
// e8m0 scale with k=0 saturates fp8 at +-240 for large scales; normalizing by k = maxS-132 keeps the
// largest block at 6*2^5 = 192 < 240 and every group of the key within the exact 13-exponent window.
// An fp4 index page (see ds_store_f4.c) holds 128 keys: 64 B of codes each (8 keys/row, rows 0..15) and
// 4 B of e8m0 scales each (128 keys/row, row 16).  The page rows come from cmp_blocks (per column) plus
// the owner's latent offset and CMP_PAGE_ROWS.  Keys at or past the slot's klen and pages of -1 get zero
// codes and scale 127 (2^0).
// tensors: 0 arena (512, PR, NB) u8  1 cmp_blocks (npages, T) i32  2 slots (8, S) i32  3 after (unread)
//          4 loff (64,) i32 | 5 codes (64, C, S) u8 (packed, low nibble first)  6 scales (4, C, S) u8 (e8m0)
//          7 sk (C, S) f32 (= 2^k)  8 ktab (C, S) i32 (= k, per-row scale for mx4_decs)
//          9 ids (C, S) i32 (= c + C*s, the identity row map for mx4_decs R=0)
// params: page_rows (rows of a fp4 latent page = CMP_PAGE_ROWS).  index space: [1, C / 32, S]
#include "common.h"

#define KEYS 32
#define ROW(r) (r) & pm, (r) >> psh

void main(tensor arena, tensor pages, tensor slots, tensor after, tensor loff, tensor codes, tensor scales,
          tensor sk, tensor ktab, tensor ids, int page_rows)
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const int PR = get_dim_size(arena, 1);
    const int C = get_dim_size(codes, 1);
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
                const int c = j * KEYS + i;
                // 64 packed code bytes of key (row row0 + key/8, byte (key%8)*64)
                int5 ic = {(key & 7) * 64, ROW(row0 + (key >> 3)), 0, 0};
                const uchar256 c64 = live ? v_u8_ld_tnsr_b(ic, arena) : zero8;
                int5 oc = {0, c, s, 0, 0};
                v_u8_st_tnsr_partial(oc, codes, c64, 63, 0);
                // 4 e8m0 scale bytes of key (row row0 + 16, byte 4*key); dead keys -> 127
                int5 sc = {4 * key, ROW(row0 + 16), 0, 0};
                uchar256 s4 = live ? v_u8_ld_tnsr_b(sc, arena) : zero8;
                s4 = live ? s4 : v_u8_or_b(s4, (uchar256)127);
                v_u8_st_tnsr_partial(oc, scales, s4, 3, 0);
                // per-key tensor scale k = maxS - 132 (largest block 6*2^5 = 192 < 240); sk = 2^k
                const int s32 = live ? s_i32_ld_g(gen_addr(sc, arena)) : 0x7F7F7F7F;
                int maxS = s32 & 0xFF;
                const int b1 = (s32 >> 8) & 0xFF, b2 = (s32 >> 16) & 0xFF, b3 = (s32 >> 24) & 0xFF;
                maxS = maxS > b1 ? maxS : b1;
                maxS = maxS > b2 ? maxS : b2;
                maxS = maxS > b3 ? maxS : b3;
                maxS = maxS < 6 ? 6 : maxS;  // keep 2^k a normal f32 (tiny keys are ~0 anyway)
                const int k = maxS - 132;
                const float64 skv = as_float64((int64)(maxS - 5) << 23);  // 2^k, exponent k+127 = maxS-5
                int5 osk = {c, s, 0, 0, 0};  // sk/ktab/ids are (C, S): dim0=c, dim1=s
                v_f32_st_tnsr_partial(osk, sk, skv, 0, 0);
                v_i32_st_tnsr_partial(osk, ktab, (int64)k, 0, 0);
                v_i32_st_tnsr_partial(osk, ids, (int64)(c + C * s), 0, 0);
            }
        }
    }
}