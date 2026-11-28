// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// ds_ix_gather_f4 for a WHOLE TILE instead of one slot: the tiled index chain's fp4 key gather, shared by every
// slot of the tile.
//
// ds_ix_gather_f4 gathers one slot's C keys and zeroes the ones past that slot's ``klen``. The chain called it
// with S slots, but every slot of a tile resolves to the SAME pages: ds_ix_gslot's lanes 3 (``key0``) and 7
// (the tile's first index page) are per-tile, and a prefill step's columns are one row, so
// ``cmp_blocks[:, col]`` is the same for all of them. Only ``len`` (the query block's visibility) and the
// block's columns differ. So the chain gathered and dequantized the same 2 MB of fp4 codes sixteen times per
// tile -- 128 KB of it distinct -- ~35 % of long-context prefill feeding arithmetic worth 4 %.
//
// This is that kernel with the ``klen`` gate replaced by the whole tile: nothing is zeroed and every key's real
// scale is published. The per-slot mask stays where it already lives, in ds_ix_qacc's ``lim = min(seen, klen)``,
// which writes -inf past the block's visibility. Called with S = 1, reading the per-tile record ix_t{i}.
//
// tensors: 0 arena (512, PR, NB) u8  1 cmp_blocks (npages, T) i32  2 slots (8, S) i32  3 after (unread)
//          4 loff (64,) i32 | 5 codes (64, C, S) u8  6 scales (4, C, S) u8  7 sk (C, S) f32  8 ktab (C, S) i32
//          9 ids (C, S) i32
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
        // THE DIFFERENCE: the whole tile is live, not this slot's visibility
        const int klen = key0 + C;
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