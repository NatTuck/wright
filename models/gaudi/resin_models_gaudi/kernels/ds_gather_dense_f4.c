// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// MLA dense slot gather, fp4 KV (DeepSeek-V4.1, D = 512, no rotary key): one slot's C cached compressed
// latents into a CONTIGUOUS staging slab of packed fp4 e2m1 codes + e4m3 group scales (NO dequant here).
// The dequant is a separate producer (ds_ldec_f4) so the compiler SRAM-slices it feeding batch_gemm and
// overlaps TPC-dequant with MME-GEMM.  The fp4 latent page (ds_store_f4.c): 256 B codes/entry (2/row,
// rows 0..63), 32 B e4m3 scales/entry (8/row, rows 64..79).  Keys at or past the slot's klen and pages of
// -1 are zeros.
// slots (8, S) i32: [group, qcol0, qpos0, key0, klen, ...].
// tensors: 0 arena (512, PR, NB) u8  1 lat_pages (C / 128, S) i32  2 rope_pages (unused)  3 slots
//          4 after (unread) | 5 codes (256, C, S) u8  6 scales (32, C, S) u8
// index space: [1, C / 32, S]
#include "common.h"

#define VG_KEYS 32
#define ROW(r) (r) & pm, (r) >> psh

void main(tensor arena, tensor lat_pages, tensor rope_pages, tensor slots, tensor after, tensor codes,
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
    for (int s = off[2]; s < end[2]; s++) {
        int5 kc = {3, s, 0, 0, 0};
        const int key0 = s_i32_ld_g(gen_addr(kc, slots));
        kc[0] = 4;
        const int klen = s_i32_ld_g(gen_addr(kc, slots));
        for (int j = off[1]; j < end[1]; j++) {
            const int page = j / (128 / VG_KEYS);
            int5 pc = {page, s, 0, 0, 0};
            const int lat0 = s_i32_ld_g(gen_addr(pc, lat_pages));
            const int cnt = klen - key0 - page * 128;
            const int q0 = (j % (128 / VG_KEYS)) * VG_KEYS;
            #pragma unroll 4
            for (int t = 0; t < VG_KEYS; t++) {
                const int key = q0 + t;
                const int live = lat0 >= 0 && key < cnt;
                const int r = live ? key : 0;
                int5 ac = {(r & 1) * 256, ROW(lat0 + (r >> 1)), 0, 0};
                const uchar256 c256 = live ? v_u8_ld_tnsr_b(ac, arena) : zero8;
                int5 oc = {0, j * VG_KEYS + t, s, 0, 0};
                v_u8_st_tnsr_partial(oc, codes, c256, 255, 0);
                int5 sc = {(r & 7) * 32, ROW(lat0 + 64 + (r >> 3)), 0, 0};
                const uchar256 s32 = live ? v_u8_ld_tnsr_b(sc, arena) : zero8;
                v_u8_st_tnsr_partial(oc, scales, s32, 31, 0);
            }
        }
    }
}