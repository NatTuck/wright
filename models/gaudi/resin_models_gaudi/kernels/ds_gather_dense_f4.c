// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// MLA dense slot gather, fp4 KV (DeepSeek-V4.1, D = 512, no rotary key): one slot's C cached compressed
// latents into a dense bf16 slab, dequantized EXACTLY (fp4 e2m1 * e4m3 group scale -> bf16).  The fp4
// latent page (ds_store_f4.c): 256 B codes/entry (2/row, rows 0..63), 32 B e4m3 scales/entry (8/row,
// rows 64..79).  Keys at or past the slot's klen and pages of -1 are zeros.
// slots (8, S) i32: [group, qcol0, qpos0, key0, klen, ...].
// tensors: 0 arena (512, PR, NB) u8  1 lat_pages (C / 128, S) i32  2 rope_pages (unused)  3 slots
//          4 after (unread) | 5 out (512, C, S) bf16
// index space: [1, C / 32, S]
#include "common.h"
#include "fp4_common.h"

#define VG_KEYS 32
#define ROW(r) (r) & pm, (r) >> psh

void main(tensor arena, tensor lat_pages, tensor rope_pages, tensor slots, tensor after, tensor out)
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
                const uchar512 nib = v_convert_u4_to_u8_all_b(v_u8_ld_tnsr_b(ac, arena, SW_UNPACK | SW_UNPCK_4_TO_8));
                int5 sc = {(r & 7) * 32, ROW(lat0 + 64 + (r >> 3)), 0, 0};
                const uchar256 sb0 = v_u8_ld_tnsr_b(sc, arena);
                sc[0] += 16;
                const uchar256 sb1 = v_u8_ld_tnsr_b(sc, arena);
                const bfloat256 a = fp4_to_bf16_256(nib.v1, sb0);
                const bfloat256 b = fp4_to_bf16_256(nib.v2, sb1);
                int5 oc = {0, j * VG_KEYS + t, s, 0, 0};
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