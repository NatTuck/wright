// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// DeepSeek-V4.1 candidate blocks, sparse scoring, fp4 KV: gather one slot's 256 candidate blocks of 8
// fp4 index keys (packed e2m1 codes + e8m0 scales) into CONTIGUOUS staging for mx4_decs (fp4 -> fp8 with
// a PER-KEY tensor scale k = maxS-132, sk = 2^k restored by qacc; k=0 would saturate fp8 for large
// scales), and each slab entry's position.  Slot s is column c = slots[5, s], slab entries
// key0 .. key0+2047 (key0 = slots[3, s]); block u of cand[u, c] holds positions 8b..8b+7.  An fp4 index
// page (ds_store_f4.c): 64 B codes/key (8/row, rows 0..15), 4 B e8m0 scales/key (row 16).  Blocks past
// the column's count read position 0; slots of group -1 are skipped.
// tensors: 0 arena (512, PR, NB) u8  1 cand (KB, T) i32  2 cslots (8, T) i32  3 blocks (P, T) i32
//          4 loff (64,) i32  5 slots (8, S) i32  6 after (unread)
//        | 7 codes (64, 2048, S) u8  8 scales (4, 2048, S) u8  9 sk (2048, S) f32 (=2^k)
//          10 ktab (2048, S) i32 (=k)  11 ids (2048, S) i32 (=c+2048*s)  12 cpos (KC, T) i32
// params: page_rows (rows of a fp4 latent page = CMP_PAGE_ROWS).  index space: [1, 32, S]
#include "common.h"

#define ROW(r) (r) & pm, (r) >> psh

void main(tensor arena, tensor cand, tensor cslots, tensor blocks, tensor loff, tensor slots, tensor after,
          tensor codes, tensor scales, tensor sk, tensor ktab, tensor ids, tensor cpos, int page_rows)
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const int PR = get_dim_size(arena, 1);
    const int C = get_dim_size(codes, 1);
    int psh = 0;
    while ((1 << psh) < PR)
        psh++;
    const int pm = PR - 1;
    const int64 lane = (int64)read_lane_id_4b_b();
    const uchar256 zero8 = 0;
    scalar_cache_reset();
    int5 lc = {0, 0, 0, 0, 0};
    const int ioff = s_i32_ld_g(gen_addr(lc, loff)) + page_rows;
    for (int s = off[2]; s < end[2]; s++) {
        int5 kc = {0, s, 0, 0, 0};
        if (s_i32_ld_g(gen_addr(kc, slots)) < 0)
            continue;
        kc[0] = 5;
        const int c = s_i32_ld_g(gen_addr(kc, slots));
        kc[0] = 3;
        const int key0 = s_i32_ld_g(gen_addr(kc, slots));
        int5 nc = {4, c, 0, 0, 0};
        const int n = s_i32_ld_g(gen_addr(nc, cslots));
        for (int j = off[1]; j < end[1]; j++) {
            const int u0 = (key0 >> 3) + 8 * j;
            int p0[8], r0[8];
            #pragma unroll
            for (int u = 0; u < 8; u++) {
                int5 cc = {u0 + u, c, 0, 0, 0};
                const int P = u0 + u < n ? s_i32_ld_g(gen_addr(cc, cand)) : 0;
                p0[u] = 8 * ((P & ~63) + 8 * (P & 7) + ((P >> 3) & 7));
            }
            #pragma unroll
            for (int u = 0; u < 8; u++) {
                int5 bc = {p0[u] >> 7, c, 0, 0, 0};
                r0[u] = s_i32_ld_g(gen_addr(bc, blocks)) + ioff;
            }
            int64 pv = 0;
            #pragma unroll
            for (int u = 0; u < 8; u++) {
                pv = v_i32_sel_eq_i32_b(lane >> 3, (int64)u, (int64)p0[u] + (lane & 7), pv);
                const int r = p0[u] & 127;  // multiple of 8
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    const int key = r + e;
                    int5 ic = {e * 64, ROW(r0[u] + (r >> 3)), 0, 0};
                    const uchar256 c64 = v_u8_ld_tnsr_b(ic, arena);
                    const int cc = 64 * j + 8 * u + e;
                    int5 oc = {0, cc, s, 0, 0};
                    v_u8_st_tnsr_partial(oc, codes, c64, 63, 0);
                    int5 sc = {4 * key, ROW(r0[u] + 16), 0, 0};
                    const uchar256 s4 = v_u8_ld_tnsr_b(sc, arena);
                    v_u8_st_tnsr_partial(oc, scales, s4, 3, 0);
                    const int s32 = s_i32_ld_g(gen_addr(sc, arena));
                    int maxS = s32 & 0xFF;
                    const int b1 = (s32 >> 8) & 0xFF, b2 = (s32 >> 16) & 0xFF, b3 = (s32 >> 24) & 0xFF;
                    maxS = maxS > b1 ? maxS : b1;
                    maxS = maxS > b2 ? maxS : b2;
                    maxS = maxS > b3 ? maxS : b3;
                    maxS = maxS < 6 ? 6 : maxS;
                    const int k = maxS - 132;
                    int5 osk = {cc, s, 0, 0, 0};  // sk/ktab/ids are (2048, S): dim0=cc, dim1=s
                    v_f32_st_tnsr_partial(osk, sk, as_float64((int64)(maxS - 5) << 23), 0, 0);
                    v_i32_st_tnsr_partial(osk, ktab, (int64)k, 0, 0);
                    v_i32_st_tnsr_partial(osk, ids, (int64)(cc + C * s), 0, 0);
                }
            }
            int5 pc = {key0 + 64 * j, c, 0, 0, 0};
            v_i32_st_tnsr(pc, cpos, pv);
        }
    }
}