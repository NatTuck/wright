// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// ds_ix_qacc over a (W, C) score slab: the tiled index chain's prefill accumulator, for a tile whose keys were
// gathered and dequantized ONCE and scored by ONE GEMM with the operands swapped.
//
// ds_ix_qacc_o reads S (C, NH*Q, S) -- keys in the first axis -- so a column's 64 keys are NH*Q elements apart
// and it reads one element per line. batch_gemm is y(M,N) = x(K,N).w(K,M)^T, so scoring with x = the key slab
// and w = the query slab puts C in the second axis instead: S (W, C), where a column's run of 128 keys is
// contiguous and the whole block is one 256-byte bf16 vector. That is the read pattern this kernel was written
// for, and it costs nothing upstream: the key slab is still the single shared tile, so there is no broadcast and
// no expansion copy.
//
// Identical otherwise -- same per-slot lim mask, columns, per-head weights and sink -- and `sk` is (C,) because
// the shared gather publishes one scale per key (the scale is a property of the key, not of the slot).
//
// guid: rg_ds_ix_qacc_w_f32 (the glue derives it as rg_<file stem>_f32).
// tensors: 0 S (NH*T, C) bf16  1 w (NH, T) f32  2 slots  3 sq (NH Q, S) f32  4 sk (C,) f32
//        | 5 sink (64, 1) i32  6 score (K, T) f32
// params: scale. index space: [C/1024, (NH*T)/256, S]. NH = 32.
#define SCORE_OUT  // the score slab is a pure OUTPUT, so its parameter comes after sink: without this the
                    // #else signature takes score as an INPUT and every tensor after it shifts by one
#include "common.h"

#define NINF (-3.0e38f)
#define NH 32
#define TQ 8
#define NB 16

#ifdef SCORE_OUT
void main(tensor S, tensor w, tensor slots, tensor sq, tensor sk, tensor sink, tensor score, float scale)
#else
void main(tensor S, tensor w, tensor slots, tensor score, tensor sq, tensor sk, tensor sink, float scale)
#endif
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const float64 lanef = v_convert_i32_to_f32_b((int64)read_lane_id_4b_b(), SW_RHNE, 0, 1, 0);
    scalar_cache_reset();
    for (int s = off[2]; s < end[2]; s++) {
        int5 fc = {0, s, 0, 0, 0};
        if (s_i32_ld_g(gen_addr(fc, slots)) < 0)
            continue;
        fc[0] = 2;
        const int qpos0 = s_i32_ld_g(gen_addr(fc, slots));
        fc[0] = 3;
        const int key0 = s_i32_ld_g(gen_addr(fc, slots));
        fc[0] = 4;
        const int klen = s_i32_ld_g(gen_addr(fc, slots));
        fc[0] = 5;
        const int tok0 = s_i32_ld_g(gen_addr(fc, slots));
        fc[0] = 6;
        int ratio = s_i32_ld_g(gen_addr(fc, slots));
        ratio = ratio > 0 ? ratio : 1;
        for (int g = off[0]; g < end[0]; g++) {
            for (int tq = off[1]; tq < end[1]; tq++)
                for (int t = TQ * tq; t < TQ * tq + TQ; t++) {
                    const int col = tok0 + t;
                    const int seen = (qpos0 + t + 1) / ratio;
                    const int lim = seen < klen ? seen : klen;
                    float wt[NH];
                    #pragma unroll
                    for (int h = 0; h < NH; h++) {
                        int5 wc = {h, col, 0, 0, 0};
                        int5 qc = {h + NH * t, s, 0, 0, 0};
                        wt[h] = s_f32_ld_g(gen_addr(wc, w)) * scale * s_f32_ld_g(gen_addr(qc, sq));
                    }
                    #pragma unroll 1
                    for (int j = 0; j < NB; j += 2) {  // two vectors of keys per bf16 read
                        const int b = NB * g + j;
                        float64 a0 = 0.0f, a1 = 0.0f, c0 = 0.0f, c1 = 0.0f;  // even and odd heads, keys b | b + 1
                        // (W, C): the row is the column's head lane (GLOBAL column, tok0 + t), and the run of
                        // 128 keys is contiguous
                        int5 sc = {NH * col, 64 * b, 0, 0, 0};
                        #pragma unroll
                        for (int h = 0; h < NH; h += 2) {
                            sc[0] = h + NH * col;
                            float128 x = convert_bfloat128_to_float128(v_bf16_ld_tnsr_b(sc, S), SW_LINEAR);
                            a0 = v_f32_mac_b(v_f32_max_b(x.v1, 0.0f), wt[h], a0);
                            c0 = v_f32_mac_b(v_f32_max_b(x.v2, 0.0f), wt[h], c0);
                            sc[0] = h + 1 + NH * col;
                            x = convert_bfloat128_to_float128(v_bf16_ld_tnsr_b(sc, S), SW_LINEAR);
                            a1 = v_f32_mac_b(v_f32_max_b(x.v1, 0.0f), wt[h + 1], a1);
                            c1 = v_f32_mac_b(v_f32_max_b(x.v2, 0.0f), wt[h + 1], c1);
                        }
                        #pragma unroll
                        for (int u = 0; u < 2; u++) {
                            int5 kc = {64 * (b + u), 0, 0, 0, 0};
                            const float64 acc = (u ? c0 + c1 : a0 + a1) * v_f32_ld_tnsr_b(kc, sk);
                            const float64 key = lanef + (float)(key0 + 64 * (b + u));
                            int5 oc = {key0 + 64 * (b + u), col, 0, 0, 0};
                            v_f32_st_tnsr(oc, score, v_f32_sel_less_f32_b(key, (float)lim, acc, NINF));
                        }
                    }
                }
        }
    }
    if (off[0] == 0 && off[1] == 0 && off[2] == 0) {
        int5 zc = {0, 0, 0, 0, 0};
        v_i32_st_tnsr(zc, sink, (int64)1);
    }
}
