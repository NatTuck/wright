// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// Routed experts, between the two batch_gemms: per slot the gate | up rows of the first product (by row
// scale 2^(C-132) and the token's input scale), clamped SwiGLU, then per-slot FP8 quantization (pair order) for the
// second batch_gemm. Width w = w0 + t: the main rows [gate 0..w0 | up 0..w0] come from y, the tail rows
// [gate w0..w | up w0..w] (t = 0 / 32 / 64) from yt. Each member writes one 128-chunk of h (the last: t long) and
// recomputes the whole slot's h for the max.
// tensors: 0 y (2 w0, 1, S) f32, or bf16 with Y16  1 yt (2t, 1, S) f32 (any tensor when t = 0)
//          2 c (2w, E) f32 ([main | tail] rows)  3 sx (1, T) f32  4 ids (64, T) i32 | 5 hq (w, 1, S) fp8  6 hs (1, S) f32
// params: R, lim, t, psh.  index space: [S, ceil(w/128)] (slot = t * R + k).  w0 % 128 == 0, w <= 4096.
// Prefill pages (R = 0): S = rows of 2^psh-row pages, ids = pe (P,) (expert of page s >> psh), sx (1, S) per row, 0
// on rows no pair uses, which are skipped. A member then owns a whole slot (index space [S]).
// LIN (ds_xact_lin.c): hq in natural order, and the tail rows come from y: y (2w, 1, S) holds
// [gate 0..w0 | up 0..w0 | gate w0..w | up w0..w] (yt is not read; t <= 64).
//
// Y16 (ds_xact_lin_b16.c): y is the w13 product in BF16 -- the reference's own expert output dtype
// (``fp8_gemm out_dtype=BF16``), so the tile's bytes halve.  A bf16 tensor load is 128 elements and the widening
// splits them the way the register packs them: even elements in the first vector, odd in the second.  Two facts
// make that harmless here: ``c`` is a per-expert row scale repeated over the rows (so lane order does not matter),
// and the FP8 store's *pair* packer (``pack_nat128``, out[2i + j] = x[64 j + i]) turns an even/odd pair back into
// natural row order.  So h stays in the loader's order and the tensor it writes is the same one the f32 build
// writes.  The tail pair [gate w0..w0+t | up w0..w0+t] is the tile's last 2t rows, contiguous, so one load covers
// both when 2t == 128 (t = 64, the 4-rank width); at t = 32 the pairing needs a lane swap and the 8-rank width
// keeps f32 (``layers.expert_out_dtype``).
#define LIM  // before common.h: it selects the clamped CLAMP_G / CLAMP_U
#include "common.h"

__local__ float64 HB[66];

#ifdef Y16
// 128 consecutive rows of y from dim0 index i (a multiple of 128), widened: rows i, i+2, .. in v1, the odd rows in
// v2 (the loader's order -- see the header).
static inline float128 ld_y(tensor y, int i, int s)
{
    int5 c = {i, 0, s, 0, 0};
    return v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c, y));
}
#endif

void main(tensor y, tensor yt, tensor c, tensor sx, tensor ids, tensor hq, tensor hs, int R, float lim, int t, int psh)
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const int w = get_dim_size(hq, 0), w0 = w - t, E = get_dim_size(c, 1);
    const float64 lanef = v_convert_i32_to_f32_b((int64)read_lane_id_4b_b(), SW_RHNE, 0, 1, 0);
    DECL_IDX(IDX)
    scalar_cache_reset();
    for (int s = off[0]; s < end[0]; s++) {
        const int tk = R ? s / R : s;
        int5 ic = {R ? s % R : s >> psh, R ? tk : 0, 0, 0, 0};
        int e = s_i32_ld_g(gen_addr(ic, ids));
        e = e < 0 ? 0 : (e >= E ? E - 1 : e);
        int5 xc = {0, tk, 0, 0, 0};
        const float xs = s_f32_ld_g(gen_addr(xc, sx));
        if (!R && xs == 0.0f)
            continue;
        float64 amax = 0.0f;
#ifdef Y16
        for (int i = 0; i < w0; i += 128) {
            const float128 g = ld_y(y, i, s), u = ld_y(y, i + w0, s);
            int5 cc = {i, e, 0, 0, 0};
            const float64 gt0 = g.v1 * v_f32_ld_tnsr_b(cc, c) * xs;
            cc[0] = i + w0;
            const float64 up0 = u.v1 * v_f32_ld_tnsr_b(cc, c) * xs;
            const float64 h0 = nl_silus(CLAMP_G(gt0)) * CLAMP_U(up0);
            HB[i / 64] = h0;
            cc[0] = i + 64;
            const float64 gt1 = g.v2 * v_f32_ld_tnsr_b(cc, c) * xs;
            cc[0] = i + w0 + 64;
            const float64 up1 = u.v2 * v_f32_ld_tnsr_b(cc, c) * xs;
            const float64 h1 = nl_silus(CLAMP_G(gt1)) * CLAMP_U(up1);
            HB[i / 64 + 1] = h1;
            amax = v_f32_max_b(v_f32_max_b(amax, v_f32_abs_b(h0)), v_f32_abs_b(h1));
        }
#else
        for (int i = 0; i < w0; i += 64) {
            int5 yc = {i, 0, s, 0, 0}, cc = {i, e, 0, 0, 0};
            const float64 gt = v_f32_ld_tnsr_b(yc, y) * v_f32_ld_tnsr_b(cc, c) * xs;
            yc[0] = i + w0; cc[0] = i + w0;
            const float64 up = v_f32_ld_tnsr_b(yc, y) * v_f32_ld_tnsr_b(cc, c) * xs;
            const float64 h = nl_silus(CLAMP_G(gt)) * CLAMP_U(up);
            HB[i / 64] = h;
            amax = v_f32_max_b(amax, v_f32_abs_b(h));
        }
#endif
#ifdef LIN
        if (t) {
#ifdef Y16
            // the tail pair as one load: the tile's last 2t rows are [gate w0..w0+t | up w0..w0+t], so the even
            // rows of the first half are the gate tail and the even rows of the second half the up tail
            const float128 p = ld_y(y, 2 * w0, s);
            int5 cc = {2 * w0, e, 0, 0, 0};
            const float64 cg = v_f32_ld_tnsr_b(cc, c);
            cc[0] = 2 * w0 + t;
            const float64 cu = v_f32_ld_tnsr_b(cc, c);
            // lanes < t/2 hold the tail's even rows (v1) and its odd rows (v2); the rest is the other tail
            const float64 m = v_f32_sel_less_f32_b(lanef, (float)(t / 2), 1.0f, 0.0f);
            const float64 h0 = nl_silus(CLAMP_G(p.v1 * cg * xs)) * CLAMP_U(SWAP_HALVES(p.v1) * cu * xs) * m;
            const float64 h1 = nl_silus(CLAMP_G(p.v2 * cg * xs)) * CLAMP_U(SWAP_HALVES(p.v2) * cu * xs) * m;
            HB[w0 / 64] = h0;
            HB[w0 / 64 + 1] = h1;
            amax = v_f32_max_b(v_f32_max_b(amax, v_f32_abs_b(h0)), v_f32_abs_b(h1));
#else
            // the tail chunk as one vector: h[w0 .. w0 + t) in lanes < t
            int5 yc = {2 * w0, 0, s, 0, 0}, cc = {2 * w0, e, 0, 0, 0};
            const float64 gt = v_f32_ld_tnsr_b(yc, y) * v_f32_ld_tnsr_b(cc, c) * xs;
            yc[0] = 2 * w0 + t; cc[0] = 2 * w0 + t;
            const float64 up = v_f32_ld_tnsr_b(yc, y) * v_f32_ld_tnsr_b(cc, c) * xs;
            const float64 h = v_f32_sel_less_f32_b(lanef, (float)t, nl_silus(CLAMP_G(gt)) * CLAMP_U(up), 0.0f);
            HB[w0 / 64] = h;
            HB[w0 / 64 + 1] = 0.0f;
            amax = v_f32_max_b(amax, v_f32_abs_b(h));
#endif
        }
#else
        if (t) {
            // pack halves of the tail chunk: a = h[w0 .. w0 + t/2), b = h[w0 + t/2 .. w0 + t) (lanes < t/2 valid)
            #pragma unroll 1
            for (int hb = 0; hb < 2; hb++) {
                const int o = hb * (t / 2);
                int5 yc = {o, 0, s, 0, 0}, cc = {2 * w0 + o, e, 0, 0, 0};
                const float64 gt = v_f32_ld_tnsr_b(yc, yt) * v_f32_ld_tnsr_b(cc, c) * xs;
                yc[0] = t + o; cc[0] = 2 * w0 + t + o;
                const float64 up = v_f32_ld_tnsr_b(yc, yt) * v_f32_ld_tnsr_b(cc, c) * xs;
                const float64 h = v_f32_sel_less_f32_b(lanef, (float)(t / 2), nl_silus(CLAMP_G(gt)) * CLAMP_U(up), 0.0f);
                HB[w0 / 64 + hb] = h;
                amax = v_f32_max_b(amax, v_f32_abs_b(h));
            }
        }
#endif
        NL_REDUCE_MAX(amax, IDX, amax);
        const float64 sc = token_scale(amax);
        const float64 inv = token_inv(sc);
        const int q0 = R ? off[1] : 0, q1 = R ? end[1] : (w + 127) >> 7;
        for (int q = q0; q < q1; q++) {
            int5 qc = {128 * q, 0, s, 0, 0};
#ifdef Y16
            // the pair packer takes the even/odd pair back to natural row order
            const int n = 128 * q < w0 ? 128 : t;
            v_f8_st_tnsr_partial(qc, hq, pack_nat128(HB[2 * q] * inv, HB[2 * q + 1] * inv), n - 1, 0);
#else
#ifdef LIN
            // LIN: hq in natural order (the natural-order activation)
            if (128 * q < w0)
                v_f8_st_tnsr_partial(qc, hq, pack_lin(HB[2 * q] * inv, HB[2 * q + 1] * inv, 0.0f, 0.0f), 127, 0);
            else
                v_f8_st_tnsr_partial(qc, hq, pack_lin(HB[w0 / 64] * inv, 0.0f, 0.0f, 0.0f), t - 1, 0);
#else
            // base build: pair order (pack_nat128), as ds_xq.c and the header document; the w2 weight is
            // stored with pair_rows so out[2i+j] = x[64j+i].  (glm5_3_flash's rg_ds_xact_f32 uses this.)
            if (128 * q < w0)
                v_f8_st_tnsr_partial(qc, hq, pack_nat128(HB[2 * q] * inv, HB[2 * q + 1] * inv), 127, 0);
            else
                v_f8_st_tnsr_partial(qc, hq, pack_nat128(HB[w0 / 64] * inv, HB[w0 / 64 + 1] * inv), t - 1, 0);
#endif
#endif
            if (q == 0)
                STORE_TOKEN_SCALE(hs, s, sc)
        }
    }
}
