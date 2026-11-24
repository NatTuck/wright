// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// DeepSeek-V4.1 fp4 code+scale KV store: a KV owner's compressed latent and index key as true fp4
// e2m1 codes + group scales (NOT a re-quantization of the round-tripped value).  Dequant in the gather
// reproduces bf16(f32(code)*f32(scale)) bit-for-bit (the checkpoint's fp4_act_quant inplace value).
//   latent (512 dims): 32 groups of 16, scale = OCP E4M3 of max(amax, 6*2^-9)/6 (nearest even, <=448).
//   index  (128 dims): 4 groups of 32, scale = 2^ceil(log2(max(amax, 6*2^-126)/6)) (E8M0 byte = e+127).
// Layout in the byte arena (512, PR, NB), per entry e of a page whose first row is lat0 (rows[0]):
//   latent codes: 256 B (2 entries/row) at row lat0 + (e>>1), byte (e&1)*256; low nibble = even dim.
//   latent scales: 32 B (8 entries/row) at row lat0 + 64 + (e>>3), byte (e&7)*32.
//   index page first row ix0 = lat0 + 80 (CMP_PAGE_ROWS).
//   index codes: 64 B (8 entries/row) at row ix0 + (e>>3), byte (e&7)*64.
//   index scales: 4 B (128 entries/row) at row ix0 + 16, byte 4*e.
// A code byte c (0..15) is written raw by feeding c*2^-9 to the fp8 encoder (fp8(c*2^-9) == c for
// 0..15), then the SW_PACK store flag nibble-packs adjacent codes (the inverse of mx4_dec's unpack load).
// tensors: 0 arena (512, PR, NB) u8  1 rows (8, G) i32 [lat0, e, ...]  2 lat (512, G) f32
//          3 ix (128, G) f32 | 4 sink (64, 1) i32.  index space: [G]; member 0 writes the sink.
#include "common.h"

#define ROW(r) (r) & pm, (r) >> psh
#define CMP_PAGE_ROWS 80

// the OCP E4M3 byte (bias 7, <=448) of the scale max(amax, 6*2^-9)/6, nearest even (as fp4_ref)
static inline int64 e4m3_scale_byte(float64 amax)
{
    const float64 v = v_f32_max_b(amax, 6.0f * 0.001953125f) * (1.0f / 6.0f);
    const int64 bits = as_int64(v);
    int64 e = ((bits >> 23) & 255) - 127;
    e = v_i32_max_b(e, (int64)-6);
    int64 q = v_convert_f32_to_i32_b(v * as_float64((127 - e + 3) << 23), SW_RHNE, 0, 1, 0);
    const int64 carry = v_i32_sel_eq_i32_b(q, (int64)16, (int64)1, (int64)0);
    e = e + carry;
    q = v_i32_sel_eq_i32_b(q, (int64)16, (int64)8, q);
    const int64 byte = v_i32_sel_eq_i32_b(e, (int64)-6, q, ((e + 7) << 3) | (q - 8));
    return v_i32_min_b(byte, (int64)0x7E);
}

// the e2m1 magnitude code (0..7) of |v| under scale s, ties to the even code (as ds_fq4.c e2m1s)
static inline float64 e2m1_code(float64 v, float64 s)
{
    const float64 a = v_f32_abs_b(v);
    float64 c = 7.0f;
    c = v_f32_sel_leq_f32_b(a, 5.0f * s, 6.0f, c);
    c = v_f32_sel_less_f32_b(a, 3.5f * s, 5.0f, c);
    c = v_f32_sel_leq_f32_b(a, 2.5f * s, 4.0f, c);
    c = v_f32_sel_less_f32_b(a, 1.75f * s, 3.0f, c);
    c = v_f32_sel_leq_f32_b(a, 1.25f * s, 2.0f, c);
    c = v_f32_sel_less_f32_b(a, 0.75f * s, 1.0f, c);
    c = v_f32_sel_leq_f32_b(a, 0.25f * s, 0.0f, c);
    return c;
}

// the low bytes of an i32 vector (lanes 0..N hold byte values) as a contiguous uchar256: bitcast to
// bytes (byte 4j = lane j's low byte) then two even-lane packs collapse the stride-4 layout.
static inline uchar256 pack_bytes(int64 v)
{
    const uchar256 xb = __builtin_bit_cast(uchar256, v);
    const uchar256 p1 = v_u8_pack_b(xb, SW_PCK_8_TO_4, xb);
    return v_u8_pack_b(p1, SW_PCK_8_TO_4, p1);
}

// broadcast lane l (0..63) of an i32 vector to every lane (via the proven f32 dual-group broadcast)
static inline int64 lane_bcast_i32(int64 v, int l)
{
    const float64 f = __builtin_bit_cast(float64, v);
    const float64 b = lane_bcast(f, l);
    return __builtin_bit_cast(int64, b);
}

// the 16-lane group max of |v| broadcast to the group's lanes (4 groups of 16 in a 64-lane vector)
static inline float64 group16_max(float64 a)
{
    float64 r0 = v_f32_shuffle_b(a, SHUF_IDX(0), 0, a), r1 = v_f32_shuffle_b(a, SHUF_IDX(1), 0, a);
    float64 r2 = v_f32_shuffle_b(a, SHUF_IDX(2), 0, a), r3 = v_f32_shuffle_b(a, SHUF_IDX(3), 0, a);
    #pragma unroll
    for (int t = 4; t < 16; t += 4) {
        r0 = v_f32_max_b(r0, v_f32_shuffle_b(a, SHUF_IDX(t), 0, a));
        r1 = v_f32_max_b(r1, v_f32_shuffle_b(a, SHUF_IDX(t + 1), 0, a));
        r2 = v_f32_max_b(r2, v_f32_shuffle_b(a, SHUF_IDX(t + 2), 0, a));
        r3 = v_f32_max_b(r3, v_f32_shuffle_b(a, SHUF_IDX(t + 3), 0, a));
    }
    return v_f32_max_b(v_f32_max_b(r0, r1), v_f32_max_b(r2, r3));
}

void main(tensor arena, tensor rows, tensor lat, tensor ix, tensor sink)
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const int PR = get_dim_size(arena, 1);
    int psh = 0;
    while ((1 << psh) < PR)
        psh++;
    const int pm = PR - 1;
    const int64 lane = (int64)read_lane_id_4b_b();
    scalar_cache_reset();
    for (int n = off[0]; n < end[0]; n++) {
        int5 rc = {0, n, 0, 0, 0};
        const int lat0 = s_i32_ld_g(gen_addr(rc, rows));
        rc[0] = 1;
        const int e = s_i32_ld_g(gen_addr(rc, rows));
        if (lat0 < 0)
            continue;
        // ---- latent: 8 vectors of 64 dims, 4 groups of 16 each
        int5 cc = {(e & 1) * 256, ROW(lat0 + (e >> 1)), 0, 0};
        int5 sc = {(e & 7) * 32, ROW(lat0 + 64 + (e >> 3)), 0, 0};
        float64 fv[8];
        #pragma unroll
        for (int b = 0; b < 8; b++) {
            int5 kc = {64 * b, n, 0, 0, 0};
            const float64 v = v_f32_ld_tnsr_b(kc, lat);
            const float64 m = group16_max(v_f32_abs_b(v));
            const int64 sbyte = e4m3_scale_byte(m);
            const int64 se = (sbyte >> 3) & 15;
            const int64 sm = sbyte & 7;
            const float64 smf = v_convert_i32_to_f32_b(sm, SW_RHNE, 0, 1, 0);
            const float64 sef = v_convert_i32_to_f32_b(se, SW_RHNE, 0, 1, 0);
            const float64 normal = as_float64(((se - 7 + 127) << 23)) * (8.0f + smf) * (1.0f / 8.0f);
            const float64 sub = smf * 0.001953125f;
            const float64 sv = v_f32_sel_grt_f32_b(sef, 0.0f, normal, sub);
            float64 cf = e2m1_code(v, sv);
            cf = v_f32_sel_less_f32_b(v, 0.0f, cf + 8.0f, cf);  // code 0..15 (sign bit 3)
            fv[b] = cf * 0.001953125f;  // fp8(code*2^-9) == code byte
            // 4 group scale bytes (lanes 0,16,32,48) -> 4 consecutive bytes
            int64 s4 = 0;
            #pragma unroll
            for (int g = 0; g < 4; g++)
                s4 = v_i32_sel_eq_i32_b(lane, (int64)g, lane_bcast_i32(sbyte, 16 * g), s4);
            v_u8_st_tnsr_partial(sc, arena, pack_bytes(s4), 3, 0);
            sc[0] += 4;
        }
        // 512 codes -> 256 packed bytes (two 256-code fp8 rounds, each SW_PACKed to 128 bytes)
        v_u8_st_tnsr_partial(cc, arena, __builtin_bit_cast(uchar256, pack_lin(fv[0], fv[1], fv[2], fv[3])), 127, 0, SW_PACK | SW_PCK_8_TO_4);
        cc[0] += 128;
        v_u8_st_tnsr_partial(cc, arena, __builtin_bit_cast(uchar256, pack_lin(fv[4], fv[5], fv[6], fv[7])), 127, 0, SW_PACK | SW_PCK_8_TO_4);
        // ---- index: 2 vectors of 64 dims, 2 groups of 32 each (E8M0 scale)
        const int ix0 = lat0 + CMP_PAGE_ROWS;
        int5 ic = {(e & 7) * 64, ROW(ix0 + (e >> 3)), 0, 0};
        int5 isc = {4 * e, ROW(ix0 + 16), 0, 0};
        float64 gi[2];
        #pragma unroll
        for (int b = 0; b < 2; b++) {
            int5 kc = {64 * b, n, 0, 0, 0};
            const float64 v = v_f32_ld_tnsr_b(kc, ix);
            float64 m16 = group16_max(v_f32_abs_b(v));
            float64 m = v_f32_max_b(m16, v_f32_mov_dual_group_all_b(m16, 0xFFFFFFFF, 1, 0, 3, 2, WR_ALL, m16));
            m = v_f32_max_b(m, 6.0f * 1.1754944e-38f);
            const float64 rr = m * (1.0f / 6.0f);
            const int64 eb = ((as_int64(rr) >> 23) & 255);
            const int64 man_nonzero = v_i32_sel_eq_i32_b(as_int64(rr) & 0x7FFFFF, (int64)0, (int64)0, (int64)1);
            const int64 ee = eb - 127 + man_nonzero;  // ceil(log2(m/6))
            const float64 s = as_float64((ee + 127) << 23);
            float64 cf = e2m1_code(v, s);
            cf = v_f32_sel_less_f32_b(v, 0.0f, cf + 8.0f, cf);
            gi[b] = cf * 0.001953125f;
            const int64 sbyte = ee + 127;
            int64 s2 = 0;
            #pragma unroll
            for (int g = 0; g < 2; g++)
                s2 = v_i32_sel_eq_i32_b(lane, (int64)g, lane_bcast_i32(sbyte, 32 * g), s2);
            v_u8_st_tnsr_partial(isc, arena, pack_bytes(s2), 1, 0);
            isc[0] += 2;
        }
        // 128 codes -> 64 packed bytes
        v_u8_st_tnsr_partial(ic, arena, __builtin_bit_cast(uchar256, pack_lin(gi[0], gi[1], 0.0f, 0.0f)), 63, 0, SW_PACK | SW_PCK_8_TO_4);
    }
    if (off[0] == 0) {
        int5 zc = {0, 0, 0, 0, 0};
        v_i32_st_tnsr(zc, sink, (int64)1);
    }
}