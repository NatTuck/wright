// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// fp4 e2m1 decode helpers for the DeepSeek-V4.1 KV gathers.  The SW_UNPACK|SW_UNPCK_4_TO_8 load expands
// 256 packed bytes to 512 natural-order nibbles (uchar512: v1 = codes 0..255, v2 = codes 256..511).
#pragma once

// e2m1 nibble (0..15) -> the fp8 e4m3 byte of its value (0, +-0.5, 1, 1.5, 2, 3, 4, 6); exact in fp8.
static inline uchar256 e2m1_to_fp8(uchar256 n)
{
    const uchar256 m = v_u8_and_b(n, (uchar256)7);
    const uchar256 normal = v_u8_or_b(
        v_u8_shl_b(v_u8_add_b(v_u8_shr_b(m, (char256)1), (uchar256)6), (char256)3),
        v_u8_shl_b(v_u8_and_b(m, (uchar256)1), (char256)2));
    const uchar256 mag = v_u8_sel_eq_u8_b(m, (uchar256)0, (uchar256)0,
                         v_u8_sel_eq_u8_b(m, (uchar256)1, (uchar256)0x30, normal));
    return v_u8_or_b(mag, v_u8_shl_b(v_u8_and_b(n, (uchar256)8), (char256)4));
}

// 16 e4m3 scale bytes (sb lanes 0..15) -> bf16 in lanes 0..15 (OCP e4m3 -> bf16 is exact).  Subnormal
// scales (se == 0) are normalized: sm*2^-9.  KV scales are >= 2^-9; the se==0 path covers them.
static inline bfloat128 e4m3_scale_to_bf16(uchar256 sb)
{
    const ushort128 sb16 = __builtin_bit_cast(ushort128, v_u8_unpack_b(sb, SW_UNPCK_8_TO_16 | SW_GROUP_0, (uchar256)0));
    const ushort128 se = v_u16_and_b(v_u16_shr_b(sb16, (short128)3), (ushort128)15);
    const ushort128 sm = v_u16_and_b(sb16, (ushort128)7);
    const ushort128 normal = v_u16_or_b(v_u16_shl_b(v_u16_add_b(se, (ushort128)120), (short128)7),
                                        v_u16_shl_b(sm, (short128)4));
    // subnormal: sm * 2^-9 -> bf16 (sm in 1..7)
    const ushort128 sub = v_u16_sel_eq_u16_b(sm, (ushort128)1, (ushort128)0x3B00,
                          v_u16_sel_eq_u16_b(sm, (ushort128)2, (ushort128)0x3B80,
                          v_u16_sel_eq_u16_b(sm, (ushort128)3, (ushort128)0x3BC0,
                          v_u16_sel_eq_u16_b(sm, (ushort128)4, (ushort128)0x3C00,
                          v_u16_sel_eq_u16_b(sm, (ushort128)5, (ushort128)0x3C20,
                          v_u16_sel_eq_u16_b(sm, (ushort128)6, (ushort128)0x3C40,
                          v_u16_sel_eq_u16_b(sm, (ushort128)7, (ushort128)0x3C60, (ushort128)0)))))));
    const ushort128 bits = v_u16_sel_eq_u16_b(se, (ushort128)0, sub, normal);
    return __builtin_bit_cast(bfloat128, bits);
}
// broadcast sf lane (base+g) to group g's 16 lanes of a 64-lane f32 vector (g=0..3)
static inline float64 scale4_broadcast(float64 sf, int base)
{
    const int64 lane = (int64)read_lane_id_4b_b();
    float64 s = 0.0f;
    #pragma unroll
    for (int g = 0; g < 4; g++)
        s = v_f32_sel_eq_i32_b(lane, (int64)(16 * g), lane_bcast(sf, base + g), s);
    return v_f32_shuffle_b(s, SHUF_IDX(0), 0, s);
}

// bf (128 bf16 codes, 8 groups of 16) * sbf (lanes base..base+7 = 8 e4m3 scales) -> 128 bf16 (natural).
// Direct bf16 multiply (as the fp8 gather ds_gather_list does f.v1 * su): the fp4 product
// e2m1(1 mantissa bit) x e4m3(3 bits) needs <= 4 significant bits and fits bf16 exactly.  The scale is
// broadcast in f32 natural order (scale4_broadcast) then mapped to the bf16 REGISTER order with
// convert_float128_to_bfloat128 SW_LINEAR -- the exact inverse of the codes' bf16->f32 SW_LINEAR widen,
// so it matches bf's register order (v_convert_f32_to_bf16_all_b does NOT; it misaligns the groups).
// bf16 multiply by a positive scale preserves -0.
static inline bfloat128 mul_scale128(bfloat128 bf, bfloat128 sbf, int base)
{
    const float128 sf = convert_bfloat128_to_float128(sbf, SW_LINEAR);
    float128 sb2;
    sb2.v1 = scale4_broadcast(sf.v1, base);
    sb2.v2 = scale4_broadcast(sf.v1, base + 4);
    const bfloat128 su = convert_float128_to_bfloat128(sb2, SW_RHNE | SW_LINEAR);
    const bfloat128 out = bf * su;
    const ushort128 sign = v_u16_and_b(__builtin_bit_cast(ushort128, bf), (ushort128)0x8000);
    return __builtin_bit_cast(bfloat128, v_u16_or_b(__builtin_bit_cast(ushort128, out), sign));
}

// 256 natural-order e2m1 nibbles + 16 e4m3 scale bytes (sb lanes 0..15) -> 256 bf16 (bfloat256)
static inline bfloat256 fp4_to_bf16_256(uchar256 nib, uchar256 sb)
{
    const uchar256 fp8 = e2m1_to_fp8(nib);
    const bfloat256 bf = convert_minifloat256_to_bfloat256(__builtin_bit_cast(minifloat256, fp8), SW_LINEAR);
    const bfloat128 sbf = e4m3_scale_to_bf16(sb);
    bfloat256 r;
    r.v1 = mul_scale128(bf.v1, sbf, 0);
    r.v2 = mul_scale128(bf.v2, sbf, 8);
    return r;
}
