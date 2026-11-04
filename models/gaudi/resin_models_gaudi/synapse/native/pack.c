// Host-side weight packing for the Gaudi loaders: GGUF Q4_K / Q5_K / Q8_0 blocks into the
// planar layout of the lq_qmv8 GEMVs, and reference dequantization. Plain C with no Synapse
// dependency, called through ctypes (which releases the GIL, so a thread pool runs row blocks
// in parallel).
#include <stdint.h>
#include <string.h>

/* GGUF Q4_K / Q5_K / Q8_0 native blocks -> the planar layout of the lq_qmv8 GEMVs (numpy model:
 * resin_models_gaudi.gguf.planar.gguf_planar). One 128-row block: rows
 * r0 .. r0 + 127 of src (row stride rs bytes, rows past M zero), superblocks sb0 .. sb0 + nsb - 1 of each row.
 * dst: [nsb][SBB] for this row block. type: 0 Q4_K, 1 Q5_K, 2 Q8_0. Returns 0, or -1 for an unknown type. */
long lq_planar_rb(int type, const uint8_t* src, long rs, int M, int r0, int sb0, int nsb, uint8_t* dst)
{
    const int SB = type == 2 ? 32 : 256, blk = type == 0 ? 144 : type == 1 ? 176 : 34;
    const int nf16 = type == 2 ? 1 : 2, nbp = type == 2 ? 0 : 12;
    const int p4sz = type == 2 ? 0 : SB / 4 * 256, p1sz = type == 1 ? SB / 16 * 256 : 0;
    const int SBB = nf16 * 256 + nbp * 128 + p4sz + p1sz + (type == 2 ? SB / 2 * 256 : 0);
    if (type < 0 || type > 2) return -1;
    uint8_t codes[256 * 128];                  /* [k][row] */
    for (int s = 0; s < nsb; s++) {
        uint8_t* d = dst + (long)s * SBB;
        memset(d, 0, SBB);
        memset(codes, 0, sizeof(codes));
        for (int r = 0; r < 128 && r0 + r < M; r++) {
            const uint8_t* b = src + (long)(r0 + r) * rs + (long)(sb0 + s) * blk;
            for (int f = 0; f < nf16; f++) memcpy(d + f * 256 + 2 * r, b + 2 * f, 2);
            for (int p = 0; p < nbp; p++) d[nf16 * 256 + p * 128 + r] = b[4 + p];
            for (int e = 0; e < SB; e++) {
                uint8_t c;
                if (type == 2) {
                    c = b[2 + e];
                } else {
                    const int ch = e / 64, rr = e % 64;
                    c = (b[(type == 0 ? 16 : 48) + ch * 32 + rr % 32] >> (rr < 32 ? 0 : 4)) & 15;
                    if (type == 1) c |= ((b[16 + rr % 32] >> (2 * ch + (rr >= 32))) & 1) << 4;
                }
                codes[e * 128 + r] = c;
            }
        }
        uint8_t* p = d + nf16 * 256 + nbp * 128;
        if (type == 2) {                       /* P8 */
            for (int m = 0; m < SB / 2; m++)
                for (int o = 0; o < 2; o++)
                    for (int r = 0; r < 128; r++)
                        p[256 * m + 4 * (r & 63) + 2 * (r >> 6) + o] = codes[(2 * m + o) * 128 + r];
            continue;
        }
        for (int m = 0; m < SB / 4; m++)       /* P4 */
            for (int o = 0; o < 2; o++)
                for (int r = 0; r < 128; r++)
                    p[256 * m + 4 * (r & 63) + 2 * (r >> 6) + o] =
                        (codes[(4 * m + o) * 128 + r] & 15) | ((codes[(4 * m + 2 + o) * 128 + r] & 15) << 4);
        if (type == 1) {                       /* P1 */
            uint8_t* q = p + p4sz;
            for (int m = 0; m < SB / 16; m++)
                for (int o = 0; o < 2; o++)
                    for (int r = 0; r < 128; r++) {
                        uint8_t v = 0;
                        for (int t = 0; t < 8; t++) v |= ((codes[(16 * m + 2 * t + o) * 128 + r] >> 4) & 1) << t;
                        q[256 * m + 4 * (r & 63) + 2 * (r >> 6) + o] = v;
                    }
        }
    }
    return 0;
}

/* GGUF dequantization (ggml-quants.c reference semantics) of nrows rows (row stride rs bytes) -> f32 [nrows][K]:
 * type 0 Q4_K, 1 Q5_K, 2 Q8_0. */
static inline float f16f(const uint8_t* p)
{
    const uint16_t h = (uint16_t)(p[0] | (p[1] << 8));
    const uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 31, m = h & 1023;
    uint32_t u;
    if (e == 0) {
        if (m == 0) { u = s; }
        else { int k = -1; uint32_t mm = m; do { k++; mm <<= 1; } while (!(mm & 1024)); u = s | ((uint32_t)(127 - 15 - k) << 23) | ((mm & 1023) << 13); }
    } else if (e == 31) u = s | 0x7F800000 | (m << 13);
    else u = s | ((e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &u, 4); return f;
}

static inline void scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m)
{
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

long gguf_dequant_rows(int type, const uint8_t* src, long rs, int nrows, int K, float* out)
{
    for (int r = 0; r < nrows; r++) {
        const uint8_t* b = src + (long)r * rs;
        float* y = out + (long)r * K;
        if (type == 2) {
            for (int s = 0; s < K / 32; s++, b += 34) {
                const float d = f16f(b);
                for (int e = 0; e < 32; e++) y[32 * s + e] = d * (float)(int8_t)b[2 + e];
            }
            continue;
        }
        const int blk = type == 0 ? 144 : 176;
        for (int s = 0; s < K / 256; s++, b += blk) {
            const float d = f16f(b), dmin = f16f(b + 2);
            const uint8_t* sc = b + 4;
            const uint8_t* qh = b + 16;
            const uint8_t* ql = b + (type == 0 ? 16 : 48);
            float* yy = y + 256 * s;
            int is = 0;
            uint8_t u1 = 1, u2 = 2;
            for (int j = 0; j < 256; j += 64) {
                uint8_t s1, m1, s2, m2;
                scale_min_k4(is, sc, &s1, &m1);
                scale_min_k4(is + 1, sc, &s2, &m2);
                const float d1 = d * s1, mm1 = dmin * m1, d2 = d * s2, mm2 = dmin * m2;
                for (int l = 0; l < 32; l++)
                    yy[j + l] = d1 * (float)((ql[l] & 0xF) + (type == 1 && (qh[l] & u1) ? 16 : 0)) - mm1;
                for (int l = 0; l < 32; l++)
                    yy[j + 32 + l] = d2 * (float)((ql[l] >> 4) + (type == 1 && (qh[l] & u2) ? 16 : 0)) - mm2;
                ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
            }
        }
    }
    return 0;
}
