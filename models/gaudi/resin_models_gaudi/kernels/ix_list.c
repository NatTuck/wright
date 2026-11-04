// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Required Notice: Copyright Aaron Botsis
// DSA key lists (Mistral-Large-4): per column c, the positions of its kept keys (sparse decode
// attention): first the keys whose mkey(score) is above the threshold (ix_thr), then the ties
// (equal), each in ascending position, at most KS in all (a column that sees at most KS keys
// lists all of them: its threshold 1 is below every key). ix_bcnt's block counts give every
// member its list offsets: member m takes blocks
// [m nb / NM, (m + 1) nb / NM) of nb = ceil(vis / 64).
// Also writes column c's slot record for the sparse attention launch:
//   slots[:, c] = [group (0, or -1 with no visible keys), H c, KS, 0, count, c, 0, 0]
// (qcol0 = H c: query block 1; qpos0 = KS: every listed key is visible; key0 = 0).
// tensors: 0 score (K, T) f32  1 vis (T) i32  2 thr (64, T) i32  3 cnt (K / 64, T) i32
//          4 gsum (K / 4096, T) i32 (ix_bcnt) | 5 list (KS, T) i32  6 slots (8, T) i32
// params: NM (members per column), KS, H. index space: [NM, T]
#include "common.h"

static inline uint64 mkey(float64 v)
{
    const uint64 b = as_uint64(v);
    return as_uint64(v_f32_sel_less_f32_b(v, 0.0f, as_float64(~b), as_float64(b | (uint64)0x80000000u)));
}

// entries [a, b) of the list vector whose first entry is vb, from lanes [a - vb, b - vb) of v
#define FLUSH(vb, a, b) \
    { \
        int5 co_ = {(a), c, 0, 0, 0}; \
        v_i32_st_tnsr_partial(co_, list, v, (b) - (a) - 1, (a) - (vb)); \
    }

void main(tensor score, tensor vis, tensor thr, tensor cnt, tensor gsum, tensor list, tensor slots,
          int NM, int KS, int H)
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const int64 lane = (int64)read_lane_id_4b_b();
    DECL_IDX(IDX)
    scalar_cache_reset();
    for (int c = off[1]; c < end[1]; c++) {
        int5 vc = {c, 0, 0, 0, 0};
        const int lim = s_i32_ld_g(gen_addr(vc, vis));
        const int nb = (lim + 63) >> 6;
        int5 tc = {0, c, 0, 0, 0};
        const uint64 th = v_u32_ld_tnsr_b(tc, thr);
        for (int m = off[0]; m < end[0]; m++) {
            const int b0 = m * nb / NM, b1 = (m + 1) * nb / NM;
            // list offsets: strict keys before b0, all strict keys, ties before b0, all ties
            // (group sums of 64 blocks, then the blocks of b0's group before it)
            int s0 = 0, st = 0, t0 = 0, tt = 0;
            for (int q = 0; q < (nb + 63) >> 6; q++) {
                int5 gc = {q, c, 0, 0, 0};
                const int x = s_i32_ld_g(gen_addr(gc, gsum));
                const int xs = x & 0xFFFF, xt = x >> 16;
                if (q < b0 >> 6) {
                    s0 += xs;
                    t0 += xt;
                }
                st += xs;
                tt += xt;
            }
            for (int b = b0 & ~63; b < b0; b++) {
                int5 cc = {b, c, 0, 0, 0};
                const int x = s_i32_ld_g(gen_addr(cc, cnt));
                s0 += x & 0xFFFF;
                t0 += x >> 16;
            }
            if (m == 0) {  // the slot record
                const int count = st + tt < KS ? st + tt : KS;
                int64 rec = 0;
                rec = v_i32_sel_eq_i32_b(lane, (int64)0, (int64)(lim > 0 ? 0 : -1), rec);
                rec = v_i32_sel_eq_i32_b(lane, (int64)1, (int64)(H * c), rec);
                rec = v_i32_sel_eq_i32_b(lane, (int64)2, (int64)KS, rec);
                rec = v_i32_sel_eq_i32_b(lane, (int64)4, (int64)count, rec);
                rec = v_i32_sel_eq_i32_b(lane, (int64)5, (int64)c, rec);
                int5 rc = {0, c, 0, 0, 0};
                v_i32_st_tnsr_partial(rc, slots, rec, 7, 0);
            }
            // pass 0: strict keys at s0..; pass 1: ties at st + t0..
            for (int pass = 0; pass < 2; pass++) {
                int g = pass == 0 ? s0 : st + t0;  // the next entry's list index
                int vb = g & ~63, va = g;
                int64 v = 0;
                for (int b = b0; b < b1 && g < KS; b++) {
                    int5 cc = {b, c, 0, 0, 0};
                    const int x = s_i32_ld_g(gen_addr(cc, cnt));
                    const int n = pass == 0 ? (x & 0xFFFF) : (x >> 16);
                    if (n == 0)
                        continue;
                    int5 sc = {b * 64, c, 0, 0, 0};
                    const uint64 k = mkey(v_f32_ld_tnsr_b(sc, score));
                    const int64 pos = lane + (int64)(b * 64);
                    // candidate lanes: visible and (above | equal to) the threshold
                    const uint64 none = (uint64)0x7FFFFFFF, upos = as_uint64(pos);
                    int64 cand = as_int64(pass == 0 ? v_u32_sel_less_u32_b(th, k, upos, none)
                                                    : v_u32_sel_eq_u32_b(k, th, upos, none));
                    cand = v_i32_sel_less_i32_b(pos, (int64)lim, cand, (int64)0x7FFFFFFF);
                    for (int i = 0; i < n && g < KS; i++, g++) {
                        int64 low;
                        NL_REDUCE_I32(cand, IDX, low, v_i32_min_b);
                        if (g - vb == 64) {
                            FLUSH(vb, va, g)
                            vb = g;
                            va = g;
                        }
                        v = v_i32_sel_eq_i32_b(lane, (int64)(g - vb), low, v);
                        cand = v_i32_sel_eq_i32_b(cand, low, (int64)0x7FFFFFFF, cand);
                    }
                }
                if (g > va)
                    FLUSH(vb, va, g)
            }
        }
    }
}
