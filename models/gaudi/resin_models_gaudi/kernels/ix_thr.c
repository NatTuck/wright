// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Required Notice: Copyright Aaron Botsis
// DSA top-k threshold per query column (Mistral-Large-4): over the column's visible keys
// 0 .. vis[c] - 1 of the score scratch (ix_slot_acc), the KS-th largest score as a monotone
// u32 key (bisection; exact ties may keep a few more). Columns that see at
// most KS keys keep every key (threshold 1: every float maps to a key >= 1). A key is kept when
// mkey(score) >= thr; mla_slot_pk applies it.
// tensors: 0 score (K, T) f32  1 vis (T) i32 | 2 thr (64, T) i32 (every lane holds the column's
//          threshold)
// params: KS. index space: [T]
#include "common.h"

static inline uint64 mkey(float64 v)
{
    const uint64 b = as_uint64(v);
    return as_uint64(v_f32_sel_less_f32_b(v, 0.0f, as_float64(~b), as_float64(b | (uint64)0x80000000u)));
}

void main(tensor score, tensor vis, tensor thr, int KS)
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const float64 lanef = v_convert_i32_to_f32_b((int64)read_lane_id_4b_b(), SW_RHNE, 0, 1, 0);
    const uint64 one = as_uint64((float64)1.0f);
    DECL_IDX(IDX)
    scalar_cache_reset();
    for (int c = off[0]; c < end[0]; c++) {
        int5 vc = {c, 0, 0, 0, 0};
        const int lim = s_i32_ld_g(gen_addr(vc, vis));
        const float limf = (float)lim;
        uint64 lo = 1;
        if (lim > KS) {
            uint64 hi = (uint64)0xFFFFFFFFu;
            for (int it = 0; it < 32; it++) {
                const uint64 d = hi - lo;
                const uint64 mid = lo + v_u32_shr_b(d, (int64)1) + (d & (uint64)1);
                float64 c0 = 0.0f, c1 = 0.0f;
                for (int j = 0; j < lim; j += 128) {
                    int5 sc = {j, c, 0, 0, 0};
                    const uint64 k0 = mkey(v_f32_ld_tnsr_b(sc, score));
                    sc[0] = j + 64;
                    const uint64 k1 = mkey(v_f32_ld_tnsr_b(sc, score));
                    const float64 r0 = lanef + (float)j, r1 = r0 + 64.0f;
                    c0 += v_f32_sel_less_f32_b(r0, limf, as_float64(v_u32_sel_less_u32_b(k0, mid, (uint64)0, one)), 0.0f);
                    c1 += v_f32_sel_less_f32_b(r1, limf, as_float64(v_u32_sel_less_u32_b(k1, mid, (uint64)0, one)), 0.0f);
                }
                float64 tot;
                NL_REDUCE_ADD(c0 + c1, IDX, tot);
                const float64 ok = v_f32_sel_less_f32_b(tot, (float)KS, 0.0f, 1.0f);  // >= KS at or above mid
                lo = as_uint64(v_f32_sel_eq_f32_b(ok, 1.0f, as_float64(mid), as_float64(lo)));
                hi = as_uint64(v_f32_sel_eq_f32_b(ok, 1.0f, as_float64(hi), as_float64(mid - (uint64)1)));
            }
        }
        int5 oc = {0, c, 0, 0, 0};
        v_i32_st_tnsr(oc, thr, __builtin_bit_cast(int64, lo));
    }
}
