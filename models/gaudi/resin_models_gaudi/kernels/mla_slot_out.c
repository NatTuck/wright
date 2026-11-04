// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Required Notice: Copyright Aaron Botsis
// MLA slot attention output (Mistral-Large-4), for steps whose attention is one slot launch in
// which every query column is in at most one slot (decode: one dense slot per row of at most
// SLOT_KEYS keys, or one sparse slot per column): out = O / l per column, as an output tensor.
// Columns no slot covers, and columns without visible keys, are 0.
// tensors: 0 O (>= D, M, S) f32 (rows past D ignored)  1 mx (Mp, S) f32 (unused)
//          2 l (Mp, S) f32  3 slots (8, S) i32 | 4 out (D, W, G) f32
// index space: [ceil(W / 64), G]
#include "common.h"

// lane k of v broadcast to all lanes
#define GET_LANE(v, k, out)                                                                  \
    { const float lo_ = (float)(k) - 0.5f, hi_ = (float)(k) + 0.5f;                          \
      float64 t_ = v_f32_sel_less_f32_b(lanef, lo_, 0.0f, v_f32_sel_less_f32_b(lanef, hi_, v, 0.0f)); \
      NL_REDUCE_ADD(t_, IDX, out) }

void main(tensor O, tensor mx, tensor l, tensor slots, tensor out)
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const int M = get_dim_size(O, 1), S = get_dim_size(O, 2), W = get_dim_size(out, 1);
    const int D = get_dim_size(out, 0);
    const float64 lanef = v_convert_i32_to_f32_b((int64)read_lane_id_4b_b(), SW_RHNE, 0, 1, 0);
    DECL_IDX(IDX)
    scalar_cache_reset();
    for (int g = off[1]; g < end[1]; g++) {
        for (int cb = off[0]; cb < end[0]; cb++) {
            const int c0 = cb * 64, c1 = c0 + 64 < W ? c0 + 64 : W;
            for (int c = c0; c < c1; c++) {
                int5 oc = {0, c, g, 0, 0};
                for (int d = 0; d < D; d += 64) {
                    oc[0] = d;
                    v_f32_st_tnsr(oc, out, (float64)0.0f);
                }
            }
            for (int s = 0; s < S; s++) {
                int5 fc = {0, s, 0, 0, 0};
                if (s_i32_ld_g(gen_addr(fc, slots)) != g)
                    continue;
                fc[0] = 1;
                const int q0 = s_i32_ld_g(gen_addr(fc, slots));
                const int lo = c0 > q0 ? c0 : q0, hi = c1 < q0 + M ? c1 : q0 + M;
                for (int c = lo; c < hi; c++) {
                    const int m = c - q0;
                    int5 sc = {m & ~63, s, 0, 0, 0};
                    float64 ls;
                    GET_LANE(v_f32_ld_tnsr_b(sc, l), m & 63, ls)
                    const float64 inv = v_f32_sel_less_f32_b(ls, 1.0e-30f, 0.0f, nl_recip(v_f32_max_b(ls, 1.0e-30f)));
                    int5 oc = {0, c, g, 0, 0}, pc = {0, m, s, 0, 0};
                    for (int d = 0; d < D; d += 64) {
                        oc[0] = d;
                        pc[0] = d;
                        v_f32_st_tnsr(oc, out, v_f32_ld_tnsr_b(pc, O) * inv);
                    }
                }
            }
        }
    }
}
