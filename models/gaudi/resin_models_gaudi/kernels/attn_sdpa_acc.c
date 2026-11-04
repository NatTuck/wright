// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Required Notice: Copyright Aaron Botsis
// Slot attention, accumulate a launch's normalized slot results (attn_slot_acc's merge). Slot s
// holds, per head h and token t of its Tq query tokens (chunk tokens qtok0 .. qtok0 + Tq - 1), the
// normalized output O, the row maximum mx and the inverse weight sum linv. Token t of head h is
// accumulator column h + H t of the slot's group:
//   n = max(acc_m, mx),  wa = exp(acc_m - n) acc_l,  ws = exp(mx - n) / linv,
//   acc_o = (wa acc_o + ws O) / (wa + ws),  acc_m = n,  acc_l = wa + ws
// acc_m and acc_l are token-major per head here: element (t + T h, g).
// A member owns 64 tokens of one head and group; padding slots (group -1) are skipped. Query
// blocks start at and span multiples of 64 tokens.
// slots (8, S) i32 per slot: [group, qcol0, qpos0, key0, klen, qtok0, 0, 0]
// tensors: 0 O (256, Tq, H, S) f32  1 mx (Tq, H, S) f32  2 linv (Tq, H, S) f32  3 slots (8, S)
//          4 acc_o (256, W, G) f32  5 acc_m (W, G) f32  6 acc_l (W, G) f32 | 7 sink (64, 1) i32
// params: T (chunk tokens, W / H). index space: [ceil(T / 64), H, G]; member (0, 0, 0) writes
// the sink.
#include "common.h"

// lane k of v broadcast to all lanes
#define GET_LANE(v, k, out)                                                                  \
    { const float lo_ = (float)(k) - 0.5f, hi_ = (float)(k) + 0.5f;                          \
      float64 t_ = v_f32_sel_less_f32_b(lanef, lo_, 0.0f, v_f32_sel_less_f32_b(lanef, hi_, v, 0.0f)); \
      NL_REDUCE_ADD(t_, IDX, out) }

void main(tensor O, tensor mx, tensor linv, tensor slots, tensor acc_o, tensor acc_m,
          tensor acc_l, tensor sink, int T)
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const int Tq = get_dim_size(O, 1), H = get_dim_size(O, 2), S = get_dim_size(O, 3);
    const float64 lanef = v_convert_i32_to_f32_b((int64)read_lane_id_4b_b(), SW_RHNE, 0, 1, 0);
    DECL_IDX(IDX)
    scalar_cache_reset();
    for (int g = off[2]; g < end[2]; g++) {
        for (int h = off[1]; h < end[1]; h++) {
            for (int tb = off[0]; tb < end[0]; tb++) {
                const int t0 = tb * 64, nt = t0 + 64 < T ? 64 : T - t0;
                int5 ac = {T * h + t0, g, 0, 0, 0};
                float64 am = v_f32_ld_tnsr_b(ac, acc_m), al = v_f32_ld_tnsr_b(ac, acc_l);
                for (int s = 0; s < S; s++) {
                    int5 fc = {0, s, 0, 0, 0};
                    if (s_i32_ld_g(gen_addr(fc, slots)) != g)
                        continue;
                    fc[0] = 5;
                    const int tl = t0 - s_i32_ld_g(gen_addr(fc, slots));
                    if (tl < 0 || tl >= Tq)
                        continue;
                    int5 sc = {tl, h, s, 0, 0};
                    const float64 ms = v_f32_ld_tnsr_b(sc, mx), li = v_f32_ld_tnsr_b(sc, linv);
                    const float64 n = v_f32_max_b(am, ms);
                    const float64 wa = nl_exp(am - n) * al;
                    const float64 ws = nl_exp(ms - n) * nl_recip(v_f32_max_b(li, 1.0e-30f));
                    const float64 den = wa + ws;
                    const float64 inv = nl_recip(v_f32_max_b(den, 1.0e-30f));
                    const float64 a = wa * inv, b = ws * inv;
                    int5 oc = {0, h + H * t0, g, 0, 0}, pc = {0, tl, h, s, 0};
                    for (int r = 0; r < nt; r++) {
                        float64 ar, br;
                        GET_LANE(a, r, ar)
                        GET_LANE(b, r, br)
                        #pragma unroll
                        for (int d = 0; d < 256; d += 64) {
                            oc[0] = d;
                            pc[0] = d;
                            const float64 acc = v_f32_ld_tnsr_b(oc, acc_o) * ar;
                            v_f32_st_tnsr(oc, acc_o, v_f32_mac_b(v_f32_ld_tnsr_b(pc, O), br, acc));
                        }
                        oc[1] += H;
                        pc[1]++;
                    }
                    am = n;
                    al = den;
                }
                v_f32_st_tnsr(ac, acc_m, am);
                v_f32_st_tnsr(ac, acc_l, al);
            }
        }
    }
    if (off[0] == 0 && off[1] == 0 && off[2] == 0) {
        int5 zc = {0, 0, 0, 0, 0};
        v_i32_st_tnsr(zc, sink, (int64)1);
    }
}
