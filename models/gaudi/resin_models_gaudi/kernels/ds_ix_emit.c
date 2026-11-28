// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// NODE B of the split index merge (DeepSeek-V4.1): turns node A's ordered keyed rows (`lanes` (KS, T), see
// ds_ix_lanes.c) into the column's three outputs -- the list of positions the attention reads, the score, and the
// position compacted into the next group's running rows (sout, pos_out) -- and writes the NINF / -1 tail past the
// listed entries. It also does the invisible tile's identity copy, which is why node A writes no outputs for it.
//
// The split exists because a partial store's source lane is a SCALAR, and a scalar cannot be made from a vector
// inside one kernel: a vector cannot be cast to one, and a vector-store -> s_i32_ld_g round trip through a
// one-element tensor is hoisted out of the loop (the compiler assumes the two intrinsic families do not alias, so
// the read never sees the write; measured: all 512 listed positions came back identical). Across two nodes the
// ordering is the graph's dependency, so this kernel reads `lanes` as a scalar with no aliasing question -- and
// then node A needs ONE reduction per entry instead of the three a single kernel needed to broadcast the position
// and the score at the chosen lane to the slot's lane.
//
// Every value here is a single element, so this node is entirely scalar: one scalar load of the keyed row, one of
// the position (or a scalar add), one of the score's bits, and three scalar stores. The score is moved as an i32
// bit pattern, so the f32 output needs no float scalar path.
// tensors: 0 score (K, T) f32  1 vis (T) i32  2 gsum (K / 4096, T) i32  3 pos_old (512, T) i32
//          4 gbase (1,) i32 (the group's first key; 0 for the first group)  5 lanes (KS, T) i32
//        | 6 list (KS, T) i32  7 sout (KS, T) f32  8 pos_out (512, T) i32
// params: NM, KS, run. index space: [NM, T] -- the SAME member shape as the merge (ds_ix_lanes.c), so a member
// walks a range of slots. It must not be [KS, T]: one member per slot is 512 x T index points, and the bmax chain
// runs at T = 2048 with KS = 2048, i.e. 4.2M members per launch (measured: it took the index chain from 52% to
// 83% of a 93k prefill).
#include "common.h"

#define NINF (-3.0e38f)
#define BIAS 0x80000000
#define NINF_BITS ((int)0xff61b1e6)

void main(tensor score, tensor vis, tensor gsum, tensor pos_old, tensor gbase, tensor lanes,
          tensor list, tensor sout, tensor pos_out, int NM, int KS, int run)
{
    int5 zc = {0, 0, 0, 0, 0};
    const int base = s_i32_ld_g(gen_addr(zc, gbase));
    const int tile_row0 = base < run ? base : run;  // running rows actually filled (matches ix_gvis row0)
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const int64 lane = (int64)read_lane_id_4b_b();
    const float64 ninf = NINF;
    const int64 none32 = (int64)-1;
    scalar_cache_reset();
    for (int c = off[1]; c < end[1]; c++) {
        int5 vc = {c, 0, 0, 0, 0};
        const int lim = s_i32_ld_g(gen_addr(vc, vis));
        const int nb = (lim + 63) >> 6;
        // the column's strict and tie counts: gsum is the packed (strict | ties << 16) sum over 4096-row groups
        int st = 0, tt = 0;
        for (int q = 0; q < (nb + 63) >> 6; q++) {
            int5 gc = {q, c, 0, 0, 0};
            const int x = s_i32_ld_g(gen_addr(gc, gsum));
            st += x & 0xFFFF;
            tt += x >> 16;
        }
        if (lim > 0 && lim <= tile_row0 && lim <= KS && st == lim && tt == 0) {
            // An INVISIBLE tile: every visible row is a running row and none is a tie, so the selection is exactly
            // rows 0 .. lim - 1 copied through. One member per column does it (node A skipped the column too).
            if (off[0] == 0) {
                for (int r = 0; r < lim; r += 64) {
                    int5 sc = {r, c, 0, 0, 0};
                    const float64 sv = v_f32_ld_tnsr_b(sc, score);
                    const int64 pv = v_i32_ld_tnsr_b(sc, pos_old);
                    const int n = lim - r - 1;
                    if (n >= 63) {
                        v_f32_st_tnsr(sc, sout, sv);
                        v_i32_st_tnsr(sc, list, pv);
                        v_i32_st_tnsr(sc, pos_out, pv);
                    } else {
                        v_f32_st_tnsr_partial(sc, sout, sv, n, 0);
                        v_i32_st_tnsr_partial(sc, list, pv, n, 0);
                        v_i32_st_tnsr_partial(sc, pos_out, pv, n, 0);
                    }
                }
                for (int i = lim & ~63; i < KS; i += 64) {  // the slots past the listed entries
                    const int first = i < lim ? lim - i : 0;
                    int5 oc = {i + first, c, 0, 0, 0};
                    if (first) {
                        v_f32_st_tnsr_partial(oc, sout, ninf, 63 - first, first);
                        v_i32_st_tnsr_partial(oc, list, none32, 63 - first, first);
                        v_i32_st_tnsr_partial(oc, pos_out, none32, 63 - first, first);
                    } else {
                        v_f32_st_tnsr(oc, sout, ninf);
                        v_i32_st_tnsr(oc, list, none32);
                        v_i32_st_tnsr(oc, pos_out, none32);
                    }
                }
            }
            continue;
        }
        const int used = st + tt < KS ? st + tt : KS;
        // the index space's dim 0 is the MEMBER (as in the merge), so a member walks its share of the slots
        for (int m = off[0]; m < end[0]; m++)
            for (int a = m * KS / NM; a < (m + 1) * KS / NM; a++) {
            int5 ac = {a, c, 0, 0, 0};
            if (a < used) {
                // the listed entry's keyed row (node A's output), decoded back to a row
                int5 lc = {a, c, 0, 0, 0};
                const int row = s_i32_ld_g(gen_addr(lc, lanes)) ^ (int)BIAS;
                int5 rc = {row, c, 0, 0, 0};
                // a running row's position is pos_old's; this group's own rows follow gbase
                const int pos = row >= tile_row0 ? base + row - tile_row0 : s_i32_ld_g(gen_addr(rc, pos_old));
                const int sc = s_i32_ld_g(gen_addr(rc, score));  // the f32's bits, moved as an i32
                s_i32_st_g(gen_addr(ac, list), pos);
                s_i32_st_g(gen_addr(ac, sout), sc);
                s_i32_st_g(gen_addr(ac, pos_out), pos);
            } else {  // the next merge ranks these last and lists none of them
                s_i32_st_g(gen_addr(ac, list), -1);
                s_i32_st_g(gen_addr(ac, sout), NINF_BITS);
                s_i32_st_g(gen_addr(ac, pos_out), -1);
            }
        }
    }
}
