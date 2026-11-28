// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// The tiled index selection's merge, NODE A (DeepSeek-V4.1): ix_list over one group's merge buffer — the running
// candidates of the earlier groups in rows 0 .. tile_row0, the group's own scores in rows tile_row0 .. vis — and
// each listed entry's keyed row is written to `lanes` in list order.
//
// NODE A of the merge, split in two because the emit needs a SCALAR row and a scalar cannot be made from a
// vector inside one kernel: a partial store's source lane is a scalar, a vector cannot be cast to one, and a
// vector-store -> s_i32_ld_g round trip through a one-element tensor is hoisted out of the loop by the compiler
// (the two intrinsic families are assumed not to alias, so the read never sees the write; measured: all 512
// listed positions identical). Across two nodes the ordering is the graph's dependency, so node B may read this
// kernel's output as a scalar and use it as a store's source lane. This node therefore does ONLY the ordering:
// one reduction and one store per entry, instead of the three reductions one kernel needed to broadcast the
// position and the score at the chosen lane to the slot's lane.
// The order is ix_list's: the entries above the threshold in ascending row, then the ties (equal to it) in
// ascending row, at most KS in all; every listed entry's position is its row's position, pos_old[row] for a
// running row (an earlier group's entry) or base + row - tile_row0 for one of this group's. Because a group's
// keys all follow the running set's, row order restricted to equal scores is position order, so a merge of merges
// lists exactly what one pass over every key would (ds41's tiling keeps the full selection bit-identical).
// Slots past the listed ones are set to NINF / -1 so that a later merge ranks them last, and the slot record's
// count is the fill's exact count (the column's visible entries, at most KS) rather than the count that the
// padded running rows would give.
// tensors: 0 score (K, T) f32  1 vis (T) i32  2 thr (64, T) i32  3 cnt (K / 64, T) i32
//          4 gsum (K / 4096, T) i32  5 exact (T) i32  6 pos_old (512, T) i32
//        | 7 gbase (1,) i32 (the group's first key; 0 for the first group)
//          8 slots (8, T) i32  9 lanes (KS, T) i32 (the listed entries' keyed rows, in list order)
// params: NM, KS, H, run. index space: [NM, T]
// node A of the split merge: the ordering only (see ds_ix_emit.c for the emit)
#include "common.h"

#define NINF (-3.0e38f)

static inline uint64 mkey(float64 v)
{
    const uint64 b = as_uint64(v);
    return as_uint64(v_f32_sel_less_f32_b(v, 0.0f, as_float64(~b), as_float64(b | (uint64)0x80000000u)));
}

// entry (slot, keyed row) -> the slot's listed entry. The keyed row is uniform across the lanes, so a one-lane
// partial store writes it with no reduction at all: this node never needs a scalar, which is exactly why the
// emit can live in a second kernel (see the header).
#define DECODE(v) ((v) ^ (int64)BIAS)
#define STORE_ENTRY(slot, keyed)                                                                  \
    {                                                                                             \
        int5 co_ = {(slot), c, 0, 0, 0};                                                          \
        v_i32_st_tnsr_partial(co_, lanes, (keyed), 0, 0);                                         \
    }

#define BIAS 0x80000000
// min / max over the 64 lanes into every lane: dual groups, then a butterfly within a dual group
#define RED(x, out, OP) \
    { \
        int64 s_ = OP((x), v_i32_mov_dual_group_all_b((x), 0xFFFFFFFF, 1, 0, 3, 2, WR_ALL, (x))); \
        s_ = OP(s_, v_i32_mov_dual_group_all_b(s_, 0xFFFFFFFF, 2, 3, 0, 1, WR_ALL, s_)); \
        s_ = OP(s_, v_i32_shuffle_b(s_, BF1, 0, s_)); \
        s_ = OP(s_, v_i32_shuffle_b(s_, BF2, 0, s_)); \
        s_ = OP(s_, v_i32_shuffle_b(s_, BF4, 0, s_)); \
        out = OP(s_, v_i32_shuffle_b(s_, BF8, 0, s_)); \
    }

// lane l <- element e (0..15) of its dual group
static inline uchar256 ctl_of(int64 e)
{
    const int64 c = (0x80 | ((e >> 3) << 5) | (e & 7)) * 0x01010101;
    return __builtin_bit_cast(uchar256, c);
}

void main(tensor score, tensor vis, tensor thr, tensor cnt, tensor gsum, tensor exact, tensor pos_old,
          tensor gbase,
#ifdef CPOS
          tensor cpos,
#endif
          tensor slots, tensor lanes,
          int NM, int KS, int H, int run)
{
    int5 zc = {0, 0, 0, 0, 0};
    const int base = s_i32_ld_g(gen_addr(zc, gbase));
    const int tile_row0 = base < run ? base : run;  // running rows actually filled (matches ix_gvis row0)
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const int64 lane = (int64)read_lane_id_4b_b();
    const int64 e16 = lane & 15;
    const uchar256 BF1 = ctl_of(e16 ^ 1), BF2 = ctl_of(e16 ^ 2), BF4 = ctl_of(e16 ^ 4), BF8 = ctl_of(e16 ^ 8);
    const float64 ninf = NINF;
    const int64 none32 = (int64)-1;
    scalar_cache_reset();
    for (int c = off[1]; c < end[1]; c++) {
        int5 vc = {c, 0, 0, 0, 0};
        const int lim = s_i32_ld_g(gen_addr(vc, vis));
        const int nb = (lim + 63) >> 6;
        int5 tc = {0, c, 0, 0, 0};
        const uint64 th = v_u32_ld_tnsr_b(tc, thr);
        int5 ec = {c, 0, 0, 0, 0};
        const int want = s_i32_ld_g(gen_addr(ec, exact));
        // An INVISIBLE tile: every visible row of the buffer is a running row (lim == tile_row0, so the group
        // added no key of its own) and the column lists at most KS entries (lim <= KS), so ix_thr leaves its
        // threshold at 1 and every visible row is strictly above it — the selection is exactly rows
        // 0 .. lim - 1 in row order, i.e. the running set copied through. That is the whole of a padded tile
        // (the chain runs ceil(tiles/G) recipes of G tiles for the MAXIMUM context, so most tiles of a short
        // prefill are past the visible prefix) and it costs one load + three stores per 64 rows instead of the
        // per-entry selection below. Bit-exact: the general path lists the strict class (key > thr) ascending
        // then the tie class (key == thr) ascending, so the identity needs every visible row strict and none a
        // tie — and that is exactly `tot == lim`, where tot is ix_bcnt's packed (strict | ties << 16) count
        // summed over the column's blocks (a scalar load, so the test is a scalar compare).
        int tot = 0;
        for (int q = 0; q < (nb + 63) >> 6; q++) {
            int5 gc = {q, c, 0, 0, 0};
            tot += s_i32_ld_g(gen_addr(gc, gsum));
        }
        // An INVISIBLE tile lists the running set copied through (every visible row strict, none a tie). Node B
        // does that copy -- this node has no position/score output -- so all this needs is to skip the work.
        if (lim > 0 && lim <= tile_row0 && lim <= KS && tot == lim)
            continue;
        for (int m = off[0]; m < end[0]; m++) {
            const int b0 = m * nb / NM, b1 = (m + 1) * nb / NM;
            // list offsets: strict keys before b0, all strict keys, ties before b0, all ties
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
            {  // the slot record: the fill's count, and the tie cut as a position
                const int all = st + tt <= KS ? 0x7FFFFFFF : st >= KS ? -1 : 0;
                int64 rec = 0;
                rec = v_i32_sel_eq_i32_b(lane, (int64)0, (int64)(lim > 0 ? 0 : -1), rec);
                rec = v_i32_sel_eq_i32_b(lane, (int64)1, (int64)(H * c), rec);
                rec = v_i32_sel_eq_i32_b(lane, (int64)2, (int64)KS, rec);
                rec = v_i32_sel_eq_i32_b(lane, (int64)4, (int64)want, rec);
                rec = v_i32_sel_eq_i32_b(lane, (int64)5, (int64)c, rec);
                rec = v_i32_sel_eq_i32_b(lane, (int64)7, (int64)all, rec);
                int5 rc = {0, c, 0, 0, 0};
                v_i32_st_tnsr_partial(rc, slots, rec, 7, 0);
            }
            int64 tw = (int64)BIAS;  // the largest tie listed (keyed row)
            for (int pass = 0; pass < 2; pass++) {
                int g = pass == 0 ? s0 : st + t0;  // the next entry's list index
                for (int b = b0; b < b1 && g < KS; b++) {
                    int5 cc = {b, c, 0, 0, 0};
                    const int x = s_i32_ld_g(gen_addr(cc, cnt));
                    const int n = pass == 0 ? (x & 0xFFFF) : (x >> 16);
                    if (n == 0)
                        continue;
                    int5 sc = {b * 64, c, 0, 0, 0};
                    const int64 svec = as_int64(v_f32_ld_tnsr_b(sc, score));
                    const uint64 k = mkey(as_float64(svec));
                    const int64 pos = lane + (int64)(b * 64);
                    const int64 key = pos ^ (int64)BIAS;
                    // the entry's position: an earlier group's, or this group's from cpos
#ifdef CPOS
                    const int64 pv = b * 64 >= tile_row0
                        ? v_i32_ld_tnsr_b((int5){b * 64 - tile_row0, c, 0, 0, 0}, cpos)
                        : v_i32_ld_tnsr_b(sc, pos_old);
#else
                    const int64 pv = b * 64 >= tile_row0
                        ? lane + (int64)(base + b * 64 - tile_row0)
                        : v_i32_ld_tnsr_b(sc, pos_old);
#endif
                    const uint64 none = (uint64)0x7FFFFFFF, ukey = as_uint64(key);
                    int64 lo = as_int64(pass == 0 ? v_u32_sel_less_u32_b(th, k, ukey, none)
                                                  : v_u32_sel_eq_u32_b(k, th, ukey, none));
                    lo = v_i32_sel_less_i32_b(pos, (int64)lim, lo, (int64)0x7FFFFFFF);
                    int64 hi = v_i32_sel_eq_i32_b(lo, (int64)0x7FFFFFFF, (int64)BIAS, lo);
                    const int take = n < KS - g ? n : KS - g;
                    int a = g, z = g + take - 1;
                    if (take == n)
                        for (; a < z; a++, z--) {
                            int64 l, h;
                            RED(lo, l, v_i32_min_b);
                            RED(hi, h, v_i32_max_b);
                            STORE_ENTRY(a, l)
                            STORE_ENTRY(z, h)
                            if (pass == 1)
                                tw = v_i32_max_b(tw, h);
                            lo = v_i32_sel_eq_i32_b(lo, l, (int64)0x7FFFFFFF, lo);
                            lo = v_i32_sel_eq_i32_b(lo, h, (int64)0x7FFFFFFF, lo);
                            hi = v_i32_sel_eq_i32_b(hi, h, (int64)BIAS, hi);
                            hi = v_i32_sel_eq_i32_b(hi, l, (int64)BIAS, hi);
                        }
                    for (; a <= z; a++) {
                        int64 l;
                        RED(lo, l, v_i32_min_b);
                        STORE_ENTRY(a, l)
                        if (pass == 1)
                            tw = v_i32_max_b(tw, l);
                        lo = v_i32_sel_eq_i32_b(lo, l, (int64)0x7FFFFFFF, lo);
                    }
                    g += take;
                }
            }
            {  // the tie cut as a position (the dense attention's mask reads it)
                const int all = st + tt <= KS ? 0x7FFFFFFF : st >= KS ? -1 : 0;
                int64 cv = (int64)all;
                if (all == 0) {
                    int64 twr;
                    RED(tw, twr, v_i32_max_b);
                    const int64 row = DECODE(twr);
#ifdef CPOS
                    cv = row;  // keyed by position: the largest tie's position directly
#else
                    cv = v_i32_sel_less_i32_b(row, (int64)tile_row0, (int64)-1,
                                              row + (int64)(base - tile_row0));
#endif
                }
                int5 rc = {7, c, 0, 0, 0};
                v_i32_st_tnsr_partial(rc, slots, cv, 0, 0);
            }
        }
    }
}
