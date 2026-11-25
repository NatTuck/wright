// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// DeepSeek-V4.1 WINDOW-latent BLOCK gather (D=512, fp8 -> bf16): the per-column sliding-window variant
// of mla_gather mode 0, bounded to one key-block of B=64 window keys for a query-tile of Q columns.
// The unified 640-key attention stream runs blocks 0-1 = window (this kernel, 128 keys/column) and
// blocks 2-9 = listed (ds_gather_list_blk_f4). tile_off is the STEP-COLUMN offset co (the SAME runtime
// table the listed gather uses): step-column s = tile_off[0] + q in [0, T). meta[0] = pos0, the step's
// first ABSOLUTE position, so column s's absolute position is pos0 + s. The window is POSITIONAL: window
// key k (local 0..B-1) of column s is absolute position pos = pos0 + s - 127 + b0*B + k (the 128-key
// sliding window [c-127, c] INCLUDING the column's own position c = pos0+s, per NVIDIA
// get_window_topk_idxs: offset -(window_size-1) = -127; b0=0,1 cover it). pos < 0 (or >= the column's
// visible count klen, or a non-resident page) is an INVALID key: ks_blk = 0 and idx = -1. The dequant is BIT-EXACT
// mla_gather mode 0: the GLOBAL absolute-position window page table lat_pages[pos >> 7] gives the first
// latent arena row lat0 of the 129-row page (128 fp8 token rows + 1 f32-scale row) holding positions
// [128*(pos>>7) .. +127] for THIS layer; the token's 512 fp8 latent bytes are rows lat0+(pos&127) at byte
// offsets 0 and 256, and its scale is the f32 at row lat0+128, byte 4*(pos&127), cast to bf16. lat_pages
// points at the prefix pages for pos < pos0; klen is read from the step-column slots table (in bounds).
// tensors: 0 arena (512, PR, NB) u8  1 lat_pages (WP,) i32 [lat_pages[pos>>7] = first latent row, -1=zeros]
//          2 tile_off (1,) i32 [step-column offset co]  3 slots (8, T) i32 [field 4 = visible key count
//          klen of step-column s]  4 meta (4,) i32 [meta[0] = pos0]  5 after (unread)
//        | 6 ks_blk (D, B, Q) bf16  7 idx (B, Q) i32 [source key index = absolute pos, -1 = invalid]
// params: b0 (window key-block index 0/1), B (keys/block=64), Q (tile columns). index space: [1, B/32, Q]
// (32 keys per member, like mla_gather). D = 512, rope = false. guid: rg_ds_gather_win_blk_f32.
#include "common.h"

#define VG_KEYS 32
#define ROW(r) (r) & pm, (r) >> psh

void main(tensor arena, tensor lat_pages, tensor tile_off, tensor slots, tensor meta, tensor after,
          tensor ks_blk, tensor idx, int b0, int B, int Q)
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const int PR = get_dim_size(arena, 1);
    int psh = 0;
    while ((1 << psh) < PR)
        psh++;
    const int pm = PR - 1;
    const bfloat128 zero = 0.0f;
    scalar_cache_reset();
    int5 tc = {0, 0, 0, 0, 0};
    const int co = s_i32_ld_g(gen_addr(tc, tile_off));   // the tile's first step-column
    const int pos0 = meta_i32(meta, 0);                  // the step's first absolute position
    const int kbase = b0 * B;                            // first local window key of this block
    for (int q = off[2]; q < end[2]; q++) {
        const int s = co + q;                            // the step-column (slots/list index, in [0,T))
        int5 kc = {4, s, 0, 0, 0};
        const int klen = s_i32_ld_g(gen_addr(kc, slots));  // the column's visible key count
        for (int j = off[1]; j < end[1]; j++) {
            const int k0 = j * VG_KEYS;                  // first local key of this member
            for (int u = 0; u < VG_KEYS; u++) {
                const int k = k0 + u;                    // local key within the block
                const int pos = pos0 + s - 127 + kbase + k;  // absolute window key position (WINDOW=128,
                // offset -(WINDOW-1)=-127: the window [c-127, c] INCLUDES the column's own position c)
                bfloat128 v0 = zero, v1 = zero, v2 = zero, v3 = zero;
                int ixv = -1;
                if (pos >= 0 && pos < klen) {
                    int5 pc = {pos >> 7, 0, 0, 0, 0};
                    const int lat0 = s_i32_ld_g(gen_addr(pc, lat_pages));
                    if (lat0 >= 0) {
                        const int r = pos & 127;
                        int5 sc = {4 * r, ROW(lat0 + 128), 0, 0};
                        const bf16 sv = (bf16)s_f32_ld_g((__global float*)gen_addr(sc, arena));
                        int5 ac = {0, ROW(lat0 + r), 0, 0};
                        const bfloat256 f = convert_minifloat256_to_bfloat256(v_f8_ld_tnsr_b(ac, arena), SW_LINEAR);
                        ac[0] = 256;
                        const bfloat256 g = convert_minifloat256_to_bfloat256(v_f8_ld_tnsr_b(ac, arena), SW_LINEAR);
                        v0 = f.v1 * sv;
                        v1 = f.v2 * sv;
                        v2 = g.v1 * sv;
                        v3 = g.v2 * sv;
                        ixv = pos;
                    }
                }
                int5 oc = {0, k, q, 0, 0};
                v_bf16_st_tnsr(oc, ks_blk, v0);
                oc[0] = 128;
                v_bf16_st_tnsr(oc, ks_blk, v1);
                oc[0] = 256;
                v_bf16_st_tnsr(oc, ks_blk, v2);
                oc[0] = 384;
                v_bf16_st_tnsr(oc, ks_blk, v3);
                int5 ic = {k, q, 0, 0, 0};
                v_i32_st_tnsr_partial(ic, idx, (int64)ixv, 0, 0);
            }
        }
    }
}