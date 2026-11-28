// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// Broadcast a 2-D slab along a new trailing axis: out (D, C, S) = in (D, C) for every s.
//
// The tiled index chain gathers, dequantizes and scores an index tile ONCE for all of the step's slots (every
// slot of a tile resolves to the same pages -- ds_ix_gslot's key0 and first-page lanes are per-tile). The
// batched GEMM wants the key operand (D, C, S) and Habana's batch_gemm does not broadcast a batch-1 operand,
// so the single tile is replicated here. The write is C*D*S bytes per tile, the same volume the per-slot
// dequant already wrote, so the saving is upstream: the 16x gather and the 16x read of the fp4 codes.
//
// guid: rg_ds_bcast3_f32 (the glue derives it as rg_<file stem>_f32).
// tensors: 0 in (D, C) u8 | 1 out (D, C, S) u8.  index space: [1, C / 2, S]; one member is a 256-byte vector,
// which is TWO whole columns of a 128-wide slab (D is 128, so a vector never spans a partial column).
#include "common.h"

void main(tensor in, tensor out)
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    for (int s = off[2]; s < end[2]; s++)
        for (int c = off[1]; c < end[1]; c++)
            for (int d = off[0]; d < end[0]; d++) {
                int5 ic = {0, 2 * c, 0, 0, 0};
                int5 oc = {0, 2 * c, s, 0, 0};
                v_u8_st_tnsr(oc, out, v_u8_ld_tnsr_b(ic, in));
            }
}
