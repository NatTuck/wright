// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// Broadcast a 2-D slab along a new trailing axis: out (D, C*S) holds in (D, C) at every slot's column block,
// i.e. out[:, c + C*s] = in[:, c].
//
// The tiled index chain gathers, dequantizes and scores an index tile ONCE for all of the step's slots (every
// slot of a tile resolves to the same pages -- ds_ix_gslot's key0 and first-page lanes are per-tile), and
// Habana's batch_gemm will not broadcast a batch-1 operand, so the single tile is replicated here. The write is
// C*D*S bytes per tile, the same volume the per-slot dequant already wrote, so the saving is upstream: the 16x
// gather and the 16x read of the fp4 codes.
//
// Both sides are declared 2-D on purpose. A 256-byte vector written to a 3-D tensor crosses its innermost
// extent (S), and the descriptor wraps: the shapes still line up and the values are silently wrong. Flattened,
// a vector is two whole columns of a 128-wide slab and never crosses anything. The caller views the result as
// (D, C, S) for the GEMM.
//
// guid: rg_ds_bcast3_f32 (the glue derives it as rg_<file stem>_f32).
// tensors: 0 in (D, C) u8 | 1 out (D, C*S) u8.  params: C.  index space: [1, C, S]; one member copies one
// whole column (D elements) of one slot.
#include "common.h"

void main(tensor in, tensor out, int C)
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    for (int s = off[2]; s < end[2]; s++)
        for (int c = off[1]; c < end[1]; c++)
            for (int d = off[0]; d < end[0]; d++) {
                int5 ic = {0, c, 0, 0, 0};
                int5 oc = {0, c + C * s, 0, 0, 0};
                // a column is one vector's low half: the full 256-byte store would cross into the next column
                v_u8_st_tnsr_partial(oc, out, v_u8_ld_tnsr_b(ic, in), 127, 0);
            }
}
