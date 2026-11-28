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
// TWO LAYOUT FACTS, both of which cost a debugging cycle:
//
//  1. Device tensors are column-major with dim 0 fastest. out (D, C*S) therefore has flat memory [s][c][d] with
//     d fastest -- C-order shape (s, c, d). Reading a d2h dump as (d, s, c) or (d, c, s) is a fixed misread of
//     a correct buffer.
//  2. A vector load needs a VECTOR-GRANULAR dim 0. A 256-byte vector at {0, c} on a (128, C) u8 tensor does not
//     address by c at all -- dim 0 is half a vector, so the descriptor collapses and EVERY load returns column
//     0's data. So both sides are declared 2-D and one member copies a full 256-byte vector, which is two whole
//     columns of the 128-wide slab at the vector-aligned flat offset 2c*128. This is x8_copy's shape
//     ({256*nt, ...}) and it is what the hardware wants; do not "optimise" it back to a partial store.
//
// guid: rg_ds_bcast3_f32 (the glue derives it as rg_<file stem>_f32).
// tensors: 0 in (2D, C/2) u8 | 1 out (2D, C/2, S) u8 -- the same memory as (D, C) and (D, C, S), viewed so dim 0
// is a whole 256-byte vector (dim 0 must be vector-granular: see fact 2 above). One member copies one vector,
// which is two columns of the 128-wide slab, of one slot. index space: [1, C / 2, S].
#include "common.h"

void main(tensor in, tensor out)
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    for (int s = off[2]; s < end[2]; s++)
        for (int c = off[1]; c < end[1]; c++)
            for (int d = off[0]; d < end[0]; d++) {
                int5 ic = {0, c, 0, 0, 0};
                int5 oc = {0, c, s, 0, 0};
                v_u8_st_tnsr(oc, out, v_u8_ld_tnsr_b(ic, in));
            }
}
