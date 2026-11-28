// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// Concatenate the tile stream's key blocks along the key axis: out (D, N*B, Q) holds block i's ks_i (D, B, Q) at
// key rows i*B .. (i+1)*B.
//
// The bounded online-attention stream keeps the block recurrence only to bound the workspace: it gathers and
// scores one block at a time, so block i+1's gather waits on block i's acc and the MME sees ten small GEMMs
// (M = B = 128 keys, N = nh = 8, batch = Q) instead of one. The keys of the whole 640-key set fit easily at
// Q = 256 (5 MB of scores), so the blocks can be concatenated and scored once.
//
// Five gathers cannot write one tensor -- Synapse rejects a tensor with more than one producer -- and a block's
// slice of (D, N*B, Q) is strided in Q, so this is the join: one node, five inputs, one output.
//
// guid: rg_ds_cat_blk_f32 (the glue derives it as rg_<file stem>_f32).
// tensors: 0..4 ks_i (D, B, Q) bf16 | 5 out (D, N*B, Q) bf16.  index space: [1, B, Q]; one member copies one
// key of one column -- a bfloat128 is D = 128 lanes, so the whole D fits one vector.
#include "common.h"

void main(tensor k0, tensor k1, tensor k2, tensor k3, tensor k4, tensor out)
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const int B = get_dim_size(k0, 1);
    for (int q = off[2]; q < end[2]; q++)
        for (int b = off[1]; b < end[1]; b++)
            // a bfloat128 is 128 BYTES = 64 bf16 values, so a key of D = 128 is two of them -- kv_store's COPY
            // macro does exactly this (load at dim0 0 and 64, store at 0 and 64)
            for (int h2 = 0; h2 < 2; h2++) {
                int5 sc = {64 * h2, b, q, 0, 0};
                const bfloat128 v0 = v_bf16_ld_tnsr_b(sc, k0);
                const bfloat128 v1 = v_bf16_ld_tnsr_b(sc, k1);
                const bfloat128 v2 = v_bf16_ld_tnsr_b(sc, k2);
                const bfloat128 v3 = v_bf16_ld_tnsr_b(sc, k3);
                const bfloat128 v4 = v_bf16_ld_tnsr_b(sc, k4);
                int5 oc = {64 * h2, b, q, 0, 0};
                v_bf16_st_tnsr(oc, out, v0);
                oc[1] = B + b;
                v_bf16_st_tnsr(oc, out, v1);
                oc[1] = 2 * B + b;
                v_bf16_st_tnsr(oc, out, v2);
                oc[1] = 3 * B + b;
                v_bf16_st_tnsr(oc, out, v3);
                oc[1] = 4 * B + b;
                v_bf16_st_tnsr(oc, out, v4);
            }
}
