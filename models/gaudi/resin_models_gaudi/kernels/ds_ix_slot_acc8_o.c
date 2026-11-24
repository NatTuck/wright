// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// ds_ix_slot_acc8 with the score slab as a pure OUTPUT (SCORE_OUT): the unrolled index chain's acc writes its
// tile's scores to a graph-local tensor the assemble reads, so the write is a normal single-producer dataflow
// edge (no in-place input, no read-modify-write section). Same numerics as ds_ix_slot_acc8.
#define RATIO
#define SCALED
#define SCORE_OUT
#include "ix_slot_acc.c"