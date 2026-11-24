// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// ds_ix_qacc with the score slab as a pure OUTPUT (SCORE_OUT): the unrolled index chain's prefill acc writes its
// tile's scores to a graph-local tensor the assemble reads (single-producer dataflow, no in-place input).
#define SCORE_OUT
#include "ds_ix_qacc.c"