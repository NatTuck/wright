// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// ix_cgather_f4 with CPOS_LOCAL: the unrolled sparse index chain's candidate gather writes each tile's entry
// positions to a tile-local cpos slab (rows = the tile's 2048 keys) instead of the full (kc, T) array — the
// candidate-block offset still comes from the slot's key0 (global), only the cpos row is taken mod the tile size.
#define CPOS_LOCAL
#include "ix_cgather_f4.c"