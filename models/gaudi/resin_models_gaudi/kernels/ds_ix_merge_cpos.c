// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// ds_ix_merge with CPOS: the sparse (candidate-entry) index chain's merge. The tile's entries are the column's
// candidate blocks' keys, whose positions are NOT sequential — they come from the candidate gather's cpos slab
// (rows, T). A tile row's listed position is cpos[row - tile_row0] instead of base + row - tile_row0; the running
// set's positions still come from pos_old. The list order (strict-by-row then ties-by-row) is unchanged, so the
// selection matches the compact-score ix_list/ix_list_map path bit-for-bit.
#define CPOS
#include "ds_ix_merge.c"