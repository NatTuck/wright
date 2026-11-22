// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// ix_list over compact entries (DeepSeek-V4.1 candidate blocks, sparse scoring): each listed entry is written as
// its position, the lane of cpos (KC, T) (ix_cgather) at the entry; tensor 5 is cpos and the outputs follow it.
// An entry is keyed as that position (see ix_list.c): a ratio-1 layer's positions reach 524287 at a 512k context,
// so the compact row index, which only ordered the key, is not packed beside them.
#define MAP
#include "ix_list.c"
