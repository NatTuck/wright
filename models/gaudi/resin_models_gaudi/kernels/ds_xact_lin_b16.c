// SPDX-License-Identifier: AGPL-3.0-only
// Copyright Aaron Botsis
// ds_xact_lin over the BF16 w13 product (Y16: the reference's own expert output dtype, half the bytes): hq in
// natural order, power-of-two scales, the tail rows in y.
#define LIN
#define POW2
#define Y16
#include "ds_xact.c"
