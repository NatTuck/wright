// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright 2026 Nat Tuck
// Splice image rows into the prompt embedding, in place: for the columns in
// [first, first + count), x[c, t] = img[c, t - first]. Every other column is left alone.
//
// The vision tower's features replace the embedding of the image_pad tokens, and they have
// to be in place before the input norm runs, so this sits between the embedding cast and
// add_rms8 (a host-side write into the carry afterwards is too late - the norm has already
// been computed from the un-spliced rows).
//
// Out of place, because Synapse will not take one tensor as both the input and the output
// of a node (synNodeCreate returns status 26), so an in-place version cannot exist and the
// text path cannot be skipped by returning early: y has to be written either way. That is
// why the embed graph only carries this node when the request actually has an image.
//
// tensors: 0 x (n, T) f32 | 1 img (n, T) f32 | 2 span (2,) i32 [first, count] |
//          3 y (n, T) f32. index space: [T].
#include "common.h"

void main(tensor x, tensor img, tensor span, tensor y)
{
    const int first = meta_i32(span, 0);
    const int count = meta_i32(span, 1);
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const int n = get_dim_size(y, 0);
    for (int t = off[0]; t < end[0]; t++) {
        // the image columns take their row from img, everything else is a straight copy
        const int take = count > 0 && t >= first && t < first + count;
        const int src_t = take ? t - first : t;
        for (int i = 0; i < n; i += 64) {
            int5 c = {i, t, 0, 0, 0}, sc = {i, src_t, 0, 0, 0};
            v_f32_st_tnsr(c, y, take ? v_f32_ld_tnsr_b(sc, img) : v_f32_ld_tnsr_b(sc, x));
        }
    }
}
