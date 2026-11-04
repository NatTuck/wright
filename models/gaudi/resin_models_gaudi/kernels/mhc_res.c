// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Required Notice: Copyright Aaron Botsis
// mHC residual update after a sub-layer (mhc_post without the rms, spread over the hidden dim):
//   Xo_j = post_j y + sum_i comb[i][j] X_i        (y summed over NR rank slices when NR > 1)
// The next site's 1 / rms(Xo) comes from mhc_mix.
// tensors: 0 X (4H, T) f32  1 y (H, T, NR) f32  2 coef (T, 24) f32 | 3 Xo (4H, T) f32.  index space: [H/256, T]
#include "common.h"

#define NS 4

void main(tensor X, tensor y, tensor coef, tensor Xo)
{
    const int5 off = get_index_space_offset();
    const int5 end = off + get_index_space_size();
    const int H = get_dim_size(y, 0), NR = get_dim_size(y, 2);
    scalar_cache_reset();
    for (int t = off[1]; t < end[1]; t++) {
        float post[NS], cm[16];
        for (int s = 0; s < NS; s++) {
            int5 c = {t, 4 + s, 0, 0, 0};
            post[s] = s_f32_ld_g(gen_addr(c, coef));
        }
        for (int m = 0; m < 16; m++) {
            int5 c = {t, 8 + m, 0, 0, 0};
            cm[m] = s_f32_ld_g(gen_addr(c, coef));
        }
        for (int b = off[0]; b < end[0]; b++)
            for (int i = 256 * b; i < 256 * b + 256 && i < H; i += 64) {
                int5 yc = {i, t, 0, 0, 0};
                float64 yv = v_f32_ld_tnsr_b(yc, y);
                for (yc[2] = 1; yc[2] < NR; yc[2]++) yv += v_f32_ld_tnsr_b(yc, y);
                float64 xs[NS];
                #pragma unroll
                for (int s = 0; s < NS; s++) {
                    int5 c = {s * H + i, t, 0, 0, 0};
                    xs[s] = v_f32_ld_tnsr_b(c, X);
                }
                #pragma unroll
                for (int j = 0; j < NS; j++) {
                    float64 o = yv * post[j];
                    #pragma unroll
                    for (int s = 0; s < NS; s++) o = v_f32_mac_b(xs[s], cm[4 * s + j], o);
                    int5 c = {j * H + i, t, 0, 0, 0};
                    v_f32_st_tnsr(c, Xo, o);
                }
            }
    }
}
