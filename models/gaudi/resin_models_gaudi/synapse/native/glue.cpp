// Graph-compiler glue for Resin's custom TPC kernels (a GC_KERNEL_PATH plugin).
#include <tpc_kernel_lib_interface.h>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include "kernels_elf.h"   // generated: rg_elf_<name>[] and the kElfs table

using namespace tpc_lib_api;

typedef bool (*GeomFn)(const HabanaKernelParams*, uint64_t is[5], uint32_t* rank);

static uint64_t sz(const HabanaKernelParams* p, bool out, int t, int d) {
    return (out ? p->outputTensors : p->inputTensors)[t].geometry.maxSizes[d];
}

// attn_prep
static bool geom_attn_prep(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, false, 0, 1); *rank = 1; return true;
}
// attn_softmax
static bool geom_rows1(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, false, 0, 1); *rank = 1; return true;
}
// attn_gate, add_rms, quant_fp8, requant_fp8
static bool geom_out_tokens(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, true, 0, 1); *rank = 1; return true;
}
// elementwise kernels
static bool geom_elem(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    uint64_t c = sz(p, true, 0, 0);
    is[0] = (c + 127) / 128; is[1] = sz(p, true, 0, 1); *rank = 2; return true;
}
static bool geom_one(const HabanaKernelParams*, uint64_t is[5], uint32_t* rank) {
    is[0] = 1; *rank = 1; return true;
}
// gdn_fused
static bool geom_gdn_fused(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, true, p->outputTensorNr - 1, 2); *rank = 1; return true;
}
// gdn_decode
static bool geom_gdn_decode(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, true, 0, 0) / 128; is[1] = sz(p, true, 0, 1); *rank = 2; return true;
}
// sampling: samp_scan, samp_mass, samp_select, samp_full, samp_draw
static bool geom_samp_scan(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, true, 0, 2); is[1] = sz(p, true, 0, 3); *rank = 2; return true;
}
static bool geom_samp_mass(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, true, 0, 1); is[1] = sz(p, true, 0, 2); *rank = 2; return true;
}
static bool geom_samp_rows_in(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, false, 0, 1); *rank = 1; return true;
}
static bool geom_samp_rows_out(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, true, 0, 1); *rank = 1; return true;
}
// score_gather
static bool geom_score_gather(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = (sz(p, true, 0, 0) + 63) / 64; *rank = 1; return true;
}
// hidden_store
static bool geom_hidden_store(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, false, 2, 1); *rank = 1; return true;
}
// gdn_verify, gdn_commit
static bool geom_gdn_verify(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, true, 1, 2); is[1] = sz(p, true, 1, 4); *rank = 2; return true;
}
static bool geom_gdn_commit(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, false, 1, 2); is[1] = sz(p, false, 1, 4); *rank = 2; return true;
}
// argmax1, argmax2
static uint64_t sz1(const HabanaKernelParams* p, bool out, int t, int d) {
    const uint64_t n = sz(p, out, t, d); return n ? n : 1;
}
static bool geom_argmax1(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, true, 0, 1); is[1] = sz1(p, true, 0, 2); *rank = 2; return true;
}
static bool geom_argmax2(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz1(p, true, 0, 1); *rank = 1; return true;
}
// attn_kvcat
static bool geom_kvcat(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = (sz(p, true, 0, 1) + 63) / 64; *rank = 1; return true;
}
// attn_dec_p, attn_dec_comb
static bool geom_dec_p(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, true, 0, 2); *rank = 1; return true;
}
static bool geom_dec_comb(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, false, p->inputTensorNr - 1, 1); *rank = 1; return true;
}
// chunked GDN
static bool geom_ck_out0(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, true, 0, 2); is[1] = sz(p, true, 0, 3); *rank = 2; return true;
}
static bool geom_ck_in0(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, false, 0, 2); is[1] = sz(p, false, 0, 3); *rank = 2; return true;
}
// gdn_ck_qk
static bool geom_ck_qk(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, true, 0, 2); is[1] = sz(p, true, 0, 3); *rank = 2; return true;
}
static bool geom_ck_step(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, true, 0, 2); is[1] = 16; *rank = 2; return true;
}
// page_store
static bool geom_page_store(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, false, 2, 1); *rank = 1; return true;
}

// kv_gather
static bool geom_kv_gather(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, true, 0, 0) / 256; is[1] = sz(p, true, 0, 1) / 32; is[2] = sz(p, true, 0, 2);
    *rank = 3; return true;
}

// state_load, state_store
// (R rows in members of 16; the kernels stop at R, which need not be a multiple of 16)
static bool geom_state_load(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = (sz(p, true, 0, 1) + 15) / 16; is[1] = sz(p, true, 0, 2); *rank = 2; return true;
}
static bool geom_state_store(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = (sz(p, false, 2, 1) + 15) / 16; is[1] = sz(p, false, 2, 2); *rank = 2; return true;
}

// slot attention: attn_slot_q, attn_slot_p, attn_slot_acc, attn_acc_init, kv_store
static uint64_t blocks64(uint64_t n) { return (n + 63) / 64; }
static bool geom_slot_q(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = 1; is[1] = blocks64(sz(p, true, 0, 1)); is[2] = sz(p, true, 0, 2); *rank = 3; return true;
}
static bool geom_slot_p(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = blocks64(sz(p, false, 0, 1)); is[1] = sz(p, false, 0, 2); *rank = 2; return true;
}
// ds_slot_pw1, ds_slot_p1: one member per column
static bool geom_slot_p1(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, false, 0, 1); is[1] = sz(p, false, 0, 2); *rank = 2; return true;
}
static bool geom_slot_acc(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = blocks64(sz(p, false, 4, 1)); is[1] = sz(p, false, 4, 2); *rank = 2; return true;
}
// attn_slot_fin: a member per 64 output columns of a group (out (256, W, G))
static bool geom_slot_fin(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = blocks64(sz(p, true, 0, 1)); is[1] = sz(p, true, 0, 2); *rank = 2; return true;
}
// attn_slot_mask, attn_sdpa_acc
static bool geom_slot_mask(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = blocks64(sz(p, true, 0, 1)); is[1] = sz(p, true, 0, 3); *rank = 2; return true;
}
// attn_sdpa_acc: a member per 64 query tokens of the slots, head and group
static bool geom_sdpa_acc(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = blocks64(sz(p, false, 0, 1)); is[1] = sz(p, false, 0, 2); is[2] = sz(p, false, 4, 2);
    *rank = 3; return true;
}
static bool geom_acc_init(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = blocks64(sz(p, false, 0, 1)); is[1] = sz(p, false, 0, 2); *rank = 2; return true;
}
// mla_store, mla_slot_out
static bool geom_mla_store(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, false, 1, 1); *rank = 1; return true;
}
static bool geom_slot_out(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = (sz(p, true, 0, 1) + 63) / 64; is[1] = sz(p, true, 0, 2); *rank = 2; return true;
}
static bool geom_slot_out1(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, true, 0, 1); is[1] = sz(p, true, 0, 2); *rank = 2; return true;
}
// ds_slot_ml8: 8 columns per member
static bool geom_slot_out8(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = (sz(p, true, 0, 1) + 7) / 8; is[1] = sz(p, true, 0, 2); *rank = 2; return true;
}
static bool geom_kv_store(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, false, 1, 1); *rank = 1; return true;
}

// Mistral-Large-4. lq_* (GEMVs, kernels/lq)
static bool geom_param0(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    const int n = p->nodeParams.nodeParams && p->nodeParams.nodeParamsSize >= 4 ? *(const int*)p->nodeParams.nodeParams : 1;
    is[0] = n > 0 ? n : 1; *rank = 1; return true;
}
// ml4_rope, ml4_perm
static bool geom_out_d12(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, true, 0, 1) * sz(p, true, 0, 2); *rank = 1; return true;
}
static bool geom_perm(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, true, 0, 1); is[1] = sz(p, true, 0, 2); *rank = 2; return true;
}
// ml4_ix_acc, ix_bcnt, ix_mask
static bool geom_elem64(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = (sz(p, true, 0, 0) + 63) / 64; is[1] = sz(p, true, 0, 1); *rank = 2; return true;
}
// ix_bmax, ix_cmask: 512 entries of a column per member (input 0's dims)
static bool geom_k512(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = (sz(p, false, 0, 0) + 511) / 512; is[1] = sz(p, false, 0, 1); *rank = 2; return true;
}
// ds_ix_bmax_nat: input 0 is the (K/8, 8, T) reshape of the tile scores, so K = dim0*8 and the column is dim2
static bool geom_bmax_nat(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = (sz(p, false, 0, 0) * 8 + 511) / 512; is[1] = sz(p, false, 0, 2); *rank = 2; return true;
}
// ix_rcnt: (64, M, T) partial counts, one member per (slice, column)
static bool geom_rcnt(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, true, 0, 1); is[1] = sz(p, true, 0, 2); *rank = 2; return true;
}
// ix_cgather: 64 slab entries (8 candidate blocks, 32 rows of the (256, 1024, S) slab) of a slot per member
static bool geom_cgather(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = 1; is[1] = sz(p, true, 0, 1) / 32; is[2] = sz(p, true, 0, 2); *rank = 3; return true;
}
// ix_cgather_f4: 64 keys (8 blocks) of the (64, 2048, S) code staging per member
static bool geom_cgather_f4(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = 1; is[1] = sz(p, true, 0, 1) / 64; is[2] = sz(p, true, 0, 2); *rank = 3; return true;
}
// ix_bgather: 64 listed blocks of a column per member (input 1 (NBL, T))
static bool geom_bgather(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, false, 1, 0) / 64; is[1] = sz(p, false, 1, 1); *rank = 2; return true;
}
// ds_ix_gather: 32 keys (16 rows of the (256, C / 2, S) slab) per member
static bool geom_pairs16(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = 1; is[1] = sz(p, true, 0, 1) / 16; is[2] = sz(p, true, 0, 2); *rank = 3; return true;
}
// ds_ix_gather_f4: 32 keys of the (64, C, S) code staging per member
static bool geom_ix_gather_f4(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = 1; is[1] = sz(p, true, 0, 1) / 32; is[2] = sz(p, true, 0, 2); *rank = 3; return true;
}
// ds_ix_gather_tile: the same shape and index space as ds_ix_gather_f4 (it gathers a whole tile, S = 1)
static bool geom_ix_gather_tile(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    return geom_ix_gather_f4(p, is, rank);
}
// ds_mk_add: 64 vocabulary entries of one rank's slice for one row per member
static bool geom_mk_add(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = (sz(p, false, 0, 0) + 63) / 64; is[1] = sz(p, true, 0, 1); is[2] = sz(p, false, 0, 2); *rank = 3; return true;
}
// ds_ix_gslot: one member per (merge group, slot of that group); params: S2 (lpg*slots), nblk, qb, heads,
// ratio, keys, run, tp, page. rec (8, groups*S2) -> [groups, S2]
static bool geom_ix_gslot(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    const int32_t* prm = (const int32_t*)p->nodeParams.nodeParams;
    const uint64_t S2 = prm[0] > 0 ? uint64_t(prm[0]) : 1;
    is[0] = (sz(p, true, 0, 1) + S2 - 1) / S2;
    is[1] = S2;
    *rank = 2;
    return true;
}
// ix_gvis: one member per column (its outputs are (T,) vectors)
static bool geom_ix_gvis(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, true, 0, 0);
    *rank = 1;
    return true;
}
// ix_list
static bool geom_ix_list(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    const int n = p->nodeParams.nodeParams && p->nodeParams.nodeParamsSize >= 4 ? *(const int*)p->nodeParams.nodeParams : 1;
    is[0] = n > 0 ? n : 1; is[1] = sz(p, true, 0, 1); *rank = 2; return true;
}
// ix_unmask
static bool geom_columns(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, true, 0, 1); *rank = 1; return true;
}
// ds_ix_qacc: 1024 keys of 8 query tokens of a slot per member (S (C, 32 Q, S))
static bool geom_ix_qacc(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, false, 0, 0) / 1024; is[1] = sz(p, false, 0, 1) / (32 * 8); is[2] = sz(p, false, 0, 2); *rank = 3;
    return true;
}
// ds_ix_qacc_tile: the same per-slot slab and index space as ds_ix_qacc (only its sk input is (C,))
static bool geom_ix_qacc_tile(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    return geom_ix_qacc(p, is, rank);
}
// ix_slot_acc
static bool geom_ix_slot_acc(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, false, 0, 0) / 64; is[1] = sz(p, false, 0, 2); *rank = 2; return true;
}
// lq_qdqp_*
static bool geom_qdqp(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    if (!p->nodeParams.nodeParams || p->nodeParams.nodeParamsSize < 16) return false;
    const int32_t* prm = (const int32_t*)p->nodeParams.nodeParams;
    const uint64_t K = uint64_t(prm[1]), M = uint64_t(prm[2]), SBk = uint64_t(prm[3]);
    if (SBk == 0 || K % SBk) return false;
    is[0] = (M + 127) / 128; is[1] = K / SBk; is[2] = sz(p, true, 0, 2); *rank = 3; return true;
}
// mla_gather
static bool geom_mla_gather(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = 1; is[1] = sz(p, true, 0, 1) / 32; is[2] = sz(p, true, 0, 2); *rank = 3; return true;
}

// GLM-5.3-Flash
static bool geom_tok1(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = 1; is[1] = sz(p, true, 0, 1); *rank = 2; return true;
}
static bool geom_blk64(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = (sz(p, true, 0, 0) + 63) / 64; *rank = 1; return true;
}
// mhc_coeft, mhc_coefp: one member per token (output 0 dim 0)
static bool geom_out0_d0(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, true, 0, 0); *rank = 1; return true;
}
// mhc_mixc: [row groups of 4, K chunks, token groups of 4] of mixp (T, 24, C)
static bool geom_mixc(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, true, 0, 1) / 4; is[1] = sz(p, true, 0, 2); is[2] = (sz(p, true, 0, 0) + 3) / 4;
    *rank = 3; return true;
}
static bool geom_nw24(const HabanaKernelParams*, uint64_t is[5], uint32_t* rank) {
    is[0] = 24; *rank = 1; return true;
}
// mhc_res
static bool geom_mhc_res(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    const uint64_t H = sz(p, false, 1, 0), T = sz(p, false, 1, 1);
    uint64_t nb = 48 / T;
    nb = nb < 1 ? 1 : (nb > H / 256 ? H / 256 : nb);
    const uint64_t blk = (H / nb + 63) / 64 * 64;
    is[0] = T; is[1] = (H + blk - 1) / blk; *rank = 2; return true;
}
// mhc_mixp: four rows of one chunk per member
static bool geom_mixp(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, true, 0, 0) / 4; is[1] = sz(p, true, 0, 1); *rank = 2; return true;
}
// mhc_mix
static bool geom_mhc_mix(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    // one member per output row (column j); the per-column full-store branch is correct for every T
    // (the T>2 partial-store branch replicated lane 0 into lanes 11-23).
    is[0] = sz(p, true, 0, 1); *rank = 1; return true;
}
// kda_ck_step
static bool geom_kda_step(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, true, 0, 2); *rank = 1; return true;
}
// x8_copy, x8_requant
static bool geom_x8_copy(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, true, 0, 0) / 256; is[1] = sz(p, true, 0, 2); is[2] = (sz(p, true, 0, 1) + 127) / 128; *rank = 3; return true;
}
static bool geom_x8_requant(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, true, 1, 0) / 256; is[1] = sz(p, true, 1, 1); *rank = 2; return true;
}
// ds_xact
static bool geom_xact(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    const int R = p->nodeParams.nodeParams && p->nodeParams.nodeParamsSize >= 4 ? *(const int*)p->nodeParams.nodeParams : 1;
    is[0] = sz(p, true, 0, 2); is[1] = R ? (sz(p, true, 0, 0) + 127) / 128 : 1; *rank = 2; return true;
}
// ds_q8
static bool geom_q8(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    const int mode = p->nodeParams.nodeParams && p->nodeParams.nodeParamsSize >= 4 ? *(const int*)p->nodeParams.nodeParams : 0;
    is[0] = (mode & 8) ? 1 : sz(p, true, 0, 0) / 256; is[1] = sz(p, true, 0, 1); *rank = 2; return true;
}
// ds_xact_pg: one page per member (input 3: pe (P,))
static bool geom_in3_d0(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, false, 3, 0); *rank = 1; return true;
}
// ds_xg: 4 pooled rows per member (input 2: gidx)
static bool geom_xg(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = (sz(p, false, 2, 0) + 3) / 4; *rank = 1; return true;
}
// ds_xqg, ds_xq
static bool geom_xqg(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = (sz(p, false, 1, 0) + 3) / 4; *rank = 1; return true;
}
static bool geom_xq(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, false, 0, 0) / 256; is[1] = sz(p, false, 0, 1); *rank = 2; return true;
}
// gx_mask
static bool geom_gx_mask(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, true, 0, 0) / 256; is[1] = sz(p, true, 0, 1); *rank = 2; return true;
}
// eg_gather: one hashed row of one token per member
static bool geom_eg_gather(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, false, 2, 0); is[1] = sz(p, false, 2, 1); *rank = 2; return true;
}
// eg_gate_g: one stream (of 4) of one token per member
static bool geom_streams(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = 4; is[1] = sz(p, true, 0, 1); *rank = 2; return true;
}
// ds_attn_comb: one column of one group per member
static bool geom_sink2(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, true, 0, 1); is[1] = sz(p, true, 0, 2); *rank = 2; return true;
}
// ds_cpool: one group per member (output 0's dim 1); ds_cstate: one column per member (input 0's dim 1)
static bool geom_in0_cols(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, false, 0, 1); *rank = 1; return true;
}
// ds_gather_list: 16 keys of a column per member
static bool geom_gather16(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = 1; is[1] = sz(p, true, 0, 1) / 16; is[2] = sz(p, true, 0, 2); *rank = 3; return true;
}
// mx4_dec, mx4_decs: 2048 elements of a row per member
static bool geom_mx4(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = (sz(p, true, 0, 0) + 2047) / 2048; is[1] = sz(p, true, 0, 1); *rank = 2; return true;
}
// ds_ldec_f4: one column (512 latent elements) of the staging slab per member
static bool geom_ldec_f4(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = 1; is[1] = sz(p, true, 0, 1); *rank = 2; return true;
}
// ds_attn_online / ds_attn_acc / ds_attn_online_fin: one member per (head, tile column); input 1 is the
// (nh, Q) running stats (max / sscale), so the index space is [nh, Q] regardless of the in/out acc_o split.
static bool geom_attn_stats(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, false, 1, 0); is[1] = sz(p, false, 1, 1); *rank = 2; return true;
}
// ds_attn_online_init: [nh, Q] from acc_o (D, nh, Q) — output 0 when the buffers are fresh outputs,
// input 0 when they are passed in place (mla_acc_init style).
static bool geom_attn_init(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    const bool out = p->outputTensorNr > 0;
    is[0] = sz(p, out, 0, 1); is[1] = sz(p, out, 0, 2); *rank = 2; return true;
}
static GeomFn geom_glm(const char* n) {
    auto is = [&](const char* pre) { return strncmp(n, pre, strlen(pre)) == 0; };
    if (is("mhc_mixp")) return geom_mixp;               // (prefix: before mhc_mix)
    if (is("mhc_mixc")) return geom_mixc;
    if (is("mhc_mix")) return geom_mhc_mix;
    if (is("mhc_res")) return geom_mhc_res;
    if (is("mhc_pre") || is("mhc_post")) return geom_tok1;
    if (is("mhc_coeft") || is("mhc_coefp")) return geom_out0_d0;   // (prefix: before mhc_coef)
    if (is("mhc_coef")) return geom_blk64;
    if (is("kda_ck_in") || is("kda_ck_solve")) return geom_ck_out0;
    if (is("kda_ck_out")) return geom_ck_in0;
    if (is("kda_ck_step")) return geom_kda_step;
    if (is("kda_decode")) return geom_gdn_decode;
    if (is("kda_verify")) return geom_gdn_verify;
    if (is("kda_commit")) return geom_gdn_commit;
    if (is("moe_cnt") || is("moe_asg")) return geom_nw24;
    if (is("moe_pages")) return geom_one;
    if (is("x8_copy")) return geom_x8_copy;
    if (is("x8_requant")) return geom_x8_requant;
    if (is("ds_xact_pg")) return geom_in3_d0;        // (prefix: before ds_xact)
    if (is("ds_xact")) return geom_xact;
    if (is("ds_xsum") || is("ds_tscale") || is("pack16")) return geom_elem;
    if (is("ds_xcomb")) return geom_out_tokens;
    if (is("ds_xqg")) return geom_xqg;                // (prefix: before ds_xq)
    if (is("ds_xq1")) return geom_columns;
    if (is("ds_xg")) return geom_xg;
    if (is("ds_xq")) return geom_xq;
    if (is("ds_q8")) return geom_q8;
    if (is("gx_pool")) return geom_mla_store;
    if (is("gx_gather")) return geom_mla_gather;
    if (is("gx_mask")) return geom_gx_mask;
    if (is("mx4_dec")) return geom_mx4;
    if (is("ds_slot_pw1") || is("ds_slot_p1")) return geom_slot_p1;   // (prefix: before ds_slot_pw)
    if (is("ds_slot_fused_dense")) return geom_slot_p;
    if (is("ds_slot_fused")) return geom_slot_p1;
    if (is("ds_slot_pw")) return geom_slot_p;
    if (is("ds_slot_pk")) return geom_slot_p;
    if (is("ds_slot_out")) return geom_slot_out;
    if (is("eg_gather")) return geom_eg_gather;
    if (is("eg_gate_g")) return geom_streams;
    if (is("ds_fq4")) return geom_elem64;
    if (is("ds_cstate")) return geom_in0_cols;
    if (is("ds_slot_ml1")) return geom_slot_out1;    // (prefix: before ds_slot_ml)
    if (is("ds_slot_ml8")) return geom_slot_out8;
    if (is("ds_slot_mld")) return geom_slot_out;
    if (is("ds_slot_ml")) return geom_slot_out;
    if (is("ds_attn_comb")) return geom_sink2;
    if (is("ds_attn_fused_dense")) return geom_sink2;
    if (is("ds_attn_fused")) return geom_sink2;
    if (is("ds_attn_online_init")) return geom_attn_init;   // (prefix: before ds_attn_online)
    if (is("ds_attn_online_fin")) return geom_attn_stats;   // (prefix: before ds_attn_online)
    if (is("ds_attn_online")) return geom_attn_stats;
    if (is("ds_attn_acc")) return geom_attn_stats;
    if (is("ds_ix_slot_acc")) return geom_ix_slot_acc;
    if (is("ds_ix_qacc_tile")) return geom_ix_qacc_tile;  // before the ds_ix_qacc prefix
    if (is("ds_ix_qacc")) return geom_ix_qacc;
    if (is("ds_gather_win_blk")) return geom_mla_gather;
    if (is("ds_gather_list_blk_f4")) return geom_gather16;  // (prefix: before ds_gather_list)
    if (is("ds_gather_list")) return geom_gather16;
    if (is("ds_ldec_f4")) return geom_ldec_f4;
    if (is("ds_gather_dense_f4")) return geom_mla_gather;
    if (is("ds_smean")) return geom_elem64;
    if (is("ds_argmax")) return geom_columns;
    if (is("ds_mk_add")) return geom_mk_add;
    if (is("ds_ix_merge")) return geom_ix_list;
    if (is("ds_ix_assemble")) return geom_elem64;
    if (is("ds_ix_bmax_nat")) return geom_k512;
    if (is("ds_ix_bmax")) return geom_k512;
    if (is("ds_ix_pin")) return geom_columns;
    if (is("ds_ix_gather_tile")) return geom_ix_gather_tile;
    if (is("ds_ix_gather_f4")) return geom_ix_gather_f4;
    if (is("ds_ix_gather")) return geom_pairs16;
    if (is("ds_ix_gslot")) return geom_ix_gslot;
    if (is("ix_gvis")) return geom_ix_gvis;
    return nullptr;
}

// by kernel family (file name prefix; longer prefixes first)
// ---- Qwen3.8-Flash-Next (qwen4_exp): hyper connections, PLE, QSA, expert-parallel MoE
// hc_combine: [NS = 4 streams, T] from xo (NS H, T)
static bool geom_hc_combine(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = 4; is[1] = sz(p, true, 0, 1); *rank = 2; return true;
}
// work-list MoE kernels: a fixed number of members (NW in the kernels)
static bool geom_nw48(const HabanaKernelParams*, uint64_t is[5], uint32_t* rank) {
    is[0] = 48; *rank = 1; return true;
}
// moe_sum_w: [H/64, T] from y (H, T)
static bool geom_out64(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, true, 0, 0) / 64; is[1] = sz(p, true, 0, 1); *rank = 2; return true;
}
// moe_up: [NB, K, T] from its output h (128, NB, K, T)
static bool geom_moe_up(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, true, 0, 1); is[1] = sz(p, true, 0, 2); is[2] = sz(p, true, 0, 3); *rank = 3; return true;
}
// moe_deq_p: [NOB, NIB, P] from W (256, 128, NOB, NIB, E) and the page table pe (input 2)
static bool geom_moe_deq_p(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, false, 0, 2); is[1] = sz(p, false, 0, 3); is[2] = sz(p, false, 2, 0); *rank = 3; return true;
}
// moe_act_g: [N/64] row blocks of h (F, N)
static bool geom_rows64(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = (sz(p, true, 0, 1) + 63) / 64; *rank = 1; return true;
}
// ple_conv: [C/64] channel blocks of xo (C, T); ple_conv_b: [C/64, T]
static bool geom_ple_conv(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, true, 0, 0) / 64; *rank = 1; return true;
}
static bool geom_ple_conv_b(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, true, 0, 0) / 64; is[1] = sz(p, true, 0, 1); *rank = 2; return true;
}
// qsa_kb: [NBc] blocks (kbs (128, 4 NBc), output 1); qsa_keys / qsa_keys_pf: [Lc / 1024, T] from K (Lc, T)
static bool geom_qsa_kb(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, true, 1, 1) / 4; *rank = 1; return true;
}
static bool geom_qsa_keys(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = (sz(p, true, 0, 0) + 1023) / 1024; is[1] = sz(p, true, 0, 1); *rank = 2; return true;
}
static GeomFn geom_fnext(const char* n) {
    auto is = [&](const char* pre) { return strncmp(n, pre, strlen(pre)) == 0; };
    if (is("hc_combine_t")) return geom_tok1;          // (prefix: before hc_combine)
    if (is("hc_combine")) return geom_hc_combine;
    if (is("hc_lo") || is("hc_mix")) return geom_elem;
    if (is("ple_hash")) return geom_one;
    if (is("ple_cvt") || is("ple_gate")) return geom_out_tokens;
    if (is("ple_conv_b")) return geom_ple_conv_b;      // (prefix: before ple_conv)
    if (is("ple_conv")) return geom_ple_conv;
    if (is("qsa_select") || is("qsa_mask_pk") || is("qsa_mask_pf")) return geom_columns;
    if (is("qsa_keys")) return geom_qsa_keys;           // qsa_keys and qsa_keys_pf
    if (is("qsa_kb_b")) return geom_columns;            // [B] rows (prefix: before qsa_kb)
    if (is("qsa_kb")) return geom_qsa_kb;
    if (is("qsa_q")) return geom_rows1;
    if (is("moe_up_w") || is("moe_down_w")) return geom_nw48;
    if (is("moe_sum_w")) return geom_out64;
    if (is("moe_deq_p")) return geom_moe_deq_p;
    if (is("moe_act_g")) return geom_rows64;
    if (is("moe_ovf")) return geom_out_tokens;
    if (is("moe_combine_p") || is("moe_down")) return geom_tok1;
    if (is("moe_up")) return geom_moe_up;
    return nullptr;
}

static GeomFn geom_for(const char* n) {
    auto is = [&](const char* pre) { return strncmp(n, pre, strlen(pre)) == 0; };
    if (is("spec_fix")) return geom_one;
    if (is("attn_slot_p1") || is("attn_slot_pc1") || is("attn_slot_pd1"))
        return geom_slot_p1;  // (prefix: before attn_slot_p)
    if (GeomFn g = geom_glm(n)) return g;
    if (GeomFn g = geom_fnext(n)) return g;
    if (is("lq_qdqp_")) return geom_qdqp;
    if (is("lq_")) return geom_param0;
    if (is("ml4_rope")) return geom_out_d12;
    if (is("ml4_perm")) return geom_perm;
    if (is("ml4_moe_sum") || is("rsum")) return geom_elem;
    if (is("ml4_ix_acc")) return geom_elem64;
    if (is("mla_store")) return geom_mla_store;
    if (is("ds_store_f4")) return geom_mla_store;
    if (is("mla_slot_out")) return geom_slot_out;
    if (is("mla_gather")) return geom_mla_gather;
    if (is("mla_slot_q")) return geom_slot_q;
    if (is("mla_slot_acc") || is("mla_slot_am")) return geom_slot_acc;
    if (is("mla_slot_pk")) return geom_slot_p;
    if (is("ix_slot_acc")) return geom_ix_slot_acc;
    if (is("ix_bcnt")) return geom_elem64;
    if (is("ix_bmax") || is("ix_cmask")) return geom_k512;
    if (is("ix_rcnt")) return geom_rcnt;
    if (is("ix_cgather_f4")) return geom_cgather_f4;
    if (is("ix_cgather")) return geom_cgather;
    if (is("ix_bgather")) return geom_bgather;
    if (is("ix_cfix")) return geom_k512;
    if (is("ix_slot_q8")) return geom_slot_q;
    if (is("ix_q8")) return geom_columns;
    if (is("ix_rsel")) return geom_columns;
    if (is("ix_mask")) return geom_elem64;
    if (is("ix_unmask")) return geom_columns;
    if (is("ix_list")) return geom_ix_list;
    if (is("mla_acc_init")) return geom_acc_init;
    if (is("gdn_decode")) return geom_gdn_decode;
    if (is("gdn_verify")) return geom_gdn_verify;
    if (is("hidden_store")) return geom_hidden_store;
    if (is("gdn_commit")) return geom_gdn_commit;
    if (is("samp_scan")) return geom_samp_scan;
    if (is("mtp_in")) return geom_out_tokens;
    if (is("score_gather")) return geom_score_gather;
    if (is("samp_mass")) return geom_samp_mass;
    if (is("samp_select")) return geom_samp_rows_in;
    if (is("samp_full") || is("samp_draw")) return geom_samp_rows_out;
    if (is("gdn_fused")) return geom_gdn_fused;
    if (is("gdn_ck_qk")) return geom_ck_qk;
    if (is("gdn_ck_solve") || is("gdn_ck_v")) return geom_ck_out0;
    if (is("gdn_ck_out")) return geom_ck_in0;
    if (is("gdn_ck_step")) return geom_ck_step;
    if (is("mtp_tok")) return geom_one;
    if (is("page_store")) return geom_page_store;
    if (is("kv_gather")) return geom_kv_gather;
    if (is("attn_slot_fin")) return geom_slot_fin;
    if (is("attn_slot_q")) return geom_slot_q;
    if (is("attn_slot_p") || is("mla_slot_p")) return geom_slot_p;
    if (is("attn_slot_acc")) return geom_slot_acc;
    if (is("attn_slot_mask")) return geom_slot_mask;
    if (is("attn_sdpa_acc")) return geom_sdpa_acc;
    if (is("attn_acc_init")) return geom_acc_init;
    if (is("kv_store")) return geom_kv_store;
    if (is("state_load")) return geom_state_load;
    if (is("state_store")) return geom_state_store;
    if (is("attn_prep") || is("dfl_prep")) return geom_attn_prep;  // one member per column
    if (is("attn_softmax")) return geom_rows1;
    if (is("attn_kvcat")) return geom_kvcat;
    if (is("attn_dec_p")) return geom_dec_p;
    if (is("attn_dec_comb")) return geom_dec_comb;
    if (is("swiglu")) return geom_elem;
    if (is("argmax1")) return geom_argmax1;
    if (is("argmax2")) return geom_argmax2;
    return geom_out_tokens;
}

// Per-kernel access patterns.
typedef void (*ApFn)(const HabanaKernelParams*, HabanaKernelInstantiation*);

static void amap(TensorAccessPattern& t, unsigned dim, unsigned axis, int a, int first, int last) {
    t.mapping[dim].indexSpaceDim = axis;
    t.mapping[dim].a = float(a);
    t.mapping[dim].start_b = float(first);
    t.mapping[dim].end_b = float(last);
}
// kv_gather, kv_gather8 (the same (256, C, S) slab)
static void ap_kv_gather(const HabanaKernelParams*, HabanaKernelInstantiation* inst) {
    TensorAccessPattern& o = inst->outputTensorAccessPattern[0];
    o.allRequired = 0;
    amap(o, 0, 0, 256, 0, 255); amap(o, 1, 1, 32, 0, 31); amap(o, 2, 2, 1, 0, 0);
}

// attn_slot_q
static void ap_slot_q(const HabanaKernelParams*, HabanaKernelInstantiation* inst) {
    TensorAccessPattern& o = inst->outputTensorAccessPattern[0];
    o.allRequired = 0;
    amap(o, 0, 0, 256, 0, 255); amap(o, 1, 1, 64, 0, 63); amap(o, 2, 2, 1, 0, 0);
}

// mla_gather, mla_slot_q
static void ap_mla_gather(const HabanaKernelParams* p, HabanaKernelInstantiation* inst) {
    const int D = int(sz(p, true, 0, 0));
    TensorAccessPattern& o = inst->outputTensorAccessPattern[0];
    o.allRequired = 0;
    amap(o, 0, 0, D, 0, D - 1); amap(o, 1, 1, 32, 0, 31); amap(o, 2, 2, 1, 0, 0);
}
// ds_gather_list
static void ap_gather8(const HabanaKernelParams* p, HabanaKernelInstantiation* inst) {
    const int D = int(sz(p, true, 0, 0));
    TensorAccessPattern& o = inst->outputTensorAccessPattern[0];
    o.allRequired = 0;
    amap(o, 0, 0, D, 0, D - 1); amap(o, 1, 1, 8, 0, 7); amap(o, 2, 2, 1, 0, 0);
}
// ds_gather_list_f4 staging: codes (256, C, T) + scales (32, C, T), 16 keys a member
static void ap_gather_list_f4(const HabanaKernelParams* p, HabanaKernelInstantiation* inst) {
    TensorAccessPattern& c = inst->outputTensorAccessPattern[0];
    c.allRequired = 0;
    amap(c, 0, 0, 256, 0, 255); amap(c, 1, 1, 16, 0, 15); amap(c, 2, 2, 1, 0, 0);
    TensorAccessPattern& s = inst->outputTensorAccessPattern[1];
    s.allRequired = 0;
    amap(s, 0, 0, 32, 0, 31); amap(s, 1, 1, 16, 0, 15); amap(s, 2, 2, 1, 0, 0);
}
// ds_gather_list_blk_f4 staging: codes (256, B, Q) + scales (32, B, Q) + idx (B, Q), 16 keys a member
static void ap_gather_list_blk_f4(const HabanaKernelParams* p, HabanaKernelInstantiation* inst) {
    TensorAccessPattern& c = inst->outputTensorAccessPattern[0];
    c.allRequired = 0;
    amap(c, 0, 0, 256, 0, 255); amap(c, 1, 1, 16, 0, 15); amap(c, 2, 2, 1, 0, 0);
    TensorAccessPattern& s = inst->outputTensorAccessPattern[1];
    s.allRequired = 0;
    amap(s, 0, 0, 32, 0, 31); amap(s, 1, 1, 16, 0, 15); amap(s, 2, 2, 1, 0, 0);
    TensorAccessPattern& x = inst->outputTensorAccessPattern[2];
    x.allRequired = 0;
    amap(x, 0, 1, 16, 0, 15); amap(x, 1, 2, 1, 0, 0);
}
// ds_gather_win_blk: ks_blk (D, B, Q) bf16 + idx (B, Q) i32, 32 keys a member (mla_gather layout)
static void ap_gather_win_blk(const HabanaKernelParams* p, HabanaKernelInstantiation* inst) {
    const int D = int(sz(p, true, 0, 0));
    TensorAccessPattern& o = inst->outputTensorAccessPattern[0];
    o.allRequired = 0;
    amap(o, 0, 0, D, 0, D - 1); amap(o, 1, 1, 32, 0, 31); amap(o, 2, 2, 1, 0, 0);
    TensorAccessPattern& x = inst->outputTensorAccessPattern[1];
    x.allRequired = 0;
    amap(x, 0, 1, 32, 0, 31); amap(x, 1, 2, 1, 0, 0);
}
// ds_ldec_f4: dequant one column of codes (256) + scales (32) -> 512 bf16
static void ap_ldec_f4(const HabanaKernelParams* p, HabanaKernelInstantiation* inst) {
    for (int t = 0; t < 2; t++) {
        TensorAccessPattern& w = inst->inputTensorAccessPattern[t];
        w.allRequired = 0;
        amap(w, 0, 0, t ? 32 : 256, 0, t ? 31 : 255); amap(w, 1, 1, 1, 0, 0);
    }
    TensorAccessPattern& o = inst->outputTensorAccessPattern[0];
    o.allRequired = 0;
    amap(o, 0, 0, 512, 0, 511); amap(o, 1, 1, 1, 0, 0);
}
// ds_gather_dense_f4 staging: codes (256, C, S) + scales (32, C, S), 32 keys a member
static void ap_gather_dense_f4(const HabanaKernelParams* p, HabanaKernelInstantiation* inst) {
    TensorAccessPattern& c = inst->outputTensorAccessPattern[0];
    c.allRequired = 0;
    amap(c, 0, 0, 256, 0, 255); amap(c, 1, 1, 32, 0, 31); amap(c, 2, 2, 1, 0, 0);
    TensorAccessPattern& s = inst->outputTensorAccessPattern[1];
    s.allRequired = 0;
    amap(s, 0, 0, 32, 0, 31); amap(s, 1, 1, 32, 0, 31); amap(s, 2, 2, 1, 0, 0);
}
static void ap_mla_slot_q(const HabanaKernelParams* p, HabanaKernelInstantiation* inst) {
    const int D = int(sz(p, true, 0, 0));
    TensorAccessPattern& o = inst->outputTensorAccessPattern[0];
    o.allRequired = 0;
    amap(o, 0, 0, D, 0, D - 1); amap(o, 1, 1, 64, 0, 63); amap(o, 2, 2, 1, 0, 0);
}

// lq_qdqp_*
static void ap_qdqp(const HabanaKernelParams* p, HabanaKernelInstantiation* inst) {
    const int SBk = ((const int32_t*)p->nodeParams.nodeParams)[3];
    TensorAccessPattern& o = inst->outputTensorAccessPattern[0];
    o.allRequired = 0;
    amap(o, 0, 0, 128, 0, 127); amap(o, 1, 1, SBk, 0, SBk - 1); amap(o, 2, 2, 1, 0, 0);
}

// x8_copy
static void ap_x8_copy(const HabanaKernelParams* p, HabanaKernelInstantiation* inst) {
    const int E = int(sz(p, false, 0, 2));
    TensorAccessPattern& w = inst->inputTensorAccessPattern[0];
    w.allRequired = 0;
    amap(w, 0, 0, 256, 0, 255); amap(w, 1, 2, 128, 0, 127); amap(w, 2, 1, 0, 0, E - 1);
    TensorAccessPattern& o = inst->outputTensorAccessPattern[0];
    o.allRequired = 0;
    amap(o, 0, 0, 256, 0, 255); amap(o, 1, 2, 128, 0, 127); amap(o, 2, 1, 1, 0, 0);
}

// mx4_dec, mx4_decs over [units of 2048 elements, rows]: codes and scales by unit, the decoded output by unit and
// row; mx4_decs reads the row of codes and scales that ids selects (the full range of dim 1).
static void ap_mx4(const HabanaKernelParams* p, HabanaKernelInstantiation* inst, bool slots) {
    const int E = int(sz(p, false, 0, 1));
    // mx4_decs param 1: the unit offset into each row of codes and scales
    const int u0 = slots && p->nodeParams.nodeParams && p->nodeParams.nodeParamsSize >= 8
                       ? ((const int32_t*)p->nodeParams.nodeParams)[1] : 0;
    for (int t = 0; t < 2; t++) {
        TensorAccessPattern& w = inst->inputTensorAccessPattern[t];
        w.allRequired = 0;
        const int a = t ? 64 : 1024;
        amap(w, 0, 0, a, u0 * a, u0 * a + a - 1);
        if (slots) amap(w, 1, 1, 0, 0, E - 1); else amap(w, 1, 1, 1, 0, 0);
    }
    TensorAccessPattern& o = inst->outputTensorAccessPattern[0];
    o.allRequired = 0;
    amap(o, 0, 0, 2048, 0, 2047); amap(o, 1, 1, 1, 0, 0);
}
static void ap_mx4_dec(const HabanaKernelParams* p, HabanaKernelInstantiation* inst) { ap_mx4(p, inst, false); }
static void ap_mx4_decs(const HabanaKernelParams* p, HabanaKernelInstantiation* inst) { ap_mx4(p, inst, true); }

static ApFn ap_for(const char* n) {
    if (strcmp(n, "mx4_dec") == 0) return ap_mx4_dec;
    if (strcmp(n, "mx4_decs") == 0) return ap_mx4_decs;
    if (strcmp(n, "x8_copy") == 0) return ap_x8_copy;
    if (strcmp(n, "gx_gather") == 0) return ap_mla_gather;
    if (strncmp(n, "lq_qdqp_", 8) == 0) return ap_qdqp;
    if (strcmp(n, "mla_slot_q") == 0) return ap_mla_slot_q;
    if (strcmp(n, "mla_gather") == 0 || strcmp(n, "mla_gather_list") == 0) return ap_mla_gather;
    if (strcmp(n, "ds_gather_dense_f4") == 0) return ap_gather_dense_f4;
    if (strcmp(n, "ds_gather_list") == 0) return ap_gather8;
    if (strcmp(n, "ds_gather_list_f4") == 0) return ap_gather_list_f4;
    if (strcmp(n, "ds_gather_list_blk_f4") == 0) return ap_gather_list_blk_f4;
    if (strcmp(n, "ds_gather_win_blk") == 0) return ap_gather_win_blk;
    if (strcmp(n, "ds_ldec_f4") == 0) return ap_ldec_f4;
    if (strcmp(n, "attn_slot_q") == 0) return ap_slot_q;
    if (strcmp(n, "kv_gather") == 0 || strcmp(n, "kv_gather8") == 0) return ap_kv_gather;
    return nullptr;
}

struct KDef { char guid[MAX_NODE_NAME]; const unsigned char* elf; unsigned len; GeomFn geom; ApFn ap; };
static const unsigned kCount = sizeof(kElfs) / sizeof(kElfs[0]);
static KDef kKernels[sizeof(kElfs) / sizeof(kElfs[0])];
static bool kInit = false;

static void init_table() {
    if (kInit) return;
    for (unsigned i = 0; i < kCount; i++) {
        snprintf(kKernels[i].guid, MAX_NODE_NAME, "rg_%s_f32", kElfs[i].name);
        kKernels[i].elf = kElfs[i].elf;
        kKernels[i].len = kElfs[i].len;
        kKernels[i].geom = geom_for(kElfs[i].name);
        kKernels[i].ap = ap_for(kElfs[i].name);
    }
    kInit = true;
}

extern "C" {

__attribute__((visibility("default")))
GlueCodeReturn GetKernelGuids(DeviceId deviceId, uint32_t* kernelCount, GuidInfo* guids) {
    if (deviceId != DEVICE_ID_GAUDI2) {
        if (kernelCount) *kernelCount = 0;
        return GLUE_SUCCESS;
    }
    init_table();
    if (guids) {
        for (unsigned i = 0; i < kCount && i < *kernelCount; i++) {
            memset(&guids[i], 0, sizeof(GuidInfo));
            strncpy(guids[i].name, kKernels[i].guid, MAX_NODE_NAME - 1);
        }
    }
    if (kernelCount) *kernelCount = kCount;
    return GLUE_SUCCESS;
}

__attribute__((visibility("default")))
GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams* p, HabanaKernelInstantiation* inst) {
    init_table();
    const KDef* k = nullptr;
    for (unsigned i = 0; i < kCount; i++)
        if (strcmp(p->guid.name, kKernels[i].guid) == 0) { k = &kKernels[i]; break; }
    if (!k) return GLUE_NODE_NOT_FOUND;

    uint64_t is[5] = {1, 1, 1, 1, 1};
    uint32_t rank = 1;
    if (!k->geom(p, is, &rank)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    inst->indexSpaceRank = rank;
    for (int d = 0; d < 5; d++) inst->indexSpaceGeometry[d] = d < (int)rank ? is[d] : 1;

    for (uint32_t t = 0; t < p->inputTensorNr; t++) {
        TensorAccessPattern& ap = inst->inputTensorAccessPattern[t];
        memset(&ap, 0, sizeof(ap));
        ap.allRequired = 1;
        for (uint32_t d = 0; d < MAX_TENSOR_DIM; d++) { ap.mapping[d].end_b = float(p->inputTensors[t].geometry.maxSizes[d]) - 1; }
    }
    for (uint32_t t = 0; t < p->outputTensorNr; t++) {
        TensorAccessPattern& ap = inst->outputTensorAccessPattern[t];
        memset(&ap, 0, sizeof(ap));
        ap.allRequired = 1;
        for (uint32_t d = 0; d < MAX_TENSOR_DIM; d++) { ap.mapping[d].end_b = float(p->outputTensors[t].geometry.maxSizes[d]) - 1; }
    }
    if (k->ap) k->ap(p, inst);

    // scalar params: raw copy of the node params (u32 words)
    uint32_t nwords = p->nodeParams.nodeParamsSize / 4;
    if (nwords > MAX_SCALAR_PARAMS) nwords = MAX_SCALAR_PARAMS;
    if (nwords && p->nodeParams.nodeParams) memcpy(inst->kernel.scalarParams, p->nodeParams.nodeParams, nwords * 4);
    inst->kernel.paramsNr = nwords;

    uint32_t given = inst->kernel.elfSize;
    inst->kernel.elfSize = k->len;
    if (given < k->len) return GLUE_INSUFFICIENT_ELF_BUFFER;
    memcpy(inst->kernel.kernelElf, k->elf, k->len);
    return GLUE_SUCCESS;
}

// Static shapes only.
__attribute__((visibility("default")))
GlueCodeReturn GetShapeInference(DeviceId, ShapeInferenceParams*, ShapeInferenceOutput*) { return GLUE_SUCCESS; }

__attribute__((visibility("default")))
uint64_t GetLibVersion() { return 1; }

}  // extern "C"
