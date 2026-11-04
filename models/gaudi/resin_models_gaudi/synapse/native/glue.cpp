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
static bool geom_state_load(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, true, 0, 1) / 16; is[1] = sz(p, true, 0, 2); *rank = 2; return true;
}
static bool geom_state_store(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, false, 2, 1) / 16; is[1] = sz(p, false, 2, 2); *rank = 2; return true;
}

// slot attention: attn_slot_q, attn_slot_p, attn_slot_acc, attn_acc_init, kv_store
static uint64_t blocks64(uint64_t n) { return (n + 63) / 64; }
static bool geom_slot_q(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = 1; is[1] = blocks64(sz(p, true, 0, 1)); is[2] = sz(p, true, 0, 2); *rank = 3; return true;
}
static bool geom_slot_p(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = blocks64(sz(p, false, 0, 1)); is[1] = sz(p, false, 0, 2); *rank = 2; return true;
}
static bool geom_slot_acc(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = blocks64(sz(p, false, 4, 1)); is[1] = sz(p, false, 4, 2); *rank = 2; return true;
}
// attn_slot_mask, attn_sdpa_acc
static bool geom_slot_mask(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = blocks64(sz(p, true, 0, 1)); is[1] = sz(p, true, 0, 3); *rank = 2; return true;
}
static bool geom_sdpa_acc(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    const uint64_t H = sz(p, false, 0, 2);
    is[0] = blocks64(sz(p, false, 4, 1) / H); is[1] = H; is[2] = sz(p, false, 4, 2);
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
// ix_list
static bool geom_ix_list(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    const int n = p->nodeParams.nodeParams && p->nodeParams.nodeParamsSize >= 4 ? *(const int*)p->nodeParams.nodeParams : 1;
    is[0] = n > 0 ? n : 1; is[1] = sz(p, true, 0, 1); *rank = 2; return true;
}
// ix_unmask
static bool geom_columns(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = sz(p, true, 0, 1); *rank = 1; return true;
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
static bool geom_nw24(const HabanaKernelParams*, uint64_t is[5], uint32_t* rank) {
    is[0] = 24; *rank = 1; return true;
}
// mhc_res
static bool geom_mhc_res(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    is[0] = (sz(p, false, 1, 0) + 255) / 256; is[1] = sz(p, false, 1, 1); *rank = 2; return true;
}
// mhc_mix
static bool geom_mhc_mix(const HabanaKernelParams* p, uint64_t is[5], uint32_t* rank) {
    const uint64_t T = sz(p, true, 0, 0);
    if (T > 16) { is[0] = sz(p, true, 0, 1) / 4; is[1] = (T + 3) / 4; *rank = 2; return true; }
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
static GeomFn geom_glm(const char* n) {
    auto is = [&](const char* pre) { return strncmp(n, pre, strlen(pre)) == 0; };
    if (is("mhc_mix")) return geom_mhc_mix;
    if (is("mhc_res")) return geom_mhc_res;
    if (is("mhc_pre") || is("mhc_post")) return geom_tok1;
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
    if (is("ds_xact")) return geom_xact;
    if (is("ds_xsum") || is("ds_tscale") || is("pack16")) return geom_elem;
    if (is("ds_xcomb")) return geom_out_tokens;
    if (is("ds_xqg")) return geom_xqg;                // (prefix: before ds_xq)
    if (is("ds_xq")) return geom_xq;
    if (is("ds_q8")) return geom_q8;
    if (is("gx_pool")) return geom_mla_store;
    if (is("gx_gather")) return geom_mla_gather;
    if (is("gx_mask")) return geom_gx_mask;
    return nullptr;
}

// by kernel family (file name prefix; longer prefixes first)
static GeomFn geom_for(const char* n) {
    auto is = [&](const char* pre) { return strncmp(n, pre, strlen(pre)) == 0; };
    if (GeomFn g = geom_glm(n)) return g;
    if (is("lq_qdqp_")) return geom_qdqp;
    if (is("lq_")) return geom_param0;
    if (is("ml4_rope")) return geom_out_d12;
    if (is("ml4_perm")) return geom_perm;
    if (is("ml4_moe_sum") || is("rsum")) return geom_elem;
    if (is("ml4_ix_acc")) return geom_elem64;
    if (is("mla_store")) return geom_mla_store;
    if (is("mla_slot_out")) return geom_slot_out;
    if (is("mla_gather")) return geom_mla_gather;
    if (is("mla_slot_q")) return geom_slot_q;
    if (is("mla_slot_acc") || is("mla_slot_am")) return geom_slot_acc;
    if (is("mla_slot_pk")) return geom_slot_p;
    if (is("ix_slot_acc")) return geom_ix_slot_acc;
    if (is("ix_bcnt")) return geom_elem64;
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
    if (is("attn_slot_q")) return geom_slot_q;
    if (is("attn_slot_p") || is("mla_slot_p")) return geom_slot_p;
    if (is("attn_slot_acc")) return geom_slot_acc;
    if (is("attn_slot_mask")) return geom_slot_mask;
    if (is("attn_sdpa_acc")) return geom_sdpa_acc;
    if (is("attn_acc_init")) return geom_acc_init;
    if (is("kv_store")) return geom_kv_store;
    if (is("state_load")) return geom_state_load;
    if (is("state_store")) return geom_state_store;
    if (is("attn_prep")) return geom_attn_prep;
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
// kv_gather
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

static ApFn ap_for(const char* n) {
    if (strcmp(n, "x8_copy") == 0) return ap_x8_copy;
    if (strcmp(n, "gx_gather") == 0) return ap_mla_gather;
    if (strncmp(n, "lq_qdqp_", 8) == 0) return ap_qdqp;
    if (strcmp(n, "mla_slot_q") == 0) return ap_mla_slot_q;
    if (strcmp(n, "mla_gather") == 0 || strcmp(n, "mla_gather_list") == 0) return ap_mla_gather;
    if (strcmp(n, "attn_slot_q") == 0) return ap_slot_q;
    if (strcmp(n, "kv_gather") == 0) return ap_kv_gather;
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
