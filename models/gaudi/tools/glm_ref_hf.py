"""Oracle for GLM-5.3-Flash (glm5_next) from the transformers implementation, on CPU in float32.

  python tools/glm_ref_hf.py MODEL_DIR N OUT.npz [--decode K] [--dump L] TOKEN_IDS...

Runs the first N decoder layers (N = 45: the whole text model) + the HC head mean + final norm + LM
head, streaming: one decoder layer is built, loaded, run over the sequence and freed at a time
(host RAM ~ one layer); the per-layer caches (KDA conv / recurrent state, MLA + indexer keys) live in
one DynamicCache, so K greedy decode steps follow the prefill (cached, the recurrent KDA path).
Experts are dequantized on demand (only the hit ones, nothing cached).

Saved (npz): logits (P + K, V) of the prompt positions and of each decode step, tokens (the greedy
continuation), x{i} = residual streams (T, 4, H) after layer i (prefill), and for layers < L
(--dump, default 4) the module internals of the prefill:
  l{i}.hca_post / hca_comb / hca_in   attention-site mHC: post (T,4), comb (T,4,4), collapsed input (T,H)
  l{i}.hcf_*                          the same for the MLP site
  l{i}.attn_out                       attention / KDA module output (T,H)
  l{i}.kda_q / kda_k / kda_v / kda_g / kda_beta / kda_o / kda_state   KDA kernel inputs / output / final state
  l{i}.idx                            indexer top-k token indices (T, 2048 + 3), -1 = none
  l{i}.router_ids / router_w          routing (T, 8)
  l{i}.mlp_out                        MLP / MoE output (T,H)
"""
import argparse
import json
import os
import struct

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("path"); ap.add_argument("N", type=int); ap.add_argument("out")
ap.add_argument("--decode", type=int, default=8); ap.add_argument("--dump", type=int, default=4)
ap.add_argument("--fp8", choices=("act", "weight", "both"), help="round as Wright does on FP8: per-token activations and/or per-row weights at its FP8 GEMMs; everything else stays f32")
ap.add_argument("--small", choices=("fp8", "bf16"), help="with --routed: precision of the dense, shared-expert and MLA q_a / kv_a GEMMs, weights and activations; overrides --fp8")
ap.add_argument("--routed", choices=("fp8", "bf16"), help="with --small: precision of the routed experts' GEMMs")
ap.add_argument("--compare", help="an oracle npz: print each layer's stream correlation against it, and the logits' at the end")
ap.add_argument("ids", nargs="+", type=int)
a = ap.parse_args()
torch.set_num_threads(int(os.environ.get("REF_THREADS", 32)))

from transformers.cache_utils import DynamicCache
from transformers.models.glm5_next import modeling_glm5_next as M
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig

top = json.load(open(os.path.join(a.path, "config.json")))
tc = dict(top["text_config"])
N = a.N
tc["num_hidden_layers"] = N
for k in ("layer_types", "mlp_layer_types", "indexer_types"):
    if k in tc: tc[k] = tc[k][:N]
tc["num_nextn_predict_layers"] = 0
cfg = Glm5NextTextConfig(**{k: v for k, v in tc.items() if k != "model_type"})
cfg._attn_implementation = "eager"
cfg.dtype = torch.float32

# ---- lazy safetensors
idx = json.load(open(os.path.join(a.path, "model.safetensors.index.json")))["weight_map"]
_hdr, _map = {}, {}
def raw(name):
    f = os.path.join(a.path, idx[name])
    if f not in _hdr:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            _hdr[f] = (json.loads(fh.read(n)), 8 + n)
        _map[f] = np.memmap(f, np.uint8, "r")
    h, base = _hdr[f]
    v = h[name]
    s, e = v["data_offsets"]
    dt = {"BF16": np.uint16, "F8_E4M3": np.uint8, "I64": np.int64, "F32": np.float32}[v["dtype"]]
    return _map[f][base + s:base + e].view(dt).reshape(v["shape"]), v["dtype"]

def e4m3fn_lut():
    u = np.arange(256)
    s = np.where(u & 0x80, -1.0, 1.0); e = (u >> 3) & 15; m = u & 7
    v = s * np.where(e == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * 2.0 ** (e - 7.0))
    v[(u & 0x7F) == 0x7F] = np.nan
    return v.astype(np.float32)
LUT = e4m3fn_lut()

def f32(name):
    t, dt = raw(name)
    if dt == "BF16":
        return torch.from_numpy((t.astype(np.uint32) << 16).view(np.float32).copy())
    if dt == "F8_E4M3":
        s = f32(name + "_scale_inv").numpy()
        w = LUT[t]
        Mb, Kb = s.shape
        w = w.reshape(Mb, w.shape[0] // Mb, Kb, w.shape[1] // Kb) * s[:, None, :, None]
        return torch.from_numpy(w.reshape(t.shape).astype(np.float32))
    return torch.from_numpy(np.asarray(t).copy())

# ---- FP8 rounding
FP8_ACT = a.fp8 in ("act", "both")
FP8_WEIGHT = a.fp8 in ("weight", "both")
assert bool(a.small) == bool(a.routed) and not (a.small and a.fp8), "--small and --routed go together, without --fp8"
FP8_LINEARS = {"gate_proj", "up_proj", "down_proj", "q_a_proj", "kv_a_proj_with_mqa"}

def fp8_round(v):
    e = torch.floor(torch.log2(v.abs().clamp_min(1e-30)))
    step = torch.pow(2.0, e - 3).clamp_min(2.0 ** -10)
    return (torch.round(v / step) * step).clamp(-240.0, 240.0)

def fp8_rows(x):
    """fp8(x / s) s, with s from the largest magnitude per row of the last dim"""
    s = x.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 240.0
    return fp8_round(x / s) * s

def bf16_round(x):
    return x.to(torch.bfloat16).to(torch.float32)

ROUND = {"fp8": fp8_rows, "bf16": bf16_round}

def act(x):
    if a.routed:
        return ROUND[a.routed](x)
    return fp8_rows(x) if FP8_ACT else x

def wt(w):
    if a.routed:
        return ROUND[a.routed](w)
    return fp8_rows(w) if FP8_WEIGHT else w

pre = "model.language_model."
def ckpt_name(i, k):
    """checkpoint name of decoder-layer-i parameter k (module naming)"""
    L = f"{pre}layers.{i}."
    for site, tag in (("attn_hc.", "hc_attn_"), ("ffn_hc.", "hc_ffn_")):
        if k.startswith(site):
            return L + tag + k[len(site):]
    if k.startswith("self_attn.forget_gate."):
        return L + "self_attn." + k[len("self_attn.forget_gate."):]
    return L + k

# ---- lazy experts (clamped SwiGLU as Glm5NextTextExperts)
class LazyExperts(torch.nn.Module):
    def __init__(self, layer, limit):
        super().__init__()
        self.p, self.limit = f"{pre}layers.{layer}.mlp.experts.", limit
    def forward(self, hidden_states, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden_states)
        for e in torch.unique(top_k_index).tolist():
            tok, slot = torch.nonzero(top_k_index == e, as_tuple=True)
            p = f"{self.p}{e}."
            h = act(hidden_states[tok])
            g = h @ wt(f32(p + "gate_proj.weight")).T
            u = h @ wt(f32(p + "up_proj.weight")).T
            g = g.clamp(max=self.limit); u = u.clamp(-self.limit, self.limit)
            y = act(torch.nn.functional.silu(g) * u) @ wt(f32(p + "down_proj.weight")).T
            out.index_add_(0, tok, y * top_k_weights[tok, slot, None])
        return out

M.Glm5NextPreTrainedModel._init_weights = lambda self, m: None
_real_experts = M.Glm5NextTextExperts

def build_layer(i):
    M.Glm5NextTextExperts = lambda config: torch.nn.Module()
    layer = M.Glm5NextTextDecoderLayer(cfg, i)
    M.Glm5NextTextExperts = _real_experts
    if isinstance(layer.mlp, M.Glm5NextTextMoE):
        layer.mlp.experts = LazyExperts(i, cfg.swiglu_limit)
    sd = layer.state_dict()
    for k in list(sd.keys()):
        if ".experts." in k:
            continue
        if k == "self_attn.conv1d.weight":           # checkpoint: separate q / k / v short convs
            t = torch.cat([f32(f"{pre}layers.{i}.self_attn.{c}_conv1d.weight") for c in "qkv"])
        else:
            t = f32(ckpt_name(i, k))
        assert t.shape == sd[k].shape, (i, k, t.shape, sd[k].shape)
        sd[k].copy_(t.to(sd[k].dtype))
    layer = layer.float().eval()
    if a.small:
        round_small = ROUND[a.small]
        for name, m in layer.named_modules():
            if isinstance(m, torch.nn.Linear) and name.split(".")[-1] in FP8_LINEARS:
                m.weight.data.copy_(round_small(m.weight.data))
                m.register_forward_pre_hook(lambda _m, inp: (round_small(inp[0]),) + tuple(inp[1:]))
    elif a.fp8:
        for name, m in layer.named_modules():
            if isinstance(m, torch.nn.Linear) and name.split(".")[-1] in FP8_LINEARS:
                if FP8_WEIGHT:
                    m.weight.data.copy_(fp8_rows(m.weight.data))
                if FP8_ACT:
                    m.register_forward_pre_hook(lambda _m, inp: (fp8_rows(inp[0]),) + tuple(inp[1:]))
    return layer

# ---- module-internal dumps (prefill, layers < --dump)
dumps = {}
cur = {"i": None, "on": False}
def put(name, t):
    if cur["on"] and cur["i"] is not None and cur["i"] < a.dump:
        dumps[f"l{cur['i']}.{name}"] = t.detach().float().reshape(-1, *t.shape[2:]).numpy().copy() if t.dim() > 2 else t.detach().float().numpy().copy()

_chunk, _rec = M.chunk_kimi_delta_attention, M.recurrent_kimi_delta_attention
def chunk_rec(q, k, v, g, beta, **kw):
    o, st = _chunk(q, k, v, g, beta, **kw)
    for n_, t_ in (("kda_q", q), ("kda_k", k), ("kda_v", v), ("kda_g", g), ("kda_beta", beta), ("kda_o", o)):
        put(n_, t_[0])
    if st is not None:
        put("kda_state", st[0])
    return o, st
M.chunk_kimi_delta_attention = chunk_rec

def hooks(layer):
    hs = []
    def hc(tag):
        def f(m, inp, out):
            post, comb, coll = out
            put(f"{tag}_post", post[0]); put(f"{tag}_comb", comb[0]); put(f"{tag}_in", coll[0])
        return f
    hs.append(layer.attn_hc.register_forward_hook(hc("hca")))
    hs.append(layer.ffn_hc.register_forward_hook(hc("hcf")))
    hs.append(layer.self_attn.register_forward_hook(lambda m, i_, o: put("attn_out", (o[0] if isinstance(o, tuple) else o)[0])))
    if getattr(layer.self_attn, "indexer", None) is not None:
        hs.append(layer.self_attn.indexer.register_forward_hook(lambda m, i_, o: put("idx", o[0])))
    hs.append(layer.mlp.register_forward_hook(lambda m, i_, o: put("mlp_out", o[0])))
    if isinstance(layer.mlp, M.Glm5NextTextMoE):
        def rt(m, i_, o):
            _, w, ids = o
            put("router_w", w); put("router_ids", ids)
        hs.append(layer.mlp.gate.register_forward_hook(rt))
    return hs

# ---- streaming forward
emb = f32(pre + "embed_tokens.weight")
norm_w = f32(pre + "norm.weight")
lm_head = f32("lm_head.weight")
cache = DynamicCache(config=cfg)
keep, KEEP_N = {}, int(os.environ.get("REF_KEEP_N", 8))

def run(tokens, prefill):
    x = emb[torch.tensor(tokens)][None].unsqueeze(2).expand(-1, -1, cfg.hc_mult, -1).contiguous()
    mask = torch.ones(1, len(tokens), dtype=torch.bool)
    topk, acts = None, {}
    for i in range(N):
        layer = keep[i] if i in keep else build_layer(i)
        if N <= KEEP_N:                              # small models: keep the built layers for the decode steps
            keep[i] = layer
        hs = hooks(layer)
        cur.update(i=i, on=prefill)
        with torch.no_grad():
            x, topk = layer(x, attention_mask=mask, past_key_values=cache, use_cache=True, prev_topk_indices=topk)
        for h in hs: h.remove()
        if prefill:
            acts[f"x{i}"] = x[0].numpy().copy()
            if compare is not None:
                P_ = acts[f"x{i}"].shape[0]
                c = np.corrcoef(acts[f"x{i}"][2:].ravel().astype(np.float64), compare[f"x{i}"][2:P_].ravel().astype(np.float64))[0, 1]
                print(f"  x{i} corr against the oracle, past column 1: {c:.5f}", flush=True)
        del layer
        print(f"  layer {i} done", flush=True)
    with torch.no_grad():
        h = x.mean(dim=2)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * norm_w
        return (h[0] @ lm_head.T).numpy(), acts

compare = np.load(a.compare) if a.compare else None
logits, acts = run(a.ids, True)
if compare is not None:
    r_ = compare["logits"][:len(a.ids)]
    c = np.corrcoef(logits[2:].ravel().astype(np.float64), r_[2:].ravel().astype(np.float64))[0, 1]
    print(f"logits corr against the oracle, past column 1: {c:.5f}, argmax equal {int((logits[2:].argmax(-1) == r_[2:].argmax(-1)).sum())}/{len(a.ids) - 2}", flush=True)
lg = [logits]
toks = [int(logits[-1].argmax())]
for s in range(a.decode):
    l1, _ = run([toks[-1]], False)
    lg.append(l1)
    toks.append(int(l1[-1].argmax()))
    print(f"decode {s}: token {toks[-2]} -> {toks[-1]}", flush=True)
np.savez(a.out, logits=np.concatenate(lg), tokens=np.array(toks[:a.decode + 1]), **acts, **dumps)
print("saved", a.out, "prompt", len(a.ids), "greedy", toks[:a.decode + 1])
