# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "transformers", "numpy"]
# ///
"""Reference logits for Qwen3.8-Flash-Next from the transformers implementation (``qwen4_exp``), on the CPU in
float32, for a truncated model (the first N decoder layers, the final hyper-connection mixer and the LM head).

    python qwen4_exp_ref_hf.py MODEL_DIR N OUT.npz [--no-table] TOKEN_IDS...

Run it with a venv that has torch and a transformers with ``qwen4_exp``. Experts
and n-gram table rows are read lazily from the safetensors (only the hit experts are dequantized), so the
reference fits in host RAM and works on a truncated download. ``--no-table``: the n-gram embedding contributes
zero rows (for a download without the table; the device runs with ``table=False`` to match). Saves ``logits``
(T, V) f32 and every decoder layer's residual streams ``x{i}`` (T, 4H).
"""

from __future__ import annotations

import json
import os
import struct
import sys

import numpy as np
import torch

args = [a for a in sys.argv[1:] if not a.startswith("--")]
NO_TABLE = "--no-table" in sys.argv
path, N, out = args[0], int(args[1]), args[2]
ids = [int(x) for x in args[3:]]
from transformers.models.qwen4_exp import modeling_qwen4_exp as M  # noqa: E402
from transformers.models.qwen4_exp.configuration_qwen4_exp import (
    Qwen4ExpTextConfig,  # noqa: E402
)

top = json.load(open(os.path.join(path, "config.json")))
tc = dict(top["text_config"])
tc["num_hidden_layers"] = N
tc["layer_types"] = tc["layer_types"][:N]
tc["ple_layer_ids"] = [i for i in tc.get("ple_layer_ids", []) if i <= N]
tc.pop("mtp", None)
tc["mtp_num_hidden_layers"] = 0
cfg = Qwen4ExpTextConfig(**{k: v for k, v in tc.items() if k != "model_type"})
cfg._attn_implementation = "eager"
cfg.dtype = torch.float32

# ---- lazy safetensors
idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
_hdr, _map = {}, {}


def raw(name):
    f = os.path.join(path, idx[name])
    if f not in _hdr:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            _hdr[f] = (json.loads(fh.read(n)), 8 + n)
        _map[f] = np.memmap(f, np.uint8, "r")
    h, base = _hdr[f]
    v = h[name]
    a, b = v["data_offsets"]
    dt = {"BF16": np.uint16, "F8_E4M3": np.uint8, "I64": np.int64, "F32": np.float32}[
        v["dtype"]
    ]
    return _map[f][base + a : base + b].view(dt).reshape(v["shape"]), v["dtype"]


def e4m3fn_lut():
    u = np.arange(256)
    s = np.where(u & 0x80, -1.0, 1.0)
    e = (u >> 3) & 15
    m = u & 7
    v = s * np.where(e == 0, m / 8.0 * 2.0**-6, (1 + m / 8.0) * 2.0 ** (e - 7.0))
    v[(u & 0x7F) == 0x7F] = np.nan
    return v.astype(np.float32)


LUT = e4m3fn_lut()


def f32(name):
    a, dt = raw(name)
    if dt == "BF16":
        return torch.from_numpy((a.astype(np.uint32) << 16).view(np.float32).copy())
    if dt == "F8_E4M3":
        s = f32(name + "_scale_inv").numpy()
        w = LUT[a]
        Mb, Kb = s.shape
        w = w.reshape(Mb, w.shape[0] // Mb, Kb, w.shape[1] // Kb) * s[:, None, :, None]
        return torch.from_numpy(w.reshape(a.shape).astype(np.float32))
    return torch.from_numpy(np.asarray(a).copy())


class LazyExperts(torch.nn.Module):
    """Experts dequantized on demand, tokens grouped per expert, nothing cached."""

    def __init__(self, config, layer):
        super().__init__()
        self.layer, self.I = layer, config.moe_intermediate_size
        self.act_fn = torch.nn.functional.silu

    def forward(self, hidden_states, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden_states)
        for e in torch.unique(top_k_index).tolist():
            tok, slot = torch.nonzero(top_k_index == e, as_tuple=True)
            p = f"{self.layer}.mlp.experts.{e}."
            w12 = torch.cat([f32(p + "gate_proj.weight"), f32(p + "up_proj.weight")])
            w3 = f32(p + "down_proj.weight")
            g, u = (hidden_states[tok] @ w12.T).chunk(2, dim=-1)
            y = (self.act_fn(g) * u) @ w3.T
            out.index_add_(0, tok, y * top_k_weights[tok, slot, None])
        return out


pre = "model.language_model."


class _Emb(torch.nn.Module):
    def __init__(self, *a, **k):
        super().__init__()
        self.weight = torch.zeros(1)
        self.padding_idx = None


_real_emb = torch.nn.Embedding
M.Qwen4ExpPreTrainedModel._init_weights = lambda self, m: (
    None
)  # weights come from the checkpoint
torch.nn.Embedding = _Emb  # no 320M-row table allocation
_real_experts = M.Qwen4ExpTextExperts
M.Qwen4ExpTextExperts = lambda config: torch.nn.Module()
model = M.Qwen4ExpTextModel(cfg)
torch.nn.Embedding = _real_emb
M.Qwen4ExpTextExperts = _real_experts
model.embed_tokens = torch.nn.Embedding.from_pretrained(
    f32(pre + "embed_tokens.weight")
)
for i, layer in enumerate(model.layers):
    layer.mlp.experts = LazyExperts(cfg, f"{pre}layers.{i}")
    if layer.ple is not None:
        q = f"{pre}layers.{i}.ple.ple_embedding."
        ne = layer.ple.ple_embedding
        shards = sorted(
            (k for k in idx if k.startswith(q + "ngram_embedding.shard_")),
            key=lambda k: int(k.rsplit("_", 1)[1].split(".")[0]),
        )
        D = 160
        scale = float(f32(q + "ngram_embedding.weight_scale")[0])
        if NO_TABLE:

            class Rows(torch.nn.Module):
                weight = torch.zeros(1)

                def forward(self, rid):
                    return torch.zeros(*rid.shape, D, dtype=torch.float32)
        else:
            sizes = [raw(k)[0].shape[0] for k in shards]
            starts = np.cumsum([0] + sizes)
            D = raw(shards[0])[0].shape[1]

            class Rows(torch.nn.Module):
                weight = torch.zeros(1)

                def forward(self, rid):
                    r = rid.reshape(-1).numpy()
                    o = np.empty((len(r), D), np.float32)
                    for n, v in enumerate(r):
                        s = int(np.searchsorted(starts, v, side="right") - 1)
                        o[n] = LUT[raw(shards[s])[0][v - starts[s]]] * scale
                    return torch.from_numpy(o).reshape(*rid.shape, -1)

        ne.ngram_embedding = Rows()
sd = model.state_dict()
missing = []
for k in list(sd.keys()):
    if "experts" in k or "ngram_embedding" in k or k.startswith("embed_tokens"):
        continue
    hk = pre + k
    if hk not in idx:
        missing.append(k)
        continue
    t = f32(hk)
    if k.endswith(
        ("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes")
    ):
        assert torch.equal(t.long(), sd[k].long()), k
        continue
    assert t.shape == sd[k].shape, (k, t.shape, sd[k].shape)
    sd[k].copy_(t.to(sd[k].dtype))
print("not in checkpoint:", missing)
lm_head = f32("lm_head.weight")
model.eval()
acts = {}
for i, layer in enumerate(model.layers):
    layer.register_forward_hook(
        lambda m, a, o, i=i: acts.__setitem__(
            f"x{i}", (o[0] if isinstance(o, tuple) else o)[0].float().numpy().copy()
        )
    )
with torch.no_grad():
    o = model(input_ids=torch.tensor([ids]), use_cache=False)
    h = o.last_hidden_state[0]
    logits = (h @ lm_head.T).float().numpy()
np.savez(out, logits=logits, **acts)
print("saved", out, logits.shape, "argmax", logits.argmax(-1)[-8:].tolist())
