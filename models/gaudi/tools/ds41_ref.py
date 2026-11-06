"""CPU reference for DeepSeek-V4.1-Flash: the checkpoint's own inference/model.py, run on CPU.

usage: ds41_ref.py CHECKPOINT OUT.npz [--mode math|quant|gaudi] [--layers N] [--decode K]
                   [--dump L] [--last M] (--ids-file FILE | --text TEXT | IDS...)

The reference's tilelang kernels (kernel.py) are replaced by torch versions. The model is built on
the meta device, and its parameters are memory maps of the checkpoint shards, so only the experts
and Engram rows that a prompt touches are read. The run covers the first N layers (no vision, no
DSpark), the final collapse, the norm and the head, then K greedy decode steps.

Modes:
- math: f32 activations and exact weights, no activation quantization.
- quant: the reference's numerics: bf16 activations, FP8 (per-32 UE8M0) activations into the FP8
  and FP4 GEMMs, FP8 window KV, FP4 compressed KV and FP4 indexer keys and queries.
- quant32: quant's roundings with f32 activations in place of bf16.
- gaudi: quant, then each stored KV row (window, compressed, index keys) rounded to E4M3 with
  bias 7 (+-240, subnormals to 2^-9) and one power-of-two scale per row, as the Gaudi cache
  stores it.

Saved (npz): the logits of the last M prompt positions and of each decode step; the argmax of every
prompt position; the greedy tokens; x{i}, the streams (P, 4, 5120) after layer i, and p{i}, the
pre-mix (P, 4) after layer i (full for i < L, the last M positions otherwise); for layers < L,
l{i}.attn_out, l{i}.ffn_out, l{i}.router_w, l{i}.router_ids, l{i}.engram_out, l{i}.topk and
l{i}.latent; buffers of the attention (its caches), compressor and indexer of every KV owner below
N, after the prefill, as buf.l{i}.<module>.<name>.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import types
import warnings

import numpy as np
import torch

# ------------------------------------------------------------------ arguments
parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
parser.add_argument("checkpoint")
parser.add_argument("out")
parser.add_argument(
    "--mode", choices=("math", "quant", "quant32", "gaudi"), default="quant"
)
parser.add_argument("--layers", type=int, default=40)
parser.add_argument("--decode", type=int, default=8)
parser.add_argument("--dump", type=int, default=4)
parser.add_argument("--last", type=int, default=64)
parser.add_argument("--ids-file")
parser.add_argument("--text")
parser.add_argument("ids", nargs="*", type=int)
args = parser.parse_args()
torch.set_num_threads(int(os.environ.get("REF_THREADS", 32)))
QUANT = args.mode != "math"
GAUDI = args.mode == "gaudi"
F32 = args.mode in (
    "math",
    "quant32",
)  # f32 activations and f32 copies of the bf16 weights
torch.set_default_dtype(torch.float32 if F32 else torch.bfloat16)
INFERENCE = os.path.join(args.checkpoint, "inference")

# ------------------------------------------------------------------ kernel.py in torch
FP4_TABLE = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
)
FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _store_lines() -> set[int]:
    """Lines of model.py whose in-place quantization rounds a value that goes into a cache."""
    marks = ("act_quant(kv,", "fp4_act_quant(latent,", "fp4_act_quant(k,")
    with open(os.path.join(INFERENCE, "model.py")) as f:
        return {n for n, line in enumerate(f, 1) if any(m in line for m in marks)}


STORE_LINES = _store_lines()


def gaudi_row(x: torch.Tensor) -> torch.Tensor:
    """Round each row to E4M3 bias 7 (+-240, subnormals to 2^-9) with a power-of-two row scale."""
    xf = x.float()
    amax = xf.abs().amax(-1, keepdim=True)
    scale = torch.exp2(torch.ceil(torch.log2(torch.clamp(amax, min=2.0**-120) / 240.0)))
    y = xf / scale
    e = torch.floor(torch.log2(y.abs().clamp_min(2.0**-30))).clamp(-6, 7)
    quantum = torch.exp2(e - 3)
    q = (torch.round(y / quantum) * quantum).clamp(-240.0, 240.0)
    return q * scale


def _stored() -> bool:
    caller = sys._getframe(2)
    return (
        GAUDI
        and caller.f_code.co_filename.endswith("model.py")
        and caller.f_lineno in STORE_LINES
    )


def _pow2_ceil(v):
    return torch.exp2(torch.ceil(torch.log2(v)))


def act_quant(
    x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False
):
    """FP8 (OCP E4M3, +-448) per-block quantization; identity in math mode."""
    if not QUANT:
        return x if inplace else (x, None)
    shape = x.shape
    xf = x.float().unflatten(-1, (-1, block_size))
    s = xf.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448.0
    if scale_fmt is not None:
        s = _pow2_ceil(s)
    y = (xf / s).clamp(-448, 448).to(torch.float8_e4m3fn)
    if inplace:
        v = (y.float() * s).reshape(shape)
        x.copy_(gaudi_row(v) if _stored() else v)
        return x
    return y.reshape(shape), s.squeeze(-1)


def _fp4_round(v):
    """The nearest E2M1 value, ties to the even code (as the reference kernel's cast)."""
    av = v.abs()
    d = (av.unsqueeze(-1) - FP4_GRID).abs()
    best = d.min(-1, keepdim=True).values
    code = d.argmin(-1)
    tie = (d == best).sum(-1) > 1
    code = torch.where(tie & (code % 2 == 1), code + 1, code)
    return torch.copysign(FP4_GRID[code.clamp(max=7)], v)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    if not QUANT:
        return x
    assert inplace
    shape = x.shape
    xf = x.float().unflatten(-1, (-1, block_size))
    am = xf.abs().amax(-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        s = (am.clamp_min(6 * 2.0**-9) / 6.0).to(torch.float8_e4m3fn).float()
    else:
        s = _pow2_ceil(am.clamp_min(6 * 2.0**-126) / 6.0)
    v = (_fp4_round((xf / s).clamp(-6, 6)) * s).reshape(shape)
    x.copy_(gaudi_row(v) if _stored() else v)
    return x


def _deq_fp8(w, ws, block=32):
    """[N, K] FP8 with [N/32, K/32] E8M0 -> f32."""
    n, k = w.shape
    s = ws.float().repeat_interleave(block, 0)[:n].repeat_interleave(block, 1)[:, :k]
    return w.float() * s


def _deq_fp4(w, ws):
    """[N, K/2] packed E2M1, low nibble first, with [N, K/32] E8M0 -> f32 [N, K]."""
    u = w.view(torch.uint8)
    v = torch.stack(
        [FP4_TABLE[(u & 0x0F).long()], FP4_TABLE[(u >> 4).long()]], -1
    ).flatten(-2)
    return v * ws.float().repeat_interleave(32, 1)


def _act(x, s, block):
    return (
        x.float() if s is None else x.float() * s.float().repeat_interleave(block, -1)
    )


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    return (_act(a, a_s, block_size) @ _deq_fp8(b, b_s, block_size).t()).to(
        torch.get_default_dtype()
    )


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    return (_act(a, a_s, act_block_size) @ _deq_fp4(b, b_s).t()).to(
        torch.get_default_dtype()
    )


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Softmax over the gathered keys plus the per-head sink, which has no value."""
    b = q.shape[0]
    i = topk_idxs.long()
    valid = i >= 0
    g = kv.float()[torch.arange(b)[:, None, None], i.clamp_min(0)]
    sc = torch.einsum("bshd,bskd->bshk", q.float(), g) * softmax_scale
    sc = sc.masked_fill(~valid[:, :, None, :], float("-inf"))
    m = torch.maximum(sc.amax(-1, keepdim=True), torch.full_like(sc[..., :1], -1e30))
    e = torch.exp(sc - m)
    den = e.sum(-1, keepdim=True) + torch.exp(
        attn_sink.float()[None, None, :, None] - m
    )
    return torch.einsum("bshk,bskd->bshd", e / den, g).to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    mixes, hc_scale, hc_base = mixes.float(), hc_scale.float(), hc_base.float()
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(
        mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc]
    )
    comb = (mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).unflatten(
        -1, (hc, hc)
    )
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


kernel = types.ModuleType("kernel")
for fn in (
    act_quant,
    fp4_act_quant,
    fp4_gemm,
    fp8_gemm,
    hc_split_sinkhorn,
    sparse_attn,
):
    setattr(kernel, fn.__name__, fn)
sys.modules["kernel"] = kernel
sys.path.insert(0, INFERENCE)
import model as M  # noqa: E402

# ------------------------------------------------------------------ checkpoint, memory-mapped
weight_map = json.load(
    open(os.path.join(args.checkpoint, "model.safetensors.index.json"))
)["weight_map"]
_headers: dict[str, tuple[dict, int]] = {}
_maps: dict[str, np.memmap] = {}
_NP = {"BF16": np.uint16, "F32": np.float32, "F8_E4M3": np.uint8, "F8_E8M0": np.uint8,
       "I8": np.uint8, "U8": np.uint8, "F16": np.float16, "I64": np.int64, "I32": np.int32}  # fmt: skip
_TORCH = {"BF16": torch.bfloat16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn,
          "F8_E8M0": torch.float8_e8m0fnu, "I8": torch.float4_e2m1fn_x2, "U8": torch.uint8}  # fmt: skip


def raw(name: str) -> torch.Tensor:
    """A zero-copy torch view of a checkpoint tensor (I8 is packed FP4 pairs)."""
    path = os.path.join(args.checkpoint, weight_map[name])
    if path not in _headers:
        with open(path, "rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            _headers[path] = (json.loads(f.read(size)), 8 + size)
        _maps[path] = np.memmap(path, np.uint8, "r")
    header, base = _headers[path]
    info = header[name]
    start, end = info["data_offsets"]
    array = (
        _maps[path][base + start : base + end]
        .view(_NP[info["dtype"]])
        .reshape(info["shape"])
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = torch.from_numpy(array)
    dtype = _TORCH[info["dtype"]]
    return (
        t.view(dtype) if dtype.itemsize == t.element_size() and dtype != t.dtype else t
    )


class _Tokenizer:
    """What the reference's Engram hash reads from a tokenizer."""

    def __init__(self, path: str) -> None:
        from tokenizers import Tokenizer

        self.backend_tokenizer = Tokenizer.from_file(path)

    def __len__(self) -> int:
        return self.backend_tokenizer.get_vocab_size(with_added_tokens=True)


tokenizer = _Tokenizer(os.path.join(args.checkpoint, "tokenizer.json"))
if args.ids_file:
    ids = [int(t) for t in open(args.ids_file).read().split()]
elif args.text is not None:
    ids = tokenizer.backend_tokenizer.encode(args.text, add_special_tokens=False).ids
else:
    ids = args.ids
if not ids:
    raise SystemExit("no prompt: give IDS, --ids-file or --text")

config = json.load(open(os.path.join(INFERENCE, "config.json")))
config.update(
    n_layers=args.layers, n_mtp_layers=0, dspark_block_size=0, vision_n_layers=0,
    max_batch_size=1, max_seq_len=max(256, (len(ids) + args.decode + 255) // 256 * 256),
)  # fmt: skip
model_args = M.ModelArgs(
    **{k: tuple(v) if isinstance(v, list) else v for k, v in config.items()
       if k in M.ModelArgs.__dataclass_fields__}
)  # fmt: skip

# The Engram hash state and the rope tables are built on the CPU, everything else on meta.
M.precompute_freqs_cis.cache_clear()
_hash_state = M.NgramHashState
M.NgramHashState = lambda *a, **k: None
with torch.device("meta"):
    model = M.Transformer(model_args)
M.NgramHashState = _hash_state
M.precompute_freqs_cis.cache_clear()
layout = M.EngramLayout.from_args(model_args)
model.engram_hash = (
    M.NgramHashState(model_args, layout, tokenizer) if layout is not None else None
)

state = {}
for name, p in model.named_parameters():
    t = raw(name)
    if p.dtype in (torch.float8_e4m3fn, torch.float4_e2m1fn_x2, torch.float8_e8m0fnu):
        assert t.dtype == p.dtype, (name, t.dtype, p.dtype)
    elif name.endswith("wo_a.weight"):  # FP8 in the checkpoint, bf16 in the model
        t = _deq_fp8(t, raw(name.replace("weight", "scale")))
        t = t.to(torch.float32 if F32 else p.dtype)
    elif F32 and t.is_floating_point():
        t = t.float()
    elif t.dtype != p.dtype:
        t = t.to(p.dtype)
    assert tuple(t.shape) == tuple(p.shape), (name, t.shape, p.shape)
    state[name] = t
model.load_state_dict(state, strict=True, assign=True)
for module in model.modules():
    if isinstance(module, M.Linear) and module.scale is not None:
        module.weight.scale = module.scale
for module in model.modules():
    for name, buf in list(module._buffers.items()):
        if buf is None or not buf.is_meta:
            continue
        if name == "freqs_cis":
            ratio = module.compress_ratio
            original, theta = (
                (model_args.original_seq_len, model_args.compress_rope_theta)
                if ratio
                else (0, model_args.rope_theta)
            )
            new = M.precompute_freqs_cis(
                model_args.rope_head_dim, model_args.max_seq_len, original, theta,
                model_args.rope_factor, model_args.beta_fast, model_args.beta_slow,
            )  # fmt: skip
        elif name == "score_state":
            new = torch.full(buf.shape, -torch.inf, dtype=buf.dtype)
        else:
            new = torch.zeros(buf.shape, dtype=buf.dtype)
        module._buffers[name] = new
model.eval()

# ------------------------------------------------------------------ dumps
dumps: dict[str, np.ndarray] = {}
recording = {"on": True}


def put(name: str, t: torch.Tensor) -> None:
    if recording["on"]:
        t = t.detach().float()
        dumps[name] = (t.reshape(-1, *t.shape[2:]) if t.dim() > 2 else t).numpy().copy()


def hook(fn):
    def call(module, inputs, output):
        fn(output)

    return call


for i, block in enumerate(model.layers):
    if i >= args.dump:
        continue
    block.attn.register_forward_hook(hook(lambda o, i=i: put(f"l{i}.attn_out", o)))
    block.ffn.register_forward_hook(hook(lambda o, i=i: put(f"l{i}.ffn_out", o)))
    block.ffn.gate.register_forward_hook(
        hook(
            lambda o, i=i: (put(f"l{i}.router_w", o[0]), put(f"l{i}.router_ids", o[1]))
        )
    )
    if block.engram is not None:
        block.engram.register_forward_hook(
            hook(lambda o, i=i: put(f"l{i}.engram_out", o))
        )
    if block.attn.indexer is not None:
        block.attn.indexer.register_forward_hook(
            hook(lambda o, i=i: put(f"l{i}.topk", o))
        )
    if block.attn.compressor is not None:
        block.attn.compressor.register_forward_hook(
            hook(lambda o, i=i: o is not None and put(f"l{i}.latent", o))
        )


@torch.inference_mode()
def forward(
    tokens: torch.Tensor, start: int, streams: dict | None = None
) -> torch.Tensor:
    """Transformer.forward with logits at every position, and the streams after each layer."""
    hashes = model.engram_hash(tokens, start) if model.engram_hash is not None else None
    h = model.embed(tokens).unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
    pre_mix = M.make_identity_pre_mix(h, model.hc_mult)
    for i, layer in enumerate(model.layers):
        if layer.engram is not None:
            h = layer.engram(h, hashes[:, :, layer.engram.layer_hash_index, :])
        h, pre_mix = layer(h, start, pre_mix, None)
        if streams is not None:
            rows = slice(None) if i < args.dump else slice(-args.last, None)
            streams[f"x{i}"] = h[0, rows].float().numpy().copy()
            streams[f"p{i}"] = pre_mix[0, rows].float().numpy().copy()
    h = layer.hc_pre(h, pre_mix)
    return model.head(model.norm(h), full_logits=True)[0].float()


prompt = torch.tensor([ids], dtype=torch.long)
streams: dict[str, np.ndarray] = {}
logits_prompt = forward(prompt, 0, streams)
print(
    f"prefill {len(ids)} tokens: argmax {int(logits_prompt[-1].argmax())}", flush=True
)
recording["on"] = False
for i, block in enumerate(model.layers):
    if i in model_args.kv_source_layers:
        for part in ("attn", "compressor", "indexer"):
            module = block.attn if part == "attn" else getattr(block.attn, part, None)
            for name, buf in module._buffers.items() if module is not None else ():
                if buf is not None and buf.dim() > 0:
                    dumps[f"buf.l{i}.{part}.{name}"] = buf.float().numpy().copy()
logits = [logits_prompt[-args.last :].numpy()]
argmax = logits_prompt.argmax(-1).numpy()
tokens = [int(logits_prompt[-1].argmax())]
for k in range(args.decode):
    step = forward(torch.tensor([[tokens[-1]]]), len(ids) + k)
    logits.append(step.numpy())
    tokens.append(int(step[-1].argmax()))
    print(f"decode {k}: token {tokens[-2]} -> {tokens[-1]}", flush=True)
np.savez(
    args.out, logits=np.concatenate(logits), argmax=argmax,
    tokens=np.array(tokens[: args.decode + 1]), last=args.last, prompt=np.array(ids),
    mode=args.mode, layers=args.layers, **streams, **dumps,
)  # fmt: skip
print("saved", args.out, "prompt", len(ids), "greedy", tokens[: args.decode + 1])
