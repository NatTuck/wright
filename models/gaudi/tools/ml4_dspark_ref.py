"""CPU PyTorch reference of Mistral-Large-4's DSpark speculative-decoding drafter.

Computes the drafter's 3 greedy drafts from the target's hidden states: the outputs of target layers
1, 18, 36, 54 and 71 are projected and injected into the drafter's KV cache at every context position,
one non-causal block (the anchor and two mask slots) runs through the drafter's MLA layers, and the
RNN Markov head adds its bias to the softcapped LM-head logits. ``simulate_acceptance`` replays a
greedy sequence and returns how many drafts the target accepts at each step.

The algorithm follows the Mistral-Large-4 drafter in llama.cpp (``conversion/mistral4.py``,
``src/models/dflash.cpp``, ``common/speculative.cpp``) and the generic DSpark code in SGLang
(``srt/models/dspark.py``, ``srt/models/dflash.py``).

  CKPT=Mistral-Large-4-1T-A52B-NVFP4 python models/gaudi/tools/ml4_dspark_ref.py

runs a smoke test on random hidden states (``SMOKE_LEN`` context positions, default 64).
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

CKPT_DIR = Path(os.environ.get("CKPT", "Mistral-Large-4-1T-A52B-NVFP4"))
DRAFT_SHARD = "consolidated-00032-of-00032.safetensors"
PREFIX = "mtp_dspark."


# ----------------------------------------------------------------------------- config
@dataclass(frozen=True)
class DSparkConfig:
    hidden: int
    n_heads: int
    qk_nope: int
    qk_rope: int
    v_head: int
    kv_lora: int
    vocab: int
    eps: float
    rope_theta: float
    yarn_factor: float
    yarn_orig: int
    yarn_beta_fast: float
    yarn_beta_slow: float
    softcap: float
    n_layers: int
    depth: int
    window: int
    target_layers: tuple[int, ...]
    markov_rank: int

    @staticmethod
    def from_params(path: Path) -> DSparkConfig:
        p = json.loads(path.read_text())
        mtp = p["mtp"]
        ds = mtp["dspark"]
        if ds["head_type"] != "rnn" or ds["draft_input"] != "position_specific":
            raise ValueError(f"unsupported DSpark config {ds}")
        windows = {int(w) for w in mtp["ragged_attention"].split(",")}
        if len(windows) != 1:
            raise ValueError(f"per-layer windows differ: {mtp['ragged_attention']}")
        yarn = p["yarn"]
        if yarn.get("apply_scale", False):
            raise ValueError("YaRN magnitude scaling is not implemented")
        return DSparkConfig(
            hidden=p["dim"],
            n_heads=p["n_heads"],
            qk_nope=p["qk_nope_head_dim"],
            qk_rope=p["qk_rope_head_dim"],
            v_head=p["v_head_dim"],
            kv_lora=p["kv_lora_rank"],
            vocab=p["vocab_size"],
            eps=float(p["norm_eps"]),
            rope_theta=float(p["rope_theta"]),
            yarn_factor=float(yarn["factor"]),
            yarn_orig=int(yarn["original_max_position_embeddings"]),
            yarn_beta_fast=float(yarn["beta"]),
            yarn_beta_slow=float(yarn["alpha"]),
            softcap=float(p["softmax_tanh_gating"]),
            n_layers=int(mtp["n_layers"]),
            depth=int(mtp["depth"]),
            window=windows.pop(),
            target_layers=tuple(int(v) for v in mtp["target_activation_layers"].split(",")),
            markov_rank=int(ds["markov_rank"]),
        )


def yarn_inv_freq(cfg: DSparkConfig) -> torch.Tensor:
    """(qk_rope/2,) f32 inverse frequencies: transformers/ggml YaRN with the truncated
    correction range, no magnitude scaling, no LeRoPE."""
    dim, base = cfg.qk_rope, cfg.rope_theta
    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    extra, inter = 1.0 / pos_freqs, 1.0 / (cfg.yarn_factor * pos_freqs)

    def correction(rot: float) -> float:
        return dim * math.log(cfg.yarn_orig / (rot * 2 * math.pi)) / (2 * math.log(base))

    low = max(math.floor(correction(cfg.yarn_beta_fast)), 0)
    high = min(math.ceil(correction(cfg.yarn_beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
    extra_mask = 1.0 - ramp  # 1 -> keep the original (extrapolated) frequency
    return (inter * (1 - extra_mask) + extra * extra_mask).float()


def rope_interleaved(x: torch.Tensor, pos: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent pairs (x[2i], x[2i+1]) by pos * inv_freq[i]. x: (T, ..., D), pos: (T,)."""
    ang = pos.to(torch.float32)[:, None] * inv_freq[None, :]  # (T, D/2)
    shape = (x.shape[0],) + (1,) * (x.dim() - 2) + (ang.shape[1],)
    c, s = torch.cos(ang).view(shape), torch.sin(ang).view(shape)
    x0, x1 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x0 * c - x1 * s
    out[..., 1::2] = x1 * c + x0 * s
    return out


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


# ----------------------------------------------------------------------------- weights
@dataclass
class Layer:
    attn_norm: torch.Tensor
    wq: torch.Tensor  # (H*(nope+rope), hidden)
    wkv_a: torch.Tensor  # (kv_lora+rope, hidden)
    kv_a_norm: torch.Tensor
    wk_b: torch.Tensor  # (H, nope, kv_lora)
    wv_b: torch.Tensor  # (H, v_head, kv_lora)
    wo: torch.Tensor  # (hidden, H*v_head)
    ffn_norm: torch.Tensor
    w1: torch.Tensor
    w3: torch.Tensor
    w2: torch.Tensor


@dataclass
class DraftContext:
    """The drafter's KV cache over one sequence: per layer, the injected latent (kv_a_norm'd)
    and the roped key part for positions 0 .. n-1 (positions below ``start`` are not filled)."""

    latent: list[torch.Tensor] = field(default_factory=list)  # per layer (cap, kv_lora)
    krope: list[torch.Tensor] = field(default_factory=list)  # per layer (cap, rope)
    start: int = 0
    n: int = 0


class DSparkRef:
    """Greedy reference of Mistral-Large-4's DSpark drafter (see the module docstring)."""

    def __init__(
        self,
        ckpt_dir: Path | str = CKPT_DIR,
        *,
        weight_dtype: torch.dtype = torch.float32,
        window: int | None = None,
        softcap_base: bool = True,
        chunk: int = 256,
    ) -> None:
        ckpt_dir = Path(ckpt_dir)
        self.cfg = cfg = DSparkConfig.from_params(ckpt_dir / "params.json")
        self.window = cfg.window if window is None else window
        self.softcap_base = softcap_base
        self.chunk = chunk
        self.wdt = weight_dtype
        self.inv_freq = yarn_inv_freq(cfg)
        self.scale = 1.0 / math.sqrt(cfg.qk_nope + cfg.qk_rope)

        H, nope, vh = cfg.n_heads, cfg.qk_nope, cfg.v_head
        with safe_open(str(ckpt_dir / DRAFT_SHARD), framework="pt") as f:

            def get(name: str, dtype: torch.dtype | None = None) -> torch.Tensor:
                t = f.get_tensor(PREFIX + name)
                return t.to(dtype or self.wdt).contiguous()

            def vec(name: str) -> torch.Tensor:
                return get(name, torch.float32)

            self.layers: list[Layer] = []
            for i in range(cfg.n_layers):
                p = f"backbone.layers.{i}."
                kv_b = get(p + "attention.wkv_b.weight").view(H, nope + vh, cfg.kv_lora)
                self.layers.append(
                    Layer(
                        attn_norm=vec(p + "attention_norm.weight"),
                        wq=get(p + "attention.wq.weight"),
                        wkv_a=get(p + "attention.wkv_a_with_mqa.weight"),
                        kv_a_norm=vec(p + "attention.kv_a_norm.weight"),
                        wk_b=kv_b[:, :nope].contiguous(),
                        wv_b=kv_b[:, nope:].contiguous(),
                        wo=get(p + "attention.wo.weight"),
                        ffn_norm=vec(p + "ffn_norm.weight"),
                        w1=get(p + "feed_forward.w1.weight"),
                        w3=get(p + "feed_forward.w3.weight"),
                        w2=get(p + "feed_forward.w2.weight"),
                    )
                )
            self.target_projection = get("backbone.target_projection.weight")
            self.target_norm = vec("backbone.target_norm.weight")
            self.final_norm = vec("backbone.final_norm.weight")
            self.mask_embedding = vec("backbone.mask_embedding")  # (depth-1, hidden)
            self.markov_emb = f.get_tensor(PREFIX + "autoregressive_head.embedding.weight")  # bf16 gather table
            self.markov_out = get("autoregressive_head.output.weight")
            self.joint_w = get("autoregressive_head.joint_proj.weight")
            self.joint_b = vec("autoregressive_head.joint_proj.bias")
        # the target's embedding and LM head (bf16 in the release's first shard)
        index = json.loads((ckpt_dir / "consolidated.safetensors.index.json").read_text())
        wm = index["weight_map"]
        with safe_open(str(ckpt_dir / wm["tok_embeddings.weight"]), framework="pt") as f:
            self.tok_emb = f.get_tensor("tok_embeddings.weight")  # bf16 gather table
        with safe_open(str(ckpt_dir / wm["output.weight"]), framework="pt") as f:
            self.lm_head = f.get_tensor("output.weight").to(self.wdt).contiguous()
        if self.mask_embedding.shape[0] != cfg.depth - 1:
            raise ValueError("mask_embedding rows != depth - 1")
        if self.target_projection.shape[1] != len(cfg.target_layers) * cfg.hidden:
            raise ValueError("target_projection width != 5 * hidden")

    # ------------------------------------------------------------------ helpers
    def _mm(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """x @ w.T in float32 (w is stored in ``weight_dtype``)."""
        if w.dtype == torch.float32:
            return x.float() @ w.T
        return (x.to(w.dtype) @ w.T).float()

    # ------------------------------------------------------------------ context (KV injection)
    def new_context(self, capacity: int, start: int = 0) -> DraftContext:
        c = self.cfg
        ctx = DraftContext(start=start, n=start)
        for _ in self.layers:
            ctx.latent.append(torch.zeros(capacity, c.kv_lora))
            ctx.krope.append(torch.zeros(capacity, c.qk_rope))
        return ctx

    def inject(self, ctx: DraftContext, hidden: np.ndarray | torch.Tensor, upto: int) -> None:
        """Fill ``ctx`` for positions ctx.n .. upto-1 from ``hidden`` (5, >=upto, hidden)."""
        c = self.cfg
        if upto > ctx.latent[0].shape[0]:
            grow = max(upto, 2 * ctx.latent[0].shape[0])
            for i in range(len(self.layers)):
                for buf in (ctx.latent, ctx.krope):
                    nb = torch.zeros(grow, buf[i].shape[1])
                    nb[: buf[i].shape[0]] = buf[i]
                    buf[i] = nb
        hid = torch.as_tensor(hidden)
        if hid.dim() != 3 or hid.shape[0] != len(c.target_layers) or hid.shape[2] != c.hidden:
            raise ValueError(f"hidden must be (5, L, {c.hidden}), got {tuple(hid.shape)}")
        for t0 in range(ctx.n, upto, self.chunk):
            t1 = min(upto, t0 + self.chunk)
            feats = hid[:, t0:t1].float().permute(1, 0, 2).reshape(t1 - t0, -1)  # [h1|h18|h36|h54|h71]
            g = rms_norm(self._mm(feats, self.target_projection), self.target_norm, c.eps)
            pos = torch.arange(t0, t1)
            for i, L in enumerate(self.layers):
                kv = self._mm(g, L.wkv_a)
                ctx.latent[i][t0:t1] = rms_norm(kv[:, : c.kv_lora], L.kv_a_norm, c.eps)
                ctx.krope[i][t0:t1] = rope_interleaved(kv[:, c.kv_lora :], pos, self.inv_freq)
        ctx.n = max(ctx.n, upto)

    # ------------------------------------------------------------------ one block
    @torch.no_grad()
    def draft_block(self, ctx: DraftContext, anchor: int, anchor_pos: int) -> dict:
        """Draft ``depth`` tokens after ``anchor`` at ``anchor_pos``; ``ctx`` must hold the
        injected context for positions max(0, anchor_pos - window) .. anchor_pos - 1.

        Returns tokens (list[int]), the softcapped base logits (depth, V), the step logits
        (depth, V) and the final-normed slot hiddens (depth, hidden)."""
        c = self.cfg
        p = anchor_pos
        lo = 0 if self.window is None else max(0, p - self.window)
        if ctx.n < p or ctx.start > lo:
            raise ValueError(f"context covers [{ctx.start}, {ctx.n}), need [{lo}, {p})")
        D = c.depth
        H, nope, rope = c.n_heads, c.qk_nope, c.qk_rope
        pos = torch.arange(p, p + D)

        x = torch.cat([self.tok_emb[anchor : anchor + 1].float(), self.mask_embedding], 0)  # (D, hidden)
        for i, L in enumerate(self.layers):
            h = rms_norm(x, L.attn_norm, c.eps)
            q = self._mm(h, L.wq).view(D, H, nope + rope)
            q_nope = q[..., :nope]
            q_pe = rope_interleaved(q[..., nope:], pos, self.inv_freq)
            kv = self._mm(h, L.wkv_a)
            lat_b = rms_norm(kv[:, : c.kv_lora], L.kv_a_norm, c.eps)
            kr_b = rope_interleaved(kv[:, c.kv_lora :], pos, self.inv_freq)
            K_lat = torch.cat([ctx.latent[i][lo:p], lat_b], 0)  # (W+D, kv_lora)
            K_pe = torch.cat([ctx.krope[i][lo:p], kr_b], 0)  # (W+D, rope)
            q_abs = torch.einsum("thd,hdr->thr", q_nope, L.wk_b.float())  # absorbed k_nope
            s = (torch.einsum("thr,kr->thk", q_abs, K_lat) + torch.einsum("thd,kd->thk", q_pe, K_pe)) * self.scale
            a = torch.softmax(s, dim=-1)  # non-causal: every slot sees the window and the block
            o_lat = torch.einsum("thk,kr->thr", a, K_lat)
            o = torch.einsum("thr,hvr->thv", o_lat, L.wv_b.float()).reshape(D, H * c.v_head)
            x = x + self._mm(o, L.wo)
            h2 = rms_norm(x, L.ffn_norm, c.eps)
            x = x + self._mm(F.silu(self._mm(h2, L.w1)) * self._mm(h2, L.w3), L.w2)
        hf = rms_norm(x, self.final_norm, c.eps)

        base = self._mm(hf, self.lm_head)
        if self.softcap_base and c.softcap:
            base = c.softcap * torch.tanh(base / c.softcap)

        state = torch.zeros(c.markov_rank)
        prev = int(anchor)
        tokens: list[int] = []
        steps = []
        R = c.markov_rank
        for k in range(D):
            z = torch.cat([state, self.markov_emb[prev].float(), hf[k]])
            j = self._mm(z[None], self.joint_w)[0] + self.joint_b
            gate, cand, out = torch.sigmoid(j[:R]), torch.tanh(j[R : 2 * R]), torch.tanh(j[2 * R :])
            bias = self._mm(out[None], self.markov_out)[0]
            state = gate * state + (1.0 - gate) * cand
            logits = base[k] + bias
            prev = int(torch.argmax(logits))
            tokens.append(prev)
            steps.append(logits)
        return {"tokens": tokens, "base_logits": base, "step_logits": torch.stack(steps), "hidden": hf}

    # ------------------------------------------------------------------ public API
    def draft(
        self,
        hidden: np.ndarray | torch.Tensor,
        context_tokens: Sequence[int] | None,
        last_token: int,
        *,
        ctx: DraftContext | None = None,
    ) -> list[int]:
        """3 greedy drafts after ``last_token``.

        hidden: (5, L, 6144) target taps (layer outputs 1, 18, 36, 54, 71, un-normalized) for
        the L context positions 0 .. L-1. ``last_token`` is the anchor at position L (not yet
        processed by the target). ``context_tokens`` (length L) is not consumed by the
        algorithm (the drafter's context is the target taps only); it is checked for length.
        ``ctx``: optional cache reused across calls (positions already injected are kept).
        """
        L = int(hidden.shape[1])
        if context_tokens is not None and len(context_tokens) != L:
            raise ValueError(f"{len(context_tokens)} context tokens for {L} hidden positions")
        if ctx is None:
            lo = 0 if self.window is None else max(0, L - self.window)
            ctx = self.new_context(L + 1, start=lo)
        self.inject(ctx, hidden, L)
        return self.draft_block(ctx, int(last_token), L)["tokens"]


def accepted_prefix(drafts: Sequence[int], targets: Sequence[int]) -> int:
    """Greedy accepted-block rule: the longest prefix of ``drafts`` equal to ``targets``."""
    n = 0
    for d, t in zip(drafts, targets):
        if int(d) != int(t):
            break
        n += 1
    return n


def simulate_acceptance(
    ref: DSparkRef,
    hidden: np.ndarray | torch.Tensor,
    tokens: Sequence[int],
    prompt_len: int,
    *,
    record: list | None = None,
) -> list[int]:
    """Replay speculative decoding over an already generated greedy sequence.

    ``tokens``: prompt + generated tokens. ``hidden``: (5, len(tokens), 6144) target taps for
    every position. The first token after the prompt, ``tokens[prompt_len]``, comes from
    prefill. Each step's anchor is the last committed token ``tokens[n-1]`` at position n-1,
    and the drafter's context is the taps at positions 0 .. n-2. The 3 drafts are compared
    with ``tokens[n : n+3]``, then the replay advances by accepted + 1 (the target's bonus
    token). Returns the per-step accepted counts (0..3; fewer than 3 targets remain at the end
    of the sequence). ``record``, if given, receives (anchor_pos, drafts, targets) per step.
    """
    T = len(tokens)
    if torch.as_tensor(hidden).shape[1] < T - 1:
        raise ValueError("hidden must cover every position of tokens")
    ctx = ref.new_context(max(T, 1))
    accepted: list[int] = []
    n = prompt_len + 1
    while n < T:
        anchor_pos = n - 1
        ref.inject(ctx, hidden, anchor_pos)
        drafts = ref.draft_block(ctx, int(tokens[anchor_pos]), anchor_pos)["tokens"]
        targets = [int(t) for t in tokens[n : n + ref.cfg.depth]]
        a = accepted_prefix(drafts, targets)
        accepted.append(a)
        if record is not None:
            record.append((anchor_pos, drafts, targets))
        n += a + 1
    return accepted


# ----------------------------------------------------------------------------- smoke test
def _main() -> None:
    torch.manual_seed(0)
    t0 = time.perf_counter()
    ref = DSparkRef()
    print(f"loaded drafter + target embedding/LM head in {time.perf_counter() - t0:.1f} s; cfg={ref.cfg}")
    c = ref.cfg
    print(f"yarn inv_freq[:6]={ref.inv_freq[:6].tolist()}  window={ref.window}")

    L = int(os.environ.get("SMOKE_LEN", "64"))
    # residual-stream-like scale: the ML4 residual grows with depth; random is only a smoke input
    scales = torch.tensor([0.5, 2.0, 4.0, 8.0, 16.0])[:, None, None]
    hidden = (torch.randn(len(c.target_layers), L + 8, c.hidden) * scales).numpy().astype(np.float32)
    tokens = torch.randint(1000, c.vocab, (L + 8,)).tolist()

    for rep in range(2):
        t = time.perf_counter()
        d = ref.draft(hidden[:, :L], tokens[:L], tokens[L])
        print(f"draft (fresh context, L={L}) -> {d}  {1e3 * (time.perf_counter() - t):.0f} ms")

    ctx = ref.new_context(L + 8)
    ref.inject(ctx, hidden, L)
    t = time.perf_counter()
    out = ref.draft_block(ctx, tokens[L], L)
    print(f"draft_block (cached context) -> {out['tokens']}  {1e3 * (time.perf_counter() - t):.0f} ms")
    assert out["tokens"] == d, "cached and fresh contexts disagree"
    top = torch.topk(out["step_logits"], 3, dim=-1)
    print("step top-3 logits:", [[round(v, 3) for v in row] for row in top.values.tolist()])

    t = time.perf_counter()
    acc = simulate_acceptance(ref, hidden, tokens, prompt_len=L - 8)
    print(f"simulate_acceptance on random data: {acc} ({1e3 * (time.perf_counter() - t):.0f} ms)")


if __name__ == "__main__":
    _main()
