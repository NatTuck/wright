"""CPU PyTorch reference of the DSpine drafter of Qwen3.8-27B (``tencent/Qwen3.8-27B-DSpine-b16``).

The drafter is five Qwen3 decoder layers at the target's width (hidden 5120, 32 query heads, 8 KV
heads of 128, full NeoX RoPE, SwiGLU 17408) run as a DFlash drafter, with three changes:

- **KV injection** as DFlash: the target's residual outputs after layers 1, 16, 31, 46 and 61
  (concatenated, 25600 wide) through ``fc`` and ``hidden_norm`` give one context feature per
  position; each layer keeps k_norm(k_proj) with RoPE and v_proj of it as the position's entry.
- **A causal block** of 16: the target's embeddings of [anchor, mask x 15] at p .. p + 15. Every
  query sees the block's keys up to its own position and the cache entries within 8192 positions
  (key >= query - 8191).
- **The code bridge.** After every layer, the residual stream ``raw`` gets a write from the row
  before it in the block: each row's code is the unit-RMS projection ``read_l(raw / rms(raw))``
  scaled to norm sqrt(C) (the anchor row's code is ``sqrt(C) * codebook[anchor]``); ``packed =
  [gate | predecessor_gate | prev] @ code``; row t > 0 gets ``message_t = sigmoid(q_t + s_{t-1} +
  bias) * v_{t-1}`` and ``raw_t += rms(raw_t) * write_l @ message_t``. The next layer's input norm
  (the final ``norm`` after the last layer) follows.
- **The serial refinement.** The target's LM head scores the 15 mask rows; the 16 largest logits
  of each are its candidates. A greedy walk from the anchor redoes the LAST write of each row with
  the exact code of the token chosen before it (``sqrt(C) * codebook[prev]`` in place of the
  predicted code) and picks the candidate with the largest refined logit.

The algorithm follows ``Tencent/DSpine`` (``sglang_dspine/models/dspine.py``: ``DSpineCodeBridge``,
``DSpineDraftModel``; ``sglang_dspine/kernels/dspine.py``: ``serial_table``, ``serial_chain``; the
plain-torch bridge in ``tests/ref_utils.py``). Math runs in float32 over the checkpoint's bf16
weights, so it tracks the bf16 original closely but not to the bit.

  DRAFT=/path/to/Qwen3.8-27B-DSpine-b16 TARGET=/path/to/Qwen3.8-27B-FP8 \\
      python models/gaudi/tools/qwen_dspine_ref.py

runs a smoke test on random context features (``SMOKE_LEN`` positions, default 64).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_dflash2_ref import (  # noqa: E402
    apply_rope,
    rms_norm,
    rope_cos_sin,
    target_embeddings,
)

DRAFT_DIR = Path(os.environ.get("DRAFT", "Qwen3.8-27B-DSpine-b16"))


def _e4m3_table() -> torch.Tensor:
    """The 256 e4m3 values with exponent bias 7 (codes 0x78 .. 0x7E unused at |x| <= 240)."""
    u = torch.arange(256)
    sign = torch.where(u & 0x80 > 0, -1.0, 1.0)
    e, m = (u >> 3) & 15, (u & 7).double()
    value = torch.where(e == 0, m / 8 * 2.0**-6, (1 + m / 8) * 2.0 ** (e.double() - 7))
    return (sign * value).float()


E4M3_TABLE = _e4m3_table()
TARGET_DIR = Path(os.environ.get("TARGET", "Qwen3.8-27B-FP8"))


# ----------------------------------------------------------------------------- config
@dataclass(frozen=True)
class DSpineConfig:
    hidden: int
    ffn: int
    vocab: int
    heads: int
    kv_heads: int
    head_dim: int
    layers: int
    eps: float
    rope_theta: float
    window: int
    block_size: int
    mask_token_id: int
    code_size: int
    rank: int
    bridge_eps: float
    head_topk: int
    target_layer_ids: tuple[int, ...]
    num_target_layers: int

    @staticmethod
    def from_json(path: Path) -> DSpineConfig:
        c = json.loads(path.read_text())
        d = c["dflash_config"]
        bridge = d.get("selected_prefix_code_bridge")
        if c.get("architectures") != ["DFlashDraftModel"] or not bridge:
            raise ValueError(f"{path}: not a DSpine drafter")
        if not (c.get("is_causal") and d.get("draft_block_causal")):
            raise ValueError("the reference implements causal DSpine blocks")
        return DSpineConfig(
            hidden=c["hidden_size"],
            ffn=c["intermediate_size"],
            vocab=c["vocab_size"],
            heads=c["num_attention_heads"],
            kv_heads=c["num_key_value_heads"],
            head_dim=c["head_dim"],
            layers=c["num_hidden_layers"],
            eps=float(c["rms_norm_eps"]),
            rope_theta=float(c["rope_parameters"]["rope_theta"]),
            window=int(c["sliding_window"]),
            block_size=int(c["block_size"]),
            mask_token_id=int(d["mask_token_id"]),
            code_size=int(bridge["code_size"]),
            rank=int(bridge["rank"]),
            bridge_eps=float(bridge["eps"]),
            head_topk=int(d["selected_prefix_last_write_refine"]["head_topk"]),
            target_layer_ids=tuple(int(i) for i in d["target_layer_ids"]),
            num_target_layers=int(c["num_target_layers"]),
        )


# ----------------------------------------------------------------------------- weights
@dataclass
class Layer:
    attn_norm: torch.Tensor
    post_norm: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    wq: torch.Tensor  # (heads * D, H)
    wk: torch.Tensor  # (kv * D, H)
    wv: torch.Tensor
    wo: torch.Tensor  # (H, heads * D)
    w_gate: torch.Tensor
    w_up: torch.Tensor
    w_down: torch.Tensor
    read: torch.Tensor  # (C, H) the bridge's read after this layer
    write: torch.Tensor  # (H, R) the bridge's write after this layer


@dataclass
class Weights:
    fc: torch.Tensor  # (H, 5 H)
    hidden_norm: torch.Tensor
    layers: list[Layer]
    norm: torch.Tensor
    codebook: torch.Tensor  # (V, C)
    gate: torch.Tensor  # (R, C)
    gate_bias: torch.Tensor  # (R,)
    pred_gate: torch.Tensor  # (R, C)
    prev: torch.Tensor  # (R, C)
    embed: torch.Tensor  # (V, H) the target's
    head: torch.Tensor  # (V, H) the target's

    @staticmethod
    def load(draft: Path, target: Path) -> Weights:
        with safe_open(str(draft / "model.safetensors"), framework="pt") as f:
            keys = set(f.keys())

            def t(name: str) -> torch.Tensor:
                if name in keys:
                    return f.get_tensor(name)
                # fp8_inc: e4m3 bytes (|x| <= 240) times one f32 scale
                codes = f.get_tensor(name + "_quant").to(torch.int64)
                scale = f.get_tensor(name + "_scale").float().reshape(())
                return (E4M3_TABLE[codes] * scale).bfloat16()

            b = "selected_prefix_bridge."
            early, last = t(b + "early_read_weights"), t(b + "read_weight")
            writes = t(b + "feature_write_weights")
            layers = []
            i = 0
            while {
                f"layers.{i}.input_layernorm.weight" + x for x in ("", "_quant")
            } & keys:
                p = f"layers.{i}."
                layers.append(
                    Layer(
                        attn_norm=t(p + "input_layernorm.weight"),
                        post_norm=t(p + "post_attention_layernorm.weight"),
                        q_norm=t(p + "self_attn.q_norm.weight"),
                        k_norm=t(p + "self_attn.k_norm.weight"),
                        wq=t(p + "self_attn.q_proj.weight"),
                        wk=t(p + "self_attn.k_proj.weight"),
                        wv=t(p + "self_attn.v_proj.weight"),
                        wo=t(p + "self_attn.o_proj.weight"),
                        w_gate=t(p + "mlp.gate_proj.weight"),
                        w_up=t(p + "mlp.up_proj.weight"),
                        w_down=t(p + "mlp.down_proj.weight"),
                        read=early[i] if i < len(early) else last,
                        write=writes[i],
                    )
                )
                i += 1
            if len(early) != len(layers) - 1:
                raise ValueError("one early read per layer but the last")
            weights = dict(
                fc=t("fc.weight"),
                hidden_norm=t("hidden_norm.weight"),
                norm=t("norm.weight"),
                codebook=t(b + "codebook"),
                gate=t(b + "gate.weight"),
                gate_bias=t(b + "gate.bias"),
                pred_gate=t(b + "predecessor_gate.weight"),
                prev=t(b + "prev.weight"),
            )
        embed, head = target_embeddings(target)
        return Weights(layers=layers, embed=embed, head=head, **weights)


# ----------------------------------------------------------------------------- the drafter
@dataclass
class Stash:
    """The last write's state before it: the residual ``raw`` (T, H), its RMS (T,) and the codes'
    projection ``packed`` (T, 3R) [query | sender | value]."""

    raw: torch.Tensor
    rms: torch.Tensor
    packed: torch.Tensor


class DSpineRef:
    """The drafter of one sequence: ``inject`` the target's features of new positions (in order),
    ``draft`` a block from the anchor token at the next position."""

    def __init__(self, cfg: DSpineConfig, w: Weights) -> None:
        self.cfg, self.w = cfg, w
        self.positions = torch.zeros(0, dtype=torch.int64)
        D = cfg.head_dim
        self.keys = [torch.zeros(0, cfg.kv_heads, D) for _ in w.layers]
        self.values = [torch.zeros(0, cfg.kv_heads, D) for _ in w.layers]
        self.packed_w = torch.cat([w.gate, w.pred_gate, w.prev]).float()  # (3R, C)

    # ---- the context
    def inject(self, taps: torch.Tensor, positions: torch.Tensor) -> None:
        self.inject_projected(taps.float() @ self.w.fc.float().T, positions)

    def inject_projected(
        self, projected: torch.Tensor, positions: torch.Tensor
    ) -> None:
        """``inject`` from the taps already through ``fc`` (n, H) f32, before ``hidden_norm``."""
        cfg, w = self.cfg, self.w
        g = rms_norm(projected.float(), w.hidden_norm, cfg.eps)
        cos, sin = rope_cos_sin(positions, cfg.head_dim, cfg.rope_theta)
        for i, layer in enumerate(w.layers):
            k = (g @ layer.wk.float().T).reshape(len(g), cfg.kv_heads, cfg.head_dim)
            k = apply_rope(rms_norm(k, layer.k_norm, cfg.eps), cos, sin)
            v = (g @ layer.wv.float().T).reshape(len(g), cfg.kv_heads, cfg.head_dim)
            self.keys[i] = torch.cat([self.keys[i], k])
            self.values[i] = torch.cat([self.values[i], v])
        self.positions = torch.cat([self.positions, positions])

    def crop(self, length: int) -> None:
        """Forget the entries at positions >= length (a rejected speculation)."""
        keep = self.positions < length
        self.positions = self.positions[keep]
        self.keys = [k[keep] for k in self.keys]
        self.values = [v[keep] for v in self.values]

    # ---- a block
    def _attention(
        self, layer: Layer, i: int, h: torch.Tensor, positions: torch.Tensor
    ):
        cfg = self.cfg
        T, D, G, NH = len(h), cfg.head_dim, cfg.kv_heads, cfg.heads
        cos, sin = rope_cos_sin(positions, D, cfg.rope_theta)
        q = apply_rope(
            rms_norm((h @ layer.wq.float().T).reshape(T, NH, D), layer.q_norm, cfg.eps),
            cos,
            sin,
        )
        kb = apply_rope(
            rms_norm((h @ layer.wk.float().T).reshape(T, G, D), layer.k_norm, cfg.eps),
            cos,
            sin,
        )
        vb = (h @ layer.wv.float().T).reshape(T, G, D)
        keys = torch.cat([self.keys[i], kb])
        values = torch.cat([self.values[i], vb])
        kpos = torch.cat([self.positions, positions])
        # causal sliding window: 0 <= query - key < window
        gap = positions[:, None] - kpos[None, :]
        visible = (gap >= 0) & (gap < cfg.window)
        hpg = NH // G
        qh = q.reshape(T, G, hpg, D)
        scores = torch.einsum("tghd,sgd->ghts", qh, keys) / D**0.5
        scores = scores.masked_fill(~visible[None, None], float("-inf"))
        probs = torch.softmax(scores, -1)
        o = torch.einsum("ghts,sgd->tghd", probs, values).reshape(T, NH * D)
        return o @ layer.wo.float().T

    def code(self, token: int) -> torch.Tensor:
        """The exact code of ``token``: sqrt(C) * codebook[token] (C,)."""
        return math.sqrt(self.cfg.code_size) * self.w.codebook[token].float()

    def bridge(
        self, raw: torch.Tensor, layer: Layer, anchor: int
    ) -> tuple[torch.Tensor, Stash]:
        """The write after a layer: raw (T, H) -> (raw + rms * write @ message, the stash)."""
        cfg, w = self.cfg, self.w
        C, R, eps = cfg.code_size, cfg.rank, cfg.bridge_eps
        rms = raw.square().mean(-1).clamp_min(torch.finfo(torch.float32).tiny).sqrt()
        proj = (raw / (rms[:, None] + eps)) @ layer.read.float().T
        codes = math.sqrt(C) * proj / (proj.norm(dim=-1, keepdim=True) + eps)
        codes[0] = self.code(anchor)
        packed = codes @ self.packed_w.T
        q, s, v = packed[:, :R], packed[:, R : 2 * R], packed[:, 2 * R :]
        message = torch.zeros_like(q)
        message[1:] = torch.sigmoid(q[1:] + s[:-1] + w.gate_bias.float()) * v[:-1]
        delta = message @ layer.write.float().T
        scale = rms.clone()
        scale[0] = 0.0  # the anchor row gets no write
        return raw + scale[:, None] * delta, Stash(raw, rms, packed)

    def block_forward(
        self, anchor_id: int, anchor_pos: int, trace: list | None = None
    ) -> tuple[torch.Tensor, Stash]:
        """The final normed hidden states (block_size, H) of the block [anchor, mask ...] at
        anchor_pos .. and the last write's stash; ``trace`` collects each layer's residual output
        (after its write)."""
        cfg, w = self.cfg, self.w
        T = cfg.block_size
        ids = torch.full((T,), cfg.mask_token_id, dtype=torch.int64)
        ids[0] = anchor_id
        positions = anchor_pos + torch.arange(T)
        h = w.embed[ids].float()
        stash = None
        for i, layer in enumerate(w.layers):
            hn = rms_norm(h, layer.attn_norm, cfg.eps)
            h = h + self._attention(layer, i, hn, positions)
            hn = rms_norm(h, layer.post_norm, cfg.eps)
            m = (
                F.silu(hn @ layer.w_gate.float().T) * (hn @ layer.w_up.float().T)
            ) @ layer.w_down.float().T
            h, stash = self.bridge(h + m, layer, anchor_id)
            if trace is not None:
                trace.append(h.clone())
        assert stash is not None
        return rms_norm(h, w.norm, cfg.eps), stash

    def candidates(self, hidden: torch.Tensor) -> torch.Tensor:
        """The ``head_topk`` largest logits' ids of each mask row, ascending: (T - 1, K)."""
        logits = hidden[1:] @ self.w.head.float().T
        return torch.topk(logits, self.cfg.head_topk, dim=-1).indices.sort(-1).values

    def refine(
        self,
        stash: Stash,
        cands: torch.Tensor,
        anchor_id: int,
        path: list[int] | None = None,
        scores: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """The greedy serial walk: row t's last write redone from the exact code of the token
        chosen at row t - 1 (the anchor for row 1); the candidate with the largest refined logit.
        ``path`` forces the choices of its first rows (another walk's); ``scores`` collects each
        row's refined logits of its candidates."""
        cfg, w = self.cfg, self.w
        R, last = cfg.rank, self.w.layers[-1]
        gb = w.gate_bias.float()
        forced, prev, path = list(path or []), anchor_id, []
        for r in range(cfg.block_size - 1):
            row = r + 1
            e = self.code(prev)
            m = torch.sigmoid(stash.packed[row, :R] + w.pred_gate.float() @ e + gb) * (
                w.prev.float() @ e
            )
            refined = stash.raw[row] + stash.rms[row] * (last.write.float() @ m)
            refined = rms_norm(refined, w.norm, cfg.eps)
            logits = w.head[cands[r]].float() @ refined
            if scores is not None:
                scores.append(logits)
            prev = (
                forced[r]
                if r < len(forced)
                else int(cands[r, int(torch.argmax(logits))])
            )
            path.append(prev)
        return torch.tensor(path)

    def draft(self, anchor_id: int, anchor_pos: int) -> torch.Tensor:
        """The block's drafts (block_size - 1,) after ``anchor_id`` at ``anchor_pos``."""
        hidden, stash = self.block_forward(anchor_id, anchor_pos)
        return self.refine(stash, self.candidates(hidden), anchor_id)


# ----------------------------------------------------------------------------- checks
def _smoke(cfg: DSpineConfig, w: Weights, length: int) -> None:
    torch.manual_seed(0)
    ref = DSpineRef(cfg, w)
    taps = torch.randn(length, len(cfg.target_layer_ids) * cfg.hidden) * 2
    start = time.perf_counter()
    ref.inject(taps, torch.arange(length))
    drafts = ref.draft(anchor_id=1000, anchor_pos=length)
    print(
        f"context {length}: drafts {drafts.tolist()} in {time.perf_counter() - start:.1f} s"
    )


if __name__ == "__main__":
    cfg = DSpineConfig.from_json(DRAFT_DIR / "config.json")
    print(cfg)
    start = time.perf_counter()
    weights = Weights.load(DRAFT_DIR, TARGET_DIR)
    print(f"weights loaded in {time.perf_counter() - start:.1f} s")
    _smoke(cfg, weights, int(os.environ.get("SMOKE_LEN", "64")))
