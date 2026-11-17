"""CPU PyTorch reference of the DFlash 2 drafter of Qwen3.8-27B (``incoai/Qwen3.8-27B-DFlash2``).

The drafter is five Qwen3 decoder layers at the target's width (hidden 5120, 32 query heads, 8 KV
heads of 128, full NeoX RoPE, SwiGLU 17408) with three additions:

- **KV injection.** The target's residual outputs after layers 5, 19, 33, 47 and 61 (concatenated,
  25600 wide) go through ``fc`` and ``hidden_norm`` into one context feature per position. Every
  drafter layer projects it with its own ``k_proj`` / ``v_proj`` (k_norm, RoPE at the position) and
  keeps the result as that position's cache entry. Nothing else of the context is kept.
- **A drafting block** at the anchor position p (the last emitted token, which the target has not
  processed): the target's embeddings of [anchor, mask x 7] at positions p .. p + 7 run through the
  layers; queries attend to the whole block (bidirectional) and to the cache entries within 2048
  positions before each query. Each sublayer is wrapped in a two-tap grouped dynamic convolution
  along the block (``GroupedDynamicConv``): its input is convolved before the sublayer and its
  output after it, with coefficients from one projection of the input.
- **The candidate selector.** The target's LM head scores the 7 mask positions; the 16 largest
  logits of each are its candidates. A greedy walk from the anchor picks, at each position, the
  candidate maximizing ``logit + <A[prev] * P(h), B[cand]>`` (rank-256 codebooks A, B and a
  projection P of the position's final hidden state).

The algorithm follows ``z-lab/dflash`` (``dflash/model.py``: ``DFlash2DraftModel``,
``GroupedDynamicCausalConv``, ``CandidateSelector``, ``dflash_generate``). Math runs in float32
over the checkpoint's bf16 weights, so it tracks the bf16 original closely but not to the bit.

  DRAFT=/path/to/Qwen3.8-27B-DFlash2 TARGET=/path/to/Qwen3.8-27B-FP8 \\
      python models/gaudi/tools/qwen_dflash2_ref.py

runs a smoke test on random context features (``SMOKE_LEN`` positions, default 64). With
``DFLASH_ZLAB=/path/to/z-lab/dflash`` (and transformers installed) it also checks one block against
the original implementation.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

DRAFT_DIR = Path(os.environ.get("DRAFT", "Qwen3.8-27B-DFlash2"))
TARGET_DIR = Path(os.environ.get("TARGET", "Qwen3.8-27B-FP8"))


# ----------------------------------------------------------------------------- config
@dataclass(frozen=True)
class DFlash2Config:
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
    conv_kernel: int
    conv_group: int
    mask_token_id: int
    selector_rank: int
    selector_top_k: int
    target_layer_ids: tuple[int, ...]
    num_target_layers: int

    @staticmethod
    def from_json(path: Path) -> DFlash2Config:
        c = json.loads(path.read_text())
        d = c["dflash_config"]
        if (
            c.get("architectures") != ["DFlash2DraftModel"]
            or c.get("model_type") != "qwen3"
        ):
            raise ValueError(f"{path}: not a DFlash 2 Qwen3 drafter")
        if c.get("is_causal", False) or not c.get("use_sliding_window", False):
            raise ValueError(
                "the reference implements non-causal sliding-window drafters"
            )
        if set(c["layer_types"]) != {"sliding_attention"}:
            raise ValueError(f"unsupported layer types {c['layer_types']}")
        if c["rope_parameters"].get("rope_type", "default") != "default":
            raise ValueError("unsupported RoPE scaling")
        return DFlash2Config(
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
            block_size=int(d["block_size"]),
            conv_kernel=int(d["conv_kernel_size"]),
            conv_group=int(d["conv_group_size"]),
            mask_token_id=int(d["mask_token_id"]),
            selector_rank=int(d["selector_rank"]),
            selector_top_k=int(d["selector_top_k"]),
            target_layer_ids=tuple(int(i) for i in d["target_layer_ids"]),
            num_target_layers=int(c["num_target_layers"]),
        )

    @property
    def groups(self) -> int:
        return self.hidden // self.conv_group


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3RMSNorm: x / rms(x) * w (no +1)."""
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.float()


def rope_cos_sin(
    positions: torch.Tensor, dim: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """(n, dim) cos and sin of the default RoPE (inverse frequencies over the whole head)."""
    inv = theta ** (-torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    ang = positions.to(torch.float32)[:, None] * inv[None, :]
    emb = torch.cat([ang, ang], -1)
    return torch.cos(emb), torch.sin(emb)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], -1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x (n, heads, D) at positions with cos / sin (n, D)."""
    return x * cos[:, None, :] + rotate_half(x) * sin[:, None, :]


# ----------------------------------------------------------------------------- weights
@dataclass
class Conv:
    base: torch.Tensor  # (2, taps, H): [prepare | finish]
    proj: torch.Tensor  # (2 * taps * groups, H)


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
    attn_conv: Conv
    mlp_conv: Conv


@dataclass
class Weights:
    fc: torch.Tensor  # (H, 5 H)
    hidden_norm: torch.Tensor
    layers: list[Layer]
    norm: torch.Tensor
    pred: torch.Tensor  # (V, rank)
    succ: torch.Tensor  # (V, rank)
    hproj: torch.Tensor  # (rank, H)
    embed: torch.Tensor  # (V, H) the target's
    head: torch.Tensor  # (V, H) the target's

    @staticmethod
    def load(draft: Path, target: Path) -> Weights:
        with safe_open(str(draft / "model.safetensors"), framework="pt") as f:

            def t(name: str) -> torch.Tensor:
                return f.get_tensor(name)

            layers = []
            i = 0
            while f"layers.{i}.input_layernorm.weight" in f.keys():
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
                        attn_conv=Conv(
                            t(p + "attention_conv.base_kernel"),
                            t(p + "attention_conv.kernel_projection.weight"),
                        ),
                        mlp_conv=Conv(
                            t(p + "mlp_conv.base_kernel"),
                            t(p + "mlp_conv.kernel_projection.weight"),
                        ),
                    )
                )
                i += 1
            fc, hidden_norm, norm = (
                t("fc.weight"),
                t("hidden_norm.weight"),
                t("norm.weight"),
            )
            pred = t("candidate_selector.predecessor_codebook")
            succ = t("candidate_selector.successor_codebook")
            hproj = t("candidate_selector.hidden_projection.weight")
        embed, head = target_embeddings(target)
        return Weights(fc, hidden_norm, layers, norm, pred, succ, hproj, embed, head)


def target_embeddings(target: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """The target's (bf16) token embedding and LM head from its safetensors."""
    index = target / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
    else:
        weight_map = {}
        for shard in sorted(target.glob("*.safetensors")):
            with safe_open(str(shard), framework="pt") as f:
                for name in f.keys():
                    weight_map[name] = shard.name
    names = [n for n in weight_map if n.endswith("embed_tokens.weight")]
    if len(names) != 1:
        raise ValueError(f"{target}: expected one embedding table, found {names}")
    embed_name = names[0]
    head_name = "lm_head.weight" if "lm_head.weight" in weight_map else embed_name
    out = {}
    for name in (embed_name, head_name):
        with safe_open(str(target / weight_map[name]), framework="pt") as f:
            out[name] = f.get_tensor(name)
    return out[embed_name], out[head_name]


# ----------------------------------------------------------------------------- the drafter
def grouped_dynamic_conv(
    h: torch.Tensor, dyn: torch.Tensor, base: torch.Tensor, group: int
) -> torch.Tensor:
    """``_grouped_dynamic_convolve`` over one block: h (T, H), dyn (T, taps, groups), base
    (taps, H). Tap k reads the block's column t - k (zero before the block's first column)."""
    T, H = h.shape
    taps = base.shape[0]
    blocks = h.reshape(T, H // group, group)
    out = torch.zeros_like(blocks)
    for k in range(taps):
        shifted = blocks if k == 0 else F.pad(blocks[:-k], (0, 0, 0, 0, k, 0))
        coef = base[k].reshape(1, H // group, group) + dyn[:, k, :, None]
        out = out + coef * shifted
    return out.reshape(T, H)


class DFlash2Ref:
    """The drafter of one sequence: ``inject`` the target's features of new positions (in order),
    ``draft`` a block from the anchor token at the next position."""

    def __init__(self, cfg: DFlash2Config, w: Weights) -> None:
        self.cfg, self.w = cfg, w
        self.positions = torch.zeros(0, dtype=torch.int64)
        D = cfg.head_dim
        self.keys = [torch.zeros(0, cfg.kv_heads, D) for _ in w.layers]
        self.values = [torch.zeros(0, cfg.kv_heads, D) for _ in w.layers]

    # ---- the context
    def context_feature(self, taps: torch.Tensor) -> torch.Tensor:
        """taps (n, 5 H) f32: the target's residual outputs after the tap layers, concatenated in
        ``target_layer_ids`` order -> (n, H) the normed fused feature."""
        return rms_norm(
            taps.float() @ self.w.fc.float().T, self.w.hidden_norm, self.cfg.eps
        )

    def inject(self, taps: torch.Tensor, positions: torch.Tensor) -> None:
        self.inject_projected(taps.float() @ self.w.fc.float().T, positions)

    def inject_projected(
        self, projected: torch.Tensor, positions: torch.Tensor
    ) -> None:
        """``inject`` from the taps already through ``fc`` (the sum of the Gaudi drafter's five
        ``feat`` parts), (n, H) f32, before ``hidden_norm``."""
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
    def _conv(
        self, h: torch.Tensor, conv: Conv, side: int, dyn: torch.Tensor
    ) -> torch.Tensor:
        cfg = self.cfg
        d = dyn.reshape(len(h), 2, cfg.conv_kernel, cfg.groups)[:, side]
        return grouped_dynamic_conv(h, d, conv.base[side].float(), cfg.conv_group)

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
        # non-causal sliding window: |query - key| < window
        gap = positions[:, None] - kpos[None, :]
        visible = (gap < cfg.window) & (-gap < cfg.window)
        hpg = NH // G
        qh = q.reshape(T, G, hpg, D)
        scores = torch.einsum("tghd,sgd->ghts", qh, keys) / D**0.5
        scores = scores.masked_fill(~visible[None, None], float("-inf"))
        probs = torch.softmax(scores, -1)
        o = torch.einsum("ghts,sgd->tghd", probs, values).reshape(T, NH * D)
        return o @ layer.wo.float().T

    def block_forward(
        self, anchor_id: int, anchor_pos: int, trace: list | None = None
    ) -> torch.Tensor:
        """The final normed hidden states (block_size, H) of the block [anchor, mask ...] at
        anchor_pos ..; ``trace`` collects each layer's residual output."""
        cfg, w = self.cfg, self.w
        T = cfg.block_size
        ids = torch.full((T,), cfg.mask_token_id, dtype=torch.int64)
        ids[0] = anchor_id
        positions = anchor_pos + torch.arange(T)
        h = w.embed[ids].float()
        for i, layer in enumerate(w.layers):
            hn = rms_norm(h, layer.attn_norm, cfg.eps)
            dyn = hn @ layer.attn_conv.proj.float().T
            a = self._attention(
                layer, i, self._conv(hn, layer.attn_conv, 0, dyn), positions
            )
            h = h + self._conv(a, layer.attn_conv, 1, dyn)
            hn = rms_norm(h, layer.post_norm, cfg.eps)
            dyn = hn @ layer.mlp_conv.proj.float().T
            x = self._conv(hn, layer.mlp_conv, 0, dyn)
            m = (
                F.silu(x @ layer.w_gate.float().T) * (x @ layer.w_up.float().T)
            ) @ layer.w_down.float().T
            h = h + self._conv(m, layer.mlp_conv, 1, dyn)
            if trace is not None:
                trace.append(h.clone())
        return rms_norm(h, w.norm, cfg.eps)

    def select(
        self, hidden: torch.Tensor, logits: torch.Tensor, anchor_id: int
    ) -> torch.Tensor:
        """Greedy candidate path: hidden, logits of the block's mask positions -> their drafts."""
        w, k = self.w, self.cfg.selector_top_k
        unary, cands = torch.topk(logits, k, dim=-1)
        hp = hidden @ w.hproj.float().T
        prev, path = anchor_id, []
        for pos in range(len(hidden)):
            scores = unary[pos] + (
                w.succ[cands[pos]].float() * (w.pred[prev].float() * hp[pos])
            ).sum(-1)
            prev = int(cands[pos, int(torch.argmax(scores))])
            path.append(prev)
        return torch.tensor(path)

    def draft(self, anchor_id: int, anchor_pos: int) -> torch.Tensor:
        """The block's drafts (block_size - 1,) after ``anchor_id`` at ``anchor_pos``."""
        hidden = self.block_forward(anchor_id, anchor_pos)[1:]
        logits = hidden @ self.w.head.float().T
        return self.select(hidden, logits, anchor_id)


# ----------------------------------------------------------------------------- checks
def _smoke(cfg: DFlash2Config, w: Weights, length: int) -> None:
    torch.manual_seed(0)
    ref = DFlash2Ref(cfg, w)
    taps = torch.randn(length, len(cfg.target_layer_ids) * cfg.hidden) * 2
    start = time.perf_counter()
    ref.inject(taps, torch.arange(length))
    drafts = ref.draft(anchor_id=1000, anchor_pos=length)
    print(
        f"context {length}: drafts {drafts.tolist()} in {time.perf_counter() - start:.1f} s"
    )


def _zlab_check(cfg: DFlash2Config, w: Weights, length: int) -> None:
    """One block against z-lab's DFlash2DraftModel on the same random features."""
    import importlib.util
    import sys

    path = Path(os.environ["DFLASH_ZLAB"]) / "dflash" / "model.py"
    spec = importlib.util.spec_from_file_location("zlab_dflash_model", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.manual_seed(1)
    model = module.DFlash2DraftModel.from_pretrained(
        str(DRAFT_DIR), dtype=torch.float32
    )
    model.eval()
    T = cfg.block_size
    taps = torch.randn(length, len(cfg.target_layer_ids) * cfg.hidden) * 2
    anchor = 1000
    ids = torch.full((1, T), cfg.mask_token_id, dtype=torch.int64)
    ids[0, 0] = anchor
    with torch.inference_mode():
        cache = module._make_cache(model.config)
        hidden = model(
            target_hidden=taps[None],
            noise_embedding=w.embed[ids].float(),
            position_ids=torch.arange(length + T)[None],
            past_key_values=cache,
            use_cache=True,
        )[0, 1:]
        drafts, _, _ = model.propose(
            hidden[None], torch.tensor([anchor]), lambda h: h @ w.head.float().T, 0.0
        )
    ref = DFlash2Ref(cfg, w)
    ref.inject(taps, torch.arange(length))
    mine = ref.block_forward(anchor, length)[1:]
    my_drafts = ref.select(mine, mine @ w.head.float().T, anchor)
    err = (mine - hidden).abs().max() / hidden.abs().max()
    print(
        f"z-lab check: hidden max rel err {err:.2e}, drafts {my_drafts.tolist()} vs {drafts[0].tolist()}"
    )
    if err > 1e-3 or not torch.equal(my_drafts, drafts[0]):
        raise SystemExit("reference disagrees with z-lab/dflash")


if __name__ == "__main__":
    cfg = DFlash2Config.from_json(DRAFT_DIR / "config.json")
    print(cfg)
    start = time.perf_counter()
    weights = Weights.load(DRAFT_DIR, TARGET_DIR)
    print(f"weights loaded in {time.perf_counter() - start:.1f} s")
    length = int(os.environ.get("SMOKE_LEN", "64"))
    _smoke(cfg, weights, length)
    if os.environ.get("DFLASH_ZLAB"):
        _zlab_check(cfg, weights, length)
