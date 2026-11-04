"""DSpark speculative drafting for Mistral-Large-4 on Gaudi: the release's draft head, tensor
parallel as the target is (heads and the FFN's width sliced per rank, partials summed between
launches). Its inputs, the target's residual stream and embedding, are replicated, as are its KV
injection and Markov head, so every rank drafts the same tokens.

The algorithm follows ``models/gaudi/tools/ml4_dspark_ref.py`` (the CPU reference):

- **KV injection.** Every position the target processes feeds the drafter's cache: the target's
  residual outputs at layers 1, 18, 36, 54, 71 (``DRAFT_TAPS``, captured by ``Ml4Gaudi.capture``)
  -> ``target_projection`` (as five 6144-wide GEMVs) -> ``target_norm`` -> per drafter layer
  ``wkv_a``, the latent's ``kv_a_norm``, rope, and ``mla_store`` into that layer's pages (3 extra
  layers in the arena, ``CacheGeometry.draft_layers``).
- **A drafting block** at the anchor position p (the last accepted token, not yet processed by the
  target): columns [anchor embedding, mask row 0, mask row 1] at p, p + 1, p + 2 through the 3
  drafter layers (the target's own MLA and dense FFN graphs at the drafter's dims), attending to the whole block and the 2048 positions before p (bidirectional within the block:
  every query column sees keys up to p + 2; the window's start is the slot table's ``kmin``).
- **Logits.** ``final_norm``, the target's vocabulary-sharded LM head, an all-gather.
- **The Markov head** (one launch, 3 sequential steps): ``z = [state | E[prev] | h_k]``,
  ``joint_proj`` -> gate, candidate, output; the state update; draft ``k`` = argmax of
  ``softcap(base_k) + W2 tanh(output)``; the draft feeds the next step's ``E[prev]``.

Weights come from the release's safetensors (``consolidated-00032-of-00032.safetensors``, bf16), as
Q8_0 planar for the GEMVs (the target's lq kernels) and f32 for norms.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from resin.models.mistral_large4.config import MistralLarge4Config
from resin_models_gaudi.gguf import pack
from resin_models_gaudi.mistral_large4 import layers as L
from resin_models_gaudi.mistral_large4.cache import PAGE_TOKENS, STORE_FIELDS
from resin_models_gaudi.mistral_large4.weights import RankSlices
from resin_models_gaudi.qwen3_5.checkpoint import bf16_rne
from resin_models_gaudi.synapse.graph import Graph
from resin_models_gaudi.synapse.runtime import F32, I8, I32, Device

if TYPE_CHECKING:
    from resin_models_gaudi.mistral_large4.model import Ml4Gaudi

DRAFT_TAPS = (
    1,
    18,
    36,
    54,
    71,
)  # the target layers whose residual outputs the drafter reads
DRAFT_LAYERS = 3
DEPTH = 3  # drafts per block (block columns)
WINDOW = 2048  # context positions before the anchor each drafter layer sees
MAX_ROWS = (
    4  # sequences per drafting step (verify steps of rows x 4 columns fit 16 columns)
)
SHARD = "consolidated-00032-of-00032.safetensors"
Q8 = "Q8_0"
KIND = L.AttnKind(rope=True, owns_kv=True, owns_index=False, index_queries=False)


def drafter_dims(
    cfg: MistralLarge4Config, rank: int, nranks: int, gather_tokens: int
) -> L.Dims:
    """One rank's drafter dimensions: the target's MLA and dense FFN slices (8 of 64 heads, 3072
    of the dense FFN's 24576 on 8 ranks), attention scale 1 / sqrt(128) (no YaRN mscale)."""
    d = L.Dims.of(cfg, RankSlices.of(cfg, rank, nranks))
    return replace(
        d, ranks=nranks, gather_tokens=gather_tokens, scale=cfg.qk_head_dim**-0.5
    )


# ------------------------------------------------------------------ weights
def _q8_0(w: np.ndarray) -> np.ndarray:
    """f32 rows (M, K) -> GGUF Q8_0 blocks (M, K / 32 * 34) u8 (round to nearest, d = amax / 127
    as f16, as ggml's quantize_row_q8_0)"""
    M, K = w.shape
    x = w.reshape(M, K // 32, 32)
    d = np.abs(x).max(-1) / 127.0
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1), 0.0)
    q = np.rint(x * inv[..., None]).clip(-127, 127).astype(np.int8)
    out = np.empty((M, K // 32, 34), np.uint8)
    out[..., :2] = d.astype(np.float16)[..., None].view(np.uint8)
    out[..., 2:] = q.view(np.uint8)
    return out.reshape(M, -1)


def _planar_q8(w: np.ndarray) -> np.ndarray:
    return pack.planar(_q8_0(np.asarray(w, np.float32)), Q8).reshape(-1)


def _embedding_q8(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(V, K) f32 -> Q8_0 gather tables (q int8 (V, K), d f32 (V, K / 32)), as embedding_q8"""
    V, K = w.shape
    b = _q8_0(w).reshape(V, K // 32, 34)
    q = b[:, :, 2:].view(np.int8).reshape(V, K)
    d = np.ascontiguousarray(b[:, :, :2]).view(np.float16)[..., 0].astype(np.float32)
    return q, d


def prepare_drafter(
    checkpoint: Path | str, cfg: MistralLarge4Config, rank: int = 0, nranks: int = 1
) -> Iterator[tuple[str, np.ndarray, str]]:
    """One rank's drafter weights from the release's safetensors (bf16): ``d{l}.*`` per layer
    (the target layer graphs' names, sliced as the target's), ``d.*`` shared. GEMV matrices as
    Q8_0 planar."""
    import torch
    from safetensors import safe_open

    path = Path(checkpoint) / SHARD
    with safe_open(str(path), framework="pt") as f:

        def t(name: str) -> np.ndarray:
            return f.get_tensor("mtp_dspark." + name).to(torch.float32).numpy()

        sl = RankSlices.of(cfg, rank, nranks)
        H, heads, qk = cfg.hidden_size, cfg.num_attention_heads, cfg.qk_head_dim
        hr = slice(sl.heads.start * qk, sl.heads.stop * qk)  # wq rows, wo inputs
        for layer in range(DRAFT_LAYERS):
            p, q = f"backbone.layers.{layer}.", f"d{layer}."
            a = p + "attention."
            yield q + "attn_norm", t(p + "attention_norm.weight"), "f32"
            yield q + "ffn_norm", t(p + "ffn_norm.weight"), "f32"
            yield q + "wq", _planar_q8(t(a + "wq.weight")[hr]), "planar:" + Q8
            yield q + "wkv", _planar_q8(t(a + "wkv_a_with_mqa.weight")), "planar:" + Q8
            yield q + "kv_norm", t(a + "kv_a_norm.weight"), "f32"
            # [k_nope 64 | v 128] rows per head
            wkv_b = t(a + "wkv_b.weight").reshape(heads, 192, 512)[sl.heads]
            kb = np.ascontiguousarray(
                wkv_b[:, :64].transpose(0, 2, 1)
            )  # (h, 512, nope)
            vb = np.ascontiguousarray(wkv_b[:, 64:])  # (h, v, 512)
            yield q + "kb", bf16_rne(kb), "bf16"
            yield q + "vb", bf16_rne(vb), "bf16"
            # no SSMax in the drafter: scale s log(pos + 1) + b = 1
            yield q + "ss_s", np.zeros(len(wkv_b), np.float32), "f32"
            yield q + "ss_b", np.ones(len(wkv_b), np.float32), "f32"
            yield q + "wo", _planar_q8(t(a + "wo.weight")[:, hr]), "planar:" + Q8
            ff = p + "feed_forward."
            gu = np.concatenate(
                [t(ff + "w1.weight")[sl.dense], t(ff + "w3.weight")[sl.dense]]
            )
            yield q + "du", _planar_q8(gu), "planar:" + Q8
            yield q + "dd", _planar_q8(t(ff + "w2.weight")[:, sl.dense]), "planar:" + Q8
        proj = t(
            "backbone.target_projection.weight"
        )  # (6144, 5 x 6144), tap-major inputs
        for k in range(len(DRAFT_TAPS)):
            yield (
                f"d.tproj{k}",
                _planar_q8(proj[:, k * H : (k + 1) * H]),
                "planar:" + Q8,
            )
        del proj
        yield "d.target_norm", t("backbone.target_norm.weight"), "f32"
        yield "d.final_norm", t("backbone.final_norm.weight"), "f32"
        yield "d.mask", t("backbone.mask_embedding"), "f32"  # (2, 6144)
        h = "autoregressive_head."
        eq, es = _embedding_q8(t(h + "embedding.weight"))
        yield "d.mk_eq", eq, "i8"
        yield "d.mk_es", es, "f32"
        yield "d.mk_w2", _planar_q8(t(h + "output.weight")), "planar:" + Q8
        yield "d.joint", _planar_q8(t(h + "joint_proj.weight")), "planar:" + Q8
        yield "d.joint_b", t(h + "joint_proj.bias"), "f32"


def drafter_config(checkpoint: Path | str) -> dict:
    """``params.json``'s ``mtp`` entry, checked against what this port implements"""
    mtp = json.loads((Path(checkpoint) / "params.json").read_text())["mtp"]
    taps = tuple(int(x) for x in str(mtp["target_activation_layers"]).split(","))
    dspark = mtp["dspark"]
    if (
        taps != DRAFT_TAPS
        or int(mtp["n_layers"]) != DRAFT_LAYERS
        or int(mtp["depth"]) != DEPTH
        or dspark.get("head_type") != "rnn"
        or dspark.get("draft_input") != "position_specific"
        or int(dspark.get("markov_rank", 0)) != 512
    ):
        raise ValueError(f"unsupported DSpark configuration {mtp}")
    return mtp


# ------------------------------------------------------------------ graphs
def inject_graph(
    device: Device, dims: L.Dims, tokens: int, blocks: int, mme: bool
) -> Graph:
    """KV injection of a target step's T columns into the drafter's 3 layers: the five taps
    ``tap0 .. tap4`` (H, T) through the target projection and ``target_norm``, then per layer l
    ``wkv{l}``, ``kv_norm{l}``, rope at each column's position (``cs``, the drafter's table) and
    ``mla_store`` at ``kv_rows{l}``."""
    H, T = dims.hidden, tokens
    lg = L.LayerGraph(device, f"ml4_dspark_inject_T{T}_NB{blocks}", dims, T, mme)
    g = None
    for k in range(len(DRAFT_TAPS)):
        x = lg.input(f"tap{k}", (H, T), F32)
        y = lg.view(lg.gemv(f"tproj{k}", Q8, x, H, H, T), (H, T))
        g = y if g is None else lg.add(g, y)
    g = lg.norm(g, "target_norm")
    D = dims.latent + dims.rope
    unused = lg.cast16(g)  # mla_store's index-key input (no index rows)
    for layer in range(DRAFT_LAYERS):
        kv = lg.view(lg.gemv(f"wkv{layer}", Q8, g, H, D, T), (D, T))
        lat = lg.norm(kv, f"kv_norm{layer}", dims.latent)
        _, kr = lg.split0(kv, [dims.latent, dims.rope])
        kr = lg.rope(kr, 1, 0, *lg.no_ssmax(), False)
        lg.node(
            "rg_mla_store_f32",
            [
                lg.arena(blocks),
                lg.input(f"kv_rows{layer}", (STORE_FIELDS, T), I32),
                lg.concat([lat, kr], 0),
                unused,
            ],
            [lg.input(f"sink_d{layer}", (64, 1), I32)],
        )
    return lg.g


def logits_graph(
    device: Device, dims: L.Dims, tokens: int, vocab: int, head_qtype: str
) -> Graph:
    """The drafter's output: x + p, ``final_norm`` -> ``hf`` (H, T), and the target's LM head
    slice of this rank over it -> ``logits`` (vocab * T,) (raw: the Markov graph softcaps)"""
    H, T = dims.hidden, tokens
    lg = L.LayerGraph(device, f"ml4_dspark_logits_T{T}_V{vocab}", dims, T, T >= 4)
    hf = lg.input("hf", (H, T), F32)
    lg.node(
        "rg_ml4_norm_f32",
        [lg.residual(), lg.input("final_norm", (H,), F32), lg.dep()],
        [hf],
        struct.pack("fi", dims.eps, 0),
    )
    lg.gemv(
        "head", head_qtype, hf, H, vocab, T, out=lg.input("logits", (vocab * T,), F32)
    )
    return lg.g


def markov_graph(
    device: Device, dims: L.Dims, tokens: int, rows: int, vocab: int, nranks: int
) -> Graph:
    """The RNN Markov head over a block's 3 columns per row (row r: columns 3 r .. 3 r + 2), all
    3 steps in one launch: per step k, ``z = [s | E[prev] | hf_k]``, ``joint_proj`` -> gate g,
    candidate c, output o; ``s = sigmoid(g) s + (1 - sigmoid(g)) tanh(c)``; draft k = argmax of
    ``cap tanh(base_k / cap) + W2 tanh(o)``; prev = draft k (first: the anchor).
    ``logits_all`` (vocab, T, nranks) is the all-gathered base logits. -> ``drafts`` (3, rows)."""
    H, T, R, Rk = dims.hidden, tokens, rows, 512
    lg = L.LayerGraph(device, f"ml4_dspark_markov_T{T}_R{R}_n{nranks}", dims, T)
    V = vocab * nranks
    if nranks > 1:
        all_ = lg.input("logits_all", (vocab, T, nranks), F32)
        full = lg.view(lg.perm(all_), (V, T))
    else:
        full = lg.input("logits_all", (V, T), F32)
    hf = lg.input("hf", (H, T), F32)
    s = lg.input("mk_s0", (Rk, R), F32)
    prev = lg.input("anchors", (R,), I32)
    eq, es = lg.input("mk_eq", (Rk, V), I8), lg.input("mk_es", (Rk // 32, V), F32)
    bias_in = lg.input("joint_b", (3 * Rk, 1), F32)
    drafts = []

    def pick(x: L.Tensor, k: int) -> L.Tensor:  # columns 3 r + k of x (n, T) -> (n, R)
        y = lg.tensor((x.shape[0], R), F32)
        lg.node(
            "gather_fwd_f32",
            [x, lg.input(f"mk_cols{k}", (R,), I32)],
            [y],
            struct.pack("i", 1),
        )
        return y

    def unary(guid: str, x: L.Tensor) -> L.Tensor:
        y = lg.tensor(x.shape, F32)
        lg.node(guid, [x], [y])
        return y

    def binary(guid: str, a: L.Tensor, b: L.Tensor) -> L.Tensor:
        y = lg.tensor(a.shape, F32)
        lg.node(guid, [a, b], [y])
        return y

    for k in range(DEPTH):
        base = pick(full, k)
        base = binary("mult_fwd_f32", base, lg.input("inv_cap", (1, 1), F32))
        base = binary(
            "mult_fwd_f32", unary("tanh_fwd_f32", base), lg.input("cap", (1, 1), F32)
        )
        # E[prev]: Q8_0 rows, as the target's embedding
        e8, esc, e32 = (
            lg.tensor((Rk, R), I8),
            lg.tensor((Rk // 32, R), F32),
            lg.tensor((Rk, R), F32),
        )
        lg.node("gather_fwd_i8", [eq, prev], [e8], struct.pack("i", 1))
        lg.node("gather_fwd_f32", [es, prev], [esc], struct.pack("i", 1))
        lg.node("cast_i8_to_f32", [e8], [e32])
        e = lg.tensor((32, Rk // 32, R), F32)
        lg.node(
            "mult_fwd_f32",
            [lg.view(e32, (32, Rk // 32, R)), lg.view(esc, (1, Rk // 32, R))],
            [e],
        )
        z = lg.concat([s, lg.view(e, (Rk, R)), pick(hf, k)], 0)  # (7168, R)
        y = lg.view(lg.gemv("joint", Q8, z, Rk * 2 + H, 3 * Rk, R), (3 * Rk, R))
        y = binary("add_fwd_f32", y, bias_in)
        gr, cr, orr = lg.split0(y, [Rk, Rk, Rk])
        gate = unary("sigmoid_fwd_f32", gr)
        cand = unary("tanh_fwd_f32", cr)
        # s = gate s + (1 - gate) cand = cand + gate (s - cand)
        s = binary(
            "add_fwd_f32",
            cand,
            binary("mult_fwd_f32", gate, binary("sub_fwd_f32", s, cand)),
        )
        bias = lg.view(
            lg.gemv("mk_w2", Q8, unary("tanh_fwd_f32", orr), Rk, V, R), (V, R)
        )
        logits = binary("add_fwd_f32", base, bias)
        NB = 24
        part = lg.tensor((64, NB, R), F32)
        lg.node("rg_argmax1_f32", [logits], [part])
        tok = lg.tensor((1, R), I32)
        lg.node("rg_argmax2_f32", [part], [tok])
        drafts.append(tok)
        prev = lg.view(tok, (R,))
    lg.node(
        "concat", drafts, [lg.input("drafts", (DEPTH, R), I32)], struct.pack("I", 0)
    )
    return lg.g


def graphs(
    device: Device,
    model_dims: L.Dims,
    dims: L.Dims,
    blocks: int,
    head_qtype: str,
    nranks: int,
    buckets: tuple[int, ...],
    inject_buckets: tuple[int, ...],
) -> dict[tuple, Graph]:
    """Every drafter recipe: KV injection per target bucket; per drafting bucket (rows x 3
    columns) the 3 layers' graphs, the logits and the Markov head"""
    out: dict[tuple, Graph] = {}
    for T in inject_buckets:
        out["d_inject", T] = inject_graph(device, dims, T, blocks, T >= 4)
    qtypes = {"wq": Q8, "wkv": Q8}
    for T in buckets:
        mme = T >= 4
        out["d_pre", T] = L.attn_pre_graph(device, dims, T, KIND, qtypes, blocks, mme)
        out["d_slots", T] = L.attn_slots_graph(
            device, dims, T, DEPTH, _slots(T), dims.latent + dims.rope, blocks
        )
        out["d_post", T] = L.attn_post_graph(device, dims, T, Q8, mme)
        out["d_ffn", T] = L.ffn_graph(
            device, dims, T, False, {"du": Q8, "dd": Q8}, mme, False
        )
        out["d_logits", T] = logits_graph(device, dims, T, model_dims.vocab, head_qtype)
        out["d_markov", T] = markov_graph(
            device, dims, T, _rows(T), model_dims.vocab, nranks
        )
    return out


def _rows(T: int) -> int:
    """sequences a drafting bucket of T columns holds (3 columns each)"""
    return T // DEPTH


# ------------------------------------------------------------------ the runtime
class Drafter:
    """The drafter of an ``Ml4Gaudi`` built with a draft checkpoint. ``inject`` after every
    target step (its captured taps -> the drafter's cache at the step's positions); ``draft``
    for up to ``MAX_ROWS`` sequences -> their 3 greedy drafts, which it waits for (the next block's
    positions follow the target's acceptance, which the host reads)."""

    def __init__(self, m: Ml4Gaudi) -> None:
        from resin_models_gaudi.mistral_large4.model import BUCKETS
        from resin_models_gaudi.mistral_large4.weights import rope_table

        self.m, dev, d = m, m.device, m.dd
        H, V, R = d.hidden, m.d.vocab, m.nranks
        TMAX = m.prefill_columns
        self.buckets = tuple(b for b in BUCKETS if b >= DEPTH)
        TD = self.buckets[-1]

        def buf(nbytes: int) -> int:
            return dev.malloc(max(256, -(-nbytes // 256) * 256))

        def const(a: np.ndarray) -> int:
            address = buf(a.nbytes)
            dev.h2d(address, np.ascontiguousarray(a))
            return address

        self.x = [buf(4 * H * TD), buf(4 * H * TD)]
        self.pa, self.pb = buf(4 * H * TD), buf(4 * H * TD)
        G = (
            4 * H * d.gather_tokens * R
        )  # the partials all-gathered (Dims.gathered launches)
        self.pa_all, self.pb_all = buf(G), buf(G)
        self.hf = buf(4 * H * TD)
        self.logits = buf(4 * V * TD)
        self.logits_all = buf(4 * V * TD * R) if R > 1 else self.logits
        self.drafts = buf(4 * DEPTH * TD)
        self.s0 = const(np.zeros(512 * TD, np.float32))
        self.lp = const(np.zeros(TMAX, np.float32))  # the drafter has no SSMax
        self.cols = {
            (T, k): const(np.arange(_rows(T), dtype=np.int32) * DEPTH + k)
            for T in self.buckets
            for k in range(DEPTH)
        }
        self.cs = buf(4 * 128 * TMAX)
        self.kv = [buf(4 * STORE_FIELDS * TMAX) for _ in range(DRAFT_LAYERS)]
        self.slots = [buf(4 * 8 * 16) for _ in range(DRAFT_LAYERS)]
        self.pages = [buf(4 * 2 * 16 * 16) for _ in range(DRAFT_LAYERS)]
        self.tok, self.anchors = buf(4 * TD), buf(4 * TD)
        self.sinks = [
            buf(256) for _ in range(DRAFT_LAYERS)
        ]
        self.rope = rope_table(m.cfg, None, m.max_seq_len + TMAX)  # YaRN, no LeRoPE
        self._ops: dict[tuple, object] = {}

    # ---- bindings
    def _weights(self, recipe, layer: int | None) -> dict[str, int]:
        w, out = self.m.w, {}
        for n in recipe.persistent:
            base = n.rstrip("0123456789")
            idx = n[len(base) :]
            if layer is not None and f"d{layer}.{n}" in w:
                out[n] = w[f"d{layer}.{n}"]
            elif idx and f"d{idx}.{base}" in w and base in ("wkv", "kv_norm"):
                out[n] = w[f"d{idx}.{base}"]  # the inject graph's per-layer names
            elif f"d.{n}" in w:
                out[n] = w[f"d.{n}"]
            elif n == "head":
                out[n] = w["head"]
            elif n in self.m.consts:
                out[n] = self.m.consts[n]
        return out

    def _op(self, key: tuple, names: dict[str, int], layer: int | None = None):
        if (op := self._ops.get(key)) is None:
            r = self.m.r[key[:2]]
            op = self._ops[key] = self.m._op(r, {**self._weights(r, layer), **names})
        return op

    # ---- KV injection
    def inject(self, T: int, positions: np.ndarray, col_blocks: np.ndarray) -> None:
        """The last target step's T columns (``positions``, their KV block rows; -1 entries
        skip) into the drafter's cache, from the taps the step captured"""
        m, dev, H = self.m, self.m.device, self.m.d.hidden
        pos = np.zeros(T, np.int64)
        pos[: len(positions)] = positions
        blocks = np.full(T, -1, np.int64)
        blocks[: len(col_blocks)] = col_blocks
        dev.h2d_nb(self.cs, np.ascontiguousarray(self.rope[pos]))
        for layer in range(DRAFT_LAYERS):
            dev.h2d_nb(self.kv[layer], m.geo.draft_store_rows(layer, blocks, pos))
        names = {
            **{f"tap{k}": m.hidden + 4 * k * H * T for k in range(len(DRAFT_TAPS))},
            **{f"kv_rows{layer}": self.kv[layer] for layer in range(DRAFT_LAYERS)},
            **{f"sink_d{layer}": self.sinks[layer] for layer in range(DRAFT_LAYERS)},
            "cs": self.cs,
            "lp": self.lp,
            "arena": m.arena.address,
        }
        self._op(("d_inject", T), {**m.io, **names}).run(dev.lib)

    # ---- drafting
    def draft(self, rows: list[tuple[int, int, np.ndarray]]) -> np.ndarray:
        """3 greedy drafts for each (anchor token, anchor position, KV block rows) -> (B, 3)"""
        m, dev, d = self.m, self.m.device, self.m.dd
        B, H = len(rows), d.hidden
        if not 0 < B <= MAX_ROWS:
            raise ValueError(f"{B} rows: the drafter takes 1 .. {MAX_ROWS}")
        T = m.bucket(DEPTH * B)
        R = _rows(T)
        # columns 3 r + j: the anchor (j = 0) and mask rows 0, 1 at positions p + j
        tok = np.zeros(T, np.int32)
        pos = np.zeros(T, np.int64)
        blocks = np.full(T, -1, np.int64)
        anchors = np.zeros(R, np.int32)
        for r, (token, p, blk) in enumerate(rows):
            for j in range(DEPTH):
                c = DEPTH * r + j
                pos[c] = p + j
                blocks[c] = blk[(p + j) // PAGE_TOKENS]
            tok[DEPTH * r] = token
            anchors[r] = token
        dev.h2d_nb(self.tok, tok)
        dev.h2d_nb(self.anchors, anchors)
        dev.h2d_nb(self.cs, np.ascontiguousarray(self.rope[pos]))
        for layer in range(DRAFT_LAYERS):
            dev.h2d_nb(self.kv[layer], m.geo.draft_store_rows(layer, blocks, pos))
            slots, lat, rope = self._slot_table(layer, rows, _slots(T))
            dev.h2d_nb(self.slots[layer], slots)
            dev.h2d_nb(self.pages[layer], np.concatenate([lat.ravel(), rope.ravel()]))
        lib, x0, x1 = dev.lib, self.x[0], self.x[1]
        # the block's input: the target's embedding of each anchor, mask rows at the others
        self._op(
            ("embed", T),
            {
                "tok": self.tok,
                "emb_q": m.w["emb_q"],
                "emb_s": m.w["emb_s"],
                "x_out": x0,
            },
        ).run(lib)
        src, size, dst = [], [], []
        for r in range(B):
            for j in (1, 2):
                src.append(m.w["d.mask"] + 4 * H * (j - 1))
                size.append(4 * H)
                dst.append(x0 + 4 * H * (DEPTH * r + j))
        dev.d2d_multi(np.array(src), np.array(size), np.array(dst))
        common = {
            **m.consts,
            **m.io,
            "cs": self.cs,
            "lp": self.lp,
            "arena": m.arena.address,
        }
        # the residual partials, as the target's: all-gathered or all-reduced between launches
        gather = d.gathered(T)
        pa, pb = (self.pa_all, self.pb_all) if gather else (self.pa, self.pb)
        collective = None
        if m.nranks > 1:
            collective = lib.rg_ccl_allgather if gather else lib.rg_ccl_allreduce
        count = H * T
        for layer in range(DRAFT_LAYERS):
            p_in = m.pzero if layer == 0 else pb
            pages = self.pages[layer]
            S = _slots(T)
            self._op(
                ("d_pre", T, layer),
                {**common, "x": x0, "p": p_in, "x_out": x1, "kv_rows": self.kv[layer]},
                layer,
            ).run(lib)
            self._op(
                ("d_slots", T, layer),
                {
                    **common,
                    "slots": self.slots[layer],
                    "lat_pages": pages,
                    "rope_pages": pages + 4 * 16 * S,
                },
                layer,
            ).run(lib)
            self._op(("d_post", T, layer), {**common, "p_out": self.pa}, layer).run(lib)
            if collective is not None and collective(self.pa, pa, count, 0):
                dev.check(-1, "drafter collective")
            self._op(
                ("d_ffn", T, layer),
                {**common, "x": x1, "p": pa, "x_out": x0, "p_out": self.pb},
                layer,
            ).run(lib)
            if collective is not None and collective(self.pb, pb, count, 0):
                dev.check(-1, "drafter collective")
        self._op(
            ("d_logits", T),
            {
                **common,
                "x": x0,
                "p": pb,
                "x_out": x1,
                "hf": self.hf,
                "logits": self.logits,
            },
        ).run(lib)
        if m.nranks > 1:
            dev.allgather(self.logits, self.logits_all, m.d.vocab * T)
        self._op(
            ("d_markov", T),
            {
                **m.consts,
                "logits_all": self.logits_all,
                "hf": self.hf,
                "mk_s0": self.s0,
                "anchors": self.anchors,
                "drafts": self.drafts,
                **{f"mk_cols{k}": self.cols[T, k] for k in range(DEPTH)},
            },
        ).run(lib)
        dev.sync()
        return dev.d2h(self.drafts, (R, DEPTH), np.int32)[:B].copy()

    def _slot_table(
        self, layer: int, rows: list[tuple[int, int, np.ndarray]], S: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The block's dense slots: row r's 3 query columns (query block 3 at qcol0 = 64 * 3 r)
        over keys [kmin, p + 3), kmin = max(0, p - WINDOW): every column sees the whole block
        (qpos0 = p + 2) and the window (``kmin``, slot field 6). Chunks of 2048 keys from kmin's
        page; (8, S) slots and (S, 16) first latent / rotary rows of each chunk's pages."""
        geo, heads = self.m.geo, self.m.dd.heads
        C = L.SLOT_KEYS
        slots = np.zeros((S, 8), np.int32)
        slots[:, 0] = -1
        lat = np.full((S, C // PAGE_TOKENS), -1, np.int32)
        rope = np.full((S, C // PAGE_TOKENS), -1, np.int32)
        s = 0
        for r, (_, p, blk) in enumerate(rows):
            kmin = max(0, p - WINDOW)
            for key0 in range(kmin // PAGE_TOKENS * PAGE_TOKENS, p + DEPTH, C):
                slots[s] = [
                    0,
                    heads * DEPTH * r,
                    p + DEPTH - 1,
                    key0,
                    p + DEPTH,
                    0,
                    kmin,
                    0,
                ]
                for j in range(C // PAGE_TOKENS):
                    page = key0 // PAGE_TOKENS + j
                    if (
                        page * PAGE_TOKENS < p + DEPTH
                        and page < len(blk)
                        and blk[page] >= 0
                    ):
                        lat[s, j] = blk[page] + geo.draft_latent_offset(layer)
                        rope[s, j] = blk[page] + geo.draft_rope_offset(layer)
                s += 1
        if s > S:
            raise ValueError(f"{s} drafter slots exceed {S}")
        return slots, lat, rope


def _slots(T: int) -> int:
    """slots of a drafting bucket's one attention launch: two per row (the window and the block
    may straddle a 2048-key chunk)"""
    return 2 * _rows(T)


__all__ = [
    "DEPTH",
    "DRAFT_LAYERS",
    "DRAFT_TAPS",
    "MAX_ROWS",
    "WINDOW",
    "drafter_config",
    "drafter_dims",
    "graphs",
    "prepare_drafter",
]
