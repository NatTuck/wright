"""GLM-5.3-Flash device weights for one tensor-parallel rank, prepared from the block-FP8 release.

``prepare`` yields ``(name, payload, kind)`` per device tensor:

- ``f32``, ``bf16`` (u16 bits): an array, uploaded as is;
- ``fp8``: ``Fp8(bytes (M, K) tau-permuted along K, block scales (M/128, K/128))`` for a dense
  ``fp8_gemm``;
- ``x8``: ``Fp8(bytes (E, K, M) input-major, K in pair order, block scales (E, K/128, M/128))`` for
  the routed experts' FP8 ``batch_gemm`` over run-time slots.

``device.py`` converts them and requantizes them to per-row scales.

Semantics: ``models/gaudi/tools/glm_np.py``. Rank r of n
holds KDA and MLA heads (rows of q / k / v / q_b / kv_b, columns of o), every routed expert's intermediate
slice, the shared expert's and the dense MLP's widths, and the vocabulary slice of the head. Replicated:
mHC, norms, the router, the DSA indexer, the latent projection and the embedding.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from resin.models.glm5_3_flash.config import Glm53Config
from resin_models_gaudi.loading.convert import (
    BLOCK,
    POOL,
    bf16_rne,
    bf16_to_f32,
    disable_hugepage_madvise,
    kperm,
    pcat,
    tperm,
)
from resin_models_gaudi.loading.source import Safetensors
from resin_models_gaudi.loading.upload import Fp8

disable_hugepage_madvise()

NS = 4  # mHC streams
TEXT = "model.language_model."


@dataclass(frozen=True, slots=True)
class RankSlices:
    kda_heads: slice
    mla_heads: slice
    expert: slice
    """each routed expert's (and the shared expert's) intermediate rows"""
    dense: slice
    vocab: slice

    @classmethod
    def of(cls, cfg: Glm53Config, rank: int, nranks: int) -> RankSlices:
        def part(n: int, align: int) -> slice:
            s = n // nranks
            if s * nranks != n or s % align:
                raise ValueError(f"{n} does not split into {nranks} parts of {align}s")
            return slice(rank * s, (rank + 1) * s)

        if not 0 <= rank < nranks:
            raise ValueError(f"rank {rank} of {nranks}")
        return cls(
            kda_heads=part(cfg.linear_num_heads, 1),
            mla_heads=part(cfg.num_attention_heads, 1),
            expert=part(cfg.moe_intermediate_size, 256),
            dense=part(cfg.intermediate_size, 256),
            vocab=part(cfg.vocab_size, 1),
        )


def pair_rows(n: int) -> np.ndarray:
    """Row order of the routed experts' K, per 128-row block."""
    if n % BLOCK:
        raise ValueError(f"{n} rows are not whole blocks")
    block = np.array([64 * (s % 2) + s // 2 for s in range(BLOCK)])
    return (np.arange(0, n, BLOCK)[:, None] + block[None, :]).ravel()


def _experts(src: Safetensors, cfg: Glm53Config, prefix: str, sl: RankSlices):
    """Every expert's slice: ``x13`` rows [gate | up] (M = 2w, K = H), ``x2`` (M = H, K = w), both
    input-major and pair-ordered along K, as (E, K, M) with (E, K/128, M/128) block scales."""
    H, E = cfg.hidden_size, cfg.n_routed_experts
    w = sl.expert.stop - sl.expert.start
    x13 = np.empty((E, H, 2 * w), np.uint8)
    s13 = np.empty((E, H // BLOCK, 2 * w // BLOCK), np.float32)
    x2 = np.empty((E, w, H), np.uint8)
    s2 = np.empty((E, w // BLOCK, H // BLOCK), np.float32)
    pH, pw = pair_rows(H), pair_rows(w)
    cols = slice(sl.expert.start // BLOCK, sl.expert.stop // BLOCK)

    def one(e: int) -> None:
        x = f"{prefix}{e}."
        g, gs = src.fp8(x + "gate_proj.weight", sl.expert)
        u, us = src.fp8(x + "up_proj.weight", sl.expert)
        x13[e] = np.concatenate([g, u]).T[pH]
        s13[e] = np.concatenate([gs, us]).T
        d, ds = src.fp8(x + "down_proj.weight")
        x2[e] = d[:, sl.expert].T[pw]
        s2[e] = ds[:, cols].T

    list(POOL.map(one, range(E)))
    return Fp8(x13, s13), Fp8(x2, s2)


def _fp8_rows(parts: list[tuple[np.ndarray, np.ndarray]]) -> Fp8:
    return Fp8(
        tperm(pcat([p[0] for p in parts])),
        np.ascontiguousarray(np.concatenate([p[1] for p in parts], 0), np.float32),
    )


def _fp8_cols(src: Safetensors, name: str, cols: slice) -> Fp8:
    w, s = src.fp8(name)
    c = slice(cols.start // BLOCK, cols.stop // BLOCK)
    return Fp8(tperm(np.ascontiguousarray(w[:, cols])), np.ascontiguousarray(s[:, c]))


def _head_rows(sl: slice, dim: int) -> slice:
    return slice(sl.start * dim, sl.stop * dim)


def _attention_mla(
    src: Safetensors, cfg: Glm53Config, sl: RankSlices, A: str, g: str
) -> Iterator[tuple[str, object, str]]:
    H = cfg.hidden_size
    nope, v, R = cfg.qk_nope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank
    nh = cfg.num_attention_heads
    yield (
        g + "w_a8",
        _fp8_rows(
            [
                src.fp8(A + "q_a_proj.weight"),
                src.fp8(A + "kv_a_proj_with_mqa.weight"),
            ]
        ),
        "fp8",
    )
    yield g + "q_a_ln", src.f32(A + "q_a_layernorm.weight"), "f32"
    yield g + "kv_a_ln", src.f32(A + "kv_a_layernorm.weight"), "f32"
    # folded per local head j at load: attention runs in the latent space directly
    #   wqlat rows [512 j, 512 j + 512) = W_UK_j^T q_b_j  (q_lat = wqlat . qr, natural latent order)
    #   wov columns [512 j, ..)         = o_j W_UV_j      (y = wov . o_lat)
    qb = src.fp8_f32(A + "q_b_proj.weight", _head_rows(sl.mla_heads, nope))
    kvb = bf16_to_f32(
        src.bf16(A + "kv_b_proj.weight", _head_rows(sl.mla_heads, nope + v))
    )
    local = sl.mla_heads.stop - sl.mla_heads.start
    kvb = kvb.reshape(local, nope + v, R)
    qb = qb.reshape(local, nope, cfg.q_lora_rank)
    wq = np.concatenate([kvb[j, :nope].T @ qb[j] for j in range(local)])
    yield g + "wqlat", kperm(bf16_rne(wq)), "bf16"
    o = src.fp8_f32(A + "o_proj.weight")  # (H, nh v)
    o = o[:, _head_rows(sl.mla_heads, v)].reshape(H, local, v)
    wov = np.concatenate([o[:, j] @ kvb[j, nope:] for j in range(local)], 1)
    yield g + "wov", kperm(bf16_rne(wov)), "bf16"
    del qb, kvb, o, wq, wov, nh
    ix = A + "indexer."
    # [wk | kpool gate | weights] share the input: one GEMM, rows 128 + 128 + 32
    yield (
        g + "w_ix",
        kperm(
            pcat(
                [
                    src.bf16(ix + "wk.weight"),
                    src.bf16(ix + "index_kpool_compress_gate"),
                    src.bf16(ix + "weights_proj.weight"),
                ]
            )
        ),
        "bf16",
    )
    yield g + "ix_wq", kperm(src.bf16(ix + "wq_b.weight")), "bf16"  # all 32 heads
    yield g + "ix_kn_w", src.f32(ix + "k_norm.weight"), "f32"
    yield g + "ix_kn_b", src.f32(ix + "k_norm.bias"), "f32"
    yield g + "ix_ape", src.f32(ix + "index_kpool_compress_ape"), "f32"  # (4, 128)


def _attention_kda(
    src: Safetensors, cfg: Glm53Config, sl: RankSlices, A: str, g: str
) -> Iterator[tuple[str, object, str]]:
    hd = cfg.linear_head_dim
    rows = _head_rows(sl.kda_heads, hd)
    nl = sl.kda_heads.stop - sl.kda_heads.start
    qkv = [src.bf16(A + f"{c}_proj.weight", rows) for c in "qkv"]
    cw = np.concatenate(
        [src.f32(A + f"{c}_conv1d.weight", rows)[:, 0] for c in "qkv"]
    )  # (3 nl hd, 4)
    yield g + "conv_w", np.ascontiguousarray(cw.T), "f32"  # (4, 3C): channel contiguous
    H = cfg.hidden_size
    # one GEMM from h: [q | k | v (3 C) | f_a | g_a | b (local heads) | 0], padded to 16 rows
    b = src.bf16(A + "b_proj.weight", sl.kda_heads)
    sm = pcat(
        [
            src.bf16(A + "f_a_proj.weight"),
            src.bf16(A + "g_a_proj.weight"),
            b,
            np.zeros((-(2 * hd + nl) % 16, H), np.uint16),
        ]
    )
    yield g + "w_in", kperm(pcat([*qkv, sm])), "bf16"
    # f_b / g_b (C, 128) read their 128 inputs straight from that GEMM's output: zero-padded K
    Ks = sm.shape[0]
    fb = np.zeros((nl * hd, Ks), np.float32)
    fb[:, :hd] = src.f32(A + "f_b_proj.weight", rows)
    gb = np.zeros((nl * hd, Ks), np.float32)
    gb[:, hd : 2 * hd] = src.f32(A + "g_b_proj.weight", rows)
    yield g + "w_fg", np.concatenate([fb, gb]), "f32"
    yield g + "dt_bias", src.f32(A + "dt_bias")[rows].copy(), "f32"
    yield g + "A_log", src.f32(A + "A_log")[sl.kda_heads].copy(), "f32"
    yield g + "o_norm", src.f32(A + "o_norm.weight"), "f32"
    o = src.bf16(A + "o_proj.weight")  # (H, nh hd)
    yield g + "o", kperm(np.ascontiguousarray(o[:, rows])), "bf16"


def _mlp(
    src: Safetensors, cfg: Glm53Config, sl: RankSlices, i: int, M_: str, g: str
) -> Iterator[tuple[str, object, str]]:
    moe = i >= cfg.first_k_dense_replace
    if not moe:
        yield (
            g + "du8",
            _fp8_rows(
                [
                    src.fp8(M_ + "gate_proj.weight", sl.dense),
                    src.fp8(M_ + "up_proj.weight", sl.dense),
                ]
            ),
            "fp8",
        )
        yield g + "dd8", _fp8_cols(src, M_ + "down_proj.weight", sl.dense), "fp8"
        return
    yield g + "router", src.f32(M_ + "gate.weight"), "f32"  # f32 GEMM on the f32 input
    yield g + "router_bias", src.f32(M_ + "gate.e_score_correction_bias"), "f32"
    S_ = M_ + "shared_experts."
    yield (
        g + "sh_gu8",
        _fp8_rows(
            [
                src.fp8(S_ + "gate_proj.weight", sl.expert),
                src.fp8(S_ + "up_proj.weight", sl.expert),
            ]
        ),
        "fp8",
    )
    yield g + "sh_down8", _fp8_cols(src, S_ + "down_proj.weight", sl.expert), "fp8"
    x13, x2 = _experts(src, cfg, M_ + "experts.", sl)
    yield g + "x13", x13, "x8"
    del x13
    yield g + "x2", x2, "x8"


def prepare(
    src: Safetensors,
    cfg: Glm53Config,
    rank: int = 0,
    nranks: int = 1,
    layers: range | None = None,
    mtp: bool = False,
) -> Iterator[tuple[str, object, str]]:
    sl = RankSlices.of(cfg, rank, nranks)
    layers = range(cfg.num_hidden_layers) if layers is None else layers
    yield "emb", src.bf16(TEXT + "embed_tokens.weight"), "bf16"
    yield "head", kperm(src.bf16("lm_head.weight", sl.vocab)), "bf16"
    yield "norm", src.f32(TEXT + "norm.weight"), "f32"
    mtp_layer = cfg.num_hidden_layers
    for i in [*layers, *([mtp_layer] if mtp else [])]:
        L, g = f"{TEXT}layers.{i}.", f"l{i}."
        # mHC (both sites): mapping rows padded 24 -> 32; the MTP layer has one plain residual
        for tag, site in (
            () if i == mtp_layer else (("hca", "hc_attn_"), ("hcf", "hc_ffn_"))
        ):
            fn = src.f32(L + site + "fn")
            pad = np.zeros((32, fn.shape[1]), np.float32)
            pad[: fn.shape[0]] = fn
            yield g + tag + "_fn", pad, "f32"
            yield g + tag + "_base", src.f32(L + site + "base"), "f32"
            yield g + tag + "_scale", src.f32(L + site + "scale"), "f32"
        yield g + "ln_a", src.f32(L + "input_layernorm.weight"), "f32"
        yield g + "ln_m", src.f32(L + "post_attention_layernorm.weight"), "f32"
        A = L + "self_attn."
        if i == mtp_layer or cfg.layer_types[i] == "deepseek_sparse_attention":
            yield from _attention_mla(src, cfg, sl, A, g)
        else:
            yield from _attention_kda(src, cfg, sl, A, g)
        yield from _mlp(src, cfg, sl, i, L + "mlp.", g)
        if i == mtp_layer:
            yield "mtp.enorm", src.f32(L + "enorm.weight"), "f32"
            yield "mtp.hnorm", src.f32(L + "hnorm.weight"), "f32"
            yield "mtp.eh_proj", kperm(src.bf16(L + "eh_proj.weight")), "bf16"
            yield "mtp.sh_norm", src.f32(L + "shared_head.norm.weight"), "f32"
