"""Qwen3.5/3.8 block-FP8 safetensors, re-laid out for the Gaudi2 graphs.

Layout rules:

* bf16 GEMM weights have K in "sigma order" inside each 128-element block (``kperm``).
* FP8 GEMM weights have K in "tau" order (``tperm``).
* The attention output-gate rows are sigma-permuted, and ``o_proj`` sees sigma then tau.
* Gated-delta-net value heads are reordered so value head ``j`` uses key head ``j % nk``.
* RMSNorm weights carry the ``+1`` (except the gated norm), ``A_log`` becomes
  ``ssm_a = -exp(A_log)``.
* OCP-range checkpoint bytes are halved and their block scales doubled; others load as-is.

Weights stream one fused tensor at a time, so host memory stays far below the model size.
"""

from __future__ import annotations

import json
import os
from collections.abc import Collection, Iterator
from pathlib import Path

import numpy as np

from resin.models.qwen3_5.config import Qwen35Config
from resin_models_gaudi.common.weights import DenseWeight, Fp8Weight
from resin_models_gaudi.loading.convert import (
    BLOCK,
    bf16_to_f32,
    halve,
    kperm,
    needs_halving,
    pcat,
    pcopy,
    sigma_perm,
    tperm,
)
from resin_models_gaudi.loading.shard import DenseShard, channels
from resin_models_gaudi.loading.source import Safetensors


class Qwen35Checkpoint:
    """Hugging Face Qwen3.5/3.8 block-FP8 checkpoint in the Gaudi kernel layouts."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        top = json.loads((self.path / "config.json").read_text())
        self.config = Qwen35Config.from_hugging_face(top, allow_tied_embeddings=True)
        text = top.get("text_config", top)
        quantization = top.get("quantization_config") or text.get("quantization_config")
        if not quantization or quantization.get("quant_method") != "fp8":
            raise ValueError(f"{self.path} is not a block-FP8 checkpoint")
        block = quantization.get("weight_block_size", [BLOCK, BLOCK])
        if list(block) != [BLOCK, BLOCK]:
            raise ValueError(f"unsupported FP8 block size {block}")
        self.files = Safetensors(self.path, mode="mmap")
        prefix = "model.language_model."
        self.prefix = (
            prefix if any(k.startswith(prefix) for k in self.files.meta) else "model."
        )
        c = self.config
        r = c.linear_num_value_heads // c.linear_num_key_heads
        nk = c.linear_num_key_heads
        self.value_head_order = np.array(
            [(j % nk) * r + j // nk for j in range(c.linear_num_value_heads)]
        )
        self._halve: bool | None = None

    # ---- raw tensors
    def _layer(self, index: int) -> str:
        if index == self.config.num_hidden_layers:
            return "mtp.layers.0."
        return f"{self.prefix}layers.{index}."

    def _f32(self, name: str) -> np.ndarray:
        array, dtype = self.files.raw(name)
        if dtype == "BF16":
            return bf16_to_f32(array)
        if dtype == "F32":
            return np.array(array)
        raise ValueError(f"{name}: expected bf16 or f32, got {dtype}")

    def _bf16(self, name: str) -> np.ndarray:
        array, dtype = self.files.raw(name)
        if dtype != "BF16":
            raise ValueError(f"{name}: expected bf16, got {dtype}")
        return array

    def _fp8(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        array, dtype = self.files.raw(name)
        if dtype != "F8_E4M3":
            raise ValueError(f"{name}: expected block FP8, got {dtype}")
        scales = self._f32(name + "_scale_inv").copy()
        weight = np.asarray(array)
        if self._halve is None:  # OCP-range bytes (the original release)?
            self._halve = needs_halving(weight)
        if self._halve:
            weight = halve(weight)
            scales *= 2
        return weight, scales

    def _value_rows(self, w: np.ndarray, scales: np.ndarray, lo: int, order=None):
        """Reorder value-head row blocks (128 rows each) starting at row ``lo`` (``order``: the value heads'
        kernel order, by default the whole model's)."""
        order = self.value_head_order if order is None else order
        nv = len(order)
        rows = np.arange(w.shape[0])
        heads = rows[lo : lo + nv * BLOCK].reshape(nv, BLOCK)[order]
        rows = np.concatenate([rows[:lo], heads.reshape(-1), rows[lo + nv * BLOCK :]])
        blocks = np.arange(scales.shape[0])
        b0 = lo // BLOCK
        blocks = np.concatenate(
            [
                blocks[:b0],
                blocks[b0 : b0 + nv][order],
                blocks[b0 + nv :],
            ]
        )
        return w[rows], scales[blocks]

    def _fp8_cat(self, parts: list[tuple[np.ndarray, np.ndarray]]) -> Fp8Weight:
        w = pcat([p[0] for p in parts])
        return Fp8Weight(
            tperm(w), np.ascontiguousarray(np.concatenate([p[1] for p in parts], 0))
        )

    def _norm(self, name: str) -> DenseWeight:
        return DenseWeight(np.ascontiguousarray(self._f32(name) + np.float32(1.0)))

    def head(self) -> np.ndarray:
        """The LM head (vocab, hidden) as bf16 bits in natural order."""
        if self.config.tie_word_embeddings:
            return self._bf16(self.prefix + "embed_tokens.weight")
        return self._bf16("lm_head.weight")

    def shard(self, rank: int, nranks: int) -> DenseShard:
        """Rank ``rank`` of ``nranks``'s part of the decoder (``loading.shard``)."""
        c = self.config
        return DenseShard(
            rank,
            nranks,
            heads=c.num_attention_heads,
            groups=c.num_key_value_heads,
            nk=c.linear_num_key_heads,
            nv=c.linear_num_value_heads,
            ffn=c.intermediate_size,
        )

    def _fp8_rows(self, name: str, rows: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        """An FP8 weight's rows ``rows`` (whole 128-row blocks; None: all) with their block scales."""
        w, s = self._fp8(name)
        if rows is None:
            return w, s
        return np.ascontiguousarray(w[rows]), np.ascontiguousarray(s[rows[::BLOCK] // BLOCK])

    def _fp8_cols(self, name: str, cols: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        """An FP8 weight's columns ``cols`` (whole 128-column blocks; None: all) with their block scales."""
        w, s = self._fp8(name)
        if cols is None:
            return w, s
        return np.ascontiguousarray(w[:, cols]), np.ascontiguousarray(s[:, cols[::BLOCK] // BLOCK])

    # ---- prepared weights
    def weights(
        self,
        *,
        mtp: bool = False,
        layers: Collection[int] | None = None,
        shard: DenseShard | None = None,
    ) -> Iterator[tuple[str, Fp8Weight | DenseWeight]]:
        """Yield ``(name, weight)`` one fused device tensor at a time; ``layers`` limits the
        decoder layers prepared (the embeddings and output norm always are). With ``shard``, a
        tensor-parallel rank's part of every decoder layer (the MTP layer stays whole)."""
        c = self.config
        p = self.prefix
        embed = self._bf16(p + "embed_tokens.weight")
        yield "E_gather", DenseWeight(pcopy(np.empty(embed.shape, embed.dtype), embed))
        head = embed if c.tie_word_embeddings else self._bf16("lm_head.weight")
        yield "E_head", DenseWeight(kperm(head))
        yield "out_norm", self._norm(p + "norm.weight")
        gate_perm = sigma_perm(c.head_dim)
        nv, nk, dk = (
            c.linear_num_value_heads,
            c.linear_num_key_heads,
            c.linear_key_head_dim,
        )
        indices = list(range(c.num_hidden_layers))
        if mtp:
            c.require_mtp()
            indices.append(c.num_hidden_layers)
        if layers is not None:
            indices = [i for i in indices if i in layers]
        full = set(c.full_layer_ids)
        r = nv // nk
        for i in indices:
            layer = self._layer(i)
            sh = None if i == c.num_hidden_layers else shard  # the MTP layer stays whole
            hd = c.head_dim
            heads = np.arange(c.num_attention_heads) if sh is None else sh.attn_heads
            groups = np.arange(c.num_key_value_heads) if sh is None else sh.attn_groups
            keys = np.arange(nk) if sh is None else sh.key_heads
            values = np.arange(nv) if sh is None else sh.value_heads
            ffn = (0, c.intermediate_size) if sh is None else sh.ffn_range
            nk_l, nv_l = len(keys), len(values)
            order = np.array([(j % nk_l) * r + j // nk_l for j in range(nv_l)])
            yield f"l{i}.attn_norm", self._norm(layer + "input_layernorm.weight")
            yield (
                f"l{i}.post_norm",
                self._norm(layer + "post_attention_layernorm.weight"),
            )
            if i in full or i == c.num_hidden_layers:
                q, qs = self._fp8_rows(
                    layer + "self_attn.q_proj.weight",
                    None if sh is None else channels(heads, 2 * hd),
                )
                q = q.copy()
                for h in range(len(heads)):  # gate rows sigma-permuted
                    g0 = 2 * hd * h + hd
                    q[g0 : g0 + hd] = q[g0 : g0 + hd][gate_perm]
                kv_rows = None if sh is None else channels(groups, hd)
                yield (
                    f"l{i}.w_qkv",
                    self._fp8_cat(
                        [
                            (q, qs),
                            self._fp8_rows(layer + "self_attn.k_proj.weight", kv_rows),
                            self._fp8_rows(layer + "self_attn.v_proj.weight", kv_rows),
                        ]
                    ),
                )
                o, os_ = self._fp8_cols(
                    layer + "self_attn.o_proj.weight",
                    None if sh is None else channels(heads, hd),
                )
                yield f"l{i}.w_o", Fp8Weight(tperm(kperm(o)), os_)
                qk = np.concatenate(
                    [
                        self._f32(layer + "self_attn.q_norm.weight"),
                        self._f32(layer + "self_attn.k_norm.weight"),
                    ]
                )
                yield (
                    f"l{i}.qk_norm",
                    DenseWeight(np.ascontiguousarray(qk + np.float32(1.0))),
                )
            else:
                a = layer + "linear_attn."
                # q, k (key heads) and v (their value heads) channels of the conv and the projection
                qkv_rows = np.concatenate(
                    [
                        channels(keys, dk),
                        nk * dk + channels(keys, dk),
                        2 * nk * dk + channels(values, dk),
                    ]
                )
                qkv = self._value_rows(
                    *self._fp8_rows(
                        a + "in_proj_qkv.weight", None if sh is None else qkv_rows
                    ),
                    2 * nk_l * dk,
                    order,
                )
                z = self._value_rows(
                    *self._fp8_rows(
                        a + "in_proj_z.weight",
                        None if sh is None else channels(values, dk),
                    ),
                    0,
                    order,
                )
                yield f"l{i}.w_in", self._fp8_cat([qkv, z])
                ba = pcat(
                    [
                        self._bf16(a + "in_proj_b.weight")[values][order],
                        self._bf16(a + "in_proj_a.weight")[values][order],
                    ]
                )
                yield f"l{i}.w_ba", DenseWeight(kperm(ba))
                o, os_ = self._fp8_cols(
                    a + "out_proj.weight",
                    None if sh is None else channels(values, dk),
                )
                m = o.shape[0]
                o = np.ascontiguousarray(
                    o.reshape(m, nv_l, dk)[:, order].reshape(m, -1)
                )
                yield (
                    f"l{i}.w_out",
                    Fp8Weight(tperm(o), np.ascontiguousarray(os_[:, order])),
                )
                conv = self._f32(a + "conv1d.weight").reshape(c.linear_conv_dim, -1)
                conv = conv[qkv_rows]
                lo = 2 * nk_l * dk
                conv = np.concatenate(
                    [
                        conv[:lo],
                        conv[lo:].reshape(nv_l, dk, -1)[order].reshape(nv_l * dk, -1),
                    ]
                )
                yield f"l{i}.conv_w", DenseWeight(np.ascontiguousarray(conv.T))
                a_log = self._f32(a + "A_log").astype(np.float64)[values]
                yield (
                    f"l{i}.ssm_a",
                    DenseWeight((-np.exp(a_log))[order].astype(np.float32)),
                )
                yield (
                    f"l{i}.dt_bias",
                    DenseWeight(
                        np.ascontiguousarray(self._f32(a + "dt_bias")[values][order])
                    ),
                )
                yield (
                    f"l{i}.ssm_norm",
                    DenseWeight(np.ascontiguousarray(self._f32(a + "norm.weight"))),
                )
            ffn_rows = None if sh is None else np.arange(*ffn)
            yield (
                f"l{i}.w_gu",
                self._fp8_cat(
                    [
                        self._fp8_rows(layer + "mlp.gate_proj.weight", ffn_rows),
                        self._fp8_rows(layer + "mlp.up_proj.weight", ffn_rows),
                    ]
                ),
            )
            d, ds = self._fp8_cols(layer + "mlp.down_proj.weight", ffn_rows)
            yield f"l{i}.w_down", Fp8Weight(tperm(d), ds)
        if mtp:
            fc = self._bf16("mtp.fc.weight")
            yield "mtp.fc", DenseWeight(kperm(fc))
            for name, source in (
                ("enorm", "mtp.pre_fc_norm_embedding.weight"),
                ("hnorm", "mtp.pre_fc_norm_hidden.weight"),
                ("norm", "mtp.norm.weight"),
            ):
                yield f"mtp.{name}", self._norm(source)
