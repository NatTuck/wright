"""The Qwen3.8-27B vision tower as a Synapse graph, on the card.

Companion to `vision.py` (the CPU reference, and the oracle this is diffed against).
Everything below runs on device: GEMMs on the MME, and the small TPC kernels of ours for
the operations the stock library would not accept (`rg_ln_pre_f32`, `rg_ln_affine_f32`,
`rg_softmax_rows_f32`, `rg_gelu_tanh_f32`, `rg_add_f32`, `rg_bias_f32`,
`rg_resid_bias_f32`, `rg_rope_vision_f32`, `rg_merge4_f32`).

Layout, from `layers.py`: activations are (features, tokens) with dim 0 contiguous, so host
arrays are bound feature-major. A checkpoint weight is (out, in), and `gemm(..., nt=False)`
takes exactly that: y(M, N) = x(K, N) . w(M, K)^T with M = rows.

Attention, head by head (the fused qkv weight is split on the host):
    q_h = gemm(a, Wq_h, 72)            (72, tokens)
    k_h = gemm(a, Wk_h, 72)            (72, tokens)
    v_h = gemm(a, Wv_h, 72)            (72, tokens)
    S   = gemm(q_h, k_h, rows=tokens, nt=True)   y[m, n] = q_n . k_m, so m indexes keys
    P   = softmax over dim 0 of S      (keys), which is what rg_softmax_rows_f32 does
    O_h = gemm(P, v_h, 72)             (72, queries) = sum_m P[m, q] v[:, m]
    proj = gemm(O_h, Wp_h, hidden)     and the heads are summed, which is the projection
                                       applied to the concatenated heads.

The whole tower is one graph and one launch. `stage` exists so a single stage can be read
back and diffed on its own; "full" is the production path and is the one that is verified.
"""

from __future__ import annotations

import os
import struct

from resin_models_gaudi.qwen3_5.layers import F32, GEMM_NN, LayerGraph

HEAD_DIM = 72  # the checkpoint's head width
PADDED = 128  # head width padded to a multiple of 64 for our 64-lane kernels; the padding
              # is zeros, so it contributes nothing to the scores, the values or the projection


class _Dims:
    """LayerGraph wants the decoder's dims; the tower only needs `hidden`."""

    def __init__(self, hidden: int) -> None:
        self.hidden = hidden
        self.vocab = 0
        self.eps = 1e-6


def _gemm(lg: LayerGraph, act, name: str, rows: int, nt: bool = False):
    """y(rows, tokens) = act(K, tokens) . weight(rows, K)^T  (nt=False: checkpoint order)."""
    w = lg.input(name, (rows, act.shape[0]), F32)
    return lg.gemm(act, w, rows, nt=nt)


def _node1(lg: LayerGraph, guid: str, act, shape, params: bytes = b""):
    out = lg.tensor(shape, F32)
    lg.node(guid, [act], [out], params)
    return out


def _bias(lg: LayerGraph, act, name: str, rows: int, tokens: int):
    """Add the layer's bias (per row) to every column."""
    b = lg.input(name, (rows,), F32)
    out = lg.tensor((rows, tokens), F32)
    lg.node("rg_bias_f32", [act, b], [out])
    return out


def _resid_bias(lg: LayerGraph, x, m, name: str, rows: int, tokens: int):
    """x + m + bias[row] in one node, so a residual costs one TPC pass instead of two."""
    b = lg.input(name, (rows,), F32)
    out = lg.tensor((rows, tokens), F32)
    lg.node("rg_resid_bias_f32", [x, m, b], [out])
    return out


def _ln_pre(lg: LayerGraph, act, off, name: str, hidden: int, tokens: int):
    """LayerNorm(act + off): the position add fused into the norm, so the norm reads the
    patch GEMM's output directly."""
    w = lg.input(name + ".weight", (hidden,), F32)
    b = lg.input(name + ".bias", (hidden,), F32)
    out = lg.tensor((hidden, tokens), F32)
    lg.node("rg_ln_pre_f32", [act, off, w, b], [out], struct.pack("f", 1e-6))
    return out


def _ln(lg: LayerGraph, act, name: str, hidden: int, tokens: int):
    w = lg.input(name + ".weight", (hidden,), F32)
    b = lg.input(name + ".bias", (hidden,), F32)
    out = lg.tensor((hidden, tokens), F32)
    lg.node("rg_ln_affine_f32", [act, w, b], [out], struct.pack("f", 1e-6))
    return out


def _add(lg: LayerGraph, a, b, hidden: int, tokens: int):
    out = lg.tensor((hidden, tokens), F32)
    lg.node("rg_add_f32", [a, b], [out])
    return out


def _rope(lg: LayerGraph, act, cos, sin, shape):
    """Axial vision rope on q or k: y = x*cos + rotate_half(x)*sin, per token."""
    out = lg.tensor(shape, F32)
    lg.node("rg_rope_vision_f32", [act, cos, sin], [out])
    return out


def _copy(lg: LayerGraph, act, shape, tokens: int):
    """Hand a pooled tensor back through a bound output, so the harness can read it."""
    out = lg.input("out", shape, F32)
    lg.node("rg_add_f32", [act, lg.input("zeros", shape, F32)], [out])
    return out


def vision_graph(device, config, patches: int, merged: int, tokens: int, width: int,
                 stage: str = "full") -> LayerGraph:
    """Build the tower. ``patches`` = temporal*h*w, ``merged`` = patches/4, ``width`` = w.

    Stages, each writing its result to the bound ``out`` tensor: ``embed`` (patch GEMM plus
    position), ``ln1``, ``q0`` (first head's scaled q), ``scores`` (first head's q.k),
    ``o<n>`` (head n's attention output), ``head<n>`` (head n's projection), ``solo<n>``
    (head n built alone), ``gdirect<n>`` (head n's projection GEMM writing straight into a
    bound tensor), ``sum<n>`` (the first n heads accumulated), ``attn`` (all heads summed,
    before the projection bias), ``block`` (block 0 in full). ``full`` writes ``outb``.
    """
    hidden, heads, depth = config.hidden, config.heads, config.depth
    inter = config.intermediate
    tag = os.environ.get("VISION_TAG", "")
    probe = 0     # which head o<n> / head<n> / gdirect<n> reads back
    only = None   # solo<n>: build that head alone
    top = None    # sum<n>: build the first n heads and hand back their sum
    if stage.startswith("sum") and stage[3:].isdigit():
        top = int(stage[3:])
    if stage.startswith("solo") and stage[4:].isdigit():
        probe = only = int(stage[4:])
    elif stage.startswith("gdirect"):
        probe = int(stage[7:]) if stage[7:].isdigit() else 0
    elif stage.startswith("head") and stage[4:].isdigit():
        probe = int(stage[4:])
    elif stage.startswith("o") and stage[1:].isdigit():
        probe = int(stage[1:])
    lg = LayerGraph(device, f"qwen35_vision_N{patches}_P{PADDED}_D{depth}_{stage}{tag}",
                    _Dims(hidden), tokens)

    # ---- patch embed. `pos` carries the embed's bias and the interpolated position table,
    # summed on the host, so the tower needs one add rather than two.
    pix = lg.input("pix", (config.patch_features, patches), F32)
    h = _gemm(lg, pix, "patch_embed.proj.weight", hidden)  # (hidden, patches)
    pos = lg.input("pos_bias", (hidden, patches), F32)
    if stage == "embed":
        _copy(lg, _add(lg, h, pos, hidden, tokens), (hidden, patches), tokens)
        return lg
    x = _add(lg, h, pos, hidden, tokens)  # the residual stream the blocks add into

    for i in range(depth):
        if i == 0:
            # block 0's norm takes the position offset instead of reading the sum, which
            # saves a pass and keeps the norm off the add's output
            a = _ln_pre(lg, h, pos, f"blocks.{i}.norm1", hidden, tokens)
        else:
            a = _ln(lg, x, f"blocks.{i}.norm1", hidden, tokens)
        if stage == "ln1":
            _copy(lg, a, (hidden, tokens), tokens)
            return lg

        cos = lg.input("rope_cos", (PADDED, tokens), F32)
        sin = lg.input("rope_sin", (PADDED, tokens), F32)
        proj = None
        for hd in ([only] if only is not None else (range(top) if top else range(heads))):
            q = _bias(lg, _gemm(lg, a, f"blocks.{i}.attn.q{hd}", PADDED),
                      f"blocks.{i}.attn.qb{hd}", PADDED, tokens)
            k = _bias(lg, _gemm(lg, a, f"blocks.{i}.attn.k{hd}", PADDED),
                      f"blocks.{i}.attn.kb{hd}", PADDED, tokens)
            v = _bias(lg, _gemm(lg, a, f"blocks.{i}.attn.v{hd}", PADDED),
                      f"blocks.{i}.attn.vb{hd}", PADDED, tokens)
            if stage == "q0" and hd == 0:
                _copy(lg, q, (PADDED, tokens), tokens)
                return lg
            if stage == "scores" and hd == 0:
                y = lg.input("out", (tokens, tokens), F32)
                lg.node("gemm", [q, k], [y], b"\x00\x01")  # y[m, n] = q_n . k_m
                return lg
            # rope after the bias, on q and k only (v supplies the values and is not rotated)
            q = _rope(lg, q, cos, sin, (PADDED, tokens))
            k = _rope(lg, k, cos, sin, (PADDED, tokens))
            s = lg.gemm(q, k, tokens, nt=True)  # (keys, queries)
            p = _node1(lg, "rg_softmax_rows_f32", s, (tokens, tokens))
            o = lg.gemm(p, v, PADDED, nt=False)
            if stage.startswith("o") and stage[1:].isdigit() and hd == probe:
                _copy(lg, o, (PADDED, tokens), tokens)
                return lg
            head = _gemm(lg, o, f"blocks.{i}.attn.proj{hd}", hidden)
            if hd == probe and (stage.startswith("head") or stage.startswith("solo")
                                or stage.startswith("gdirect")):
                if stage.startswith("gdirect"):
                    # the projection GEMM writing straight into a bound tensor: no pooled
                    # output and no copy node in the way
                    y = lg.input("out", (hidden, tokens), F32)
                    w = lg.input(f"blocks.{i}.attn.proj{hd}", (hidden, PADDED), F32)
                    lg.node("gemm", [o, w], [y], GEMM_NN)
                    return lg
                if stage.startswith("solo"):
                    _copy(lg, o, (PADDED, tokens), tokens)  # o and head from one launch
                _copy(lg, head, (hidden, tokens), tokens)
                return lg
            proj = head if proj is None else _add(lg, proj, head, hidden, tokens)

        if top is not None or stage == "attn":
            _copy(lg, proj, (hidden, tokens), tokens)
            return lg
        x = _resid_bias(lg, x, proj, f"blocks.{i}.attn.proj.bias", hidden, tokens)

        m = _ln(lg, x, f"blocks.{i}.norm2", hidden, tokens)
        m = _bias(lg, _gemm(lg, m, f"blocks.{i}.mlp.fc1", inter), f"blocks.{i}.mlp.b1",
                  inter, tokens)
        m = _node1(lg, "rg_gelu_tanh_f32", m, (inter, tokens))
        m = _gemm(lg, m, f"blocks.{i}.mlp.fc2", hidden)
        x = _resid_bias(lg, x, m, f"blocks.{i}.mlp.b2", hidden, tokens)

    if stage == "block":
        _copy(lg, x, (hidden, tokens), tokens)
        return lg

    # ---- merger: norm, 2x2 spatial merge, then two GEMMs with a gelu between
    x = _ln(lg, x, "merger.norm", hidden, tokens)
    rows = 4 * hidden  # the merger's input width, not the merged-token count
    # patches are already in merge-block order, so this is four consecutive columns into one
    # row - a strided copy on the card, not the 2x2 gather the raster assumption needed
    x = _node1(lg, "rg_merge4_f32", x, (rows, merged))
    x = _bias(lg, _gemm(lg, x, "merger.fc1", rows), "merger.b1", rows, merged)
    x = _node1(lg, "rg_gelu_tanh_f32", x, (rows, merged))
    w = lg.input("merger.fc2", (config.out_hidden, rows), F32)
    y = lg.input("out", (config.out_hidden, merged), F32)
    lg.node("gemm", [x, w], [y], GEMM_NN)
    out = lg.input("outb", (config.out_hidden, merged), F32)
    lg.node("rg_bias_f32", [y, lg.input("merger.b2", (config.out_hidden,), F32)], [out])
    lg.out = out
    return lg
