"""Host-side pieces of the DSpine drafter of Qwen3.8 (no card): the configuration checks, the
serial refinement's tables, the rescored-logit walk ``dsp_select`` implements against the
reference's redone write, and the block step's slot launches over the 8192-position window."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from test_qwen_dflash_plan import TARGET

from resin.models.qwen3_5.config import Qwen35Config
from resin_models_gaudi.common import dflash
from resin_models_gaudi.common.dflash import DrafterConfig, DraftRow
from resin_models_gaudi.loading.convert import bf16_rne, bf16_to_f32

TOOLS = Path(__file__).resolve().parents[1] / "tools"

DRAFTER = {
    "architectures": ["DFlashDraftModel"],
    "model_type": "qwen3",
    "block_size": 16,
    "is_causal": True,
    "use_sliding_window": True,
    "sliding_window": 8192,
    "layer_types": ["sliding_attention"] * 5,
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "vocab_size": 248320,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "num_hidden_layers": 5,
    "num_target_layers": 64,
    "rms_norm_eps": 1e-6,
    "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
    "code_bridge_second_pass": {"first_exact": 4, "head_topk": 16, "mode": "serial_table"},
    "dflash_config": {
        "draft_block_causal": True,
        "mask_token_id": 248070,
        "selected_prefix_code_bridge": {"code_size": 1024, "eps": 1e-6, "kappa": 10.0, "rank": 512},
        "selected_prefix_feature_layers": [1, 2, 3, 4, 5],
        "selected_prefix_feature_write_mode": "independent",
        "selected_prefix_feature_writes": True,
        "selected_prefix_last_write_refine": {"head_topk": 16, "kind": "exact_sender", "version": 1, "weight": 1.0},
        "selected_prefix_terminal": False,
        "target_layer_ids": [1, 16, 31, 46, 61],
    },
}  # fmt: skip


@pytest.fixture
def target() -> Qwen35Config:
    return Qwen35Config.from_hugging_face(TARGET, allow_tied_embeddings=True)


def write_drafter(path: Path, **changes) -> Path:
    config = json.loads(json.dumps(DRAFTER))
    for key, value in changes.items():
        if key in config["dflash_config"] or key.startswith("selected_prefix_"):
            config["dflash_config"][key] = value
        elif key.startswith("dflash_config."):
            config["dflash_config"][key.removeprefix("dflash_config.")] = value
        else:
            config[key] = value
    (path / "config.json").write_text(json.dumps(config))
    return path


@pytest.fixture
def cfg(tmp_path, target) -> DrafterConfig:
    return DrafterConfig.load(write_drafter(tmp_path), target)


def test_config_reads_the_release(cfg, tmp_path):
    assert (cfg.kind, cfg.causal, cfg.block, cfg.drafts) == ("dspine", True, 16, 15)
    assert (cfg.window, cfg.taps) == (8192, (1, 16, 31, 46, 61))
    assert (cfg.code_size, cfg.rank, cfg.top_k, cfg.bridge_eps) == (1024, 512, 16, 1e-6)
    assert (cfg.conv_kernel, cfg.groups, cfg.kp_rows) == (0, 0, 0)
    assert dflash.drafter_draft_tokens(tmp_path) == 15


@pytest.mark.parametrize(
    "changes",
    [
        {"is_causal": False},
        {"draft_block_causal": False},
        {"selected_prefix_terminal": True},
        {"selected_prefix_feature_write_mode": "shared"},
        {"selected_prefix_feature_layers": [2, 3, 4, 5]},
        {"selected_prefix_last_write_refine": {"head_topk": 16, "kind": "other"}},
        {
            "selected_prefix_last_write_refine": {
                "head_topk": 32,
                "kind": "exact_sender",
            }
        },
        {
            "code_bridge_second_pass": {
                "first_exact": 3,
                "head_topk": 16,
                "mode": "serial_table",
            }
        },
        {"selected_prefix_code_bridge": {"code_size": 1024, "eps": 1e-6, "rank": 256}},
        {"selected_prefix_terminal_residual_rank": 8},
        {"dflash_config.conv_kernel_size": 2},
        {"block_size": 32},
        {"hidden_size": 4096},
    ],
)
def test_config_rejects_other_drafters(tmp_path, target, changes):
    with pytest.raises(ValueError):
        DrafterConfig.load(write_drafter(tmp_path, **changes), target)


def test_tables_follow_their_definition():
    """tg = bf16(head * norm) . write_last, tsv = [pg | prev] . bf16(sqrt(C) codebook), by
    chunk as on the full vocabulary."""
    rng = np.random.default_rng(0)
    V, H, C, R = 100, 64, 32, 8
    head = bf16_rne(rng.standard_normal((V, H)).astype(np.float32))
    codebook = bf16_rne(rng.standard_normal((V, C)).astype(np.float32) * 0.1)
    norm = rng.uniform(0.5, 1.5, H).astype(np.float32)
    write = bf16_to_f32(bf16_rne(rng.standard_normal((H, R)).astype(np.float32)))
    sender = bf16_to_f32(bf16_rne(rng.standard_normal((2 * R, C)).astype(np.float32)))
    tg, tsv = dflash.dspine_tables(head, norm, write, sender, codebook, chunk=37)
    h = bf16_to_f32(bf16_rne(bf16_to_f32(head) * norm))
    np.testing.assert_allclose(bf16_to_f32(tg), h @ write, rtol=1e-2, atol=1e-2)
    code = bf16_to_f32(bf16_rne(bf16_to_f32(codebook) * np.float32(math.sqrt(C))))
    np.testing.assert_allclose(bf16_to_f32(tsv), code @ sender.T, rtol=1e-2, atol=1e-2)


def test_rescored_walk_equals_the_redone_write():
    """dsp_select scores candidate j by ratio * logit_j + tg[j] . (m - mp), the forward pass's
    logits rescaled to its last residual plus the write's change; the reference redoes the last
    write and rescores with the LM head. Both pick the same tokens on a small random drafter."""
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(TOOLS))
    from qwen_dspine_ref import DSpineConfig, DSpineRef, Layer, Weights

    torch.manual_seed(0)
    H, C, R, V, T, NH, KV, D, F = 128, 32, 16, 400, 6, 4, 2, 32, 96
    cfg = DSpineConfig(
        hidden=H, ffn=F, vocab=V, heads=NH, kv_heads=KV, head_dim=D, layers=2,
        eps=1e-6, rope_theta=1e4, window=64, block_size=T, mask_token_id=V - 1,
        code_size=C, rank=R, bridge_eps=1e-6, head_topk=8,
        target_layer_ids=(0, 1), num_target_layers=2,
    )  # fmt: skip

    def rnd(*shape, s=0.2):
        return (torch.randn(*shape) * s).bfloat16()

    layers = [
        Layer(
            attn_norm=1 + rnd(H, s=0.1), post_norm=1 + rnd(H, s=0.1),
            q_norm=1 + rnd(D, s=0.1), k_norm=1 + rnd(D, s=0.1),
            wq=rnd(NH * D, H, s=0.05), wk=rnd(KV * D, H, s=0.05), wv=rnd(KV * D, H, s=0.05),
            wo=rnd(H, NH * D, s=0.05), w_gate=rnd(F, H, s=0.05), w_up=rnd(F, H, s=0.05),
            w_down=rnd(H, F, s=0.05), read=rnd(C, H, s=0.1), write=rnd(H, R, s=0.1),
        )
        for _ in range(cfg.layers)
    ]  # fmt: skip
    w = Weights(
        fc=rnd(H, 2 * H, s=0.05), hidden_norm=1 + rnd(H, s=0.1), layers=layers,
        norm=1 + rnd(H, s=0.1), codebook=rnd(V, C, s=0.3), gate=rnd(R, C),
        gate_bias=rnd(R), pred_gate=rnd(R, C), prev=rnd(R, C),
        embed=rnd(V, H, s=1.0), head=rnd(V, H, s=0.3),
    )  # fmt: skip
    ref = DSpineRef(cfg, w)
    ref.inject(torch.randn(20, 2 * H), torch.arange(20))
    gb = w.gate_bias.float()
    last = layers[-1].write.float()
    tg = (w.head.float() * w.norm.float()) @ last  # exact, no bf16 rounding
    tsv_g, tsv_v = (
        torch.stack([w.pred_gate.float() @ ref.code(v) for v in range(V)]),
        torch.stack([w.prev.float() @ ref.code(v) for v in range(V)]),
    )
    for anchor in (3, 77, 150):
        hidden, stash = ref.block_forward(anchor, 20)
        cands = ref.candidates(hidden)
        want = ref.refine(stash, cands, anchor)
        out = hidden  # rms_norm(x_fin) * norm
        x_fin = stash.raw + stash.rms[:, None] * (
            torch.cat(
                [
                    torch.zeros(1, R),
                    torch.sigmoid(
                        stash.packed[1:, :R] + stash.packed[:-1, R : 2 * R] + gb
                    )
                    * stash.packed[:-1, 2 * R :],
                ]
            )
            @ last.T
        )
        prev, got = anchor, []
        for r in range(T - 1):
            col = r + 1
            ratio = torch.sqrt(x_fin[col].square().mean() + cfg.eps) / stash.rms[col]
            q = stash.packed[col, :R] + gb
            mp = (
                torch.sigmoid(q + stash.packed[col - 1, R : 2 * R])
                * stash.packed[col - 1, 2 * R :]
            )
            m = torch.sigmoid(q + tsv_g[prev]) * tsv_v[prev]
            logits = w.head[cands[r]].float() @ out[col]
            scores = ratio * logits + tg[cands[r]] @ (m - mp)
            prev = int(cands[r, int(torch.argmax(scores))])
            got.append(prev)
        assert got == want.tolist(), (anchor, got, want.tolist())


def test_block_slots_reach_back_a_window_of_8192(cfg):
    """A block at 10000: the window [1809, 10000) spans five 2048-key chunks (0 .. 8192), and the
    block is a sixth slot per pair; one row's launches fit the bound."""
    blocks = 50_000 + np.arange(100) * 5120  # logical blocks 0 .. 99
    row = DraftRow(10_000, blocks, 0)
    scratch = np.arange(1 * 5 * 4 * 2).reshape(1, 5, 4, 2) * 128 + 900_000
    S = dflash.slots_per_launch(1, cfg.pairs)
    launches = dflash.plan_block_slots(cfg, [row], scratch, S)
    slots = np.concatenate([lnch[0] for lnch in launches])
    real = slots[slots[:, 0] >= 0]
    assert len(real) == 4 * (5 + 1)
    assert sorted({int(v) for v in real[:, 3]}) == [0, 2048, 4096, 6144, 8192, 10_000]
    assert (real[:, 6] == [1809] * 20 + [10_000] * 4).all()
    assert (real[:, 7] == 8192).all()
    assert (
        real[-4:, 4] == 10_000 + 16
    ).all()  # the block: its 16 keys, causally masked
    assert len(launches) <= cfg.window // dflash.SLOT_KEYS + 2
    # the widest batch: every row's launches still fit the bound
    B = 32
    scratch = np.arange(B * 5 * 4 * 2).reshape(B, 5, 4, 2) * 128 + 900_000
    rows = [DraftRow(10_000 + 37 * b, blocks, 0) for b in range(B)]
    launches = dflash.plan_block_slots(
        cfg, rows, scratch, dflash.slots_per_launch(B, cfg.pairs)
    )
    assert len(launches) <= cfg.window // dflash.SLOT_KEYS + 2


def test_verify_width_follows_the_expected_rate():
    """The native executor's choose_width: few accepted drafts take a narrow step, every draft accepted the
    widest; a batch weighs all its rows (one row's long acceptance does not widen eight rows' steps)."""
    native = pytest.importorskip("resin._resin")
    if not hasattr(native, "QwenDraftTables"):
        pytest.skip("resin built without the gaudi feature")
    from resin_models_gaudi.qwen3_5.model import verify_mode

    def choose_width(widths, accepted, width, bucket=1):
        return native.QwenDraftTables.choose_width(
            list(widths), [int(a) for a in accepted], width, bucket
        )

    widths = (4, 8, 16)
    assert choose_width(widths, np.array([0]), 16) == 8
    assert choose_width(widths, np.array([15]), 16) == 16
    assert choose_width(widths, np.array([7]), 8) == 16  # all accepted at 8: grow
    assert choose_width(widths, np.array([0, 1, 2, 3, 0, 1, 15, 2]), 16, 8) == 4
    assert choose_width(widths, np.array([15] * 8), 16, 8) == 16
    assert choose_width((16,), np.array([0]), 16) == 16
    assert (verify_mode(8, 16, 16), verify_mode(8, 4, 16)) == ("v8", "v8w4")


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> Path:
    """A one-file checkpoint (dtypes from the arrays: uint16 as BF16)."""
    names = {np.dtype(np.uint16): "BF16", np.dtype(np.float32): "F32"}
    names |= {np.dtype(np.uint8): "U8", np.dtype(np.int64): "I64"}
    header, blobs, offset = {}, [], 0
    for name, array in tensors.items():
        data = np.ascontiguousarray(array).tobytes()
        header[name] = {
            "dtype": names[array.dtype],
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    text = json.dumps(header).encode()
    text += b" " * (-len(text) % 8)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "model.safetensors", "wb") as f:
        f.write(len(text).to_bytes(8, "little") + text + b"".join(blobs))
    return path


def fp8_inc(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The quantizer's form: per-tensor scale amax / 240, e4m3 bytes of x / scale."""
    from resin_models_gaudi.loading.convert import e4m3_encode

    scale = np.float32(np.abs(x).max() / 240.0)
    return e4m3_encode(x / scale), np.array([scale], np.float32)


def test_fp8_inc_drafter_tensors(tmp_path):
    """DrafterTensors detects an fp8_inc checkpoint and reads it: dense tensors dequantized,
    GEMM weights from the stored bytes with each part's scale on its row blocks."""
    from resin_models_gaudi.loading.convert import E4M3, kperm, tau_perm

    rng = np.random.default_rng(3)
    a = rng.standard_normal((256, 512)).astype(np.float32)
    b = rng.standard_normal((128, 512)).astype(np.float32) * 5
    norm = rng.standard_normal(512).astype(np.float32)
    tensors = {}
    for name, x in (("a", a), ("b", b), ("norm", norm)):
        tensors[name + "_quant"], tensors[name + "_scale"] = fp8_inc(x)
    tensors["version"] = np.ones((), np.int64)
    fp8 = dflash.DrafterTensors(write_safetensors(tmp_path / "fp8", tensors))
    plain = dflash.DrafterTensors(
        write_safetensors(tmp_path / "bf16", {"a": bf16_rne(a), "b": bf16_rne(b)})
    )
    assert fp8.fp8 and not plain.fp8 and "a" in fp8 and "version" in fp8
    assert dflash.drafter_is_fp8(tmp_path / "fp8")

    def deq(name):
        return E4M3[tensors[name + "_quant"]] * tensors[name + "_scale"][0]

    np.testing.assert_array_equal(fp8.f32("norm"), deq("norm"))
    np.testing.assert_array_equal(fp8.bf16("norm"), bf16_rne(deq("norm")))

    def unpack(w):  # Fp8Weight -> f32 (M, K) natural
        q = np.empty_like(w.bytes)
        q[:, tau_perm(w.bytes.shape[1])] = w.bytes
        scales = np.repeat(np.repeat(w.block_scales, 128, 0), 128, 1)
        return E4M3[q] * scales

    want = np.concatenate([deq("a"), deq("b")])
    w = fp8.fp8_gemm(["a", "b"])
    assert w.block_scales.shape == (3, 4)
    np.testing.assert_array_equal(unpack(w), want)
    np.testing.assert_array_equal(
        unpack(fp8.fp8_gemm(["a"], permute_k=True)), kperm(deq("a"))
    )
    # a bf16 checkpoint goes through block_fp8 as before
    got = plain.fp8_gemm(["a", "b"])
    ref = dflash.block_fp8(
        np.concatenate([bf16_to_f32(bf16_rne(a)), bf16_to_f32(bf16_rne(b))])
    )
    np.testing.assert_array_equal(got.bytes, ref.bytes)
    np.testing.assert_array_equal(got.block_scales, ref.block_scales)
