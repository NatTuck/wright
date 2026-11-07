from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from resin_models_gaudi.qwen3_5.vision import VisionConfig, VisionTower, vision_rope

torch = pytest.importorskip("torch")

CHECKPOINT = os.environ.get("RESIN_GAUDI_QWEN35")

pytestmark = pytest.mark.skipif(
    CHECKPOINT is None or not (Path(CHECKPOINT) / "config.json").exists(),
    reason="requires RESIN_GAUDI_QWEN35, a Qwen3.8-27B FP8 checkpoint",
)


def tiny_config() -> VisionConfig:
    return VisionConfig(
        depth=2,
        hidden=8,
        heads=2,
        intermediate=16,
        patch=2,
        temporal=2,
        merge=2,
        positions=4,  # 2x2 grid
        out_hidden=6,
        in_channels=3,
    )


def tiny_tensors(config: VisionConfig, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    def rand(*shape: int) -> np.ndarray:
        return rng.normal(0.0, 0.1, size=shape).astype(np.float32)

    tensors = {
        "patch_embed.proj.weight": rand(
            config.hidden, config.in_channels, config.temporal, config.patch, config.patch
        ),
        "patch_embed.proj.bias": rand(config.hidden),
        "pos_embed.weight": rand(config.positions, config.hidden),
        "merger.norm.weight": np.ones(config.hidden, np.float32),
        "merger.norm.bias": np.zeros(config.hidden, np.float32),
        "merger.linear_fc1.weight": rand(config.merge**2 * config.hidden, config.merge**2 * config.hidden),
        "merger.linear_fc1.bias": rand(config.merge**2 * config.hidden),
        "merger.linear_fc2.weight": rand(config.out_hidden, config.merge**2 * config.hidden),
        "merger.linear_fc2.bias": rand(config.out_hidden),
    }
    for i in range(config.depth):
        tensors.update(
            {
                f"blocks.{i}.norm1.weight": np.ones(config.hidden, np.float32),
                f"blocks.{i}.norm1.bias": np.zeros(config.hidden, np.float32),
                f"blocks.{i}.norm2.weight": np.ones(config.hidden, np.float32),
                f"blocks.{i}.norm2.bias": np.zeros(config.hidden, np.float32),
                f"blocks.{i}.attn.qkv.weight": rand(3 * config.hidden, config.hidden),
                f"blocks.{i}.attn.qkv.bias": rand(3 * config.hidden),
                f"blocks.{i}.attn.proj.weight": rand(config.hidden, config.hidden),
                f"blocks.{i}.attn.proj.bias": rand(config.hidden),
                f"blocks.{i}.mlp.linear_fc1.weight": rand(config.intermediate, config.hidden),
                f"blocks.{i}.mlp.linear_fc1.bias": rand(config.intermediate),
                f"blocks.{i}.mlp.linear_fc2.weight": rand(config.hidden, config.intermediate),
                f"blocks.{i}.mlp.linear_fc2.bias": rand(config.hidden),
            }
        )
    return tensors


def test_patch_features_match_the_conv_kernel() -> None:
    config = tiny_config()
    assert config.patch_features == 3 * 2 * 2 * 2
    assert config.grid == 2


def test_forward_merges_the_patch_grid() -> None:
    config = tiny_config()
    tower = VisionTower(config, tiny_tensors(config))
    # One frame, a 4x4 patch grid: 16 patches, merged 2x2 into 4 tokens.
    patches = np.random.default_rng(1).normal(size=(16, config.patch_features)).astype(np.float32)
    out = tower.forward(patches, [[1, 4, 4]])
    assert out.shape == (4, config.out_hidden)
    assert np.isfinite(out).all()


def test_forward_rejects_a_grid_that_does_not_match_the_patches() -> None:
    config = tiny_config()
    tower = VisionTower(config, tiny_tensors(config))
    patches = np.zeros((16, config.patch_features), np.float32)
    with pytest.raises(ValueError, match="does not match"):
        tower.forward(patches, [[1, 3, 3]])
    with pytest.raises(ValueError, match="does not divide"):
        tower.forward(patches, [[1, 1, 16]])


def test_missing_tensors_are_reported() -> None:
    config = tiny_config()
    tensors = tiny_tensors(config)
    del tensors["blocks.1.mlp.linear_fc2.weight"]
    with pytest.raises(ValueError, match="missing tensors"):
        VisionTower(config, tensors)


def test_position_embedding_interpolates_to_the_patch_grid() -> None:
    config = tiny_config()
    tower = VisionTower(config, tiny_tensors(config))
    table = tower._pos_embed(4, 4)
    assert table.shape == (16, config.hidden)
    assert np.isfinite(table.numpy()).all()


def test_real_checkpoint_tower_runs() -> None:
    """Smoke test against the pinned checkpoint: 27 blocks, 460.7M parameters."""
    config = json.loads((Path(CHECKPOINT) / "config.json").read_text())
    tower = VisionTower.from_checkpoint(CHECKPOINT, config)
    vision = config["vision_config"]
    patch = int(vision["patch_size"])
    temporal = int(vision["temporal_patch_size"])
    # A 32x32 image: 2x2 patches, merged to a single token.
    features = 3 * temporal * patch * patch
    patches = np.random.default_rng(0).normal(size=(4, features)).astype(np.float32)
    out = tower.forward(patches, [[1, 2, 2]])
    assert out.shape == (1, int(vision["out_hidden_size"]))
    assert np.isfinite(out).all()
    assert out.std() > 0


def test_attention_runs_over_tokens_and_concatenates_heads() -> None:
    """The arithmetic the oracle itself has to get right.

    `_attention` used to permute the token and head axes, which made
    `scaled_dot_product_attention` attend across *heads* instead of across patches and
    then scrambled the head concatenation. Nothing about the shape gives that away, so
    the semantics are pinned here against an explicit per-head reference.
    """
    config = tiny_config()
    tower = VisionTower(config, tiny_tensors(config))
    x = np.random.default_rng(2).normal(size=(16, config.hidden)).astype(np.float32)
    params = {name: value.numpy() for name, value in tower.params.items()}
    qkv = x @ params["blocks.0.attn.qkv.weight"].T + params["blocks.0.attn.qkv.bias"]
    qkv = qkv.reshape(x.shape[0], 3, config.heads, config.head_dim)
    # 16 patches as a 4x4 grid, in the merge-block order the model consumes them
    rope = vision_rope(torch, config.head_dim, 4, 4, config.merge)
    cos, sin = (part.numpy() for part in rope)

    def rotate(t):
        half = t.shape[-1] // 2
        return np.concatenate([-t[..., half:], t[..., :half]], axis=-1)

    want = np.zeros((x.shape[0], config.hidden), np.float64)
    for h in range(config.heads):
        q, k, v = (qkv[:, i, h, :].astype(np.float64) for i in range(3))
        q = q * cos + rotate(q) * sin
        k = k * cos + rotate(k) * sin
        s = (q @ k.T) / np.sqrt(config.head_dim)  # (queries, keys)
        e = np.exp(s - s.max(1, keepdims=True))
        o = (e / e.sum(1, keepdims=True)) @ v  # (queries, head_dim)
        want[:, h * config.head_dim : (h + 1) * config.head_dim] = o
    want = want @ params["blocks.0.attn.proj.weight"].T + params["blocks.0.attn.proj.bias"]
    got = tower._attention(torch.from_numpy(x), 0, rope).numpy()
    assert np.abs(got - want).max() < 1e-5 * max(np.abs(want).max(), 1.0)


def test_recipe_takes_its_media_config_from_the_checkpoint() -> None:
    """The preprocessor config has to actually reach the Rust media config.

    A silent fall back to `Config::default()` looks exactly like success, so check a number
    that distinguishes them: the checkpoint sets min_pixels = 65536, so a 96x96 image is
    upscaled to 256x256 and becomes a 16x16 patch grid. With the defaults (min_pixels unset)
    it would stay 96x96, i.e. a 6x6 grid of 36 patches.
    """
    from dataclasses import replace

    from deploy.models import create_recipe, get_model
    from resin.recipe import Image, Message, Role, Text

    # the preset itself is still media=False; this exercises the branch without flipping it
    spec = replace(get_model("qwen3_5-gaudi2"), media=True)
    recipe = create_recipe(spec, CHECKPOINT)
    axis = np.linspace(0, 1, 96, dtype=np.float32)
    horizontal = np.broadcast_to(axis[None], (96, 96))
    pixels = np.stack((horizontal, horizontal.T, 1 - horizontal), axis=-1).copy()
    prompt = recipe.render([Message(Role.USER, [Image(source=pixels), Text("hi")])])

    media = prompt.media
    assert media is not None
    grid = np.asarray(media.image_grid_thw)
    assert grid.shape == (1, 3)
    assert tuple(int(v) for v in grid[0]) == (1, 16, 16), (
        f"grid {grid[0].tolist()} - the checkpoint's media config did not apply"
    )
    # temporal 2, patch 16, three channels
    assert np.asarray(media.pixel_values).shape == (256, 3 * 2 * 16 * 16)


def test_vision_switch_rejects_an_unknown_value(monkeypatch) -> None:
    from resin_models_gaudi.qwen3_5.media import VISION_SWITCH, tower_from_checkpoint

    monkeypatch.setenv(VISION_SWITCH, "gpu")
    with pytest.raises(ValueError, match="expected 'device' or 'cpu'"):
        tower_from_checkpoint("/nonexistent", {})


def test_vision_switch_defaults_to_the_card(monkeypatch) -> None:
    """Unset means `device`: the card path is the one the accuracy set was measured with.

    This test asserted the opposite while image support did not work - "nothing may start using
    the card by accident" was the right policy then. It is now a deliberate default, so the test
    states it, and `cpu` is checked separately as still available.
    """
    from resin_models_gaudi.qwen3_5.media import VISION_SWITCH, tower_from_checkpoint
    from resin_models_gaudi.qwen3_5.vision_device import DeviceTower

    monkeypatch.delenv(VISION_SWITCH, raising=False)
    tower = tower_from_checkpoint(CHECKPOINT, json.loads((Path(CHECKPOINT) / "config.json").read_text()))
    assert isinstance(tower, DeviceTower)


def test_vision_switch_still_offers_the_cpu_oracle(monkeypatch) -> None:
    """`cpu` has to keep working: it needs no card, and it is the comparison the record uses."""
    from resin_models_gaudi.qwen3_5.media import VISION_SWITCH, tower_from_checkpoint

    monkeypatch.setenv(VISION_SWITCH, "cpu")
    tower = tower_from_checkpoint(CHECKPOINT, json.loads((Path(CHECKPOINT) / "config.json").read_text()))
    assert isinstance(tower, VisionTower)


def test_prepare_image_prompt_joins_recipe_and_tower() -> None:
    from types import SimpleNamespace

    from resin_models_gaudi.qwen3_5.media import prepare_image_prompt

    config = tiny_config()
    tower = VisionTower(config, tiny_tensors(config))
    patches = np.zeros((16, config.patch_features), np.float32)  # 4x4 patch grid
    # 5 tokens: text, 4 image tokens (merged 2x2 from a 4x4 grid), text.
    position_ids = np.array(
        [
            [0, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 1, 0, 1, 1],
        ]
    )[:, :6]
    media = SimpleNamespace(
        pixel_values=patches,
        image_grid_thw=np.array([[1, 4, 4]]),
        position_ids=position_ids,
        token_ids=list(range(6)),
        image_positions=[1, 2, 3, 4],
    )
    rope_config = SimpleNamespace(rotary_dim=64, rope_theta=1e7, mrope_section=(11, 11, 10))
    prompt = prepare_image_prompt(media, tower, rope_config)
    assert prompt.tokens == 6
    assert prompt.image_tokens == 4
    assert prompt.first == 1  # the payload's positions, not a scan of the token ids
    assert prompt.features.shape == (4, config.out_hidden)
    assert prompt.rope.shape == (6, 128)


def test_prepare_image_prompt_rejects_a_mismatched_prompt() -> None:
    from types import SimpleNamespace

    from resin_models_gaudi.qwen3_5.media import prepare_image_prompt

    config = tiny_config()
    tower = VisionTower(config, tiny_tensors(config))
    media = SimpleNamespace(
        pixel_values=np.zeros((16, config.patch_features), np.float32),
        image_grid_thw=np.array([[1, 4, 4]]),
        position_ids=np.zeros((3, 6), np.int64),
        token_ids=list(range(5)),  # one short of the six position rows
        image_positions=[1, 2, 3, 4],
    )
    rope_config = SimpleNamespace(rotary_dim=64, rope_theta=1e7, mrope_section=(11, 11, 10))
    with pytest.raises(ValueError, match="rope rows for 5 prompt tokens"):
        prepare_image_prompt(media, tower, rope_config)


def test_image_block_is_token_major() -> None:
    """The host array `splice` reads must be (tokens, hidden), features in the first rows.

    Device tensors are token-major, so the (H, T) tensor the embed graph declares takes a
    (T, H) host array. The upload once sent the compact (H, count) features instead, which
    put feature k at flat h * count where the kernel read some other (h, t): the image
    columns held the right values in the wrong places. That kept the model able to tell an
    image was present while destroying what it was - it scored 2/19 on the accuracy set and
    scores 19/19 since. A layout convention cannot be caught by any end-to-end check that
    only looks at magnitudes, so it is pinned here.
    """
    from resin_models_gaudi.qwen3_5.model import image_block

    hidden, count, tokens = 6, 3, 8
    features = np.arange(count * hidden, dtype=np.float32).reshape(count, hidden)
    block = image_block(features, count, tokens, hidden)

    assert block.shape == (tokens, hidden)
    assert block.dtype == np.float32
    # the features, row for row, and nothing else anywhere
    assert np.array_equal(block[:count], features)
    assert not block[count:].any()
    # the failure this exists for: the compact (hidden, count) array the upload used to
    # send holds every one of these values, in a layout the kernel cannot read
    transposed = features.T
    assert not np.array_equal(block[:count], transposed)
    assert sorted(block[:count].ravel()) == sorted(transposed.ravel())


def test_image_block_rejects_the_wrong_feature_count() -> None:
    from resin_models_gaudi.qwen3_5.model import image_block

    with pytest.raises(ValueError, match="features must be"):
        image_block(np.zeros((3, 6), np.float32), 4, 8, 6)


def test_the_image_span_comes_from_the_payload_not_a_token_scan() -> None:
    """`prepare_image_prompt` must use the payload's `image_positions`.

    The runner only ever sees one prefill chunk's tokens, and a chunk boundary (or a
    prefix-cache restore) can split the placeholder run - scanning chunk-local ids for a
    maximal run of identical tokens is exactly the crash this replaces.
    """
    from types import SimpleNamespace

    from resin_models_gaudi.qwen3_5.media import prepare_image_prompt
    from resin_models_gaudi.qwen3_5.mrope import text_position_ids

    rope_config = SimpleNamespace(rotary_dim=8, rope_theta=1e6, mrope_section=(2, 1, 1))
    tower = SimpleNamespace(
        config=SimpleNamespace(merge=2),
        forward=lambda pixels, grids: np.zeros(
            (int(np.prod(np.asarray(grids)[:, 1:]).sum()) // 4, 6), np.float32
        ),
    )
    media = SimpleNamespace(
        pixel_values=np.zeros((4, 24), np.float32),
        image_grid_thw=np.array([[1, 2, 2]]),
        position_ids=text_position_ids(np.arange(5)),
        token_ids=list(range(5)),
        image_positions=[3],
    )
    prompt = prepare_image_prompt(media, tower, rope_config)
    assert prompt.image_tokens == 1
    assert prompt.first == 3
    assert prompt.delta == 4 + 1 - 5  # max(position_ids) + 1 - tokens


def test_noncontiguous_image_positions_are_refused() -> None:
    """The contiguity the Rust side validated is re-checked, cheaply, on the Python side."""
    from types import SimpleNamespace

    from resin_models_gaudi.qwen3_5.media import prepare_image_prompt
    from resin_models_gaudi.qwen3_5.mrope import text_position_ids

    rope_config = SimpleNamespace(rotary_dim=8, rope_theta=1e6, mrope_section=(2, 1, 1))
    tower = SimpleNamespace(
        config=SimpleNamespace(merge=2),
        forward=lambda pixels, grids: np.zeros(
            (int(np.prod(np.asarray(grids)[:, 1:]).sum()) // 4, 6), np.float32
        ),
    )
    media = SimpleNamespace(
        pixel_values=np.zeros((16, 24), np.float32),
        image_grid_thw=np.array([[1, 4, 4]]),
        position_ids=text_position_ids(np.arange(8)),
        token_ids=list(range(8)),
        image_positions=[1, 2, 4, 5],
    )
    with pytest.raises(ValueError, match="contiguous run"):
        prepare_image_prompt(media, tower, rope_config)


def test_a_chunked_prefill_and_a_prefix_restore_still_place_the_image() -> None:
    """The served crash, end to end: a 729-token placeholder run split across prefill
    chunks has no maximal run in any single chunk, and a prefix-cache restore can start
    a chunk inside the run. Both must complete, agree, and agree with a fresh prefill.
    """
    import shutil
    from dataclasses import replace

    from deploy.models import create_recipe, get_model
    from resin import Engine, SamplingParams, SchedulerConfig
    from resin.recipe import Image as RImage
    from resin.recipe import Message, RenderOptions, Role, Text

    if not Path("/usr/include/habanalabs/synapse_api.h").exists() or shutil.which("tpc-clang") is None:
        pytest.skip("requires the Synapse SDK and tpc-clang")
    from resin_models_gaudi.qwen3_5.runner import Qwen3_5GaudiRunner
    from resin_models_gaudi.synapse.runtime import SynapseError

    try:
        runner = Qwen3_5GaudiRunner(
            CHECKPOINT,
            max_batch_size=4,
            max_seq_len=8192,
            parent_capacity=24,
            media=True,
        )
    except SynapseError as error:
        pytest.skip(f"no Gaudi card available: {error}")

    recipe = create_recipe(replace(get_model("qwen3_5-gaudi2"), media=True), CHECKPOINT)
    # ~300 tokens of text pushes the image past the 256-token chunk size.
    text = "The quick brown fox jumps over the lazy dog. " * 45
    axis = np.linspace(0, 1, 256, dtype=np.float32)
    square = np.broadcast_to(axis[None], (256, 256))
    pixels = np.stack((square, square.T, 1 - square), axis=-1).copy()

    def run_once(engine, prompt):
        lease = engine.submit(
            media=prompt.media,
            sampling_params=SamplingParams(
                max_output_tokens=16,
                stop_token_ids=recipe.stop_token_ids,
                temperature=0.0,
            ),
        )
        while engine.has_work():
            engine.step()
        return list(lease.take_result())

    try:
        engine = Engine(
            runner,
            max_running_requests=4,
            context_length=8192,
            prefix_cache=True,
            scheduler_config=SchedulerConfig(prefill_chunk_size=256),
        )
        try:
            prompt = recipe.render(
                [Message(Role.USER, [Text(text), RImage(source=pixels), Text("What color?")])],
                options=RenderOptions(enable_thinking=False),
            )
            assert len(prompt.media.token_ids) > 256  # the chunk size, so this splits
            chunked = run_once(engine, prompt)
            restored = run_once(engine, prompt)  # the prefix cache restores mid-run
        finally:
            engine.close()
        assert chunked == restored  # a prefix-cache hit replays the same tokens
        assert len(chunked) > 0
    finally:
        runner.close()
