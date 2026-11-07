"""The rope rows a media plan uploads are the rows its graph reads.

`vision_rope_binding` was a harness no test ran. These are its four assertions.

The finding they exist to overturn: feeding HF's rope rows into the engine instead of ours
gave a bit-identical result, so the rows were provably in a buffer and provably not read. A
media plan bound `cs_tab` to the text table's buffer and uploaded the mrope rows over it
*behind* the launches, then restored it afterwards - so the launches read whatever the text
table was. The fix gives media plans a `cs_tab_media` buffer of their own and an extra
binding of the pre graph's `cs_tab` slot.

The first test is the one that must have failed before the fix: three materially different
row sets, everything else identical, must give materially different outputs. Holding the
rows was never in doubt - that is why magnitude probes over a buffer passed - so the second
test asserts what the plan *binds*, not what some buffer happens to contain.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

CHECKPOINT = os.environ.get("RESIN_GAUDI_QWEN35")

pytestmark = pytest.mark.skipif(
    CHECKPOINT is None
    or not Path("/usr/include/habanalabs/synapse_api.h").exists()
    or shutil.which("tpc-clang") is None,
    reason="requires RESIN_GAUDI_QWEN35, the Synapse SDK and tpc-clang",
)

GRID, PAD_ID, NEW, MERGE = 16, 248056, 10, 2

# the recorded max |logit difference| between row sets is 1.65 at the smallest, so a floor
# of 1.0 leaves margin while still failing if the rows stop reaching the graph
ROWS_MATTER = 1.0


def payload(tok, colour, grid: int = GRID):
    """One solid-colour image behind a fixed question, built here rather than loaded."""
    from resin_models_gaudi.qwen3_5.vision import merge_order

    patches = grid * grid
    merged = (grid // MERGE) ** 2
    text_ids = tok.encode("What colour is this image? Answer with one word:").ids
    base = np.tile(np.array(colour, np.float32), (patches, 1))
    patch = np.repeat(base, 16 * 16, axis=1)
    pixels = np.concatenate([patch, patch], axis=1).astype(np.float32)
    ids = text_ids + [PAD_ID] * merged + [NEW]
    pos = np.zeros((3, len(ids)), np.int64)
    pos[:, : len(text_ids)] = np.arange(len(text_ids))
    order = merge_order(grid, grid, MERGE)
    rows, cols = np.divmod(order, grid)
    for k in range(merged):
        pos[1, len(text_ids) + k] = rows[k]
        pos[2, len(text_ids) + k] = cols[k]
    pos[:, len(text_ids) + merged :] = int(max(rows.max(), cols.max())) + 1
    return SimpleNamespace(
        pixel_values=pixels,
        image_grid_thw=np.array([[1, grid, grid]], np.int64),
        position_ids=pos,
        token_ids=ids,
        image_positions=list(range(len(text_ids), len(text_ids) + merged)),
    )


@pytest.fixture(scope="module")
def engine():
    from resin_models_gaudi.qwen3_5.model import Qwen35Gaudi
    from resin_models_gaudi.synapse.runtime import SynapseError, open_device

    card = os.environ.get("RESIN_GAUDI_CARD")
    try:
        device = open_device(None if card is None else int(card))
    except SynapseError as error:
        pytest.skip(f"no Gaudi card available: {error}")
    built = Qwen35Gaudi(device, CHECKPOINT, parent_capacity=12, media=True)
    yield built
    built.close()
    device.close()


@pytest.fixture(scope="module")
def prompt(engine):
    """The image prompt as serving prepares it: features, span, mrope rows and delta."""
    from tokenizers import Tokenizer

    from resin_models_gaudi.qwen3_5.media import prepare_image_prompt
    from resin_models_gaudi.qwen3_5.vision_device import DeviceTower

    raw = json.loads((Path(CHECKPOINT) / "config.json").read_text())
    tok = Tokenizer.from_file(str(Path(CHECKPOINT) / "tokenizer.json"))
    tower = DeviceTower.from_checkpoint(CHECKPOINT, raw, device=engine.device)
    media = payload(tok, (1.0, -1.0, -1.0))  # red
    prepared = prepare_image_prompt(media, tower, engine.config)
    return SimpleNamespace(
        ids=media.token_ids,
        prepared=prepared,
        first=prepared.first,
        tok=tok,
    )


@pytest.fixture(scope="module")
def seq(engine):
    """One sequence over the first six slabs, reset before each use."""
    from resin_models_gaudi.qwen3_5.model import SingleSequence

    return SingleSequence(engine, range(1, 7))


def prefill_media(engine, prompt, rope, seq):
    """Prefill the image prompt with ``rope`` as the per-token table; returns the sequence."""
    prepared = prompt.prepared
    seq.reset()
    seq.prefill(
        prompt.ids,
        media=(
            prepared.features,
            prompt.first,
            prepared.image_tokens,
            np.asarray(rope, np.float32),
            prepared.delta,
        ),
    )
    return seq


def text_logits(ids, seq) -> np.ndarray:
    seq.reset()
    seq.prefill(ids)
    return seq.logits().copy()


def the_media_plan(engine):
    """The cached target plan a media prefill built: its key and its launches."""
    keys = [key for key in engine._plans if key[0] == "target" and key[-1] is True]
    assert keys, "a media prefill cached no media plan, so there is nothing to inspect"
    return keys[0], engine._plans[keys[0]]


def pre_step_binding(engine, plan, mode: str, name: str = "cs_tab") -> tuple[int, int]:
    """The address the plan's pre launch binds ``name`` to, and how many bindings it makes.

    The last binding of a slot wins (`LaunchPlan`), so the media plan's extra entry is the
    one that counts. The slot index comes from the *same mode's* recipe: slots are per
    compiled graph, so asking another bucket's recipe for it finds a different tensor's
    slot. Step indices are per compiled graph too, so the launch is found by recipe name
    first - hence `_steps`, which stores that name for exactly this kind of diagnosis.
    """
    recipe = engine.recipe[("pre", mode)]
    slot = recipe.slot(name)
    steps = [i for i, step in enumerate(plan._steps) if step[4] == recipe.name]
    assert steps, f"the plan holds no launch of {recipe.name}"
    start, stop = int(plan.starts[steps[0]]), int(plan.starts[steps[0] + 1])
    hits = [i for i in range(start, stop) if plan.slots[i] == slot]
    assert hits, f"the pre graph's {name} slot is bound by no entry"
    return int(plan.addresses[hits[-1]]), len(hits)


def test_different_rope_rows_change_the_logits(engine, prompt, seq):
    """The check that must have failed before the fix, and must pass after it.

    Two materially different row sets, everything else identical, must give different
    outputs. A permuted or ignored table gives the same answer for all of them.
    """
    from resin_models_gaudi.qwen3_5.checkpoint import rope_table

    T = len(prompt.ids)
    rows = {
        "mrope": prompt.prepared.rope,
        "zeros": np.zeros_like(prompt.prepared.rope),
        "text table": rope_table(T, engine.config),
    }
    logits = {
        label: prefill_media(engine, prompt, rope, seq).logits().copy()
        for label, rope in rows.items()
    }
    labels = list(rows)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            difference = float(np.abs(logits[labels[i]] - logits[labels[j]]).max())
            assert difference > ROWS_MATTER, (
                f"{labels[i]} vs {labels[j]}: max |logit difference| {difference:.6f} - "
                "the rope rows are not reaching the graph"
            )


def test_the_media_plan_binds_the_media_rope_buffer(engine, prompt, seq):
    """What the plan binds, not what a buffer holds: holding the rows was never in doubt."""
    prefill_media(engine, prompt, prompt.prepared.rope, seq)
    key, plan = the_media_plan(engine)
    bound, hits = pre_step_binding(engine, plan, key[1])
    assert bound == int(engine.buf["cs_tab_media"]), (
        "the media plan's pre graph does not read the media buffer "
        f"(it binds {bound:#x}, the text table is {int(engine.buf['cs_tab']):#x})"
    )
    assert hits >= 2, (
        f"the pre graph's cs_tab slot is bound {hits} time(s), so no override was appended"
    )


def test_the_two_rope_buffers_hold_what_they_should(engine, prompt, seq):
    """The media buffer holds this prompt's rows; the text table is left as it was."""
    from resin_models_gaudi.qwen3_5.checkpoint import rope_table

    prefill_media(engine, prompt, prompt.prepared.rope, seq)
    T = len(prompt.ids)
    shape = (engine.max_positions, 128)
    media_rows = engine.device.d2h(engine.buf["cs_tab_media"], shape, np.float32)
    text_rows = engine.device.d2h(engine.buf["cs_tab"], shape, np.float32)
    want = np.ascontiguousarray(np.asarray(prompt.prepared.rope, np.float32))
    assert np.array_equal(media_rows[:T], want), (
        "the media buffer lost this prompt's rows"
    )
    assert np.array_equal(text_rows, rope_table(engine.max_positions, engine.config)), (
        "a media prefill wrote over the shared text table"
    )


def test_a_media_prefill_leaves_a_text_prefill_unchanged(engine, prompt, seq):
    """The original bug's other half: the restore happened behind the launches."""
    text_ids = prompt.tok.encode("The capital of France is").ids
    before = text_logits(text_ids, seq)
    prefill_media(engine, prompt, prompt.prepared.rope, seq)
    after = text_logits(text_ids, seq)
    difference = float(np.abs(before - after).max())
    assert difference == 0.0, (
        f"a text prefill differs by {difference:.6f} after a media prefill ran"
    )
