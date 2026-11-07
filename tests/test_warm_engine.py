"""`warm_engine` has to fail loudly for text and quietly for images.

The image warmup exists so the media tower's recipes are compiled at startup instead of on a
user's first image. But it runs before the server accepts traffic, so anything that raises
there decides whether the server comes up at all - and a bad checkpoint, a missing
preprocessor config or a grid that will not compile should cost the image feature, not a
server that could still serve text.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from deploy.models import ModelSpec, get_model, warm_engine


class Engine:
    """Just enough of the engine: submit returns a lease, and nothing is ever pending."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.submitted: list[list[str]] = []

    def submit(self, **kwargs):
        self.submitted.append(sorted(kwargs))
        if self.fail_on in kwargs:
            raise RuntimeError(f"{self.fail_on} is not supported here")
        return SimpleNamespace(take_result=lambda: None)

    def has_work(self) -> bool:
        return False

    def step(self) -> None:  # pragma: no cover - has_work is always False
        raise AssertionError("nothing should be pending")


class Recipe:
    """Renders a prompt that carries both a text side and a media side."""

    stop_token_ids: tuple[int, ...] = ()

    def render(self, messages):
        return SimpleNamespace(media=object(), token_ids=[1, 2, 3])


def spec(image_warmup: bool) -> ModelSpec:
    """A real preset with the warmup switch flipped, which is all `warm_engine` reads."""
    return replace(get_model("qwen3_5-gaudi2"), image_warmup=image_warmup)


def test_image_warmup_failure_warns_and_text_still_warms() -> None:
    engine = Engine(fail_on="media")
    with pytest.warns(RuntimeWarning, match="serve text only"):
        warm_engine(engine, Recipe(), spec(image_warmup=True))
    # the text warmup ran, and the image warmup was attempted rather than skipped
    assert any("token_ids" in keys for keys in engine.submitted)
    assert any("media" in keys for keys in engine.submitted)


def test_image_warmup_is_not_attempted_when_the_preset_does_not_warm_it() -> None:
    engine = Engine(fail_on="media")
    warm_engine(engine, Recipe(), spec(image_warmup=False))
    assert engine.submitted and not any("media" in keys for keys in engine.submitted)


def test_text_warmup_failure_is_fatal() -> None:
    """A broken text path must not come up as a server that pretends to work."""
    engine = Engine(fail_on="token_ids")
    with pytest.raises(RuntimeError, match="not supported here"):
        warm_engine(engine, Recipe(), spec(image_warmup=True))
