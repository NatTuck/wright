"""HTTP serving for a token-native Resin engine.

The networking, scheduling bridge, wire protocols, and streaming adapters run
in Rust. Stock model formats run in resin-recipe; custom recipes implement the same
structural Python protocol.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from resin._resin import Engine, _HttpServer
from resin.protocols import ResponseParser
from resin.recipe import ParserEvent, Recipe, RenderedPrompt

if TYPE_CHECKING:
    from resin.backend.distributed import DistributedEngine


@dataclass(frozen=True, kw_only=True)
class ModelCapabilities:
    """What the served model accepts and makes, for ``GET /v1/models``.

    The response does not include a field that is None. False tells that the model does
    not support the capability. The response tells that the model supports vision when
    input_modalities includes "image". The engine gives the token limits and prompt
    caching.

    thinking_levels are the reasoning_effort values that the model takes, from the least
    to the most effort, for example ``NativeRecipe.reasoning_efforts``. They require
    reasoning=True.

    default_temperature, default_top_p and default_top_k are the sampling settings of a
    request that gives none: the model's recommended values, as in its
    ``generation_config.json``. Without them the server uses temperature 1.0, top_p 1.0 and
    no top_k.
    """

    function_calling: bool | None = None
    reasoning: bool | None = None
    thinking_levels: tuple[str, ...] | None = None
    input_modalities: tuple[str, ...] | None = None
    output_modalities: tuple[str, ...] | None = None
    default_temperature: float | None = None
    default_top_p: float | None = None
    default_top_k: int | None = None

    def __post_init__(self) -> None:
        for name in ("function_calling", "reasoning"):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise ValueError(f"{name} must be a bool or None")
        for name in ("thinking_levels", "input_modalities", "output_modalities"):
            value = getattr(self, name)
            if value is None:
                continue
            if (
                type(value) is not tuple
                or not value
                or any(type(item) is not str or not item.strip() for item in value)
            ):
                raise ValueError(
                    f"{name} must be a nonempty tuple of nonempty strings or None"
                )
        if self.thinking_levels is not None and self.reasoning is not True:
            raise ValueError("thinking_levels must be None unless reasoning=True")
        for name, low, high in (
            ("default_temperature", 0.0, float("inf")),
            ("default_top_p", 0.0, 1.0),
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if (
                type(value) not in (float, int)
                or type(value) is bool
                or value != value
                or not (low <= value <= high)
                or (name == "default_top_p" and value == 0.0)
                or value == float("inf")
            ):
                raise ValueError(f"{name} must be a finite number in its range or None")
        if self.default_top_k is not None and (
            type(self.default_top_k) is not int or self.default_top_k < 1
        ):
            raise ValueError("default_top_k must be a positive int or None")

    def to_wire(self) -> dict[str, object]:
        """The fields that are not None, as the native server takes them."""
        return {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in (
                ("function_calling", self.function_calling),
                ("reasoning", self.reasoning),
                ("thinking_levels", self.thinking_levels),
                ("input_modalities", self.input_modalities),
                ("output_modalities", self.output_modalities),
                ("default_temperature", self.default_temperature),
                ("default_top_p", self.default_top_p),
                ("default_top_k", self.default_top_k),
            )
            if value is not None
        }


class HttpServer:
    """Serve a local or distributed engine through the shared HTTP frontend.

    With DistributedEngine, every attention-DP driver constructs a server and
    calls serve. The first driver listens; the others participate in coordinated
    inference. Requests select a replica with routed_dp_rank or the
    resin-routed-dp-rank header; unassigned requests use round robin.

    stream_interval batches later output until N generated model tokens have
    accumulated, checked at step boundaries. The first semantic output is sent
    immediately, and completion flushes the remainder. Stop matching and
    cancellation still run on every step.

    request_queue_capacity bounds requests waiting for engine admission across
    all replicas. It does not change the engine's active batch size. A full
    queue returns HTTP 429 so clients can retry with backoff.

    On SIGTERM the server stops new connections, and requests in flight run for up to
    drain_seconds before they are cancelled. Ctrl-C cancels them at once.

    OpenAI chat and Responses requests without an output-token limit use the
    engine's remaining context budget after rendering and tokenizing the prompt.

    ``GET /v1/models`` gives the engine's context length as the context and output
    limits, and tells whether the engine has a prefix cache. capabilities adds what
    the engine cannot tell.
    """

    def __init__(
        self,
        engine: Engine | DistributedEngine,
        *,
        recipe: Recipe,
        model: str | None = None,
        stream_interval: int = 1,
        request_queue_capacity: int = 1024,
        drain_seconds: float = 30.0,
        enable_openai: bool = True,
        enable_anthropic: bool = True,
        enable_demo: bool = True,
        capabilities: ModelCapabilities | None = None,
    ) -> None:
        from resin.backend.distributed import (
            DistributedEngine,
            DistributedExecutionError,
        )

        error = None
        try:
            if (
                type(stream_interval) is not int
                or not 1 <= stream_interval <= 2**32 - 1
            ):
                raise ValueError(
                    "stream_interval must be a positive integer fitting u32"
                )
            if (
                type(request_queue_capacity) is not int
                or not 1 <= request_queue_capacity <= 2**32 - 1
            ):
                raise ValueError(
                    "request_queue_capacity must be a positive integer fitting u32"
                )
            if model is None:
                tokenizer = recipe.tokenizer
                model = None if tokenizer is None else tokenizer.name_or_path
            if not isinstance(model, str) or not model.strip():
                raise ValueError(
                    "model is required when it cannot be inferred from the recipe"
                )
            if any(
                type(flag) is not bool
                for flag in (enable_openai, enable_anthropic, enable_demo)
            ):
                raise ValueError("HTTP route flags must be booleans")
            if enable_demo and not enable_openai:
                raise ValueError("enable_demo=True requires enable_openai=True")
            if capabilities is not None and not isinstance(
                capabilities, ModelCapabilities
            ):
                raise ValueError("capabilities must be a ModelCapabilities or None")
        except BaseException as exc:
            error = exc
        if isinstance(engine, DistributedEngine):
            # No driver may return before its peers have validated startup.
            # Inference of a recipe's model name can itself run user code.
            config = (
                None
                if error is not None
                else (
                    model,
                    enable_openai,
                    enable_anthropic,
                    enable_demo,
                    stream_interval,
                    request_queue_capacity,
                    engine.local.metrics_enabled,
                    capabilities,
                )
            )
            states = engine.worker._drivers(
                "http configuration", (config, None if error is None else str(error))
            )
            group = engine.worker._driver_group
            failure = next(
                (
                    f"HTTP configuration failed on rank {rank}: {error}"
                    for rank, (_, error) in zip(group.ranks, states, strict=True)
                    if error
                ),
                None,
            )
            if failure is None and any(config != states[0][0] for config, _ in states):
                failure = "HTTP server configuration must agree across replica drivers"
            if failure:
                try:
                    engine.close()
                finally:
                    raise DistributedExecutionError(failure)
        elif error is not None:
            raise error
        assert isinstance(model, str)
        self._model = model
        self._coordinator = None
        if isinstance(engine, DistributedEngine):
            from resin.server.distributed import DistributedHttpCoordinator

            self._coordinator = DistributedHttpCoordinator(
                engine, recipe, model=model, stream_interval=stream_interval
            )
        self._native = None
        error = None
        try:
            if self._coordinator is None or self._coordinator.is_frontend:
                self._native = _HttpServer(
                    engine.local if isinstance(engine, DistributedEngine) else engine,
                    recipe,
                    model=model,
                    stream_interval=stream_interval,
                    request_queue_capacity=request_queue_capacity,
                    drain_seconds=drain_seconds,
                    enable_openai=enable_openai,
                    enable_anthropic=enable_anthropic,
                    enable_demo=enable_demo,
                    distributed=self._coordinator,
                    capabilities=None
                    if capabilities is None
                    else capabilities.to_wire(),
                )
        except BaseException as exc:
            error = exc
        if self._coordinator is not None:
            failure = self._coordinator._vote(
                "frontend initialize", None if error is None else str(error)
            )
            if failure:
                self._coordinator._close(failure)
        elif error is not None:
            raise error

    @property
    def model(self) -> str:
        return self._model

    @property
    def models(self) -> list[str]:
        """Every served model's id, in registration order."""
        if self._native is not None:
            return list(self._native.models)
        return [self._model]

    def register(
        self,
        engine: Engine,
        *,
        recipe: Recipe,
        model: str | None = None,
        stream_interval: int = 1,
        capabilities: ModelCapabilities | None = None,
        share: float = 1.0,
    ) -> None:
        """Serve one more model, on its own Engine, before ``serve``.

        A request goes to the model that its ``model`` field names, and
        ``GET /v1/models`` lists every model. The Engines step in turn on one
        thread (they share one card); ``share`` weighs a model's card time
        against the others'. Give the Engines one ``CachePool`` so that the
        models share cache memory.
        """
        from resin.backend.distributed import DistributedEngine

        if self._coordinator is not None or isinstance(engine, DistributedEngine):
            raise ValueError("a server with a DistributedEngine serves one model")
        if type(stream_interval) is not int or not 1 <= stream_interval <= 2**32 - 1:
            raise ValueError("stream_interval must be a positive integer fitting u32")
        if model is None:
            tokenizer = recipe.tokenizer
            model = None if tokenizer is None else tokenizer.name_or_path
        if not isinstance(model, str) or not model.strip():
            raise ValueError(
                "model is required when it cannot be inferred from the recipe"
            )
        if capabilities is not None and not isinstance(capabilities, ModelCapabilities):
            raise ValueError("capabilities must be a ModelCapabilities or None")
        assert self._native is not None
        self._native.register(
            engine,
            recipe,
            model=model,
            stream_interval=stream_interval,
            capabilities=None if capabilities is None else capabilities.to_wire(),
            share=float(share),
        )

    def register_replicas(self, router, *, model: str) -> None:
        """Serve one more model whose Engines run in other processes, one per card.

        ``router`` is a ``resin._resin._ReplicaRouter`` whose replicas have joined
        (``router.accept``). Each request goes to one replica: the replica of its
        conversation, unless that replica is much busier than another. It can sit
        beside a ``DistributedEngine``.
        """
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty string")
        if self._native is None:
            raise ValueError("only the HTTP frontend serves replicas")
        self._native.register_replicas(router, model=model)

    @property
    def running(self) -> bool:
        if self._native is not None:
            return self._native.running
        return self._coordinator is not None and self._coordinator.running

    def serve(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8000,
        on_listening: Callable[[], object] | None = None,
    ) -> None:
        """Block while serving, until Ctrl-C or another thread calls shutdown.

        ``on_listening`` is called once the listener is bound and SIGTERM drains the server:
        a client that connects after it is called is served. Its error stops the server and
        raises here. A non-root distributed driver has no listener and does not call it.
        """

        try:
            if self._native is not None:
                self._native.serve(host=host, port=port, on_listening=on_listening)
            elif self._coordinator is not None:
                self._coordinator.serve()
        finally:
            if self._coordinator is not None:
                self._coordinator.close()

    def shutdown(self) -> None:
        """Request shutdown after the current inference step finishes.

        This does not block the caller. Join the thread running ``serve()`` to
        wait for shutdown; active requests are cancelled and the Engine closes.
        """

        if self._native is not None:
            self._native.shutdown()
        elif self._coordinator is not None:
            self._coordinator.shutdown()


__all__ = [
    "Recipe",
    "HttpServer",
    "ModelCapabilities",
    "ParserEvent",
    "RenderedPrompt",
    "ResponseParser",
]
