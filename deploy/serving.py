"""The server: load one registry model on this host and serve it through Resin's HTTP APIs.

The `wright` command and scripts/serve_local.py both start here. Multi-card presets spawn one
process per card, and each process takes a free card from --gpu.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import secrets
import shutil
import signal
import socket
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from deploy.gaudi_cli import (
    add_gaudi_arguments,
    apply_gaudi_arguments,
    check_listen_address,
    decode_share,
    describe_settings,
    feature_enabled,
    nonnegative_int,
    positive_int,
)
from deploy.logs import setup_logging
from deploy.models import (
    MODEL_SPECS,
    NATIVE_RANK_FAMILIES,
    ModelSpec,
    check_cache_types,
    check_gaudi_drafts,
    check_gaudi_ranks,
    checkpoint_weight_bytes,
    create_recipe,
    create_runner,
    default_draft_tokens,
    gaudi_defers_cache,
    gaudi_supports_ranks,
    get_model,
    model_capabilities,
    require_checkpoint,
    validate_gaudi_options,
    warm_engine,
)

logger = logging.getLogger("wright")
QWEN_NEEDS_NATIVE = (
    "a Qwen3.5/3.8 model on more than one card runs on the native executor: "
    "do not give -f oldqwen"
)

# A rank exits with this code after it reports a startup problem that needs no traceback.
EXIT_REPORTED = 78
# A rank exits with this code when a signal stops it during the load.
EXIT_STOPPED = 79
# Recipe workspaces and slack that the KV pool leaves free on a card.
_WORKSPACE_BYTES = 3 << 30
# Rank 0 and its followers need this long to shut down after the drain.
_SHUTDOWN_SECONDS = 60.0
# How long rank 0 waits for the replicas of a model to load and join, and they for it.
_REPLICA_TIMEOUT = 3600.0
# The interpreter's switch interval in a rank that serves a replica beside its own model.
_SWITCH_INTERVAL = float(os.environ.get("RESIN_COLOCATED_SWITCH_INTERVAL", "0.0005"))
# The share of card memory that a replica on the cards of the first model of a layout may fill
# (weights, workspace and cache) when the two models do not share the cards' KV cache (a first model
# that makes its cache when it loads), unless it has --gpu-memory-utilization. The first model sizes
# its cache from what is left. A Qwen3.5-9B replica gets about 10 GiB of cache.
COLOCATED_MEMORY = 0.25
# A shared KV cache arena holds a multiple of this many slabs on each card, so that small changes of
# free memory do not change every recipe's arena shape.
_ARENA_QUANTUM = 8


def batch_tokens(runner, prefill_chunk_size: int) -> int:
    """Prefill tokens per step, for all rows together: no more than one chunk."""
    limit = getattr(runner.runner_config, "max_batch_tokens", None)
    return prefill_chunk_size if limit is None else min(limit, prefill_chunk_size)


def current_device_free() -> int:
    from resin_models_gaudi.synapse.runtime import current_device

    return current_device().memory_info()[0]


def estimate_kv_pool(free: int, weight_bytes: int, ranks: int) -> None:
    """The limit on the KV pool, before the load: what the weights leave free on one card. Graph
    buffers and the memory share of the preset take more, so the report after the load is smaller."""
    left = free - weight_bytes // ranks - _WORKSPACE_BYTES
    logger.info(
        f"KV pool: at most {left / 2**30:.0f} GiB per card, {left * ranks / 2**30:.0f} GiB in "
        f"total, from {free / 2**30:.0f} GiB free and {weight_bytes / ranks / 2**30:.0f} GiB of "
        "weights per card"
    )


def tokens_held(layout) -> int | None:
    """The longest one sequence that the pool holds, or None when no length fills it."""
    low, high = 0, 1 << 31
    if layout.estimated_parent_count([high]) <= layout.parent_capacity:
        return None
    while high - low > 1:
        middle = (low + high) // 2
        if layout.estimated_parent_count([middle]) <= layout.parent_capacity:
            low = middle
        else:
            high = middle
    return low


def report_kv_pool(
    runner, engine, ranks: int, context_length: int, max_batch_size: int
) -> None:
    """The KV pool after the load: its size, its free slabs, the tokens of one sequence, and how
    many sequences at full context it holds."""
    layout = runner.cache_layout
    nbytes = getattr(runner, "kv_pool_bytes", None)
    if nbytes is None:
        return
    free = getattr(engine, "local", engine).cache_free_parent_count
    free = "?" if free is None else f"{free:,}"
    tokens = tokens_held(layout)
    message = (
        f"KV pool: {nbytes / 2**30:.1f} GiB per card, {nbytes * ranks / 2**30:.1f} GiB in total; "
        f"{free} of {layout.parent_capacity:,} slabs free"
    )
    if tokens is not None:
        message += f"; holds {tokens:,} tokens in one sequence"
    logger.info(message)
    fits = 0
    for count in range(1, max_batch_size + 1):
        needed = layout.estimated_parent_count([context_length] * count)
        if needed > layout.parent_capacity:
            break
        fits = count
    message = f"KV pool: holds {fits} of {max_batch_size} sequences at {context_length:,} tokens"
    if fits < max_batch_size:
        logger.warning(message + ". More sequences than this wait, or give back cache")
    else:
        logger.info(message)


def _stop_while_loading(signum, frame) -> None:
    raise SystemExit(EXIT_STOPPED)


def _ignore_signal(signum, frame) -> None:
    """The HTTP server handles SIGTERM once it runs. Its handler chains to this one, which must
    not turn a clean drain into an error."""


_renderer = None


def _start_progress(board, name: str | list[str]) -> None:
    """Show startup progress (every rank and model of ``board``) from this process."""
    global _renderer
    from resin_models_gaudi.loading.progress import Renderer

    # Under systemd (Type=notify), `systemctl status` shows the step that loads now.
    status = (lambda text: _notify(f"STATUS={text}")) if os.environ.get("NOTIFY_SOCKET") else None
    _renderer = Renderer(
        board, [name] if isinstance(name, str) else name, status=status
    ).start()


def _stop_progress() -> None:
    global _renderer
    if _renderer is not None:
        _renderer.stop()
        _renderer = None


def _remember_load(spec, checkpoint, **options) -> None:
    """Weight totals of this checkpoint and these options, measured at the last load."""
    from resin_models_gaudi.loading.progress import progress
    from resin_models_gaudi.synapse.build import default_cache_root

    path = Path(checkpoint).resolve()
    marker = next(
        (
            f
            for f in (path / "model.safetensors.index.json", path / "config.json")
            if f.exists()
        ),
        path,
    )
    progress.remember(
        default_cache_root(),
        model=spec.name,
        checkpoint=str(path),
        changed=marker.stat().st_mtime_ns if marker.exists() else 0,
        **options,
    )


def _notify(message: str) -> None:
    """Send ``message`` to systemd (``Type=notify`` units), if it listens."""
    address = os.environ.get("NOTIFY_SOCKET")
    if not address:
        return
    if address.startswith("@"):
        address = "\0" + address[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.sendto(message.encode(), address)
    except OSError as error:
        logger.debug(f"systemd notification failed: {error}")


def _ready(status: str) -> None:
    from resin_models_gaudi.loading.progress import progress

    progress.phase("ready")
    _stop_progress()
    _notify(f"READY=1\nSTATUS={status}")


def _serve_http(server, host: str, port: int, status: str, line: str) -> None:
    """Serve on ``host``:``port``. Startup ends (``_ready``, then the ready ``line``) only when
    the listener is bound, so a client that starts at that line, or when systemd marks the
    service active, connects. A SIGTERM before then drains the server once it listens: a
    normal stop, not a stop during startup."""
    # One handler for the whole call: the server's own SIGTERM handler chains to it, and a
    # handler set after the server's would replace the server's.
    state = {"listening": False, "pending": False}

    def on_sigterm(signum, frame) -> None:
        if not state["listening"]:
            state["pending"] = True

    def listening() -> None:
        _ready(status)
        logger.info(line)
        state["listening"] = True
        if state["pending"]:
            os.kill(os.getpid(), signal.SIGTERM)  # to the server's handler, which drains it

    signal.signal(signal.SIGTERM, on_sigterm)
    try:
        server.serve(host=host, port=port, on_listening=listening)
    except KeyboardInterrupt:
        pass  # Ctrl-C reaches Python as well as the HTTP server


def _gaudi_presets() -> str:
    lines = ["Presets (ranks, context length, maximum batch, draft tokens):"]
    for spec in MODEL_SPECS.values():
        lines.append(
            f"  {spec.name}: {spec.world_size}, {spec.context_length}, "
            f"{spec.max_batch_size}, {spec.draft_tokens}"
        )
    return "\n".join(lines)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=_gaudi_presets(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=MODEL_SPECS,
        metavar="MODEL",
        required=True,
        help="the preset to serve; the presets are below",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="local checkpoint directory (default: the preset's published volume path)",
    )
    parser.add_argument(
        "--alias", help="model name in the API (default: the repository name)"
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="local tokenizer directory, for presets whose weights ship without one",
    )
    parser.add_argument(
        "--ranks",
        type=positive_int,
        help="cards to split a tensor-parallel preset across (default: the preset's). A new "
        "count needs a cold compile. Mistral-Large-4 needs 8 for the full model",
    )
    parser.add_argument(
        "--num-layers",
        type=positive_int,
        help="test only: run the first N layers of a tensor-parallel preset, skip the fit check",
    )
    parser.add_argument(
        "--numa",
        choices=("auto", "off", "required"),
        default="auto",
        help="bind each process to its card's NUMA node (default: %(default)s)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="IP address to listen on. The server has no authentication (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="port to listen on (default: %(default)s)",
    )
    parser.add_argument(
        "--context-length",
        type=positive_int,
        help="maximum tokens per sequence (default: the preset's)",
    )
    parser.add_argument(
        "--max-batch-size",
        type=positive_int,
        help="maximum sequences in a batch (default: the preset's)",
    )
    parser.add_argument(
        "--prefill-chunk-size",
        type=positive_int,
        help="tokens per prefill chunk. A smaller chunk stops decode for a shorter time "
        "(default: the preset's; Qwen3.8-27B takes 512, 1024, 2048 or 4096, GLM-5.3-Flash "
        "1024, 2048 or 4096, Mistral-Large-4 a power of two >= 1024, its widest launch, sized "
        "at startup)",
    )
    parser.add_argument(
        "--draft-tokens",
        type=nonnegative_int,
        help="tokens to draft per decode step, 0 for none (default: the preset's)",
    )
    parser.add_argument(
        "--assistant-checkpoint",
        type=Path,
        help="local draft model checkpoint directory (default: the preset's volume)",
    )
    for kind in ("k", "v"):
        parser.add_argument(
            f"--cache-type-{kind}",
            choices=("bf16", "fp8"),
            help=f"type of the {kind.upper()} cache pages: Qwen3.5/3.8 takes bf16 or fp8, the "
            "other models one type (default: the model's)",
        )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        help="share of free card memory for the KV cache, above 0 and below 1. A new value "
        "recompiles recipes at startup. One-card presets only (default: the preset's)",
    )
    parser.add_argument(
        "--overlap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="start the next decode step while the card runs the current step",
    )
    parser.add_argument(
        "--step-transport",
        choices=("auto", "gloo", "socket", "native"),
        default="auto",
        help="how the ranks of a multi-card model exchange each step: Unix sockets "
        "(socket), torch Gloo (gloo), socket when all ranks are on this host (auto), or "
        "native: rank 0's native executor sends each step to the others, with no Python "
        "in the step path (Qwen3.5/3.8, DeepSeek-V4.1 and GLM-5.3-Flash). Multi-card presets "
        "only (default: %(default)s)",
    )
    parser.add_argument(
        "--optimize",
        choices=("latency", "throughput"),
        default="latency",
        help="for replicas on the cards of the first model of a layout: latency lets the first "
        "model go first; throughput lets both models run at once (default: %(default)s)",
    )
    parser.add_argument(
        "--decode-share",
        type=decode_share,
        default="50:90",
        metavar="MIN[:MAX]",
        help="percent of card time for decode while a prompt prefills: MIN with no sequence "
        "in decode, up to MAX when every batch position decodes. One number fixes it. 0 runs "
        "each prompt's whole prefill before decode (default: %(default)s)",
    )
    parser.add_argument(
        "--prefix-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="let a new request reuse cached prompt prefixes",
    )
    parser.add_argument(
        "--request-queue-capacity",
        type=positive_int,
        default=1024,
        help="requests that can wait for admission before HTTP 429 (default: %(default)s)",
    )
    parser.add_argument(
        "--stream-interval",
        type=positive_int,
        default=1,
        help="generated tokens per streamed chunk (default: %(default)s)",
    )
    parser.add_argument(
        "--drain-timeout",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="on SIGTERM, time for requests in flight to finish (default: %(default)s)",
    )
    add_gaudi_arguments(parser)
    return parser


def _validate(parser, args, spec) -> None:
    """Report what a run would otherwise find only after a load of several minutes."""
    apply_gaudi_arguments(args)
    try:
        validate_gaudi_options(
            spec,
            max_batch_size=args.max_batch_size or spec.max_batch_size,
            prefill_chunk_size=args.prefill_chunk_size or spec.prefill_chunk_size,
            draft_tokens=resolve_draft_tokens(spec, args),
            assistant=args.assistant_checkpoint,
        )
        resolve_cache_types(spec, args)
    except ValueError as error:
        parser.error(str(error))
    if (
        args.gpu_memory_utilization is not None
        and not 0 < args.gpu_memory_utilization < 1
    ):
        parser.error("--gpu-memory-utilization takes a value above 0 and below 1")
    problem = check_listen_address(args.host, args.port)
    if problem is not None:
        parser.error(problem)


def main(argv=None, on_error=None) -> None:
    """``on_error`` replaces the usage error of this module's own parser."""
    parser = argument_parser()
    if on_error is not None:
        parser.error = on_error
    args = parser.parse_args(argv)
    if not logging.getLogger("wright").handlers:
        setup_logging()
    spec = get_model(args.model)
    _validate(parser, args, spec)
    if args.optimize == "throughput":
        parser.error(
            "--optimize throughput is for replicas on the cards of the first model of a layout"
        )
    if _spans_cards(spec, args):
        return serve_gaudi_ranks(parser, args, spec)
    _check_one_card(parser, args, spec)
    from resin_models_gaudi.loading.progress import Board, progress
    from resin_models_gaudi.synapse.runtime import claim_card

    logger.info(f"settings: {describe_settings()}")
    signal.signal(signal.SIGTERM, _stop_while_loading)
    board = Board()
    progress.attach(board)
    _start_progress(board, args.alias or spec.repo_id)
    try:
        with claim_card(0, numa=args.numa) as claim:
            if claim is None:
                logger.error(f"{spec.name} needs 1 card, and none is free")
                raise SystemExit(1)
            logger.info(f"card: module {claim.card.module}")
            try:
                _serve_one(args, spec, f"hpu:{claim.card.accel}")
            except KeyboardInterrupt:
                logger.info("stopped during startup")
            except SystemExit as stopped:
                if stopped.code != EXIT_STOPPED:
                    raise
                logger.info("stopped during startup")
    finally:
        _stop_progress()


def _spans_cards(spec, args) -> bool:
    """Whether the model runs on several cards: its runner splits it and it takes more than one rank (the
    Qwen models run on one card unless ``--ranks`` asks for more)."""
    return gaudi_supports_ranks(spec) and (getattr(args, "ranks", None) or spec.world_size) > 1


def _check_one_card(
    parser, args, spec, *, replica: bool = False, spans: bool = False
) -> None:
    """Refuse the flags that a one-card model does not use. A replica of a layout can take
    ``--ranks 1``: the layout gives each model a ``--ranks``. ``spans``: a replica that spans cards
    (its ``--ranks``), which main_layout checked."""
    if replica and not spans and args.ranks not in (None, 1):
        parser.error(
            f"{spec.name} runs one replica on each card of its --gpu list: give it --ranks 1"
        )
    for flag, value in (
        ("--ranks", None if replica else args.ranks),
        ("--num-layers", args.num_layers),
        ("--tokenizer", args.tokenizer),
        # Qwen3.8-27B takes a DFlash 2 or DSpine drafter directory, which validate_gaudi_options
        # checked
        (
            "--assistant-checkpoint",
            None if spec.family == "qwen3_5" else args.assistant_checkpoint,
        ),
    ):
        if value is not None:
            parser.error(f"{spec.name} runs on one card and does not use {flag}")


def main_models(
    argvs: list[list[str]],
    *,
    shares: list[float] | None = None,
    floors: list[int] | None = None,
    on_error=None,
) -> None:
    """Serve several models on one card: one ``main`` argument list per model (each checked as
    for one model). Server settings (host, port, flags of the frontend) come from the first; each
    model keeps its own limits, prefix cache, overlap and decode share. The card's memory share
    (``--gpu-memory-utilization``) is one for all the models."""
    parser = argument_parser()
    if on_error is not None:
        parser.error = on_error
    if not logging.getLogger("wright").handlers:
        setup_logging()
    shares = shares or [1.0] * len(argvs)
    floors = floors or [0] * len(argvs)
    if len(shares) != len(argvs) or len(floors) != len(argvs):
        parser.error("give one --share and one --kv-floor for each model, or none")
    models, first = [], None
    for argv, share, floor in zip(argvs, shares, floors, strict=True):
        args = parser.parse_args(argv)
        spec = get_model(args.model)
        _validate(parser, args, spec)
        if args.optimize == "throughput":
            parser.error(
                "--optimize throughput is for replicas on the cards of the first model of a "
                "layout; models on one card take turns"
            )
        if _spans_cards(spec, args) or spec.family != "qwen3_5":
            parser.error(
                f"{spec.name} cannot share a card yet: only one-card Qwen models do"
            )
        if args.assistant_checkpoint is not None:
            parser.error("a model that shares a card cannot use a DFlash drafter yet")
        if not share > 0:
            parser.error("--share takes a positive number")
        if floor < 0:
            parser.error("--kv-floor takes 0 or more slabs")
        if first is not None and args.gpu_memory_utilization != first.gpu_memory_utilization:
            parser.error(
                "several models on one card take one --gpu-memory-utilization, the share of the "
                "card for all of them"
            )
        first = first or args
        models.append(
            CardModel(
                spec,
                Path(args.checkpoint) if args.checkpoint else spec.path,
                alias=args.alias,
                context_length=args.context_length,
                max_batch_size=args.max_batch_size,
                prefill_chunk_size=args.prefill_chunk_size,
                draft_tokens=resolve_draft_tokens(spec, args),
                cache_types=resolve_cache_types(spec, args),
                share=share,
                kv_floor=floor,
                prefix_cache=args.prefix_cache,
                overlap=args.overlap,
                decode_share=args.decode_share,
            )
        )
    assert first is not None
    logger.info(f"settings: {describe_settings()}")
    serve_card_models(
        models,
        host=first.host,
        port=first.port,
        numa=first.numa,
        stream_interval=first.stream_interval,
        request_queue_capacity=first.request_queue_capacity,
        drain_timeout=first.drain_timeout,
        memory_utilization=first.gpu_memory_utilization or 0.9,
    )


def main_layout(
    argvs: list[list[str]], *, shares: list[float] | None = None, on_error=None
) -> None:
    """Serve several models on the cards of their ``--gpu`` lists: one ``main`` argument list per
    model.

    The first model spans the cards of its ``--gpu`` list (its ``--ranks``); rank 0 runs the
    HTTP server. Every other model runs on one card, as one replica per card of its ``--gpu``
    list, and rank 0 routes each request to one replica. A replica model's cards are cards of its
    own, or cards of the first model (rank 0's too): then each replica shares its card with a rank
    of the first model. ``shares`` are the models' weights in the card time of a shared
    card (default: 1 for the first model, 0 for the others, which then step only when the first
    model is idle on the card). Server settings come from the first model."""
    parser = argument_parser()
    if on_error is not None:
        parser.error = on_error
    if not logging.getLogger("wright").handlers:
        setup_logging()
    parsed = []
    for argv in argvs:
        args = parser.parse_args(argv)
        spec = get_model(args.model)
        _validate(parser, args, spec)
        if args.gpu is None:
            parser.error(f"give each model its cards with --gpu ({spec.name} has none)")
        try:
            modules = [int(module) for module in str(args.gpu).split(",") if module]
        except ValueError:
            parser.error(f"--gpu {args.gpu}: give module IDs such as 0,1,2,3")
        parsed.append((argv, args, spec, modules))
    (_, first, first_spec, first_modules), others = parsed[0], parsed[1:]
    if not _spans_cards(first_spec, first):
        parser.error(
            f"the first model must span several cards; {first_spec.name} runs on one"
        )
    if first.gpu_memory_utilization is not None:
        parser.error(
            "the first model of a layout makes its KV cache from the memory that the replicas "
            "leave: give --gpu-memory-utilization to a replica model"
        )
    throughput = first.optimize == "throughput"
    if throughput and shares is not None:
        parser.error(
            "--share weighs the card time of --optimize latency; with throughput both models "
            "run at once"
        )
    if shares is None:
        shares = [1.0] + [0.0] * len(others)
    if len(shares) != len(parsed):
        parser.error("give one --share for each model, or none")
    if not shares[0] > 0 or any(share < 0 for share in shares):
        parser.error("--share takes a positive weight for the first model and 0 or more for the others")
    rank_modules = first_modules[: first.ranks or first_spec.world_size]
    used: set[int] = set()
    replicated = []
    for (argv, args, spec, modules), share in zip(others, shares[1:], strict=True):
        if gaudi_supports_ranks(spec) and spec.world_size > 1:
            parser.error(
                f"only the first model can span cards; {spec.name} runs on one card a replica"
            )
        if used & set(modules):
            parser.error(
                f"{spec.name} and another model both use modules {sorted(used & set(modules))}"
            )
        used |= set(modules)
        shared = set(modules) & set(first_modules)
        if shared and not set(modules) <= set(rank_modules):
            parser.error(
                f"{spec.name}: give its replicas cards of the first model's ranks "
                f"(modules {','.join(map(str, rank_modules))}) or cards of their own, not both"
            )
        if shared and spec.family != "qwen3_5":
            parser.error(
                f"{spec.name} cannot share the first model's cards yet: only one-card Qwen models can"
            )
        # A Qwen3.5/3.8 replica can span R cards of the first model's ranks (--ranks R): R cards in --gpu order
        # a replica, its rank 0 in the process of the first card's rank. Its ranks exchange each step natively.
        replica_ranks = args.ranks or 1
        if replica_ranks > 1:
            if spec.family != "qwen3_5" or not shared:
                parser.error(
                    f"{spec.name} runs one replica on each card of its --gpu list: give it --ranks 1 "
                    "(a Qwen3.5/3.8 replica spans cards only on cards of the first model's ranks)"
                )
            if feature_enabled(first, "oldqwen"):
                parser.error(
                    f"{spec.name} replicas of {replica_ranks} cards: {QWEN_NEEDS_NATIVE}"
                )
            if len(modules) % replica_ranks:
                parser.error(
                    f"{spec.name}: --ranks {replica_ranks} does not divide its {len(modules)} cards"
                )
        # Replicas on the cards of a first model that defers its cache share the KV cache of
        # those cards with it; else (or with their own --gpu-memory-utilization, or when either
        # model runs a drafter model) each takes a fixed share of its card's memory.
        shared_kv = (
            bool(shared)
            and replica_ranks == 1
            and gaudi_defers_cache(first_spec)
            and args.gpu_memory_utilization is None
            and not _runs_drafter(first_spec, first)
            and not _runs_drafter(spec, args)
        )
        if shared and not shared_kv and args.gpu_memory_utilization is None:
            argv = [*argv, "--gpu-memory-utilization", str(COLOCATED_MEMORY)]
            args = parser.parse_args(argv)
        # Check what the replicas check when they load, before the first model loads.
        _check_one_card(parser, args, spec, replica=True, spans=replica_ranks > 1)
        draft_tokens = resolve_draft_tokens(spec, args)
        try:
            checkpoint = require_checkpoint(
                spec, checkpoint=args.checkpoint, draft_tokens=draft_tokens
            )
            check_gaudi_drafts(
                spec, checkpoint, draft_tokens, args.assistant_checkpoint
            )
        except (RuntimeError, FileNotFoundError, ValueError) as error:
            parser.error(f"{spec.name}: {error}")
        name = args.alias or spec.repo_id
        served = dict(name=name, modules=modules, argv=argv)
        if shared:
            # rank -> the replica that shares its card
            served["ranks"] = {rank_modules.index(m): i for i, m in enumerate(modules)}
            served["share"] = share / shares[0]
            served["optimize"] = first.optimize
            served["shared_kv"] = shared_kv
            if replica_ranks > 1:
                served["replica_ranks"] = replica_ranks
        replicated.append(served)
    if throughput and not any("ranks" in served for served in replicated):
        parser.error(
            "--optimize throughput is for replicas on the cards of the first model; these "
            "replicas have cards of their own"
        )
    names = [first.alias or first_spec.repo_id] + [
        served["name"] for served in replicated
    ]
    if len(set(names)) != len(names):
        parser.error("give each model its own --alias")
    # The first model's settings (its --gpu) apply in this process and its ranks.
    apply_gaudi_arguments(first)
    serve_gaudi_ranks(parser, first, first_spec, replicated)


def _runs_drafter(spec, args) -> bool:
    """Whether the model drafts with a drafter model of its own (``--assistant-checkpoint``: DFlash 2
    or DSpine). Its runner does not take a cache arena that other models share yet."""
    return args.assistant_checkpoint is not None and resolve_draft_tokens(spec, args) > 0


def resolve_cache_types(spec, args) -> tuple[str, str]:
    """``--cache-type-k`` and ``--cache-type-v``, else the model's types (ValueError: a type the model does not
    take)."""
    return check_cache_types(spec, (args.cache_type_k, args.cache_type_v))


def resolve_draft_tokens(spec, args) -> int:
    """``--spec-draft-n-max``, else the preset's drafts, or a given drafter's block."""
    if args.draft_tokens is not None:
        return args.draft_tokens
    return default_draft_tokens(spec, args.assistant_checkpoint)


@contextmanager
def _one_card_engine(args, spec, device: str, rank: int = 0, *, shared_card: bool = False):
    """Load a one-card model on ``device`` and warm it up; yield (engine, recipe, checkpoint).
    The Engine and the runner close when the body ends; with ``shared_card``, the card stays
    open for the other model on it."""
    with _one_card_model(args, spec, device, rank, shared_card=shared_card) as loaded:
        yield loaded.engine, loaded.recipe, loaded.checkpoint


@contextmanager
def _one_card_model(
    args,
    spec,
    device: str,
    rank: int = 0,
    *,
    shared_card: bool = False,
    defer_cache: bool = False,
    native_ranks=None,
):
    """Load a one-card model on ``device``; yield an object with its ``runner``, ``recipe``,
    ``checkpoint`` and warmed-up ``engine``. ``native_ranks``: this rank's group of a Qwen replica that
    spans cards (``resin._resin.GaudiRanks``); a rank above 0 has no Engine (its runner serves rank 0's
    steps). With ``defer_cache``, the runner loads its weights
    only: ``engine`` is None until ``start_engine(cache_pool)``, after the runner's ``attach``.
    The Engine and the runner close when the body ends; with ``shared_card``, the card stays open
    for the other model on it."""
    from types import SimpleNamespace

    context_length = args.context_length or spec.context_length
    max_batch_size = args.max_batch_size or spec.max_batch_size
    prefill_chunk_size = args.prefill_chunk_size or spec.prefill_chunk_size
    draft_tokens = resolve_draft_tokens(spec, args)
    try:
        checkpoint = require_checkpoint(
            spec, checkpoint=args.checkpoint, draft_tokens=draft_tokens
        )
        check_gaudi_drafts(spec, checkpoint, draft_tokens, args.assistant_checkpoint)
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        logger.error(f"{spec.name}: {error}")
        raise SystemExit(1) from None
    ranks = 1 if native_ranks is None else native_ranks.size
    estimate_kv_pool(
        current_device_free(), checkpoint_weight_bytes(spec, checkpoint), ranks
    )
    _remember_load(
        spec,
        checkpoint,
        ranks=ranks,
        draft_tokens=draft_tokens,
        assistant=args.assistant_checkpoint,
    )

    from resin import Engine, SchedulerConfig

    runner = create_runner(
        spec,
        checkpoint,
        device=device,
        context_length=context_length,
        max_batch_size=max_batch_size,
        prefill_chunk_size=prefill_chunk_size,
        draft_tokens=draft_tokens,
        assistant_checkpoint=args.assistant_checkpoint,
        gpu_memory_utilization=args.gpu_memory_utilization,
        defer_cache=defer_cache,
        native_ranks=native_ranks,
        cache_types=resolve_cache_types(spec, args),
    )
    loaded = SimpleNamespace(
        runner=runner, recipe=None, checkpoint=checkpoint, engine=None
    )

    def start_engine(cache_pool=None):
        """Make the Engine (in ``cache_pool``, else a pool of its own) and warm it up."""
        loaded.engine = Engine(
            runner,
            max_running_requests=max_batch_size,
            context_length=context_length,
            prefix_cache=args.prefix_cache,
            enable_overlap=args.overlap,
            scheduler_config=SchedulerConfig(
                max_batch_tokens=batch_tokens(runner, prefill_chunk_size),
                prefill_chunk_size=prefill_chunk_size,
                decode_share=args.decode_share,
            ),
            cache_pool=cache_pool,
        )
        report_kv_pool(runner, loaded.engine, 1, context_length, max_batch_size)
        from resin_models_gaudi.loading.progress import progress

        progress.phase("warm-up")
        logger.info("warming up")
        warm_engine(loaded.engine, loaded.recipe, spec, rank)
        return loaded.engine

    loaded.start_engine = start_engine
    try:
        loaded.recipe = create_recipe(spec, checkpoint, runner=runner)
        follower = native_ranks is not None and native_ranks.rank > 0
        if not defer_cache and not follower:
            start_engine()
        yield loaded
    finally:
        engine = loaded.engine
        if engine is not None and not engine.closed:
            engine.close()
        compiles = getattr(runner, "compiles_after_start", None)
        if compiles is not None:
            logger.info(f"graph compilations after startup: {compiles}")
        close = getattr(runner, "close", None)
        if close is not None:
            if shared_card:
                close(device=False)
            else:
                close()


def _serve_one(args, spec, device: str) -> None:
    from resin.server import HttpServer

    with _one_card_engine(args, spec, device) as (engine, recipe, checkpoint):
        server = HttpServer(
            engine,
            recipe=recipe,
            model=args.alias or spec.repo_id,
            stream_interval=args.stream_interval,
            request_queue_capacity=args.request_queue_capacity,
            drain_seconds=args.drain_timeout,
            enable_anthropic=spec.enable_anthropic,
            capabilities=model_capabilities(spec, recipe, checkpoint),
        )
        _serve_http(
            server,
            args.host,
            args.port,
            f"serving {spec.name} at http://{args.host}:{args.port}",
            f"{spec.name} is ready at http://{args.host}:{args.port}",
        )


@dataclass(frozen=True)
class CardModel:
    """One of several models that share one card: its preset, checkpoint and limits (None:
    the preset's), its API name, its weight in the card's time, its reserved cache slabs, and the
    settings of its Engine."""

    spec: ModelSpec
    checkpoint: Path
    alias: str | None = None
    context_length: int | None = None
    max_batch_size: int | None = None
    prefill_chunk_size: int | None = None
    draft_tokens: int | None = None
    cache_types: tuple[str, str] | None = None
    share: float = 1.0
    kv_floor: int = 0
    prefix_cache: bool = True
    overlap: bool = True
    decode_share: tuple[float, float] | None = None

    @property
    def name(self) -> str:
        return self.alias or self.spec.repo_id


def serve_card_models(
    models: list[CardModel],
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    numa: str = "auto",
    stream_interval: int = 1,
    request_queue_capacity: int = 1024,
    drain_timeout: float = 30.0,
    memory_utilization: float = 0.9,
) -> None:
    """Serve several one-card models on one card: one Engine each, one HTTP server, one cache
    pool that they share. Each model loads its weights first; the cache arena then takes what
    the weights leave free, and every model compiles its recipes for that arena."""
    from resin import CachePool, Engine, SchedulerConfig
    from resin.server import HttpServer
    from resin_models_gaudi.arena import plan_shared_arena
    from resin_models_gaudi.loading.progress import Board, progress
    from resin_models_gaudi.synapse.runtime import claim_card, current_device

    names = [model.name for model in models]
    if len(set(names)) != len(names):
        raise ValueError(f"two models of one card have one name: {names}")
    signal.signal(signal.SIGTERM, _stop_while_loading)
    board = Board(1, len(models))
    progress.attach(board, 0, 0)
    _start_progress(board, names)
    runners, engines, recipes = [], [], []
    try:
        with claim_card(0, numa=numa) as claim:
            if claim is None:
                logger.error(f"{len(models)} models need 1 card, and none is free")
                raise SystemExit(1)
            logger.info(f"card: module {claim.card.module}, {len(models)} models")
            for index, model in enumerate(models):
                progress.attach(board, 0, index)
                draft_tokens = (
                    model.draft_tokens
                    if model.draft_tokens is not None
                    else model.spec.draft_tokens
                )
                checkpoint = require_checkpoint(
                    model.spec, checkpoint=model.checkpoint, draft_tokens=draft_tokens
                )
                _remember_load(
                    model.spec,
                    checkpoint,
                    ranks=1,
                    draft_tokens=draft_tokens,
                    assistant=None,  # the same key as a load of this model alone
                )
                runners.append(
                    create_runner(
                        model.spec,
                        checkpoint,
                        device=f"hpu:{claim.card.accel}",
                        context_length=model.context_length
                        or model.spec.context_length,
                        max_batch_size=model.max_batch_size
                        or model.spec.max_batch_size,
                        prefill_chunk_size=model.prefill_chunk_size
                        or model.spec.prefill_chunk_size,
                        draft_tokens=draft_tokens,
                        defer_cache=True,
                        cache_types=model.cache_types,
                    )
                )
                recipes.append(
                    create_recipe(model.spec, checkpoint, runner=runners[-1])
                )
                # its recipes compile once every model is loaded and the arena is known
                progress.phase("recipes", note="waits for the other models to load")
            device = current_device()
            arena = plan_shared_arena(
                device,
                [runner.own_slab_rows for runner in runners],
                [runner.scratch_rows for runner in runners],
                memory_utilization=memory_utilization,
                reserve=max(runner.model.workspace_reserve for runner in runners),
            )
            logger.info(
                f"KV pool: {arena.nbytes / 2**30:.1f} GiB shared by {len(models)} models; "
                f"{arena.parent_capacity:,} slabs of {arena.slab_rows * 512 / 2**20:.0f} MiB"
            )
            pool = CachePool(arena.parent_capacity)
            for index, (model, runner, recipe) in enumerate(
                zip(models, runners, recipes, strict=True)
            ):
                progress.attach(board, 0, index)
                runner.attach(arena)
                chunk = model.prefill_chunk_size or model.spec.prefill_chunk_size
                context_length = model.context_length or model.spec.context_length
                engine = Engine(
                    runner,
                    max_running_requests=model.max_batch_size
                    or model.spec.max_batch_size,
                    context_length=context_length,
                    prefix_cache=model.prefix_cache,
                    enable_overlap=model.overlap,
                    scheduler_config=SchedulerConfig(
                        max_batch_tokens=batch_tokens(runner, chunk),
                        prefill_chunk_size=chunk,
                        decode_share=model.decode_share,
                    ),
                    cache_pool=pool,
                    cache_floor=model.kv_floor,
                )
                engines.append(engine)
                progress.phase("warm-up")
                logger.info(f"{model.name}: warming up")
                warm_engine(engine, recipe, model.spec)
                progress.phase("ready")
            server = HttpServer(
                engines[0],
                recipe=recipes[0],
                model=names[0],
                stream_interval=stream_interval,
                request_queue_capacity=request_queue_capacity,
                drain_seconds=drain_timeout,
                enable_anthropic=models[0].spec.enable_anthropic,
                capabilities=model_capabilities(
                    models[0].spec, recipes[0], models[0].checkpoint
                ),
            )
            for model, engine, recipe in zip(
                models[1:], engines[1:], recipes[1:], strict=True
            ):
                server.register(
                    engine,
                    recipe=recipe,
                    model=model.name,
                    stream_interval=stream_interval,
                    capabilities=model_capabilities(
                        model.spec, recipe, model.checkpoint
                    ),
                    share=model.share,
                )
            for runner in runners:
                runner.mark_started()
            _serve_http(
                server,
                host,
                port,
                f"serving {', '.join(names)} at http://{host}:{port}",
                f"{', '.join(names)} ready at http://{host}:{port}",
            )
    finally:
        _stop_progress()
        for engine in engines:
            if not engine.closed:
                engine.close()
        for model, runner in zip(models, runners):
            compiles = getattr(runner, "compiles_after_start", None)
            if compiles is not None:
                logger.info(
                    f"{model.name}: graph compilations after startup: {compiles}"
                )
            runner.close(device=False)
        if runners:
            runners[0].device.close()


def card_kv_sides(replicated: list[dict]) -> dict[int, int]:
    """rank -> the slab table side of the replica on that rank's card, for the replicas (of any model) that
    share their card's KV cache with the first model: 1, 2, ... in rank order (the first model holds side 0)."""
    sharing = sorted(
        rank
        for served in replicated
        if served.get("shared_kv")
        for rank in served.get("ranks", {})
    )
    return {rank: side for side, rank in enumerate(sharing, 1)}


def serve_gaudi_ranks(parser, args, spec, replicated: list[dict] | None = None) -> None:
    """One process per card: every rank takes a free card and binds to its NUMA node, builds the
    runner, rank 0 serves the Engine and HTTP, the others follow over the Gloo control plane.

    ``replicated`` (from :func:`main_layout`) adds models that run as one-card replicas, each with
    its own Engine, which rank 0's HTTP server routes requests to: one process per card of their
    own, or, for a model whose cards are the ranks' cards, in the process of that rank, on a
    runtime context of its own and behind the card's gate (``card_gate.CardGate``)."""
    from resin.process import ProcessError, spawn
    from resin_models_gaudi.synapse.cards import visible_cards

    ranks = args.ranks or spec.world_size
    draft_tokens = resolve_draft_tokens(spec, args)
    if args.gpu_memory_utilization is not None:
        parser.error(f"{spec.name} sizes its KV cache from free memory on every rank")
    if spec.family == "qwen3_5" and feature_enabled(args, "oldqwen"):
        parser.error(f"{spec.name} on {ranks} cards: {QWEN_NEEDS_NATIVE}")
    if args.step_transport == "native" and spec.family not in NATIVE_RANK_FAMILIES:
        parser.error(
            f"--step-transport native: {spec.name} has no native rank dispatch "
            "(Qwen3.5/3.8, DeepSeek-V4.1 and GLM-5.3-Flash only)"
        )
    if args.step_transport == "native" and replicated and spec.family not in ("deepseek_v41", "glm5_3_flash"):
        parser.error(
            "--step-transport native: in layouts of several models, DeepSeek-V4.1 and GLM-5.3-Flash only (their "
            "ranks take the cards' gates)"
        )
    try:
        checkpoint = require_checkpoint(
            spec,
            checkpoint=args.checkpoint,
            tokenizer_checkpoint=args.tokenizer,
            assistant_checkpoint=args.assistant_checkpoint,
            draft_tokens=draft_tokens,
        )
    except (RuntimeError, FileNotFoundError) as error:
        parser.error(f"{spec.name}: {error}")
    try:
        cards = visible_cards()
    except ValueError as error:
        parser.error(f"--gpu: {error}")
    try:
        check_gaudi_ranks(spec, checkpoint, ranks)
    except ValueError as error:
        parser.error(f"--ranks {ranks}: {error}")
    if len(cards) < ranks:
        parser.error(
            f"{spec.name} at {ranks} ranks needs {ranks} cards, and {len(cards)} are visible"
        )
    logger.info(f"settings: {describe_settings()}")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    options = dict(
        model=spec.name,
        ranks=ranks,
        num_layers=args.num_layers,
        numa=args.numa,
        checkpoint=str(checkpoint),
        weight_bytes=checkpoint_weight_bytes(spec, checkpoint),
        tokenizer=None if args.tokenizer is None else str(args.tokenizer),
        alias=args.alias,
        assistant=(
            None
            if args.assistant_checkpoint is None
            else str(args.assistant_checkpoint)
        ),
        host=args.host,
        port=args.port,
        context_length=args.context_length or spec.context_length,
        max_batch_size=args.max_batch_size or spec.max_batch_size,
        prefill_chunk_size=args.prefill_chunk_size or spec.prefill_chunk_size,
        draft_tokens=draft_tokens,
        cache_types=resolve_cache_types(spec, args),
        overlap=args.overlap,
        decode_share=args.decode_share,
        step_transport=args.step_transport,
        prefix_cache=args.prefix_cache,
        stream_interval=args.stream_interval,
        request_queue_capacity=args.request_queue_capacity,
        drain_timeout=args.drain_timeout,
        parent=os.getpid(),
        control_port=port,
    )
    import tempfile

    from resin_models_gaudi.loading.progress import Board

    with (
        tempfile.TemporaryDirectory(prefix="wright-progress-") as board_directory,
        ExitStack() as cleanup,
    ):
        # The ranks report startup progress to one shared board, and rank 0 shows it.
        options["progress"] = str(Path(board_directory) / "board")
        Board(ranks, 1, options["progress"])
        if args.step_transport == "native":
            from resin.backend.step_transport import socket_directory

            # rank 0 listens on this socket, and the other ranks join it with the token
            group = socket_directory("wright-ranks-", "group")
            cleanup.callback(shutil.rmtree, group, ignore_errors=True)
            options["native_path"] = os.path.join(group, "group")
            options["native_token"] = secrets.token_bytes(16)
        roles: list[tuple] = [("rank", rank) for rank in range(ranks)]
        if replicated:
            from resin.backend.step_transport import socket_directory

            # Rank 0 listens on one socket a replicated model; its replicas join with the token.
            # The socket paths must fit the 108-byte limit of Unix sockets.
            sockets = socket_directory(
                "wright-replicas-", f"replicas-{len(replicated) - 1}"
            )
            cleanup.callback(shutil.rmtree, sockets, ignore_errors=True)
            options["replica_token"] = secrets.token_bytes(16)
            options["replicated"] = []
            options["hosted"] = {}
            options["replica_groups"] = {}
            for model, served in enumerate(replicated):
                path = os.path.join(sockets, f"replicas-{model}")
                spans = served.get("replica_ranks", 1)
                count = len(served["modules"]) // spans
                options["replicated"].append(
                    dict(name=served["name"], count=count, path=path)
                )
                if "ranks" in served:
                    # each replica shares the card of a rank, in that rank's process; a replica that spans
                    # cards has a rank in the process of each of its cards
                    for rank, card in served["ranks"].items():
                        options["hosted"][rank] = (model, card // spans, card % spans)
                    if spans > 1:
                        options["replica_groups"][model] = [
                            os.path.join(sockets, f"group-{model}-{index}")
                            for index in range(count)
                        ]
                else:
                    roles += [
                        ("replica", model, index)
                        for index in range(len(served["modules"]))
                    ]
            options["replica_models"] = replicated
            options["kv_sides"] = card_kv_sides(replicated)
            options["shared_kv"] = bool(options["kv_sides"])
            if options["hosted"]:
                # A rank takes the card of its own module: its replica's card.
                options["rank_modules"] = [
                    int(module) for module in str(args.gpu).split(",")
                ][:ranks]
        # A rank binds itself once it holds a card, so spawn does not place it.
        context = spawn(
            _serve_layout_process,
            args=(options, roles),
            devices=["hpu"] * len(roles),
            numa="off",
            join=False,
        )
        try:
            context.join_stopping_on_signals(args.drain_timeout + _SHUTDOWN_SECONDS)
        except ProcessError as error:
            if error.exitcode == EXIT_REPORTED:
                raise SystemExit(1) from None
            if error.exitcode == EXIT_STOPPED:
                logger.info("stopped during startup")
                return
            raise


def _serve_layout_process(index: int, options: dict, roles: list[tuple]) -> None:
    """A rank of the first model, or a replica of another (:func:`serve_gaudi_ranks`)."""
    role = roles[index]
    if role[0] == "rank":
        _serve_gaudi_rank(role[1], options)
    else:
        _serve_replica(role[1], role[2], options)


def _serve_replica(model: int, index: int, options: dict) -> None:
    """One card of a replicated model: load it, join rank 0's router, serve until rank 0 stops
    it. A failed Engine tells the router, which routes around this replica, and the process
    ends without taking the others down."""
    from resin._resin import _ReplicaServer
    from resin.process import die_with_parent
    from resin_models_gaudi.synapse.runtime import claim_card

    served = options["replica_models"][model]
    name = f"{served['name']} {index}"
    setup_logging(name=name)
    die_with_parent(options["parent"])
    # Rank 0 stops the replicas through their sockets.
    signal.signal(signal.SIGTERM, _ignore_signal)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ["HABANA_VISIBLE_MODULES"] = ",".join(map(str, served["modules"]))
    args = argument_parser().parse_args(served["argv"])
    spec = get_model(args.model)
    with claim_card(index, numa=options["numa"]) as claim:
        if claim is None:
            logger.error(f"{name}: no card of modules {served['modules']} is free")
            raise SystemExit(EXIT_REPORTED)
        logger.info(f"card: module {claim.card.module}")
        device = f"hpu:{claim.card.accel}"
        with _one_card_engine(args, spec, device) as (engine, recipe, checkpoint):
            capabilities = model_capabilities(spec, recipe, checkpoint)
            server = _ReplicaServer(
                options["replicated"][model]["path"],
                options["replica_token"],
                index,
                engine,
                recipe,
                model=served["name"],
                stream_interval=args.stream_interval,
                capabilities=capabilities.to_wire(),
                timeout=_REPLICA_TIMEOUT,
            )
            logger.info(f"{name} is ready")
            try:
                server.serve()
            except Exception as error:  # noqa: BLE001 (the router has the error and serves on)
                logger.error(f"{name} stopped: {error}")


def _serve_gaudi_rank(rank: int, options: dict) -> None:
    from resin.process import die_with_parent
    from resin_models_gaudi.loading.progress import Board, progress
    from resin_models_gaudi.synapse.runtime import claim_card

    setup_logging(rank)
    die_with_parent(options["parent"])
    # The parent asks rank 0 to stop, and rank 0 stops the others in order.
    signal.signal(signal.SIGTERM, _stop_while_loading)
    if rank != 0:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    board = Board(options["ranks"], 1, options["progress"])
    progress.attach(board, rank)
    if rank == 0:
        _start_progress(board, options["alias"] or get_model(options["model"]).repo_id)
    modules = options.get("rank_modules")
    try:
        with claim_card(
            rank,
            numa=options["numa"],
            module=None if modules is None else modules[rank],
        ) as claim:
            try:
                with _hosted_replica(rank, options, claim) as replica:
                    _serve_claimed_rank(rank, options, claim, replica)
            except KeyboardInterrupt:
                raise SystemExit(EXIT_STOPPED) from None
    finally:
        _stop_progress()


def _startup_problem(options: dict, states: list[tuple[int, int] | None]) -> str | None:
    """Why this rank group cannot start, before any rank spends minutes on weights. ``states`` is
    each rank's (module, free card memory), or None for a rank with no card."""
    held = [state[0] for state in states if state is not None]
    if len(held) < len(states):
        return (
            f"{options['model']} needs {len(states)} cards, and {len(held)} are available "
            f"(modules {','.join(map(str, sorted(held))) or 'none'})"
        )
    if options["num_layers"] is None:
        free = min(state[1] for state in states if state is not None)
        share = options["weight_bytes"] // len(states)
        if share > free:
            return (
                f"{options['model']} does not fit at {len(states)} ranks: its weights need "
                f"{share / 2**30:.0f} GiB per card, and {free / 2**30:.0f} GiB are free"
            )
    return None


@contextmanager
def _within(context, manager):
    """Enter and exit ``manager`` with runtime ``context`` current; the body runs on the context
    before it."""
    with context.use():
        value = manager.__enter__()
    try:
        yield value
    except BaseException as error:
        with context.use():
            if not manager.__exit__(type(error), error, error.__traceback__):
                raise
    else:
        with context.use():
            manager.__exit__(None, None, None)


@contextmanager
def _hosted_replica(rank: int, options: dict, claim):
    """The replica that shares this rank's card, if it has one: it loads first, on a runtime
    context of its own, because the rank's model sizes its cache from the memory that the
    replica leaves. Yields None, or an object with the card's ``gate`` (None with ``--optimize
    throughput``: both models run at once) and a ``start()`` that serves the replica on a thread
    of its own.

    When its card's KV cache is shared (``options["kv_sides"]`` has the rank) the replica has no
    cache yet: ``attach(arena)`` gives it the
    card's shared arena and returns its floor, then ``start_engine(cache_pool)`` makes its Engine
    in its pool of the card's slab table and warms it up."""
    hosted = options.get("hosted", {}).get(rank)
    if hosted is None or claim is None:
        yield None
        return
    import sys
    import threading
    from types import SimpleNamespace

    from resin._resin import _ReplicaServer
    from resin_models_gaudi.loading.progress import Board, progress
    from resin_models_gaudi.synapse.card_gate import CardGate
    from resin_models_gaudi.synapse.runtime import current_device

    model, index, local = hosted
    served = options["replica_models"][model]
    spans = served.get("replica_ranks", 1)
    name = f"{served['name']} {index}" + (f" rank {local}" if spans > 1 else "")
    args = argument_parser().parse_args(served["argv"])
    spec = get_model(args.model)
    device = current_device()
    # With no share, the replica steps only in the rank's idle time, on the queue slot of the rank's own stream.
    # With a share, it takes a slot of its own. With --optimize throughput, no gate and a slot of its own: the two
    # models run at once.
    #
    # A replica that spans cards takes a slot of its own. Its rank 0 steps in its card's gate; its other ranks
    # make rank 0's steps as they come (no gate).
    throughput = served.get("optimize") == "throughput"
    shared_slot = served["share"] == 0 and not throughput and spans == 1
    context = device.context(device.default_context.affinity if shared_slot else 0)
    logger.info(
        f"{name} shares this card (queue slot {context.affinity:#x}, "
        f"{'no gate' if throughput else 'the card gate'}) and loads first"
    )
    # The board has one row for each rank, where the rank's own model reports: the replica
    # reports to a board of its own, which nobody shows.
    board, hidden = progress.board, Board()
    progress.attach(hidden)
    # With a shared cache, the replica loads its weights only: the card's cache arena comes when
    # every model of the rank group is loaded (_share_card_cache).
    side = options.get("kv_sides", {}).get(rank)
    shared_kv = side is not None
    # It clears the slabs that cross the boundary of the card's cache, on a stream of their own.
    scrub_context = device.context(context.affinity) if shared_kv else None
    native_ranks = None
    if spans > 1:
        from resin._resin import GaudiRanks

        # the replica's ranks join rank 0's socket; the group makes the replica's communicator on its context
        native_ranks = GaudiRanks(
            options["replica_groups"][model][index],
            options["replica_token"],
            local,
            spans,
            timeout=_REPLICA_TIMEOUT,
        )
    follower = local > 0
    # rank 0 of its own: it runs its own warm-up request
    loading = _one_card_model(
        args,
        spec,
        f"hpu:{claim.card.accel}",
        shared_card=True,
        defer_cache=shared_kv,
        native_ranks=native_ranks,
    )
    with ExitStack() as stack:
        loaded = stack.enter_context(_within(context, loading))
        recipe, checkpoint = loaded.recipe, loaded.checkpoint
        progress.attach(board, rank)
        gate = None if throughput or follower else CardGate(device, share=served["share"])
        capabilities = model_capabilities(spec, recipe, checkpoint).to_wire()
        context_length = args.context_length or spec.context_length

        @contextmanager
        def replica_turn():
            """The replica's context, and its progress on its own board."""
            progress.attach(hidden)
            try:
                with context.use():
                    yield
            finally:
                progress.attach(board, rank)

        shared = {}
        scrub_lock = threading.Lock()

        def attach(arena) -> int:
            """Take the card's shared cache arena (its recipes compile for it); returns the slabs
            of one sequence at the replica's context length, its floor."""
            with replica_turn():
                loaded.runner.attach(arena)
            shared["arena"] = arena
            return loaded.runner.cache_layout.estimated_parent_count([context_length])

        def scrub(slabs: list[int]) -> None:
            """Clear ``slabs`` of the card's arena and wait (``CachePool(scrub=)``)."""
            from resin_models_gaudi.arena import ROW_BYTES

            arena = shared["arena"]
            nbytes = arena.slab_rows * ROW_BYTES
            assert scrub_context is not None  # a shared cache has one
            with scrub_lock, scrub_context.use():
                for slab in slabs:
                    device.memset32(arena.address + slab * nbytes, 0, nbytes // 4)
                device.sync()

        def start_engine(cache_pool) -> None:
            with replica_turn():
                loaded.start_engine(cache_pool)

        def serve() -> None:
            with context.use():
                if follower:
                    try:
                        calls = loaded.runner.serve()
                        logger.info(f"{name}: made {calls} runner calls of its rank 0")
                    except Exception as error:  # noqa: BLE001 (rank 0 meets it at its next step)
                        logger.error(f"{name} stopped: {error}")
                    return
                try:
                    server = _ReplicaServer(
                        options["replicated"][model]["path"],
                        options["replica_token"],
                        index,
                        loaded.engine,
                        recipe,
                        model=served["name"],
                        stream_interval=args.stream_interval,
                        capabilities=capabilities,
                        timeout=_REPLICA_TIMEOUT,
                        gate=gate,
                    )
                    logger.info(f"{name} is ready")
                    server.serve()
                except Exception as error:  # noqa: BLE001 (the router has the error)
                    logger.error(f"{name} stopped: {error}")

        thread = threading.Thread(target=serve, name=name, daemon=True)

        def start() -> None:
            # The two models' threads hand the interpreter over sooner (default 5 ms).
            sys.setswitchinterval(_SWITCH_INTERVAL)
            if thread.ident is None:
                thread.start()

        if follower:
            start()  # it makes rank 0's warm-up steps, which come before the rank's model loads

        closed = []

        def close() -> None:
            """Stop the replica: before the rank's model, which closes the card."""
            if closed:
                return
            closed.append(True)
            if thread.ident is not None:
                thread.join(options["drain_timeout"] + _SHUTDOWN_SECONDS)
            if gate is not None:
                counts = gate.counts
                logger.info(
                    f"{name}: {counts.idle_steps} steps in idle card time, "
                    f"{counts.shared_steps} in its share; {counts.tp_waits} steps of "
                    f"{options['model']} waited for it on the device"
                )
            stack.close()

        yield SimpleNamespace(
            gate=gate,
            start=start,
            close=close,
            side=side,
            runner=loaded.runner,
            attach=attach,
            start_engine=start_engine,
            scrub=scrub,
        )
        close()


def _share_card_cache(rank: int, options: dict, runner, replica):
    """One cache arena on each card of the rank group, which the first model and the replica of
    the card share through the server's slab table (``resin.SlabTable``): the first model holds
    the slabs below the table's boundary on every card, each sharing replica the slabs from the
    boundary up on its card (table side ``replica.side``; ``replica`` is None on a card with no
    sharing replica). Every rank takes part. Each model keeps a floor of one sequence at its
    context length. Returns (table, slabs per card), or None when a rank cannot go on (rank 0
    raises, after it reports)."""
    import torch.distributed as dist

    from resin import SlabTable
    from resin_models_gaudi.arena import ARENA_BLOCK_ROWS, ROW_BYTES, plan_shared_arena
    from resin_models_gaudi.synapse.runtime import current_device

    ranks = options["ranks"]
    device = current_device()
    rows, scratch, reserve = [runner.own_slab_rows], 0, runner.workspace_reserve
    units = [getattr(runner, "slab_unit", ARENA_BLOCK_ROWS)]
    if replica is not None:
        rows.append(replica.runner.own_slab_rows)
        units.append(getattr(replica.runner, "slab_unit", ARENA_BLOCK_ROWS))
        scratch = replica.runner.scratch_rows
        # the replica's runtime context keeps a workspace of its own
        reserve += replica.runner.model.workspace_reserve
    scratch_bytes = -(-scratch // ARENA_BLOCK_ROWS) * ARENA_BLOCK_ROWS * ROW_BYTES
    free = device.memory_info()[0] - reserve - scratch_bytes
    states: list = [None] * ranks
    dist.all_gather_object(states, (free, max(rows), math.lcm(*units)))
    # The same slab on every card (a slab of the first model is one column of the arenas): the largest of the
    # models', in whole units of each.
    unit = math.lcm(ARENA_BLOCK_ROWS, *(state[2] for state in states))
    slab_rows = -(-max(state[1] for state in states) // unit) * unit
    capacity = min(runner.shared_capacity(state[0], slab_rows) for state in states)
    capacity = capacity // _ARENA_QUANTUM * _ARENA_QUANTUM
    # Every rank reports what failed, so that all of them stop together: the arena, then (after the first
    # model's attach, which every rank enters) the replica's.
    arena, floor, local = None, None, None
    if capacity > 0:
        try:
            arena = plan_shared_arena(
                device, [slab_rows], [scratch], parent_capacity=capacity
            )
        except Exception as error:  # noqa: BLE001 (rank 0 reports it)
            local = f"rank {rank}: {error}"
    allocated: list = [None] * ranks
    dist.all_gather_object(allocated, local)
    if arena is not None and not any(allocated):
        runner.attach(capacity, slab_rows=slab_rows, arena=arena.address)
        if replica is not None:
            try:
                floor = (replica.side, replica.attach(arena))
            except Exception as error:  # noqa: BLE001 (rank 0 reports it)
                local = f"rank {rank}: {error}"
    reports: list = [None] * ranks
    dist.all_gather_object(reports, (floor, local))
    path = str(Path(options["progress"]).with_name("slabs"))
    problem, table = None, None
    if rank == 0:
        name = get_model(options["model"]).name
        problems = [report[1] for report in reports if report[1] is not None]
        sides = dict(report[0] for report in reports if report[0] is not None)
        replica_floors = [sides[side] for side in sorted(sides)]
        if capacity <= 0:
            tightest = min(state[0] for state in states)
            problem = (
                f"{name} and its replicas leave no memory for a shared KV cache: after both models' "
                f"weights and runtime workspaces, the tightest card has {tightest / 2**30:.1f} GiB free, "
                f"and a slab is {slab_rows * ROW_BYTES / 2**20:.0f} MiB. Give {name} more cards, or "
                "use a smaller replica model"
            )
        elif problems:
            problem = "the shared KV cache: " + "; ".join(problems)
        else:
            first_floor = runner.cache_layout.estimated_parent_count(
                [options["context_length"]]
            )
            try:
                table = SlabTable.create(
                    path, capacity, len(replica_floors), first_floor, replica_floors
                )
            except ValueError:
                problem = (
                    f"the KV cache of each card holds {capacity:,} slabs: one sequence of "
                    f"{options['context_length']:,} tokens of {name} needs {first_floor:,}, and "
                    f"one of a replica up to {max(replica_floors):,}. Give a smaller "
                    "--context-length"
                )
            except OSError as error:
                problem = f"the shared KV cache's slab table: {error}"
            else:
                share = capacity + 1 - table.boundary
                logger.info(
                    f"KV cache: {capacity:,} slabs of {slab_rows * ROW_BYTES / 2**20:.0f} MiB on "
                    f"each card, shared: {name} starts with {table.boundary - 1:,} on every "
                    f"card and each of {len(replica_floors)} replicas with {share:,} on its card; "
                    f"each keeps at least one sequence ({first_floor:,} and up to "
                    f"{max(replica_floors):,} slabs)"
                )
    verdict = [problem]
    dist.broadcast_object_list(verdict, src=0)
    if verdict[0] is not None:
        if rank == 0:
            logger.error(verdict[0])
            raise SystemExit(EXIT_REPORTED)
        return None
    return (table if table is not None else SlabTable.open(path)), capacity


def _serve_claimed_rank(rank: int, options: dict, claim, replica=None) -> None:
    import torch.distributed as dist

    from resin import Engine, Parallelism, SchedulerConfig
    from resin.backend.distributed import DistributedEngine, DistributedWorker
    from resin.backend.step_transport import create_step_transport
    from resin.backend.topology import ParallelGroups
    from resin.server import HttpServer
    from resin_models_gaudi.synapse.runtime import current_device

    spec = get_model(options["model"])
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    ranks = options["ranks"]
    routers = []
    if rank == 0:
        from resin._resin import _ReplicaRouter

        for served in options.get("replicated", ()):
            router = _ReplicaRouter(
                served["path"], options["replica_token"], served["count"]
            )
            routers.append((served["name"], router))
    topology = Parallelism(world_size=ranks, attn_dp=1, attn_tp=ranks, ep=1)
    # loading takes minutes per rank: the control plane must outlast it
    timeout = timedelta(minutes=60)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{options['control_port']}",
        rank=rank,
        world_size=topology.world_size,
        timeout=timeout,
    )
    # Every rank joins, with or without a card, so that all of them learn the outcome together.
    state = None
    if claim is not None:
        state = (claim.card.module, current_device().memory_info()[0])
    states: list[tuple[int, int] | None] = [None] * ranks
    dist.all_gather_object(states, state)
    problem = _startup_problem(options, states)
    if problem is not None:
        if rank != 0:
            return  # rank 0 reports, and the parent reads only its exit code
        logger.error(problem)
        raise SystemExit(EXIT_REPORTED)
    if rank == 0:
        modules = ",".join(str(state[0]) for state in states if state is not None)
        logger.info(f"cards: modules {modules}")
        if options["num_layers"] is None:
            estimate_kv_pool(
                min(state[1] for state in states if state is not None),
                options["weight_bytes"],
                ranks,
            )
    groups = ParallelGroups.create(topology, device_backend="gloo", timeout=timeout)
    checkpoint = options["checkpoint"]
    _remember_load(
        spec,
        checkpoint,
        ranks=ranks,
        num_layers=options["num_layers"],
        draft_tokens=options["draft_tokens"],
        assistant=options["assistant"],
    )
    shared_kv = options.get("shared_kv", False)
    native_ranks = None
    if options["step_transport"] == "native":
        from resin._resin import GaudiRanks

        # the native executors exchange each step over this group (no Gloo after the start)
        native_ranks = GaudiRanks(
            options["native_path"],
            options["native_token"],
            rank,
            ranks,
            timeout=timeout.total_seconds(),
        )
    runner = create_runner(
        spec,
        Path(checkpoint),
        device=f"hpu:{claim.card.accel}",
        groups=None if native_ranks is not None else groups,
        native_ranks=native_ranks,
        context_length=options["context_length"],
        max_batch_size=options["max_batch_size"],
        prefill_chunk_size=options["prefill_chunk_size"],
        draft_tokens=options["draft_tokens"],
        assistant_checkpoint=options["assistant"],
        num_layers=options["num_layers"],
        defer_cache=shared_kv,
        cache_types=tuple(options["cache_types"]),
    )
    engine = None
    try:
        pool = None
        if shared_kv:
            from resin import CachePool

            # a replica with a fixed share of its card (--gpu-memory-utilization) made its cache when it loaded
            sharing = replica if replica is not None and replica.side is not None else None
            shared = _share_card_cache(rank, options, runner, sharing)
            if shared is None:
                return  # rank 0 reports
            table, capacity = shared
            if sharing is not None:
                sharing.start_engine(
                    CachePool(
                        capacity,
                        boundary=table.side(sharing.side),
                        scrub=sharing.scrub,
                    )
                )
            if rank == 0:
                pool = CachePool(capacity, boundary=table.side(0))
        if native_ranks is not None:
            # rank 0's Engine runs the native executor, which sends each runner call to the other ranks; with a
            # replica on this rank's card, the executor takes the card's gate around each runner call
            if replica is not None and replica.gate is not None:
                runner.executor.attach_gate(replica.gate)
            worker = runner
            make_engine = Engine
        else:
            # the worker takes the runner's config, which a deferred cache completes
            transport = create_step_transport(
                groups, options["step_transport"], timeout=timeout.total_seconds()
            )
            if replica is not None and replica.gate is not None:
                from resin_models_gaudi.synapse.card_gate import GatedTransport

                transport = GatedTransport(transport, replica.gate)
            worker = DistributedWorker(runner, groups, transport=transport)
            make_engine = DistributedEngine
        if not groups.is_driver:
            from resin_models_gaudi.loading.progress import progress

            progress.phase("ready")  # rank 0 runs the warm-up
            signal.signal(signal.SIGTERM, _ignore_signal)
            if replica is not None:
                replica.start()
            if native_ranks is not None:
                calls = runner.serve()
                logger.info(f"rank {rank}: served {calls} runner calls")
            else:
                worker.serve()
            return
        recipe = create_recipe(
            spec,
            Path(checkpoint),
            runner=runner,
            tokenizer_checkpoint=options["tokenizer"],
        )
        engine = make_engine(
            worker,
            max_running_requests=options["max_batch_size"],
            context_length=options["context_length"],
            prefix_cache=options["prefix_cache"],
            enable_overlap=options["overlap"]
            and callable(getattr(runner, "enqueue", None)),
            scheduler_config=SchedulerConfig(
                max_batch_tokens=batch_tokens(runner, options["prefill_chunk_size"]),
                prefill_chunk_size=options["prefill_chunk_size"],
                decode_share=options["decode_share"],
            ),
            cache_pool=pool,
        )
        report_kv_pool(
            runner,
            engine,
            ranks,
            options["context_length"],
            options["max_batch_size"],
        )
        from resin_models_gaudi.loading.progress import progress

        progress.phase("warm-up")
        logger.info("warming up")
        warm_engine(engine, recipe, spec, rank)
        if replica is not None:
            replica.start()  # it joins the router of this process
        for name, router in routers:
            logger.info(f"waiting for {router.count} replicas of {name}")
            router.accept(_REPLICA_TIMEOUT)
            logger.info(f"{name} on {router.count} cards joined")
        server = HttpServer(
            engine,
            recipe=recipe,
            model=options["alias"] or spec.repo_id,
            stream_interval=options["stream_interval"],
            request_queue_capacity=options["request_queue_capacity"],
            drain_seconds=options["drain_timeout"],
            enable_anthropic=spec.enable_anthropic,
            capabilities=model_capabilities(spec, recipe, checkpoint),
        )
        for name, router in routers:
            server.register_replicas(router, model=name)
        address = f"http://{options['host']}:{options['port']}"
        _serve_http(
            server,
            options["host"],
            options["port"],
            f"serving {spec.name} at {address}",
            f"{spec.name} on {ranks} cards is ready at {address}",
        )
    finally:
        if engine is not None and not engine.closed:
            engine.close()
        compiles = getattr(runner, "compiles_after_start", None)
        if compiles is not None:
            logger.info(f"graph compilations after startup: {compiles}")
        if replica is not None:
            replica.close()
        runner.close()
        groups.close()


if __name__ == "__main__":
    main()
