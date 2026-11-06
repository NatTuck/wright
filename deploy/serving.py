"""The server: load one registry model on this host and serve it through Resin's HTTP APIs.

The `wright` command and scripts/serve_local.py both start here. Multi-card presets spawn one
process per card, and each process takes a free card from --gpu.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
from datetime import timedelta
from pathlib import Path

from deploy.gaudi_cli import (
    add_gaudi_arguments,
    apply_gaudi_arguments,
    check_listen_address,
    decode_share,
    describe_settings,
    nonnegative_int,
    positive_int,
)
from deploy.logs import setup_logging
from deploy.models import (
    MODEL_SPECS,
    check_gaudi_drafts,
    check_gaudi_ranks,
    checkpoint_weight_bytes,
    create_recipe,
    create_runner,
    gaudi_supports_ranks,
    get_model,
    require_checkpoint,
    validate_gaudi_options,
    warm_engine,
)

logger = logging.getLogger("wright")

# A rank exits with this code after it reports a startup problem that needs no traceback.
EXIT_REPORTED = 78
# A rank exits with this code when a signal stops it during the load.
EXIT_STOPPED = 79
# Recipe workspaces and slack that the KV pool leaves free on a card.
_WORKSPACE_BYTES = 3 << 30
# Rank 0 and its followers need this long to shut down after the drain.
_SHUTDOWN_SECONDS = 60.0


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
            draft_tokens=spec.draft_tokens
            if args.draft_tokens is None
            else args.draft_tokens,
        )
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
    if gaudi_supports_ranks(spec):
        return serve_gaudi_ranks(parser, args, spec)
    for flag, value in (
        ("--ranks", args.ranks),
        ("--num-layers", args.num_layers),
        ("--tokenizer", args.tokenizer),
        ("--assistant-checkpoint", args.assistant_checkpoint),
    ):
        if value is not None:
            parser.error(f"{spec.name} runs on one card and does not use {flag}")
    from resin_models_gaudi.synapse.runtime import claim_card

    logger.info(f"settings: {describe_settings()}")
    signal.signal(signal.SIGTERM, _stop_while_loading)
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


def _serve_one(args, spec, device: str) -> None:
    context_length = args.context_length or spec.context_length
    max_batch_size = args.max_batch_size or spec.max_batch_size
    prefill_chunk_size = args.prefill_chunk_size or spec.prefill_chunk_size
    draft_tokens = spec.draft_tokens if args.draft_tokens is None else args.draft_tokens
    try:
        checkpoint = require_checkpoint(
            spec, checkpoint=args.checkpoint, draft_tokens=draft_tokens
        )
        check_gaudi_drafts(spec, checkpoint, draft_tokens)
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        logger.error(f"{spec.name}: {error}")
        raise SystemExit(1) from None
    estimate_kv_pool(
        current_device_free(), checkpoint_weight_bytes(spec, checkpoint), 1
    )

    from resin import Engine, SchedulerConfig
    from resin.server import HttpServer

    runner = create_runner(
        spec,
        checkpoint,
        device=device,
        context_length=context_length,
        max_batch_size=max_batch_size,
        prefill_chunk_size=prefill_chunk_size,
        draft_tokens=draft_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    engine = None
    try:
        recipe = create_recipe(spec, checkpoint, runner=runner)
        engine = Engine(
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
        )
        report_kv_pool(runner, engine, 1, context_length, max_batch_size)
        logger.info("warming up")
        warm_engine(engine, recipe, spec)
        logger.info(f"{spec.name} is ready at http://{args.host}:{args.port}")
        server = HttpServer(
            engine,
            recipe=recipe,
            model=args.alias or spec.repo_id,
            stream_interval=args.stream_interval,
            request_queue_capacity=args.request_queue_capacity,
            drain_seconds=args.drain_timeout,
            enable_anthropic=spec.enable_anthropic,
        )
        signal.signal(signal.SIGTERM, _ignore_signal)
        try:
            server.serve(host=args.host, port=args.port)
        except KeyboardInterrupt:
            pass  # Ctrl-C reaches Python as well as the HTTP server
    finally:
        if engine is not None and not engine.closed:
            engine.close()
        compiles = getattr(runner, "compiles_after_start", None)
        if compiles is not None:
            logger.info(f"graph compilations after startup: {compiles}")
        close = getattr(runner, "close", None)
        if close is not None:
            close()


def serve_gaudi_ranks(parser, args, spec) -> None:
    """One process per card: every rank takes a free card and binds to its NUMA node, builds the
    runner, rank 0 serves the Engine and HTTP, the others follow over the Gloo control plane."""
    from resin.process import ProcessError, spawn
    from resin_models_gaudi.synapse.cards import visible_cards

    ranks = args.ranks or spec.world_size
    draft_tokens = spec.draft_tokens if args.draft_tokens is None else args.draft_tokens
    if args.gpu_memory_utilization is not None:
        parser.error(f"{spec.name} sizes its KV cache from free memory on every rank")
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
        overlap=args.overlap,
        decode_share=args.decode_share,
        prefix_cache=args.prefix_cache,
        stream_interval=args.stream_interval,
        request_queue_capacity=args.request_queue_capacity,
        drain_timeout=args.drain_timeout,
        parent=os.getpid(),
        control_port=port,
    )
    # A rank binds itself once it holds a card, so spawn does not place it.
    context = spawn(
        _serve_gaudi_rank,
        args=(options,),
        devices=["hpu"] * ranks,
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


def _serve_gaudi_rank(rank: int, options: dict) -> None:
    from resin.process import die_with_parent
    from resin_models_gaudi.synapse.runtime import claim_card

    setup_logging(rank)
    die_with_parent(options["parent"])
    # The parent asks rank 0 to stop, and rank 0 stops the others in order.
    signal.signal(signal.SIGTERM, _stop_while_loading)
    if rank != 0:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    with claim_card(rank, numa=options["numa"]) as claim:
        try:
            _serve_claimed_rank(rank, options, claim)
        except KeyboardInterrupt:
            raise SystemExit(EXIT_STOPPED) from None


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


def _serve_claimed_rank(rank: int, options: dict, claim) -> None:
    import torch.distributed as dist

    from resin import Parallelism, SchedulerConfig
    from resin.backend.distributed import DistributedEngine, DistributedWorker
    from resin.backend.topology import ParallelGroups
    from resin.server import HttpServer
    from resin_models_gaudi.synapse.runtime import current_device

    spec = get_model(options["model"])
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    ranks = options["ranks"]
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
    runner = create_runner(
        spec,
        Path(checkpoint),
        device=f"hpu:{claim.card.accel}",
        groups=groups,
        context_length=options["context_length"],
        max_batch_size=options["max_batch_size"],
        prefill_chunk_size=options["prefill_chunk_size"],
        draft_tokens=options["draft_tokens"],
        assistant_checkpoint=options["assistant"],
        num_layers=options["num_layers"],
    )
    worker = DistributedWorker(runner, groups)
    engine = None
    try:
        if not groups.is_driver:
            signal.signal(signal.SIGTERM, _ignore_signal)
            worker.serve()
            return
        recipe = create_recipe(
            spec,
            Path(checkpoint),
            runner=runner,
            tokenizer_checkpoint=options["tokenizer"],
        )
        engine = DistributedEngine(
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
        )
        report_kv_pool(
            runner,
            engine,
            ranks,
            options["context_length"],
            options["max_batch_size"],
        )
        logger.info("warming up")
        warm_engine(engine, recipe, spec, rank)
        logger.info(
            f"{spec.name} on {ranks} cards is ready at "
            f"http://{options['host']}:{options['port']}"
        )
        server = HttpServer(
            engine,
            recipe=recipe,
            model=options["alias"] or spec.repo_id,
            stream_interval=options["stream_interval"],
            request_queue_capacity=options["request_queue_capacity"],
            drain_seconds=options["drain_timeout"],
            enable_anthropic=spec.enable_anthropic,
        )
        signal.signal(signal.SIGTERM, _ignore_signal)
        try:
            server.serve(host=options["host"], port=options["port"])
        except KeyboardInterrupt:
            pass  # Ctrl-C reaches Python as well as the HTTP server
    finally:
        if engine is not None and not engine.closed:
            engine.close()
        compiles = getattr(runner, "compiles_after_start", None)
        if compiles is not None:
            logger.info(f"graph compilations after startup: {compiles}")
        runner.close()
        groups.close()


if __name__ == "__main__":
    main()
