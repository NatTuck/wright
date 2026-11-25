"""Run resident prefill/decode benchmarks on this host's Gaudi cards.

uv run --no-default-groups --group gaudi scripts/bench.py --model qwen3_5-gaudi2 \
    --checkpoint /path/to/Qwen3.8-27B-FP8 --output bench/qwen.jsonl
The measurement API lives in resin.bench; model definitions are shared with serving.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

# Direct script execution must also make the repository packages importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist

import resin
from deploy.gaudi_cli import add_gaudi_arguments, apply_gaudi_arguments
from deploy.models import (
    MODEL_SPECS,
    check_cache_types,
    create_runner,
    default_draft_tokens,
    get_model,
    require_checkpoint,
)
from resin import Engine, Parallelism, SchedulerConfig
from resin.backend.distributed import DistributedEngine, DistributedWorker
from resin.backend.topology import ParallelGroups
from resin.bench import decode_output_budget

ROOT = Path(__file__).resolve().parents[1]


def _synapse_trace(*args, **kwargs):
    from resin_models_gaudi.synapse.trace import perfetto_trace as synapse_trace

    return synapse_trace(*args, **kwargs)


def _runner(
    model, checkpoint, *, device, groups, cases, options, context_capacity, native_ranks=None
):
    return create_runner(
        get_model(model),
        checkpoint,
        device=device,
        groups=groups,
        native_ranks=native_ranks,
        context_length=context_capacity,
        max_batch_size=max(case.batch_size for case in cases),
        prefill_chunk_size=options["prefill_chunk_size"],
        draft_tokens=options["draft_tokens"],
        gpu_memory_utilization=options["gpu_memory_utilization"],
        assistant_checkpoint=options.get("assistant_checkpoint"),
        cache_types=options.get("cache_types"),
    )


def _result_callback(report_path):
    """Keep a valid snapshot of completed cases and print untimed progress."""
    records = []
    path = Path(report_path)

    def save(record):
        records.append(record)
        pending = path.with_suffix(path.suffix + ".tmp")
        pending.write_text(json.dumps(records))
        pending.replace(path)
        print("[bench-result] " + json.dumps(record), flush=True)

    return save


def _decode_budget(options, raw=None):
    overlap = options.get("overlap", True) and (
        raw is None or callable(getattr(raw, "enqueue", None))
    )
    config = getattr(raw, "runner_config", None)
    return decode_output_budget(
        warmup_steps=options["warmup_steps"],
        measure_steps=options["measure_steps"],
        draft_tokens=options["draft_tokens"],
        overlap=overlap,
        trace_passes=2 + int(overlap) if options.get("trace_dir") else 0,
        cache_write_lookahead_tokens=getattr(
            config, "cache_write_lookahead_tokens", None
        ),
    )


def _context_capacity(cases, options):
    # Preserve existing untraced graph geometry. Traced cohorts additionally
    # reserve the continuation guard through their last requested completion.
    headroom = (options["warmup_steps"] + options["measure_steps"] + 4) * (
        options["draft_tokens"] + 1
    )
    if options.get("prompt_files"):
        headroom = max(
            headroom, options["text_output_tokens"] + options["draft_tokens"]
        )
    if options.get("trace_dir") and any(case.phase == "decode" for case in cases):
        headroom = max(headroom, _decode_budget(options))
    return max(case.context_length for case in cases) + headroom


def _cache_preflight(raw, cases, options, groups=None):
    """Reject a partial decode cohort before spending time on capture or prefill."""
    plan = getattr(raw, "cache_capacity_plan", None)
    layout = getattr(raw, "cache_layout", None)
    if layout is None:
        return cases
    requirements = []
    for case in cases:
        context = case.context_length
        if case.phase == "decode":
            if options.get("trace_dir"):
                # Include the same output reservation as admission, plus actual
                # cache writes beyond the final public token position.
                config = getattr(raw, "runner_config", None)
                lookahead = getattr(
                    config, "cache_write_lookahead_tokens", options["draft_tokens"]
                )
                context += _decode_budget(options, raw) + lookahead
            elif options.get("prompt_files"):
                context += options["text_output_tokens"] + options["draft_tokens"]
            else:
                context += (options["warmup_steps"] + options["measure_steps"] + 1) * (
                    options["draft_tokens"] + 1
                ) + options["draft_tokens"]
        parents = layout.estimated_parent_count(
            [context] * case.batch_size, shared_prefix_tokens=case.shared_prefix_tokens
        )
        requirements.append(
            dict(
                **asdict(case),
                peak_context=context,
                required_parents=parents,
            )
        )
    rank = 0 if groups is None else groups.rank
    print(
        "[bench-cache] "
        + json.dumps(
            dict(
                rank=rank,
                capacity=None if plan is None else asdict(plan),
                parent_capacity=layout.parent_capacity,
                cases=requirements,
            )
        ),
        flush=True,
    )
    failures = [
        f"rank {rank} {row['phase']} ctx={row['context_length']} "
        f"batch={row['batch_size']} needs {row['required_parents']} cache parents, "
        f"capacity={layout.parent_capacity} (peak context {row['peak_context']})"
        for row in requirements
        if row["required_parents"] > layout.parent_capacity
    ]
    errors = [failures]
    if groups is not None and groups.domain.size > 1:
        errors = [None] * groups.domain.size
        groups.distributed.all_gather_object(
            errors, failures, group=groups.domain.handle
        )
    failures = [failure for values in errors if values for failure in values]
    if failures and options.get("skip_unfit"):
        print("[bench-cache] skipped: " + "; ".join(failures), flush=True)
        unfit = {
            (row["phase"], row["context_length"], row["batch_size"])
            for row in requirements
            if row["required_parents"] > layout.parent_capacity
        }
        return [
            case
            for case in cases
            if (case.phase, case.context_length, case.batch_size) not in unfit
        ]
    if failures:
        raise ValueError("benchmark cache preflight failed: " + "; ".join(failures))
    return cases


def _text_workloads(raw, spec, checkpoint, options) -> list:
    """The real-text workloads: every prompt file with every sampling mode and seed."""
    from deploy.models import create_recipe
    from resin.bench import TextWorkload
    from resin.recipe import Message, RenderOptions, Role, Text

    if not options.get("prompt_files"):
        return []
    recipe = create_recipe(
        spec,
        checkpoint,
        runner=raw,
        tokenizer_checkpoint=options.get("tokenizer_checkpoint"),
    )
    render_options = RenderOptions(**options["render_options"])

    def render(instruction: str, passage: str) -> list[int]:
        message = Message(Role.USER, [Text(f"{instruction}\n\n{passage}")])
        return list(recipe.render([message], options=render_options).token_ids)

    workloads = []
    for name, path in options["prompt_files"].items():
        rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
        fitted: dict[tuple[int, int], list[int]] = {}

        def prompts(index: int, context: int, rows=rows, fitted=fitted) -> list[int]:
            """A passage cut from its start so that the rendered prompt fills the context. A
            passage that is too short gives way to the next one that is long enough."""
            index %= len(rows)
            if (index, context) not in fitted:
                instruction = rows[index]["instruction"]
                keep = context - len(render(instruction, ""))
                for offset in range(len(rows)):
                    ids = recipe.tokenizer.encode(
                        rows[(index + offset) % len(rows)]["text"]
                    )
                    if keep <= len(ids):
                        break
                else:
                    raise ValueError(
                        f"no passage has the {keep} tokens of context {context}"
                    )
                while True:
                    tokens = render(instruction, recipe.tokenizer.decode(ids[:keep]))
                    if len(tokens) <= context:
                        break
                    keep -= len(tokens) - context
                fitted[index, context] = tokens
            return fitted[index, context]

        common: dict[str, Any] = dict(
            prompts=prompts,
            stop_token_ids=tuple(recipe.stop_token_ids),
            output_tokens=options["text_output_tokens"],
            steps=options["text_steps"],
        )
        for mode in options["text_modes"]:
            if mode == "greedy":
                workloads.append(TextWorkload(name=f"{name}/greedy", **common))
                continue
            for seed in options["text_seeds"]:
                workloads.append(
                    TextWorkload(
                        name=f"{name}/sampled",
                        temperature=options["temperature"],
                        top_k=options["top_k"],
                        top_p=options["top_p"],
                        seed=seed,
                        **common,
                    )
                )
    return workloads


# a grid prompt is the grid's context and then the template's closing tokens; the capacity
# and the cache checks reserve this many for them
GRID_TAIL_TOKENS = 256
# the families whose runner switches its drafts at run time (``drafting``)
SWITCHED_DRAFT_FAMILIES = ("deepseek_v41",)


def _grid_texts(raw, spec, checkpoint, options, grid) -> list:
    """The grid's text workloads: every prompt file with every sampling mode and seed, and the
    prompt stream of each row (``resin.bench.GridText``). Row r takes passage r mod n of its
    file (n passages), from a token offset that differs for each r div n."""
    import numpy as np

    from deploy.models import create_recipe
    from resin.bench import GridText, grid_rows
    from resin.recipe import Message, RenderOptions, Role, Text

    recipe = create_recipe(
        spec,
        checkpoint,
        runner=raw,
        tokenizer_checkpoint=options.get("tokenizer_checkpoint"),
    )
    render_options = RenderOptions(**options["render_options"])

    def render(instruction: str, passage: str) -> list[int]:
        message = Message(Role.USER, [Text(f"{instruction}\n\n{passage}")])
        return list(recipe.render([message], options=render_options).token_ids)

    longest = max(grid.cells())
    rows_needed = grid_rows(grid)
    texts = []
    for name, path in options["prompt_files"].items():
        rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
        first, second = (render(rows[0]["instruction"], p) for p in ("x", "y"))
        size = 0
        while size < min(len(first), len(second)) and first[-1 - size] == second[-1 - size]:
            size += 1
        tail = tuple(first[len(first) - size :])
        if not tail or len(tail) > GRID_TAIL_TOKENS:
            raise ValueError(f"{name}: the template's closing tokens are {len(tail)} tokens")
        slots = -(-rows_needed // len(rows))  # offsets per passage
        encoded: dict[int, list[int]] = {}
        streams: dict[int, np.ndarray] = {}

        def stream(row: int, length: int, rows=rows, encoded=encoded, streams=streams, tail=tail):
            if row not in streams:
                index, slot = row % len(rows), row // len(rows)
                if index not in encoded:
                    # a passage too short for every slot takes the next passages after it
                    text, ids, extra = rows[index]["text"], [], 0
                    while True:
                        ids = recipe.tokenizer.encode(text)
                        if len(ids) >= longest + 256 * slots or extra == len(rows) - 1:
                            break
                        extra += 1
                        text += "\n\n" + rows[(index + extra) % len(rows)]["text"]
                    encoded[index] = ids
                ids = encoded[index]
                step = max(1, (len(ids) - longest) // slots)
                start = slot * step
                piece = ids[start : start + longest]
                prompt = render(rows[index]["instruction"], recipe.tokenizer.decode(piece))
                if tuple(prompt[len(prompt) - len(tail) :]) != tail:
                    raise ValueError(f"{name} row {row}: the prompt does not end with the closing tokens")
                body = prompt[: len(prompt) - len(tail)]
                if len(body) < longest:
                    raise ValueError(
                        f"{name} row {row}: passage {index} from token {start} gives "
                        f"{len(body)} of the {longest} prompt tokens"
                    )
                streams[row] = np.asarray(body, dtype=np.uint32)
            return streams[row][:length]

        from resin.bench import TextWorkload

        def workload(mode: str, seed: int | None) -> TextWorkload:
            sampled = mode != "greedy"
            return TextWorkload(
                name=f"{name}/{mode}",
                prompts=lambda index, context: [],
                stop_token_ids=tuple(recipe.stop_token_ids),
                output_tokens=options["text_output_tokens"],
                steps=options["text_steps"],
                temperature=options["temperature"] if sampled else 0.0,
                top_k=options["top_k"] if sampled else None,
                top_p=options["top_p"] if sampled else 1.0,
                seed=seed,
            )

        for mode in options["text_modes"]:
            seeds = [None] if mode == "greedy" else options["text_seeds"]
            for seed in seeds:
                texts.append(GridText(workload(mode, seed), stream, tail))
    return texts


def draft_name(spec, draft_tokens: int, assistant) -> str:
    """A draft configuration's name in the results: none, mtpK, dsparkK, dflashK or dspineK."""
    if not draft_tokens:
        return "none"
    if spec.family in ("deepseek_v41", "mistral_large4"):
        return f"dspark{draft_tokens}"
    if assistant is None:
        return f"mtp{draft_tokens}"
    config = json.loads((Path(assistant) / "config.json").read_text())
    kind = (
        "dspine"
        if (config.get("dflash_config") or {}).get("selected_prefix_code_bridge")
        else "dflash"
    )
    return f"{kind}{draft_tokens}"


def _rank(
    rank,
    port,
    report_path,
    model,
    checkpoint,
    case_values,
    topology,
    options,
):
    """A rank's process. It takes a free visible card first, then binds to its node."""
    run = (rank, port, report_path, model, checkpoint, case_values, topology, options)
    from resin_models_gaudi.synapse.runtime import claim_card

    if options.get("trace_dir"):
        # BEFORE claim_card: it opens the device, and Synapse loads its profiler plugin only when
        # HABANA_PROFILE is set at the first Synapse call. Enabled later, the plugin is never loaded
        # and synProfilerStart returns status=18 (synUnsupported) from libSynapse's stub.
        from resin_models_gaudi.synapse.trace import enable_profiler

        enable_profiler(Path(tempfile.mkdtemp(prefix="resin-synapse-profiler-")))

    with claim_card(rank, numa=options["numa"]) as claim:
        if claim is None:
            raise RuntimeError(f"rank {rank} found no free Gaudi card")
        Path(report_path).with_name(f"binding-{rank}.json").write_text(
            json.dumps(None if claim.binding is None else asdict(claim.binding))
        )
        return _run_rank(*run, device=f"hpu:{claim.card.accel}")


def _run_rank(
    rank,
    port,
    report_path,
    model,
    checkpoint,
    case_values,
    topology,
    options,
    device,
):
    from resin.bench import (
        BenchCase,
        BenchmarkRunner,
        distributed_reducer,
        run_sweep,
    )

    spec = get_model(model)
    groups = native = None
    if topology.world_size > 1 and options.get("step_transport") == "native":
        from resin._resin import GaudiRanks

        # the native rank group: HCCL, the capacity and barriers at the start, then each runner call; no Gloo
        native = GaudiRanks(
            options["native_path"], options["native_token"], rank, topology.world_size, timeout=3600.0
        )
    elif topology.world_size > 1:
        # A Gloo control plane; the runner owns HCCL. Loading takes minutes per rank.
        dist.init_process_group(
            "gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=topology.world_size,
            timeout=timedelta(minutes=60),
        )
        groups = ParallelGroups.create(
            topology, device_backend="gloo", timeout=timedelta(minutes=60)
        )
    cases = [BenchCase(**case) for case in case_values]
    prefill_budget = (
        max(case.batch_size for case in cases) * options["prefill_chunk_size"]
    )
    # Reserve real generated positions, including draft write lookahead. Context
    # lengths on the charts remain the prompt lengths recorded by the harness.
    capacity = _context_capacity(cases, options)
    if rank == 0:
        print(f"[bench] loading {model} over {topology.world_size} cards", flush=True)
    started = time.perf_counter()
    raw = _runner(
        model,
        checkpoint,
        device=device,
        groups=groups,
        cases=cases,
        options=options,
        context_capacity=capacity,
        native_ranks=native,
    )
    # A batch is a cohort per attention replica; never exceed the runner's
    # supported flattened row count when prefill spans several requests.
    prefill_budget = min(prefill_budget, raw.runner_config.max_batch_tokens)
    cases = _cache_preflight(raw, cases, options, groups)
    observed = BenchmarkRunner(raw, accepted_drafts=options["accepted_drafts"])
    # The runners compile and load every Synapse recipe during construction.
    synchronize = raw.device.sync
    synchronize()
    loaded_seconds = time.perf_counter() - started
    if rank == 0:
        print(f"[bench] runner ready in {loaded_seconds:.1f}s", flush=True)
    overlap = options.get("overlap", True) and callable(getattr(raw, "enqueue", None))
    metadata = {
        **options["metadata"],
        "model": model,
        "checkpoint": str(checkpoint),
        "hardware": f"{spec.gpu} (Gaudi2, Synapse {raw.device.driver_version})",
        "parallelism": {
            axis: getattr(topology, axis)
            for axis in ("world_size", "attn_dp", "attn_tp", "cp", "ep")
        },
        "graphs": {"prefill": "synapse_recipes", "decode": "synapse_recipes"},
        "quant": "checkpoint",
        "expert_backend": "none",
        "load_seconds": loaded_seconds,
        "overlap": overlap,
        "timing": "continuous_completions" if overlap else "synchronized_steps",
        "prefill_chunk_size": options["prefill_chunk_size"],
        "context_capacity": capacity,
        "max_batch_tokens": prefill_budget,
    }
    if options.get("grid"):
        metadata["grid"] = {
            key: options["grid"][key]
            for key in ("row_contexts", "row_streams", "column_contexts", "column_streams", "reuse")
        }
    engine_options: dict[str, Any] = dict(
        max_running_requests=max(case.batch_size for case in cases),
        context_length=capacity,
        # the grid keeps its prompts in the prefix cache
        prefix_cache=bool(options.get("prefix_cache"))
        or bool(options.get("grid"))
        or any(case.shared_prefix_tokens for case in cases),
        enable_overlap=options.get("overlap", True),
        scheduler_config=SchedulerConfig(
            max_batch_tokens=prefill_budget,
            prefill_chunk_size=options["prefill_chunk_size"],
        ),
    )
    if native is not None and rank > 0:
        raw.serve()  # rank 0's native executor sends every runner call
        os._exit(0)
    worker = None if groups is None else DistributedWorker(observed, groups)
    if groups is not None and worker is not None and not groups.is_driver:
        worker.serve()
        os._exit(0)
    engine = (
        Engine(observed, **engine_options)
        if worker is None
        else DistributedEngine(worker, **engine_options)
    )
    try:
        if options.get("grid"):
            _run_grid(
                engine,
                raw,
                observed,
                spec,
                checkpoint,
                cases,
                options,
                metadata=metadata,
                report_path=report_path,
                synchronize=synchronize,
                reduce_step=None if groups is None else distributed_reducer(groups),
                gpu_count=topology.world_size,
                budget=prefill_budget,
            )
        else:
            # A seeded synthetic prompt avoids tokenization overhead and special
            # token stops. Every replica has distinct, reproducible input tokens.
            generator = torch.Generator().manual_seed(41 + rank)
            prompt = torch.randint(
                1000, min(30_000, raw.vocab_size), (capacity,), generator=generator
            ).tolist()
            run_sweep(
                engine,
                cases,
                runner=observed,
                prompt_tokens=prompt,
                workloads=_text_workloads(raw, spec, checkpoint, options),
                warmup_steps=options["warmup_steps"],
                measure_steps=options["measure_steps"],
                gpu_count=topology.world_size,
                attn_dp=topology.attn_dp,
                synchronize=synchronize,
                reduce_step=None if groups is None else distributed_reducer(groups),
                metadata=metadata,
                on_result=_result_callback(report_path) if rank == 0 else None,
                trace_step=(
                    None
                    if options.get("trace_dir") is None
                    else _synapse_trace(
                        options["trace_dir"],
                        rank,
                        metadata,
                        device=raw.device,
                        decode_passes=2 + int(overlap),
                    )
                ),
            )
    finally:
        # Closing a poisoned domain still tells followers to stop. Preserve the
        # original failure/report if close repeats that same domain error.
        failed = engine.poisoned or sys.exc_info()[0] is not None
        try:
            engine.close()
        except Exception:
            if not failed:
                raise
    # Every rank exits together, so the runners release their cards together.
    if native is not None:
        raw.close()  # the followers stop serving
    if groups is not None or native is not None:
        os._exit(0)


def _run_grid(
    engine,
    raw,
    observed,
    spec,
    checkpoint,
    cases,
    options,
    *,
    metadata,
    report_path,
    synchronize,
    reduce_step,
    gpu_count,
    budget,
):
    """The driver's grid (``resin.bench.run_grid``): the cells that the cache checks kept, each
    draft configuration of ``options["grid"]["drafts"]``, the progress file of each row."""
    from resin.bench import Chunking, CrossGrid, DraftConfig, run_grid, set_drafting

    settings = options["grid"]
    grid = CrossGrid(
        **{
            key: tuple(settings[key])
            for key in ("row_contexts", "row_streams", "column_contexts", "column_streams")
        }
    )
    kept = {
        (case.context_length - GRID_TAIL_TOKENS, case.batch_size)
        for case in cases
        if case.phase == "decode"
    }
    rows = raw.runner_config.max_speculative_rows
    drafts = settings["drafts"]
    assistant = options.get("assistant_checkpoint")
    configs = [
        DraftConfig(
            draft_name(spec, k, assistant), k, rows=rows if len(drafts) > 1 and k else None
        )
        for k in drafts
    ]
    switch = None
    if len(configs) > 1:
        if not hasattr(raw, "drafting"):
            raise ValueError(f"{spec.name}'s runner cannot switch its drafts at run time")
        switch = lambda config: set_drafting(  # noqa: E731
            engine, raw, config.draft_tokens > 0, rows
        )
    run_grid(
        engine,
        grid,
        _grid_texts(raw, spec, checkpoint, options, grid),
        runner=observed,
        chunking=Chunking.of(raw, options["prefill_chunk_size"], budget),
        configs=configs,
        switch=switch,
        fits=lambda context, streams: (context, streams) in kept,
        reuse=settings["reuse"],
        phases=options.get("phases", ("prefill", "decode")),
        warmup_steps=options["warmup_steps"],
        gpu_count=gpu_count,
        synchronize=synchronize,
        reduce_step=reduce_step,
        metadata=metadata,
        on_result=_result_callback(report_path),
    )


def _validate_configuration(model, cases, topology, options):
    from resin.bench import BenchCase

    spec = get_model(model)
    draft_tokens = options["draft_tokens"]
    accepted_drafts = options["accepted_drafts"]
    prefill_chunk_size = options["prefill_chunk_size"]
    numa = options["numa"]
    for name, minimum in (
        ("warmup_steps", 0),
        ("measure_steps", 1),
        ("prefill_chunk_size", 1),
        ("draft_tokens", 0),
    ):
        value = options[name]
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if accepted_drafts is not None and type(accepted_drafts) is not int:
        raise ValueError("accepted_drafts must be an integer or None")
    if not 0 < options["gpu_memory_utilization"] < 1:
        raise ValueError("gpu_memory_utilization must be between zero and one")
    check_cache_types(spec, options.get("cache_types"))
    if topology.world_size != spec.world_size:
        raise ValueError(f"{model} runs on {spec.world_size} Gaudi cards")
    if (
        spec.family != "mistral_large4"
        and prefill_chunk_size != spec.prefill_chunk_size
    ):
        raise ValueError(f"{model} prefills in {spec.prefill_chunk_size}-token chunks")
    if numa not in ("auto", "off", "required"):
        raise ValueError("numa must be 'auto', 'off', or 'required'")
    if not cases:
        raise ValueError("the benchmark needs at least one case")
    for values in cases:
        BenchCase(**values)
    if (
        draft_tokens < 0
        or accepted_drafts is not None
        and not 0 <= accepted_drafts <= draft_tokens
    ):
        raise ValueError("accepted drafts must be between zero and draft tokens")


def run_benchmark(
    model: str,
    checkpoint: str | Path,
    cases: list[dict],
    *,
    parallelism: Parallelism | None = None,
    draft_tokens: int | None = None,
    accepted_drafts: int | None = None,
    warmup_steps: int = 5,
    measure_steps: int = 20,
    prefill_chunk_size: int = 4096,
    gpu_memory_utilization: float = 0.9,
    metadata: dict | None = None,
    overlap: bool = True,
    numa: Literal["auto", "off", "required"] = "auto",
    trace_dir: str | None = None,
    assistant_checkpoint: str | Path | None = None,
    progress_path: str | Path | None = None,
    text: dict | None = None,
    step_transport: Literal["gloo", "native"] = "gloo",
    cache_types: tuple[str | None, str | None] | None = None,
    grid: dict | None = None,
    prefix_cache: bool | None = None,
) -> list[dict]:
    """One load and graph catalog for all compatible points in this sweep.

    ``progress_path`` optionally retains a JSON array of completed cases even
    if a later case or worker fails. It must be a new file. Updates happen
    outside timing; successful results also include worker NUMA placement.
    ``cache_types``: the types of the K and V cache pages (None: the model's); the results' metadata names
    them.
    ``grid``: measure ``resin.bench.run_grid``'s cross grid in place of ``cases`` (these then only size the
    runner and check the cache): the ``CrossGrid`` lists, ``reuse``, and ``drafts``, the draft tokens of each
    configuration (0,K: off and on, one load).
    """
    spec = get_model(model)
    cache_types = check_cache_types(spec, cache_types)
    topology = parallelism or Parallelism()
    if draft_tokens is None:
        draft_tokens = default_draft_tokens(spec, assistant_checkpoint)
    options = dict(
        draft_tokens=draft_tokens,
        accepted_drafts=accepted_drafts,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
        prefill_chunk_size=prefill_chunk_size,
        gpu_memory_utilization=gpu_memory_utilization,
        metadata={**(metadata or {}), "cache_types": list(cache_types)},
        cache_types=cache_types,
        overlap=overlap,
        numa=numa,
        trace_dir=trace_dir,
        step_transport=step_transport,
        grid=grid,
        prefix_cache=prefix_cache,
        assistant_checkpoint=None
        if assistant_checkpoint is None
        else str(assistant_checkpoint),
        **(text or {}),
    )
    _validate_configuration(model, cases, topology, options)
    with tempfile.TemporaryDirectory(prefix="resin-bench-") as directory:
        output = (
            Path(directory) / "results.json"
            if progress_path is None
            else Path(progress_path).expanduser().resolve()
        )
        if progress_path is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x") as stream:
                stream.write("[]")
        port = 0
        # the native rank group's socket (rank 0 listens) and run token
        options["native_path"] = str(Path(directory) / "ranks")
        options["native_token"] = os.urandom(16)
        if topology.world_size > 1:
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
        resin.process.spawn(
            _rank,
            args=(
                port,
                output,
                model,
                checkpoint,
                cases,
                topology,
                {**options, "numa": numa},
            ),
            devices=["hpu"] * topology.world_size,
            # A rank binds itself once it holds a card.
            numa="off",
        )
        records = json.loads(output.read_text())
        placement = {
            "mode": numa,
            "bindings": [
                json.loads(output.with_name(f"binding-{rank}.json").read_text())
                for rank in range(topology.world_size)
            ],
        }
        for rank in range(topology.world_size):
            output.with_name(f"binding-{rank}.json").unlink(missing_ok=True)
        for record in records:
            record["numa"] = placement
        if progress_path is not None:
            pending = output.with_suffix(output.suffix + ".tmp")
            pending.write_text(json.dumps(records))
            pending.replace(output)
        return records


def write_results(path: str | Path, records: list[dict]) -> None:
    """Append raw points; plotting deliberately resolves repeated points later."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


def source_metadata() -> dict:
    """Identify the submitted source, including local edits and new source files."""

    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=ROOT)

    try:
        revision = git("rev-parse", "HEAD").decode().strip()
        dirty = bool(git("status", "--porcelain"))
        source = hashlib.sha256(revision.encode())
        source.update(git("diff", "--binary", "HEAD"))
        for name in sorted(
            git("ls-files", "--others", "--exclude-standard").decode().splitlines()
        ):
            path = ROOT / name
            if path.suffix in {".py", ".rs", ".toml", ".lock", ".cu", ".cuh"}:
                source.update(name.encode())
                source.update(path.read_bytes())
        return dict(git_commit=revision, dirty=dirty, source_digest=source.hexdigest())
    except (OSError, subprocess.CalledProcessError):
        return dict(git_commit=None, dirty=None, source_digest=None)


def checkpoint_revision(spec, checkpoint: Path) -> str:
    """The pinned revision, or the commit Hugging Face recorded in a local download."""
    if Path(checkpoint) == spec.path:
        return spec.revision
    recorded = Path(checkpoint) / ".cache/huggingface/download/config.json.metadata"
    try:
        commit = recorded.read_text().split()[0]
    except (OSError, IndexError):
        return f"unpinned:{checkpoint}"
    return commit


def _integers(value: str) -> list[int]:
    try:
        values = [int(part) for part in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "expected comma-separated positive integers"
        ) from None
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return list(dict.fromkeys(values))


def _integers_or_none(value: str) -> list[int]:
    return [] if value.strip().lower() == "none" else _integers(value)


def _draft_list(value: str) -> list[int]:
    try:
        values = [int(part) for part in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "expected comma-separated integers >= 0"
        ) from None
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("draft tokens must be integers >= 0")
    return list(dict.fromkeys(values))


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_SPECS, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="local checkpoint directory (required)",
    )
    parser.add_argument(
        "--assistant-checkpoint",
        type=Path,
        help="local assistant/draft checkpoint when required",
    )
    parser.add_argument(
        "--tokenizer-checkpoint",
        type=Path,
        help="local tokenizer directory, for presets whose weights ship without one",
    )
    add_gaudi_arguments(parser)
    for axis in ("world-size", "attn-dp", "attn-tp", "cp", "ep"):
        parser.add_argument(f"--{axis}", type=int)
    parser.add_argument(
        "--phases", default="prefill,decode", help="prefill, decode, or prefill,decode"
    )
    parser.add_argument(
        "--context-lengths",
        type=_integers,
        help="prompt lengths for both phases; phase-specific options take precedence",
    )
    parser.add_argument(
        "--prefill-context-lengths",
        type=_integers,
        help="prefill prompt lengths (default: 4096,8192,16384,32768,65536)",
    )
    parser.add_argument(
        "--decode-context-lengths",
        type=_integers,
        help="initial decode context lengths (default: 8192)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=_integers,
        help="requests per attention-DP replica for both phases; phase-specific options take precedence",
    )
    parser.add_argument(
        "--prefill-batch-sizes", type=_integers, help="prefill batches (default: 1)"
    )
    parser.add_argument(
        "--decode-batch-sizes",
        type=_integers,
        help="decode batches (default: 1,2,4,8,16,32,64)",
    )
    parser.add_argument("--shared-prefix-tokens", type=int, default=0)
    parser.add_argument(
        "--prefix-cache",
        action="store_true",
        default=None,
        help="force the prefix cache on (the shipping server's default). Otherwise it is on only "
        "for a --grid run or a nonzero --shared-prefix-tokens, which is the historical behaviour; "
        "the flag matters because the cache also sets the checkpoint alignment the prefill chunker "
        "rounds to",
    )
    parser.add_argument(
        "--draft-tokens",
        type=_draft_list,
        help="draft positions; defaults to model configuration. With --grid, 0,K measures "
        "each cell with the drafts off and on in one load (DeepSeek-V4.1: its DSpark switch)",
    )
    parser.add_argument(
        "--accepted-drafts",
        type=int,
        help="force this many accepted drafts per step; omit for natural acceptance",
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=20)
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        help="execution chunk size, separate from prompt length (default: model preset, at least 4096; subject to kernel limits)",
    )
    parser.add_argument("--gpu-memory-utilization", type=float)
    for kind in ("k", "v"):
        parser.add_argument(
            f"--cache-type-{kind}",
            choices=("bf16", "fp8"),
            help=f"type of the {kind.upper()} cache pages (default: the model's)",
        )
    text = parser.add_argument_group(
        "real text", "decode real passages in place of the synthetic prompt"
    )
    text.add_argument(
        "--prompt-file",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="JSONL of {text, instruction} passages, from scripts/fetch_bench_corpus.py; "
        "repeat for each kind of text",
    )
    text.add_argument(
        "--text-modes",
        type=lambda value: value.split(","),
        default=["sampled"],
        help="sampled, greedy, or sampled,greedy (default: sampled)",
    )
    text.add_argument("--temperature", type=float, default=1.0)
    text.add_argument("--top-k", type=int)
    text.add_argument("--top-p", type=float, default=1.0)
    text.add_argument(
        "--seeds",
        type=_integers,
        default=(1,),
        help="sampling seeds to repeat each point with; each row adds its index",
    )
    text.add_argument(
        "--text-steps",
        type=int,
        default=200,
        help="measured decode steps per point (default: %(default)s)",
    )
    text.add_argument(
        "--text-output-tokens",
        type=int,
        default=512,
        help="longest generation per passage, before a new passage starts (default: %(default)s)",
    )
    text.add_argument(
        "--render-options",
        type=json.loads,
        default={},
        help='JSON recipe options, such as {"enable_thinking": false}',
    )
    text.add_argument(
        "--skip-unfit",
        action="store_true",
        help="skip points that do not fit the KV cache in place of failing the run",
    )
    grid = parser.add_argument_group(
        "grid",
        "decode a grid of cells on real text, growing the prompts from context to context "
        "and decoding subsets of them (needs --prompt-file); in place of the phase options",
    )
    grid.add_argument(
        "--grid",
        choices=("cross",),
        help="cross: rows (each --grid-row-contexts at each --grid-row-streams) and "
        "columns (each --grid-column-streams at each --grid-column-contexts)",
    )
    for name, default, what in (
        ("row-contexts", "2048,4096,8192,16384,32768,65536", "the rows' contexts"),
        ("row-streams", "1,8", "the rows' stream counts"),
        ("column-contexts", "4096,65536", "the columns' contexts"),
        ("column-streams", "1,2,4,8,16,32,64", "the columns' stream counts"),
    ):
        grid.add_argument(
            f"--grid-{name}",
            type=_integers_or_none,
            help=f"{what} (default: {default}; none for no rows or columns)",
        )
    grid.add_argument(
        "--grid-reuse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="grow the prompts and decode subsets of them; --no-grid-reuse prefills each "
        "cell from an empty prefix cache (to check the method)",
    )
    parser.add_argument(
        "--overlap", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--numa", choices=("auto", "off", "required"), default="auto")
    parser.add_argument(
        "--step-transport",
        choices=("gloo", "native"),
        default="gloo",
        help="how a multi-card model's ranks exchange each step: Gloo (the Python DistributedWorker), or native: "
        "rank 0's native executor sends each runner call to the others (Qwen3.5/3.8, DeepSeek-V4.1 and GLM-5.3-Flash)",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        help="capture two untimed passes per case, then merge driver-rank Perfetto traces",
    )
    parser.add_argument("--output", type=Path, default=Path("bench/results.jsonl"))
    parser.add_argument(
        "--plot", action="store_true", help="plot measured results after writing JSONL"
    )
    return parser


def _configuration(args):
    from resin.bench import BenchCase

    spec = get_model(args.model)
    world = spec.world_size if args.world_size is None else args.world_size
    topology = dict(world_size=world, cp=1 if args.cp is None else args.cp)
    for axis in ("attn_dp", "attn_tp", "ep"):
        explicit = getattr(args, axis)
        default = getattr(spec, axis)
        topology[axis] = (
            explicit
            if explicit is not None
            else (
                world if spec.world_size > 1 and default == spec.world_size else default
            )
        )
    parallelism = Parallelism(**topology)
    phases = args.phases.split(",")
    if not phases or any(phase not in {"prefill", "decode"} for phase in phases):
        raise ValueError("phases must be prefill, decode, or prefill,decode")
    contexts = {
        "prefill": args.prefill_context_lengths
        or args.context_lengths
        or [4096, 8192, 16384, 32768, 65536],
        "decode": args.decode_context_lengths or args.context_lengths or [8192],
    }
    batches = {
        "prefill": args.prefill_batch_sizes or args.batch_sizes or [1],
        "decode": args.decode_batch_sizes
        or args.batch_sizes
        or [1, 2, 4, 8, 16, 32, 64],
    }
    drafts = (
        [default_draft_tokens(spec, args.assistant_checkpoint)]
        if args.draft_tokens is None
        else args.draft_tokens
    )
    grid = None
    if args.grid is not None:
        from resin.bench import CrossGrid

        defaults = CrossGrid()
        lists = {
            name: getattr(defaults, name)
            if getattr(args, f"grid_{name}") is None
            else tuple(getattr(args, f"grid_{name}"))
            for name in ("row_contexts", "row_streams", "column_contexts", "column_streams")
        }
        CrossGrid(**lists)  # check the lists
        if not args.prompt_file:
            raise ValueError("--grid decodes real text: give --prompt-file")
        if args.trace_dir is not None or args.shared_prefix_tokens:
            raise ValueError("--grid takes no --trace-dir or --shared-prefix-tokens")
        if args.accepted_drafts is not None:
            raise ValueError("--grid measures natural acceptance only")
        if parallelism.attn_dp != 1:
            raise ValueError("--grid runs one attention replica")
        grid = dict(lists, reuse=args.grid_reuse, drafts=drafts)
        cells = CrossGrid(**lists).cells()
        # the cache checks and the capacity: each cell's prompts with their closing tokens
        cases = [
            asdict(BenchCase(phase="decode", context_length=c + GRID_TAIL_TOKENS, batch_size=n))
            for c, streams in cells.items()
            for n in streams
        ]
        if "prefill" in phases:
            cases += [
                asdict(BenchCase(phase="prefill", context_length=c + GRID_TAIL_TOKENS, batch_size=1))
                for c in cells
            ]
    else:
        cases = [
            asdict(
                BenchCase(
                    phase=phase,
                    context_length=context,
                    batch_size=batch,
                    shared_prefix_tokens=args.shared_prefix_tokens,
                )
            )
            for phase in dict.fromkeys(phases)
            for context in contexts[phase]
            for batch in batches[phase]
        ]
    if len(drafts) > 1:
        if grid is None or len(drafts) != 2 or 0 not in drafts:
            raise ValueError(
                "several --draft-tokens values: 0,K with --grid only (drafts off and on)"
            )
        if spec.family not in SWITCHED_DRAFT_FAMILIES:
            raise ValueError(
                f"--draft-tokens 0,K: {spec.name} cannot switch its drafts at run time "
                "(one load for each draft configuration)"
            )
        if parallelism.world_size > 1 and args.step_transport != "native":
            raise ValueError("--draft-tokens 0,K on several cards needs --step-transport native")
    chunk = (
        spec.prefill_chunk_size
        if args.prefill_chunk_size is None
        else args.prefill_chunk_size
    )
    options = dict(
        draft_tokens=max(drafts),
        grid=grid,
        prefix_cache=args.prefix_cache,
        accepted_drafts=args.accepted_drafts,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        prefill_chunk_size=chunk,
        gpu_memory_utilization=spec.gpu_memory_utilization
        if args.gpu_memory_utilization is None
        else args.gpu_memory_utilization,
        overlap=args.overlap,
        numa=args.numa,
        step_transport=args.step_transport,
        cache_types=(args.cache_type_k, args.cache_type_v),
        text=dict(
            tokenizer_checkpoint=None
            if args.tokenizer_checkpoint is None
            else str(args.tokenizer_checkpoint),
            prompt_files=dict(item.split("=", 1) for item in args.prompt_file),
            text_modes=args.text_modes,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            text_seeds=list(args.seeds),
            text_steps=args.text_steps,
            text_output_tokens=args.text_output_tokens,
            render_options=args.render_options,
            skip_unfit=args.skip_unfit or grid is not None,
            phases=phases,
        ),
    )
    _validate_configuration(spec.name, cases, parallelism, options)
    return spec, cases, parallelism, options


def main(argv=None) -> None:
    parser = argument_parser()
    args = parser.parse_args(argv)
    apply_gaudi_arguments(args)
    if args.checkpoint is None:
        parser.error("--checkpoint is required")
    try:
        spec, cases, topology, options = _configuration(args)
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    options["metadata"] = source_metadata()
    checkpoint = require_checkpoint(
        spec,
        checkpoint=args.checkpoint,
        assistant_checkpoint=args.assistant_checkpoint,
        tokenizer_checkpoint=args.tokenizer_checkpoint,
        draft_tokens=options["draft_tokens"],
    )
    options["metadata"]["checkpoint_revision"] = checkpoint_revision(spec, checkpoint)
    local_trace_dir = None
    if args.trace_dir is not None:
        from uuid import uuid4

        local_trace_dir = args.trace_dir / uuid4().hex
    options.update(
        trace_dir=None if local_trace_dir is None else str(local_trace_dir),
        assistant_checkpoint=args.assistant_checkpoint,
    )
    records = run_benchmark(
        spec.name, checkpoint, cases, parallelism=topology, **options
    )
    write_results(args.output, records)
    if args.trace_dir:
        from scripts.merge_bench_traces import main as merge_main

        merge_main([str(local_trace_dir)])
    print(f"Appended {len(records)} raw points to {args.output}")
    if args.plot:
        from scripts.plot_bench import main as plot_main

        plot_main(
            [str(args.output), "--mode", "both" if "," in args.phases else args.phases]
        )


if __name__ == "__main__":
    main()
