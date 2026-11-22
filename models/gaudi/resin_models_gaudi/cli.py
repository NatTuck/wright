"""The `wright` command: serve a model on Intel Gaudi 2. The flags follow llama-server.

With several models (``-m`` more than once), the flags after an ``-m`` are the flags of that model, up to the next
``-m``. The model flags before the first ``-m`` are defaults for every model. A server flag can come anywhere, once.
"""

from __future__ import annotations

import logging
import re
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from enum import Enum
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from typer.core import TyperCommand

from resin_models_gaudi import features
from resin_models_gaudi.detect import DetectionError, choose_preset, detect

# deploy/ holds the preset registry and the server. It sits at the top of the source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

app = typer.Typer(
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)
logger = logging.getLogger("wright")

_MODEL = "Each model (after its -m)"
_DRAFT = "Speculative decoding (after its -m)"
_CARDS = "Cards (after its -m)"
_SEVERAL = "Several models (after its -m)"
_SERVER = "Server"
_RUNTIME = "Runtime"
_LOGS = "Logging"

@dataclass(frozen=True)
class ModelFlags:
    """The flags that each model of a server takes for itself, with the defaults of `serve`. The other flags
    are the server's."""

    alias: str | None = None
    ctx_size: int = 0
    parallel: int = -1
    batch_size: int | None = None
    cache_prompt: bool = True
    decode_share: str | None = None
    overlap: bool = True
    gpu_memory_utilization: float | None = None
    tokenizer: Path | None = None
    spec_draft_n_max: int | None = None
    engram: str | None = None
    engram_gather: str | None = None
    engram_prewarm: bool | None = None
    model_draft: Path | None = None
    cache_type_k: CacheType | None = None
    cache_type_v: CacheType | None = None
    device: str | None = None
    ranks: int | None = None
    share: float | None = None
    kv_floor: int | None = None


PER_MODEL = frozenset(field.name for field in fields(ModelFlags))

_SCOPED = "wright.models"


class Colors(str, Enum):
    on = "on"
    off = "off"
    auto = "auto"


class Numa(str, Enum):
    auto = "auto"
    off = "off"
    required = "required"


class StepTransport(str, Enum):
    auto = "auto"
    socket = "socket"
    gloo = "gloo"
    native = "native"


class Optimize(str, Enum):
    latency = "latency"
    throughput = "throughput"


class CacheType(str, Enum):
    bf16 = "bf16"
    fp8 = "fp8"


def fail(message: str) -> NoReturn:
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(2)


# The server's names for the flags that this command names differently.
_SERVER_FLAGS = {
    "--context-length": "--ctx-size",
    "--max-batch-size": "--parallel",
    "--prefill-chunk-size": "--batch-size",
    "--draft-tokens": "--spec-draft-n-max",
    "--assistant-checkpoint": "--model-draft",
    "--gpu": "--device",
    "--log-dir": "--synapse-log-dir",
}


def server_error(message: str) -> NoReturn:
    """A usage error from deploy.serving, with the flag names of this command."""
    for server, ours in _SERVER_FLAGS.items():
        message = re.sub(re.escape(server) + r"(?![\w-])", ours, message)
    fail(message)


def _module(name: str, text: str) -> int:
    module = name.strip()
    module = module[3:] if module.upper().startswith("HPU") else module
    if not module.isdigit():
        fail(
            f"--device takes names such as HPU0,HPU1, module IDs such as 0,1, or ranges such "
            f"as 0-7, not {text!r}"
        )
    return int(module)


def device_modules(text: str) -> str:
    """`-dev` names cards as HPU<module>, as --list-devices shows them, as bare module IDs, or as ranges of
    either (0-3, HPU0-HPU3)."""
    modules: list[int] = []
    for name in (name.strip() for name in text.split(",")):
        if not name:
            continue
        low, dash, high = name.partition("-")
        first = _module(low, text)
        last = _module(high, text) if dash else first
        if last < first:
            fail(f"--device: the range {name} ends before it starts")
        modules += range(first, last + 1)
    if not modules:
        _module("", text)
    return ",".join(map(str, modules))


def list_devices(value: bool) -> None:
    if not value:
        return
    from resin_models_gaudi.synapse.cards import all_cards

    typer.echo("Available devices:")
    for card in sorted(all_cards(), key=lambda card: card.module):
        state = "" if card.operational else ", not operational"
        typer.echo(
            f"  HPU{card.module}: /dev/accel/accel{card.accel}, {card.pci}{state}"
        )
    raise typer.Exit()


def split_models(
    args: list[str], names: frozenset[str] = frozenset()
) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """The arguments before the first -m, and each -m's path with the arguments after it, up to the next -m.
    No parts when an -m has no path (the parser then reports it). ``names``: the command's option names, so
    that an option such as -md is not read as -m with the path d."""
    head: list[str] = []
    parts: list[tuple[str, list[str]]] = []
    tokens = iter(args)
    for token in tokens:
        if token in ("-m", "--model"):
            path = next(tokens, None)
            if path is None:
                return head, []
            parts.append((path, []))
        elif token.startswith("--model="):
            parts.append((token.partition("=")[2], []))
        elif (
            token.startswith("-m")
            and not token.startswith("--")
            and not any(token == name or token.startswith(name + "=") for name in names - {"-m"})
        ):  # -m with its path attached (-mPATH), as the parser reads it
            parts.append((token[2:], []))
        else:
            (parts[-1][1] if parts else head).append(token)
    return head, parts


def _flag(param) -> str:
    return next((name for name in param.opts if name.startswith("--")), param.opts[0])


class ScopedCommand(TyperCommand):
    """The serve command. With several models, it reads the flags of each -m from the arguments after it (see
    the module docstring), into ``ctx.meta[_SCOPED]``: the model flags before the first -m, and each model's
    own."""

    def parse_args(self, ctx, args: list[str]) -> list[str]:
        names = frozenset(name for param in self.params for name in (*param.opts, *param.secondary_opts))
        head, parts = split_models(args, names)
        if set(self.get_help_option_names(ctx)) & set(args):
            pass
        elif len(parts) <= 1:  # one model: its flags may come anywhere, once
            self._given(ctx, list(args), None, {})
        else:
            server: dict[str, str] = {}
            defaults = self._given(ctx, head, "before the first -m", server)
            own = [self._given(ctx, tokens, f"after -m {path}", server) for path, tokens in parts]
            ctx.meta[_SCOPED] = (
                {name: value for name, value in defaults.items() if name in PER_MODEL},
                [{name: value for name, value in values.items() if name in PER_MODEL} for values in own],
            )
        return super().parse_args(ctx, args)

    def _given(self, ctx, tokens: list[str], where: str | None, server: dict[str, str]) -> dict:
        """The options that ``tokens`` give, converted. ``where``: the part of the arguments, None with one
        model. ``server``: where each server option was given."""
        scratch = self.context_class(
            self, info_name=ctx.info_name, parent=ctx.parent, **self.context_settings
        )
        _, _, order = self.make_parser(scratch).parse_args(list(tokens))
        for name, count in Counter(param.name or "" for param in order).items():
            param = next(param for param in order if param.name == name)
            if count > 1 and name not in ("model", "feature"):  # (-f may come more than once)
                fail(
                    f"{_flag(param)} is given {count} times: give it once"
                    if where is None
                    else f"{_flag(param)} is given {count} times {where}: give each model its own "
                    f"{_flag(param)} after its -m"
                    if name in PER_MODEL
                    else f"{_flag(param)} is given {count} times: it is a flag of the server"
                )
            if name not in PER_MODEL and param.name != "model":
                if name in server:
                    fail(
                        f"{_flag(param)} is given {server[name]} and {where}: it is a flag of "
                        "the server, so give it once"
                    )
                server[name] = where or ""
        super().parse_args(scratch, list(tokens))
        return {name: scratch.params[name] for name in (param.name or "" for param in order)}


def _features_option(values: list[str] | None) -> list[str] | None:
    """``-f``: ``help`` shows the features and exits; an unknown name stops the command."""
    if not values:
        return values
    if "help" in (name.strip().lower() for value in values for name in value.split(",")):
        typer.echo(features.describe())
        raise typer.Exit()
    try:
        features.parse(values)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from None
    return values


def server_arguments(
    preset: str,
    checkpoint: Path,
    *,
    tokenizer: Path | None = None,
    alias: str | None = None,
    ctx_size: int = 0,
    parallel: int = -1,
    batch_size: int | None = None,
    draft_tokens: int | None = None,
    engram: str | None = None,
    engram_gather: str | None = None,
    engram_prewarm: bool | None = None,
    model_draft: Path | None = None,
    cache_types: tuple[CacheType | None, CacheType | None] = (None, None),
    device: str | None = None,
    ranks: int | None = None,
    numa: Numa = Numa.auto,
    step_transport: StepTransport = StepTransport.auto,
    optimize: Optimize = Optimize.latency,
    host: str = "127.0.0.1",
    port: int = 8080,
    cache_prompt: bool = True,
    overlap: bool = True,
    decode_share: str | None = None,
    drain_timeout: float = 30.0,
    stream_interval: int = 1,
    request_queue_capacity: int = 1024,
    cache_dir: str | None = None,
    synapse_log_dir: str | None = None,
    load_threads: int | None = None,
    gpu_memory_utilization: float | None = None,
    spin_wait: float | None = None,
    features: Sequence[str] = (),
) -> list[str]:
    """The same settings in the terms of deploy.serving, which checks them all."""
    line = ["--model", preset, "--checkpoint", str(checkpoint)]
    for flag, value in (
        ("--tokenizer", tokenizer),
        ("--alias", alias),
        ("--context-length", ctx_size if ctx_size > 0 else None),
        ("--max-batch-size", parallel if parallel > 0 else None),
        ("--prefill-chunk-size", batch_size),
        ("--draft-tokens", draft_tokens),
        ("--engram", engram),
        ("--engram-gather", engram_gather),
        ("--assistant-checkpoint", model_draft),
        # (a model's flags of a layout come as their text)
        ("--cache-type-k", None if cache_types[0] is None else CacheType(cache_types[0]).value),
        ("--cache-type-v", None if cache_types[1] is None else CacheType(cache_types[1]).value),
        ("--gpu", device_modules(device) if device else None),
        ("--ranks", ranks),
        ("--cache-dir", cache_dir),
        ("--log-dir", synapse_log_dir),
        ("--load-threads", load_threads),
        ("--gpu-memory-utilization", gpu_memory_utilization),
        ("--spin-wait", spin_wait),
        ("--decode-share", decode_share),
    ):
        if value is not None:
            line += [flag, str(value)]
    if engram_prewarm is not None:
        line += ["--engram-prewarm" if engram_prewarm else "--no-engram-prewarm"]
    line += [
        "--numa",
        numa.value,
        "--step-transport",
        step_transport.value,
        "--optimize",
        optimize.value,
        "--host",
        host,
        "--port",
        str(port),
        "--prefix-cache" if cache_prompt else "--no-prefix-cache",
        "--overlap" if overlap else "--no-overlap",
        "--drain-timeout",
        str(drain_timeout),
        "--stream-interval",
        str(stream_interval),
        "--request-queue-capacity",
        str(request_queue_capacity),
    ]
    if features:
        line += ["--features", ",".join(features)]
    return line


@app.command(cls=ScopedCommand, context_settings={"help_option_names": ["-h", "--help"]})
def serve(
    ctx: typer.Context,
    model: Annotated[
        list[Path] | None,
        typer.Option(
            "-m",
            "--model",
            metavar="PATH",
            help="Checkpoint directory, GGUF directory, or one .gguf file. Wright finds the "
            "runner from its contents. Give -m more than once to serve several models: on one "
            "card, on cards of their own, or as replicas on the cards of the first model "
            "(refer to --device). Each model's flags then follow its -m.",
            rich_help_panel=_MODEL,
        ),
    ] = None,
    alias: Annotated[
        str | None,
        typer.Option(
            "-a",
            "--alias",
            metavar="NAME",
            help="Model name in the API. (default: the checkpoint's repository name)",
            rich_help_panel=_MODEL,
        ),
    ] = None,
    ctx_size: Annotated[
        int,
        typer.Option(
            "-c",
            "--ctx-size",
            metavar="N",
            help="Maximum tokens per sequence, not shared across slots as in llama-server. "
            "0 uses the model's default.",
            rich_help_panel=_MODEL,
        ),
    ] = 0,
    parallel: Annotated[
        int,
        typer.Option(
            "-np",
            "--parallel",
            metavar="N",
            help="Maximum sequences in one batch. -1 uses the model's default.",
            rich_help_panel=_MODEL,
        ),
    ] = -1,
    batch_size: Annotated[
        int | None,
        typer.Option(
            "-b",
            "--batch-size",
            metavar="N",
            help="Tokens per prefill chunk. A smaller chunk stops decode for a shorter time. "
            "(default: the model's)",
            rich_help_panel=_MODEL,
        ),
    ] = None,
    cache_prompt: Annotated[
        bool,
        typer.Option(
            "--cache-prompt/--no-cache-prompt",
            help="Let a new request reuse cached prompt prefixes.",
            rich_help_panel=_MODEL,
        ),
    ] = True,
    decode_share: Annotated[
        str | None,
        typer.Option(
            metavar="MIN[:MAX]",
            help="Percent of card time for decode while a prompt prefills: MIN with no "
            "sequence in decode, up to MAX when every batch position decodes. One number "
            "fixes it. 0 runs each prompt's whole prefill before decode. (default: 50:90)",
            rich_help_panel=_MODEL,
        ),
    ] = None,
    overlap: Annotated[
        bool,
        typer.Option(
            help="Start the next decode step while the card runs the current step.",
            rich_help_panel=_MODEL,
        ),
    ] = True,
    gpu_memory_utilization: Annotated[
        float | None,
        typer.Option(
            metavar="F",
            help="Share of the card's memory that a one-card model may fill: weights, "
            "workspace and KV cache. A new value recompiles recipes at startup. With several "
            "models on one card, it is the share for all of them: give it before the first -m. "
            "For a replica model on the cards of the first model, it is a fixed share for each "
            "replica, and the replicas do not share the KV cache of the first model. The first "
            "model of several cards takes the memory that is left. (default: the model's; "
            "replicas on the cards of DeepSeek-V4.1-Flash or GLM-5.3-Flash share its KV cache "
            "when neither model has a drafter (-md); other replicas on the first model's cards: "
            "0.25)",
            rich_help_panel=_MODEL,
        ),
    ] = None,
    tokenizer: Annotated[
        Path | None,
        typer.Option(
            metavar="DIR",
            help="Tokenizer directory, for a GGUF checkpoint that has no tokenizer.json.",
            rich_help_panel=_MODEL,
        ),
    ] = None,
    spec_draft_n_max: Annotated[
        int | None,
        typer.Option(
            "--spec-draft-n-max",
            metavar="N",
            help="Draft tokens per decode step. 0 stops speculative decoding. (default: the "
            "model's)",
            rich_help_panel=_DRAFT,
        ),
    ] = None,
    model_draft: Annotated[
        Path | None,
        typer.Option(
            "-md",
            "--model-draft",
            "--spec-draft-model",
            metavar="DIR",
            help=(
                "Draft model directory: the DSpark drafter for Mistral-Large-4, the DFlash 2 "
                "drafter for Qwen3.8-27B and GLM-5.3-Flash, or the DSpine drafter for "
                "Qwen3.8-27B."
            ),
            rich_help_panel=_DRAFT,
        ),
    ] = None,
    cache_type_k: Annotated[
        CacheType | None,
        typer.Option(
            "-ctk",
            "--cache-type-k",
            help="Type of the K cache: bf16, or fp8 (about half the memory of bf16). Qwen3.5/3.8 takes "
            "either. DeepSeek-V4.1-Flash, GLM-5.3-Flash and Mistral-Large-4 keep an fp8 cache, and "
            "Qwen3.8-Flash-Next a bf16 cache. (default: the model's: bf16 for Qwen3.5/3.8)",
            rich_help_panel=_MODEL,
        ),
    ] = None,
    cache_type_v: Annotated[
        CacheType | None,
        typer.Option(
            "-ctv",
            "--cache-type-v",
            help="Type of the V cache, as --cache-type-k. (default: the model's)",
            rich_help_panel=_MODEL,
        ),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option(
            "-dev",
            "--device",
            metavar="LIST",
            help="Cards that the model may use, such as HPU0,HPU1, 0,1 or 0-7. Busy cards are "
            "skipped. With several models, give each model its own -dev to put the models on "
            "different cards: the first model spans its cards, and every other model runs one "
            "replica per card, on cards of its own or on the first model's cards (rank 0's "
            "too). With no -dev after any -m, the models share one card. In a layout, a model with "
            "no -dev of its own takes the -dev before the first -m. (default: "
            "HABANA_VISIBLE_MODULES, then all cards)",
            rich_help_panel=_CARDS,
        ),
    ] = None,
    ranks: Annotated[
        int | None,
        typer.Option(
            metavar="N",
            help="Cards to split the model across. GLM-5.3-Flash takes 4 or 8. Mistral-Large-4 "
            "and DeepSeek-V4.1-Flash need 8. Qwen3.5/3.8 runs on 1, 2, 4 or 8, and on 1 only with -f "
            "oldqwen. (default: the model's)",
            rich_help_panel=_CARDS,
        ),
    ] = None,
    share: Annotated[
        float | None,
        typer.Option(
            "--share",
            metavar="W",
            help="The model's weight in the card's time while several models have work. "
            "(default: 1 for each model on one card. For replicas on the cards of the first "
            "model, 1 for the first model and 0 for the replicas, which then step only when the "
            "first model is idle on their card)",
            rich_help_panel=_SEVERAL,
        ),
    ] = None,
    kv_floor: Annotated[
        int | None,
        typer.Option(
            "--kv-floor",
            metavar="SLABS",
            help="With several models on one card: cache slabs kept for the model. On the "
            "cards of a multi-card model, each model keeps one sequence at its context length. "
            "(default: 0)",
            rich_help_panel=_SEVERAL,
        ),
    ] = None,
    optimize: Annotated[
        Optimize,
        typer.Option(
            "-o",
            "--optimize",
            help="For replicas on the cards of the first model: latency lets the first model go "
            "first, and the replicas step in its idle time (or in their --share); throughput lets "
            "both models run at once, for more work in total, and the first model decodes slower.",
            rich_help_panel=_SERVER,
        ),
    ] = Optimize.latency,
    host: Annotated[
        str,
        typer.Option(
            "--host",
            metavar="HOST",
            help="IP address to listen on. The server has no authentication.",
            rich_help_panel=_SERVER,
        ),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option(
            "--port", metavar="PORT", help="Port to listen on.", rich_help_panel=_SERVER
        ),
    ] = 8080,
    drain_timeout: Annotated[
        float,
        typer.Option(
            metavar="SECONDS",
            help="On SIGTERM, time for requests in flight to finish.",
            rich_help_panel=_SERVER,
        ),
    ] = 30.0,
    stream_interval: Annotated[
        int,
        typer.Option(
            metavar="N",
            help="Generated tokens per streamed chunk.",
            rich_help_panel=_SERVER,
        ),
    ] = 1,
    request_queue_capacity: Annotated[
        int,
        typer.Option(
            metavar="N",
            help="Requests that can wait for admission before HTTP 429.",
            rich_help_panel=_SERVER,
        ),
    ] = 1024,
    _list_devices: Annotated[
        bool,
        typer.Option(
            "--list-devices",
            callback=list_devices,
            is_eager=True,
            help="Show the cards and stop.",
            rich_help_panel=_RUNTIME,
        ),
    ] = False,
    numa: Annotated[
        Numa,
        typer.Option(
            help="Bind each process to the NUMA node of its card.",
            rich_help_panel=_RUNTIME,
        ),
    ] = Numa.auto,
    step_transport: Annotated[
        StepTransport,
        typer.Option(
            help="How the cards of a model on several cards exchange each step: Unix "
            "sockets, or torch Gloo. auto uses sockets when all the cards are on this host. "
            "native (Qwen3.5/3.8, DeepSeek-V4.1): rank 0's native executor sends each step to the other "
            "cards, with no Python in the step path.",
            rich_help_panel=_RUNTIME,
        ),
    ] = StepTransport.auto,
    cache_dir: Annotated[
        str | None,
        typer.Option(
            metavar="DIR",
            help="Directory for compiled kernels and recipes.",
            rich_help_panel=_RUNTIME,
        ),
    ] = None,
    load_threads: Annotated[
        int | None,
        typer.Option(
            metavar="N",
            help="Threads that prepare GGUF weights.",
            rich_help_panel=_RUNTIME,
        ),
    ] = None,
    engram: Annotated[
        str | None,
        typer.Option(
            metavar="MODE",
            help="Where the DeepSeek-V4.1 Engram tables live: 'card' or 'host'. Host keeps them "
            "in the checkpoint's file and gathers rows each step, which is what makes four cards "
            "possible. (default: the model's)",
            rich_help_panel=_RUNTIME,
        ),
    ] = None,
    engram_gather: Annotated[
        str | None,
        typer.Option(
            metavar="HOW",
            help="How the host gather reads the tables: 'pooled', 'mmap', or 'auto' by measured "
            "page-cache residency. (default: the model's)",
            rich_help_panel=_RUNTIME,
        ),
    ] = None,
    engram_prewarm: Annotated[
        bool | None,
        typer.Option(
            "--engram-prewarm/--no-engram-prewarm",
            help="Read the two Engram shards once at startup so the gather can map them. "
            "(default: the model's)",
            rich_help_panel=_RUNTIME,
        ),
    ] = None,
    spin_wait: Annotated[
        float | None,
        typer.Option(
            metavar="SECONDS",
            help="Time to poll for a launch before blocking, which keeps one core busy.",
            rich_help_panel=_RUNTIME,
        ),
    ] = None,
    feature: Annotated[
        list[str] | None,
        typer.Option(
            "-f",
            "--features",
            metavar="NAME[,NAME...]",
            help="Optional features, each off unless you name it: oldqwen (Qwen3.5/3.8 steps in the Python "
            "executor, on one card only) and skipvalidation (no check of each launch's tensor bindings). Give names separated by "
            "commas, or -f more than once. -f help shows the features.",
            callback=_features_option,
            is_eager=True,
            rich_help_panel=_RUNTIME,
        ),
    ] = None,
    verbosity: Annotated[
        int,
        typer.Option(
            "-lv",
            "--verbosity",
            "--log-verbosity",
            metavar="N",
            help="Log level: 0 critical, 1 error, 2 warning, 3 info, 4 debug.",
            rich_help_panel=_LOGS,
        ),
    ] = 3,
    log_colors: Annotated[
        Colors,
        typer.Option(help="Colored log output.", rich_help_panel=_LOGS),
    ] = Colors.auto,
    log_file: Annotated[
        str | None,
        typer.Option(
            metavar="FNAME",
            help="Also write the log to this file.",
            rich_help_panel=_LOGS,
        ),
    ] = None,
    synapse_log_dir: Annotated[
        str | None,
        typer.Option(
            metavar="DIR", help="Directory for Synapse logs.", rich_help_panel=_LOGS
        ),
    ] = None,
) -> None:
    """Serve a model on Intel Gaudi 2 over OpenAI and Anthropic HTTP APIs.

    With several models, the flags after an -m are the flags of that model, up to the next -m. The flags of the
    groups marked "after its -m" apply to one model; give them before the first -m to set them for every model.
    The other flags are the server's: give them anywhere, once.
    """
    if not model:
        fail("give a model with -m PATH")
    from deploy.logs import configure_logging
    from deploy.serving import main as serve_main
    from deploy.serving import main_layout, main_models

    configure_logging(verbosity, log_colors.value, log_file)
    scoped = ctx.meta.get(_SCOPED)
    if scoped is None:  # one model: its flags may come anywhere
        own: list[dict] = [{}]
        models = [
            ModelFlags(
                alias=alias,
                ctx_size=ctx_size,
                parallel=parallel,
                batch_size=batch_size,
                cache_prompt=cache_prompt,
                decode_share=decode_share,
                overlap=overlap,
                gpu_memory_utilization=gpu_memory_utilization,
                tokenizer=tokenizer,
                spec_draft_n_max=spec_draft_n_max,
                engram=engram,
                engram_gather=engram_gather,
                engram_prewarm=engram_prewarm,
                model_draft=model_draft,
                cache_type_k=cache_type_k,
                cache_type_v=cache_type_v,
                device=device,
                ranks=ranks,
                share=share,
                kv_floor=kv_floor,
            )
        ]
    else:
        defaults, own = scoped
        models = [ModelFlags(**{**defaults, **values}) for values in own]
    several = len(model) > 1
    # A -dev for each model puts the models on different cards; else they share one card.
    layout = several and any("device" in values for values in own)
    if not several and (share is not None or kv_floor is not None):
        fail("--share and --kv-floor are for several models")
    if optimize is Optimize.throughput and not layout:
        fail(
            "-o throughput is for replicas on the cards of the first model: give each model "
            "its -dev after its -m"
        )
    if layout:
        missing = [str(path) for path, flags in zip(model, models) if flags.device is None]
        if missing:
            fail(f"give each model its cards with -dev after its -m ({', '.join(missing)} has none)")
        if any(flags.kv_floor is not None for flags in models):
            fail("--kv-floor is for several models on one card")
        if "gpu_memory_utilization" in own[0]:
            fail(
                "the first model of several cards makes its KV cache from the memory that the "
                "replicas leave: give --gpu-memory-utilization after a replica model's -m"
            )
        models[0] = replace(models[0], gpu_memory_utilization=None)
    elif several:
        if any(flags.ranks not in (None, 1) for flags in models):
            fail("give each model its -dev after its -m to put several models on cards of their own")
        if any("gpu_memory_utilization" in values for values in own):
            fail(
                "with several models on one card, --gpu-memory-utilization is the share of the "
                "card for all of them: give it before the first -m"
            )
    argument_lists = []
    for path, flags in zip(model, models, strict=True):
        try:
            detected = detect(path)
        except DetectionError as error:
            fail(str(error))
        preset = choose_preset(detected, flags.ranks, flags.ctx_size or None)
        logger.info(
            f"model: {detected.name} from {detected.checkpoint}, preset {preset}"
        )
        if not detected.tokenizer_included and flags.tokenizer is None:
            fail(f"{detected.name} from GGUF needs its tokenizer: give --tokenizer DIR")
        argument_lists.append(
            server_arguments(
                preset,
                detected.checkpoint,
                tokenizer=flags.tokenizer,
                alias=flags.alias,
                ctx_size=flags.ctx_size,
                parallel=flags.parallel,
                batch_size=flags.batch_size,
                draft_tokens=flags.spec_draft_n_max,
                engram=flags.engram,
                engram_gather=flags.engram_gather,
                engram_prewarm=flags.engram_prewarm,
                model_draft=flags.model_draft,
                cache_types=(flags.cache_type_k, flags.cache_type_v),
                device=flags.device,
                ranks=None
                if flags.ranks == 4 and preset.endswith("-tp4")
                else flags.ranks,
                numa=numa,
                step_transport=step_transport,
                optimize=optimize,
                host=host,
                port=port,
                cache_prompt=flags.cache_prompt,
                overlap=flags.overlap,
                decode_share=flags.decode_share,
                drain_timeout=drain_timeout,
                stream_interval=stream_interval,
                request_queue_capacity=request_queue_capacity,
                cache_dir=cache_dir,
                synapse_log_dir=synapse_log_dir,
                load_threads=load_threads,
                gpu_memory_utilization=flags.gpu_memory_utilization,
                spin_wait=spin_wait,
                features=features.parse(feature or ()),
            )
        )
    shares = [flags.share for flags in models]
    if layout:
        main_layout(
            argument_lists,
            shares=None
            if all(weight is None for weight in shares)
            else [
                (1.0 if index == 0 else 0.0) if weight is None else weight
                for index, weight in enumerate(shares)
            ],
            on_error=server_error,
        )
    elif several:
        main_models(
            argument_lists,
            shares=[1.0 if weight is None else weight for weight in shares],
            floors=[flags.kv_floor or 0 for flags in models],
            on_error=server_error,
        )
    else:
        serve_main(argument_lists[0], on_error=server_error)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
