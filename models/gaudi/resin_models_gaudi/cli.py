"""The `wright` command: serve a model on Intel Gaudi 2. The flags follow llama-server."""

from __future__ import annotations

import logging
import re
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from resin_models_gaudi.detect import DetectionError, choose_preset, detect

# deploy/ holds the preset registry and the server. It sits at the top of the source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

app = typer.Typer(
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)
logger = logging.getLogger("wright")

_MODEL = "Model"
_LOAD = "Context and batch"
_DRAFT = "Speculative decoding"
_CARDS = "Cards"
_SERVER = "Server"
_LOGS = "Logging"
_WRIGHT = "Wright only"


class Colors(str, Enum):
    on = "on"
    off = "off"
    auto = "auto"


class Numa(str, Enum):
    auto = "auto"
    off = "off"
    required = "required"


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


def device_modules(text: str) -> str:
    """`-dev` names cards as HPU<module>, as --list-devices shows them, or as bare module IDs."""
    names = [name.strip() for name in text.split(",") if name.strip()]
    modules = [name[3:] if name.upper().startswith("HPU") else name for name in names]
    if not modules or not all(module.isdigit() for module in modules):
        fail(
            f"--device takes names such as HPU0,HPU1 or module IDs such as 0,1, not {text!r}"
        )
    return ",".join(modules)


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
    model_draft: Path | None = None,
    device: str | None = None,
    ranks: int | None = None,
    numa: Numa = Numa.auto,
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
        ("--assistant-checkpoint", model_draft),
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
    line += [
        "--numa",
        numa.value,
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
    return line


@app.command(context_settings={"help_option_names": ["-h", "--help"]})
def serve(
    model: Annotated[
        Path | None,
        typer.Option(
            "-m",
            "--model",
            metavar="PATH",
            help="Checkpoint directory, GGUF directory, or one .gguf file. Wright finds the "
            "runner from its contents.",
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
            rich_help_panel=_LOAD,
        ),
    ] = 0,
    parallel: Annotated[
        int,
        typer.Option(
            "-np",
            "--parallel",
            metavar="N",
            help="Maximum sequences in one batch. -1 uses the model's default.",
            rich_help_panel=_LOAD,
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
            rich_help_panel=_LOAD,
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
            help="Draft model directory: the DSpark drafter for Mistral-Large-4.",
            rich_help_panel=_DRAFT,
        ),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option(
            "-dev",
            "--device",
            metavar="LIST",
            help="Cards that this server may use, such as HPU0,HPU1 or 0,1. Busy cards are "
            "skipped. (default: HABANA_VISIBLE_MODULES, then all cards)",
            rich_help_panel=_CARDS,
        ),
    ] = None,
    _list_devices: Annotated[
        bool,
        typer.Option(
            "--list-devices",
            callback=list_devices,
            is_eager=True,
            help="Show the cards and stop.",
            rich_help_panel=_CARDS,
        ),
    ] = False,
    numa: Annotated[
        Numa,
        typer.Option(
            help="Bind each process to the NUMA node of its card.",
            rich_help_panel=_CARDS,
        ),
    ] = Numa.auto,
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
    cache_prompt: Annotated[
        bool,
        typer.Option(
            "--cache-prompt/--no-cache-prompt",
            help="Let a new request reuse cached prompt prefixes.",
            rich_help_panel=_SERVER,
        ),
    ] = True,
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
    tokenizer: Annotated[
        Path | None,
        typer.Option(
            metavar="DIR",
            help="Tokenizer directory, for a GGUF checkpoint that has no tokenizer.json.",
            rich_help_panel=_WRIGHT,
        ),
    ] = None,
    ranks: Annotated[
        int | None,
        typer.Option(
            metavar="N",
            help="Cards to split the model across. GLM-5.3-Flash takes 4 or 8. Mistral-Large-4 "
            "needs 8. Qwen3.8-27B runs on 1. (default: the model's)",
            rich_help_panel=_WRIGHT,
        ),
    ] = None,
    gpu_memory_utilization: Annotated[
        float | None,
        typer.Option(
            metavar="F",
            help="Share of free card memory for the KV cache on the one-card models. A new "
            "value recompiles recipes at startup.",
            rich_help_panel=_WRIGHT,
        ),
    ] = None,
    spin_wait: Annotated[
        float | None,
        typer.Option(
            metavar="SECONDS",
            help="Time to poll for a launch before blocking, which keeps one core busy.",
            rich_help_panel=_WRIGHT,
        ),
    ] = None,
    overlap: Annotated[
        bool,
        typer.Option(
            help="Start the next decode step while the card runs the current step.",
            rich_help_panel=_WRIGHT,
        ),
    ] = True,
    decode_share: Annotated[
        str | None,
        typer.Option(
            metavar="MIN[:MAX]",
            help="Percent of card time for decode while a prompt prefills: MIN with no "
            "sequence in decode, up to MAX when every batch position decodes. One number "
            "fixes it. 0 runs each prompt's whole prefill before decode. (default: 50:90)",
            rich_help_panel=_WRIGHT,
        ),
    ] = None,
    drain_timeout: Annotated[
        float,
        typer.Option(
            metavar="SECONDS",
            help="On SIGTERM, time for requests in flight to finish.",
            rich_help_panel=_WRIGHT,
        ),
    ] = 30.0,
    stream_interval: Annotated[
        int,
        typer.Option(
            metavar="N",
            help="Generated tokens per streamed chunk.",
            rich_help_panel=_WRIGHT,
        ),
    ] = 1,
    request_queue_capacity: Annotated[
        int,
        typer.Option(
            metavar="N",
            help="Requests that can wait for admission before HTTP 429.",
            rich_help_panel=_WRIGHT,
        ),
    ] = 1024,
    cache_dir: Annotated[
        str | None,
        typer.Option(
            metavar="DIR",
            help="Directory for compiled kernels and recipes.",
            rich_help_panel=_WRIGHT,
        ),
    ] = None,
    synapse_log_dir: Annotated[
        str | None,
        typer.Option(
            metavar="DIR", help="Directory for Synapse logs.", rich_help_panel=_WRIGHT
        ),
    ] = None,
    load_threads: Annotated[
        int | None,
        typer.Option(
            metavar="N",
            help="Threads that prepare GGUF weights.",
            rich_help_panel=_WRIGHT,
        ),
    ] = None,
) -> None:
    """Serve a model on Intel Gaudi 2 over OpenAI and Anthropic HTTP APIs."""
    if model is None:
        fail("give a model with -m PATH")
    from deploy.logs import configure_logging
    from deploy.serving import main as serve_main

    configure_logging(verbosity, log_colors.value, log_file)
    try:
        detected = detect(model)
    except DetectionError as error:
        fail(str(error))
    preset = choose_preset(detected, ranks, ctx_size or None)
    logger.info(f"model: {detected.name} from {detected.checkpoint}, preset {preset}")
    if not detected.tokenizer_included and tokenizer is None:
        fail(f"{detected.name} from GGUF needs its tokenizer: give --tokenizer DIR")
    arguments = server_arguments(
        preset,
        detected.checkpoint,
        tokenizer=tokenizer,
        alias=alias,
        ctx_size=ctx_size,
        parallel=parallel,
        batch_size=batch_size,
        draft_tokens=spec_draft_n_max,
        model_draft=model_draft,
        device=device,
        ranks=None if ranks == 4 and preset.endswith("-tp4") else ranks,
        numa=numa,
        host=host,
        port=port,
        cache_prompt=cache_prompt,
        overlap=overlap,
        decode_share=decode_share,
        drain_timeout=drain_timeout,
        stream_interval=stream_interval,
        request_queue_capacity=request_queue_capacity,
        cache_dir=cache_dir,
        synapse_log_dir=synapse_log_dir,
        load_threads=load_threads,
        gpu_memory_utilization=gpu_memory_utilization,
        spin_wait=spin_wait,
    )
    serve_main(arguments, on_error=server_error)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
