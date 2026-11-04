"""The command-line settings that deploy/serving.py and scripts/bench.py share on Gaudi.

Each flag sets the variable that the runtime reads, so spawned ranks inherit it. The order of
precedence is flag, then variable, then default.
"""

from __future__ import annotations

import argparse
import ipaddress
import os
import socket


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text}")
    return value


def nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"expected 0 or more, got {text}")
    return value


def module_list(text: str) -> str:
    from resin_models_gaudi.synapse.cards import parse_modules

    try:
        parse_modules(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None
    return text


def add_gaudi_arguments(parser: argparse.ArgumentParser) -> None:
    gaudi = parser.add_argument_group("Gaudi")
    gaudi.add_argument(
        "--gpu",
        type=module_list,
        metavar="MODULES",
        help="module IDs that this process may use, as hl-smi shows them, such as 0,1,2,3. "
        "Sets HABANA_VISIBLE_MODULES. Ranks skip busy cards in the list. "
        "(default: HABANA_VISIBLE_MODULES, then every operational card)",
    )
    gaudi.add_argument(
        "--cache-dir",
        help="directory for compiled kernels and recipes. Sets RESIN_GAUDI_CACHE. "
        "(default: ~/.cache/resin/gaudi)",
    )
    gaudi.add_argument(
        "--log-dir",
        help="directory for Synapse logs. Sets HABANA_LOGS. (default: HABANA_LOGS if it is "
        "writable, otherwise a directory in the cache)",
    )
    gaudi.add_argument(
        "--log-level",
        type=nonnegative_int,
        help="Synapse log level, where a lower value logs more. Sets LOG_LEVEL_ALL. (default: 4)",
    )
    gaudi.add_argument(
        "--load-threads",
        type=positive_int,
        help="threads that prepare GGUF weights. Sets RESIN_GAUDI_LOAD_THREADS. (default: 32)",
    )
    gaudi.add_argument(
        "--spin-wait",
        type=float,
        metavar="SECONDS",
        help="time to poll for a launch before blocking, which keeps one core busy. "
        "Sets RESIN_GAUDI_SPIN_WAIT. (default: 0.2)",
    )
    gaudi.add_argument(
        "--cc", help="C compiler for native code. Sets CC. (default: gcc)"
    )
    gaudi.add_argument(
        "--cxx", help="C++ compiler for native code. Sets CXX. (default: g++)"
    )


_VARIABLES = {
    "gpu": "HABANA_VISIBLE_MODULES",
    "cache_dir": "RESIN_GAUDI_CACHE",
    "log_dir": "HABANA_LOGS",
    "log_level": "LOG_LEVEL_ALL",
    "load_threads": "RESIN_GAUDI_LOAD_THREADS",
    "spin_wait": "RESIN_GAUDI_SPIN_WAIT",
    "cc": "CC",
    "cxx": "CXX",
}


def apply_gaudi_arguments(args: argparse.Namespace) -> None:
    """Set the variables for this process and its ranks."""
    for name, variable in _VARIABLES.items():
        value = getattr(args, name, None)
        if value is not None:
            os.environ[variable] = str(value)


def describe_settings() -> str:
    """The settings in effect, for the startup log."""
    values = {
        variable: os.environ[variable]
        for variable in _VARIABLES.values()
        if variable in os.environ
    }
    return " ".join(f"{name}={value}" for name, value in values.items()) or "defaults"


def check_listen_address(host: str, port: int) -> str | None:
    """Why the server cannot listen on this address, or None. The server binds only after a
    load of several minutes, so check this first."""
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return f"--host takes an IP address, such as 127.0.0.1, got {host!r}"
    if not 1 <= port <= 65535:
        return f"--port takes a value from 1 to 65535, got {port}"
    family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
    with socket.socket(family) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError as error:
            return f"cannot listen on {host}:{port}: {error.strerror}"
    return None
