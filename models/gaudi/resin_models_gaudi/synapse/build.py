"""Build the Synapse runtime wrapper, the custom TPC kernel library and the host packing library.

Both libraries are built from the sources in this package and published through
``resin.kernel_cache.ArtifactCache``, keyed by source content, compiler versions, build
flags and the installed Synapse build. Nothing is keyed by modification time. Building
happens under a file lock, so concurrent ranks on one host build once.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from blake3 import blake3

from resin.kernel_cache import ArtifactCache

PACKAGE = Path(__file__).resolve().parent.parent
NATIVE = Path(__file__).resolve().parent / "native"
KERNELS = PACKAGE / "kernels"
HABANA_INCLUDE = Path("/usr/include/habanalabs")
HABANA_LIB = Path("/usr/lib/habanalabs")

CXX = os.environ.get("CXX", "g++")
CC = os.environ.get("CC", "gcc")
CFLAGS = ("-O3", "-g", "-fPIC", "-std=c11", "-Wall")
TPC_CLANG = "tpc-clang"
CXXFLAGS = ("-O2", "-g", "-fPIC", "-std=c++17", "-Wall", "-Wno-unused-function")
TPCFLAGS = ("-x", "c++", "-march=gaudi2", "-O2")
RUNTIME_LIBS = ("-lSynapse", "-lhcl")

RUNTIME_LIBRARY = "libresin_gaudi_rt.so"
KERNEL_LIBRARY = "libresin_gaudi_tpc.so"
KERNEL_MANIFEST = "kernels.json"
PACK_LIBRARY = "libresin_gaudi_pack.so"

_INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)


@dataclass(frozen=True, slots=True)
class NativeLibraries:
    """Validated native artifacts and the digests recipes are keyed by."""

    runtime: Path
    kernels: Path
    kernel_digests: Mapping[str, str]
    """Kernel name (file stem) to the digest of its sources, headers and build flags."""
    glue_digest: str
    runtime_digest: str
    """The runtime builds every graph, so its source is part of each recipe's key."""
    synapse_build: str


def default_cache_root() -> Path:
    root = os.environ.get("RESIN_GAUDI_CACHE")
    return Path(root) if root else Path.home() / ".cache" / "resin" / "gaudi"


@cache
def _tool_version(tool: str) -> str:
    try:
        out = subprocess.run(
            [tool, "--version"], check=True, capture_output=True, text=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"{tool} is required to build the Gaudi runtime") from error
    return out.splitlines()[0].strip()


TESTED_SYNAPSE = "1.24.1"


@cache
def synapse_build() -> str:
    """Identity of the installed Synapse and graph-compiler build."""
    try:
        version = subprocess.run(
            ["dpkg-query", "-W", "-f=${Version}", "habanalabs-graph"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if version:
            return f"habanalabs-graph-{version}"
    except (OSError, subprocess.CalledProcessError):
        pass
    digest = blake3(max_threads=blake3.AUTO)
    for name in ("libSynapse.so", "libtpc_kernels.so"):
        digest.update_mmap(HABANA_LIB / name)
    return f"synapse-{digest.hexdigest()[:32]}"


def _abi() -> str:
    tpc = re.sub(r"[^A-Za-z0-9.]+", "-", _tool_version(TPC_CLANG)).strip("-")
    return f"gaudi2-{synapse_build()}-{tpc}"


def _dependencies(path: Path, seen: set[Path] | None = None) -> set[Path]:
    """``path`` and every quoted include it reaches, resolved next to the includer."""
    seen = set() if seen is None else seen
    if path in seen:
        return seen
    seen.add(path)
    for name in _INCLUDE.findall(path.read_text()):
        included = (path.parent / name).resolve()
        if not included.exists():
            raise FileNotFoundError(f"{path.name} includes missing {name}")
        _dependencies(included, seen)
    return seen


def _digest(*parts: str | bytes | Path) -> str:
    digest = blake3()
    for part in parts:
        if isinstance(part, Path):
            data = part.read_bytes()
            digest.update(f"{part.name}:{len(data)}:".encode())
            digest.update(data)
        else:
            data = part.encode() if isinstance(part, str) else part
            digest.update(f"{len(data)}:".encode())
            digest.update(data)
    return digest.hexdigest()


def kernel_sources() -> dict[str, Path]:
    return {path.stem: path for path in sorted(KERNELS.glob("*.c"))}


def kernel_digests() -> dict[str, str]:
    tpc = _tool_version(TPC_CLANG)
    flags = " ".join(TPCFLAGS)
    return {
        name: _digest(tpc, flags, name, *sorted(_dependencies(path)))
        for name, path in kernel_sources().items()
    }


def _identity(digests: Mapping[str, str]) -> tuple[str, str, str]:
    glue = _digest(_tool_version(CXX), " ".join(CXXFLAGS), NATIVE / "glue.cpp")
    runtime = _digest(
        _tool_version(CXX),
        " ".join(CXXFLAGS + RUNTIME_LIBS),
        NATIVE / "runtime.cpp",
    )
    kernels = json.dumps(dict(sorted(digests.items())))
    return _digest(runtime, glue, kernels), glue, runtime


def _run(*command: str | Path) -> None:
    result = subprocess.run(
        [str(c) for c in command], capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise RuntimeError(
            f"{' '.join(str(c) for c in command)} failed:\n{result.stderr}"
        )


def _embed(header: Path, objects: Mapping[str, Path]) -> None:
    lines = ["// generated by resin_models_gaudi.synapse.build", "#pragma once"]
    for name, path in objects.items():
        data = path.read_bytes()
        lines.append(
            f"static const unsigned char rg_elf_{name}[] = {{"
            + ",".join(str(b) for b in data)
            + "};"
        )
    lines.append(
        "struct RgElf { const char* name; const unsigned char* elf; unsigned len; };"
    )
    lines.append("static const RgElf kElfs[] = {")
    lines.extend(
        f'    {{"{name}", rg_elf_{name}, sizeof(rg_elf_{name})}},' for name in objects
    )
    lines.append("};")
    header.write_text("\n".join(lines) + "\n")


def _build(scratch: Path) -> None:
    objects = scratch / "k"
    objects.mkdir()
    sources = kernel_sources()

    def compile_kernel(item: tuple[str, Path]) -> tuple[str, Path]:
        name, source = item
        out = objects / f"{name}.o"
        _run(TPC_CLANG, source, "-c", *TPCFLAGS, f"-I{KERNELS}", "-o", out)
        return name, out

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        built = dict(pool.map(compile_kernel, sorted(sources.items())))
    _embed(scratch / "kernels_elf.h", built)
    include = f"-I{HABANA_INCLUDE}"
    glue = NATIVE / "glue.cpp"
    _run(
        CXX,
        *CXXFLAGS,
        include,
        f"-I{scratch}",
        "-shared",
        "-o",
        scratch / KERNEL_LIBRARY,
        glue,
    )
    _run(
        CXX,
        *CXXFLAGS,
        include,
        "-shared",
        "-o",
        scratch / RUNTIME_LIBRARY,
        NATIVE / "runtime.cpp",
        f"-L{HABANA_LIB}",
        *RUNTIME_LIBS,
        f"-Wl,-rpath,{HABANA_LIB}",
    )


def build_native(cache_root: str | os.PathLike[str] | None = None) -> NativeLibraries:
    """Return the validated native libraries, building them on a cache miss."""
    digests = kernel_digests()
    identity, glue, runtime_digest = _identity(digests)
    store = ArtifactCache(cache_root or default_cache_root(), _abi())
    directory = Path("native") / identity[:32]
    names = (RUNTIME_LIBRARY, KERNEL_LIBRARY, KERNEL_MANIFEST)
    manifest = json.dumps({"identity": identity, "kernels": digests}, sort_keys=True)

    def artifacts_valid() -> bool:
        return all(
            store.validate(directory / name, identity=f"{identity}:{name}")
            for name in names
        )

    store.root.mkdir(parents=True, exist_ok=True)
    with open(store.root / "native.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not artifacts_valid():
            logging.getLogger("wright").info("building the kernels for this Synapse version")
            with tempfile.TemporaryDirectory(prefix="resin-gaudi-build-") as tmp:
                scratch = Path(tmp)
                _build(scratch)
                (scratch / KERNEL_MANIFEST).write_text(manifest)
                for name in names:
                    store.publish_from(
                        scratch / name,
                        directory / name,
                        identity=f"{identity}:{name}",
                    )
            if not artifacts_valid():
                raise RuntimeError("published Gaudi native artifacts failed validation")
    root = store.root / directory
    return NativeLibraries(
        runtime=root / RUNTIME_LIBRARY,
        kernels=root / KERNEL_LIBRARY,
        kernel_digests=json.loads((root / KERNEL_MANIFEST).read_text())["kernels"],
        glue_digest=glue,
        runtime_digest=runtime_digest,
        synapse_build=synapse_build(),
    )


def build_pack(cache_root: str | os.PathLike[str] | None = None) -> Path:
    """Return the validated host packing library (``native/pack.c``), building it on a miss.

    It has no Synapse dependency and its own ABI namespace, so loaders and host-only tests
    need neither the SDK nor tpc-clang.
    """
    cc = _tool_version(CC)
    identity = _digest(cc, " ".join(CFLAGS), NATIVE / "pack.c")
    abi = "host-" + re.sub(r"[^A-Za-z0-9.]+", "-", cc).strip("-")
    store = ArtifactCache(cache_root or default_cache_root(), abi)
    artifact = Path("pack") / identity[:32] / PACK_LIBRARY
    store.root.mkdir(parents=True, exist_ok=True)
    with open(store.root / "pack.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not store.validate(artifact, identity=identity):
            with tempfile.TemporaryDirectory(prefix="resin-gaudi-pack-") as tmp:
                out = Path(tmp) / PACK_LIBRARY
                _run(CC, *CFLAGS, "-shared", "-o", out, NATIVE / "pack.c")
                store.publish_from(out, artifact, identity=identity)
            if not store.validate(artifact, identity=identity):
                raise RuntimeError("published Gaudi pack library failed validation")
    return store.root / artifact
