# Persistent kernel-artifact cache

Compiled kernel persistence is an execution-runtime concern, not Engine cache
policy. The Rust control plane does not select compilers or know where generated
libraries live. A Python runner or deployment harness may opt into
`resin.kernel_cache` before importing its accelerator framework.

Keep these stores conceptually separate:

- model checkpoints contain weights;
- the native flat cache pool contains live history and recurrent state;
- the kernel-artifact store contains compiled libraries and tuning records.

Tokenizer files may accompany a model checkpoint. Live tokenizer and
conversation-parser state belongs to the recipe layer, independently of the
device and kernel caches.

## Portable API

`resin.kernel_cache` uses the Python standard library and has no import-time
dependency on PyTorch or an accelerator framework.

```python
from resin.kernel_cache import ArtifactCache

cache = ArtifactCache(storage_root, abi_namespace)
```

The host supplies both values. `abi_namespace` is one safe path component that
should encode every compatibility input relevant to the caller: the accelerator,
Python, driver, framework, compiler, and kernel-library versions.
Changing any such input should select another namespace rather than mutate an
old one.

The main methods are:

```python
cache.validate(artifact, identity=...)
cache.publish(artifact, identity=...)
cache.publish_from(local_source, durable_artifact, identity=...)
cache.load_or_build(artifact, identity=..., load=..., build=...)
cache.load_or_build_from(
    durable_artifact,
    source=local_source,
    identity=...,
    load=...,
    build=...,
)
```

`hits`, `misses`, `root`, `abi`, and `record_path()` expose diagnostics. Paths
must remain inside the selected ABI namespace.

## Validation and publication

An artifact is reusable only when its sidecar record exactly matches the record
schema, ABI namespace, logical identity, relative path, byte count, and current
content digest. The current `ArtifactRecord` schema stores that digest in its
`blake3` field. Legacy schema 1 sidecars are misses; there is no legacy digest
compatibility path. This artifact identity belongs only to the persistent
kernel-artifact store; the live request cache has no content index.

Resin never uses mtime as proof that compiled bytes are reusable. Missing,
malformed, wrong-ABI, wrong-identity, truncated, or corrupted artifacts are
misses and enter the caller's normal build path.

`publish_from` supports building on compiler-friendly local scratch and copying
the finished result into durable storage. It flushes and atomically replaces the
artifact, then publishes the readiness sidecar last. A reader may see a miss
during publication but cannot validate a partial destination. A failed build or
publication does not create a new readiness record.

## Compiler adapters

The generic cache does not turn every compiler directory into a safe portable
cache. Each backend needs an adapter that knows how to build, publish, and load
its artifact.

The Gaudi package builds its kernels and Synapse recipes through `ArtifactCache`,
keyed by their source content and the compiler versions.

## Runtime integration

A cache miss builds in local scratch and publishes the completed artifact
through `ArtifactCache`; later processes validate that specific artifact before
loading it. The Gaudi kernel build holds a file lock, so concurrent processes on
one host build once.

Atomic artifact publication is not a distributed compiler lock. Cold writers on
different hosts that share storage can duplicate work.

Valid cache hits can avoid compilation for covered specializations. They do not
remove framework import, device-context creation, weight loading, graph capture,
or model warmup. A new layout, bucket, or backend may still compile.

## Current limitations

- The portable store is filesystem-backed; an object-store API needs an adapter.
- Concurrent cache misses have no distributed compiler lock.
- A validated compiled artifact says nothing about live cache correctness.
- Similar TTFT values do not prove absence of compilation. That claim requires
  compilation-event instrumentation or before/after artifact evidence, with
  the measured cache paths and process scope recorded.
