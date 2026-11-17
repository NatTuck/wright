# Distributed execution

`resin.Parallelism` defines rank placement in the Rust core and exposes that
immutable configuration to Python. `resin.backend.topology.ParallelGroups` owns
the Python process-group handles. `resin.backend.distributed` coordinates
Engine steps and eligible decode continuations. Each rank owns a model runtime;
each attention-DP replica has one native request Engine, on its lowest global
rank. The other ranks in that replica follow its driver.

The boundaries are:

| Component           | Responsibility                                                                       |
| ------------------- | ------------------------------------------------------------------------------------ |
| `Parallelism`       | Rust-defined dimensions, validation, rank placement, and partitions                  |
| `ParallelGroups`    | Explicit device groups and Gloo control groups                                       |
| `DistributedEngine` | Phase agreement, normal local Engine steps, empty participation, failure propagation |
| Native `Engine`     | Local admission, requests, pages, leases, output validation, commit                  |
| `DistributedWorker` | Replica commands and model callback invocation                                       |
| `DistributedRunner` | Weight placement, CP partitioning, expert dispatch/combine, device synchronization   |

This preserves the existing checkpoint-to-runtime preparation boundary. Create
groups before preparing model weights and inject the appropriate group into each
prepared runtime. Attention weights use `groups.attn_tp`; prefill query ownership
uses `groups.cp`; experts use the independent `groups.ep` and `groups.expert_tp`
coordinates. Expert payloads travel over `groups.device_domain`, the device group
covering the complete serving world. The coordinator never
infers model weight layouts or uses the default world group for those operations.

Runners declare supported axes in `parallelism_capabilities`, a
[`ParallelismCapabilities`](../resin/backend/topology.py) value with flags for
`attn_tp`, `attn_dp`, `cp`, `ep`, and `expert_tp`. Each runner opts into the axes
its implementation supports. Deployment presets in
[`deploy/models.py`](../deploy/models.py) supply concrete starting topologies.

Attention replication is always allowed; it repeats the same attention
computation without introducing another sharding axis.
These axes compose through the explicit `parallel_groups=` preparation path.
Model dimensions, quantization blocks, and kernel capabilities still constrain
valid layouts. `DistributedWorker` validates a declared capability set collectively before
serving begins. Custom distributed runners may supply the same declaration.

## Rank placement

`world_size` is the total number of ranks. The five configuration fields are
`world_size`, `attn_dp`, `attn_tp`, `cp`, and `ep`, each defaulting to one.
`Parallelism` is immutable and accepts keyword arguments only:

```python
from resin import Parallelism

parallelism = Parallelism(world_size=4, attn_dp=4)
```

Attention TP is explicit; attention replication and expert TP are derived:

```python
attn_replication = world_size // (attn_dp * cp * attn_tp)
expert_tp = world_size // ep
```

All sizes must be positive integers. Both `attn_dp * cp * attn_tp` and `ep` must
divide `world_size`. Attention uses an
`[ATTN_DP, CP, ATTN_REPLICATION, ATTN_TP]` mesh, keeping each attention-TP group
contiguous. The derived replication axis repeats the same attention shard and
receives the same logical plan; it does not create additional request Engines.
Experts use `[EP, EXPERT_TP]` over the same ranks. Expert weights are sharded
across that complete mesh, with no separate expert-replica axis. CP describes
attention token partitioning within those ranks; it is not an additional
multiplier in the expert mesh. For example,
`Parallelism(world_size=8, ep=8, attn_tp=8)` gives attention TP eight and
expert TP one on eight ranks.

`parallelism.coordinates(rank)` returns `(dp_index, cp_index, replication_index,
tp_index)`, and `parallelism.rank(...)` maps those four coordinates back to the
global rank. `parallelism.attn_replication` is the derived replication count;
`groups.attn_replication` contains ranks with the same attention shard at fixed
DP, CP, and TP coordinates. The runner owns any device communication across
that group.

For `Parallelism(world_size=8, ep=4, cp=2, attn_dp=2, attn_tp=2)`, both
attention TP and expert TP are two, and attention replication is one. The groups
containing rank zero are:

| Group                  | Ranks      |
| ---------------------- | ---------- |
| Attention TP           | 0, 1       |
| CP                     | 0, 2       |
| Attention DP           | 0, 4       |
| Attention replication  | 0          |
| EP                     | 0, 2, 4, 6 |
| Expert TP              | 0, 1       |
| Request replica        | 0, 1, 2, 3 |
| Serving domain         | 0–7        |
| Device request replica | 0, 1, 2, 3 |
| Device serving domain  | 0–7        |
| Domain drivers         | 0, 4       |

`Parallelism(world_size=4, attn_dp=4)` gives four attention replicas with
attention TP one, CP one, and expert TP four. Each rank drives its own request
Engine while the experts span all four ranks.

`Parallelism(world_size=8, attn_tp=2)` gives attention replication four and
expert TP eight. There is one request replica and one driver for all eight
ranks. Its four attention-TP groups receive the same logical plan. In general,
each request replica contains `cp * attn_replication * attn_tp` ranks, and there
are exactly `attn_dp` request Engines.

The complete expert mesh connects every attention replica, so all drivers
synchronize their serving loops. This also holds with `ep=1`, where expert TP
spans the entire world. A replica with no requests participates in empty
execution while its peers have work. The serving domain is the complete world;
no connected-component inference is needed.

The Rust `Parallelism` value contains numeric configuration and rank layouts;
Python `ParallelGroups` creates and owns the communication handles.
`ParallelGroups.create(parallelism)` is collective over an initialized default
world group, checks agreement on configuration/backend, creates groups in
deterministic order, and reuses identical groups. Its `.topology` field refers
to the supplied `Parallelism`. Device collectives default to NCCL; control
collectives use Gloo. The CPU conformance test uses Gloo for both.
`groups.close()` destroys only the groups it created, after workers stop; the
caller owns the default world.

Each attention-DP replica still has its own native scheduler, request queue, and
logical cache pool. Request limits, token budgets, and cache-parent capacity are
per replica; `attn_dp` does not scale one shared cache pool. The runner supplies
a logical cache layout backed by every rank in that replica. Attention TP can
reduce per-rank KV head storage; CP currently retains complete KV copies and
does not multiply logical capacity. The native configuration type does not add a global
request router or move the current Python serving coordinator into Rust. The
HTTP bridge below adds explicit replica routing above these local engines.

## One coordinated step

Drivers in the same world call `has_work()`, `step()`, and `close()` in matching
order. `submit()` and `cancel()` remain local operations. Applications can drive
this loop directly or use the HTTP coordinator to route ingress and keep all
drivers stepping together.

1. Drivers exchange read-only `Engine.next_phase()` proposals. When replicas
   propose different phases, drivers alternate prefill and decode. The proposal shares the native admission
   fallback used by actual dispatch and reserves nothing.
2. Each driver calls `Engine.step(phase=agreed_phase)`. A mismatched local cohort
   remains queued. Cache pressure and retraction still belong to the native Engine.
3. Actual plans rendezvous with empty or failed preparation on other drivers.
   This second exchange determines whether there is real work, computes logical
   replica capacities, and catches plan serialization errors before execution.
4. Each driver broadcasts its logical plan to all ranks in its request replica. All workers
   validate inputs and vote readiness before calling the model. An idle replica
   receives `plan=None` and participates in the model's empty execution path.
5. Workers execute or enqueue the forward callback and exchange execution status.
   The native Engine validates and commits completed results, including NumPy
   [logprob columns](logprobs.md). Eligible decode steps can enqueue the next
   continuation before collecting the previous completion.
6. Drivers exchange commit status before returning events. A failure permanently
   poisons every native Engine in the domain and stops further execution.

Applications retain the same `step()` API. The Engine owns pending plans and
cache leases during overlap; workers retain per-rank completions and drain them
before resource release or close. Model tensor forwards do not schedule batches.

Replica commands also propagate resource binding and release. Page allocation
exists only on the driver; followers bind their device arenas to the same logical
pool identity and receive its page tables. Request IDs and pool IDs are local to
the request replica; cross-replica device metadata must preserve that namespace.
Bind and release callbacks are replica-local commands and must not issue
collectives spanning other drivers. Cross-replica collectives belong in forward.

Replica transport retains request state by sequence ID. The first scheduled plan
registers each request's sampling policy and prepared media on the followers.
Native media uses a versioned descriptor and separate numeric buffers; subsequent
prefill chunks and decode steps reuse the exact same follower object. Completion,
cancellation and retraction release that state after outstanding work has drained.
Plans already in flight retain their own references and array storage.

Every rank of a request replica reaches the others through a `StepTransport`
(`resin/backend/step_transport.py`): replica commands, the readiness gate before
the first device collective, the execution status after a step, the status of
bind, release and metrics commands, and the end of each engine step. Startup
checks and driver votes do not use the transport. There are two transports:

- `GlooStepTransport` uses the Gloo broadcasts and votes that this section
  describes. It is the default of `DistributedWorker`.
- `SocketStepTransport` (`resin/backend/socket_transport.py`) uses one Unix socket
  between the driver and each follower (`resin._resin.StepChannelDriver` and
  `StepChannelFollower`). A command is one frame per follower, in the same layout
  as the Gloo command. For a forward, each follower sends PREPARED and waits for
  GO, so no rank enters a device collective alone. After the step, each follower
  sends ACCEPTED and continues; only the driver waits for it, before its enqueue
  returns. Bind, release and metrics commands get one DONE per follower. A
  successful engine step sends nothing more. A failed step sends a finish command
  with the error, as the Gloo transport does.

`create_step_transport(groups, mode, timeout=...)` makes the transport on every rank
of a replica. `socket` needs all ranks of the replica on one Linux host and one
replica per communication domain (attention DP 1); `auto` uses it when those
conditions are true, and Gloo when they are not. The driver listens in a new
private directory and sends the path and a random token to the followers through
Gloo. The driver accepts only followers that send the token and run as the same
user, then removes the path. A rank that cannot start the channel stops every rank
of the replica. When a process exits, the kernel closes its sockets, and the next
read or write of each other rank fails in less than a millisecond. Each wait has a
deadline and checks for Python signals every 100 ms. Before a wait blocks, it polls
the socket for up to 200 microseconds (`RESIN_STEP_CHANNEL_SPIN_US`, 0 to block at
once), because the next frame of a step usually arrives within that time. Multi-card Gaudi serving uses
`create_step_transport` with `--step-transport` (default `auto`).

Successful step commands use typed Gloo broadcasts: a fixed integer header and
a native plan archive. The archive schema is derived directly from Rust's
`BatchPlan` and cache-group types using rkyv; Python does not repeat their field
lists, sizes or alignment rules. Request sampling, media and derived positions
are excluded from the archive. Receivers validate the archive, attach retained
request state, and apply the ordinary plan validation before execution. Restored
arrays own their storage, so subsequent receives cannot overwrite an in-flight
plan. The private format is versioned and requires the same Resin build on every
rank. This is not a zero-copy receive path or a persistent storage format.
Successful step commands do not use pickle or object collectives. Media buffers
and JSON request descriptors are sent only on registration, never on repeated
decode steps. Request policy and media stay unchanged until release. Distributed
sampling `extra` values must be JSON data, and distributed media must use Resin's
native payloads. Local execution accepts custom Python `MediaInput`
implementations with the same required cache identity and token-span fields.

## Model callback

A CP, EP, or attention-DP runtime implements:

```python
def forward_distributed(
    self, plan: BatchPlan | None, context: StepContext
) -> ForwardOutput | None:
    # Model-owned CP partitioning and expert communication, including plan=None.
    # Return ForwardOutput for a local batch; idle ranks may return None.
    ...
```

`context` carries the local groups, agreed phase, step ID, local logical sequence
and token counts, and `(sequence_count, token_count)` for each replica in driver
rank order. `max_num_sequences` and `max_num_tokens` describe logical per-replica
capacity. The model derives physical shard sizes and graph buckets from them.
CP groups are static across phases. The runner uses `context.phase` to shard
attention tokens across CP ranks during prefill and replicate attention
execution across those ranks during decode. The coordinator broadcasts the same
logical plan within a replica; phase-specific CP behavior and expert routing
belong to the model callback.

An optional `validate_distributed(plan, context)` performs host checks before the
readiness vote. If it returns a non-`None` prepared value, the worker passes that
value to the execution callback as the keyword-only `prepared` argument. This
lets runners reuse validated metadata without retaining pending state.

The synchronous forward callback must preserve collective ordering and finish
its device work before returning. A runner can instead provide
`enqueue_distributed(plan, context, *, prepared=..., feedback=...)` for deferred
completion and eligible decode continuations. It returns an owned completion
with `result()` and device feedback for queued work, or the ordinary synchronous
result on fallback. The Engine retains the plan and cache leases until
completion, and every replica must enqueue together. See
[overlap scheduling](scheduler.md) for the completion contract.

Mid-collective process/device failures rely on
process-group timeouts and process teardown. The coordinator propagates errors;
it cannot roll back in-place model state or globally undo a peer's completed local
commit. Failure marks all native Engines poisoned, preserving resources and
preventing cleanup/reuse through normal Engine operations.

TP-only runtimes can implement ordinary `forward(plan)` whenever
`cp=ep=attn_dp=1`, including attention replication greater than one. They use
the same `DistributedWorker` and `DistributedEngine` lifecycle; the runtime
owns weight preparation and communication across its groups.

## Process integration

For HTTP serving, every attention-DP driver constructs the same server and
calls `serve()`. Non-driver ranks stay in the existing worker loop:

```python
worker = DistributedWorker(runner, groups)
if groups.is_driver:
    engine = DistributedEngine(worker, max_running_requests=256)
    HttpServer(engine, recipe=recipe, model=model_name).serve(host="0.0.0.0", port=8000)
else:
    worker.serve()
```

Only the first driver binds a listener. Requests select an attention-DP replica
using `routed_dp_rank` in JSON or the `resin-routed-dp-rank` header. The header takes
precedence; invalid or out-of-range ranks return HTTP 400. Unassigned requests
use round robin. Each selected driver owns the actual native request lease,
recipe, media preprocessing, incremental parser, scheduler, and cache. Client
disconnects and stalled response channels cancel the request on its owner.
Generated and prompt logprobs travel through the same response bridge.

The built-in chat page chooses a random replica once per page session and sends
that rank on every turn. This preserves prefix and image-cache locality for a
conversation; it does not alter automatic routing for other clients. The page's
first-token timer includes upload, queueing, preparation and inference, and does
not identify any one of those phases.

Set `HttpServer(..., stream_interval=N)` to batch later streaming output until N
generated model tokens accumulate, checked at step boundaries after the first
semantic output. Completion flushes the remainder. It defaults to 1 and must be
the same positive integer on every driver, along with the model and route
configuration. Every driver's native
session uses that interval; coordinated inference, stop matching, and
cancellation continue on every step. Text and score records stay aligned.

`DistributedHttpCoordinator` exchanges ingress before each coordinated step,
including steps where a replica is idle. Native `_HttpSession` instances reuse
the ordinary HTTP lifecycle on each driver; the frontend retains bounded client
channels and receives native event packets. Session initialization,
admission infrastructure, execution, and parser errors are voted before the
next phase. Request rendering failures remain local request errors. Frontend
bind failure or shutdown closes waiting driver loops and model followers.
Ingress and events remain typed Rust packets while crossing Python. Remote
packets use Rust's existing JSON encoding over Gloo byte buffers; local packets
need no serialization, and empty packets carry no payload. Fixed integer votes
exchange lengths, shutdown state, and errors before payload collectives. Python
does not decode requests or events into dictionaries on this path.

For direct engine integration, applications can continue to own local
admission and event consumption explicitly:

After initializing distributed communication and preparing each rank's runtime:

```python
worker = DistributedWorker(runner, groups)
if groups.is_driver:
    engine = DistributedEngine(worker, max_running_requests=256)
    try:
        # Admit this replica's requests here. Every driver stays in the same loop.
        while engine.has_work():
            events = engine.step()
            consume_local_events(events)
    finally:
        engine.close()
else:
    worker.serve()
# Once all ranks have left their serving loops:
groups.close()
```

### Gaudi ranks

Gaudi runners have no Torch device and no NCCL. The default world and every group run on
Gloo: `ParallelGroups.create(topology, device_backend="gloo")`. The runner owns its device
communicator. `resin_models_gaudi.mistral_large4`, `glm5_3_flash`, and `deepseek_v41` make one HCCL communicator per
attention replica. Rank 0 of the replica creates the HCCL id and broadcasts it over the replica's Gloo
group. The ranks join their partial results between the step's launches.

- **Plain `forward(plan)`.** The runner is TP-only, so every rank receives the same plan,
  runs the same launches and collectives, and samples identical tokens from the all-gathered
  logits on its own device. No token broadcast is needed.
- **Timeouts.** Loading takes minutes per rank, so the Gloo timeout must outlast it.
- **Launching.** `scripts/serve_local.py` spawns one process per card for multi-card Gaudi
  presets. Each process takes a free card, and binds to the card's NUMA node.

Initialization and error diagnostics may still use object collectives;
successful execution votes, plan transport, and HTTP rounds use typed buffers.
Shared-memory command
transport, expert load balancing, and prefix-aware replica routing remain
separate work.

## Replicas of a one-card model

A server can also serve a model whose Engines run in other processes, one per card
(`crates/resin-py/src/http_replicas.rs`). These replicas are not attention-DP
replicas: they do not join a distributed step, and they do not wait for each other.

- The frontend process makes a `_ReplicaRouter` on a socket path, and
  `HttpServer.register_replicas(router, model=...)` serves its model beside the
  frontend's own model, which can be a `DistributedEngine`.
- Each replica process makes an Engine and a `_ReplicaServer` with the path and the
  run token. The replica joins the router as the step channel's followers do (the
  token, the same user), and sends READY with its model's metadata. Every replica
  must send the same metadata.
- A router thread sends each request to one replica in an INGRESS frame (`HttpIngress`
  JSON). The replica admits it into its own HTTP session, steps its Engine, and sends
  EVENTS frames (`HttpEvents` JSON) after each step that has events. Cancellations and
  shutdown travel in INGRESS frames.
- The router does not block on a replica. Each replica has its own queue of INGRESS
  frames, and the router writes it only when that replica's socket takes more. So a
  replica that reads slowly does not stop the router from reading the events of the
  others. A replica that reads none of its queue for 120 s fails.
- The router keeps the turns of a conversation on one replica: rendezvous hashing of a
  conversation key over the serving replicas. The key is the request's
  `prompt_cache_key`, `user`, or Anthropic `metadata.user_id`, else a hash of the tools
  and the messages up to the first user message. When that replica has more than
  `max(2, batch / 8)` more requests than the least busy one, the conversation moves to
  the least busy replica.
- A replica whose Engine fails sends FAILED. The router ends that replica's requests
  with the error, logs the error, removes the replica's metrics, and sends new requests
  to the others. When no replica serves, new requests fail at once. An EVENTS frame that
  is not valid, or metrics that are not valid, fail that replica only. A server that
  stops after a replica failed still exits normally.
- Once a second the router asks each replica for its engine metrics and publishes them
  as its own part of `/metrics`, with the replica number in `attn_dp_rank`. When the
  replicas' Engines record metrics, the replicas send the first-token event, so the
  router records the time to the first token.

`deploy.serving.main_layout` uses this for `wright` with one `--device` list for each
model: the first model spans its cards, and each other model runs one replica per card.

### Replicas on the cards of a multi-card model

A replica can also share its card with a rank of the first model (not rank 0, which
serves HTTP). A card can belong to only one process, so the replica runs in that rank's
process:

- The rank takes the card of its own module (`claim_card(module=...)`), because the
  replica of that card has a fixed index.
- The replica loads first, on a runtime context of its own (`Device.context()`): a
  stream with its own workspace and staging buffers. Then
  the rank's model loads and sizes its KV cache from the memory that the replica leaves.
  A replica can fill `--gpu-memory-utilization` of the card (default 0.25).
- The rank follows its driver on the main thread. A second thread serves the replica,
  with the replica's context current.
- A card gate (`resin_models_gaudi.synapse.card_gate.CardGate`) decides whose step comes
  next on the card. The rank's transport (`GatedTransport`) tells the gate when a step
  of the first model starts and when its work is on the device. The replica's
  `_ReplicaServer(gate=...)` calls `gate.acquire(0.02)` before each Engine step and
  `gate.release()` after it. When the gate gives no turn, the replica reads the
  router's frames and asks again, so its socket does not fill.
- The first model goes first. A replica step starts when the first model has had no
  step on the card for 5 ms and its last step is done on the device. With a `--share`
  weight above 0, a replica step can also start while the first model is busy, until
  the replica has used its share of the card's time.
- The gate keeps the steps of the two models whole on the device, in order, with
  device events: a replica step in the first model's busy time waits on the device for
  the first model's step before it, and the next step of the first model waits on the
  device for that replica step. The host does not wait for either.

## Source references and validation

The design follows DeepInfer's explicit device/CPU group separation in
[`parallel_state.py`](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/parallel_utils/parallel_state.py),
its read-only phase vote followed by an actual-scheduling vote in
[`scheduler.hpp`](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/csrc/engine/scheduler.hpp#L299),
and model-level empty execution in
[`worker.py`](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/worker/worker.py).
Resin uses explicit injected groups. Its native Engine owns pending plans and
cache leases, including when a decode continuation overlaps completion of the
previous step.

[`test_distributed.py`](../tests/test_distributed.py) launches eight real processes
and checks numerical TP/CP reductions and variable-sized EP dispatch/combine,
including empty senders, mixed phases, unequal request queues, cancellation,
array score output, shutdown, idle DP replicas sharing expert TP, TP-only
compatibility, and failures in initialization, preparation, serialization,
preflight, execution, output validation, release, and control failure after model
execution. These are CPU Gloo conformance tests; device kernel and multi-node
behavior are not established by them.

[`test_distributed_overlap.py`](../tests/test_distributed_overlap.py) checks
coordinated continuations across replicas, including idle participation and
synchronous fallback.
