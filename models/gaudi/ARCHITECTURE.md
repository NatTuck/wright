# Gaudi architecture — the native-executor hard rule

This note states the load-bearing execution rule for the Gaudi model package, DeepSeek-V4.1-Flash
first. It exists so a future change cannot quietly reintroduce a Python per-step executor. The gate
that enforces it is [`tests/test_ds41_native_only.py`](../../tests/test_ds41_native_only.py)
(repo root `tests/`); read this note and that test together.

## The hard rule

**A DeepSeek-V4.1 step runs only in the Rust native executor (`resin._resin.Ds41Executor`). Python
loads, builds, and records at startup — it never runs a step.**

Concretely, in the DSv4.1 per-step path there is **no Python**:

- no per-step recipe-launch loop (`.run(lib)`, a Python `model.launch(self, ...)`),
- no per-step Python table fill (`_fill(self, k, tb, tokens, positions, rows)`),
- no per-step Python table-image upload (`dev.h2d_async(self.table, ...)` inside a step method),
- no Python `runner.forward(self, plan)` / `runner.enqueue(self, plan, ...)` step entry,
- no Python launch-cost accounting (`HOST_CALL_S`, `_report_recipe_device`, `_harvest`, `_inflight`).

The Engine calls `Ds41Executor.forward / enqueue / release` directly. `runner.attach` binds
`self.forward = self.executor.forward` (and `enqueue`, `release`) to those native methods; the runner
class itself defines no Python `forward`/`enqueue`.

## What Python is allowed to do (startup only)

- **Load** weights and convert checkpoint formats (`deepseek_v41/weights.py`, `loading/`).
- **Build/compile** the graphs and recipes once per shape (`model.py::_graphs`, `_compile`).
- **Record** native programs once per shape: `model.py::_program` / `_step_program` / `_draft_native`
  emit `resin._resin.GaudiProgram` sections (`native.launch(handle, slots, addresses)`,
  `native.allreduce/allgather`, `native.select/arm`). The nested `def launch(op)` helpers in
  `model.py` are **recording** helpers — they append to a `GaudiProgram`; they do not execute a step.
  `native.py::build_executor` records the I/O program (`upload`, `tokens`, `sample`, `out`, `compact`)
  and constructs the `Ds41Executor`.
- **Build the native fill** (`model.py::_native_fill` → `resin._resin.Ds41Fill`), which fills the step
  tables in Rust. The CPU tests keep a numpy fill of the same tables as a reference.

Recording is allowed because it happens once per shape at startup, not per step. The forbidden
patterns are the *per-step execution* shapes, not the recording shapes; the gate's regexes are written
to tell them apart (`def launch(op)` recording vs `def launch(self` step method; `native.launch(...)`
vs `.run(lib)`).

## Host-Engram on 4 cards

The 188.8 GiB Engram lookup tables do not fit on 4 cards (118.81 GiB/card needed vs ~95 free), so the
TP4 preset leaves them on the host and gathers just the rows a step needs. **The production gather,
dequant, and upload run in Rust** (the native fill's host-Engram branch), under the same hard rule —
no per-step Python gather.

`weights.py::engram_rows` is the **numpy correctness oracle** the Rust gather is gated against, **not
the production path**. It is the branch's optimized gather (pooled `preadv` / resident `mmap` /
auto-mincore / 256×256 f32 LUT dequant), kept bit-identical to that reference; it may be slower than
Rust, which is fine for an oracle. Its API is `(src, layer, rows) -> (n, 256) f32`. The dequant is
f32-scoped (overflow to inf where the f32 expression did); e8m0 scale byte `0xFF` (NaN) is rejected at
load (`_reject_nan_engram_scales`), never substituted (manual ch3 fp8 landmines).

The host-mode device contract is unchanged from card mode downstream: the gathered f32 rows ride the
step-table image (`eg.{layer}` f32 `(T, 6144)`) and feed the same `engram_graph`
(`q8p` → `eg_wkv` GEMM → `rg_eg_gate_f32`). Host mode does **not** launch the card-only
`eg_hash`/`eg_gather`/`eg_gate_g` kernels.

**Correctness constraint (load-bearing):** host decode is synchronous — the next step's hash needs
this step's sampled token on the host. A continuation's one-position-stale bound token must never
reach the host hash nor be recorded in the Engram context (it would poison every later `Row.before`
→ wrong text, no error). The native enqueue path enforces this; it is not a perf regression to
"optimize away".

## Why this rule

The branch `origin/four-card-ds` ran steps in a Python executor (`model.launch` / `runner.enqueue` /
`_fill` / `.run(lib)` / `h2d_async(self.table)` / `HOST_CALL_S=44us`). main 0.3.0 deleted it for the
Rust executor. The Python per-step path is a launch/serialization cost the native executor removes,
and its `HOST_CALL_S` accounting is a measurement artifact of that deleted path — neither is carried
over. Reintroducing a Python step loop regresses the architecture; the gate fails on it (its negative
control runs the same detector against the branch's Python `model.py`/`runner.py` and requires every
forbidden pattern to fire there).

## Enforcing it

- `tests/test_ds41_native_only.py` — CPU-only gate: the step entry is the Rust executor; no per-step
  Python orchestration in `deepseek_v41/model.py` or `runner.py`; negative control proves the gate
  fires on the branch Python executor.
- `models/gaudi/tests/test_ds41_weights.py` — the oracle's dequant bit-identity (all 65,536
  scale/code pairs), the e8m0 NaN-scale rejection, same-shape-different-data gather, pooled==mmap,
  and the residency monitor's fail-safe fallback.