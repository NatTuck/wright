# Gaudi 2 model package

`resin-models-gaudi` runs models on Intel Gaudi 2 with the Synapse graph compiler and custom TPC-C
kernels. For the commands and the results, read the [root README](../../README.md).

## Layout

| Path | Contents |
|---|---|
| `resin_models_gaudi/synapse/` | The native runtime (`native/runtime.cpp`, with graph building in `native/graph.cpp`), the kernel glue (`native/glue.cpp`), the build (`build.py`), the device (`runtime.py`), card selection (`cards.py`), and graphs with a recipe cache (`graph.py`) |
| `resin_models_gaudi/kernels/` | The TPC-C sources. The build compiles them into one kernel library. A kernel has the GUID `rg_<file stem>_f32`. |
| `resin_models_gaudi/qwen3_5/` | The Qwen3.8-27B runner, cache arena, and checkpoint preparation |
| `resin_models_gaudi/mistral_large4/` | The Mistral-Large-4 runner, cache arena, DSpark drafter, and weight preparation |
| `resin_models_gaudi/glm5_3_flash/` | The GLM-5.3-Flash runner, cache arena, and weight preparation |
| `resin_models_gaudi/deepseek_v41/` | The DeepSeek-V4.1-Flash runner, cache arena, and weight preparation |
| `resin_models_gaudi/qwen4_exp/` | The Qwen3.8-Flash-Next runner (in progress: 4 or 8 cards, dense attention within the QSA indexer's 2048-token budget), cache geometry, n-gram hashing, and checkpoint preparation |
| `resin_models_gaudi/gguf/` | The GGUF reader and block formats |
| `tests/` | The tests. Most need a card |
| `tools/` | Reference models of the numerics, the references for the DSpark and DFlash 2 drafters, and DeepSeek-V4.1-Flash development tools |

## How it runs

- **One process holds one card.** A multi-card preset starts one process for each card. Each process
  takes a free card from the list, then binds to the NUMA node of that card.
- **Every graph compiles at startup, and no graph compiles after that.** The key of a recipe holds the
  graph definition, the digests of its kernels, the runtime's graph building (`native/graph.cpp` and
  `graph.h`), the Synapse build, and the variables that the compiler reads. The key does not hold device
  addresses, so a recipe is valid for all sequences. A change to the rest of the runtime (`native/runtime.cpp`:
  streams, memory, copies, launches) does not change a recipe, so the recipes do not compile again.
- **Work issues on a context.** A context is a stream with its own workspace and pinned staging buffers
  (`Device.context(affinity)`). Each thread
  issues on its current context (`Device.context()`, `DeviceContext.use()`). A thread starts on the default
  context, so a process with one model uses only that context. A process that serves several models gives each
  one a context, so that two models' copies do not share staging buffers. A recipe
  takes its workspace from the context that loads it. `Device.stream_wait(slot)` makes the current context wait on the device for an event that another
  context recorded. The host does not wait. Contexts share the card's time.
- **The Engine owns the cache and the batches.** The runner gets a `BatchPlan`, runs launches of the startup
  recipes, and gives back tokens. Sequences keep key and value rows in pages of one device arena.
- **Tensor-parallel ranks run the same plan.** Each rank holds a slice of the wide weights, and the ranks join
  their partial results between launches. All ranks sample the same tokens on their own card.
- **The card does the sampling.** Temperature, top-k, top-p, seeds and logprobs do not stop the launch sequence.
- **DeepSeek-V4.1's steps run in a native executor.** Python loads the model and builds its step programs at
  startup. Then `resin._resin.Ds41Executor` (`crates/resin-py/src/gaudi_ds41.rs`) runs every step without
  Python: it writes the step's tables (`resin_gaudi::ds41`, also prefill chunks and DSpark drafts), uploads them
  in one copy, runs the step program, samples on the card, and reads the samples behind events. It also scores
  prompt logprobs, accepts DSpark drafts, and keeps the Engram context (`Ds41Context`). The Engine calls the
  executor's `forward`, `enqueue` and `release` directly. With `--step-transport native`, rank 0's executor
  sends each runner call to the other ranks (`GaudiRanks`, the same as Qwen3.5/3.8), and no rank uses Python or
  Gloo for a step. The CPU tests keep the numpy fill and the Python launch loops that the executor replaced
  (`tests/ds41_reference.py`), and the native code must give the same bytes and calls.
- **GLM-5.3-Flash's steps run in a native executor.** `glm5_3_flash/native.py` builds every step program
  (each decode bucket, with and without the indexer and the sparse attention, and the verify buckets), the MTP
  passes and the DFlash 2 drafter's programs at startup. Then `resin._resin.GlmExecutor`
  (`crates/resin-py/src/gaudi_glm.rs`) runs every step without Python: it writes the step's tables
  (`resin_gaudi::glm`), uploads them in one copy, runs the step program, samples on the card, and reads the
  samples behind events. It also scores prompt logprobs, accepts the MTP and DFlash drafts, commits the
  accepted tokens' KDA updates, runs the MTP chain and the drafter's block steps, and chooses for each step
  whether to draft (`resin_gaudi::draft_choice`, the measured cost of each kind of step). With
  `--step-transport native`, rank 0's executor sends each runner call to the other ranks (`GaudiRanks`)
  before it waits for the last step's samples. It sends its draft decision with the signal that starts the
  step's collectives. Thus every rank settles the last verify step at the same time, and the MTP chain starts
  about as soon as with Gloo. A follower waits for a verify step's samples outside a layout's card gate. The
  CPU tests keep the numpy fill and the Python launch loops that the executor replaced
  (`tests/glm_reference.py`). `RESIN_STEP_TRACE=<dir>` writes a timed trace of each step's host events on
  each rank, for studies of the host path.
- **One step program for every model.** `GaudiProgram` (`resin_gaudi::step`) holds a step's stream calls:
  launches with their bindings, collectives, copies, uploads, downloads, memsets and event records. A
  step writes a few registers (counts, lengths, addresses), and `take` and `select` ops choose which
  calls it makes from them. `calls` records the calls that `run` makes, so a test compares a model's
  launch loop with its program without a card. These steps use it: every DeepSeek-V4.1 step, every
  Qwen3.5/3.8 step without `oldqwen`, every GLM-5.3-Flash step and MTP pass, and every Qwen3.8-Flash-Next
  step. For Qwen3.8-Flash-Next, `RESIN_GAUDI_PROGRAM=0` keeps the Python launch loops, for a comparison on
  one build.
- **Qwen3.5/3.8 KV pages are bf16 or FP8.** `--cache-type-k` and `--cache-type-v` (`wright -ctk`, `-ctv`)
  set the type of the K pages and of the V pages of the full-attention layers and the MTP layer
  (`qwen3_5/cache.py`). A bf16 page is 128 rows of 512 bytes, one key a row. An FP8 page is 65 rows: two
  keys' 256 E4M3 values a row (each key scaled by its largest magnitude over 240, rounded up to a bf16
  value), then a row of the 128 scales. `kv_store8` writes a chunk's keys, and its `kv_rows` entry for a key
  of an FP8 page is the key's 256-byte unit (2 x page row + slot). `kv_gather8` dequantizes the pages of a
  slot launch into the same bf16 slab that `kv_gather` fills, so the MME and the softmax kernels do not
  change. With two bf16 types, the graphs and recipes are the same as before. The drafters' caches (DFlash 2,
  DSpine) stay bf16.
- **Optional features are off unless the command line names them.** `wright -f NAME[,NAME...]` (and
  `--features` of `deploy/serving.py` and `scripts/bench.py`) writes them to `RESIN_GAUDI_FEATURES`, so the
  ranks get them too. `-f help` shows the list (`resin_models_gaudi/features.py`).
  - `oldqwen`: the Python executor of `qwen3_5/runner.py` and `qwen3_5/model.py` makes each Qwen3.5/3.8 step,
    on one card only. Without it, the native executor (`resin._resin.QwenExecutor`, its step programs from
    `qwen3_5/native.py`) makes them. Tensor parallel Qwen (`--ranks 2` or more, Qwen replicas over several
    cards) and `--step-transport native` need the native executor. Its wide attention launches and its prefill
    launches without a mask exist only in it.
  - `skipvalidation`: skips the check of the tensor bindings of each launch. On DeepSeek-V4.1 it measured
    +1.4% decode at 8 streams and +2.9% at 1 stream, before the native step. Use it only with a build that you
    trust. Do not use it while you change graph bindings.

## Speculative decoding

- **MTP (Qwen3.8-27B, GLM-5.3-Flash).** A decode step checks the anchor token and the drafts of each row in
  one pass, and emits the target's sample at each position. The checkpoint supplies the MTP layer.
- **DFlash 2 (Qwen3.8-27B).** A separate block-diffusion drafter of five layers makes 7 draft tokens for each
  row of a decode step, of which the step verifies up to 7. Every target pass feeds the residuals of five of its layers to the drafter's cache,
  which keeps the last 2048 positions. After a step accepts its drafts, one block step makes the next 7. The
  drafter's own keys of a block are in scratch pages, so a step writes the cache only at the positions it
  verifies. Refer to `common/dflash.py`.
- **DSpark (Mistral-Large-4).** A separate drafter makes blocks of 3 tokens, for a maximum of 4 rows. A larger
  batch runs without drafts.
- **DSpark (DeepSeek-V4.1-Flash).** The DSpark layers of the checkpoint make a maximum of 5 draft tokens, for a
  maximum of 2 rows. A larger batch runs without drafts.

Greedy and seeded decoding gives the same tokens as decoding without drafts, because each emitted token is
the target's sample at its position. A rejected draft costs time, and does not change the output.

With Mistral-Large-4, the draft build uses different graphs from the build with no drafts. To compare tokens,
turn off speculation on the draft build.

## Checks

| Check | What it shows | Command |
|---|---|---|
| Kernel and runner tests | Each kernel against a reference, and batches against rows alone | `uv run --no-default-groups --group gaudi-dev python -m pytest models/gaudi/tests` |
| Server check | The HTTP APIs, stops, logprobs, context limits, prefix cache, and a seeded MMLU sample | `uv run scripts/check_server.py URL` |
| Reference models | The numerics of GLM-5.3-Flash and DeepSeek-V4.1-Flash on a CPU | `tools/glm_ref_hf.py`, `tools/glm_np.py`, `tools/ds41_ref.py` |

The tests that need a checkpoint read `RESIN_GAUDI_QWEN35`, `RESIN_ML4_GGUF`, `RESIN_ML4_TOKENIZER` and
`DS41_CHECKPOINT`. The test
`test_gguf.py` reads `RESIN_GGUF_PY` for the `gguf` package of llama.cpp. A test skips if its variable is not set.
