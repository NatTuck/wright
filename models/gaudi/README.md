# Gaudi 2 model package

`resin-models-gaudi` runs models on Intel Gaudi 2 with the Synapse graph compiler and custom TPC-C
kernels. For the commands and the results, read the [root README](../../README.md).

## Layout

| Path | Contents |
|---|---|
| `resin_models_gaudi/synapse/` | The native runtime (`native/runtime.cpp`), the kernel glue (`native/glue.cpp`), the build (`build.py`), the device (`runtime.py`), card selection (`cards.py`), and graphs with a recipe cache (`graph.py`) |
| `resin_models_gaudi/kernels/` | The TPC-C sources. The build compiles them into one kernel library. A kernel has the GUID `rg_<file stem>_f32`. |
| `resin_models_gaudi/qwen3_5/` | The Qwen3.8-27B runner, cache arena, and checkpoint preparation |
| `resin_models_gaudi/mistral_large4/` | The Mistral-Large-4 runner, cache arena, DSpark drafter, and weight preparation |
| `resin_models_gaudi/glm5_3_flash/` | The GLM-5.3-Flash runner, cache arena, and weight preparation |
| `resin_models_gaudi/deepseek_v41/` | The DeepSeek-V4.1-Flash runner, cache arena, and weight preparation |
| `resin_models_gaudi/gguf/` | The GGUF reader and block formats |
| `tests/` | The tests. Most need a card |
| `tools/` | Reference models of the numerics, the reference for the DSpark drafter, DeepSeek-V4.1-Flash development tools, and the Qwen3.8 image harnesses (HF tower reference, preprocessing comparison, accuracy set, single-kernel check) |

## How it runs

- **One process holds one card.** A multi-card preset starts one process for each card. Each process
  takes a free card from the list, then binds to the NUMA node of that card.
- **Every graph compiles at startup, and no graph compiles after that.** The key of a recipe holds the
  graph definition, the digests of its kernels, the runtime, the Synapse build, and the variables that the
  compiler reads. The key does not hold device addresses, so a recipe is valid for all sequences.
- **The Engine owns the cache and the batches.** The runner gets a `BatchPlan`, runs launches of the startup
  recipes, and gives back tokens. Sequences keep key and value rows in pages of one device arena.
- **Tensor-parallel ranks run the same plan.** Each rank holds a slice of the wide weights, and the ranks join
  their partial results between launches. All ranks sample the same tokens on their own card.
- **The card does the sampling.** Temperature, top-k, top-p, seeds and logprobs do not stop the launch sequence.

## Speculative decoding

- **MTP (Qwen3.8-27B, GLM-5.3-Flash).** A decode step checks the anchor token and the drafts of each row in
  one pass, and emits the target's sample at each position. The checkpoint supplies the MTP layer.
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
