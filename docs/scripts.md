# Scripts and deployment modules

Run commands from the repository root. The shared entry points are the `wright` command for
serving, `scripts/bench.py` for Engine performance, and `scripts/check_server.py` for endpoint
qualification. Model-specific configuration belongs in `deploy/models.py`.

## Scripts

| File | Purpose |
| --- | --- |
| [bench.py](../scripts/bench.py) | Construct a registered runner on this host's cards; sweep prefill contexts and decode batches through `resin.bench`; save JSONL, plots and Perfetto traces. |
| [sweep.py](../scripts/sweep.py) | Measure prefill and decode by context length with llama-benchy against a real server, and run GPQA Diamond. |
| [batch_sweep.py](../scripts/batch_sweep.py) | Run `bench.py` on real text across batch sizes and draft settings. |
| [fetch_bench_corpus.py](../scripts/fetch_bench_corpus.py) | Fetch the pinned English and code passages for the real-text benchmarks. |
| [plot_bench.py](../scripts/plot_bench.py) | Plot prefill latency versus input throughput and decode interactivity versus output throughput. Actual committed output is the default; acceptance projections are explicit. |
| [merge_bench_traces.py](../scripts/merge_bench_traces.py) | Merge per-rank prefill/decode traces, align timestamp bases and isolate process/flow IDs for Perfetto. |
| [serve_local.py](../scripts/serve_local.py) | Serve one registry model by preset name (`--model KEY`) through the native HTTP server. The `wright` command finds the preset from the checkpoint instead. |
| [check_server.py](../scripts/check_server.py) | Qualify an existing HTTP endpoint: JSON/SSE, logprobs, concurrency, prefix hits, context/TTFT, optional media/reasoning, MMLU subset and stress/recovery. Writes a pass/fail/skip report. |
| [gap_bench.py](../scripts/gap_bench.py) | Start streams that decode on a real server. Send one long prompt. Record the longest gap of each stream and the TTFT of the prompt, for each `--decode-share` and `-b` setting. |
| [stress_http.py](../scripts/stress_http.py) | Standalone bounded HTTP load/protocol/disconnect/recovery checks, plus the reusable client/parser/reporting machinery used by `check_server.py`. |
| [qwen35_mtp_fp8.py](../scripts/qwen35_mtp_fp8.py) | Add the MTP weights of a dense Qwen3.5 checkpoint to its FP8 directory. |
| [bench_startup.py](../scripts/bench_startup.py) | Measure cold `resin` and Torch backend imports in separate fresh processes. This measures Python startup, not model loading or inference. |
| [lfm2_chat.py](../scripts/lfm2_chat.py) | Minimal CPU example: render a fixed LFM2 conversation with the native recipe and stream one Engine completion. |
| [lfm2_serve.py](../scripts/lfm2_serve.py) | Minimal CPU example: serve the Torch LFM2 runner through the native HTTP server on localhost. |
| [rust_build_and_types.py](../scripts/rust_build_and_types.py) | Development build: build the Rust extension with generated stubs, normalize stubs and format them. |
| [normalize_stubs.py](../scripts/normalize_stubs.py) | Add property types, protocol bases and imports missing from generated native bindings. Called by the build script. |
| [cloc.py](../scripts/cloc.py) | Count production Rust/Python code using Tokei, excluding tests and grouping the report by package. |

## Deployment modules

| File | Purpose |
| --- | --- |
| [models.py](../deploy/models.py) | Shared checkpoint revisions/inventories, card and topology defaults, runner and recipe factories, and engine warmup. |
| [serving.py](../deploy/serving.py) | The server: settings, checks before the load, card selection, and one process per card for multi-card presets. |
| [gaudi_cli.py](../deploy/gaudi_cli.py) | The command-line settings that the server and the benchmark share. |
| [logs.py](../deploy/logs.py) | Log setup for the server and its ranks. |
| [checkpoint_cache.py](../deploy/checkpoint_cache.py) | Validate checkpoint file inventories and atomically publish/require completion markers. |
| [systemd/wright.service](../deploy/systemd/wright.service) | A systemd unit for the server. |
| [__init__.py](../deploy/__init__.py) | Package marker for shared deployment imports. |

Reusable numerical tests are in `tests/`, `models/gaudi/tests/` and `models/torch/tests/`.
Performance measurement lives in `resin.bench`, with command-line orchestration in `scripts/`.
