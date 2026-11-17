# Changelog

## 0.3.0 (2026-11-16)

The rates are tok/s. "1 stream" is one request at a time.

### License

- **AGPL-3.0-only, or a commercial license from the author.** This includes the TPC-C kernels, which were
  PolyForm Noncommercial 1.0.0. v0.1.0 and v0.2.0 keep their licenses. Refer to [License](README.md#license)
  and [`COMMERCIAL.md`](COMMERCIAL.md).

### Upgrade notes

- **Mistral-Large-4 moves to In Testing.** Wright 0.3.0 does not test it again. The next tests will use a
  checkpoint with MXFP4 experts.
- **The first start rebuilds all kernels and graph recipes, one time.**
- **A Gaudi error in a step stops the engine.** Before, the engine tried the step again.

### Qwen3.8-Flash-Next

- **New: early support for Qwen3.8-Flash-Next** (FP8 safetensors, 4 cards, 🔴 Bleeding Edge). It is text only,
  contexts go up to 2,048 tokens, and the server does not score prompt logprobs. It has no server check or
  throughput sweep yet.

### Qwen3.8-27B and Qwen3.5

- **A native executor runs every step.** Long contexts gain the most. Each value is one card, prose and code
  alike:
  - Qwen3.8-27B: prefill 32K 3,662 → 4,302, 128K 1,814 → 2,526 (+39%). Decode with no drafts: 32K x4 117 → 156,
    32K x8 176 → 219, 128K x1 31 → 41, 128K x4 51 → 76 (+49%). 2K and 8K: no change to +9%.
  - Qwen3.8-27B with drafts, code: MTP-2 32K x4 190 → 279, 128K x4 97 → 145. DFlash 2 32K x4 264 → 321,
    128K x4 161 → 219. DSpine 32K x8 324 → 387, 128K x1 164 → 203. With MTP-2 on prose: 32K x8 200 → 277.
  - On prose with 4 to 32 requests, DFlash 2 decodes up to 16% slower than no drafts, and DSpine up to 5% slower.
    MTP-2 is the best choice for prose.
  - Qwen3.5 0.8B to 9B: prefill 128K +45 to +51%. Decode 128K x4 +46 to +53%, 32K x4 +25 to +32%.
    Qwen3.5-0.8B also decodes 16 to 21% faster at 2K and 8K.
- **New: an FP8 KV cache, `-ctk fp8 -ctv fp8`.** The default stays bf16. On one card, Qwen3.8-27B holds
  1,629,184 tokens of KV cache with FP8, against 836,608 with bf16 (+95%), so 8 sequences of 128K fit
  (90.6 tok/s). `-ctk fp8` alone gives +32%. The costs:
  - Decode with 4 or 8 sequences is 2 to 4% slower up to 32K. At 128K it is 1 to 2% faster. Prefill does not
    change.
  - The logprobs change a little. Against bf16, the top token is the same at 94 to 99% of the positions, and a
    greedy output can split from the bf16 output after a few tokens. FP8 keys cause almost all of the change.
- **Fixed: MTP-2 at 8 sequences could give different tokens from run to run.** The same fault could also change
  plain decode at 11 or more sequences past 2,048 tokens of context. The attention kernels now give the same
  result each time.

### Speculative decoding

- **New: the DSpine drafter for Qwen3.8-27B** (up to 15 draft tokens, `-md`). 1 stream: edits 63 → 237,
  JSON 64 → 185, code 64 → 139. Its fp8_inc copy prepares in 38 s, against 132 s for bf16.
- **New: the DFlash 2 drafter for Qwen3.8-27B** (up to 7 draft tokens). 1 stream: edits 63 → 226, JSON
  64 → 213, code 64 → 168.
- **New: the DFlash 2 drafter for GLM-5.3-Flash** (4 cards). 1 stream: edits 99 → 283, JSON 99 → 224, code
  99 → 217.
- **Qwen3.8-27B uses drafts only where they give more tokens** (`RESIN_QWEN_DRAFT_POLICY=auto`, the
  default). DFlash 2 at 32 requests: 419 → 628. At 16 requests: 389 → 491.
- **GLM-5.3-Flash selects drafts for each step**, from the acceptance of each sequence
  (`RESIN_GLM_DRAFT_POLICY=auto`, the default). DFlash 2: 16 code streams 210 → 364, 8 mixed streams
  141 → 306, 1 story stream 80 → 97. With MTP, 16 streams 310 → 630, because wide steps now decode plainly.
- **GLM-5.3-Flash tries drafts again sooner after a run of plain steps.** Before, a sequence could stay on
  plain steps after its text changed to text that the drafter predicts well. On 4 cards, sampled, at 2K and 8K
  context:
  - Code: MTP-3 with 1 stream 102 → 122 and 116 → 120, with 4 streams +4 to +7%. DFlash 2 with 1 stream
    92 → 115 and 92 → 118, with 4 streams 2 to 6% slower.
  - English: MTP-3 with 1 stream at 8K 82 → 100, with 4 streams at 8K 6% slower (221 → 208). DFlash 2 is 1 to
    4% slower.
  - With 8 streams and in prefill, there is no change.
- **Decode continuations cross KV blocks.** Qwen3.8-27B at 8 requests: +7 to +10% with DFlash 2, +4 to +7%
  with MTP-2.
- **Faster verify steps with a drafter.** The card accepts the drafts without a host wait: DFlash 2 +7% at
  1 request. A faster softmax in the Qwen3.8-27B verify and drafter blocks: +2 to +3% at 1 request, +4.5 to
  +6.5% at 8.
- The greedy output can change when the server changes between verify and plain steps.

### Batch size

- **Qwen3.8-27B, GLM-5.3-Flash and DeepSeek-V4.1-Flash batch up to 64 sequences** (16 or 32 before).
  Qwen3.8-27B at 8K context: 635 at 32 rows → 791 at 64 (+25%). On 8 cards at 2K context: GLM-5.3-Flash 613 at
  16 rows → 874 at 64, DeepSeek-V4.1-Flash 615 → 851.

### GLM-5.3-Flash

- **A native executor runs every step.** Rust fills the step tables (dense slots, indexer slots, the sparse
  block list, the KDA records and the pool tails), runs the step programs, samples on the card, and accepts
  the drafts. MTP passes, the commit of the accepted tokens, and the DFlash 2 drafter's injections and block
  steps are native too. Python runs only at startup. The output is the same as the Python steps, bit for bit:
  tokens, logprobs, prompt logprobs and accepted drafts, with no drafts, MTP-3 and DFlash-7.
- On 4 cards (`bench.py`, draft policy `always`), against the Python steps with Gloo, the median decode step.
  The drafts and their acceptance are the same in both:
  - No drafts: no change (the card limits these steps).
  - MTP-3, 1 stream: 2K context 22.2 → 19.6 ms (-12%), 8K 22.8 → 20.5 ms (-10%), 32K 23.9 → 22.3 ms (-7%).
    4 streams: -6 to -9%. A batch of 8 decodes with no drafts.
  - DFlash-7, 1 stream: 2K 28.7 → 26.7 ms (-7%), 8K 29.7 → 27.4 ms (-8%), 32K 31.4 → 28.8 ms (-8%).
    4 streams: -8%. 8 streams: 2K 178.8 → 167.3 ms (-6%), 32K 214.2 → 184.8 ms (-14%).
  - Prefill: 2K 0.582 → 0.572 s (+1.8%), 8K 2.076 → 2.050 s (+1.3%).
  - Prompt logprobs are native. An A/B scenario with two scored prompts: 10.4 s → 4.3 s.
  - The executor gives the gain with any step transport. MTP-3 at 1 stream and 2K context takes the same time
    with Gloo and with native ranks (19.44 and 19.48 ms).
- With the default draft policy (`auto`) on prose, the server uses drafts in fewer steps, so the gain is
  smaller. In tok/s against the Python steps, with the old probe rule in both (a probe after 32 or more plain
  steps, with a back-off): MTP-3 +1 to +6% at 1 stream and no change at 8 and 16 streams. DFlash-7 +1 to +6% at
  1 stream and +2 to +3% at 4 to 16 streams.
- **Fixed: with `--step-transport native`, the draft policy did not use its measured step times.** The first
  card selected drafts or plain steps from fixed estimates of the step times for the whole run. Now it selects
  with the step times that it measures, as with the other transports. In the throughput sweep, the rates with
  drafts changed by -16 to +27% with 1 or 2 requests (median 0%), and by at most 5% with 4 or more requests.
- **256K context on 8 cards.** With `--step-transport native` and `-c 262144`, a prompt of 262,143 tokens
  prefills in 187 s.
- **A layout of GLM-5.3-Flash and Qwen replicas runs no Python for a step.** GLM-5.3-Flash on 4 cards with a
  Qwen3.5-0.8B replica on each card, in a 60 s window of 4 GLM streams: 13.9 million Python calls → 56 (the
  template render and the parser of each request), and GLM 220 → 235 tok/s. A Qwen3.5-9B replica does not fit
  beside GLM-5.3-Flash on 4 cards (76.8 GiB of weights a card).
- The Python step code of GLM-5.3-Flash is removed. The CPU tests keep a reference of the Python table fill
  (`tests/glm_reference.py`), and the native fill must give the same bytes.

### DeepSeek-V4.1-Flash

- **A native executor runs every step.** Rust fills the tables, runs the launches, waits for the card, and
  accepts the DSpark drafts. Python runs only at startup. The output is the same as the Python steps that
  it replaces, bit for bit. On 8 cards (`bench.py`, DSpark-3), against the Python steps with Gloo:
  - 1 stream: 2K context 125 → 134, 8K 132 → 143, 32K 123 → 132, 128K 118 → 127 (+7 to +8%).
  - 4 and 8 streams: no change (the card limits these steps).
  - Prefill: 2K 6,927 → 7,164 tok/s (+3.4%), 8K 8,562 → 8,659 (+1.1%).
  - Prompt logprobs are native. An A/B scenario with two scored prompts: 10.7 s → 3.6 s.
- **A layout of DeepSeek-V4.1-Flash and Qwen replicas runs no Python for a step.** The card gate is native.
  The log line of each step, the metrics snapshot and the parent process's wait are native too. In a layout
  of DeepSeek on 8 cards and a Qwen3.5-9B replica on each card, the Python calls in a 93 s window went from
  376,000 to 200 (a once-a-second timer).
- **Bounded replay: prefill 1.4 times as fast at 2K context and 2.2 times at 120K.** Layers 20 to 39 compute
  only the last 256 prompt positions. This is an approximation of the full model. `RESIN_DS41_REPLAY=0`
  turns it off. The prefix cache works with it: a repeated 1,054-token prompt gets 768 cached tokens, and its
  time to first token decreases from 300 ms to 178 ms.
- **Faster decode.** 8 streams with DSpark: step 20.4–21.8 → 19.0 ms. 1 stream with DSpark: two changes cut
  5 to 6% of the step each.
- **Fixed: Engram could hash the wrong previous tokens.** With overlap on (the default), this happened when a
  request arrived, finished or was cancelled, and after each 2,048 tokens. Then each running sequence used wrong
  n-grams at its next 3 positions, and a later prefix-cache hit could restore wrong tokens. Now the context
  records each token at its position. On 8 cards, all the test scenarios give the same tokens and logprobs with
  overlap on and off.
- **Fixed: a long prompt could get a wrong first token beside a request with prompt logprobs.** When a prompt
  longer than one prefill chunk shared a step with a request that asked for prompt logprobs, its next chunk ran
  all the layers and read windows that its replayed chunks did not write. Now each prompt makes its own replay
  choice.
- **Fixed: with sampling and DSpark drafts, the tokens of one step used the same random number.** Thus they were
  not independent, and a seeded request's output depended on the scheduling. Now each token uses the random
  number of its position, as without drafts.

### Several models in one server

- **New: give `-m` more than once.** One server serves all the models on one port. `/v1/models` lists them,
  and the `model` field of a request selects one. Refer to [Serve several models](README.md#serve-several-models).
  - Several Qwen models on one card, with one KV cache.
  - Models on cards of their own. GLM-5.3-Flash on 4 cards and 4 Qwen3.8-27B replicas on the other 4 did
    not slow each other down.
  - Qwen replicas on the cards of DeepSeek-V4.1-Flash or GLM-5.3-Flash. The replicas use the card time that
    the first model does not use, and they share the KV cache of each card with it. DeepSeek-V4.1-Flash
    decodes 0.5% slower while 8 replicas are busy.
- **New flags:** `--share` (the weight of a model in the card time), `--kv-floor`, `-o latency|throughput`.
- **The flags after an `-m` apply to that model.** The flags before the first `-m` apply to all models.

### Multi-card models

- **Ranks exchange each step over Unix sockets** (`--step-transport`, `auto` by default). This replaces 7
  collectives a step. GLM-5.3-Flash on a busy host: +11% at 1 stream. DeepSeek-V4.1-Flash: +4% at 1 stream.
- **New: `--step-transport native`** (Qwen3.5/3.8, DeepSeek-V4.1-Flash and GLM-5.3-Flash). Rank 0's native
  executor sends each step, and its draft decision, to the other ranks. No rank runs Python or Gloo for a
  step. `auto` stays sockets.
- **A cold start on several cards leaves recipes that the next start can use.**

### Server

- **Prompts render beside the engine's loop.** Before, a request's template and tokenization ran between two
  steps. A 113K-token prompt held every running stream for about 200 ms.
- `--request-queue-capacity` now counts the requests that wait for their render. Up to 4 more can render at
  the same time.
- **New: `-f` / `--features` turns on optional features.** Each feature is off unless you name it, and
  `-f help` lists them. `-f oldqwen` runs Qwen3.5 and Qwen3.8 on the Python executor, on one card only. Without
  it, the native executor runs them. `-f skipvalidation` skips the check of the tensor bindings of each
  launch. Use it only with a build that you trust.
- **A request with no sampling fields takes the values in the checkpoint's `generation_config.json`.**
- **The log shows the life of each request at the `info` level:** cached tokens, prefill rate, time to first
  token, each decode or verify step and the finish. The headers and the body are at the `debug` level.
- **New: `"return_progress": true` streams the progress of a long prompt** on `/v1/chat/completions`, as in
  llama.cpp. Before the first token, the stream gets a `prompt_progress` chunk when the engine admits the
  request and after each prefill chunk: the prompt tokens, the cached tokens, the processed tokens and the
  time. The decode rate does not change. Refer to [Show the prompt progress](README.md#show-the-prompt-progress).
- **Startup shows progress for every model, rank and step.** The kernels and recipes steps show what they
  do and the time that is left. With several cards, a graph that another card compiled counts as compiled.
  The ready line gives the time of each step. `systemctl status wright` shows the step during the load.
- **Fixed: the ready line came before the server listened.** Now the server logs the "is ready at" line and
  tells systemd that it is ready only when it accepts connections. Before, a client that started at that
  line could get "connection refused".
- **A waiting request releases the prefix pages that it restored** before a batch waits or fails. Before,
  this could fail with a `MemoryError`.
- `bench.py --step-transport native` measures a multi-card model with native rank dispatch, also with
  `--accepted-drafts`. `bench.py` now shows the acceptance rate of native executors.
- **New: `bench.py --grid cross` measures the grid of the 0.3.0 tables with one load of the model:** contexts
  2K to 64K at 1 and 8 streams, and 1 to 64 streams at 4K and 64K. The prompts of a sequence share their
  tokens, so each context prefills only its new tokens. With DeepSeek-V4.1-Flash, `--draft-tokens 0,3`
  measures no drafts and DSpark in the same load. Each result also gives the host load and the steps that
  the host stalled. Refer to [The cross grid](docs/performance.md#the-cross-grid).
