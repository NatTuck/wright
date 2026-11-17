# Performance

These tables have every throughput result of the Wright 0.3.0 sweep. The README shows a summary. The results of
Wright 0.1.0 and 0.2.0 are in [`performance-0.2.0.md`](performance-0.2.0.md). They use a different method, so do
not compare the two sets value for value.

## Method

[`scripts/bench.py`](../scripts/bench.py) loads the model on its cards and runs the Engine without the HTTP
server. The sweep uses the [cross grid](#the-cross-grid): contexts from 2K to 64K at 1 and at 8 requests, and 1 to
64 requests at 4K and at 64K context, with one load of the model for each configuration.

- **Text:** passages of English prose and of source code from
  [`scripts/fetch_bench_corpus.py`](../scripts/fetch_bench_corpus.py). A prompt has as many tokens of its passage
  as its context.
- **Decode:** each request generates up to 512 tokens. The benchmark times 512 decode steps after 5 warmup steps.
  A request that ends starts again with the same prompt and the next seed.
- **Prefill:** one request, from the chunks of the prompts that grow from one context to the next. Refer to
  [Prefill](#prefill).
- **Sampling:** the values in the `generation_config.json` of each checkpoint. The Qwen3.5 dense checkpoints have no
  `generation_config.json`, so they use temperature 1.0, top-p 0.95, and top-k 20, as Qwen3.8-27B does. The Qwen
  models have thinking off. GLM-5.3-Flash and DeepSeek-V4.1-Flash use low reasoning.
- **Cache:** the prefix cache is on, because the grid uses it to reuse the prompts.
- **Runs:** one run for each text, with seed 1. A cell is valid when its stalls (steps that take more than 10
  times the median step and more than 0.1 s) take 1% of its time or less. A busy host can make stalls. The sweep
  measured the cells that were not valid again on a quiet host. A cell that still had stalls has a † mark.
- **Cards:** the results of one model come from one set of cards, for all its configurations.
- **Drafts:** the servers select drafts for each step with the default policy (`auto`). With a drafter, the decode
  rate depends on the text. These tables use one passage for each text, so a drafter can give other values with
  other passages. For example, DFlash 2 on code at 2K with 1 request gives 166 tok/s here, and it gave 199 tok/s
  in an earlier sweep that took the median of 3 passages.
- **Hardware:** HL-225 cards, Synapse 1.24.1, and the pinned checkpoints in
  [Checkpoints](../README.md#checkpoints). The results are from 2026-11-15, and
  GLM-5.3-Flash MTP-3 and DFlash 2 from 2026-11-16.

## How to read the tables

- The values are tok/s.
- **Prefill** is the prompt tokens per second of one request, with prose.
- **Decode** is the total of generated tokens per second for all requests together, in the form
  **prose / code**.
- **No fit** means that the requests of that cell do not fit in the KV cache of the cards at the same time.
- **†** means that the cell had stalls in each run. The value is the rate without the time of the stalls. The note
  of the model tells more.

## Make the results

Get the text, then run one command for each configuration. The commands are in [Commands](#commands).

```bash
uv run --no-default-groups --group gaudi scripts/fetch_bench_corpus.py --output bench/corpus
```

- **Drafts:** use `--draft-tokens 0` for no drafts. For a drafter, add `--assistant-checkpoint` with the drafter
  directory, and `--draft-tokens 7` for DFlash 2 or `--draft-tokens 15` for DSpine.
- **Other models:** use `--model glm5_3_flash-gaudi2-tp4` (with `--step-transport native`), `deepseek_v41-gaudi2`
  or `qwen3_5-9b-gaudi2` (and the other Qwen3.5 sizes). Use the sampling values of the checkpoint. For
  GLM-5.3-Flash and DeepSeek-V4.1-Flash, use `--render-options '{"reasoning_effort": "low"}'` and no `--top-k`.
- **Cards:** give each run its own cards with `--gpu`.

## The cross grid

`scripts/bench.py --grid cross` measures a grid of decode cells and the prefill of one request at each context
of the grid, with one load of the model. It reuses the prefilled prompts, so it needs much less prefill than one
fresh cohort for each cell.

- **Rows:** contexts 2K, 4K, 8K, 16K, 32K and 64K, each at 1 and at 8 streams. Change them with
  `--grid-row-contexts` and `--grid-row-streams`.
- **Columns:** 1, 2, 4, 8, 16, 32 and 64 streams, each at 4K and at 64K context. Change them with
  `--grid-column-streams` and `--grid-column-contexts`. Give `none` for no rows or no columns.
- **Fit:** the benchmark does not measure a cell whose requests do not fit in the KV cache at the same time.
- **Text:** the grid needs `--prompt-file`. It runs the whole grid for each prompt file, sampling mode and seed.

### How the grid reuses the prompts

- A prompt of context C has C tokens of its passage (with the instruction), then the closing tokens of the chat
  template (2 to 9 tokens). Thus the prompts of one sequence at two contexts start with the same tokens.
- The prefix cache is on. The first 8 sequences (the widest row) grow. At each context, their prefill computes
  only the tokens that the prefix cache does not hold from the previous context.
- A cell decodes some of the sequences that have the context of the cell. The other sequences stay in the
  prefix cache and do not decode. Each request of a cell computes only the closing tokens again. For
  DeepSeek-V4.1-Flash, it also computes the last 256 prompt tokens again (bounded replay).
- When a cell needs more sequences than the grid has at that context, the benchmark prefills the new sequences
  once. The widest cell of each context runs first, so the narrower cells use its sequences.
- A request that ends gets the same prompt again, with the next seed.
- `--no-grid-reuse` prefills each cell from an empty prefix cache. Use it to compare the method with fresh
  prompts.

The decode rate is the committed tokens divided by the time of the measured steps, as for the other results.

### Prefill

The prefill row of each context is the time of one request, from the chunks that the sequences prefilled while
they grew. The sequences grow one at a time, so each chunk is the only work of its step. Each chunk of the prompt
gets the median of its times over the sequences that grow (8 with the default rows), and the row is the sum of
these medians. Thus one slow step does not change the rows of the longer contexts. The benchmark uses the growth
chunks only when they are the same as the chunks of a fresh prefill of the prompt. This is true for Qwen3.5,
Qwen3.8 and GLM-5.3-Flash, whose cache checkpoints are at multiples of the 4,096-token chunk. Otherwise the
benchmark prefills 3 new prompts of that context and uses the median of each chunk. DeepSeek-V4.1-Flash computes
the last prompt tokens again after a prefix hit, so its prefill rows use new prompts. The `prefill_method` field
of the result is `growth` or `fresh`, and `prefill_samples` has the total time of each prompt.

### Drafts in one load

`--draft-tokens 0,K` measures each cell with the drafts off and with K drafts, with one load. Only a runner that
can turn its drafts off and on at run time can do this: DeepSeek-V4.1-Flash, with `--step-transport native` on
8 cards. DeepSeek-V4.1-Flash makes drafts only on decode steps of 2 sequences or fewer. Thus a cell of 1 or 2
streams runs once with each configuration. A wider cell runs once with no drafts, and its result is for both
configurations. Other models use one run for each draft configuration.

### The fields of a grid result

A grid result has the fields of the other results, and these fields:

| Field | Value |
|---|---|
| `grid`, `reuse` | `cross`, and whether the grid reused the prompts |
| `draft_config` | The configuration of the run: `none`, `mtpK`, `dflashK`, `dspineK` or `dsparkK` |
| `configs` | The configurations that the result is for |
| `drafting` | Whether the decode steps of the cell can make drafts |
| `prompt_tokens` | The tokens of each prompt: the context and the closing tokens |
| `grown`, `grown_rows` | Whether all sequences of the cell grew from a shorter context, and how many did |
| `resident`, `subset` | The sequences at the context of the cell, and whether the cell decodes fewer of them |
| `prefill_method`, `prefill_chunks`, `prefill_samples` | For a prefill row: `growth` or `fresh`, the chunks that it adds, and the total of each prompt |

Each result (with a grid or not) also has `host`: the 1-minute load average at the start and at the end of the
result, the busy cores while it ran, and its start and end times. `stall_steps` and `stall_seconds` count the
steps that took more than 10 times the median step and more than 0.1 s, and the time that they added. The rate
includes these steps. Use these values to find results from a busy host.

### Commands

```bash
uv run --no-default-groups --group gaudi scripts/bench.py --model qwen3_5-gaudi2 \
  --checkpoint ~/models/Qwen3.8-27B-FP8 --draft-tokens 2 --grid cross \
  --prompt-file english=bench/corpus/english.jsonl --prompt-file code=bench/corpus/coding.jsonl \
  --text-modes sampled --temperature 1.0 --top-k 20 --top-p 0.95 --seeds 1 \
  --text-output-tokens 512 --text-steps 512 --render-options '{"enable_thinking": false}' \
  --output bench/qwen-mtp2.jsonl
```

For DeepSeek-V4.1-Flash with no drafts and with DSpark-3 in one load:

```bash
uv run --no-default-groups --group gaudi scripts/bench.py --model deepseek_v41-gaudi2 \
  --checkpoint ~/models/DeepSeek-V4.1-Flash-1017 --draft-tokens 0,3 --step-transport native --grid cross \
  --prompt-file english=bench/corpus/english.jsonl --prompt-file code=bench/corpus/coding.jsonl \
  --text-modes sampled --temperature 1.0 --top-p 1.0 --seeds 1 \
  --text-output-tokens 512 --text-steps 512 --render-options '{"reasoning_effort": "low"}' \
  --output bench/deepseek.jsonl
```

## Qwen3.8-27B, 1 card

The four configurations ran on the same card.

† In these MTP-2 cells on code, some steps took 0.5 to 1 s. This happened in the same cells in two runs, on
two cards, and only with MTP-2 on code. The other configurations had no such steps.

**Prefill, 1 request** (prose)

| Context | No drafts | MTP-2 | DFlash 2 (7) | DSpine (15) |
|---:|---:|---:|---:|---:|
| 2K | 4,935 | 4,886 | 4,851 | 4,848 |
| 4K | 5,129 | 5,088 | 5,066 | 5,067 |
| 8K | 5,086 | 5,052 | 5,038 | 5,036 |
| 16K | 4,810 | 4,782 | 4,772 | 4,770 |
| 32K | 4,273 | 4,253 | 4,244 | 4,244 |
| 64K | 3,473 | 3,460 | 3,455 | 3,454 |

**Decode, 1 request** (prose / code)

| Context | No drafts | MTP-2 | DFlash 2 (7) | DSpine (15) |
|---:|---:|---:|---:|---:|
| 2K | 64 / 64 | 99 / 131 | 97 / 166 | 91 / 167 |
| 4K | 64 / 64 | 101 / 124 | 100 / 149 | 96 / 151 |
| 8K | 61 / 61 | 93 / 119† | 85 / 178 | 82 / 187 |
| 16K | 58 / 58 | 85 / 114 | 74 / 152 | 71 / 148 |
| 32K | 57 / 56 | 83 / 106 | 70 / 133 | 68 / 133 |
| 64K | 50 / 50 | 69 / 90 | 56 / 122 | 54 / 119 |

**Decode, 8 requests** (prose / code)

| Context | No drafts | MTP-2 | DFlash 2 (7) | DSpine (15) |
|---:|---:|---:|---:|---:|
| 2K | 414 / 412 | 510 / 686 | 349 / 714 | 420 / 719 |
| 4K | 386 / 387 | 483 / 604† | 334 / 576 | 399 / 585 |
| 8K | 347 / 348 | 422 / 557† | 332 / 569 | 330 / 543 |
| 16K | 290 / 290 | 351 / 474† | 284 / 515 | 289 / 476 |
| 32K | 219 / 219 | 264 / 354 | 216 / 354 | 219 / 349 |
| 64K | 148 / 147 | 179 / 235 | 140 / 244 | 158 / 243 |

**Decode at 4K context** (prose / code)

| Requests | No drafts | MTP-2 | DFlash 2 (7) | DSpine (15) |
|---:|---:|---:|---:|---:|
| 1 | 64 / 64 | 101 / 124 | 100 / 149 | 96 / 151 |
| 2 | 119 / 119 | 173 / 225† | 152 / 283 | 142 / 279 |
| 4 | 215 / 217 | 294 / 387† | 224 / 423 | 255 / 428 |
| 8 | 386 / 387 | 483 / 604† | 334 / 576 | 399 / 585 |
| 16 | 598 / 598 | 676 / 875 | 556 / 604 | 578 / 587 |
| 32 | 791 / 791 | 805 / 1,075 | 710 / 784 | 751 / 763 |
| 64 | 1,011 / 1,011 | 1,006 / 1,007 | 998 / 998 | 999 / 998 |

**Decode at 64K context** (prose / code)

| Requests | No drafts | MTP-2 | DFlash 2 (7) | DSpine (15) |
|---:|---:|---:|---:|---:|
| 1 | 50 / 50 | 69 / 90 | 56 / 122 | 54 / 119 |
| 2 | 81 / 80 | 103 / 143 | 80 / 150 | 77 / 161 |
| 4 | 116 / 116 | 145 / 192 | 106 / 198 | 128 / 217 |
| 8 | 148 / 147 | 179 / 235 | 140 / 244 | 158 / 243 |
| 16 | No fit | No fit | No fit | No fit |
| 32 | No fit | No fit | No fit | No fit |
| 64 | No fit | No fit | No fit | No fit |

## GLM-5.3-Flash, 4 cards (`-tp4`)

These results use `--step-transport native`. No drafts ran on 2026-11-15, and MTP-3 and DFlash 2 on 2026-11-16,
on the same cards. With the DFlash 2 drafter, 64 requests at 4K and 16 requests at 64K do not fit.

**Prefill, 1 request** (prose)

| Context | No drafts | MTP-3 | DFlash 2 (7) |
|---:|---:|---:|---:|
| 2K | 2,949 | 2,929 | 2,918 |
| 4K | 3,707 | 3,686 | 3,682 |
| 8K | 3,716 | 3,702 | 3,697 |
| 16K | 3,452 | 3,442 | 3,440 |
| 32K | 2,962 | 2,953 | 2,956 |
| 64K | 2,285 | 2,278 | 2,283 |

**Decode, 1 request** (prose / code)

| Context | No drafts | MTP-3 | DFlash 2 (7) |
|---:|---:|---:|---:|
| 2K | 95 / 95 | 92 / 124 | 88 / 107 |
| 4K | 95 / 94 | 96 / 130 | 89 / 117 |
| 8K | 94 / 94 | 132 / 165 | 92 / 169 |
| 16K | 93 / 93 | 120 / 137 | 87 / 108 |
| 32K | 92 / 93 | 91 / 137 | 89 / 125 |
| 64K | 91 / 91 | 95 / 121 | 86 / 113 |

**Decode, 8 requests** (prose / code)

| Context | No drafts | MTP-3 | DFlash 2 (7) |
|---:|---:|---:|---:|
| 2K | 324 / 320 | 332 / 333 | 317 / 324 |
| 4K | 320 / 321 | 329 / 328 | 313 / 318 |
| 8K | 316 / 316 | 324 / 325 | 297 / 302 |
| 16K | 305 / 307 | 315 / 316 | 300 / 300 |
| 32K | 297 / 298 | 305 / 305 | 279 / 288 |
| 64K | 284 / 284 | 290 / 291 | 276 / 276 |

**Decode at 4K context** (prose / code)

| Requests | No drafts | MTP-3 | DFlash 2 (7) |
|---:|---:|---:|---:|
| 1 | 95 / 94 | 96 / 130 | 89 / 117 |
| 2 | 160 / 161 | 152 / 224 | 150 / 157 |
| 4 | 232 / 231 | 219 / 270 | 215 / 220 |
| 8 | 320 / 321 | 329 / 328 | 313 / 318 |
| 16 | 404 / 403 | 409 / 408 | 408 / 408 |
| 32 | 467 / 468 | 467 / 467 | 466 / 465 |
| 64 | 514 / 514 | 513 / 513 | No fit |

**Decode at 64K context** (prose / code)

| Requests | No drafts | MTP-3 | DFlash 2 (7) |
|---:|---:|---:|---:|
| 1 | 91 / 91 | 95 / 121 | 86 / 113 |
| 2 | 149 / 149 | 144 / 179 | 140 / 168 |
| 4 | 209 / 211 | 196 / 238 | 195 / 194 |
| 8 | 284 / 284 | 290 / 291 | 276 / 276 |
| 16 | 348 / 350 | 352 / 352 | No fit |
| 32 | No fit | No fit | No fit |
| 64 | No fit | No fit | No fit |

## DeepSeek-V4.1-Flash, 8 cards

These results use `--step-transport native`. One load measured both configurations. DSpark makes drafts only
on decode steps of 2 requests or fewer, so with more requests the DSpark-3 values are the values with no
drafts.

**Prefill, 1 request** (prose)

| Context | Prefill |
|---:|---:|
| 2K | 5,377 |
| 4K | 6,951 |
| 8K | 7,819 |
| 16K | 8,243 |
| 32K | 8,216 |
| 64K | 7,927 |

**Decode, 1 request** (prose / code)

| Context | No drafts | DSpark-3 |
|---:|---:|---:|
| 2K | 116 / 116 | 130 / 174 |
| 4K | 115 / 115 | 139 / 168 |
| 8K | 114 / 114 | 146 / 182 |
| 16K | 113 / 113 | 140 / 175 |
| 32K | 112 / 112 | 140 / 163 |
| 64K | 109 / 109 | 127 / 168 |

**Decode, 8 requests** (prose / code)

| Context | No drafts | DSpark-3 |
|---:|---:|---:|
| 2K | 469 / 469 | 469 / 469 |
| 4K | 466 / 465 | 466 / 465 |
| 8K | 457 / 457 | 457 / 457 |
| 16K | 450 / 449 | 450 / 449 |
| 32K | 437 / 436 | 437 / 436 |
| 64K | 414 / 415 | 414 / 415 |

**Decode at 4K context** (prose / code)

| Requests | No drafts | DSpark-3 |
|---:|---:|---:|
| 1 | 115 / 115 | 139 / 168 |
| 2 | 187 / 187 | 211 / 271 |
| 4 | 295 / 296 | 295 / 296 |
| 8 | 466 / 465 | 466 / 465 |
| 16 | 606 / 605 | 606 / 605 |
| 32 | 709 / 709 | 709 / 709 |
| 64 | 831 / 831 | 831 / 831 |

**Decode at 64K context** (prose / code)

| Requests | No drafts | DSpark-3 |
|---:|---:|---:|
| 1 | 109 / 109 | 127 / 168 |
| 2 | 175 / 175 | 201 / 248 |
| 4 | 275 / 273 | 275 / 273 |
| 8 | 414 / 415 | 414 / 415 |
| 16 | 526 / 526 | 526 / 526 |
| 32 | 603 / 603 | 603 / 603 |
| 64 | 698 / 697 | 698 / 697 |

## Qwen3.5 dense, 1 card each

These results have no drafts. The four sizes ran on the same card, one after the other.

† A busy host made some steps of these cells slow.

**Prefill, 1 request** (prose)

| Context | Qwen3.5-9B | Qwen3.5-4B | Qwen3.5-2B | Qwen3.5-0.8B |
|---:|---:|---:|---:|---:|
| 2K | 13,937 | 17,813 | 34,330 | 42,107 |
| 4K | 14,654 | 18,810 | 38,481 | 47,334 |
| 8K | 14,571 | 18,540 | 38,571 | 47,240 |
| 16K | 13,794 | 17,227 | 36,789 | 44,407 |
| 32K | 12,247 | 14,852 | 32,796 | 38,645 |
| 64K | 9,936 | 11,576 | 26,694 | 30,430 |

**Decode, 1 request** (prose / code)

| Context | Qwen3.5-9B | Qwen3.5-4B | Qwen3.5-2B | Qwen3.5-0.8B |
|---:|---:|---:|---:|---:|
| 2K | 173 / 173 | 252 / 252 | 418 / 427 | 600 / 599 |
| 4K | 172 / 172 | 250 / 250 | 391 / 426 | 597 / 597† |
| 8K | 162 / 162 | 229 / 229 | 421 / 408 | 588 / 588 |
| 16K | 152 / 152 | 210 / 210 | 377 / 376 | 504 / 504 |
| 32K | 147 / 147 | 200 / 199 | 338 / 337 | 436 / 435 |
| 64K | 127 / 127 | 165 / 165 | 316 / 315 | 400 / 398 |

**Decode, 8 requests** (prose / code)

| Context | Qwen3.5-9B | Qwen3.5-4B | Qwen3.5-2B | Qwen3.5-0.8B |
|---:|---:|---:|---:|---:|
| 2K | 1,086 / 1,088 | 1,435 / 1,462 | 2,471 / 2,481 | 3,104 / 3,121 |
| 4K | 1,004 / 1,002 | 1,315 / 1,314 | 2,257 / 2,244 | 2,790 / 2,787 |
| 8K | 882 / 879 | 1,113 / 1,112 | 2,158 / 2,155 | 2,634 / 2,636 |
| 16K | 711 / 710 | 848 / 855 | 1,756 / 1,746 | 2,025 / 2,058 |
| 32K | 518 / 518 | 593 / 591 | 1,282 / 1,282 | 1,441 / 1,426 |
| 64K | 336 / 335 | 366 / 366 | 832 / 834† | 898 / 898† |

**Decode at 4K context** (prose / code)

| Requests | Qwen3.5-9B | Qwen3.5-4B | Qwen3.5-2B | Qwen3.5-0.8B |
|---:|---:|---:|---:|---:|
| 1 | 172 / 172 | 250 / 250 | 391 / 426 | 597 / 597† |
| 2 | 314 / 314† | 442 / 442† | 812 / 814 | 1,113 / 1,112 |
| 4 | 563 / 562 | 760 / 760 | 1,375 / 1,376 | 1,784 / 1,652 |
| 8 | 1,004 / 1,002 | 1,315 / 1,314 | 2,257 / 2,244 | 2,790 / 2,787 |
| 16 | 1,533 / 1,532 | 1,884 / 1,882 | 3,643 / 3,580 | 4,365 / 4,380 |
| 32 | 1,942 / 1,941 | 2,233 / 2,234 | 4,808 / 4,818 | 5,503 / 5,502 |
| 64 | 2,404 / 2,401 | 2,660 / 2,660 | 5,584 / 5,528 | 6,163 / 6,160 |

**Decode at 64K context** (prose / code)

| Requests | Qwen3.5-9B | Qwen3.5-4B | Qwen3.5-2B | Qwen3.5-0.8B |
|---:|---:|---:|---:|---:|
| 1 | 127 / 127 | 165 / 165 | 316 / 315 | 400 / 398 |
| 2 | 197 / 197 | 241 / 241 | 487 / 487 | 583 / 582 |
| 4 | 273 / 273 | 312 / 312 | 674 / 674 | 751 / 761 |
| 8 | 336 / 335 | 366 / 366 | 832 / 834† | 898 / 898† |
| 16 | 378 / 377 | 398 / 397 | 929 / 936 | 973 / 979 |
| 32 | 405 / 404 | 417 / 417 | 997 / 997 | 1,027 / 1,025 |
| 64 | No fit | No fit | 1,062 / 1,062 | 1,085 / 1,085 |

## Mistral-Large-4, 8 cards

Wright 0.3.0 does not have new results for Mistral-Large-4. Refer to
[`performance-0.2.0.md`](performance-0.2.0.md#mistral-large-4-8-cards).
