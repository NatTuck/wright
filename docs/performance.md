# Performance

These tables have every throughput result of the real-text sweep. The README shows a summary.

## Method

[`scripts/sweep.py`](../scripts/sweep.py) starts one server for each setting, and runs
[llama-benchy](https://github.com/eugr/llama-benchy) at commit `e9be344` against it, for each
context length and number of requests.

- **Text:** English from a public domain book. Each prompt fills the context, less the reply.
- **Reply:** 256 generated tokens for each request.
- **Sampling:** The values in the `generation_config.json` of each checkpoint. The Qwen models have
  thinking off. Mistral-Large-4 and GLM-5.3-Flash use low reasoning. The Qwen3.5 dense checkpoints
  have no `generation_config.json`. They use temperature 1.0, top-p 0.95, and top-k 20, the values
  of their model cards and of Qwen3.8-27B.
- **Cache:** The prefix cache is off, so every prompt goes through prefill.
- **Runs:** A cell is the mean of 3 runs after 1 warmup run. A cell with 131,072 or more prompt
  tokens in total (context times requests) runs once, with no warmup. A `¹` marks these cells.
- **Hardware:** HL-225 cards, Synapse 1.24.1, and the pinned checkpoints.
- **Dates:** UTC−7.

## How to read the tables

Each cell shows **prefill / decode / peak** in tok/s.

- **Prefill** is the prompt tokens per second. For one request, the time runs to the first token.
  For more than one request, it is the prompt rate that llama-benchy reports.
- **Decode** is the total of generated tokens per second for all requests together.
- **Peak** is the highest total in any one second of the run.
- **No fit** means that the sweep did not run that cell, because a cell with the same number of
  tokens or fewer did not fit all its requests in the KV cache at the same time.

llama-benchy sends all requests at the same time. Thus, with more than one request, the decode time
includes the prefill of the other requests, and at long context most of the time is prefill. The
sweep does not measure a server that receives requests at different times.

In agentic workloads, each turn adds to a context that the server has already seen. This is
generally not a problem there: the prefix cache holds the earlier context, so each turn prefills
little more than the new tokens. The sweep ran before Wright divided the card time between prefill
and decode. Its results are the same as with `--decode-share 0`. Refer to
[Prefill stops decode for one chunk](../README.md#prefill-stops-decode-for-one-chunk).

## Make the results

Run one command for each model. Each setting writes its own JSONL file in `--output`, and a stopped
sweep continues where it stopped. `--report` prints the tables.

```bash
uv run --no-default-groups --group gaudi scripts/sweep.py \
  --qwen ~/models/Qwen3.8-27B-FP8 --texts english \
  --only qwen-mtp0,qwen-mtp1,qwen-mtp2 --output bench/sweep

uv run --no-default-groups --group gaudi scripts/sweep.py \
  --glm ~/models/GLM-5.3-Flash --texts english \
  --only glm-8cards-mtp0,glm-8cards-mtp1,glm-8cards-mtp2,glm-4cards-mtp0,glm-4cards-mtp1,glm-4cards-mtp2 \
  --output bench/sweep

uv run --no-default-groups --group gaudi scripts/sweep.py \
  --mistral ~/models/Mistral-Large-4-1T-A52B-GGUF \
  --mistral-tokenizer ~/models/Mistral-Large-4-tokenizer \
  --mistral-drafter ~/models/Mistral-Large-4-drafter \
  --texts english --contexts 2048,8192,32768 --concurrency 1,4,16 \
  --only mistral-nodraft --output bench/sweep
uv run --no-default-groups --group gaudi scripts/sweep.py \
  --mistral ~/models/Mistral-Large-4-1T-A52B-GGUF \
  --mistral-tokenizer ~/models/Mistral-Large-4-tokenizer \
  --mistral-drafter ~/models/Mistral-Large-4-drafter \
  --texts english --contexts 2048,8192,32768 --concurrency 1,2,4 \
  --only mistral-dspark3 --output bench/sweep

uv run --no-default-groups --group gaudi scripts/sweep.py \
  --qwen35 ~/models --texts english --contexts 8192 --concurrency 1,8 \
  --only qwen35-9b-mtp0 --output bench/sweep
```

For the Qwen3.5 dense models, `--qwen35` is the folder that holds `Qwen3.5-9B-FP8` and the other
sizes. Use `--only qwen35-4b-mtp0`, `qwen35-2b-mtp0`, or `qwen35-0.8b-mtp0` for the other sizes.
To run the sizes at the same time on different cards, give each command its own `--gpu` and
`--port`.

The DeepSeek-V4.1-Flash results come from llama-benchy and one server. Start the server, then run
llama-benchy:

```bash
wright -m ~/models/DeepSeek-V4.1-Flash-1017 --spec-draft-n-max 3 -c 512000 -np 8 \
  --decode-share 25:99

uvx --from git+https://github.com/eugr/llama-benchy@e9be344578cec17745066b220798b80a0d2686d3 \
  llama-benchy --base-url http://127.0.0.1:8080/v1 \
  --model deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --served-model-name deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --tokenizer ~/models/DeepSeek-V4.1-Flash-1017 \
  --pp 1024 4096 32768 --tg 512 1024 --concurrency 1 2 4 8 \
  --runs 3 --warmup-runs 1 --no-cache --latency-mode generation --skip-coherence \
  --book-url https://www.gutenberg.org/files/1661/1661-0.txt \
  --extra-body 'temperature=1.0,top_p=1.0,reasoning_effort="low"'
```

To get the acceptance of drafts, read the `spec_accepted_tokens_total` and
`spec_draft_tokens_total` counters at `/metrics` before and after each cell.

## Qwen3.8-27B, 1 card

These results are from 2026-11-01.

**No drafts** (`qwen-mtp0`)

| Context | 1 request | 2 requests | 4 requests | 8 requests | 16 requests | 32 requests |
|---:|---:|---:|---:|---:|---:|---:|
| 1K | 3,460 / 64 / 65 | 3,295 / 125 / 126 | 3,305 / 238 / 240 | 3,291 / 362 / 400 | 3,293 / 485 / 599 | 3,300 / 567 / 758 |
| 2K | 4,624 / 64 / 64 | 4,522 / 116 / 126 | 4,419 / 188 / 215 | 4,363 / 285 / 360 | 4,350 / 362 / 522 | 4,319 / 395 / 625 |
| 4K | 4,858 / 64 / 64 | 4,491 / 104 / 115 | 4,512 / 146 / 187 | 4,547 / 184 / 281 | 4,521 / 210 / 373 | 4,530 / 227 / 435¹ |
| 8K | 4,757 / 63 / 64 | 4,554 / 85 / 102 | 4,600 / 96 / 143 | 4,620 / 113 / 200 | 4,602 / 120 / 247¹ | 4,609 / 122 / 272¹ |
| 16K | 4,351 / 60 / 61 | 4,263 / 59 / 82 | 4,296 / 58 / 105 | 4,290 / 61 / 129¹ | 4,287 / 58 / 137¹ | 4,290 / 60 / 157¹ |
| 32K | 3,659 / 53 / 53 | 3,621 / 35 / 62 | 3,637 / 30 / 68¹ | 3,632 / 27 / 74¹ | 3,632 / 27 / 80¹ | 3,346 / 26 / 79¹ |
| 64K | 2,753 / 44 / 45 | 2,734 / 16 / 42¹ | 2,747 / 12 / 39¹ | 2,744 / 11 / 40¹ | No fit | No fit |
| 128K | 1,831 / 33 / 34¹ | 1,829 / 6 / 27¹ | 1,835 / 4 / 21¹ | No fit | No fit | No fit |

**MTP-1** (`qwen-mtp1`)

| Context | 1 request | 2 requests | 4 requests | 8 requests | 16 requests | 32 requests |
|---:|---:|---:|---:|---:|---:|---:|
| 1K | 3,330 / 93 / 93 | 3,153 / 175 / 178 | 3,162 / 276 / 296 | 3,161 / 416 / 494 | 3,168 / 505 / 655 | 3,165 / 557 / 791 |
| 2K | 4,488 / 92 / 93 | 4,387 / 164 / 175 | 4,292 / 259 / 299 | 4,219 / 318 / 431 | 4,211 / 373 / 575 | 4,252 / 396 / 657 |
| 4K | 4,726 / 89 / 89 | 4,370 / 128 / 160 | 4,392 / 172 / 239 | 4,434 / 200 / 332 | 4,416 / 206 / 384 | 4,416 / 229 / 461¹ |
| 8K | 4,636 / 93 / 94 | 4,452 / 100 / 135 | 4,498 / 105 / 162 | 4,515 / 109 / 199 | 4,519 / 121 / 251¹ | 4,510 / 117 / 267¹ |
| 16K | 4,250 / 83 / 84 | 4,158 / 65 / 98 | 4,190 / 61 / 116 | 4,199 / 61 / 134¹ | 4,188 / 60 / 150¹ | 4,192 / 59 / 160¹ |
| 32K | 3,552 / 66 / 67 | 3,516 / 36 / 71 | 3,534 / 30 / 76¹ | 3,532 / 28 / 81¹ | 3,526 / 26 / 81¹ | 3,337 / 25 / 81¹ |
| 64K | 2,655 / 54 / 54 | 2,636 / 16 / 51¹ | 2,649 / 11 / 46¹ | 2,645 / 11 / 44¹ | No fit | No fit |
| 128K | 1,752 / 43 / 44¹ | 1,749 / 6 / 33¹ | 1,756 / 4 / 25¹ | No fit | No fit | No fit |

**MTP-2** (`qwen-mtp2`)

| Context | 1 request | 2 requests | 4 requests | 8 requests | 16 requests | 32 requests |
|---:|---:|---:|---:|---:|---:|---:|
| 1K | 3,219 / 115 / 115 | 3,139 / 189 / 198 | 3,136 / 304 / 372 | 3,160 / 425 / 526 | 3,167 / 501 / 671 | 3,179 / 522 / 747 |
| 2K | 4,427 / 113 / 113 | 4,360 / 206 / 215 | 4,269 / 268 / 318 | 4,219 / 311 / 446 | 4,230 / 360 / 560 | 4,207 / 372 / 627 |
| 4K | 4,711 / 108 / 109 | 4,359 / 146 / 184 | 4,395 / 173 / 254 | 4,446 / 198 / 341 | 4,426 / 216 / 412 | 4,418 / 222 / 435¹ |
| 8K | 4,627 / 103 / 104 | 4,449 / 113 / 151 | 4,495 / 104 / 170 | 4,525 / 107 / 215 | 4,509 / 112 / 252¹ | 4,514 / 119 / 267¹ |
| 16K | 4,243 / 94 / 95 | 4,158 / 67 / 102 | 4,192 / 61 / 120 | 4,196 / 59 / 141¹ | 4,188 / 59 / 151¹ | 4,193 / 60 / 157¹ |
| 32K | 3,554 / 71 / 71 | 3,514 / 37 / 77 | 3,531 / 30 / 79¹ | 3,533 / 28 / 82¹ | 3,531 / 26 / 83¹ | 3,322 / 25 / 85¹ |
| 64K | 2,657 / 62 / 62 | 2,637 / 16 / 53¹ | 2,651 / 12 / 46¹ | 2,646 / 10 / 44¹ | No fit | No fit |
| 128K | 1,753 / 45 / 45¹ | 1,750 / 6 / 34¹ | 1,757 / 4 / 26¹ | No fit | No fit | No fit |

## Mistral-Large-4, 8 cards

These results are from 2026-11-03.

The model accepted 52% to 60% of the DSpark draft tokens. The drafter works on batches of up to 4
sequences. At 32K with 16 requests, the KV cache held 11 requests at the same time. These results
replace a sweep from 2026-11-01, which gave low decode rates at 8K and longer.

**No drafts** (`mistral-nodraft`)

| Context | 1 request | 4 requests | 16 requests |
|---:|---:|---:|---:|
| 2K | 1,981 / 48 / 49 | 1,587 / 129 / 136 | 1,535 / 263 / 273 |
| 8K | 2,043 / 43 / 44 | 1,882 / 96 / 100 | 1,875 / 176 / 192¹ |
| 32K | 1,439 / 35 / 35 | 1,395 / 72 / 79¹ | 1,278 / 24 / 88¹ |

**DSpark-3** (`mistral-dspark3`)

| Context | 1 request | 2 requests | 4 requests |
|---:|---:|---:|---:|
| 2K | 1,950 / 74 / 75 | 1,838 / 110 / 122 | 1,567 / 141 / 155 |
| 8K | 2,010 / 60 / 61 | 1,874 / 76 / 85 | 1,852 / 104 / 113 |
| 32K | 1,427 / 48 / 48 | 1,389 / 64 / 67 | 1,382 / 79 / 86¹ |

## GLM-5.3-Flash, 8 cards

These results are from 2026-11-01.

**No drafts** (`glm-8cards-mtp0`)

| Context | 1 request | 2 requests | 4 requests | 8 requests | 16 requests |
|---:|---:|---:|---:|---:|---:|
| 1K | 3,268 / 100 / 101 | 3,269 / 173 / 185 | 3,175 / 273 / 301 | 3,198 / 323 / 400 | 3,229 / 387 / 483 |
| 2K | 3,881 / 99 / 99 | 3,794 / 159 / 170 | 3,468 / 223 / 268 | 3,363 / 258 / 360 | 3,427 / 262 / 409 |
| 4K | 4,323 / 97 / 98 | 3,883 / 135 / 158 | 3,752 / 165 / 241 | 3,593 / 166 / 296 | 3,609 / 169 / 318 |
| 8K | 4,228 / 90 / 90 | 3,957 / 102 / 143 | 3,903 / 105 / 182 | 3,817 / 104 / 218 | 3,847 / 103 / 249¹ |
| 16K | 3,981 / 96 / 97 | 3,845 / 72 / 124 | 3,799 / 66 / 189 | 3,765 / 55 / 190¹ | 3,781 / 57 / 198¹ |
| 32K | 3,517 / 96 / 97 | 3,446 / 38 / 108 | 3,436 / 31 / 109¹ | 3,423 / 28 / 149¹ | 3,430 / 25 / 174¹ |
| 64K | 2,873 / 91 / 91 | 2,829 / 18 / 89¹ | 2,820 / 13 / 147¹ | 2,822 / 11 / 112¹ | 2,826 / 10 / 113¹ |
| 128K | 2,092 / 85 / 86¹ | 2,073 / 8 / 79¹ | 2,071 / 5 / 67¹ | 2,073 / 4 / 128¹ | 2,074 / 4 / 156¹ |

**MTP-1** (`glm-8cards-mtp1`)

| Context | 1 request | 2 requests | 4 requests | 8 requests | 16 requests |
|---:|---:|---:|---:|---:|---:|
| 1K | 3,179 / 130 / 130 | 3,106 / 200 / 211 | 3,141 / 269 / 310 | 3,179 / 305 / 363 | 3,121 / 286 / 356 |
| 2K | 3,688 / 137 / 138 | 3,754 / 213 / 220 | 3,450 / 241 / 296 | 3,355 / 250 / 365 | 3,362 / 221 / 318 |
| 4K | 4,315 / 136 / 137 | 3,964 / 150 / 185 | 3,789 / 165 / 247 | 3,532 / 155 / 279 | 3,561 / 149 / 262 |
| 8K | 4,224 / 124 / 124 | 3,937 / 120 / 174 | 3,883 / 105 / 191 | 3,792 / 97 / 212 | 3,810 / 94 / 208¹ |
| 16K | 3,970 / 130 / 131 | 3,811 / 72 / 148 | 3,765 / 62 / 195 | 3,738 / 57 / 184¹ | 3,756 / 52 / 170¹ |
| 32K | 3,501 / 121 / 122 | 3,429 / 42 / 129 | 3,414 / 29 / 116¹ | 3,399 / 27 / 141¹ | 3,409 / 25 / 125¹ |
| 64K | 2,858 / 126 / 126 | 2,817 / 20 / 109¹ | 2,808 / 13 / 158¹ | 2,809 / 11 / 152¹ | 2,812 / 11 / 109¹ |
| 128K | 2,082 / 118 / 119¹ | 2,066 / 8 / 95¹ | 2,062 / 5 / 70¹ | 2,063 / 4 / 119¹ | 2,064 / 4 / 95¹ |

**MTP-2** (`glm-8cards-mtp2`)

| Context | 1 request | 2 requests | 4 requests | 8 requests | 16 requests |
|---:|---:|---:|---:|---:|---:|
| 1K | 3,199 / 127 / 127 | 3,090 / 171 / 186 | 3,030 / 198 / 233 | 3,076 / 200 / 238 | 3,090 / 230 / 278 |
| 2K | 3,827 / 125 / 125 | 3,745 / 174 / 185 | 3,794 / 194 / 229 | 3,398 / 173 / 237 | 3,330 / 183 / 256 |
| 4K | 4,294 / 119 / 120 | 4,132 / 138 / 168 | 3,729 / 147 / 206 | 3,557 / 123 / 192 | 3,551 / 130 / 208 |
| 8K | 4,215 / 118 / 118 | 3,947 / 109 / 159 | 3,882 / 95 / 158 | 3,791 / 84 / 157 | 3,802 / 86 / 187¹ |
| 16K | 3,956 / 130 / 130 | 3,826 / 71 / 131 | 3,756 / 59 / 149 | 3,738 / 50 / 138¹ | 3,750 / 50 / 142¹ |
| 32K | 3,499 / 120 / 120 | 3,430 / 40 / 114 | 3,416 / 29 / 91¹ | 3,393 / 26 / 105¹ | 3,413 / 25 / 116¹ |
| 64K | 2,860 / 117 / 118 | 2,806 / 19 / 88¹ | 2,804 / 14 / 132¹ | 2,805 / 11 / 108¹ | 2,810 / 11 / 105¹ |
| 128K | 2,085 / 115 / 115¹ | 2,066 / 8 / 90¹ | 2,062 / 5 / 60¹ | 2,063 / 4 / 60¹ | 2,064 / 4 / 56¹ |

## GLM-5.3-Flash, 4 cards (`-tp4`)

These results are from 2026-11-01.

**No drafts** (`glm-4cards-mtp0`)

| Context | 1 request | 2 requests | 4 requests | 8 requests | 16 requests |
|---:|---:|---:|---:|---:|---:|
| 1K | 2,396 / 91 / 91 | 2,366 / 155 / 159 | 2,376 / 222 / 233 | 2,387 / 235 / 291 | 2,374 / 270 / 335 |
| 2K | 2,814 / 90 / 90 | 2,773 / 153 / 159 | 2,723 / 184 / 213 | 2,512 / 189 / 271 | 2,507 / 192 / 288 |
| 4K | 3,365 / 88 / 88 | 2,985 / 110 / 133 | 2,913 / 127 / 182 | 2,700 / 116 / 216 | 2,717 / 123 / 228 |
| 8K | 3,266 / 87 / 88 | 3,069 / 87 / 119 | 3,006 / 81 / 142 | 2,913 / 78 / 156 | 2,927 / 78 / 169¹ |
| 16K | 3,037 / 85 / 86 | 2,916 / 54 / 104 | 2,887 / 51 / 138 | 2,854 / 43 / 145¹ | 2,863 / 40 / 141¹ |
| 32K | 2,615 / 85 / 86 | 2,555 / 29 / 89 | 2,545 / 22 / 85¹ | 2,536 / 20 / 112¹ | 2,540 / 18 / 128¹ |
| 64K | 2,046 / 82 / 83 | 2,016 / 14 / 79¹ | 2,009 / 25 / 167¹ | 2,009 / 9 / 87¹ | 2,014 / 8 / 101¹ |
| 128K | 1,424 / 77 / 77¹ | 1,411 / 5 / 68¹ | 1,409 / 4 / 100¹ | 1,412 / 3 / 130¹ | 1,394 / 3 / 52¹ |

**MTP-1** (`glm-4cards-mtp1`)

| Context | 1 request | 2 requests | 4 requests | 8 requests | 16 requests |
|---:|---:|---:|---:|---:|---:|
| 1K | 2,269 / 108 / 108 | 2,346 / 160 / 170 | 2,341 / 208 / 234 | 2,339 / 229 / 274 | 2,304 / 214 / 267 |
| 2K | 2,768 / 112 / 112 | 2,766 / 170 / 176 | 2,601 / 172 / 214 | 2,446 / 171 / 246 | 2,471 / 162 / 229 |
| 4K | 3,358 / 110 / 110 | 2,977 / 118 / 152 | 2,844 / 127 / 196 | 2,692 / 113 / 198 | 2,687 / 110 / 189 |
| 8K | 3,266 / 112 / 112 | 3,039 / 91 / 136 | 2,980 / 80 / 145 | 2,885 / 74 / 156 | 2,902 / 72 / 168¹ |
| 16K | 3,024 / 111 / 111 | 2,900 / 55 / 114 | 2,869 / 52 / 142 | 2,831 / 42 / 137¹ | 2,839 / 39 / 124¹ |
| 32K | 2,600 / 108 / 108 | 2,539 / 31 / 101 | 2,521 / 24 / 129¹ | 2,523 / 18 / 109¹ | 2,527 / 19 / 103¹ |
| 64K | 2,038 / 108 / 109 | 2,006 / 14 / 87¹ | 1,999 / 28 / 166¹ | 2,003 / 8 / 113¹ | 1,954 / 8 / 75¹ |
| 128K | 1,418 / 100 / 100¹ | 1,405 / 5 / 77¹ | 1,402 / 3 / 103¹ | No fit | No fit |

**MTP-2** (`glm-4cards-mtp2`)

| Context | 1 request | 2 requests | 4 requests | 8 requests | 16 requests |
|---:|---:|---:|---:|---:|---:|
| 1K | 2,341 / 102 / 103 | 2,297 / 140 / 154 | 2,291 / 156 / 177 | 2,293 / 146 / 174 | 2,278 / 164 / 200 |
| 2K | 2,776 / 102 / 103 | 2,746 / 128 / 140 | 2,514 / 125 / 157 | 2,500 / 125 / 166 | 2,440 / 134 / 181 |
| 4K | 3,349 / 98 / 99 | 3,100 / 103 / 127 | 2,886 / 103 / 143 | 2,680 / 90 / 142 | 2,672 / 96 / 155 |
| 8K | 3,264 / 100 / 100 | 3,040 / 83 / 121 | 2,973 / 68 / 116 | 2,883 / 63 / 112 | 2,894 / 62 / 139¹ |
| 16K | 3,020 / 106 / 107 | 2,900 / 56 / 105 | 2,862 / 47 / 109 | 2,832 / 40 / 103¹ | 2,840 / 37 / 108¹ |
| 32K | 2,598 / 103 / 103 | 2,539 / 31 / 89 | 2,530 / 22 / 76¹ | 2,514 / 18 / 77¹ | 2,523 / 18 / 112¹ |
| 64K | 2,034 / 93 / 94 | 2,000 / 13 / 79¹ | 1,989 / 13 / 92¹ | 1,995 / 8 / 78¹ | 1,947 / 8 / 67¹ |
| 128K | 1,417 / 92 / 93¹ | 1,405 / 5 / 68¹ | 1,400 / 3 / 44¹ | No fit | No fit |

## DeepSeek-V4.1-Flash, 8 cards

These results are from 2026-11-05, with DSpark-3 (`--spec-draft-n-max 3`). The method is different
from the method of the other tables:

- **Server:** One server ran all cells, with 512,000 tokens of context, `-np 8`, and
  `--decode-share 25:99`.
- **Prompt:** Each prompt has exactly 1K, 4K, or 32K tokens. The reply comes after it.
- **Reply:** Each request generates a maximum of 512 or 1,024 tokens. A request stops when the
  model ends its answer. Thus, some requests generate fewer tokens. Wright does not support
  `min_tokens` or `ignore_eos`.
- **Sampling:** The checkpoint has no `generation_config.json`. The requests use temperature 1.0,
  top-p 1.0, and low reasoning.
- **Drafts:** DSpark runs only on decode steps of 2 sequences or fewer. A larger step runs without
  drafts. Thus, with 4 or 8 requests, most steps have no drafts. The percent in each cell is the
  part of the draft tokens that the model accepted. In total, the model accepted 45%.
- **Runs:** A cell is the mean of 3 runs after 1 warmup run. A `¹` marks a cell that ran once,
  with no warmup. A `²` marks a cell that ran twice.

With 8 requests, prefill decreases to approximately 1,500 tok/s, because decode gets up to 99% of
the card time. At 32K, a prefill takes approximately 10 s, and a reply takes less time. Thus, with
more than one request, the requests decode mostly one after the other. As a result, the decode rate
is low.

**Replies of up to 512 tokens** (prefill / decode / peak, accepted drafts)

| Prompt | 1 request | 2 requests | 4 requests | 8 requests |
|---:|---:|---:|---:|---:|
| 1K | 3,204 / 127 / 128, 43% | 3,341 / 187 / 198, 45% | 2,476 / 223 / 260, 57% | 1,510 / 247 / 369, 43% |
| 4K | 4,017 / 115 / 116, 41% | 3,202 / 152 / 183, 43% | 2,356 / 151 / 224, 44% | 1,535 / 136 / 243, 44% |
| 32K | 3,284 / 113 / 114, 44% | 2,665 / 53 / 117, 44% | 2,611 / 37 / 114, 48%¹ | 2,392 / 33 / 129, 45%¹ |

**Replies of up to 1,024 tokens** (prefill / decode / peak, accepted drafts)

| Prompt | 1 request | 2 requests | 4 requests | 8 requests |
|---:|---:|---:|---:|---:|
| 1K | 3,157 / 126 / 126, 45% | 3,343 / 163 / 219, 48% | 2,376 / 193 / 277, 48% | 1,514 / 257 / 395, 47% |
| 4K | 4,021 / 128 / 128, 45% | 3,218 / 133 / 191, 42% | 2,351 / 134 / 250, 46% | 1,470 / 158 / 277, 42% |
| 32K | 3,277 / 119 / 120, 49% | 2,796 / 52 / 122, 48%² | 2,453 / 43 / 122, 47%¹ | 2,338 / 37 / 129, 46%¹ |

## Qwen3.5 dense, 1 card each

These results are from 2026-11-03.

The four models ran at the same time, on four different cards.

**Qwen3.5-9B, no drafts** (`qwen35-9b-mtp0`)

| Context | 1 request | 8 requests |
|---:|---:|---:|
| 8K | 12,994 / 169 / 170 | 12,397 / 304 / 517 |

**Qwen3.5-4B, no drafts** (`qwen35-4b-mtp0`)

| Context | 1 request | 8 requests |
|---:|---:|---:|
| 8K | 15,999 / 240 / 241 | 15,178 / 361 / 637 |

**Qwen3.5-2B, no drafts** (`qwen35-2b-mtp0`)

| Context | 1 request | 8 requests |
|---:|---:|---:|
| 8K | 30,447 / 389 / 390 | 28,741 / 674 / 1,102 |

**Qwen3.5-0.8B, no drafts** (`qwen35-0.8b-mtp0`)

| Context | 1 request | 8 requests |
|---:|---:|---:|
| 8K | 36,359 / 504 / 506 | 34,211 / 818 / 1,309 |

