"""The cross grid (``resin.bench.run_grid``) on CPU: prompts grow from context to context and
cells decode subsets of them, with the prefix cache; prefill rows come from the growth when
its chunks are a fresh prefill's; draft configurations; the bench CLI's grid options."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from output_helpers import forward_output

from resin import (
    BatchPhase,
    CacheGroupFamily,
    CacheGroupLayout,
    CacheLayout,
    Engine,
    RunnerConfig,
    SamplingParams,
)
from resin.bench import (
    BenchmarkRunner,
    Chunking,
    CrossGrid,
    DraftConfig,
    GridText,
    TextWorkload,
    host_fields,
    host_snapshot,
    run_grid,
    set_drafting,
    stall_fields,
)

CHUNK = 16
TAIL = (900, 901)


class Runner:
    """A paged runner: KV blocks of 4 tokens and STATE checkpoints of ``state`` tokens (a
    recurrent model's prefix checkpoints), or a final span that a prefix hit computes again."""

    vocab_size = 1024

    def __init__(self, *, state: int | None = 16, replay: int = 0, capacity=4096):
        groups = [CacheGroupLayout(4)]
        if state:
            groups.append(CacheGroupLayout(state, family=CacheGroupFamily.STATE))
        self.cache_layout = CacheLayout(
            parent_capacity=capacity, groups=groups, replay_suffix_tokens=replay
        )
        self.runner_config = RunnerConfig(
            cache_layout=self.cache_layout,
            continuation_prefill=True,
            prefill_chunk_size=CHUNK,
            max_batch_tokens=CHUNK,
        )
        self.plans = []

    def bind_resources(self, resources):
        pass

    def forward(self, plan):
        self.plans.append(plan)
        return forward_output([7] * plan.num_sequences)


def stream(row: int, length: int) -> list[int]:
    return [100 + row] + [(row * 7 + i) % 500 + 1 for i in range(length - 1)]


def text(name="t/greedy", steps=6, output=4) -> GridText:
    workload = TextWorkload(
        name=name,
        prompts=lambda index, context: [],
        stop_token_ids=(),
        output_tokens=output,
        steps=steps,
    )
    return GridText(workload, stream, TAIL)


GRID = CrossGrid(
    row_contexts=(16, 32, 64),
    row_streams=(1, 4),
    column_contexts=(32,),
    column_streams=(1, 2, 8),
)


def grid(raw, *, texts=None, **options):
    runner = BenchmarkRunner(raw)
    engine = Engine(
        runner, max_running_requests=8, context_length=256, prefix_cache=True
    )
    options.setdefault("configs", [DraftConfig("none", 0)])
    records = run_grid(
        engine,
        options.pop("grid", GRID),
        texts or [text()],
        runner=runner,
        chunking=Chunking.of(raw, CHUNK, CHUNK),
        warmup_steps=1,
        **options,
    )
    assert not engine.has_work()
    return records


def cells(records, phase="decode"):
    return [
        (r["context_length"], r["batch_size"]) for r in records if r["phase"] == phase
    ]


def test_the_cross_grid_lists_each_context_widest_first():
    assert CrossGrid().cells() == {
        2048: (8, 1),
        4096: (64, 32, 16, 8, 4, 2, 1),
        8192: (8, 1),
        16384: (8, 1),
        32768: (8, 1),
        65536: (64, 32, 16, 8, 4, 2, 1),
    }
    assert CrossGrid().growth == 8 and CrossGrid().streams == 64
    only = CrossGrid(row_contexts=(), column_contexts=(64,), column_streams=(2, 1))
    assert only.cells() == {64: (2, 1)} and only.growth == 1
    for bad in (dict(row_streams=(0,)), dict(row_contexts=(), column_contexts=())):
        with pytest.raises(ValueError):
            CrossGrid(**bad)


def test_grown_prompts_and_subsets_prefill_only_new_tokens():
    raw = Runner()
    records = grid(raw)
    assert cells(records) == [
        (16, 4), (16, 1),
        (32, 8), (32, 4), (32, 2), (32, 1),
        (64, 4), (64, 1),
    ]  # fmt: skip
    decode = [r for r in records if r["phase"] == "decode"]
    for r in decode:
        # every cell's prompts come from the cache up to their closing tokens
        assert r["cached_tokens_min"] == r["cached_tokens_max"] == r["context_length"]
        assert r["prompt_tokens"] == r["context_length"] + len(TAIL)
        assert r["measure_steps"] == 6 and r["committed_tokens"] == 6 * r["batch_size"]
        assert r["measured_global_batch_size_min"] == r["batch_size"]
        assert r["draft_config"] == "none" and r["configs"] == ["none"]
        assert set(r["host"]) == {
            "load1_start",
            "load1_end",
            "busy_cores",
            "cpus",
            "start",
            "end",
        }
    by = {(r["context_length"], r["batch_size"]): r for r in decode}
    # the first context is a fresh prefill; later ones grow the first 4 rows; rows above the
    # growth set are prefilled fresh when the 8-stream column first needs them
    assert by[16, 4]["grown_rows"] == 0 and not by[16, 4]["grown"]
    assert by[32, 8]["grown_rows"] == 4 and not by[32, 8]["grown"]
    assert by[32, 4]["grown"] and by[64, 1]["grown"]
    assert [by[32, n]["resident"] for n in (8, 4, 2, 1)] == [8, 8, 8, 8]
    assert [by[32, n]["subset"] for n in (8, 4, 2, 1)] == [False, True, True, True]
    assert by[64, 1]["resident"] == 4 and by[64, 1]["subset"]
    # each prefill step after the first context's computes only tokens the cache lacks
    prefills = [p for p in raw.plans if p.phase == BatchPhase.PREFILL]
    starts = [int(s) for p in prefills for s in p.context_lengths]
    assert (
        starts.count(0) == 1 + 4 + 4
    )  # warm-up, the first context's rows, the extra rows
    # the prefill rows: row 0's chunks over its contexts are a fresh prefill's
    rows = [r for r in records if r["phase"] == "prefill"]
    assert [(r["context_length"], r["prefill_method"]) for r in rows] == [
        (16, "growth"),
        (32, "growth"),
        (64, "growth"),
    ]
    for r in rows:
        length = r["context_length"] + len(TAIL)
        assert r["input_tokens"] == length
        assert [tuple(c) for c in r["prefill_chunks"]] == Chunking.of(
            raw, CHUNK, CHUNK
        ).fresh(length)
        assert r["window_seconds"] == pytest.approx(sum(r["step_seconds"]))
        assert r["prefill_seconds"] == [r["window_seconds"]]
        # each chunk's time is its median over the 4 growth rows
        assert len(r["prefill_samples"]) == 4


def test_chunks_that_a_fresh_prefill_cuts_differently_take_a_fresh_prompt():
    # a final span computed again after each hit: grown chunks are not a fresh prefill's
    raw = Runner(state=None, replay=8)
    records = grid(raw)
    rows = [r for r in records if r["phase"] == "prefill"]
    assert [r["prefill_method"] for r in rows] == ["growth", "fresh", "fresh"]
    assert [len(r["prefill_samples"]) for r in rows] == [4, 3, 3]
    for r in rows:
        assert [tuple(c) for c in r["prefill_chunks"]] == Chunking.of(
            raw, CHUNK, CHUNK
        ).fresh(r["context_length"] + len(TAIL))
    decode = [r for r in records if r["phase"] == "decode"]
    # a hit leaves the final span (rounded to the checkpoints) to compute
    assert {r["context_length"]: r["cached_tokens_min"] for r in decode} == {
        16: 8,
        32: 24,
        64: 56,
    }


@pytest.mark.parametrize(
    "state,replay,length",
    [(16, 0, 18), (16, 0, 66), (None, 8, 34), (None, 8, 18), (None, 0, 40), (8, 5, 50)],
)
def test_the_fresh_chunks_are_the_engines(state, replay, length):
    raw = Runner(state=state, replay=replay)
    engine = Engine(
        BenchmarkRunner(raw),
        max_running_requests=1,
        context_length=256,
        prefix_cache=True,
    )
    lease = engine.submit(
        stream(0, length), sampling_params=SamplingParams(max_output_tokens=1)
    )
    while not lease.done:
        engine.step()
    chunks = [
        (int(p.context_lengths[0]), int(p.context_lengths[0]) + p.num_tokens)
        for p in raw.plans
    ]
    assert chunks == Chunking.of(raw, CHUNK, CHUNK).fresh(length)


def test_without_reuse_every_cell_prefills_from_an_empty_cache():
    raw = Runner()
    records = grid(raw, reuse=False)
    assert cells(records) == cells(grid(Runner()))
    for r in records:
        assert r["cached_tokens_max"] == 0 and r["reuse"] is False
        if r["phase"] == "decode":
            assert r["resident"] == r["batch_size"] and not r["subset"]
            assert r["grown_rows"] == 0
        else:
            assert r["prefill_method"] == "fresh"


def test_the_same_prompts_and_seeds_give_the_same_tokens_with_and_without_reuse():
    class Echo(Runner):
        """the next token depends on the row's prompt and its tokens so far"""

        def forward(self, plan):
            self.plans.append(plan)
            rows = []
            for r in range(plan.num_sequences):
                end = plan.token_indptr[r + 1]
                rows.append(int(plan.token_ids[end - 1]) * 31 % 997 + 1)
            return forward_output(rows)

    reused, fresh = grid(Echo()), grid(Echo(), reuse=False)
    assert [r["sample_tokens"] for r in reused if r["phase"] == "decode"] == [
        r["sample_tokens"] for r in fresh if r["phase"] == "decode"
    ]


def test_cells_that_do_not_fit_are_skipped(capsys):
    records = grid(Runner(), fits=lambda context, streams: streams * context <= 128)
    assert cells(records) == [(16, 4), (16, 1), (32, 4), (32, 2), (32, 1), (64, 1)]
    assert "context 32 streams [8]" in capsys.readouterr().out


def test_each_cell_runs_each_configuration_that_drafts_there():
    switched = []
    records = grid(
        Runner(),
        configs=[DraftConfig("none", 0), DraftConfig("dspark3", 3, rows=2)],
        switch=lambda config: switched.append(config.name),
    )
    runs = [
        (
            r["context_length"],
            r["batch_size"],
            r["draft_config"],
            r["configs"],
            r["drafting"],
        )
        for r in records
        if r["phase"] == "decode"
    ]
    assert runs[:3] == [
        (16, 4, "none", ["none", "dspark3"], False),
        (16, 1, "none", ["none"], False),
        (16, 1, "dspark3", ["dspark3"], True),
    ]
    assert [(c, n) for c, n, name, _, _ in runs if name == "dspark3"] == [
        (16, 1), (32, 2), (32, 1), (64, 1),
    ]  # fmt: skip
    assert switched[:3] == ["none", "dspark3", "none"]
    # prefill rows stand for every configuration
    assert {tuple(r["configs"]) for r in records if r["phase"] == "prefill"} == {
        ("none", "dspark3")
    }
    with pytest.raises(ValueError, match="need a switch"):
        grid(
            Runner(),
            configs=[DraftConfig("none", 0), DraftConfig("dspark3", 3, rows=2)],
        )


def test_one_configuration_without_a_plain_one_runs_every_cell():
    records = grid(Runner(), configs=[DraftConfig("mtp2", 2)])
    assert {r["draft_config"] for r in records} == {"mtp2"}
    assert all(r["drafting"] for r in records if r["phase"] == "decode")


def test_decode_only_and_several_texts():
    records = grid(
        Runner(), phases=("decode",), texts=[text("a/greedy"), text("b/greedy")]
    )
    assert {r["phase"] for r in records} == {"decode"}
    assert [r["workload"] for r in records].count("a/greedy") == len(records) // 2


def test_the_grid_needs_the_prefix_cache_and_an_idle_engine():
    raw = Runner()
    runner = BenchmarkRunner(raw)
    engine = Engine(runner, max_running_requests=8, context_length=256)
    with pytest.raises(ValueError, match="prefix_cache"):
        run_grid(
            engine, GRID, [text()], runner=runner, chunking=Chunking(CHUNK),
            configs=[DraftConfig("none", 0)],
        )  # fmt: skip


def test_a_short_stream_fails():
    short = GridText(text().workload, lambda row, length: [1] * min(length, 10), TAIL)
    with pytest.raises(ValueError, match="no stream of 16 tokens"):
        short.prompt(0, 16)
    assert short.prompt(0, 2).tolist() == [1, 1, *TAIL]
    assert text().prompt(3, 4).dtype == np.uint32


def test_host_fields():
    import sys

    before = host_snapshot()
    if before is None:
        assert sys.platform != "linux"
        return
    fields = host_fields(before, host_snapshot())["host"]
    assert fields["load1_start"] >= 0 and fields["cpus"] >= 1
    assert host_fields(None, before) == {"host": None}


def test_set_drafting_switches_the_engine_and_the_runner_together():
    local = SimpleNamespace(max_speculative_rows=2)
    runner = SimpleNamespace(drafting=True)
    set_drafting(SimpleNamespace(local=local), runner, False, 2)
    assert (local.max_speculative_rows, runner.drafting) == (0, False)
    set_drafting(local, runner, True, 2)
    assert (local.max_speculative_rows, runner.drafting) == (2, True)

    class Refuses:
        @property
        def drafting(self):
            return True

        @drafting.setter
        def drafting(self, on):
            raise NotImplementedError("no switch")

    with pytest.raises(NotImplementedError):
        set_drafting(local, Refuses(), False, 2)
    assert local.max_speculative_rows == 2  # put back


# ---------------------------------------------------------------------------- the CLI

CORPUS = {"english": "english.jsonl"}


def configure(*arguments):
    from scripts import bench

    return bench._configuration(bench.argument_parser().parse_args(list(arguments)))


def test_the_grid_options_size_the_runner_and_check_the_cache(tmp_path):
    from scripts import bench

    corpus = tmp_path / "english.jsonl"
    corpus.write_text("")
    spec, cases, _, options = configure(
        "--model", "deepseek_v41-gaudi2", "--grid", "cross", "--draft-tokens", "0,3",
        "--step-transport", "native", "--prompt-file", f"english={corpus}",
        "--grid-column-contexts", "none", "--grid-row-contexts", "2048,4096",
    )  # fmt: skip
    assert options["draft_tokens"] == 3
    assert options["grid"]["drafts"] == [0, 3] and options["grid"]["reuse"]
    assert options["grid"]["column_contexts"] == ()
    assert options["text"]["skip_unfit"]
    tail = bench.GRID_TAIL_TOKENS
    assert [(c["phase"], c["context_length"], c["batch_size"]) for c in cases] == [
        ("decode", 2048 + tail, 8),
        ("decode", 2048 + tail, 1),
        ("decode", 4096 + tail, 8),
        ("decode", 4096 + tail, 1),
        ("prefill", 2048 + tail, 1),
        ("prefill", 4096 + tail, 1),
    ]


@pytest.mark.parametrize(
    "arguments,fragment",
    [
        (["--draft-tokens", "0,3"], "with --grid only"),
        (["--grid", "cross", "--draft-tokens", "2,3", "--prompt-file", "e=x"], "0,K with"),
        (["--grid", "cross"], "--prompt-file"),
        (["--grid", "cross", "--draft-tokens", "0,3", "--prompt-file", "e=x"], "native"),
        (
            ["--model", "qwen3_5-gaudi2", "--grid", "cross", "--draft-tokens", "0,2",
             "--prompt-file", "e=x"],
            "cannot switch",
        ),
    ],
)  # fmt: skip
def test_grid_options_that_cannot_run_fail_before_the_load(arguments, fragment):
    model = [] if "--model" in arguments else ["--model", "deepseek_v41-gaudi2"]
    with pytest.raises(ValueError, match=fragment):
        configure(*model, *arguments)


def test_draft_names(tmp_path):
    from deploy.models import get_model
    from scripts.bench import draft_name

    qwen, ds = get_model("qwen3_5-gaudi2"), get_model("deepseek_v41-gaudi2")
    dspine = tmp_path / "dspine"
    dspine.mkdir()
    (dspine / "config.json").write_text(
        json.dumps({"dflash_config": {"selected_prefix_code_bridge": True}})
    )
    dflash = tmp_path / "dflash"
    dflash.mkdir()
    (dflash / "config.json").write_text(json.dumps({"dflash_config": {}}))
    assert draft_name(qwen, 0, None) == "none"
    assert draft_name(qwen, 2, None) == "mtp2"
    assert draft_name(qwen, 15, dspine) == "dspine15"
    assert draft_name(qwen, 7, dflash) == "dflash7"
    assert draft_name(ds, 3, None) == "dspark3"


def test_grid_prompt_streams(tmp_path, monkeypatch):
    """Each row's prompt stream is its rendered passage without the closing tokens; rows differ;
    a row's prompt at one context is the start of its prompt at a longer one."""
    import deploy.models
    from scripts import bench

    class Tokenizer:
        def encode(self, text):
            return [ord(c) for c in text]

        def decode(self, ids):
            return "".join(map(chr, ids))

    class Recipe:
        tokenizer = Tokenizer()
        stop_token_ids = (3,)

        def render(self, messages, options=None):
            body = messages[0].content[0].text
            return SimpleNamespace(token_ids=[1, 2, *map(ord, body), 3, 4])

    monkeypatch.setattr(deploy.models, "create_recipe", lambda *a, **k: Recipe())
    corpus = tmp_path / "e.jsonl"
    corpus.write_text(
        "\n".join(
            json.dumps({"instruction": "go", "text": chr(65 + i) * 300 + "".join(
                chr(97 + (i * j) % 26) for j in range(300))})
            for i in range(3)
        )
    )  # fmt: skip
    small = CrossGrid(row_contexts=(64, 128), row_streams=(1, 2), column_contexts=())
    options = dict(
        prompt_files={"e": str(corpus)}, render_options={}, text_modes=["greedy", "sampled"],
        text_seeds=[1, 2], temperature=1.0, top_k=None, top_p=1.0, text_steps=5,
        text_output_tokens=7,
    )  # fmt: skip
    texts = bench._grid_texts(None, None, None, options, small)
    assert [t.workload.name for t in texts] == ["e/greedy", "e/sampled", "e/sampled"]
    assert [t.workload.seed for t in texts] == [None, 1, 2]
    t = texts[0]
    assert t.tail == (3, 4)
    prompts = [t.prompt(row, 128).tolist() for row in range(4)]
    assert len({tuple(p) for p in prompts}) == 4
    for row in range(4):
        assert t.prompt(row, 64).tolist()[:64] == prompts[row][:64]
        assert prompts[row][-2:] == [3, 4] and prompts[row][:2] == [1, 2]
    assert texts[1].stream is t.stream  # the sampled texts share the streams


def test_one_slow_chunk_does_not_move_the_prefill_row(monkeypatch):
    """A chunk's time is its median over the growth rows, so one slow step of one row (a
    host stall) does not go into the prefill rows of every longer context."""
    import resin.bench

    class Clock:
        def __init__(self):
            self.now, self.slow = 0.0, False

        def __call__(self):
            self.now += 5.0 if self.slow else 1.0
            self.slow = False
            return self.now

    clock = Clock()
    monkeypatch.setattr(resin.bench.time, "perf_counter", clock)

    class Stalls(Runner):
        seen = 0

        def forward(self, plan):
            # the third growth row's chunk 16 .. 32 (at context 32) is slow
            if plan.phase == BatchPhase.PREFILL and plan.num_tokens == 16:
                if int(plan.context_lengths[0]) == 16:
                    self.seen += 1
                    clock.slow = self.seen == 3
            return super().forward(plan)

    records = grid(Stalls())
    rows = [r for r in records if r["phase"] == "prefill"]
    assert [r["step_seconds"] for r in rows] == [[1.0] * 2, [1.0] * 3, [1.0] * 5]
    assert max(rows[1]["prefill_samples"]) > 3.0


def test_stall_fields_count_only_far_slower_steps():
    assert stall_fields([0.02] * 10 + [0.5, 0.15]) == {
        "stall_steps": 1,
        "stall_seconds": 0.48,
    }
    # a verify step a few times a plain one is not a stall
    assert stall_fields([0.02] * 10 + [0.16] * 3)["stall_steps"] == 0
    assert stall_fields([])["stall_steps"] == 0
