"""Cache pools in several processes that share one arena layout across the boundary slab of a shared-memory
SlabTable: the tensor-parallel model's pool (side 0) holds the slabs below it on every card, each replica's pool
(side k) the boundary slab and above on its card. The boundary moves one slab at a time toward the side that
wants slabs, once the other side has emptied its edge slab."""

from __future__ import annotations

import multiprocessing as mp
import random
import time
from pathlib import Path

import pytest
from output_helpers import forward_output

from resin import (
    BatchPlan,
    CacheGroupLayout,
    CacheLayout,
    CachePool,
    Engine,
    ForwardOutput,
    RunnerConfig,
    SamplingParams,
    SlabTable,
)


class SlabRunner:
    """Each step returns the last input token + 1; slabs hold one 2-token block."""

    def __init__(self, capacity: int) -> None:
        self.runner_config = RunnerConfig(
            cache_layout=CacheLayout(parent_capacity=capacity, groups=[CacheGroupLayout(2)]),
            continuation_prefill=True,
        )

    def bind_resources(self, binding) -> None:
        pass

    def forward(self, plan: BatchPlan) -> ForwardOutput:
        return forward_output(
            [plan.token_ids[end - 1] + 1 for _, end in map(plan.token_range, range(plan.num_sequences))]
        )


def _engine(pool: CachePool) -> Engine:
    return Engine(
        SlabRunner(pool.parent_capacity),
        context_length=64,
        enable_overlap=False,
        prefix_cache=True,
        cache_pool=pool,
    )


def _wait_for(check, seconds: float = 10.0) -> None:
    deadline = time.monotonic() + seconds
    while not check():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def _expected(prompt: list[int], outputs: int) -> list[int]:
    return [prompt[-1] + k for k in range(1, outputs + 1)]


def _drain(engines: list[Engine], seconds: float = 20.0) -> None:
    deadline = time.monotonic() + seconds
    while any(engine.has_work() for engine in engines):
        assert time.monotonic() < deadline, "the engines did not finish"
        for engine in engines:
            if engine.has_work():
                engine.step()
        time.sleep(0.0005)


def _agree(table: SlabTable, pools: list[CachePool]) -> bool:
    """Each pool holds its side of the boundary, less an edge slab that it gave for a move not done yet."""
    sides = table.sides()
    held = [size - (1 if given else 0) for _, size, given, _, _ in sides]
    return held == [pool.held_slabs for pool in pools]


def test_each_side_holds_its_side_of_the_boundary_and_the_table_reopens(tmp_path: Path) -> None:
    path = tmp_path / "slabs"
    table = SlabTable.create(path, 8, 2, 1, [1, 2])
    # five spare slabs beyond the floors (1 and the replicas' 2): the boundary starts at 1 + 1 + 5 // 2
    assert table.boundary == 4
    pools = [CachePool(8, boundary=table.side(side)) for side in range(3)]
    assert [pool.held_slabs for pool in pools] == [3, 5, 5]
    assert _agree(table, pools)
    # the cache metrics count the slabs of the pool's side as held, and none of the other side as occupied
    engine = Engine(SlabRunner(8), context_length=64, cache_pool=pools[1], enable_metrics=True)
    snapshot = engine.metrics_snapshot()
    assert snapshot is not None and snapshot.cache is not None
    cache = snapshot.cache
    assert (cache["capacity_slabs"], cache["held_slabs"], cache["free_slabs"], cache["occupied_slabs"]) == (8, 5, 5, 0)
    again = SlabTable.open(path)
    assert (again.capacity, again.cards, again.boundary) == (8, 2, 4)


def test_a_table_refuses_floors_that_do_not_fit_and_unknown_sides(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="do not fit in 4 slabs"):
        SlabTable.create(tmp_path / "a", 4, 1, 2, [3])
    with pytest.raises(ValueError, match="one floor for each replica card"):
        SlabTable.create(tmp_path / "b", 4, 2, 1, [1])
    table = SlabTable.create(tmp_path / "c", 4, 1, 1, [1])
    with pytest.raises(ValueError, match="1 replica cards"):
        table.side(2)
    with pytest.raises(OSError):
        SlabTable.create(tmp_path / "c", 4, 1, 1, [1])  # it exists
    (tmp_path / "d").write_bytes(b"not a table")
    with pytest.raises(ValueError, match="is not a slab table"):
        SlabTable.open(tmp_path / "d")


def test_a_replica_grows_into_the_idle_first_models_empty_slabs_and_keeps_its_prefix(tmp_path: Path) -> None:
    """The first model is idle with empty slabs and no prefix: a replica that needs slabs takes them, down to the
    first model's headroom, before it evicts its own cached prefix."""
    table = SlabTable.create(tmp_path / "slabs", 8, 1, 1, [1])
    assert table.boundary == 5  # four slabs each
    pools = [CachePool(8, boundary=table.side(0)), CachePool(8, boundary=table.side(1))]
    _first, replica = (_engine(pool) for pool in pools)
    cached = replica.submit(list(range(31, 38)), sampling_params=SamplingParams(max_output_tokens=1))
    _drain([replica])
    assert cached.take_result() == [38]  # its prefix holds slabs 8, 7 and 6
    # three blocks: slab 5 is free, and the first model gives slabs 4 and 3 (it keeps a floor and a headroom)
    live = replica.submit(list(range(50, 55)), sampling_params=SamplingParams(max_output_tokens=1))
    _drain([replica])
    assert live.take_result() == [55]
    assert table.boundary == 3
    again = replica.submit(list(range(31, 38)), sampling_params=SamplingParams(max_output_tokens=1))
    assert again.cached_tokens > 0, "the replica kept its prefix"
    _drain([replica])
    assert again.take_result() == [38]


def test_a_process_that_died_holding_the_tables_lock_does_not_stop_the_others(tmp_path: Path) -> None:
    path = tmp_path / "slabs"
    table = SlabTable.create(path, 8, 1, 1, [1])
    dead = mp.get_context("spawn").Process(target=int)
    dead.start()
    dead.join()
    assert dead.pid is not None
    with path.open("r+b") as file:  # the lock word (offset 16) holds its owner's process id
        file.seek(16)
        file.write(dead.pid.to_bytes(4, "little"))
    started = time.monotonic()
    pool = CachePool(8, boundary=table.side(1))
    assert pool.held_slabs == 4
    assert time.monotonic() - started < 5


def _first_model(path: str, done) -> None:
    """The tensor-parallel side in a process of its own: one request that needs a slab of the replica's side."""
    table = SlabTable.open(path)
    engine = _engine(CachePool(table.capacity, boundary=table.side(0)))
    lease = engine.submit([1, 2, 3, 4, 5, 6, 7], sampling_params=SamplingParams(max_output_tokens=1))
    deadline = time.monotonic() + 20
    while engine.has_work() and time.monotonic() < deadline:
        engine.step()
        time.sleep(0.001)
    done.put((lease.take_result() if not engine.has_work() else None, table.boundary))


def test_the_boundary_moves_between_processes_to_the_side_that_needs_slabs(tmp_path: Path) -> None:
    """The first model, in another process, needs a fourth slab; the replica's pool (its own thread: no step of
    the replica's Engine) gives its edge slab, and the boundary moves up."""
    path = tmp_path / "slabs"
    table = SlabTable.create(path, 6, 1, 1, [1])
    assert table.boundary == 4  # three slabs each
    replica = CachePool(6, boundary=table.side(1))
    engine = _engine(replica)
    cached = engine.submit([9, 8, 7, 6], sampling_params=SamplingParams(max_output_tokens=1))
    _drain([engine])
    assert cached.take_result() == [7]

    context = mp.get_context("spawn")
    done = context.Queue()
    first = context.Process(target=_first_model, args=(str(path), done))
    first.start()
    result, boundary = done.get(timeout=30)
    first.join(30)
    assert result == [8], "the first model's request finished"
    assert boundary == 5
    _wait_for(lambda: replica.held_slabs == 2)


def test_a_live_replica_block_at_the_boundary_holds_the_move_until_it_ends(tmp_path: Path) -> None:
    table = SlabTable.create(tmp_path / "slabs", 8, 2, 1, [1, 1])
    assert table.boundary == 5  # four slabs each
    pools = [CachePool(8, boundary=table.side(side)) for side in range(3)]
    first, card1, card2 = (_engine(pool) for pool in pools)
    # card 1's replica holds all four of its slabs with a live request (its prefill; it does not step for now)
    live = card1.submit(list(range(20, 28)), sampling_params=SamplingParams(max_output_tokens=2))
    card1.step()
    card1.step()  # the prefill: 6 tokens, then the last 2 (prefix-cache alignment)
    # the first model needs a fifth slab: card 2's replica gives slab 5, card 1's cannot yet
    lease = first.submit(list(range(1, 10)), sampling_params=SamplingParams(max_output_tokens=1))
    for _ in range(100):
        first.step()
        time.sleep(0.002)
    assert not lease.done
    assert table.boundary == 5
    assert [given for _, _, given, _, _ in table.sides()] == [0, 0, 5]
    assert table.sides()[0][4], "the first model wants a slab"
    # card 1's request ends; its slab 5 holds only a cached prefix now, which goes, and the boundary moves
    _drain([first, card1])
    assert live.take_result() == [28, 29]
    assert lease.take_result() == [10]
    _wait_for(lambda: _agree(table, pools))
    assert not card2.has_work()


def test_a_replica_that_gave_its_edge_slab_waits_for_the_other_replicas(tmp_path: Path) -> None:
    """Card 2 gives slab 5 while card 1's live block holds its slab 5: card 2 then has no edge slab until the
    boundary passes. It does not clear slab 6 or evict the prefix in it for a move that cannot happen."""
    table = SlabTable.create(tmp_path / "slabs", 8, 2, 1, [1, 1])
    assert table.boundary == 5  # four slabs each
    cleared: list[int] = []
    pools = [
        CachePool(8, boundary=table.side(0)),
        CachePool(8, boundary=table.side(1)),
        CachePool(8, boundary=table.side(2), scrub=cleared.extend),
    ]
    first, card1, card2 = (_engine(pool) for pool in pools)
    # card 2 caches a prefix in its slabs 8, 7 and 6
    cached = card2.submit(list(range(31, 38)), sampling_params=SamplingParams(max_output_tokens=1))
    _drain([card2])
    assert cached.take_result() == [38]
    blocks = card2.cache_occupied_block_count
    assert blocks == 3
    live = card1.submit(list(range(20, 28)), sampling_params=SamplingParams(max_output_tokens=2))
    card1.step()
    card1.step()  # its prefill holds slabs 8 to 5 (prefix-cache alignment: 6 tokens, then 2)
    lease = first.submit(list(range(1, 10)), sampling_params=SamplingParams(max_output_tokens=1))
    for _ in range(100):
        first.step()
        time.sleep(0.002)
    assert [given for _, _, given, _, _ in table.sides()] == [0, 0, 5]
    assert cleared == [5], "card 2 cleared only the slab it gave"
    assert card2.cache_occupied_block_count == blocks, "card 2 kept its prefix"
    _drain([first, card1])
    assert live.take_result() == [28, 29]
    assert lease.take_result() == [10]
    _wait_for(lambda: _agree(table, pools))


def test_the_first_model_runs_on_in_its_own_slabs_while_a_live_block_holds_the_edge(tmp_path: Path) -> None:
    """The first model's two requests need six slabs, it holds four, and the replica's live request holds the
    edge slab. After a short wait the first model retracts one of its own and runs the two in turn, as with a
    pool of its own; it does not stall until the replica's request ends."""
    table = SlabTable.create(tmp_path / "slabs", 8, 1, 1, [1])
    assert table.boundary == 5
    pools = [CachePool(8, boundary=table.side(0)), CachePool(8, boundary=table.side(1))]
    first, replica = (_engine(pool) for pool in pools)
    live = replica.submit(list(range(20, 28)), sampling_params=SamplingParams(max_output_tokens=2))
    replica.step()
    replica.step()  # its prefill holds slabs 8 to 5; it does not step again for now
    leases = [
        first.submit([1, 2, 3, 4, 5], sampling_params=SamplingParams(max_output_tokens=2)),
        first.submit([6, 7, 8, 9, 10], sampling_params=SamplingParams(max_output_tokens=2)),
    ]
    _drain([first], 10)
    assert [lease.take_result() for lease in leases] == [[6, 7], [11, 12]]
    assert table.boundary == 5 and not live.done
    _drain([replica])
    assert live.take_result() == [28, 29]


def test_a_replica_takes_back_its_edge_slab_when_the_move_is_off(tmp_path: Path) -> None:
    """Card 2 gave slab 5 for a move that card 1's live block holds; the first model's request ends another way
    (cancelled), so it stops wanting, and card 2 takes its slab back."""
    table = SlabTable.create(tmp_path / "slabs", 8, 2, 1, [1, 1])
    pools = [CachePool(8, boundary=table.side(side)) for side in range(3)]
    first, card1, _card2 = (_engine(pool) for pool in pools)
    live = card1.submit(list(range(20, 28)), sampling_params=SamplingParams(max_output_tokens=2))
    card1.step()
    card1.step()  # the prefill: 6 tokens, then the last 2 (prefix-cache alignment)
    lease = first.submit(list(range(1, 10)), sampling_params=SamplingParams(max_output_tokens=1))
    for _ in range(50):
        first.step()
        time.sleep(0.002)
    _wait_for(lambda: table.sides()[2][2] == 5)
    assert pools[2].held_slabs == 3
    first.cancel(lease)
    _drain([first])
    _wait_for(lambda: pools[2].held_slabs == 4 and table.sides()[2][2] == 0)
    _drain([first, card1])
    assert live.take_result() == [28, 29]
    assert table.boundary == 4  # card 1's fifth slab came from the first model


def test_a_replica_with_a_scrub_clears_each_slab_that_it_gives_or_gains(tmp_path: Path) -> None:
    """A replica clears the slabs that cross the boundary on its card: one that it gives, before the first model
    can use it, and one that it gains, before it uses it."""
    table = SlabTable.create(tmp_path / "slabs", 6, 1, 1, [1])
    assert table.boundary == 4  # three slabs each
    cleared: list[int] = []
    pools = [CachePool(6, boundary=table.side(0)), CachePool(6, boundary=table.side(1), scrub=cleared.extend)]
    first, replica = (_engine(pool) for pool in pools)
    # the first model needs a fourth slab: the replica clears slab 4, then gives it
    lease = first.submit(list(range(1, 8)), sampling_params=SamplingParams(max_output_tokens=1))
    _drain([first])
    assert lease.take_result() == [8]
    assert cleared == [4] and table.boundary == 5
    # the replica needs four slabs: the first model gives slabs 4 and 3, which the replica clears before it uses
    lease = replica.submit(list(range(10, 17)), sampling_params=SamplingParams(max_output_tokens=1))
    _drain([replica, first])
    assert lease.take_result() == [17]
    assert cleared == [4, 4, 3] and table.boundary == 3
    _wait_for(lambda: _agree(table, pools))


@pytest.mark.parametrize("seed", [7, 5, 11])
def test_three_engines_on_one_table_finish_every_request_under_churn(tmp_path: Path, seed: int) -> None:
    """Random bursts on the first model and two replicas, with repeated prompts (prefix hits): every request
    finishes with the right tokens, and the pools agree with the table once they are quiet. Waiting requests
    hold the prefixes they restored: a side that gives way releases them from its edge slab."""
    table = SlabTable.create(tmp_path / "slabs", 16, 2, 2, [1, 1])
    pools = [CachePool(16, boundary=table.side(side)) for side in range(3)]
    engines = [_engine(pool) for pool in pools]
    rng = random.Random(seed)
    prompts = [[rng.randrange(1, 900) for _ in range(rng.randrange(2, 9))] for _ in range(12)]
    leases = []
    boundaries = set()
    for _ in range(150):
        side = rng.randrange(3)
        prompt = list(rng.choice(prompts))  # repeats: prefix hits
        outputs = rng.randrange(1, 5)
        sampling = SamplingParams(max_output_tokens=outputs)
        leases.append((engines[side].submit(prompt, sampling_params=sampling), prompt, outputs))
        for engine in rng.sample(engines, 3):
            for _ in range(rng.randrange(0, 3)):
                engine.step()
        boundaries.add(table.boundary)
    _drain(engines, 60)
    for lease, prompt, outputs in leases:
        assert lease.take_result() == _expected(prompt, outputs)
    assert len(boundaries) > 2, f"the boundary moved: {sorted(boundaries)}"
    _wait_for(lambda: _agree(table, pools))
