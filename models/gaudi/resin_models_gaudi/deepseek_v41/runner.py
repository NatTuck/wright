"""DeepSeek-V4.1-Flash on Gaudi2 as a Resin managed runner: one tensor-parallel rank per process.

The Engine owns the paged cache: every plan names each row's WINDOW blocks (the 128-token pages a query can still
see) and HISTORY blocks (compressed entries). Every rank receives the same plan, runs the same launches and
collectives, and samples the same tokens from the all-gathered logits on its own device.

Python loads the model and builds its programs at startup. Every step then runs in the native executor
(``resin._resin.Ds41Executor``, ``deepseek_v41.native``): the Engine calls its forward, enqueue and release. Prefill
rows run one at a time in chunks of up to TMAX (2048) tokens; decode rows run as one batch of up to MAX_T; with
DSpark, a decode step of up to DSPARK_ROWS rows drafts and verifies. Engram hashes each position with the three
tokens before it: the executor's Engram context (``Ds41Context``) keeps each sequence's newest tokens, by position,
and the last three tokens of each WINDOW block, for a sequence that the Engine's prefix cache starts from cached
blocks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from resin import ForwardOutput, RunnerConfig, RunnerMetrics
from resin.backend.resources import CachePoolBinding
from resin.backend.speculative import SpeculativeStats
from resin.backend.topology import ParallelismCapabilities
from resin.models.deepseek_v41 import DeepSeekV41Config
from resin_models_gaudi import ranks
from resin_models_gaudi.deepseek_v41.cache import (
    CacheGeometry,
)
from resin_models_gaudi.deepseek_v41.model import (
    DSPARK_ROWS,
    MAX_T,
    TMAX,
    DeepSeekV41Gaudi,
)
from resin_models_gaudi.deepseek_v41.native import build_executor
from resin_models_gaudi.loading.progress import info

PREFILL_CHUNK_SIZE = TMAX
MAX_BATCH = 64
# decode continuations cross cache blocks; a chain ends every this many tokens of a row so that the prefix cache
# can publish (it publishes between chains)
CONTINUATION_PUBLISH_TOKENS = 2048
# bounded replay's kept span: a prefill runs the layers after the cut only on a prompt's last this many positions
REPLAY_TOKENS = 256



class DeepSeekV41GaudiRunner:
    """``ManagedRunner`` over ``DeepSeekV41Gaudi``; tensor parallel over ``groups``' attention replica."""

    parallelism_capabilities = ParallelismCapabilities(attn_tp=True, expert_tp=True)

    def __init__(
        self,
        checkpoint: str,
        *,
        groups=None,
        card: int | None = None,
        max_batch_size: int = MAX_T,
        max_seq_len: int = 16384,
        parent_capacity: int | None = None,
        num_layers: int | None = None,
        engram: str = "card",
        engram_prewarm: bool = True,
        draft_tokens: int = 0,
        replay: int | None = None,
        defer_cache: bool = False,
        prefill_chunk_size: int = PREFILL_CHUNK_SIZE,
        native_ranks=None,
        log=info,
    ) -> None:
        """``replay``: bounded replay's kept span (prompt positions whose layers after the cut run; 0: off;
        None: ``RESIN_DS41_REPLAY``, default REPLAY_TOKENS). See ``DeepSeekV41Gaudi.launch_replay``.

        ``defer_cache``: load the weights and stop; :meth:`attach` then makes the cache arena and what
        depends on its capacity. Every rank attaches at the same time (the capacity is agreed over the ranks).

        ``native_ranks``: this process's ``resin._resin.GaudiRanks``, a tensor-parallel replica without Gloo (in
        place of ``groups``): the group makes the HCCL communicator and agrees the cache capacity, and after the
        start rank 0's native executor sends each runner call to the other ranks, which ``serve``."""
        if replay is None:
            replay = int(os.environ.get("RESIN_DS41_REPLAY", REPLAY_TOKENS))
        self.replay = replay
        if max_batch_size > MAX_BATCH:
            raise ValueError(f"max_batch_size exceeds {MAX_BATCH}")
        self.prefill_chunk_size = prefill_chunk_size
        self.groups = groups
        self.native_ranks = native_ranks
        if native_ranks is not None:
            if groups is not None:
                raise ValueError("native ranks take no parallel groups")
            from resin_models_gaudi.synapse.runtime import open_device

            self.device = open_device(card)
            if native_ranks.size > 1:
                native_ranks.ccl_init(self.device.glue)
            rank, nranks = native_ranks.rank, native_ranks.size
        else:
            self.device, rank, nranks = ranks.open_rank(groups, card, "DeepSeek-V4.1-Flash")
        self.rank, self.nranks = rank, nranks
        cfg = DeepSeekV41Config.from_hugging_face(
            json.loads((Path(checkpoint) / "config.json").read_text())
        )
        self.config = cfg
        self.max_batch_size, self.max_seq_len = max_batch_size, max_seq_len
        # DSpark: decode steps of up to DSPARK_ROWS rows verify each row's anchor and draft_tokens drafts
        if draft_tokens and not 0 < draft_tokens <= cfg.dspark_block_size:
            raise ValueError(
                f"draft_tokens: 1 to {cfg.dspark_block_size} (the DSpark block)"
            )
        speculative = draft_tokens > 0
        self.speculative_tokens = draft_tokens
        self.speculate = (
            speculative  # (off: plain decoding with the same graphs, for parity)
        )
        self.lookahead = (
            draft_tokens  # a verify step writes positions p .. p + draft_tokens
        )
        self._speculation = SpeculativeStats(draft_tokens)
        geometry = CacheGeometry.from_config(
            cfg,
            layers=num_layers,
            max_write=TMAX,
            draft_layers=cfg.num_nextn_predict_layers if speculative else 0,
        )

        if draft_tokens + 1 > geometry.max_pending_columns:
            raise ValueError(
                f"draft_tokens: a verify step writes at most {geometry.max_pending_columns} positions of a row "
                "(its pending-state slots)"
            )

        self._parent_capacity = parent_capacity

        barrier = None
        if native_ranks is not None and nranks > 1:
            barrier = native_ranks.barrier
        elif groups is not None and nranks > 1:
            barrier = lambda: ranks.barrier(groups.replica.handle)  # noqa: E731
        self._barrier = barrier
        self.geometry = geometry
        self.model = DeepSeekV41Gaudi(
            self.device, checkpoint, cfg, capacity=self._fit, max_seq_len=max_seq_len, rank=rank, nranks=nranks,
            layers=num_layers, engram=engram, barrier=barrier, log=log if rank == 0 else (lambda *_: None),
            dspark=speculative, replay=replay, defer_cache=True, engram_prewarm=engram_prewarm,
        )  # fmt: skip
        if barrier is not None:
            barrier()
        self.vocab_size = cfg.vocab_size
        self.executor = None  # the native executor, made at attach
        if not defer_cache:
            self.attach()

    @property
    def own_slab_rows(self) -> int:
        """This model's slab, for planning an arena that other models share."""
        return self.geometry.own_slab_rows

    @property
    def slab_unit(self) -> int:
        """A shared slab is a multiple of this many rows (whole 1024-row arena blocks)."""
        return 1024

    @property
    def workspace_reserve(self) -> int:
        return self.model.workspace_reserve

    def shared_capacity(self, free: int, slab_rows: int) -> int:
        """The slabs of ``slab_rows`` rows, with this model's pending state for each, that ``free`` bytes
        hold (for an arena that other models share), less ``parent_capacity`` when that is smaller."""
        return ranks.requested_capacity(
            self.geometry.with_slab_rows(slab_rows).capacity_for(free), self._parent_capacity
        )

    def _fit(self, free: int) -> int:
        """the arena every rank can hold: the smallest free memory over the ranks"""
        if self.native_ranks is not None:
            free = self.native_ranks.min(free)
        else:
            free = ranks.min_over_ranks(free, self.groups)
        return ranks.requested_capacity(
            self.geometry.capacity_for(free), self._parent_capacity
        )

    def attach(
        self,
        capacity: int | None = None,
        *,
        slab_rows: int | None = None,
        arena: int | None = None,
    ) -> None:
        """Make the cache arena: ``capacity`` slabs, or the most that every rank's free memory holds (less
        ``parent_capacity`` when that is smaller); then the recipes, the cache layout and the runner config.
        ``slab_rows``: slabs of that many rows, which another model's slabs share. ``arena``: the address of
        a shared arena of ``capacity`` slabs (see ``DeepSeekV41Gaudi.attach``)."""
        if slab_rows is not None:
            self.geometry = self.geometry.with_slab_rows(slab_rows)
        cfg, geometry, replay = self.config, self.geometry, self.replay
        self.model.attach(
            self._fit if capacity is None else capacity, slab_rows=slab_rows, arena=arena
        )
        if self._barrier is not None:
            self._barrier()
        self.cache_layout = geometry.layout(self.model.capacity, replay)
        self._binding = CachePoolBinding(
            self.cache_layout, resource="DeepSeek-V4.1-Flash Gaudi cache"
        )
        self.runner_config = RunnerConfig(
            max_batch_tokens=self.prefill_chunk_size,
            prefill_chunk_size=self.prefill_chunk_size,
            continuation_prefill=True,
            cache_write_lookahead_tokens=self.lookahead,
            draft_tokens=self.speculative_tokens,
            cache_layout=self.cache_layout,
            # wider decode steps write one token a row and lease no lookahead (_decode_lookahead), so the Engine
            # keeps their continuations until a block's last token
            max_speculative_rows=DSPARK_ROWS if self.speculative_tokens else None,
            continuation_extension=CONTINUATION_PUBLISH_TOKENS,
        )
        from resin._resin import Ds41Context

        self._context = Ds41Context(
            cfg.engram_max_ngram_size - 1, geometry.window_blocks(self.model.capacity)
        )
        # the native executor (deepseek_v41.native) runs every step: the Engine calls its forward, enqueue and
        # release
        self.executor = build_executor(self)
        self.forward = self.executor.forward
        if __import__("os").environ.get("RESIN_DS41_TIMINGS"):
            import time as _t
            _fwd = self.forward
            def _timed_forward(plan, *args, **kwargs):
                t0 = dict(self.executor.timings); w0 = _t.perf_counter()
                out = _fwd(plan, *args, **kwargs)
                w1 = _t.perf_counter(); t1 = self.executor.timings
                d = {k: round((t1[k] - t0[k]) / 1e6, 1) for k in t1}
                d["wall"] = round((w1 - w0) * 1e3, 1)
                d["phase"] = str(getattr(plan, "phase", "?"))
                print(f"TIMINGS {d}", flush=True)
                return out
            self.forward = _timed_forward
        self.enqueue = self.executor.enqueue
        self.release = self.executor.release
        self.metrics_snapshot = self.executor.metrics_snapshot  # the Engine's metrics, no Python
        self._speculation = _NativeStats(self.executor, self._speculation)
        if self.native_ranks is not None:
            self.executor.attach_ranks(self.native_ranks)
        self.compiles_at_start = self._compiles_seen = self.device.compile_count
        if __import__("os").environ.get("RESIN_DS41_DUMP"):
            import threading, time as _t
            def _dump_loop():
                p = "/tmp/other-stingray/dump_now"; last = 0
                while True:
                    try:
                        m = __import__("os").path.getmtime(p)
                        if m > last:
                            last = m; self.model.dump_step("req")
                    except FileNotFoundError:
                        pass
                    _t.sleep(0.5)
            threading.Thread(target=_dump_loop, daemon=True).start()

    @property
    def kv_pool_bytes(self) -> int:
        return self.model.arena_blocks * 1024 * 512

    @property
    def drafting(self) -> bool:
        """Whether a decode step of up to DSPARK_ROWS rows drafts and verifies (DSpark loaded and switched on)."""
        return self.executor is not None and self.executor.drafting

    @drafting.setter
    def drafting(self, on: bool) -> None:
        """Switch DSpark at run time, between steps. Off: every decode step decodes one token a row with the same
        programs and leases no write lookahead; plain steps keep the drafter's inputs current, so drafts are good
        again when it goes back on. The Engine's ``max_speculative_rows`` must change with it: 0 while off,
        ``runner_config.max_speculative_rows`` while on (``runner_config`` keeps the values for drafting on).
        ``resin.bench.set_drafting`` does both. With ranks, switch rank 0: native ranks (``native_ranks``) send its
        decision with each step; a Gloo group (``groups``) cannot tell its other ranks."""
        if on and not self.speculative_tokens:
            raise ValueError("drafting needs a runner built with draft_tokens (DSpark)")
        if self.groups is not None and self.nranks > 1:
            raise NotImplementedError(
                "the drafting switch needs native ranks: the other ranks of a Gloo group cannot learn it"
            )
        if self.native_ranks is not None and self.rank > 0:
            raise ValueError("only rank 0 switches drafting: the other ranks take its decision with each step")
        if self.executor is None:
            raise RuntimeError("attach the cache before switching drafting")
        self.executor.drafting = bool(on)

    def _sync_speculation(self) -> None:
        """The native executor's acceptance counters into the shared SpeculativeStats."""
        if getattr(self, "executor", None) is not None:
            s = self._speculation
            s.blocks, s.drafts, s.accepted, s.emitted = self.executor.speculation

    @property
    def speculative_stats(self) -> dict[str, int | float]:
        self._sync_speculation()
        return self._speculation.snapshot(include_draft_tokens=True)

    def configure_metrics(self, *, enabled: bool = True) -> None:
        """Speculation counters always run, so there is nothing to switch on."""

    def metrics_snapshot(self) -> RunnerMetrics:
        self._sync_speculation()
        return RunnerMetrics(
            spec_draft_tokens=self._speculation.drafts,
            spec_accepted_tokens=self._speculation.accepted,
            spec_verify_rows=self._speculation.blocks,
        )

    def bind_resources(self, resources) -> None:
        self._binding.bind_resources(resources)

    @property
    def compiles_after_start(self) -> int:
        return self.device.compile_count - self.compiles_at_start


    def enqueue_distributed(self, plan, context, *, prepared=None, feedback=None):
        """``enqueue`` on every rank of the replica (tensor parallel: the same plan)"""
        if plan is None:
            return None
        return self.enqueue(plan, feedback=feedback)

    def forward_distributed(
        self, plan, context, *, prepared=None
    ) -> ForwardOutput | None:
        """Tensor parallel only: every rank of the replica runs the same plan."""
        return None if plan is None else self.forward(plan)

    def release(self, sequence_ids: list[int]) -> None:
        self._context.release(sequence_ids)

    def serve(self) -> int:
        """A follower rank (``native_ranks`` above rank 0): make the steps that rank 0 sends, until it closes the
        group. Returns the runner calls made."""
        return self.executor.serve()

    def close(self) -> None:
        if self.native_ranks is not None and self.rank == 0 and self.executor is not None:
            self.executor.close()  # the followers stop serving
        if self.nranks > 1:
            self.device.lib.rg_ccl_destroy()
        self.device.close()


class _NativeStats(SpeculativeStats):
    """``SpeculativeStats`` whose benchmark acceptance (``BenchmarkRunner``'s ``accepted_drafts``) goes to the native
    executor, which accepts the drafts."""

    def __init__(self, executor, stats: SpeculativeStats) -> None:
        super().__init__(stats.draft_tokens)
        self._executor = executor

    @property
    def benchmark_accepted_drafts(self) -> int | None:
        return self._executor.fixed_acceptance

    @benchmark_accepted_drafts.setter
    def benchmark_accepted_drafts(self, value: int | None) -> None:
        if getattr(self, "_executor", None) is not None:
            self._executor.fixed_acceptance = value


