"""GLM-5.3-Flash on Gaudi2 as a Resin managed runner: one tensor-parallel rank per process.

The Engine owns the paged cache: every plan names each row's KV blocks (128 tokens) and STATE blocks (4096
tokens). Every rank receives the same plan and runs the same launches and collectives, and samples the same tokens
from the all-gathered logits on its own device, so no token broadcast is needed; the driver's output is the one
the Engine reads. With a native rank group (``native_ranks``) rank 0's executor sends each runner call to the other
ranks; otherwise ``DistributedWorker`` broadcasts the plans.

The steps run in the native executor (``glm5_3_flash.native``, ``resin._resin.GlmExecutor``): prefill rows one at a
time in launches of up to 4096 tokens, decode rows as one batch of up to MAX_T (larger batches as consecutive MAX_T-row
steps), sampling and logprobs on the device with Resin's semantics (``common.sampling``). Python builds the model,
its graphs and programs at startup only.

With ``mtp`` = K, a decode step verifies each row's anchor and carried drafts (K + 1 columns per row, in groups of
``model.speculative_rows``) and emits the accepted prefix plus the target's next token. Every emitted token is the target's
sample at its position, so speculative output equals plain decoding, greedy or seeded; carried drafts only affect speed
(a row whose drafts do not continue from its anchor verifies zeros). With ``drafter`` = a DFlash 2 drafter directory
the drafts come from its block instead of the MTP layer (``mtp`` = the drafts a step verifies, 1 .. 7, 0 for all 7).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from resin import ForwardOutput, RunnerConfig
from resin.backend.resources import CachePoolBinding
from resin.backend.speculative import SpeculativeStats
from resin.backend.topology import ParallelismCapabilities
from resin.models.glm5_3_flash.config import Glm53Config
from resin_models_gaudi import ranks
from resin_models_gaudi.draft_choice import DraftChoice, StepCosts
from resin_models_gaudi.glm5_3_flash import native
from resin_models_gaudi.glm5_3_flash.cache import (
    STATE_TOKENS,
    CacheGeometry,
    tail_pools_for,
)
from resin_models_gaudi.glm5_3_flash.dflash import DrafterConfig
from resin_models_gaudi.glm5_3_flash.model import (
    MAX_T,
    MTP_T,
    PREFILL_T,
    WORKSPACE_RESERVE,
    Glm53Gaudi,
)
from resin_models_gaudi.glm5_3_flash.weights import RankSlices
from resin_models_gaudi.loading.progress import info

PREFILL_CHUNK_SIZE = 4096
# tokens per Engine prefill step that run as whole launches of the compiled prefill widths
PREFILL_CHUNK_SIZES = (1024, 2048, PREFILL_CHUNK_SIZE)
MAX_BATCH = 64
MTP_MAX = 3  # drafts of the MTP layer a step verifies, at most


def step_cost_terms(nranks: int, K: int, drafter: bool, group: int) -> dict[str, float]:
    """``step_costs``' fit: plain steps plain_0 + plain_row x rows, verify steps scale x launches x (launch_0 +
    launch_row x min(rows, group)) (also the native draft policy's ``StepCosts``)."""
    eight = nranks >= 8
    plain_0, plain_row = (7.6, 1.04) if eight else (8.8, 1.53)
    if drafter:
        launch_0, launch_row = 11.5, 13.1 * (K + 1) / 8
        scale = 0.84 if eight else 1.0
    else:
        launch_0, launch_row = (11.2, 4.55) if eight else (12.7, 6.4)
        launch_row *= (K + 1) / 4
        scale = 1.0
    return {"plain_0": plain_0, "plain_row": plain_row, "launch_0": launch_0, "launch_row": launch_row,
            "scale": scale, "group": group}  # fmt: skip


def step_costs(nranks: int, K: int, drafter: bool, group: int) -> StepCosts:
    """GLM-5.3-Flash decode step times (ms) by rows, which DraftChoice starts from before it measures steps: plain
    steps, and verify steps of K drafts a row in launches of up to ``group`` rows (more rows run one launch after
    another), from fits at 2K to 3K tokens of context (``step_cost_terms``)."""
    c = step_cost_terms(nranks, K, drafter, group)
    plain_0, plain_row, launch_0, launch_row, scale = (
        c["plain_0"], c["plain_row"], c["launch_0"], c["launch_row"], c["scale"]
    )

    def plain(rows: int) -> float:
        return plain_0 + plain_row * rows

    def verify(rows: int) -> float:
        launches = math.ceil(rows / group)
        return scale * launches * (launch_0 + launch_row * min(rows, group))

    return StepCosts(plain, verify)


def lookahead_tokens(draft_tokens: int, drafter: bool) -> int:
    """Positions past a row's token that a speculative step of ``draft_tokens`` drafts writes: p .. p + K (the
    DFlash drafter's cache entries with them, its block in scratch pages); an MTP step's draft chain also writes
    MTP keys up to p + 2K - 1."""
    if not draft_tokens:
        return 0
    return draft_tokens if drafter else 2 * draft_tokens - 1
# decode continuations cross KV blocks; a chain ends every this many tokens of a row (one STATE block, where
# a prefix can be reused) so that the prefix cache can publish (it publishes between chains)
CONTINUATION_PUBLISH_TOKENS = STATE_TOKENS


class Glm53FlashGaudiRunner:
    """``ManagedRunner`` over ``Glm53Gaudi``; tensor parallel over ``groups``' attention replica."""

    parallelism_capabilities = ParallelismCapabilities(attn_tp=True, expert_tp=True)

    def __init__(
        self,
        checkpoint: str,
        *,
        groups=None,
        card: int | None = None,
        max_batch_size: int = MAX_T,
        max_seq_len: int = 65536,
        parent_capacity: int | None = None,
        num_layers: int | None = None,
        mtp: int = 0,
        drafter: str | Path | None = None,
        defer_cache: bool = False,
        native_ranks=None,
        log=info,
    ) -> None:
        """``defer_cache``: load the weights and stop; :meth:`attach` then makes the cache arena and what
        depends on its capacity. Every rank attaches at the same time (the capacity is agreed over the ranks).

        ``native_ranks``: this process's ``resin._resin.GaudiRanks``, a tensor-parallel replica without Gloo (in
        place of ``groups``): the group makes the HCCL communicator and agrees the cache capacity, and after the
        start rank 0's native executor sends each runner call (with its draft decision) to the other ranks, which
        ``serve``."""
        if max_batch_size > MAX_BATCH:
            raise ValueError(f"max_batch_size exceeds {MAX_BATCH}")
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
            self.device, rank, nranks = ranks.open_rank(groups, card, "GLM-5.3-Flash")
        self.rank, self.nranks = rank, nranks
        raw = json.loads((Path(checkpoint) / "config.json").read_text())
        cfg = Glm53Config.from_dict(raw)
        self.config = cfg
        self.max_batch_size, self.max_seq_len = max_batch_size, max_seq_len
        dfl = None if drafter is None else DrafterConfig.load(drafter, cfg)
        if dfl is None:
            if mtp > MTP_MAX:
                raise ValueError(f"glm5_3_flash on Gaudi takes at most {MTP_MAX} draft tokens, not {mtp}")
            self.mtp = self.K = mtp
        else:
            if not 0 <= mtp <= dfl.drafts:
                raise ValueError(f"the drafter makes up to {dfl.drafts} drafts a step, not {mtp}")
            self.mtp, self.K = 0, mtp or dfl.drafts
        self.speculative_tokens = self.K
        self.lookahead = self.lookahead_tokens(self.K, dfl is not None)
        # decode steps of up to this many rows verify drafts. With MTP, wider ones decode plainly (and store the MTP
        # keys). The DFlash 2 drafter verifies every decode step, model.speculative_rows rows at a time.
        if not self.K:
            self.spec_rows = 0
        else:
            self.spec_rows = MTP_T // (self.mtp + 1) if self.mtp else max_batch_size
        sl = RankSlices.of(cfg, rank, nranks)
        geometry = CacheGeometry.from_config(
            cfg,
            sl.kda_heads.stop - sl.kda_heads.start,
            layers=num_layers,
            mtp=self.mtp > 0,
            tail_pools=tail_pools_for(self.K),
            draft_layers=0 if dfl is None else dfl.layers,
            draft_layer_rows=0 if dfl is None else dfl.layer_rows,
            draft_window=0 if dfl is None else dfl.window,
            draft_write=PREFILL_T,
        )
        self.geometry = geometry
        self._parent_capacity = parent_capacity
        self.model = Glm53Gaudi(
            self.device,
            checkpoint,
            cfg,
            capacity=self._fit,
            max_seq_len=max_seq_len,
            rank=rank,
            nranks=nranks,
            layers=num_layers,
            mtp=mtp,
            drafter=drafter,
            log=log if rank == 0 else (lambda *_: None),
            defer_cache=True,
        )
        assert self.model.geo == geometry
        self._speculation = self.model.speculation if self.K else SpeculativeStats(0)
        self._barrier()
        self.vocab_size = cfg.vocab_size
        self.continuations = 0
        # RESIN_GLM_DRAFT_POLICY (the native executor's draft policy, resin_gaudi::draft_choice): auto (default:
        # drafts where each row's acceptance makes them faster than plain steps), always (every step of up to
        # spec_rows rows verifies drafts) or never. Every rank must have the same value, so they take rank 0's. With
        # a native rank group rank 0 decides for every rank; otherwise the ranks decide with rank 0's measured step
        # times, which it shares at the same steps on every rank.
        mode = os.environ.get("RESIN_GLM_DRAFT_POLICY", "auto")
        if native_ranks is not None and nranks > 1:
            mode = bytes(native_ranks.broadcast(mode.encode())).decode()
        else:
            mode = ranks.from_first_rank(mode, groups)
        if mode not in DraftChoice.MODES:
            raise ValueError(f"draft policy {mode!r}: one of {', '.join(DraftChoice.MODES)}")
        self.policy = mode
        if self.K and rank == 0:
            info(f"draft policy: {mode} ({self.K} drafts, steps of up to {self.spec_rows} rows)")
        if not defer_cache:
            self.attach()

    def _barrier(self) -> None:
        """Every rank of the replica waits for the others here."""
        if self.nranks == 1:
            return
        if self.native_ranks is not None:
            self.native_ranks.barrier()
        elif self.groups is not None:
            ranks.barrier(self.groups.replica.handle)

    def serve(self) -> int:
        """A follower rank (``native_ranks`` above rank 0): make the steps that rank 0 sends, until it closes the
        group. Returns the runner calls made."""
        return self.executor.serve()

    def _fit(self, free: int) -> int:
        """the arena every rank can hold: the smallest free memory over the ranks"""
        if self.native_ranks is not None:
            free = self.native_ranks.min(free)
        else:
            free = ranks.min_over_ranks(free, self.groups)
        geometry = self.geometry
        return ranks.requested_capacity(
            geometry.capacity_for(free, geometry.slab_bytes), self._parent_capacity
        )

    @property
    def own_slab_rows(self) -> int:
        """This model's slab, for planning an arena that other models share."""
        return self.geometry.own_slab_rows

    @property
    def slab_unit(self) -> int:
        """A shared slab is a multiple of this many rows."""
        return self.geometry.slab_unit

    @property
    def workspace_reserve(self) -> int:
        return WORKSPACE_RESERVE

    def shared_capacity(self, free: int, slab_rows: int) -> int:
        """The slabs of ``slab_rows`` rows that ``free`` bytes hold (for an arena that other models share), less
        ``parent_capacity`` when that is smaller."""
        geometry = self.geometry.with_slab_rows(slab_rows)
        return ranks.requested_capacity(
            geometry.capacity_for(free, geometry.slab_bytes), self._parent_capacity
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
        ``slab_rows``: slabs of that many rows, which another model's slabs share. ``arena``: the address of a
        shared arena of ``capacity`` slabs (see ``Glm53Gaudi.attach``)."""
        if slab_rows is not None:
            self.geometry = self.geometry.with_slab_rows(slab_rows)
        geometry = self.geometry
        self.model.attach(
            self._fit if capacity is None else capacity, slab_rows=slab_rows, arena=arena
        )
        assert self.model.geo == geometry
        self._barrier()
        self.cache_layout = geometry.layout(self.model.capacity)
        self._binding = CachePoolBinding(self.cache_layout, resource="GLM-5.3-Flash Gaudi cache")
        self.runner_config = RunnerConfig(
            max_batch_tokens=PREFILL_CHUNK_SIZE,
            prefill_chunk_size=PREFILL_CHUNK_SIZE,
            continuation_prefill=True,
            cache_write_lookahead_tokens=self.lookahead,
            draft_tokens=self.K,
            cache_layout=self.cache_layout,
            # plain decode steps (more rows) write one token a row and lease no lookahead (_decode_lookahead)
            max_speculative_rows=self.spec_rows if self.mtp else None,
            continuation_extension=CONTINUATION_PUBLISH_TOKENS,
        )
        # the native executor (glm5_3_flash.native) runs every step: the Engine calls its forward, enqueue and
        # release
        if not native.available(self):
            raise RuntimeError(
                "GLM-5.3-Flash runs on the native executor: build resin with the gaudi feature, on a device with "
                "the runtime's glue"
            )
        self.executor = native.build_executor(self)
        self.forward = self.executor.forward
        self.enqueue = self.executor.enqueue
        self.release = self.executor.release
        self.metrics_snapshot = self.executor.metrics_snapshot  # the Engine's metrics, no Python
        self._speculation = _NativeStats(self.executor, self._speculation)
        if self.native_ranks is not None:
            self.executor.attach_ranks(self.native_ranks)
        self.compiles_at_start = self._compiles_seen = self.device.compile_count

    @staticmethod
    def lookahead_tokens(draft_tokens: int, drafter: bool) -> int:
        return lookahead_tokens(draft_tokens, drafter)

    @property
    def kv_pool_bytes(self) -> int:
        """The KV pool on one card."""
        return self.model.arena.nbytes

    @property
    def speculative_stats(self) -> dict[str, int | float]:
        if getattr(self, "executor", None) is not None:  # the native executor keeps the counters
            s = self._speculation
            s.blocks, s.drafts, s.accepted, s.emitted = self.executor.speculation
        return self._speculation.snapshot(include_draft_tokens=True)

    def configure_metrics(self, *, enabled: bool = True) -> None:
        """Speculation counters always run, so there is nothing to switch on."""

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

    def forward_distributed(self, plan, context, *, prepared=None) -> ForwardOutput | None:
        """Tensor parallel only: every rank of the replica runs the same plan."""
        return None if plan is None else self.forward(plan)

    def close(self) -> None:
        if self.native_ranks is not None and self.rank == 0 and getattr(self, "executor", None) is not None:
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


