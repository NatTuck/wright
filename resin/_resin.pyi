from _typeshed import Incomplete
from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Any, Final, final
__version__: Final[str]

@final
class Audio:
    def __new__(cls, /, *, source: Any) -> Audio: ...
    @property
    def source(self, /) -> Any: ...

@final
class BatchPhase:
    DECODE: Final[BatchPhase]
    PREFILL: Final[BatchPhase]
    def __eq__(self, value: object, /) -> bool: ...
    def __hash__(self, /) -> int: ...
    def __int__(self, /) -> int: ...
    def __ne__(self, value: object, /) -> bool: ...
    def __repr__(self, /) -> str: ...

@final
class BatchPlan:
    """
    Immutable execution plan for one prefill or decode step.
    
    This runs one forward pass on a batch of sequences. `token_ids` contains
    the newly scheduled input tokens, packed by sequence into one flat array;
    `token_indptr` is the CSR row pointer, so sequence `i` owns
    `token_ids[token_indptr[i]:token_indptr[i + 1]]`.
    
    - `context_lengths`: Tokens already processed per sequence; new token
      positions start here.
    - `prefill_lengths`: Prefill endpoints, fixed across chunks, including
      retraction replay. Defaults to each row's scheduled end.
    - `cache_groups`: One placement plan per `CacheLayout` group, covering
      every sequence.
    """
    def __new__(cls, /, *, sequence_ids: "Sequence[int] | npt.NDArray[np.uint64]", context_lengths: "Sequence[int] | npt.NDArray[np.uint32]", token_ids: "Sequence[int] | npt.NDArray[np.uint32]", token_indptr: "Sequence[int] | npt.NDArray[np.uint32]", prefill_prompt: "Sequence[int] | npt.NDArray[np.uint32] | None" = None, prefill_lengths: "Sequence[int] | npt.NDArray[np.uint32] | None" = None, phase: BatchPhase = ..., batch_id: int = 0, batch_key: int |None = None, cache_groups: Sequence[CacheGroupPlan] |None = None, sampling_mask: "Sequence[bool] | npt.NDArray[np.bool_] | None" = None, sampling_params: Sequence[SamplingParams] |None = None, media: "Sequence[MediaInput | None] | None" = None, prompt_score_indices: "Sequence[int] | npt.NDArray[np.uint32] | None" = None, prompt_score_token_ids: "Sequence[int] | npt.NDArray[np.uint32] | None" = None, prompt_score_indptr: "Sequence[int] | npt.NDArray[np.uint32] | None" = None, prompt_score_top_counts: "Sequence[int] | npt.NDArray[np.uint32] | None" = None, prompt_target_indptr: "Sequence[int] | npt.NDArray[np.uint32] | None" = None, prompt_target_token_ids: "Sequence[int] | npt.NDArray[np.uint32] | None" = None) -> BatchPlan: ...
    def __repr__(self, /) -> str: ...
    @staticmethod
    def _from_wire(data: "npt.NDArray[np.uint8]", requests: "dict[int, Any]", pending: "dict[int, Any]") -> BatchPlan: ...
    def _to_wire(self, /) -> "npt.NDArray[np.uint8]": ...
    @property
    def batch_id(self, /) -> int:
        """
        Monotonic identifier for tracing and correlating runner steps.
        """
    @property
    def batch_key(self, /) -> int |None:
        """
        Optional key that prevents incompatible requests from sharing a batch.
        """
    @property
    def cache_groups(self, /) -> tuple:
        """
        Ordered physical placement plans corresponding to CacheLayout groups.
        There is one plan per layout group, not one plan per sequence; each plan
        contains one block-table row for every sequence in this batch.
        """
    @property
    def context_lengths(self, /) -> Incomplete:
        """
        Number of tokens already processed for each sequence.
        """
    @property
    def media(self, /) -> tuple:
        """
        Opaque runner inputs, one per sequence. The scheduler never inspects them.
        """
    @property
    def needs_logprobs(self, /) -> bool:
        """
        Check natively so token-only graph replay does not box per-row policies.
        """
    @property
    def needs_prompt_logprobs(self, /) -> bool: ...
    @property
    def num_sequences(self, /) -> int:
        """
        Number of sequences (batch size).
        """
    @property
    def num_tokens(self, /) -> int:
        """
        Number of flattened scheduled tokens.
        """
    @property
    def phase(self, /) -> BatchPhase:
        """
        Whether this is a prefill or decode batch. Mixed-chunk is not supported.
        """
    @property
    def positions(self, /) -> Incomplete:
        """
        Absolute position of every flattened scheduled token.
        """
    @property
    def prefill_lengths(self, /) -> Incomplete:
        """
        Final history endpoint for each prefill, including retracted generated inputs.
        Only meaningful during prefill; constant across that history's chunks.
        """
    @property
    def prompt_score_indices(self, /) -> Incomplete: ...
    @property
    def prompt_score_indptr(self, /) -> Incomplete: ...
    @property
    def prompt_score_token_ids(self, /) -> Incomplete: ...
    @property
    def prompt_score_top_counts(self, /) -> Incomplete: ...
    @property
    def prompt_target_indptr(self, /) -> Incomplete: ...
    @property
    def prompt_target_token_ids(self, /) -> Incomplete: ...
    @property
    def sampling_mask(self, /) -> Incomplete:
        """
        Which sequences should generate a sampled token. Used for chunked
        prefill, when some sequences only advance the cache.
        """
    @property
    def sampling_params(self, /) -> tuple:
        """
        Effective sampling parameters, one per sequence.
        """
    @property
    def sequence_ids(self, /) -> Incomplete:
        """
        Stable request identifier for each sequence.
        """
    @property
    def token_ids(self, /) -> Incomplete:
        """
        Flattened scheduled token IDs, grouped by sequence.
        """
    @property
    def token_indptr(self, /) -> Incomplete:
        """
        Offsets into `token_ids` for each sequence.
        """
    def token_range(self, /, sequence_index: int) -> tuple[int, int]:
        """
        Return the half-open range in `token_ids` for one sequence.
        """

@final
class CacheGroupFamily:
    """
    Whether a group stores token history or recurrent checkpoints.
    """
    HISTORY: Final[CacheGroupFamily]
    STATE: Final[CacheGroupFamily]
    def __eq__(self, value: object, /) -> bool: ...
    def __hash__(self, /) -> int: ...
    def __int__(self, /) -> int: ...
    def __ne__(self, value: object, /) -> bool: ...
    def __repr__(self, /) -> str: ...

@final
class CacheGroupLayout:
    """
    Geometry and retention policy for one type of cache.
    
    Cache groups share a pool of equal-sized slabs. `blocks_per_slab` is how
    many of this group's blocks fit in one slab.
    """
    def __new__(cls, /, block_tokens: int, *, blocks_per_slab: int = 1, retention: CacheRetention = ..., window_tokens: int |None = None, family: CacheGroupFamily = ..., read_window_tokens: int |None = None) -> CacheGroupLayout: ...
    def __repr__(self, /) -> str: ...
    @property
    def block_tokens(self, /) -> int:
        """
        Number of token positions represented by one logical block.
        """
    @property
    def blocks_per_slab(self, /) -> int:
        """
        Number of this group's blocks packed into one slab.
        """
    @property
    def family(self, /) -> CacheGroupFamily:
        """
        Whether this group stores token history or recurrent checkpoints.
        """
    @property
    def read_window_tokens(self, /) -> int |None:
        """
        Logical read window including the current token. Defaults to the full
        retained window; reserve extra `window_tokens` for simultaneous writes.
        """
    @property
    def retention(self, /) -> CacheRetention:
        """
        Lifetime policy for logical blocks in this group.
        """
    @property
    def window_resident_block_count(self, /) -> int |None:
        """
        Maximum resident logical blocks per sequence for a window group.
        """
    @property
    def window_tokens(self, /) -> int |None:
        """
        Retained suffix length for `WINDOW` groups, otherwise `None`.
        """

@final
class CacheGroupPlan:
    """
    Plan for one type of cache in a BatchPlan.
    
    Each `CacheGroupLayout` describes full-attention history, sliding-window
    history, or recurrent state. This plan maps every batch sequence to that
    group's physical blocks and token write locations.
    
    - `group_index`: Zero-based index of the matching `CacheGroupLayout`.
    - `blocks_per_slab`: Number of this group's blocks packed into each slab.
    - `block_indptr`: CSR row pointer selecting each sequence's chronological
      block-table entries.
    - `block_slab_ids`: One-based slab IDs for the flattened block table.
    - `block_slotnums`: Zero-based block slots within those slabs.
    - `block_table_start_indices`: Logical index of each row's first block;
      zero for full history, advancing for window and recurrent rings. Block
      `b` starts at token position `b * CacheGroupLayout.block_tokens`.
    - `append_slab_ids`: Destination slab for each flattened `BatchPlan` token.
    - `append_slotnums`: Destination block slot for each token.
    - `append_block_offsets`: Token offset within each destination block.
    """
    def __new__(cls, /, *, group_index: int, block_indptr: "Sequence[int] | npt.NDArray[np.uint32]", block_slab_ids: "Sequence[int] | npt.NDArray[np.uint32]", block_slotnums: "Sequence[int] | npt.NDArray[np.uint32]", block_table_start_indices: "Sequence[int] | npt.NDArray[np.uint32]", append_slab_ids: "Sequence[int] | npt.NDArray[np.uint32]", append_slotnums: "Sequence[int] | npt.NDArray[np.uint32]", append_block_offsets: "Sequence[int] | npt.NDArray[np.uint32]", blocks_per_slab: int = 1) -> CacheGroupPlan: ...
    def __repr__(self, /) -> str: ...
    @property
    def append_block_offsets(self, /) -> Incomplete:
        """
        Token offset within each token's destination block. Entry `k`
        corresponds to `BatchPlan.token_ids[k]` and is less than the matching
        `CacheGroupLayout.block_tokens`.
        """
    @property
    def append_slab_ids(self, /) -> Incomplete:
        """
        One-based slab ID of each token's write target. Entry `k`
        corresponds to `BatchPlan.token_ids[k]`.
        """
    @property
    def append_slotnums(self, /) -> Incomplete:
        """
        Zero-based slot number of each token's write target. Entry `k`
        corresponds to `BatchPlan.token_ids[k]`.
        """
    @property
    def block_indptr(self, /) -> Incomplete:
        """
        CSR row pointer into the block-address arrays. Its length is one more
        than the batch sequence count.
        """
    @property
    def block_slab_ids(self, /) -> Incomplete:
        """
        One-based slab ID for each flattened block-table entry. Pair with
        `block_slotnums`; slab zero is reserved as null.
        """
    @property
    def block_slotnums(self, /) -> Incomplete:
        """
        Zero-based slot number for each flattened block-table entry. Pair with
        `block_slab_ids`.
        """
    @property
    def block_table_start_indices(self, /) -> Incomplete:
        """
        Absolute logical-block index at which each sequence's table row starts.
        This is zero for full-attention history and advances for sliding-window
        and recurrent rings. It has one entry per sequence and is unrelated to
        the physical slab ID.
        """
    @property
    def blocks_per_slab(self, /) -> int:
        """
        Number of this group's blocks packed into one slab. Valid slot numbers
        are `0 <= slotnum < blocks_per_slab`.
        """
    @property
    def group_index(self, /) -> int:
        """
        Zero-based index of this plan's matching CacheGroupLayout. BatchPlan
        orders group plans by this index and rejects gaps or reordering.
        """

@final
class CacheLayout:
    """
    Immutable grouped-cache geometry owned by one Engine.
    
    - `parent_capacity`: Number of physical slabs shared by all cache groups.
    - `groups`: Cache-group layouts in the stable order used by `BatchPlan`.
    - `replay_suffix_tokens`: Final prompt tokens rebuilt after a prefix-cache
      hit. Models must rebuild omitted cache state before reading it.
    """
    def __new__(cls, /, *, parent_capacity: int, groups: Sequence[CacheGroupLayout], replay_suffix_tokens: int = 0) -> CacheLayout: ...
    def __repr__(self, /) -> str: ...
    def estimated_parent_count(self, /, context_lengths: Sequence[int], *, shared_prefix_tokens: int = 0) -> int:
        """
        Physical slabs needed for contexts in an empty pool.
        Applies retention and packing across requests without allocating pages;
        an aligned shared prefix may be declared, and demand may exceed capacity.
        """
    @property
    def group_count(self, /) -> int:
        """
        Number of logical cache groups.
        """
    def group_kernel_page_capacity(self, /, group_index: int) -> int: ...
    def group_kernel_page_count(self, /, group_index: int) -> int: ...
    @property
    def groups(self, /) -> tuple:
        """
        Ordered immutable cache-group descriptors.
        """
    @property
    def parent_capacity(self, /) -> int:
        """
        Number of fungible physical slabs in the pool.
        """
    @property
    def replay_suffix_tokens(self, /) -> int:
        """
        Minimum final prompt suffix rebuilt after every prefix-cache hit.
        """

@final
class CachePool:
    """
    A pool of equal-sized cache slabs that several Engines share: one Engine
    for each model that the process serves.
    
    Give it to each Engine (`Engine(..., cache_pool=pool)`). Every runner's
    `CacheLayout` must declare `parent_capacity` slabs, and all of them address
    one device arena. A slab belongs to one Engine while it holds that Engine's
    blocks and returns to the pool when it is free. When an Engine needs a slab
    and none is free, the least recently used cached prefix of any Engine goes
    first; no Engine evicts another Engine's live requests. An Engine whose
    growth finds no slab waits while other Engines' live requests hold them.
    `cache_floor` reserves slabs for an Engine.
    
    With `boundary` (a side of a `SlabTable`), the pool shares the arena with pools in other processes: it places
    blocks only in its side of the table's boundary slab, and when the other side wants slabs it gives its edge
    slab (after evicting the prefixes in it), so the boundary moves. A thread of its own does that work between
    steps. `scrub(slabs)` (a list of slab ids), if given, clears those slabs on the device and returns when they
    are clear: the pool calls it for every slab that it gains or gives, before the slab is used on either side.
    """
    def __new__(cls, /, parent_capacity: int, *, boundary: SlabSide |None = None, scrub: Any |None = None) -> CachePool: ...
    def __repr__(self, /) -> str: ...
    def bound_slabs(self, /) -> list[int]:
        """
        Slabs that each Engine's blocks occupy (live or cached), in join order.
        """
    @property
    def engines(self, /) -> int:
        """
        The Engines that joined, in order.
        """
    @property
    def free_slabs(self, /) -> int: ...
    @property
    def held_slabs(self, /) -> int:
        """
        Slabs the pool holds: all of them, or with a boundary its side's.
        """
    @property
    def parent_capacity(self, /) -> int:
        """
        Physical slabs, without the null slab.
        """
    @property
    def pool_id(self, /) -> int:
        """
        The pool identity that every Engine's `ResourceBinding` carries.
        """

@final
class CacheRetention:
    """
    Cache-block lifetime policy.
    
    For history groups, `HISTORY` retains every logical block and `WINDOW`
    retains the suffix selected by `CacheGroupLayout.window_tokens`. Recurrent
    state groups use `HISTORY` and retain their two-page checkpoint ring.
    """
    HISTORY: Final[CacheRetention]
    WINDOW: Final[CacheRetention]
    def __eq__(self, value: object, /) -> bool: ...
    def __hash__(self, /) -> int: ...
    def __int__(self, /) -> int: ...
    def __ne__(self, value: object, /) -> bool: ...
    def __repr__(self, /) -> str: ...

@final
class CardGate:
    """
    `resin_models_gaudi.synapse.card_gate.CardGate` builds it: three event slots from `first` (the replica's two
    timed), the replica's `share` of the card's time while both models have work, and the TP model's `holdoff`.
    """
    def __new__(cls, /, glue: GaudiGlue, first: int, share: float = 0.0, holdoff: float = ...) -> CardGate: ...
    def acquire(self, /, timeout: float) -> bool:
        """
        Wait up to `timeout` seconds for the replica's turn; False: not this time.
        """
    @property
    def counts(self, /) -> GateCounts: ...
    def release(self, /) -> None:
        """
        The replica's step is on the device.
        """
    @property
    def replica_slot(self, /) -> int: ...
    def tp_begin(self, /) -> None: ...
    @property
    def tp_busy(self, /) -> bool: ...
    def tp_end(self, /) -> None: ...
    @property
    def tp_slot(self, /) -> int: ...

@final
class DeepSeekV41Image:
    def __new__(cls, /, *, start: int, patches: Any, n_vit_h: int, n_vit_w: int, types: Sequence[int]) -> DeepSeekV41Image: ...
    @property
    def n_vit_h(self, /) -> int: ...
    @property
    def n_vit_w(self, /) -> int: ...
    @property
    def patches(self, /) -> Incomplete: ...
    @property
    def start(self, /) -> int: ...
    @property
    def types(self, /) -> tuple: ...

@final
class DeepSeekV41Media:
    def __new__(cls, /, *, token_ids: Sequence[int], images: Sequence[DeepSeekV41Image], image_token_id: int = 129264) -> DeepSeekV41Media: ...
    @property
    def cache_key(self, /) -> tuple: ...
    @property
    def digest(self, /) -> str: ...
    @property
    def image_token_id(self, /) -> int: ...
    @property
    def images(self, /) -> tuple: ...
    @property
    def token_ids(self, /) -> tuple: ...
    @property
    def token_spans(self, /) -> tuple: ...

@final
class Ds41Completion:
    """
    One enqueued decode step: `result` waits for its staged samples once and records each row's sampled token
    (the token at its position + 1) as the sequence's newest in the context.
    """
    @property
    def feedback(self, /) -> Ds41Feedback: ...
    def result(self, /) -> ForwardOutput: ...

@final
class Ds41Context:
    """
    DeepSeek-V4.1's Engram context (`resin_gaudi::ds41::context`): the tokens before each row's first position.
    """
    def __new__(cls, /, n: int, window_blocks: int) -> Ds41Context: ...
    def before(self, /, sequence: int, pos0: int, first: int, ids: Sequence[int]) -> Incomplete:
        """
        The tokens at positions `pos0 - n` .. `pos0 - 1` that Engram hashes, for a row whose WINDOW table starts at
        logical block `first` with window block ids `ids`.
        """
    def has(self, /, sequence: int) -> bool:
        """
        Whether `sequence` has a record (it is not released).
        """
    @property
    def n(self, /) -> int: ...
    def release(self, /, sequences: Sequence[int]) -> None:
        """
        Forget released sequences (their blocks' records stay).
        """
    def seen(self, /, sequence: int, pos0: int, tokens: Sequence[int]) -> None:
        """
        Record `tokens`, committed at positions `pos0` and after, in place of any recorded there and after.
        """

@final
class Ds41Executor:
    REGISTERS: Final[int]
    R_BYTES: Final[int]
    R_IMAGE: Final[int]
    R_KEPT: Final[int]
    R_SCORE_BYTES: Final[int]
    R_SCORE_COL: Final[int]
    R_SKIP: Final[int]
    R_STAGE: Final[int]
    R_TOKBYTES: Final[int]
    def __new__(cls, /, *, glue: GaudiGlue, fill: Ds41Fill, context: Ds41Context, layout: CacheLayout, shapes: dict[tuple[int, bool], dict], image: tuple[int, Incomplete, int, int], staging: tuple[Incomplete, int, int, tuple[int, int, int, int, int]], sample: tuple[Incomplete, int], geometry: tuple[int, int, int, int], model: dict, draft: dict |None = None, scoring: dict |None = None) -> Ds41Executor:
        """
        `shapes`: per (columns, prefill) a dict of `step` and `io` (`GaudiProgram`s) and `compact` (kept bucket ->
        the io section). `image`: (device table, pinned host copies, stride, first event slot); `staging`: (pinned
        sets, stride, first event slot, (tok, lp, top_idx, top_lp, bytes) offsets); `sample`: (pinned sampler
        table, offset of its `last`); `geometry`: (slab rows, WINDOW block rows, HISTORY block rows, WINDOW blocks
        per slab); `model`: a dict of the model's numbers.
        """
    def attach_gate(self, /, gate: CardGate) -> None:
        """
        A layout's card gate: every runner call is a TP step on this card, which a replica on it waits for.
        """
    def attach_ranks(self, /, ranks: GaudiRanks) -> None:
        """
        Join a tensor-parallel model's rank group: rank 0 drives (it sends each runner call to the others), and
        every other rank replays the calls (`serve`).
        """
    def close(self, /) -> None:
        """
        The driver: tell the followers to stop serving.
        """
    @property
    def continuations(self, /) -> int:
        """
        Enqueued steps that continued the previous one.
        """
    @property
    def drafting(self, /) -> bool:
        """
        DSpark's switch: while it is on, a decode step of up to DSPARK_ROWS rows drafts and verifies; while it is
        off, every decode step decodes one token a row and expects no lookahead lease (the Engine's
        `max_speculative_rows` must change with it). Plain steps keep the drafter's state current, so drafts are
        good again right after the switch goes back on. Switch it only between steps.
        """
    @drafting.setter
    def drafting(self, /, on: bool) -> None: ...
    def enqueue(self, /, plan: BatchPlan, *, feedback: Any |None = None) -> Any:
        """
        Submit a decode step and return without waiting for it: a continuation (`feedback`) reads its tokens from
        the previous step's samples on the device, or continues its verification. Other plans run synchronously.
        """
    @property
    def fixed_acceptance(self, /) -> int |None:
        """
        The benchmark's fixed acceptance (`SpeculativeStats.benchmark_accepted_drafts`): every row accepts this
        many drafts (None: the samples decide).
        """
    @fixed_acceptance.setter
    def fixed_acceptance(self, /, fixed: int |None) -> None: ...
    def forward(self, /, plan: BatchPlan) -> Any:
        """
        The Engine's synchronous step: prefill chunks, a decode step or a speculative step. With ranks, the driver
        sends the call to the followers first.
        """
    def metrics_snapshot(self, /) -> RunnerMetrics:
        """
        The runner's `metrics_snapshot` (the speculation counters), with no Python: the runner binds it.
        """
    @property
    def native_steps(self, /) -> int:
        """
        Runs of the native step programs so far.
        """
    def release(self, /, sequence_ids: Sequence[int]) -> None:
        """
        Forget released sequences (their Engram records; their blocks' records stay).
        """
    def serve(self, /) -> int:
        """
        A follower: replay the driver's runner calls until it closes the group. Returns the calls replayed.
        """
    @property
    def speculation(self, /) -> tuple[int, int, int, int]:
        """
        The verify steps' acceptance counters: (blocks, drafts, accepted, emitted).
        """
    @property
    def timings(self, /) -> dict[str, int]:
        """
        Host nanoseconds so far: fill, submit, image_wait, output_wait.
        """

@final
class Ds41Feedback:
    """
    A plain decode step's continuation: each row's next position; its tokens stay on the device.
    """
    @property
    def positions(self, /) -> Incomplete: ...

@final
class Ds41Fill:
    """
    The native step fill of one DeepSeek-V4.1 model: constructed once with what the tables depend on, then
    `add_shape` per step shape (table offsets) and `fill` per step.
    """
    def __new__(cls, /, *, heads: int, index_heads: int, sliding_window: int, layer_rows: Sequence[int], owners: Sequence[tuple[int, int, int]], pending: Sequence[int], pages: int, kmax: int, kc: int, candidate: tuple[int, int] |None, engram: tuple[Sequence[int], int, Sequence[int], int, int, int, Sequence[int], Sequence[int]] |None, cs0: Incomplete, cs1: Incomplete, radix_min: int, prefilter: tuple[int, int], dense_keys: int, geometry: tuple[int, int, int, int, int, int], engram_host: Sequence[tuple[Incomplete, Incomplete]] |None = None) -> Ds41Fill: ...
    def _engram_gather(self, /, layer: int, ids: Incomplete) -> Incomplete:
        """
        Gather + dequantize global row `ids` from host Engram table `layer` (verification surface for
        the oracle gate; the step path uses the same `engram_gather_into` core). Returns `(n, 256)` f32.
        """
    @staticmethod
    def _engram_lut() -> Incomplete:
        """
        The host Engram dequant LUT (256,256) f32: `LUT[scale, code] = OCP[code] * 2^(scale-127)`.
        A verification surface for the host-gather oracle tests; not used in the step path.
        """
    def add_shape(self, /, layout: dict) -> None:
        """
        Byte offsets of a shape's tables: a dict of the `Tables` fields.
        """
    def fill(self, /, t: int, host: Incomplete, column: Incomplete, pos0: Incomplete, real: Incomplete, window_first: Incomplete, window_indptr: Incomplete, window: Incomplete, window_ids: Incomplete, history_indptr: Incomplete, history: Incomplete, before_indptr: Incomplete, before: Incomplete, positions: Incomplete, tokens: Incomplete |None = None, with_tokens: bool = True, prefill: bool = False, floor: int |None = None) -> tuple[bool, bool, int, bool, bool, int, int, int, int, bool, bool]:
        """
        Write step `t`'s tables (`prefill`: a prefill shape) into `host` (`with_tokens`: also the tokens and their
        Engram rows; `floor`: no window reaches before that position); returns (sparse, cand, ixc, radix1, radix2,
        ix1, ix2, pf1, pf2, dense1, dense2).
        """
    def fill_draft(self, /, t: int, host: Incomplete, pos0: Incomplete, window_first: Incomplete, window_indptr: Incomplete, window: Incomplete, anchors: Incomplete, block: int, noise: int, stages: int) -> None:
        """
        Write decode shape `t`'s draft-step tables into `host` (`_fill_draft`): `anchors` per row, DSpark blocks of
        `block` tokens with `noise` after the anchor, through the last `stages` layers.
        """
    def fill_tokens(self, /, t: int, host: Incomplete, column: Incomplete, pos0: Incomplete, real: Incomplete, window_first: Incomplete, window_indptr: Incomplete, window: Incomplete, window_ids: Incomplete, history_indptr: Incomplete, history: Incomplete, before_indptr: Incomplete, before: Incomplete, n: int, tokens: Incomplete |None = None, prefill: bool = False) -> None:
        """
        Write the tables of step `t` (`prefill`) that need its `n` tokens into `host`: the tokens (None: the
        device's) and their Engram rows (`_fill_tokens`).
        """

@final
class Ds41Verification:
    """
    An enqueued speculative step and its continuation's feedback (`deepseek_v41.runner._Verification`): `finish`
    waits for its samples, accepts each row's longest draft prefix that the target's samples confirm, records the
    processed tokens it commits (anchor and accepted drafts), and leaves each row's next anchor.
    """
    @property
    def anchors(self, /) -> Incomplete:
        """
        Each row's next anchor (after `finish`).
        """
    @property
    def feedback(self, /) -> Ds41Verification: ...
    def finish(self, /) -> None: ...
    @property
    def positions(self, /) -> Incomplete:
        """
        Each row's next anchor position (after `finish`).
        """
    def result(self, /) -> ForwardOutput: ...

@final
class Engine:
    def __enter__(self, /) -> Engine: ...
    def __exit__(self, /, _exc_type: Any, _exc_value: Any, _traceback: Any) -> bool: ...
    def __new__(cls, /, runner: "Runner", *, max_running_requests: int = 32, context_length: int = 32768, scheduler_config: SchedulerConfig |None = None, prefix_cache: bool = False, enable_overlap: bool = True, enable_metrics: bool = True, cache_pool: CachePool |None = None, cache_floor: int = 0) -> Engine: ...
    def __repr__(self, /) -> str: ...
    def _metrics_prometheus_protobuf(self, /, *, model_name: str, rank: int = 0, attn_dp_rank: int = 0, include_runner: bool = True) -> bytes |None:
        """
        Native serving export; avoids constructing Python inspector dictionaries.
        """
    @property
    def active_request_count(self, /) -> int: ...
    @property
    def cache_free_parent_count(self, /) -> int |None: ...
    @property
    def cache_layout(self, /) -> CacheLayout |None: ...
    @property
    def cache_occupied_block_count(self, /) -> int |None: ...
    @property
    def cache_parent_capacity(self, /) -> int |None: ...
    @property
    def cache_pool_id(self, /) -> int |None: ...
    def cancel(self, /, request: RequestLease) -> bool: ...
    def clear_prefix_cache(self, /) -> None:
        """
        Drop all reusable cache prefixes.
        """
    def close(self, /) -> None: ...
    @property
    def closed(self, /) -> bool: ...
    @property
    def context_length(self, /) -> int: ...
    def has_work(self, /) -> bool: ...
    @property
    def max_running_requests(self, /) -> int: ...
    @property
    def max_speculative_rows(self, /) -> int |None:
        """
        Decode steps with more rows than this lease no write lookahead and
        verify no drafts (the runner config's `max_speculative_rows`; None: any
        step may). Set it between steps for a runner that turns its drafts off
        (0) or on again; it fails while a step is in flight.
        """
    @max_speculative_rows.setter
    def max_speculative_rows(self, /, rows: int |None) -> None: ...
    @property
    def metrics_enabled(self, /) -> bool: ...
    def metrics_snapshot(self, /) -> EngineMetrics |None:
        """
        Collect a read-only native snapshot of engine and runner observations.
        Call periodically: cache inventory requires a scan of allocated blocks.
        """
    @property
    def new_token_ratio(self, /) -> float: ...
    def next_phase(self, /) -> BatchPhase |None:
        """
        Read-only admission-aware proposal; reserves no requests or resources.
        """
    @property
    def overlap(self, /) -> bool: ...
    @property
    def overlap_pending(self, /) -> bool: ...
    @property
    def overlap_ready(self, /) -> bool: ...
    @property
    def overlap_stats(self, /) -> tuple[int, int]: ...
    def poison(self, /, reason: str) -> None:
        """
        Contain a failure reported by a peer without releasing local resources.
        """
    @property
    def poison_error(self, /) -> str |None: ...
    @property
    def poisoned(self, /) -> bool: ...
    @property
    def prefix_cache_stats(self, /) -> tuple[int, int, int, int] |None:
        """
        `(entries, hits, misses, evictions)` when prefix caching is enabled.
        """
    @property
    def ready_request_count(self, /) -> int: ...
    @property
    def scheduler_config(self, /) -> SchedulerConfig: ...
    def step(self, /, *, phase: BatchPhase |None = None, allow_overlap: bool = True) -> list[TokenEvent]: ...
    def submit(self, /, token_ids: "Sequence[int] | npt.NDArray[np.uint32] | None" = None, *, priority: int = 0, batch_key: int |None = None, sampling_params: SamplingParams |None = None, media: "MediaInput | None" = None, prompt_target_logprobs: Sequence[Sequence[int]] |None = None) -> RequestLease: ...

@final
class EngineMetrics:
    """
    An immutable engine snapshot with request, cache, and runner observations.
    Request and cache properties return independent dictionary views.
    """
    @property
    def cache(self, /) -> Any: ...
    @property
    def requests(self, /) -> Any: ...
    @property
    def runner(self, /) -> RunnerMetrics |None: ...

@final
class FnextFill:
    """
    The native decode-step fill of one Qwen3.8-Flash-Next model: constructed once with what the tables depend
    on, then `add_shape` per decode bucket (table offsets) and `fill` per step.
    """
    def __new__(cls, /, *, linear_layers: int, full_layers: int, kv_groups: int, hpg: int, kv_layer_rows: int, state_layer_rows: int, ssm_rows: int, ple_offset: int |None, max_positions: int, ple_heads: int, ngram: tuple[Sequence[int], Sequence[int], Sequence[int], int, int, int, int] |None) -> FnextFill: ...
    def add_shape(self, /, layout: dict) -> None:
        """
        Byte offsets of decode shape `t`'s tables: a dict of the `Tables` fields (per-layer tables as
        (offset, stride)).
        """
    def fill(self, /, t: int, host: Incomplete, pos0: Incomplete, state_src: Incomplete, state_dst: Incomplete, blocks_indptr: Incomplete, blocks: Incomplete, token: Incomplete, history: Incomplete, samp: Incomplete) -> None:
        """
        Write decode step `t`'s tables into `host`: every row's position, STATE block rows, KV block rows
        (`blocks_indptr` delimits them), token and the two tokens before it, and `samp` (rows x 8) f32.
        """

@final
class ForwardOutput:
    """
    Borrowed NumPy SoA output, valid until the synchronous step returns.
    
    All arrays must have their exact declared dtype, one dimension, and contiguous
    storage. Construction retains the arrays without copying. Do not mutate or
    recycle their storage until step() returns. Token rows use token_indptr;
    score rows use logprob_indptr (defaults to token_indptr). An empty score row
    opts out; every other score row must cover the entire raw output token row.
    top_indptr partitions top_token_ids/top_logprobs by scored token. All three
    top_* arrays are optional together.
    """
    def __new__(cls, /, *, token_ids: "npt.NDArray[np.uint32]", token_indptr: "npt.NDArray[np.uint32]", logprobs: "npt.NDArray[np.float32] | None" = None, logprob_indptr: "npt.NDArray[np.uint32] | None" = None, top_indptr: "npt.NDArray[np.uint32] | None" = None, top_token_ids: "npt.NDArray[np.uint32] | None" = None, top_logprobs: "npt.NDArray[np.float32] | None" = None, prompt_logprobs: "npt.NDArray[np.float32] | None" = None, prompt_logprob_indptr: "npt.NDArray[np.uint32] | None" = None, prompt_top_indptr: "npt.NDArray[np.uint32] | None" = None, prompt_top_token_ids: "npt.NDArray[np.uint32] | None" = None, prompt_top_logprobs: "npt.NDArray[np.float32] | None" = None, prompt_target_indptr: "npt.NDArray[np.uint32] | None" = None, prompt_target_token_ids: "npt.NDArray[np.uint32] | None" = None, prompt_target_logprobs: "npt.NDArray[np.float32] | None" = None) -> ForwardOutput: ...
    @property
    def logprob_indptr(self, /) -> Incomplete |None: ...
    @property
    def logprobs(self, /) -> Incomplete |None: ...
    @property
    def prompt_logprob_indptr(self, /) -> Incomplete |None: ...
    @property
    def prompt_logprobs(self, /) -> Incomplete |None: ...
    @property
    def prompt_target_indptr(self, /) -> Incomplete |None: ...
    @property
    def prompt_target_logprobs(self, /) -> Incomplete |None: ...
    @property
    def prompt_target_token_ids(self, /) -> Incomplete |None: ...
    @property
    def prompt_top_indptr(self, /) -> Incomplete |None: ...
    @property
    def prompt_top_logprobs(self, /) -> Incomplete |None: ...
    @property
    def prompt_top_token_ids(self, /) -> Incomplete |None: ...
    @property
    def token_ids(self, /) -> Incomplete: ...
    @property
    def token_indptr(self, /) -> Incomplete: ...
    @property
    def top_indptr(self, /) -> Incomplete |None: ...
    @property
    def top_logprobs(self, /) -> Incomplete |None: ...
    @property
    def top_token_ids(self, /) -> Incomplete |None: ...

@final
class GateCounts:
    """
    Replica steps in the TP model's idle time and in its share, and TP steps that waited for a replica step.
    """
    @property
    def idle_steps(self, /) -> int: ...
    @property
    def shared_steps(self, /) -> int: ...
    @property
    def tp_waits(self, /) -> int: ...

@final
class GaudiGlue:
    """
    The glue functions of the Gaudi runtime library the device loaded (`Device.libraries.runtime`).
    """
    def __new__(cls, /, path: str) -> GaudiGlue: ...

@final
class GaudiProgram:
    """
    A model step's stream calls as one native program (`resin_gaudi::step`). It is built once after the
    recipes compile: `section`, then the ops in stream order, with `take` or `select` (and `arm`) opening a
    nested list that `end` closes, then `finish`. A step then runs a list of sections with its register array
    (`run`, without the GIL); `calls` gives the same walk's calls, recorded, for tests and check mode.
    """
    def __new__(cls, /, registers: int) -> GaudiProgram: ...
    def allgather(self, /, send: int |tuple[int, int, int], recv: int |tuple[int, int, int], count: int |tuple[int, int, int], bf16: bool = False) -> None: ...
    def allreduce(self, /, address: int |tuple[int, int, int], count: int |tuple[int, int, int], bf16: bool = False) -> None: ...
    def arm(self, /) -> int: ...
    def calls(self, /, registers: Sequence[int], sections: Sequence[int |str]) -> list[tuple[str, list[int]]]:
        """
        The calls a step with these registers makes, recorded: (kind, integers) per call.
        """
    def collective_free(self, /, section: int |str) -> bool:
        """
        Whether a section makes no collective on any path.
        """
    def copy(self, /, dst: int |tuple[int, int, int], src: int |tuple[int, int, int], bytes: int |tuple[int, int, int]) -> None: ...
    def copy_multi(self, /, src: Sequence[int], bytes: Sequence[int], dst: Sequence[int]) -> None: ...
    def download(self, /, host: int |tuple[int, int, int], src: int |tuple[int, int, int], bytes: int |tuple[int, int, int]) -> None:
        """
        To pinned host memory, which must stay alive until the stream has passed the copy.
        """
    def end(self, /) -> None: ...
    def finish(self, /) -> None: ...
    def launch(self, /, recipe: int, slots: Sequence[int], addresses: Sequence[int], dynamic: Sequence[tuple[int, int |tuple[int, int, int]]] = ...) -> None:
        """
        A graph launch: the fixed bindings, then `dynamic` (slot, operand) bindings, in order.
        """
    def memset(self, /, dst: int |tuple[int, int, int], value: int, count: int |tuple[int, int, int]) -> None: ...
    def record(self, /, slot: int) -> None: ...
    @property
    def registers(self, /) -> int: ...
    def run(self, /, glue: GaudiGlue, registers: Incomplete, sections: Sequence[int |str]) -> int:
        """
        Make the step's calls on the card through `glue`, without the GIL. Returns the number of calls. On a
        failure the RuntimeError's `issued` is the number of calls the stream took first.
        """
    @staticmethod
    def run_sequence(glue: GaudiGlue, steps: Sequence[tuple[GaudiProgram, int |str]], registers: Incomplete) -> int:
        """
        Make several programs' sections in order as one step, without the GIL (see `run`).
        """
    def section(self, /, name: str) -> int:
        """
        Start a section; returns its number.
        """
    @property
    def sections(self, /) -> list[str]: ...
    def select(self, /, keys: Sequence[tuple[int, int]], table: Sequence[int]) -> None:
        """
        Open a choice of arms: `table[index]`, where `index` combines the (register, radix) keys with the first
        key the most significant; -1 is no arm. `arm` starts each arm.
        """
    @staticmethod
    def sequence_calls(steps: Sequence[tuple[GaudiProgram, int |str]], registers: Sequence[int]) -> list[tuple[str, list[int]]]:
        """
        The calls of several programs' sections in order, recorded (see `calls`).
        """
    def take(self, /, register: int) -> None:
        """
        Open a list of which a step makes the first `registers[register]` ops (no collectives in it).
        """
    def upload(self, /, dst: int |tuple[int, int, int], host: int |tuple[int, int, int], bytes: int |tuple[int, int, int]) -> None:
        """
        From pinned host memory, which must stay alive until the stream has passed the copy.
        """
    def upload_ring(self, /, dst: int |tuple[int, int, int], host: int |tuple[int, int, int], bytes: int |tuple[int, int, int]) -> None:
        """
        Through the runtime's ring of pinned slots: the host bytes are read when the step gets here.
        """

@final
class GaudiRanks:
    """
    One rank's end of the group: the driver's sockets to every follower, or a follower's socket to the driver.
    """
    def __new__(cls, /, path: str |PathLike[str], token: Sequence[int], rank: int, size: int, timeout: float = 600.0) -> GaudiRanks:
        """
        Rank `rank` of `size`: rank 0 listens on `path` until every other rank has joined (with `token`); the
        others connect to it.
        """
    def barrier(self, /) -> None:
        """
        Wait until every rank is here.
        """
    def broadcast(self, /, payload: Sequence[int]) -> bytes:
        """
        Rank 0's bytes on every rank (the others pass anything).
        """
    def ccl_init(self, /, glue: GaudiGlue) -> None:
        """
        Make the group's HCCL communicator on the calling thread's current context: rank 0's unique id on every
        rank, then every rank's `hcclCommInitRank`.
        """
    def min(self, /, value: int) -> int:
        """
        The smallest `value` over the ranks, on every rank.
        """
    @property
    def rank(self, /) -> int: ...
    @property
    def size(self, /) -> int: ...

@final
class Gemma4Media:
    def __new__(cls, /, *, token_ids: Sequence[int], mm_token_type_ids: Sequence[int], pixel_values: Any |None = None, image_position_ids: Any |None = None, pixel_values_videos: Any |None = None, video_position_ids: Any |None = None) -> Gemma4Media: ...
    @property
    def block_ids(self, /) -> tuple: ...
    @property
    def cache_key(self, /) -> tuple: ...
    @property
    def digest(self, /) -> str: ...
    @property
    def image_position_ids(self, /) -> Incomplete |None: ...
    @property
    def image_positions(self, /) -> tuple: ...
    @property
    def mm_token_type_ids(self, /) -> tuple: ...
    @property
    def pixel_values(self, /) -> Incomplete |None: ...
    @property
    def pixel_values_videos(self, /) -> Incomplete |None: ...
    @property
    def token_ids(self, /) -> tuple: ...
    @property
    def token_spans(self, /) -> tuple: ...
    def validate_placeholder_tokens(self, /, *, image_token_id: int, video_token_id: int) -> None: ...
    @property
    def video_position_ids(self, /) -> Incomplete |None: ...
    @property
    def video_positions(self, /) -> tuple: ...
    @property
    def vision_digest(self, /) -> str:
        """
        Image/video tensor identity, independent of conversational text positions.
        """

@final
class Glm53FlashMedia:
    def __new__(cls, /, *, token_ids: Sequence[int], image_positions: Sequence[int], pixel_values: Any, image_grid_thw: Any, spatial_merge_size: int, image_token_id: int) -> Glm53FlashMedia: ...
    @property
    def cache_key(self, /) -> tuple: ...
    @property
    def digest(self, /) -> str: ...
    @property
    def image_grid_thw(self, /) -> Incomplete: ...
    @property
    def image_positions(self, /) -> tuple: ...
    @property
    def image_token_id(self, /) -> int: ...
    @property
    def pixel_values(self, /) -> Incomplete: ...
    @property
    def spatial_merge_size(self, /) -> int: ...
    @property
    def token_ids(self, /) -> tuple: ...
    @property
    def token_spans(self, /) -> tuple: ...

@final
class GlmCompletion:
    """
    One enqueued decode step: `result` waits for its staged samples once.
    """
    @property
    def feedback(self, /) -> GlmFeedback: ...
    def result(self, /) -> ForwardOutput: ...

@final
class GlmDraftPolicy:
    """
    `resin_gaudi::draft_choice::DraftChoice` for tests: the native draft policy, call for call with
    `draft_choice.DraftChoice`.
    """
    def __new__(cls, /, k: int, costs: dict, *, max_rows: int, prior: float, buckets: Sequence[int], shared: bool, mode: str) -> GlmDraftPolicy: ...
    def expected(self, /, sequence: int) -> float: ...
    def forget(self, /, sequences: Sequence[int]) -> None: ...
    def measured(self, /, kind: str, rows: int, ms: float) -> None: ...
    def ms(self, /, kind: str, rows: int) -> float: ...
    def set_shared(self, /, table: dict[tuple[str, int], float]) -> None: ...
    def speculate(self, /, sequences: Sequence[int]) -> bool: ...
    def sync_due(self, /) -> bool: ...
    def table(self, /) -> dict[tuple[str, int], float]: ...
    def verified(self, /, sequences: Sequence[int], accepted: Sequence[int], own: Sequence[bool] |None = None) -> None: ...

@final
class GlmExecutor:
    REGISTERS: Final[int]
    R_ACC: Final[int]
    R_BBYTES: Final[int]
    R_BIMAGE: Final[int]
    R_BLAUNCHES: Final[int]
    R_CHAIN_ROWS: Final[int]
    R_DIMAGE: Final[int]
    R_IMAGE: Final[int]
    R_INDEX_BYTES: Final[int]
    R_MIMAGE: Final[int]
    R_MINDEX_BYTES: Final[int]
    R_MSLOT_BYTES: Final[int]
    R_MSTAGE: Final[int]
    R_SLOT_BYTES: Final[int]
    R_STAGE: Final[int]
    def __new__(cls, /, *, glue: GaudiGlue, fill: GlmFill, layout: CacheLayout, steps: dict[tuple[int, bool, bool, int], GaudiProgram], io: dict[int, GaudiProgram], image: tuple[int, Incomplete, int, int], staging: tuple[Incomplete, int, int, tuple[int, int, int, int, int]], sample: tuple[Incomplete, int], geometry: tuple[int, int, int, int], model: dict, scoring: dict |None = None, mtp: dict |None = None, dflash: dict |None = None, commit: dict |None = None, share: Any |None = None) -> GlmExecutor:
        """
        `steps`: per (bucket, indexer, sparse, verify) its step program; `io`: per bucket its I/O program.
        `image`: (device table, pinned host copies, stride, first event slot); `staging`: (pinned sets, stride,
        first event slot, (tok, lp, top_idx, top_lp, bytes) offsets); `sample`: (pinned sampler table, offset of
        its `last`); `geometry`: (slab rows, KV block rows, STATE block rows, STATE tokens); `model`: a dict of the
        model's numbers; `scoring`: the prompt logprobs' pinned buffers and event; `mtp`: the MTP layer's side
        (`native._mtp_side`); `share`: the ranks' step times from the first rank (None: one card or a native rank
        group).
        """
    def attach_gate(self, /, gate: CardGate) -> None:
        """
        A layout's card gate: every runner call is a TP step on this card, which a replica on it waits for.
        """
    def attach_ranks(self, /, ranks: GaudiRanks) -> None:
        """
        Join a tensor-parallel model's rank group: rank 0 drives (it sends each runner call to the others, with its
        draft decision), and every other rank replays the calls (`serve`).
        """
    def carried(self, /, sequence: int) -> tuple[int, int, list[int]] |None:
        """
        A sequence's drafts carried to its next step: (the anchor's position, the anchor, the drafts).
        """
    def carried_all(self, /) -> dict[int, tuple[int, int, list[int]]]:
        """
        Every sequence's drafts carried to its next step.
        """
    def close(self, /) -> None:
        """
        The driver: tell the followers to stop serving.
        """
    @property
    def continuations(self, /) -> int:
        """
        Enqueued steps that continued the previous one.
        """
    def enqueue(self, /, plan: BatchPlan, *, feedback: Any |None = None) -> Any:
        """
        Submit a decode step and return without waiting for it: a continuation (`feedback`) reads its tokens from
        the previous step's samples on the device, at the previous positions + 1, or continues its verification.
        Other plans run synchronously.
        """
    @property
    def fixed_acceptance(self, /) -> int |None:
        """
        The benchmark's fixed acceptance (`SpeculativeStats.benchmark_accepted_drafts`).
        """
    @fixed_acceptance.setter
    def fixed_acceptance(self, /, fixed: int |None) -> None: ...
    def forward(self, /, plan: BatchPlan) -> Any:
        """
        The Engine's synchronous step: prefill chunks, a decode step or a speculative step.
        """
    def metrics_snapshot(self, /) -> RunnerMetrics:
        """
        The runner's `metrics_snapshot` (the speculation counters), with no Python: the runner binds it.
        """
    @property
    def native_steps(self, /) -> int:
        """
        Runs of the native step programs so far.
        """
    @property
    def policy(self, /) -> str:
        """
        The draft policy's mode (auto, always or never).
        """
    @policy.setter
    def policy(self, /, mode: str) -> None:
        """
        Change the draft policy's mode (tests and benchmarks; with ranks, on the driver, which decides).
        """
    def release(self, /, sequence_ids: Sequence[int]) -> None:
        """
        Forget released sequences: their acceptance and carried drafts (the Engine took the open steps' results
        first). With ranks, the driver sends the call to the followers first.
        """
    def serve(self, /) -> int:
        """
        A follower: replay the driver's runner calls until it closes the group. Returns the calls replayed.
        """
    @property
    def speculation(self, /) -> tuple[int, int, int, int]:
        """
        The verify steps' acceptance counters: (blocks, drafts, accepted, emitted).
        """
    @property
    def timings(self, /) -> dict[str, int]:
        """
        Host nanoseconds so far: fill, submit, image_wait, output_wait.
        """
    @staticmethod
    def trace_rank(rank: int) -> None:
        """
        `RESIN_STEP_TRACE`: mark this process's rank in its trace.
        """

@final
class GlmFeedback:
    """
    A plain decode step's continuation: each row's next position; its tokens stay on the device.
    """
    @property
    def positions(self, /) -> Incomplete: ...

@final
class GlmFill:
    """
    The native step fill of one GLM-5.3-Flash model (`resin_gaudi::glm`): constructed once with what the tables
    depend on, then `add_shape` per table shape (the step's and the MTP pass's) and `fill` per step.
    """
    def __new__(cls, /, *, mla_heads: int, index_heads: int, top_pools: int, tail_pools: int, pages: int, state_layer_rows: int, ssm_rows: int, max_t: int, geometry: tuple[int, int, int, int, int, int]) -> GlmFill: ...
    def add_shape(self, /, layout: dict, mtp: bool) -> None:
        """
        Byte offsets of a shape's tables (a dict of the `_Tables` fields, `offsets` its DSA layers' page offsets);
        `mtp`: the MTP pass's tables.
        """
    def fill(self, /, t: int, mtp: bool, host: Incomplete, column: Incomplete, pos0: Incomplete, real: Incomplete, state_src: Incomplete, state_dst: Incomplete, blocks_indptr: Incomplete, blocks: Incomplete, columns: int, first_position: int, tokens: Incomplete |None = None, head: bool = False, verify: int = 0, keep_tails: bool = False, meta_columns: int |None = None) -> tuple[bool, bool, int, int, list[tuple[int, int]]]:
        """
        Write a step's tables (shape `t`, `mtp`: the MTP pass's) into `host`: the rows (column, pos0, real,
        state_src, state_dst, KV blocks by `blocks_indptr`), the step's `columns` real columns with `tokens` (None:
        the device's), its first position, and whether it writes the head's columns (a prefill chunk), is a verify
        step of `verify` columns a row, keeps every pool tail, and the meta table's real columns. Returns (index on, sparse, slot launches,
        index launches, uploads).
        """

@final
class GlmVerification:
    """
    An enqueued speculative step and its continuation's feedback (`runner._Verification`): `settle` waits for its
    samples, accepts each row's longest draft prefix that the target's samples confirm, commits the accepted
    tokens' KDA state, reads each row's next first draft, and leaves each row's next anchor.
    """
    @property
    def anchors(self, /) -> Incomplete:
        """
        Each row's next anchor (after `finish`).
        """
    @property
    def feedback(self, /) -> GlmVerification: ...
    def finish(self, /) -> None: ...
    @property
    def positions(self, /) -> Incomplete:
        """
        Each row's next anchor position (after `finish`).
        """
    def result(self, /) -> ForwardOutput: ...

@final
class Image:
    def __new__(cls, /, *, source: Any) -> Image: ...
    @staticmethod
    def from_bytes(data: bytes, *, media_type: str) -> Image: ...
    @staticmethod
    def from_rgb(array: "npt.NDArray[np.float32]") -> Image: ...
    @property
    def source(self, /) -> Any: ...

@final
class InklingMedia:
    def __new__(cls, /, *, token_ids: Sequence[int], image_token_id: int, pixel_values: Any |None = None) -> InklingMedia: ...
    @property
    def cache_key(self, /) -> tuple: ...
    @property
    def digest(self, /) -> str: ...
    @property
    def image_positions(self, /) -> tuple: ...
    @property
    def image_token_id(self, /) -> int: ...
    @property
    def pixel_values(self, /) -> Incomplete |None: ...
    @property
    def token_ids(self, /) -> tuple: ...
    @property
    def token_spans(self, /) -> tuple: ...

@final
class InklingResponseParser(_NativeResponseParser):
    """
    Parse Inkling response blocks into the events accepted by HttpServer.
    """
    def __new__(cls, /, tokenizer: RecipeTokenizer, token_ids: "Mapping[str, int]", *, prompt_ids: "Sequence[int] | npt.NDArray[np.uint32]" = ...) -> InklingResponseParser: ...

@final
class Message:
    def __new__(cls, /, role: Role, content: "Sequence[ContentBlock]", *, tools: Sequence[Tool] = ..., task: str |None = None, continuation: bool = False, response_format: Any |None = None) -> Message: ...
    @property
    def content(self, /) -> tuple: ...
    @property
    def continuation(self, /) -> bool: ...
    @property
    def response_format(self, /) -> Any: ...
    @property
    def role(self, /) -> Role: ...
    @property
    def task(self, /) -> str |None: ...
    @property
    def tools(self, /) -> list[Tool]: ...

@final
class NativeRecipe:
    """
    A native model recipe constructed by get_native_recipe.
    """
    def _media_preparation_cache_info(self, /) -> dict[str, int]: ...
    def create_parser(self, /, prompt: RenderedPrompt) -> _NativeResponseParser: ...
    def debug_render_text(self, /, messages: "Sequence[Message]", *, tools: "Sequence[Tool]" = ..., options: RenderOptions |None = None) -> str:
        """
        Inspect the template before tokenization and media expansion.
        """
    def parse_response(self, /, token_ids: "Sequence[int] | npt.NDArray[np.uint32]", *, prompt: RenderedPrompt) -> Any: ...
    @property
    def reasoning_efforts(self, /) -> list[str]:
        """
        The `reasoning_effort` names that select a different prompt, from the least to the
        most effort. Empty when the recipe does not use `reasoning_effort`.
        """
    def render(self, /, messages: "Sequence[Message]", *, tools: "Sequence[Tool]" = ..., options: RenderOptions |None = None) -> RenderedPrompt: ...
    @property
    def stop_token_ids(self, /) -> list[int]:
        """
        Static model stop tokens, returned as an owned list.
        """
    @property
    def tokenizer(self, /) -> RecipeTokenizer: ...

@final
class Parallelism:
    """
    Attention and expert meshes over the same GPU ranks.
    
    Attention uses [ATTN_DP, CP, REPLICATION, ATTN_TP]; unassigned attention
    ranks replicate the same request. Experts use [EP, EXPERT_TP], where
    EXPERT_TP is world_size // ep. CP shards prefill and replicates decode.
    """
    def __eq__(self, value: object, /) -> bool: ...
    def __getnewargs_ex__(self, /) -> "tuple[tuple[()], dict[str, int]]": ...
    def __hash__(self, /) -> int: ...
    def __ne__(self, value: object, /) -> bool: ...
    def __new__(cls, /, *, world_size: "int" = 1, attn_dp: "int" = 1, attn_tp: "int" = 1, cp: "int" = 1, ep: "int" = 1) -> Parallelism: ...
    def __repr__(self, /) -> str: ...
    @property
    def attn_dp(self, /) -> int: ...
    @property
    def attn_replication(self, /) -> int: ...
    @property
    def attn_tp(self, /) -> int: ...
    def coordinates(self, /, rank: "int") -> tuple[int, int, int, int]: ...
    @property
    def cp(self, /) -> int: ...
    @property
    def ep(self, /) -> int: ...
    @property
    def expert_tp(self, /) -> int: ...
    def partitions(self, /) -> "dict[str, tuple[tuple[int, ...], ...]]": ...
    def rank(self, /, attn_dp: "int", cp: "int", replication: "int", attn_tp: "int") -> int: ...
    @property
    def world_size(self, /) -> int: ...

class ParserEvent:
    """
    A semantic parser event with token-to-text byte provenance.
    
    Empty token_spans identify synthetic text or unknown provenance.
    """
    def __eq__(self, value: object, /) -> bool: ...
    def __ne__(self, value: object, /) -> bool: ...
    def __repr__(self, /) -> str: ...
    @property
    def token_spans(self, /) -> tuple: ...
    @property
    def type(self, /) -> str: ...

@final
class Qwen35Media:
    def __new__(cls, /, *, token_ids: Sequence[int], image_positions: Sequence[int], pixel_values: Any, image_grid_thw: Any, spatial_merge_size: int, image_token_id: int) -> Qwen35Media: ...
    @property
    def cache_key(self, /) -> tuple: ...
    @property
    def digest(self, /) -> str: ...
    @property
    def image_grid_thw(self, /) -> Incomplete: ...
    @property
    def image_positions(self, /) -> tuple: ...
    @property
    def image_token_id(self, /) -> int: ...
    @property
    def pixel_values(self, /) -> Incomplete: ...
    @property
    def position_ids(self, /) -> Incomplete: ...
    @property
    def rope_delta(self, /) -> int: ...
    @property
    def spatial_merge_size(self, /) -> int: ...
    @property
    def token_ids(self, /) -> tuple: ...
    @property
    def token_spans(self, /) -> tuple: ...

@final
class QwenCompletion:
    """
    An enqueued plain step: `result()` waits for it and gives its ForwardOutput; `feedback` continues it.
    """
    @property
    def feedback(self, /) -> QwenFeedback: ...
    def result(self, /) -> ForwardOutput: ...

@final
class QwenDraftPolicy:
    """
    The native executor's draft policy (`Policy`) on its own, for the CPU checks of its decisions: `k` drafts a
    step, a drafter's (`drafter`) or the MTP layer's step-time fits, and its `mode`. A step is keyed ("plain",
    bucket) or ("verify", bucket).
    """
    BETA: Final[float]
    MARGIN: Final[float]
    PROBE_FIRST: Final[int]
    PROBE_MAX: Final[int]
    PROBE_STEPS: Final[int]
    SPIKE: Final[float]
    def __new__(cls, /, k: int, drafter: bool, mode: str = "auto") -> QwenDraftPolicy: ...
    def fit(self, /, bucket: int, width: int) -> float:
        """
        A verify step's tokens per ms by the fits, at about 60% accepted drafts.
        """
    def rate(self, /, key: tuple[str, int], widths: Sequence[int] = ...) -> float:
        """
        The tokens per ms of `key`: measured, else the fit (a verify step's at the best of `widths`).
        """
    def record(self, /, key: tuple[str, int], tokens: float, ms: float) -> None:
        """
        A measured step of `key`: the tokens it committed and its time.
        """
    def speculate(self, /, rows: int, widths: Sequence[int]) -> bool:
        """
        Whether a decode step of `rows` rows verifies drafts (a drafter's at the best of `widths`).
        """

@final
class QwenDraftTables:
    """
    The native drafter tables of a Qwen3.5/3.8 verify step (`resin_gaudi::qwen_draft`), decoded into arrays for
    the CPU checks against `common.dflash` and `qwen3_5.model` (the executor fills the images directly).
    """
    def __new__(cls, /, *, layers: int, pairs: int, layer_rows: int, block: int, window: int, causal: bool, mask_token: int, scratch_rows: int, max_rows: int) -> QwenDraftTables: ...
    @staticmethod
    def accept(drafts: Sequence[int], samples: Sequence[int], rows: int, width: int) -> tuple[list[int], list[list[int]]]:
        """
        `accept_speculative_tokens` of rows x (width - 1) drafts and rows x width samples: (accepted, emitted).
        """
    def block(self, /, bucket: int, rows: Sequence[tuple[int, Sequence[int], int]], anchors: Sequence[int] |None, low: Sequence[int] |None, width: int, max_positions: int) -> dict:
        """
        A block step of `bucket` rows for a verify of `width` columns, decoded: a dict of `tok`, `pos`, `anchors`,
        `block_rows` (layers * T * 8), `slots` / `k_pages` / `v_pages` per (launch, layer), `slots_per_launch`,
        `launches`, `records` and `patch` (offsets in i32 units from the slot tables, launch-major).
        """
    @staticmethod
    def choose_width(widths: Sequence[int], accepted: Sequence[int], width: int, bucket: int) -> int:
        """
        `qwen3_5.model.choose_width`
        """
    @staticmethod
    def fix_sizes(bucket: int, layers: int, patches: int) -> list[int]:
        """
        The bytes of a verify step's spec_fix tables (`FixLayout` order) for `bucket` rows.
        """
    def inject(self, /, tokens: int, runs: Sequence[tuple[int, int, int, tuple[int, Sequence[int], int]]]) -> tuple[list[int], list[int]]:
        """
        An injection of `tokens` columns: (positions (tokens,), kv_rows (layers * tokens * 8,)). `runs`: (column,
        position, count, (position, blocks, first)) each.
        """
    def inject_sizes(self, /, tokens: int) -> list[int]:
        """
        The bytes of an injection's tables of `tokens` columns: (positions, every layer's kv_rows).
        """
    def layout(self, /, bucket: int) -> dict:
        """
        A block step's image layout for `bucket` rows: a dict of byte offsets (`tok`, `pos`, `anchors`,
        `block_rows`, `block_stride`, `tables`, `stride`, `k_pages`, `v_pages`), `slots` a launch, `max_launches`
        and `nbytes`; launch i of layer l's tables at `tables + (i * layers + l) * stride`.
        """

@final
class QwenExecutor:
    BLOCK_SECTIONS: Final[list[str]]
    MAX_LINKS: Final[int]
    REGISTERS: Final[int]
    R_ACC: Final[int]
    R_BBYTES: Final[int]
    R_BIMAGE: Final[int]
    R_BLAUNCHES: Final[int]
    R_BYTES: Final[int]
    R_CBYTES: Final[int]
    R_CIMAGE: Final[int]
    R_FIXTIN: Final[int]
    R_IMAGE: Final[int]
    R_LAUNCHES: Final[int]
    R_LINK: Final[int]
    R_MLAUNCHES: Final[int]
    R_MPREV: Final[int]
    R_NEXT: Final[int]
    R_ROWS: Final[int]
    R_ROWS4: Final[int]
    R_STAGE: Final[int]
    R_TOK: Final[int]
    R_TOP: Final[int]
    R_WLAUNCHES: Final[int]
    SECTIONS: Final[list[str]]
    def __new__(cls, /, *, glue: GaudiGlue, fill: QwenFill, programs: dict[str, GaudiProgram], extras: dict[str, dict], prefill_buckets: Sequence[int], decode_buckets: Sequence[int], image: int, host: Incomplete, host_stride: int, image_events: int, staging: dict, tok_out: int, spin_ns: int, vocab: int, cache: tuple[int, int, int, int, int, int], max_batch_size: int, max_seq_len: int, lookahead: int, draft: dict |None, mtp: dict |None = None, scoring: dict |None = None) -> QwenExecutor:
        """
        `fill`'s modes, each with its program (`programs`, with sections of SECTIONS) and with a drafter its
        injection and verify tables (`extras`: per mode a dict of `inject` (pos, kv_rows, kv_stride) and
        `fix` (patches and the FixLayout tables' offsets)). `image` / `host` / `host_stride` / `image_events`: the
        step image, its two pinned host copies and their events. `staging`: a dict of the pinned staging sets
        (tokens, lp, top_idx, top_lp, and with a drafter accepted and next) with `columns`, `rows`, `block` and
        the first of its four event slots (`events`). `draft`: the drafter's config dict, or None.
        """
    def attach_ranks(self, /, ranks: GaudiRanks) -> None:
        """
        Join a tensor-parallel model's rank group: rank 0 drives (it sends each runner call to the others), and
        every other rank replays the calls (`serve`).
        """
    def carried(self, /, sequence: int) -> tuple[int, int, list[int]] |None:
        """
        A test aid: the drafts carried for `sequence`'s next verify step: (the next anchor's position, the
        anchor, its drafts), or None.
        """
    def close(self, /) -> None:
        """
        The driver: tell the followers to stop serving.
        """
    def enqueue(self, /, plan: BatchPlan, *, feedback: Any |None = None) -> Any:
        """
        Submit a decode step and return without waiting for it (`Qwen3_5GaudiRunner.enqueue`'s contract): the
        device continues from `feedback` (the previous step's exact positions; its tokens stay on the device).
        Other plans run synchronously.
        """
    @property
    def force_width(self, /) -> int |None:
        """
        A test aid: every verify step of the drafter at `width` columns (one of its verify widths), or None (the
        width that the last step's acceptance gives).
        """
    @force_width.setter
    def force_width(self, /, width: int |None) -> None: ...
    def forward(self, /, plan: BatchPlan) -> Any:
        """
        The Engine's synchronous step: a prefill (with its prompt logprobs) or a decode step.
        """
    def metrics_snapshot(self, /) -> RunnerMetrics:
        """
        The runner's `metrics_snapshot` (the speculation counters), with no Python: the runner binds it.
        """
    @property
    def native_steps(self, /) -> int:
        """
        Program runs this executor made.
        """
    @property
    def policy_mode(self, /) -> str |None:
        """
        The draft policy's mode: auto, always or never (None without a drafter).
        """
    @policy_mode.setter
    def policy_mode(self, /, mode: str) -> None: ...
    def release(self, /, sequence_ids: Sequence[int]) -> None:
        """
        Drop the sequences' carried drafts.
        """
    def serve(self, /) -> int:
        """
        A follower: replay the driver's runner calls until it closes the group. Returns the calls replayed.
        """
    @property
    def speculation(self, /) -> tuple[int, int, int, int]:
        """
        The verify steps' acceptance counters: (blocks, drafts, accepted, emitted).
        """
    @property
    def timings(self, /) -> dict[str, int]:
        """
        Cumulative host nanoseconds of the native steps: fill, submit, image_wait, output_wait, and with ranks
        gate (waiting until every rank is ready) and announce (the driver sending the calls).
        """

@final
class QwenFeedback:
    """
    A plain step's continuation: the rows' next positions; `tokens()` waits for the step's samples.
    """
    @property
    def positions(self, /) -> Incomplete: ...
    def tokens(self, /) -> Incomplete: ...

@final
class QwenFill:
    """
    The native step tables of one Qwen3.5/3.8 model (one rank's dimensions): constructed once, then `add_mode`
    per step mode (it returns the mode's table offsets, which the device bindings take) and `fill` per step.
    """
    def __new__(cls, /, *, gdn_layers: int, kv_groups: int, hpg: int, state_layer_rows: int, ssm_rows: int, max_positions: int, kv8: int = 0) -> QwenFill:
        """
        `kv8`: bit 0 the K pages FP8, bit 1 the V pages (`qwen3_5/cache.py`).
        """
    def add_mode(self, /, name: str, *, columns: int, per_row: int, query_block: int, slots: int, prefill: bool, max_launches: int, extras: Sequence[int] = ..., wide_slots: int = 0, max_wide: int = 0, wide_full: bool = False) -> dict:
        """
        Add step mode `name` and return its image layout: a dict of byte offsets (`tok`, `pos`, `meta`, `last`,
        `samp_params`, `kv_rows`, `state`, `state_stride`, `conv_dst`, `slots`, `launch_stride`, `k_pages`,
        `v_pages`, and of the wide launches `wide`, `wide_stride`, `wide_k_pages`, `wide_v_pages`) and the image's
        size with every launch (`nbytes`). With `wide_slots` (a multiple of `slots`), a step's first slots go in
        up to `max_wide` launches of `wide_slots`, and `max_launches` (the narrow ones) is under
        `wide_slots / slots`. With `wide_full`, the wide launches take the slots whose every key each query of
        the block sees instead, and the narrow ones the others (`wide_slots` may then equal `slots`).
        """
    def fill(self, /, mode: str, host: Incomplete, pos0: Incomplete, state_src: Incomplete, state_dst: Incomplete, blocks_indptr: Incomplete, blocks: Incomplete, real: int, tokens: Incomplete |None, params: Incomplete) -> tuple[int, int]:
        """
        Write a target pass of mode `mode` into `host` (the mode's image): every row's first position, STATE block
        rows and KV block rows (`blocks_indptr` delimits them); `real` (a prefill's real tokens, else the real
        rows); the columns' `tokens` (None: another buffer holds them); `params` (PARAMS f32 a sampled column).
        Returns (slot launches per layer, bytes to upload).
        """
    def fill_chain(self, /, decode: str, width: int, host: Incomplete, pos0: Sequence[int], state_src: Sequence[int], states: Sequence[Sequence[int]], state_first: Sequence[int], accepted: Sequence[int], anchors: Sequence[int], first: Sequence[int], blocks_indptr: Sequence[int], blocks: Sequence[int], params: Sequence[Sequence[float]], kv_groups: int, hpg: int, state: tuple[int, int, int, int]) -> tuple[list[int], int]:
        """
        Write a verify step's commit and draft chain (`qwen_mtp::fill_chain`) into `host` (the chain image of
        `mtp_chain(rows, ...)`, `decode` naming the chain's decode mode): each row's first position, STATE source
        and STATE block rows (`state_first`: the logical block of the first), accepted drafts, next anchor, first
        next draft, MTP KV block rows (`blocks_indptr` delimits them) and each link's sampler inputs. Returns (each
        link's slot launches, the bytes to upload).
        """
    def fill_mtp(self, /, mode: str, host: Incomplete, offsets: Sequence[int], starts: Sequence[int], real: int, blocks_indptr: Sequence[int], blocks: Sequence[int], cols: Sequence[int], tokens: Sequence[int] |None, params: Sequence[float] |None, kv_groups: int, hpg: int) -> int:
        """
        Write an MTP pass of mode `mode` (`qwen_mtp::fill_pass`) into `host` at the tables `offsets`
        (`mtp_pass_sizes` order): row i's columns at positions from `starts[i]` (a prefill's `real` of them),
        its MTP KV block rows (`blocks_indptr` delimits them), each column's hidden column `cols` and token
        `tokens` (None: a device buffer), and for a full pass the sampler inputs `params`. The MTP layer's
        attention: `kv_groups` KV groups of `hpg` heads. Returns the slot launches.
        """
    @staticmethod
    def launch_tables(slots: int) -> tuple[int, int, int]:
        """
        (k_pages, v_pages, stride) of a slot launch's tables with `slots` slots.
        """
    @staticmethod
    def mtp_chain(rows: int, slots: int, max_launches: int, width: int, layers: int) -> dict:
        """
        The chain image after an MTP verify step of `rows` rows at `width` columns a row (`qwen_mtp::ChainLayout`;
        its links run the decode mode of `rows` rows, `slots` slots a launch and room for `max_launches` launches)
        over `layers` GDN layers: a dict of the byte offsets (`commit_bases`, `commit_counts`, `hidden_table`,
        `following`, `meta`), `links` (each a dict of `cols`, `tok`, `pos`, `kv_rows`, `params`, `slots`,
        `stride`, `k_pages`, `v_pages`) and `nbytes`.
        """
    @staticmethod
    def mtp_pass_sizes(columns: int, slots: int, max_launches: int, full: bool) -> list[int]:
        """
        The bytes of an MTP pass's tables (`cols`, `tok`, `pos`, `kv_rows`, `params`, `slots`; `resin_gaudi::
        qwen_mtp::PassLayout`) over a mode of `columns` columns, `slots` slots a launch and room for `max_launches`
        launches; a keys pass (`full` False) takes no sampler inputs and no slot tables.
        """

@final
class ReasoningDelta(ParserEvent):
    def __new__(cls, /, delta: str, *, token_spans: Sequence[tuple[int, int, int, int, int]] = ...) -> ReasoningDelta: ...
    @property
    def delta(self, /) -> str: ...

@final
class RecipeTokenizer:
    """
    A tokenizer shared by rendering, response parsing, and token metadata.
    
    Fast Hugging Face tokenizers are snapshotted into Rust. Custom Python
    tokenizers remain supported through their decode method.
    """
    def __new__(cls, /, tokenizer: Any) -> RecipeTokenizer: ...
    def decode(self, /, token_ids: "Sequence[int] | npt.NDArray[np.uint32]", *, skip_special_tokens: bool = False) -> str: ...
    def encode(self, /, text: str, *, add_special_tokens: bool = False) -> list[int]: ...
    def id_to_token(self, /, token_id: int) -> str |None: ...
    @property
    def is_native(self, /) -> bool: ...
    @property
    def name_or_path(self, /) -> str |None: ...
    def to_str(self, /) -> str:
        """
        Serialize the prepared native snapshot, for external tokenizer tools.
        """
    def token_to_id(self, /, token: str) -> int |None: ...

@final
class RenderOptions:
    """
    Explicit rendering controls; absent fields inherit recipe defaults.
    """
    def __new__(cls, /, *, reasoning_effort: "str | int | float | None" = None, enable_thinking: bool |None = None, preserve_thinking: bool |None = None, clear_thinking: bool |None = None, add_generation_prompt: bool |None = None, add_vision_id: bool |None = None, use_system_prompt: bool |None = None, model_identity: str |None = None, knowledge_cutoff: str |None = None, conversation_start_date: str |None = None, auto_drop_analysis: bool |None = None, thinking_mode: str |None = None, response_format: Any |None = None) -> RenderOptions: ...
    def __repr__(self, /) -> str: ...
    @property
    def add_generation_prompt(self, /) -> bool |None: ...
    @property
    def add_vision_id(self, /) -> bool |None: ...
    @property
    def auto_drop_analysis(self, /) -> bool |None: ...
    @property
    def clear_thinking(self, /) -> bool |None: ...
    @property
    def conversation_start_date(self, /) -> str |None: ...
    @property
    def enable_thinking(self, /) -> bool |None: ...
    @property
    def knowledge_cutoff(self, /) -> str |None: ...
    @property
    def model_identity(self, /) -> str |None: ...
    @property
    def preserve_thinking(self, /) -> bool |None: ...
    @property
    def reasoning_effort(self, /) -> Any: ...
    @property
    def response_format(self, /) -> Any: ...
    @property
    def thinking_mode(self, /) -> str |None: ...
    @property
    def use_system_prompt(self, /) -> bool |None: ...

@final
class RenderedPrompt:
    """
    Rendered IDs and optional prepared media. Native parser state contains no pixels.
    """
    def __new__(cls, /, token_ids: "Sequence[int] | npt.NDArray[np.uint32]", *, media: "MediaInput | None" = None) -> RenderedPrompt: ...
    @property
    def media(self, /) -> Any |None: ...
    @property
    def token_ids(self, /) -> Incomplete: ...

@final
class RequestLease:
    """
    A caller-owned lifetime handle for one submitted request.
    
    The Engine keeps the shared state only while the request is active. Once
    the request retires, its generated tokens are moved into this lease. If
    the caller drops the lease without taking the result, the token buffer is
    released when the last internal reference disappears.
    """
    def __eq__(self, other: object, /) -> bool: ...
    def __index__(self, /) -> int: ...
    def __int__(self, /) -> int: ...
    def __repr__(self, /) -> str: ...
    @property
    def cached_tokens(self, /) -> int:
        """
        Number of prompt tokens restored from the prefix cache at admission.
        """
    @property
    def cancelled(self, /) -> bool: ...
    @property
    def done(self, /) -> bool: ...
    @property
    def id(self, /) -> int: ...
    @property
    def prompt_logprobs(self, /) -> list[TokenLogprobs |None] |None:
        """
        Raw input-token logprobs; the first token has no predecessor and is None.
        """
    @property
    def request_id(self, /) -> int: ...
    def take_result(self, /) -> list[int]:
        """
        Consume the completed token buffer exactly once.
        """

@final
class ResourceBinding:
    """
    Concrete Engine-owned resource identities passed to `bind_resources`.
    """
    def __repr__(self, /) -> str: ...
    @property
    def cache_layout(self, /) -> CacheLayout |None: ...
    @property
    def cache_pool_id(self, /) -> int |None: ...

@final
class Role:
    """
    Conversation role. This enum is extensible: new roles may be added.
    Code matching roles should include a fallback for unsupported roles.
    Some roles are model-specific; recipes validate supported roles.
    """
    ASSISTANT: Final[Role]
    DEVELOPER: Final[Role]
    LATEST_REMINDER: Final[Role]
    SYSTEM: Final[Role]
    TOOL: Final[Role]
    USER: Final[Role]
    def __eq__(self, value: object, /) -> bool: ...
    def __int__(self, /) -> int: ...
    def __ne__(self, value: object, /) -> bool: ...
    def __repr__(self, /) -> str: ...

@final
class RunnerConfig:
    """
    Frozen scheduling limits and persistent-resource declarations for a runner.
    
    Runners without special requirements may omit `runner_config` entirely.
    """
    def __new__(cls, /, *, max_batch_tokens: int |None = None, prefill_chunk_size: int |None = None, continuation_prefill: bool = False, cache_write_lookahead_tokens: int = 0, cache_layout: CacheLayout |None = None, max_speculative_rows: int |None = None, continuation_extension: int |None = None, draft_tokens: int = 0) -> RunnerConfig: ...
    def __repr__(self, /) -> str: ...
    @property
    def cache_layout(self, /) -> CacheLayout |None: ...
    @property
    def cache_write_lookahead_tokens(self, /) -> int: ...
    @property
    def continuation_extension(self, /) -> int |None:
        """
        Decode continuations may cross cache blocks (pages leased in place); a
        chain still ends where a row reaches a multiple of this many tokens, so
        that prefixes can be published. None: continuations stay within a block.
        """
    @property
    def continuation_prefill(self, /) -> bool: ...
    @property
    def draft_tokens(self, /) -> int:
        """
        Drafts a speculative decode step verifies for each row (0: no speculation).
        The HTTP log's verify-launch lines count acceptance against it.
        """
    @property
    def max_batch_tokens(self, /) -> int |None: ...
    @property
    def max_speculative_rows(self, /) -> int |None:
        """
        Decode steps with more rows than this do not speculate (one token a row).
        """
    @property
    def prefill_chunk_size(self, /) -> int |None: ...

@final
class RunnerMetrics:
    """
    A nonblocking snapshot of cumulative observations for one physical rank.
    
    Forward categories are exclusive: prefill, decode, target_verify, draft,
    and idle. DP wait uses a bounded exponential histogram. The core snapshot is stored directly.
    """
    def __eq__(self, value: object, /) -> bool: ...
    def __ne__(self, value: object, /) -> bool: ...
    def __new__(cls, /, *, forward_execution_seconds: dict[str, float] |None = None, spec_draft_tokens: int = 0, spec_accepted_tokens: int = 0, spec_verify_rows: int = 0, dp_attention_sync_wait_seconds: "dict[str, Any] | None" = None) -> RunnerMetrics: ...
    def as_dict(self, /) -> "dict[str, Any]":
        """
        Plain fields for inspection and explicit distributed transport.
        """
    @property
    def dp_attention_sync_wait_seconds(self, /) -> Any: ...
    @property
    def forward_execution_seconds(self, /) -> dict[str, float]: ...
    @property
    def spec_accepted_tokens(self, /) -> int: ...
    @property
    def spec_draft_tokens(self, /) -> int: ...
    @property
    def spec_verify_rows(self, /) -> int: ...

@final
class SamplingParams:
    """
    Per-request generation policy, including sampling, length, and stop tokens.
    
    A zero temperature selects the maximum-logit token. Positive temperatures
    enable stochastic sampling after optional top-k and top-p filtering.
    An omitted `max_output_tokens` uses the remaining context budget. Stop tokens
    are explicit: pass a recipe's `stop_token_ids` for recipe stopping.
    """
    def __new__(cls, /, *, max_output_tokens: int |None = None, stop_token_ids: Sequence[int] = ..., temperature: float = 1.0, top_p: float = 1.0, top_k: int |None = None, seed: int |None = None, logprobs: bool = False, top_logprobs: int = 0, prompt_logprobs: bool = False, prompt_top_logprobs: int = 0, extra: Any |None = None) -> SamplingParams: ...
    def __repr__(self, /) -> str: ...
    @property
    def extra(self, /) -> Any |None: ...
    @property
    def logprobs(self, /) -> bool: ...
    @property
    def max_output_tokens(self, /) -> int |None: ...
    @property
    def prompt_logprobs(self, /) -> bool: ...
    @property
    def prompt_top_logprobs(self, /) -> int: ...
    @property
    def seed(self, /) -> int |None: ...
    @property
    def stop_token_ids(self, /) -> tuple: ...
    @property
    def temperature(self, /) -> float: ...
    @property
    def top_k(self, /) -> int |None: ...
    @property
    def top_logprobs(self, /) -> int: ...
    @property
    def top_p(self, /) -> float: ...

@final
class SchedulerConfig:
    """
    Native admission and token-budget policy for an Engine.
    """
    def __new__(cls, /, *, max_batch_tokens: int |None = None, prefill_chunk_size: int = 8192, schedule_conservativeness: float = 1.0, decode_share: tuple[float, float] |None = None) -> SchedulerConfig: ...
    def __repr__(self, /) -> str: ...
    @property
    def decode_share(self, /) -> tuple[float, float] |None:
        """
        The `(min, max)` share of contended execution time for decode, or
        `None` to run every admitted prefill chunk before decode.
        """
    @property
    def max_batch_tokens(self, /) -> int |None: ...
    @property
    def prefill_chunk_size(self, /) -> int: ...
    @property
    def schedule_conservativeness(self, /) -> float: ...

@final
class SlabSide:
    """
    One side of a `SlabTable`, for `CachePool(..., boundary=side)`.
    """
    @property
    def side(self, /) -> int: ...

@final
class SlabTable:
    """
    A slab table that pools in several processes share (see the module docs).
    """
    @property
    def boundary(self, /) -> int:
        """
        The first slab of the replicas' side; the tensor-parallel model holds the slabs below it.
        """
    @property
    def capacity(self, /) -> int: ...
    @property
    def cards(self, /) -> int: ...
    @staticmethod
    def create(path: str |PathLike[str], capacity: int, cards: int, tp_floor: int, replica_floors: Sequence[int]) -> SlabTable:
        """
        Make the table at `path` (a new file): `capacity` slabs, `cards` replica cards, and the floor of the
        tensor-parallel model and of each replica (slabs each keeps). The boundary starts halfway between the
        floors.
        """
    @staticmethod
    def open(path: str |PathLike[str]) -> SlabTable:
        """
        Map the table that `create` made at `path`.
        """
    @property
    def path(self, /) -> Path: ...
    def side(self, /, side: int) -> SlabSide:
        """
        The side of the tensor-parallel model (0) or of the replica on card `side` (1 to `cards`).
        """
    def sides(self, /) -> list[tuple[int, int, int, int |None, bool]]:
        """
        Each side: (floor, slabs on its side of the boundary, the edge slab it gave or 0, oldest prefix stamp or
        None, whether it wants slabs).
        """

@final
class StepChannelDriver:
    """
    The driver end: one socket per follower, in rank order.
    """
    def __new__(cls, /, path: "str | os.PathLike[str]", token: "bytes", ranks: "Sequence[int]") -> StepChannelDriver:
        """
        Listen on `path` for the followers `ranks`; `token` is the secret they must send.
        """
    def accept(self, /, timeout: "float") -> "None":
        """
        Wait until every follower has joined, then remove the socket path.
        """
    def close(self, /) -> "None":
        """
        Close every socket; followers see the end of the stream.
        """
    @property
    def followers(self, /) -> list[tuple[int, int]]:
        """
        The followers' ranks and process ids, in rank order.
        """
    def gather(self, /, kind: "int", seq: "int", timeout: "float | None", what: "str") -> "list[tuple[int, int, npt.NDArray[np.uint8]]]":
        """
        One frame of `kind` and `seq` from every follower, in rank order: (rank, status, payload).
        """
    def send(self, /, kind: "int", seq: "int", status: "int", pieces: "Sequence[npt.NDArray[np.uint8]]", timeout: "float", what: "str") -> "None":
        """
        Send one frame to every follower.
        """

@final
class StepChannelFollower:
    """
    A follower end: one socket to the driver.
    """
    def __new__(cls, /, path: "str | os.PathLike[str]", token: "bytes", rank: "int", timeout: "float") -> StepChannelFollower:
        """
        Connect to the driver at `path` as `rank` and send HELLO with `token`.
        """
    def close(self, /) -> "None":
        """
        Close the socket; the driver sees the end of the stream.
        """
    def receive(self, /, timeout: "float | None", what: "str") -> "tuple[int, int, int, npt.NDArray[np.uint8]]":
        """
        The next frame from the driver: (kind, seq, status, payload). `timeout` None waits forever.
        """
    def send(self, /, kind: "int", seq: "int", status: "int", pieces: "Sequence[npt.NDArray[np.uint8]]", timeout: "float", what: "str") -> "None":
        """
        Send one frame to the driver.
        """

@final
class Text:
    def __new__(cls, /, text: str) -> Text: ...
    @property
    def text(self, /) -> str: ...

@final
class TextDelta(ParserEvent):
    def __new__(cls, /, delta: str, *, token_spans: Sequence[tuple[int, int, int, int, int]] = ...) -> TextDelta: ...
    @property
    def delta(self, /) -> str: ...

@final
class Thinking:
    def __new__(cls, /, text: str) -> Thinking: ...
    @property
    def text(self, /) -> str: ...

@final
class TokenEvent:
    def __repr__(self, /) -> str: ...
    @property
    def finish_reason(self, /) -> str |None: ...
    @property
    def finished(self, /) -> bool: ...
    @property
    def index(self, /) -> int: ...
    @property
    def logprobs(self, /) -> TokenLogprobs |None: ...
    @property
    def request_id(self, /) -> int: ...
    @property
    def token_id(self, /) -> int: ...

@final
class TokenLogprobs:
    def __new__(cls, /, logprob: float, top_logprobs: Sequence[tuple[int, float]] = ..., target_logprobs: Sequence[tuple[int, float]] = ...) -> TokenLogprobs: ...
    @property
    def logprob(self, /) -> float: ...
    @property
    def target_logprobs(self, /) -> list[tuple[int, float]]: ...
    @property
    def top_logprobs(self, /) -> list[tuple[int, float]]: ...

@final
class Tool:
    def __new__(cls, /, name: str, *, description: str |None = None, parameters: Any |None = None, response: Any |None = None, namespace: Any |None = None) -> Tool: ...
    @property
    def description(self, /) -> str |None: ...
    @property
    def name(self, /) -> str: ...
    @property
    def namespace(self, /) -> Any: ...
    @property
    def parameters(self, /) -> Any: ...
    @property
    def response(self, /) -> Any: ...

@final
class ToolCall:
    def __new__(cls, /, name: str, arguments: Any, *, id: str |None = None, namespace: Any |None = None) -> ToolCall: ...
    @property
    def arguments(self, /) -> Any: ...
    @property
    def id(self, /) -> str |None: ...
    @property
    def name(self, /) -> str: ...
    @property
    def namespace(self, /) -> Any: ...

@final
class ToolCallArgumentsDelta(ParserEvent):
    def __new__(cls, /, index: int, delta: str, *, token_spans: Sequence[tuple[int, int, int, int, int]] = ...) -> ToolCallArgumentsDelta: ...
    @property
    def delta(self, /) -> str: ...
    @property
    def index(self, /) -> int: ...

@final
class ToolCallStart(ParserEvent):
    def __new__(cls, /, index: int, id: str, name: str, *, token_spans: Sequence[tuple[int, int, int, int, int]] = ...) -> ToolCallStart: ...
    @property
    def id(self, /) -> str: ...
    @property
    def index(self, /) -> int: ...
    @property
    def name(self, /) -> str: ...

@final
class ToolResult:
    def __new__(cls, /, content: Any, *, call_id: str |None = None, name: str |None = None) -> ToolResult: ...
    @property
    def call_id(self, /) -> str |None: ...
    @property
    def content(self, /) -> Any: ...
    @property
    def name(self, /) -> str |None: ...

@final
class Video:
    def __new__(cls, /, *, source: Any) -> Video: ...
    @property
    def source(self, /) -> Any: ...

@final
class _HistogramAccumulator:
    """
    Native accumulator; snapshots are plain Python dictionaries.
    """
    def __new__(cls, /) -> _HistogramAccumulator: ...
    def attach(self, /, snapshot: RunnerMetrics) -> RunnerMetrics:
        """
        Return a snapshot with this coordinator's wait observations included.
        """
    def observe(self, /, value: float, *, weight: int = 1) -> None: ...
    def snapshot(self, /) -> "dict[str, Any]": ...

@final
class _HttpEvents:
    def __new__(cls, /) -> _HttpEvents: ...
    @staticmethod
    def combine(packets: Sequence[_HttpEvents]) -> _HttpEvents: ...
    @staticmethod
    def from_bytes(data: bytes) -> _HttpEvents: ...
    def to_bytes(self, /) -> bytes: ...
    def with_metrics(self, /, snapshots: Sequence[Sequence[int]]) -> _HttpEvents: ...

@final
class _HttpIngress:
    """
    Native ingress stays typed across the Python coordinator. Only transport
    between ranks encodes it; JSON never becomes a Python object graph.
    """
    def __new__(cls, /, *, shutdown: bool = False) -> _HttpIngress: ...
    @property
    def collect_metrics(self, /) -> bool: ...
    @property
    def error(self, /) -> str |None: ...
    @staticmethod
    def from_bytes(data: bytes) -> _HttpIngress: ...
    @property
    def shutdown(self, /) -> bool: ...
    def to_bytes(self, /) -> bytes: ...

@final
class _HttpServer:
    def __new__(cls, /, engine: Engine, recipe: "Recipe", *, model: str, stream_interval: int = 1, request_queue_capacity: int = 1024, drain_seconds: float = 30.0, enable_openai: bool = True, enable_anthropic: bool = True, enable_demo: bool = True, distributed: Any |None = None, capabilities: dict |None = None, share: float = 1.0) -> _HttpServer: ...
    def __repr__(self, /) -> str: ...
    @property
    def model(self, /) -> str:
        """
        The first model's id.
        """
    @property
    def models(self, /) -> list[str]:
        """
        Every served model's id, in registration order (replicated models last).
        """
    def register(self, /, engine: Engine, recipe: "Recipe", *, model: str, stream_interval: int = 1, capabilities: dict |None = None, share: float = 1.0) -> None:
        """
        Serve one more model on its own Engine (before `serve`). `share` weighs its card time
        against the other models': the Engines step in turn on one card.
        """
    def register_replicas(self, /, router: _ReplicaRouter, *, model: str) -> None:
        """
        Serve one more model whose Engines run in other processes, one per card, through
        `router` (a `_ReplicaRouter` whose replicas have joined). It can sit beside a
        distributed model. Each request goes to one replica.
        """
    @property
    def running(self, /) -> bool: ...
    def serve(self, /, *, host: str = "127.0.0.1", port: int = 8000, on_listening: Any |None = None) -> None:
        """
        Serve until shutdown. `on_listening`, if given, is called (with no arguments) once the
        listener is bound and SIGTERM drains the server: from then on a client connects.
        """
    def shutdown(self, /) -> None:
        """
        Request shutdown from another thread; serve returns after the current
        inference step has finished and its requests have been cancelled.
        """

@final
class _HttpSession:
    """
    The same native lifecycle used by the ordinary actor, attached to one DP driver.
    """
    def __new__(cls, /, engine: Engine, recipe: "Recipe", *, stream_interval: int = 1, emit_first_token: bool = False) -> _HttpSession: ...
    @property
    def active_count(self, /) -> int: ...
    def admit(self, /, ingress: _HttpIngress, *, rank: int = 0) -> None: ...
    def cancel(self, /, ids: Sequence[str]) -> None: ...
    def close(self, /) -> None: ...
    def process(self, /, events: Sequence[TokenEvent]) -> _HttpEvents: ...

class _NativeResponseParser:
    """
    Common native handle recognized directly by the HTTP actor.
    """
    def feed(self, /, token_ids: "Sequence[int] | npt.NDArray[np.uint32]") -> list[ParserEvent]: ...
    def finish(self, /, reason: "Literal[\"stop\", \"length\"]") -> list[ParserEvent]: ...

@final
class _ReplicaRouter:
    """
    The router end: accepts the replicas of one model, then belongs to the HTTP server.
    """
    def __new__(cls, /, path: "str | os.PathLike[str]", token: "bytes", count: "int") -> _ReplicaRouter:
        """
        Listen on `path` for `count` replicas; `token` is the secret they must send.
        """
    def accept(self, /, timeout: "float") -> "str":
        """
        Wait until every replica has joined and sent READY; return the model's metadata as JSON.
        Every replica must report the same metadata.
        """
    @property
    def count(self, /) -> int: ...

@final
class _ReplicaServer:
    """
    The replica end: serves one Engine's requests from the router until shutdown.
    """
    def __new__(cls, /, path: "str | os.PathLike[str]", token: "bytes", index: "int", engine: Engine, recipe: "Recipe", *, model: str = ..., stream_interval: int = 1, capabilities: dict |None = None, timeout: float = 600.0, gate: Any |None = None) -> _ReplicaServer:
        """
        Join the router at `path` as replica `index` (from 0) and send READY. With a `gate`, the
        replica steps only when `gate.acquire(timeout)` returns true, and calls `gate.release()`
        after each step.
        """
    @property
    def index(self, /) -> int:
        """
        The router's index of this replica.
        """
    def serve(self, /) -> "None":
        """
        Serve until the router asks this replica to stop. An error stops the replica: it tells
        the router, cancels its requests and raises.
        """

def _media_from_buffers(metadata: bytes, buffers: "Sequence[npt.NDArray[Any]]") -> "MediaInput": ...
def _media_to_buffers(media: "MediaInput") -> "tuple[bytes, tuple[npt.NDArray[Any], ...]]": ...

def _numa_get_policy() -> tuple[int, list[int]]:
    """
    Read the calling thread's allocation policy without changing existing pages.
    """

def _numa_set_policy(mode: int, nodes: Sequence[int]) -> None:
    """
    Set the calling thread's allocation policy for subsequent allocations.
    """

def _runner_metrics_prometheus_protobuf(snapshot: RunnerMetrics, *, model_name: str, rank: int, attn_dp_rank: int) -> bytes:
    """
    Encode physical-rank observations separately from logical request ownership.
    """

def get_native_recipe(name: str, tokenizer: Any, *, stop_token_ids: "Sequence[int] | None" = None, **kwargs) -> NativeRecipe:
    """
    Construct a native recipe from a canonical name and typed family parameters.
    """

def image_token_types(height: int, width: int) -> "tuple[int, ...]": ...

def qwen_philox_uniform(seed: int, offset: int) -> float:
    """
    `common.sampling.philox_uniform` of one seed and offset (for the CPU checks of the native sampling inputs).
    """

def set_log_level(level: int) -> None:
    """
    Let native code log at and above a Python logging level (10 debug, 20 info,
    30 warning, 40 error). Levels above error silence it; `logging.NOTSET` (0)
    and other levels at most 5 enable trace output.
    """

def set_log_sink(prefix: str |None = None, colors: bool = False, path: str |PathLike[str] |None = None) -> None:
    """
    Write native lines without Python, as the server's handlers do: `prefix` before each message
    (a rank's or a replica's), coloredlogs' colors on stderr when `colors`, and each line also
    appended to `path`. `None` sends native records to Python's `logging` again.
    """

def __getattr__(name: str) -> Incomplete: ...
