# Unified paged cache

Each Engine owns one pool of equal-sized physical slabs. A runner declares an
ordered `CacheLayout`; every `CacheGroupLayout` explains how one model state is
packed into those slabs:

- `family=HISTORY` stores token-addressed attention or rolling history.
- `family=STATE` stores recurrent checkpoints.
- `retention=HISTORY` keeps the full logical history. State groups retain only
  their current and preceding checkpoints.
- `retention=WINDOW` keeps a bounded chronological suffix.
- `window_tokens` is that suffix's physical retention budget;
  `read_window_tokens` is the logical window including the current token.
  It defaults to `window_tokens`. To write a chunk of `Q` tokens while its
  earliest query reads a window of `W` tokens, reserve at least `W + Q - 1`
  retained tokens and declare `read_window_tokens=W`.
- `blocks_per_slab` says how many blocks of that type fit in one slab.

Slabs are fungible. They are not divided into fixed History, Window, or State
regions. Allocation binds a free slab to whichever group currently needs it,
and retirement returns it to the common pool. Admission estimates the slab
demand for every request and group, including packing, before admitting work.

The runner allocates the matching device arena once and receives its pool ID and
frozen layout through `bind_resources`. A `BatchPlan` does not repeat that
identity. It contains only the ordered `CacheGroupPlan`s whose structured
physical addresses refer to the already-bound arena.

## Batch plans

`BatchPlan.cache_groups` has one plan per declared block type—such as full
attention, sliding-window attention, or recurrent state—not one per sequence.
The cache is a pool of equally sized physical slabs. Each slab in use is
assigned to one cache group and fits that group's `blocks_per_slab` blocks.
A physical block is identified by `(slab_id, slotnum)`. A cached token location is
`(block, block_offset)`, equivalently `(slab_id, slotnum, block_offset)`. Each
group plan covers every batch sequence and has one CSR block table:

- `block_indptr` slices the parallel `block_slab_ids` and
  `block_slotnums` arrays by sequence.
- `block_table_start_indices[i]` is the absolute chunk index represented by
  sequence `i`'s first table entry. For a group with 64-token blocks, block 37
  covers token positions 2368 through 2431. Full-history rows begin at zero;
  sliding-window rows advance when earlier chunks are evicted.
- `append_slab_ids`, `append_slotnums`, and `append_block_offsets` give one
  exact cached-token write location per scheduled token.

The Engine allocates blocks, performs copy-on-write decisions, and holds the
reader/writer lease through synchronous execution and native commit. Model code maps
each `(slab_id, slotnum)` pair into its model-specific tensor row. Slab zero is
reserved for inactive/null rows; live slab IDs are one-based. Keeping the
address structured prevents runners from assuming a different packing than the
allocator.

## Prefix caching

`Engine(..., prefix_cache=True)` enables an exact-token block trie with LRU
entries. Prefixes are published only at an alignment shared by every cache
group. A hit attaches immutable references to the cached pages and schedules
only the unmatched suffix.

A model can declare `CacheLayout(replay_suffix_tokens=N)` when a restored
history needs its final `N` inputs recomputed. Prefix lookup caps the hit at
`prefill_length - N` before applying cache alignment. The scheduler does not
interpret those inputs or choose model layers. A prefill chunk that passes
`prefill_length - N` stops at the last cache boundary at or before that point.
Thus the request publishes the longest prefix that a repeat of its prompt can
restore. `BatchPlan.prefill_lengths`
provides each request's final prefill endpoint across chunks, including
generated inputs when reconstructing a retracted request. Scored prompts still
take the full prefill path.

Live requests have priority over cached entries. When allocation needs more
slabs, the least-recently-used prefix is evicted. Advancing a shared Window or
State ring uses copy-on-write so one request cannot overwrite another request's
prefix.

State groups use two checkpoint pages per live sequence. At a restored boundary,
the cached input page remains read-only and the next checkpoint page is private
and writable. Within a checkpoint interval, decode updates the private page in
place. This keeps recurrent state prefix-compatible without a second slot
allocator.

## Physical sizing

The one-card Qwen runners use the shared
[`plan_cache_capacity`](../resin/backend/capacity.py) helper to choose slab capacity
from free card memory. They do this after model load and before cache allocation,
within their memory-utilization targets and workspace reservations.
Runners may also bound that capacity explicitly. The slab byte size is the
largest group allocation. Smaller groups set `blocks_per_slab` from their exact
byte ratio, so the allocator can compare all demand in slab units.

Slab zero is reserved as the device null target. Positive slab IDs are stable for
the lifetime of the Engine-issued lease.

Window read history and all scheduled writes share one chronological table.
The Engine caps prefill chunks by the smallest `window_tokens -
read_window_tokens + 1` across Window groups and rejects decode lookahead that
exceeds it. The same bound is checked before native cache growth or copy-on-write.
This prevents a later write in a chunk from overwriting the earlier queries'
history, including when a prefix hit shares those pages with another request.

## Several Engines on one pool

A process that serves more than one model has one Engine for each model. These Engines can use
one pool of slabs, so the models share the cache memory of the card:

```python
pool = CachePool(parent_capacity)
first = Engine(first_runner, cache_pool=pool)
second = Engine(second_runner, cache_pool=pool, cache_floor=64)
```

Each runner declares a `CacheLayout` with the pool's `parent_capacity`. All runners address one
device arena, and they use the same slab size. Every runner's `ResourceBinding` has the same
`cache_pool_id`.

The rules for slabs are these:

- A free slab goes to the Engine that needs it next. The slab belongs to that Engine while it
  holds that Engine's blocks, and it goes back to the pool when it is free.
- When an Engine needs a slab and no slab is free, the pool evicts the least recently used cached
  prefix of any Engine. The prefixes of all Engines are in one LRU order.
- An Engine never evicts the live requests of another Engine.
- An Engine admits requests only to the capacity that the other Engines leave: the pool's
  capacity less the live slabs (or the floor, if it is larger) of each other Engine.
- `cache_floor` reserves slabs for an Engine. The other Engines do not take the free slabs that
  the floor still needs. They do not evict that Engine's prefixes while it holds no more slabs
  than its floor.
- If an Engine cannot grow a request because other Engines' live requests hold the slabs, its
  step does nothing and it waits. If every other Engine that holds live slabs also waits, the
  Engine gives back the slabs of one of its own requests, which restarts later, so that the
  others can continue. A request that cannot fit, with no live request of another Engine to wait
  for, fails as it does with one Engine.

Slab counts in `/metrics` (`cache_slabs`) are the pool's. Group and prefix metrics are each
Engine's own. `CachePool.bound_slabs()` gives the slabs that each Engine's blocks occupy.
