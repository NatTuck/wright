"""FP8 forward of Qwen3.5/3.8 on one Gaudi card over the paged cache arena, one recipe launch
per layer: one sequence's prefill chunk, one decode token for each of a batch of sequences, or,
with MTP, a speculative verify step for each of a batch of sequences.

Every recipe follows from the configuration (prefill buckets, decode batch buckets, MTP draft
length, slots per attention launch, rope table length, arena size) and is compiled at
construction. The arena is sized first, from the memory left after weights,
and rounded down to a multiple of ``CAPACITY_QUANTUM`` slabs.

A step runs ``embed``, the layers and ``head``; a full-attention layer is ``attn_pre``, as many
``attn_slots`` launches as the context needs and ``attn_post``. Per-step tables are written once:
attention tables use layer 0's page rows, and each full-attention layer binds the arena at its
own row offset (a layer's pages sit at a fixed offset inside every KV block). GDN state rows are
per-layer tables at fixed addresses.

MTP (``mtp`` = K drafts): a decode step verifies [anchor, d_1 .. d_K] of every sequence in one
target pass (T = K + 1 per sequence; ``gdn_verify`` records the GDN state updates instead of
writing them), samples y_0 .. y_K, and runs the MTP layer at every verify position. The host
accepts d_1 .. d_m while d_t = y_(t-1), emits y_0 .. y_m, commits the first m + 1 tokens' state
(``gdn_commit``) and the next drafts are the MTP's draft at position m chained K - 1 times
through the MTP layer. A prefill runs the MTP layer over its chunk (filling its KV) and drafts
after the sampled token. Output equals non-speculative decoding: every token is the target's
sample at its position.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from resin.backend.capacity import plan_cache_capacity
from resin.backend.speculative import SpeculativeStats
from resin_models_gaudi.qwen3_5.attention import SlotRow, kv_rows, plan_slots
from resin_models_gaudi.qwen3_5.cache import ROW_BYTES, STATE_TOKENS, CacheGeometry
from resin_models_gaudi.qwen3_5.checkpoint import Qwen35Checkpoint, rope_table
from resin_models_gaudi.qwen3_5.layers import (
    ARENA_BLOCK_ROWS,
    QUERY_BLOCK,
    SCORE_PAIRS,
    SCORE_ROWS,
    SLOT_KEYS,
    Dims,
    attn_post_graph,
    attn_pre_graph,
    attn_proj,
    attn_slots_graph,
    commit_graph,
    embed_graph,
    gdn_layer_graph,
    head_graph,
    hidden_store_graph,
    mtp_in_graph,
    score_graph,
)
from resin_models_gaudi.qwen3_5.sampling import (
    PARAMS,
    TOP_MAX,
    philox_uniform,
    row_params,
)
from resin_models_gaudi.qwen3_5.weights import DeviceWeights
from resin_models_gaudi.synapse.graph import LaunchPlan, Recipe, RecipeCache
from resin_models_gaudi.synapse.progress import weight_file_bytes
from resin_models_gaudi.synapse.runtime import Device

PREFILL_BUCKETS = (128, 256, 512, 1024, 2048, 4096)
DECODE_BUCKETS = (1, 2, 4, 8, 16, 32)
SMALL = 256  # bytes of each small device buffer
WORKSPACE_RESERVE = (
    2 << 30
)  # bytes held back for recipe workspaces
CAPACITY_QUANTUM = 8
SCORE_TOPS = (0, 64, 1024)  # top-logprob list lengths of the scoring recipes
OFFSET_STRIDE = 32  # Philox offsets per position (resin.backend.sampling)
DYNAMIC = ("slots", "k_pages", "v_pages", "mtp_prev", "mtp_tok")  # bound per launch


def prefill_chunks(
    n: int, buckets: Sequence[int] = PREFILL_BUCKETS
) -> list[tuple[int, int]]:
    """(real tokens, bucket T) per chunk: the largest bucket while it fits, then the smallest
    bucket holding the rest when that pads by under 25%, else the largest bucket that fits."""
    chunks = []
    while n > 0:
        real = min(n, buckets[-1])
        if real < buckets[-1]:
            bucket = min(b for b in buckets if b >= real)
            if bucket > 1.25 * real:
                real = max([b for b in buckets if b <= real] or [real])
        chunks.append((real, min(b for b in buckets if b >= real)))
        n -= real
    return chunks


def _round(nbytes: int) -> int:
    return max(SMALL, -(-nbytes // SMALL) * SMALL)


@dataclass(frozen=True, slots=True)
class RowCache:
    """One sequence's cache for a step: its first position, the arena rows of its KV blocks (and
    MTP KV blocks) in logical order covering every position the step writes, and its STATE
    blocks: ``state_src`` holds its state after position pos0 - 1 (-1: fresh), ``state_dst``
    receives the state after the step's last token; ``state_rows`` (from logical block
    ``state_first``) resolve any position's block (a verify step commits a variable number)."""

    pos0: int
    kv_blocks: np.ndarray
    state_src: int
    state_dst: int
    mtp_blocks: np.ndarray | None = None
    state_first: int = 0
    state_rows: tuple[int, ...] = ()
    # what the rope index is offset by: the prompt's mrope delta (`ImagePrompt.delta`). Only
    # the rope is compressed - `pos0` is the token position and stays the cache's index, and
    # `pos0 + pos_delta` is the position the model rotates as. After an image prompt the two
    # differ by the image's token count minus its merged grid's larger side.
    pos_delta: int = 0

    @property
    def rope_at(self) -> int:
        """The rope position of this row's first token."""
        return self.pos0 + self.pos_delta

    def state_at(self, position: int) -> int:
        if not self.state_rows:
            return self.state_dst
        return self.state_rows[position // STATE_TOKENS - self.state_first]


@dataclass(frozen=True, slots=True)
class Sampling:
    """Sampler inputs of the rows (``sampling.row_params``), with each row's seed and the Philox
    offset of its first sampled position, so later positions (verify columns, drafts) get their
    own uniforms."""

    params: np.ndarray
    seeds: np.ndarray | None = None
    offsets: np.ndarray | None = None

    def at(self, rows: Sequence[int], steps: Sequence[int]) -> np.ndarray:
        """Params of ``rows`` [i] at ``steps`` [i] positions past their first sampled one."""
        rows = np.asarray(rows, np.int64)
        return self._uniforms(
            self.params[rows].copy(), rows, np.asarray(steps, np.int64)
        )

    def grid(self, rows: int, steps: Sequence[int]) -> np.ndarray:
        """(rows, len(steps), PARAMS): the first ``rows`` rows at each of ``steps``, in one
        Philox evaluation."""
        steps = np.asarray(steps, np.int64)
        index = np.repeat(np.arange(rows), len(steps))
        out = self._uniforms(self.params[index].copy(), index, np.tile(steps, rows))
        return out.reshape(rows, len(steps), -1)

    def _uniforms(
        self, out: np.ndarray, rows: np.ndarray, steps: np.ndarray
    ) -> np.ndarray:
        """Each sampled row's uniform at its step (greedy rows never read theirs)."""
        if self.seeds is None or not len(out):
            return out
        sampled = out[:, 4] > 0.5
        if sampled.any():
            offsets = self.offsets[rows[sampled]] + steps[sampled] * OFFSET_STRIDE
            out[sampled, 3] = philox_uniform(
                self.seeds[rows[sampled]], offsets.astype(np.uint64)
            ).astype(np.float32)
        return out


@dataclass(frozen=True, slots=True)
class ScoreRequest:
    """Prompt positions of a prefill to score: ``positions`` (indices into its tokens, whose
    next-token distributions are scored), per position the token ids whose raw logprobs are
    wanted (``targets``) and the number of top logprobs (``tops``, at most SCORE_TOPS[-1])."""

    positions: np.ndarray
    targets: Sequence[np.ndarray]
    tops: np.ndarray


@dataclass(frozen=True, slots=True)
class Scores:
    """Per scored position: the targets' logprobs, and the top ids and logprobs."""

    targets: list[np.ndarray]
    top_ids: list[np.ndarray]
    top_logprobs: list[np.ndarray]


@dataclass(frozen=True, slots=True)
class Prefilled:
    """A prefill's next token, its prompt scores and (MTP) the drafts after the token."""

    token: int
    scores: Scores | None = None
    drafts: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class Verified:
    """A verify step: each row's emitted tokens (y_0 .. y_m), its accepted draft count m, its
    next drafts, and the step's columns of each row (for their logprobs)."""

    tokens: list[list[int]]
    accepted: np.ndarray
    drafts: np.ndarray
    columns: list[np.ndarray] = field(default_factory=list)


class Staging:
    """Pinned host buffers one in-flight step copies its outputs into, and its two events:
    0 after the step's outputs (tokens, chosen-token logprobs and top lists per column; a
    verify step's MTP drafts per column), 1 after a verify step's draft chain (drafts 2 .. K,
    (K, B))."""

    def __init__(self, device: Device, columns: int, drafts: int, rows: int, slot: int):
        self.device, self.slot = device, slot
        self.tokens = device.host_buffer(columns, np.int32)
        self.mtp_tokens = device.host_buffer(columns, np.int32)
        self.lp = device.host_buffer(columns, np.float32)
        self.top_idx = device.host_buffer((columns, TOP_MAX), np.int32)
        self.top_lp = device.host_buffer((columns, TOP_MAX), np.float32)
        self.drafts = device.host_buffer((max(drafts, 1), rows), np.int32)

    def record(self, event: int = 0) -> None:
        self.device.record(self.slot + 2 * event)

    def wait(self, event: int = 0) -> None:
        self.device.wait(self.slot + 2 * event)


@dataclass(frozen=True, slots=True)
class _Mode:
    """A step shape: ``name`` keys its recipes (p{T} prefill, d{B} decode, v{B} verify)."""

    name: str
    columns: int
    per_row: int
    query_block: int
    slots: int


@dataclass(frozen=True, slots=True)
class Launched:
    """A verify pass left running (``Qwen35Gaudi.verify_launch``), for ``verify_finish``."""

    rows: Sequence[RowCache]
    drafts: np.ndarray | Callable[[], np.ndarray]
    sampling: Sampling | None
    stage: Staging | None
    mode: _Mode


@dataclass(frozen=True, slots=True)
class _Binds:
    slots: np.ndarray
    addresses: np.ndarray


def slice_media(media, i: int, real: int, columns: int):
    """The part of a prompt's media that falls inside the chunk starting at token ``i``.

    ``media`` is (features (image tokens, hidden), first prompt column, count, rope rows
    (prompt tokens, 128), the prompt's mrope delta); the image tokens are contiguous in the
    prompt, which is what makes a (first, count) span enough. Returns the same five for this
    chunk, or None when the chunk holds no image token at all - the usual case for a long
    text prompt, where the plan must not carry the splice.
    """
    features, first, count, rope, delta = media
    lo = max(first - i, 0)
    hi = min(first + count - i, real)
    if hi <= lo:
        return None
    k0 = lo + i - first
    rows = rope[i : i + columns]
    if len(rows) < columns:  # the last chunk is short; the tail columns are padding
        rows = np.concatenate(
            [rows, np.zeros((columns - len(rows), rope.shape[1]), np.float32)]
        )
    return (
        features[k0 : k0 + (hi - lo)],
        lo,
        hi - lo,
        np.ascontiguousarray(rows, np.float32),
        delta,
    )


def image_block(features: np.ndarray, count: int, tokens: int, hidden: int) -> np.ndarray:
    """The host array `splice`'s ``img`` input must receive: (tokens, hidden) f32.

    Device tensors are token-major - a logical ``(n, T)`` tensor lives at flat ``t * n + i``
    - so the ``(H, T)`` tensor the embed graph declares takes a ``(T, H)`` host array, and
    the features go in its first ``count`` rows. ``features`` is (count, hidden).

    Getting this wrong is silent and severe: the compact ``(hidden, count)`` array the
    upload used to send put feature ``k`` at flat ``h * count``, which the kernel read as
    some other ``(h, t)``. Every value was still there, in the wrong place, so the model
    could tell an image was present and never what it was. `test_qwen_vision` pins the
    layout; `emb_img_check` and the accuracy set pin the behaviour.
    """
    features = np.asarray(features, np.float32)
    if features.shape != (count, hidden):
        raise ValueError(f"features must be ({count}, {hidden}), got {features.shape}")
    block = np.zeros((tokens, hidden), np.float32)
    block[:count] = features
    return block


class Qwen35Gaudi:
    """``prefill(row, ids)`` runs one sequence's tokens; ``decode(rows, tokens)`` one token of
    each sequence; with MTP, ``verify(rows, anchors, drafts)`` one speculative step of each."""

    def __init__(
        self,
        device: Device,
        checkpoint: str,
        *,
        parent_capacity: int | None = None,
        memory_utilization: float = 0.9,
        workspace_reserve: int = WORKSPACE_RESERVE,
        max_positions: int = 32768,
        buckets: Sequence[int] = PREFILL_BUCKETS,
        decode_buckets: Sequence[int] = DECODE_BUCKETS,
        prefill_slots: int = 16,
        decode_slots: int | None = None,
        mtp: int = 0,
        layers: int | None = None,
        media: bool = False,
    ) -> None:
        self.device = device
        # a text-only deployment must not pay for the media embed graph, and a request that
        # arrives with media on a model built without it is a bug worth failing on
        self.media = media
        ckpt = Qwen35Checkpoint(checkpoint)
        self.config = config = ckpt.config
        self.dims = Dims.from_config(config)
        self.mtp = mtp
        # verify acceptance counters (and the benchmark's fixed-acceptance mode)
        self.speculation = SpeculativeStats(mtp)
        self.geometry = CacheGeometry.from_config(config, mtp=mtp > 0)
        self.n_layers = layers or config.num_hidden_layers
        self.mtp_layer = config.num_hidden_layers  # the MTP layer's weight index
        self.full = [i for i in range(self.n_layers) if i in set(config.full_layer_ids)]
        self.gdn = [i for i in range(self.n_layers) if i not in set(self.full)]
        self.buckets = tuple(buckets)
        self.decode_buckets = tuple(decode_buckets)
        self.max_positions = max_positions
        self.modes: dict[str, _Mode] = {}
        for T in self.buckets:
            self.modes[f"p{T}"] = _Mode(
                f"p{T}", T, T, min(T, QUERY_BLOCK), prefill_slots
            )
        for B in self.decode_buckets:
            # by default one 2048-key chunk of every sequence and KV group per launch
            S = decode_slots or max(16, self.dims.kv_groups * B)
            self.modes[f"d{B}"] = _Mode(f"d{B}", B, 1, 1, S)
            if mtp:
                Tq = mtp + 1
                self.modes[f"v{B}"] = _Mode(f"v{B}", B * Tq, Tq, Tq, S)
        self.columns = max(m.columns for m in self.modes.values())
        self.recipes = RecipeCache(device)
        weight_layers = set(range(self.n_layers)) | ({self.mtp_layer} if mtp else set())
        device.begin_load(weight_file_bytes(ckpt.path, "*.safetensors"))
        self.weights = DeviceWeights(
            device, self.recipes, ckpt, mtp=mtp > 0, layers=weight_layers
        )
        device.end_load()
        self._buffers()
        if parent_capacity is None:
            free, total = device.memory_info()
            planned = plan_cache_capacity(
                total,
                free,
                memory_utilization,
                self.geometry.slab_bytes,
                overhead_bytes=workspace_reserve,
            ).parent_capacity
            parent_capacity = max(1, planned // CAPACITY_QUANTUM * CAPACITY_QUANTUM)
        self.parent_capacity = parent_capacity
        self.blocks = self.geometry.arena_blocks(parent_capacity)
        nbytes = self.blocks * ARENA_BLOCK_ROWS * ROW_BYTES
        self.arena_bytes = nbytes
        self.arena = device.malloc(nbytes)
        device.memset32(self.arena, 0, nbytes // 4)
        logging.getLogger("wright").info("preparing recipes")
        self._compile()
        workspace = max(r.workspace for r in self.recipe.values())
        if workspace > workspace_reserve:
            raise MemoryError(
                f"recipe workspace {workspace} exceeds the reserve {workspace_reserve}"
            )
        self._bind_all()

    # ---- setup
    def _buffers(self) -> None:
        d, dev = self.dims, self.device
        H, G, V = d.hidden, d.kv_groups, d.vocab
        T = self.columns
        B = self.decode_buckets[-1]
        W = d.hpg * T

        def buf(nbytes: int) -> int:
            return dev.malloc(_round(nbytes))

        def carry() -> dict[str, int]:
            return {
                "x": buf(H * T * 4),
                "h8": buf(H * T),
                "hs": buf(T * 4),
                "h16": buf(H * T * 2),
            }

        self.carry = [carry(), carry()]
        self.buf = {
            "p_out": buf(attn_proj(d) * T * 4),
            "q_all": buf(256 * W * G * 2),
            "acc_o": buf(256 * W * G * 4),
            "acc_m": buf(max(W, 64) * G * 4),
            "acc_l": buf(max(W, 64) * G * 4),
            "kv_rows": buf(T * 8 * 4),
            "tok": buf(T * 4),
            "pos": buf(T * 4),
            # the image rows of an embed splice, and its (first, count) span
            "img": buf(H * T * 4),
            "img_span": buf(2 * 4),
            "logits": buf(V * T * 4),
            "tok_out": buf(T * 4),
            "lp_out": buf(T * 4),
            "top_idx": buf(TOP_MAX * T * 4),
            "top_lp": buf(TOP_MAX * T * 4),
            "samp_params": buf(PARAMS * T * 4),
            "score_rows": buf(SCORE_ROWS * 4),
            "score_pairs": buf(2 * SCORE_PAIRS * 4),
            "score_lp": buf(SCORE_PAIRS * 4),
            "score_stats": buf(16 * SCORE_ROWS * 4),
            "score_params": buf(PARAMS * SCORE_ROWS * 4),
            "score_top_v": buf(SCORE_TOPS[-1] * SCORE_ROWS * 4),
            "score_top_i": buf(SCORE_TOPS[-1] * SCORE_ROWS * 4),
            "cs_tab": buf(self.max_positions * 512),
            # the media plan's own rope rows: the text table is never overwritten, so there is
            # no restore and nothing can race with the launches
            "cs_tab_media": buf(self.max_positions * 512),
            "hidden_table": buf(2 * B * 4),
        }
        for name in (
            "meta",
            "one",
            "last",
            "sink_conv",
            "sink_ssm",
            "sink_kv",
            "sink_acc",
            "sink_slots",
            "sink_commit",
            "sink_hidden",
        ):
            self.buf[name] = buf(SMALL)
        # per GDN layer: prefill [conv_src, ssm_src, conv_dst, ssm_dst] (1,) each; decode and
        # verify gdn_bases (4, B) and conv_dst (B,)
        self.pre_bases = buf(len(self.gdn) * 4 * SMALL)
        self.dec_stride = _round(4 * B * 4) + _round(B * 4)
        self.dec_bases = buf(len(self.gdn) * self.dec_stride)
        # slot tables of every launch of a step
        S = max(m.slots for m in self.modes.values())
        self.launch_stride = _round(S * 8 * 4) + 2 * _round(S * 16 * 4)
        chunks = -(-self.max_positions // SLOT_KEYS)
        self.max_launches = max(
            -(
                -(m.columns // m.per_row)
                * -(-m.per_row // m.query_block)
                * G
                * chunks
                // m.slots
            )
            for m in self.modes.values()
        )
        self.tables = buf(self.max_launches * self.launch_stride)
        dev.h2d(self.buf["cs_tab"], rope_table(self.max_positions, self.config))
        dev.h2d(self.buf["one"], np.ones(64, np.float32))
        # scoring rows: greedy, with the log-sum-exp
        score = row_params(SCORE_ROWS, d.vocab, logprobs=np.ones(SCORE_ROWS))
        dev.h2d(self.buf["score_params"], score)
        # two in-flight steps' output copies (the Engine keeps one step ahead)
        self.staging = [Staging(dev, T, self.mtp, B, slot) for slot in (0, 1)]
        if self.mtp:
            Tq, L = self.mtp + 1, len(self.gdn)
            self.mtp_carry = [carry(), carry()]
            self.mtp_buf = {
                "mtp_cols": buf(T * 4),
                "mtp_tok": buf(T * 4),
                "mtp_zero": buf(H * 4),
                "mtp_tok_out": buf(T * 4),
                "mtp_lp_out": buf(T * 4),
                "mtp_top_idx": buf(TOP_MAX * T * 4),
                "mtp_top_lp": buf(TOP_MAX * T * 4),
                "mtp_logits": buf(V * T * 4),
                "mtp_params": buf(PARAMS * T * 4),
                "commit_bases": buf(4 * B * L * 4),
                "commit_counts": buf(B * L * 4),
                # an enqueued step's continuation tokens [anchor, d_1 .. d_K] per row: the
                # anchor and d_1 from the host, d_2 .. d_K copied from the draft chain
                "next_tok": buf(B * Tq * 4),
            }
            dev.memset32(self.mtp_buf["mtp_zero"], 0, H)
            self.pend_layer = 128 * 3 * d.nv * Tq * 4  # bytes per sequence and layer
            self.pconv_layer = d.conv_dim * Tq * 4
            self.pend = buf(self.pend_layer * B * L)
            self.pconv = buf(self.pconv_layer * B * L)

    def _compile(self) -> None:
        dev, d, P = self.device, self.dims, self.max_positions
        nb = self.blocks
        na = self.geometry.attention_blocks(self.parent_capacity)
        graphs = {}
        for m in self.modes.values():
            T = m.columns
            verify = m.per_row if m.name[0] == "v" else 0
            graphs[("embed", m.name)] = embed_graph(dev, d, T)
            if self.media and m.name[0] == "p":
                # media only arrives in a prefill, and only a deployment that serves images
                # builds this variant, so a text-only preset keeps exactly the graphs it has
                # today - including at startup, where this would otherwise be one extra
                # compile for something that can never be used
                graphs[("embed_img", m.name)] = embed_graph(dev, d, T, media=True)
            graphs[("gdn", m.name)] = gdn_layer_graph(
                dev, d, T, nb, decode=m.name[0] != "p", verify=verify
            )
            graphs[("pre", m.name)] = attn_pre_graph(dev, d, T, na, P)
            graphs[("slots", m.name)] = attn_slots_graph(
                dev, d, T, m.slots, na, m.query_block
            )
            graphs[("post", m.name)] = attn_post_graph(dev, d, T)
            graphs[("head", m.name)] = head_graph(
                dev, d, T, 1 if m.name[0] == "p" else T
            )
            if self.mtp:
                graphs[("mtp_in", m.name)] = mtp_in_graph(dev, d, T, self.columns)
            if verify:
                B = T // verify
                graphs[("commit", m.name)] = commit_graph(
                    dev, d, B, verify, len(self.gdn), nb
                )
        for K in SCORE_TOPS:
            graphs[("score", f"k{K}")] = score_graph(dev, d, self.columns, K)
        if self.mtp:
            graphs[("hidden", "x")] = hidden_store_graph(dev, d, self.columns, nb)
        recipes = self.recipes.compile_many(list(graphs.values()))
        self.recipe: dict[tuple[str, str], Recipe] = dict(zip(graphs, recipes))

    def _bind_all(self) -> None:
        self.binds: dict[tuple[str, str, int, str], _Binds] = {}
        # launch plans of whole passes, by (pass, mode, slot launches per layer, ...)
        self._plans: dict[tuple, LaunchPlan] = {}
        M = self.mtp_layer
        for (kind, mode), recipe in self.recipe.items():
            if kind == "gdn":
                targets = [(i, "") for i in self.gdn]
            elif kind in ("pre", "slots", "post"):
                targets = [(i, "") for i in self.full]
                if self.mtp:
                    targets.append((M, "mtp"))
            elif kind == "head":
                targets = [(-1, "")] + ([(-1, "mtp")] if self.mtp else [])
            elif kind == "mtp_in":
                targets = [(M, "mtp"), (M, "mtpc")]
            else:
                targets = [(-1, "")]
            for layer, variant in targets:
                self.binds[(kind, mode, layer, variant)] = self._bind_table(
                    recipe, kind, mode, layer, variant
                )

    def _bind_table(
        self, recipe: Recipe, kind: str, mode: str, layer: int, variant: str
    ) -> _Binds:
        """Every persistent tensor of ``recipe`` launched as ``layer`` (-1: embed / head) in
        ``variant`` ("": the target, "mtp": the MTP layer fed from the target, "mtpc": the MTP
        layer fed from its own previous output), except the per-launch DYNAMIC ones."""
        w, g = self.weights, self.geometry
        final = self.carry[self.n_layers % 2]
        names = dict(self.buf)
        if self.mtp:
            names.update(self.mtp_buf)
        if variant:
            src, dst = self.mtp_carry
            if kind == "mtp_in":
                x = final["x"] if variant == "mtp" else self.mtp_carry[1]["x"]
                norm = "out_norm" if variant == "mtp" else "mtp.norm"
                names.update({f"{k}_out": v for k, v in src.items()})
                names.update(x=x, mtp_xnorm=w[norm].address)
            elif kind == "head":
                names["h16"] = dst["h16"]
                for k in ("tok_out", "lp_out", "top_idx", "top_lp", "logits"):
                    names[k] = self.mtp_buf[f"mtp_{k}"]
                names["samp_params"] = self.mtp_buf["mtp_params"]
            else:  # the MTP decoder layer: its own KV blocks, mtp.norm as the next norm
                names.update(src)
                names.update({f"{k}_out": v for k, v in dst.items()})
                names["arena"] = self.arena
                names["next_norm"] = w["mtp.norm"].address
        else:
            src = 0 if layer < 0 else layer % 2
            names.update(self.carry[src])
            names.update({f"{k}_out": v for k, v in self.carry[1 - src].items()})
            names["arena"] = self.arena
            if (
                layer < 0
            ):  # embed writes carry 0; the head and scoring read the final one
                names.update({f"{k}_out": v for k, v in self.carry[0].items()})
                names["h16"] = final["h16"]
                names["x"] = final["x"]
                names["next_norm"] = w["l0.attn_norm"].address
                if kind == "embed_img":
                    names["img"] = self.buf["img"]
                    names["img_span"] = self.buf["img_span"]
                if self.mtp:
                    names["gdn_pend"], names["gdn_pconv"] = self.pend, self.pconv
            else:
                nxt = (
                    f"l{layer + 1}.attn_norm"
                    if layer + 1 < self.n_layers
                    else "out_norm"
                )
                names["next_norm"] = w[nxt].address
            if layer in self.full:
                a = self.full.index(layer)
                names["arena"] = self.arena + a * g.kv_layer_rows * ROW_BYTES
            if layer in self.gdn:
                j = self.gdn.index(layer)
                if mode[0] == "p":
                    for k, key in enumerate(
                        ("conv_src", "ssm_src", "conv_dst", "ssm_dst")
                    ):
                        names[key] = self.pre_bases + (4 * j + k) * SMALL
                else:
                    base = self.dec_bases + j * self.dec_stride
                    names["gdn_bases"] = base
                    names["conv_dst"] = base + _round(4 * self.decode_buckets[-1] * 4)
                if mode[0] == "v":
                    B = self.modes[mode].columns // self.modes[mode].per_row
                    names["gdn_pend"] = self.pend + j * B * self.pend_layer
                    names["gdn_pconv"] = self.pconv + j * B * self.pconv_layer
        pairs = []
        for name in recipe.persistent:
            if name in DYNAMIC:
                continue
            if layer >= 0 and f"l{layer}.{name}" in w:
                address = w[f"l{layer}.{name}"].address
            elif name in w:
                address = w[name].address
            else:
                address = names[name]
            pairs.append((recipe.slot(name), address))
        return _Binds(
            np.array([p[0] for p in pairs], np.int32),
            np.array([p[1] for p in pairs], np.uint64),
        )

    # ---- launching
    def _upload(self, address: int, array: np.ndarray) -> None:
        """Queue a table upload behind the launches already submitted.

        ``h2d_nb`` carries at most 64 KiB, so a larger table goes as consecutive queued
        chunks instead of falling back to the blocking ``h2d``. That fallback was silent
        and changed the meaning of every caller: below 64 KiB the copy was ordered after
        the launches already submitted, above it the host waited for a copy that is not
        ordered against them at all. The tables this matters for are exactly the large
        ones - ``kv_rows`` for a big prefill bucket, the image block, a long rope table -
        and a copy landing under a pending launch is how a plan reads a table that has
        not been written yet.
        """
        raw = np.ascontiguousarray(array).view(np.uint8).reshape(-1)
        for offset in range(0, raw.size, 65536):
            chunk = np.ascontiguousarray(raw[offset : offset + 65536])
            self.device.h2d_nb(address + offset, chunk)

    def _launch(
        self,
        kind: str,
        mode: str,
        layer: int = -1,
        variant: str = "",
        extra: dict[str, int] | None = None,
    ) -> None:
        recipe = self.recipe[(kind, mode)]
        b = self.binds[(kind, mode, layer, variant)]
        recipe.bind_slots(b.slots, b.addresses)
        if extra:
            recipe.bind_all(extra)
        recipe.launch()

    def _slot_tables(self, mode: _Mode, rows: Sequence[SlotRow]) -> list[tuple]:
        """Write every slot launch's tables; returns each launch's (slots, k, v) addresses."""
        S = mode.slots
        launches = plan_slots(rows, mode.query_block, self.dims.hpg, S)
        if len(launches) > self.max_launches:
            raise ValueError(
                f"{len(launches)} attention launches exceed {self.max_launches}"
            )
        stride, k_off = self.launch_stride, _round(S * 8 * 4)
        v_off = k_off + _round(S * 16 * 4)
        table = np.zeros((len(launches), stride // 4), np.int32)
        for i, launch in enumerate(launches):
            table[i, : S * 8] = launch.slots.reshape(-1)
            table[i, k_off // 4 : k_off // 4 + S * 16] = launch.k_pages.reshape(-1)
            table[i, v_off // 4 : v_off // 4 + S * 16] = launch.v_pages.reshape(-1)
        self._upload(self.tables, table)
        recipe = self.recipe[("slots", mode.name)]
        slots = np.array(
            [recipe.slot(n) for n in ("slots", "k_pages", "v_pages")], np.int32
        )
        out = []
        for i in range(len(launches)):
            base = self.tables + i * stride
            addresses = np.array([base, base + k_off, base + v_off], np.uint64)
            out.append((slots, addresses))
        return out

    def _step(
        self,
        kind: str,
        mode: str,
        layer: int = -1,
        variant: str = "",
        extra: dict[str, int] | None = None,
    ) -> tuple[Recipe, np.ndarray, np.ndarray]:
        """One launch of a plan: ``_launch``'s recipe and bindings."""
        recipe = self.recipe[(kind, mode)]
        b = self.binds[(kind, mode, layer, variant)]
        if not extra:
            return recipe, b.slots, b.addresses
        slots = np.array([recipe.slot(name) for name in extra], np.int32)
        addresses = np.array(list(extra.values()), np.uint64)
        return (
            recipe,
            np.concatenate([b.slots, slots]),
            np.concatenate([b.addresses, addresses]),
        )

    def _attention_steps(
        self, mode: _Mode, layer: int, variant: str, tables, media: bool = False
    ) -> list:
        """A full-attention layer's launches: pre, one slot launch per table, post.

        ``media`` binds the pre graph's rope table to the media buffer, so a media plan reads
        the mrope rows it uploaded and a text plan keeps the standard table untouched.
        """
        recipe = self.recipe[("slots", mode.name)]
        b = self.binds[("slots", mode.name, layer, variant)]
        extra = {"cs_tab": self.buf["cs_tab_media"]} if media else None
        steps = [self._step("pre", mode.name, layer, variant, extra=extra)]
        for slots, addresses in tables:
            steps.append(
                (
                    recipe,
                    np.concatenate([b.slots, slots]),
                    np.concatenate([b.addresses, addresses]),
                )
            )
        steps.append(self._step("post", mode.name, layer, variant))
        return steps

    def _pages(
        self, mode: _Mode, rows: Sequence[RowCache], mtp: bool, positions
    ) -> list:
        """Slot rows (one per sequence) of the target's or the MTP layer's pages; positions[i]
        is row i's first position."""
        g, out = self.geometry, []
        for i, row in enumerate(rows):
            blocks = row.mtp_blocks if mtp else row.kv_blocks
            k, v = g.kv_pages(blocks)
            real = mode.per_row if mode.name[0] != "p" else positions[1]
            out.append(SlotRow(i * mode.per_row, positions[0][i], real, k, v))
        return out

    # ---- the target pass
    def _target(
        self,
        mode: _Mode,
        rows: Sequence[RowCache],
        tokens: np.ndarray,
        real: int,
        params: np.ndarray,
        token_source: int | None = None,
        media: tuple[np.ndarray, int, int, np.ndarray] | None = None,
    ) -> None:
        """Launch the target model over ``mode``'s columns: ``tokens`` (columns,), or a
        callable returning them (called once the step's tables are queued), or with
        ``token_source`` the device buffer holding them (a previous step's sampled tokens); a
        prefill's ``real`` tokens or ``real`` rows of a decode / verify batch; ``params``
        (columns or 1, PARAMS) sampler inputs."""
        dev, prefill = self.device, mode.name[0] == "p"
        T = mode.columns
        dev.h2d_nb(self.buf["samp_params"], params)
        pos = np.zeros(T, np.int32)
        if prefill:
            # `pos` is the rope index, so it is the compressed position; `starts` is the
            # cache's, so it is not. A media chunk overwrites `pos` with arange(T) because
            # its rope table is per-token (see `_splice_media`).
            pos[:] = np.minimum(
                rows[0].rope_at + np.arange(T), self.max_positions - 1
            )
            starts = ([rows[0].pos0], real)
        else:
            q = mode.per_row
            for i, row in enumerate(rows):
                pos[i * q : (i + 1) * q] = row.rope_at + np.arange(q)
            starts = ([row.pos0 for row in rows], q)
        if (
            int(pos[: real if prefill else len(rows) * mode.per_row].max())
            >= self.max_positions
        ):
            raise ValueError(f"position exceeds {self.max_positions}")
        dev.h2d_nb(self.buf["pos"], pos)
        dev.h2d_nb(
            self.buf["meta"],
            np.array([rows[0].pos0 if prefill else 0, real, 0, 0], np.int32),
        )
        if prefill:
            dev.h2d_nb(self.buf["last"], np.array([real - 1], np.int32))
        self._write_state_tables(rows, prefill)
        slot_rows = self._pages(mode, rows, False, starts)
        self._upload(self.buf["kv_rows"], kv_rows(T, slot_rows))
        tables = self._slot_tables(mode, slot_rows)
        if token_source is None:
            tokens = tokens() if callable(tokens) else tokens
            dev.h2d_nb(self.buf["tok"], np.asarray(tokens).astype(np.int32))
        # a pass's launches depend only on its mode, slot launches and token source
        # media changes the embed graph, so it is part of the plan's identity: without it
        # the text plan would be reused for an image prompt, silently
        key = ("target", mode.name, len(tables), token_source, media is not None)
        plan = self._plans.get(key)
        if plan is None:
            extra = None if token_source is None else {"tok": token_source}
            kind = "embed_img" if media is not None else "embed"
            steps = [self._step(kind, mode.name, extra=extra)]
            for i in range(self.n_layers):
                if i in self.gdn:
                    steps.append(self._step("gdn", mode.name, i))
                else:
                    steps += self._attention_steps(mode, i, "", tables, media=media is not None)
            steps.append(self._step("head", mode.name))
            plan = self._plans[key] = LaunchPlan(self.device, steps)
        if media is not None:
            if not self.media:
                raise ValueError(
                    "this model was built without media support (media=False); "
                    "enable it on the preset rather than sending image rows"
                )
            self._splice_media(media, T)
        plan.run()

    def _splice_media(self, media, T: int) -> None:
        """Queue this chunk's image rows and its per-token rope rows.

        ``media`` is (image rows for this chunk (n, H), first column, n, rope rows (T, 128)).
        The rope is safe to express as a per-token table because `attn_pre_graph` gathers
        `cs_tab` by a per-token index: `pos = arange(T)` selects the row this token needs.
        If that gather is ever changed to a position index, mrope breaks here and a batched
        prefill collides.
        """
        img, first, count, rope, _delta = media  # the delta is the sequence's, not the plan's
        self._upload(
            self.buf["img"], image_block(img, count, T, self.dims.hidden)
        )
        self._upload(self.buf["img_span"], np.array([first, count], np.int32))
        # blocking, into the media plan's own buffer: ordered by construction (the host does not
        # proceed until the copy completes) and not at `_upload`'s size threshold, so no launch
        # can read a stale table and no other plan can clobber it
        self.device.h2d(self.buf["cs_tab_media"], np.ascontiguousarray(np.asarray(rope, np.float32)))
        self._upload(self.buf["pos"], np.arange(T, dtype=np.int32))

    def _write_state_tables(self, rows: Sequence[RowCache], prefill: bool) -> None:
        """Every GDN layer's state rows for this step: [conv_src, ssm_src, conv_dst, ssm_dst]
        per row (-1 for a fresh source and for padding)."""
        n = len(self.gdn)
        table4 = self._state_fields(
            [r.state_src for r in rows], [r.state_dst for r in rows]
        )
        if not prefill:
            B = self.decode_buckets[-1]
            table = np.full((n, self.dec_stride // 4), -1, np.int32)
            table[:, : 4 * len(rows)] = table4.reshape(n, -1)
            off = _round(4 * B * 4) // 4
            table[:, off : off + len(rows)] = table4[:, :, 2]
            self._upload(self.dec_bases, table)
        else:
            table = np.zeros((n, 4, SMALL // 4), np.int32)
            table[:, :, 0] = table4[:, 0]
            self._upload(self.pre_bases, table)

    def _state_fields(self, srcs, dsts) -> np.ndarray:
        """(layers, rows, 4) [conv_src, ssm_src, conv_dst, ssm_dst] of STATE block rows."""
        g, n = self.geometry, len(self.gdn)
        layer = np.arange(n, dtype=np.int64)[:, None] * g.state_layer_rows
        fields = []
        for bases in (srcs, dsts):
            base = np.asarray(bases, np.int64)[None, :]
            ssm = np.where(base >= 0, base + layer, -1)
            fields += [np.where(base >= 0, ssm + g.ssm_rows, -1), ssm]
        return np.stack(fields, axis=-1).astype(np.int32)

    # ---- the MTP layer pass
    def _mtp_pass(
        self,
        mode: _Mode,
        variant: str,
        rows: Sequence[RowCache],
        starts: Sequence[int],
        real: int,
        cols: np.ndarray,
        tokens: np.ndarray | int,
        prev: int,
        params: np.ndarray | None,
    ) -> None:
        """The MTP layer over ``mode``'s columns: column c reads token ``tokens`` [c] (or the
        device buffer ``tokens``), hidden column ``cols`` [c] of [prev | x] and sits at its
        row's position ``starts`` [row] + its index in the row; it writes MTP KV for the first
        ``real`` columns of each row (prefill: of the chunk). With ``params``, sample drafts."""
        dev, T = self.device, mode.columns
        dev.h2d_nb(self.mtp_buf["mtp_cols"], np.asarray(cols, np.int32))
        extra = {"mtp_prev": prev}
        if isinstance(tokens, np.ndarray):
            dev.h2d_nb(self.mtp_buf["mtp_tok"], tokens.astype(np.int32))
            extra["mtp_tok"] = self.mtp_buf["mtp_tok"]
        else:
            extra["mtp_tok"] = tokens
        pos = np.zeros(T, np.int32)
        q = mode.per_row
        for i, start in enumerate(starts):
            # the MTP layer rotates at the compressed position too; its KV rows are at
            # `starts`, which stay the token positions
            shift = rows[i].pos_delta
            pos[i * q : i * q + (real if mode.name[0] == "p" else q)] = (
                start + shift + np.arange(real if mode.name[0] == "p" else q)
            )
        dev.h2d_nb(self.buf["pos"], np.minimum(pos, self.max_positions - 1))
        slot_rows = self._pages(mode, rows, True, (list(starts), real))
        self._upload(self.buf["kv_rows"], kv_rows(T, slot_rows))
        tables = self._slot_tables(mode, slot_rows)
        key = ("mtp", mode.name, variant, len(tables))
        plan = self._plans.get(key)
        if (
            plan is None
        ):  # mtp_prev and mtp_tok (the first launch's last bindings) are patched
            first = self._step(
                "mtp_in", mode.name, self.mtp_layer, variant, dict.fromkeys(extra, 0)
            )
            steps = [first, *self._attention_steps(mode, self.mtp_layer, "mtp", tables)]
            plan = self._plans[key] = LaunchPlan(self.device, steps)
        n = int(plan.starts[1])
        plan.patch(n - 2, extra["mtp_prev"])
        plan.patch(n - 1, extra["mtp_tok"])
        plan.run()
        if params is not None:
            dev.h2d_nb(self.mtp_buf["mtp_params"], params)
            self._launch("head", mode.name, -1, "mtp")

    def _prev_hidden(self, block_row: int) -> int:
        """Device address of a STATE block's hidden slot (the zero buffer for none)."""
        row = self.geometry.hidden_row(block_row)
        return self.mtp_buf["mtp_zero"] if row < 0 else self.arena + row * ROW_BYTES

    def _chain(
        self,
        rows: Sequence[RowCache],
        anchor_pos: Sequence[int],
        first: np.ndarray,
        first_cols: np.ndarray,
        sampling: Sampling | None,
        base_steps: Sequence[int],
        stage: Staging | None = None,
        scatter: int | None = None,
    ) -> np.ndarray:
        """Drafts d_1 .. d_K of each row: d_1 = ``first``; link j (MTP at position
        anchor + j - 1, token d_j, hidden: link j - 1's MTP output, the first link's at
        ``first_cols`` of the last MTP pass) samples d_(j+1) for position anchor + j + 1, which
        is ``base_steps`` [row] + j + 1 positions past the row's first sampled position. With
        ``stage``, drafts 2 .. K are copied into ``stage.drafts`` [j] without waiting (and
        returned as zeros); with ``scatter`` (a (rows, K + 1) token buffer) also into its
        column j + 1."""
        K, B = self.mtp, len(rows)
        drafts = np.zeros((B, K), np.int64)
        drafts[:, 0] = first
        if K == 1:
            return drafts
        bucket = min(b for b in self.decode_buckets if b >= B)
        mode = self.modes[f"d{bucket}"]
        tokens = np.zeros(bucket, np.int64)
        tokens[:B] = first
        cols = np.zeros(bucket, np.int64)
        cols[:B] = first_cols
        for j in range(1, K):
            starts = [anchor_pos[i] + j - 1 for i in range(B)]
            params = np.zeros((bucket, PARAMS), np.float32)
            if sampling is not None:
                params[:B] = sampling.at(range(B), [int(s) + j + 1 for s in base_steps])
            self._mtp_pass(
                mode,
                "mtpc",
                rows,
                starts,
                1,
                cols,
                tokens if j == 1 else self.mtp_buf["mtp_tok_out"],
                self.mtp_buf["mtp_zero"],
                params,
            )
            if stage is None:
                drafts[:, j] = self.device.d2h(
                    self.mtp_buf["mtp_tok_out"], (bucket,), np.int32
                )[:B]
            else:
                self.device.d2h_async(
                    stage.drafts[j, :bucket], self.mtp_buf["mtp_tok_out"]
                )
            if scatter is not None:
                b = np.arange(B)
                self.device.d2d_multi(
                    self.mtp_buf["mtp_tok_out"] + 4 * b,
                    np.full(B, 4),
                    scatter + 4 * (b * (K + 1) + j + 1),
                )
            cols = np.arange(bucket) + 1  # the previous link's output columns
        return drafts

    # ---- steps
    def prefill(
        self,
        row: RowCache,
        ids: Sequence[int],
        sampling: Sampling | np.ndarray | None = None,
        scores: ScoreRequest | None = None,
        draft: bool = True,
        media: tuple[np.ndarray, int, int, np.ndarray] | None = None,
    ) -> Prefilled:
        """One sequence's tokens at row.pos0, in bucketed chunks: the next token, sampled with
        ``sampling`` (1 row) after the last chunk (None: greedy), the requested prompt scores
        and, with MTP and ``draft`` (the prompt ends here), the drafts after the token."""
        if isinstance(sampling, np.ndarray):
            sampling = Sampling(sampling)
        token, i = -1, 0
        chunks = prefill_chunks(len(ids), self.buckets)
        result = None
        if scores is not None:
            n = len(scores.positions)
            result = Scores([None] * n, [None] * n, [None] * n)
        drafts = None
        for n, (real, T) in enumerate(chunks):
            src = row.state_src if i == 0 else row.state_dst
            chunk = RowCache(
                row.pos0 + i, row.kv_blocks, src, row.state_dst, row.mtp_blocks
            )
            last = n == len(chunks) - 1
            mode = self.modes[f"p{T}"]
            tokens = np.zeros(T, np.int64)
            tokens[:real] = ids[i : i + real]
            params = np.zeros((1, PARAMS), np.float32)
            if last and sampling is not None:
                params[:] = sampling.params[:1]
            chunk_media = None
            if media is not None:
                # the image span is in prompt coordinates; the splice works in this chunk's
                chunk_media = slice_media(media, i, real, T)
            self._target(mode, [chunk], tokens, real, params, media=chunk_media)
            token = int(self.device.d2h(self.buf["tok_out"], (1,), np.int32)[0])
            if scores is not None:
                inside = np.flatnonzero(
                    (scores.positions >= i) & (scores.positions < i + real)
                )
                self._score(scores, inside, i, result)
            # An image prompt does not use MTP. The MTP layer over the prompt's image tokens
            # would need those tokens' mrope rows, and `_mtp_pass` binds the position-indexed
            # text table; the columns are also shifted by one from the target's, so the rows
            # are not simply this chunk's. Rather than bind the wrong table, the whole MTP
            # pass is skipped for a media prompt - no MTP KV, no drafts - and the runner
            # decodes such a sequence without speculation. (A decode step after an image is
            # fine either way: the text table is indexed by position, and the positions are
            # the compressed ones `RowCache.pos_delta` passes.)
            if self.mtp and media is None:
                drafts = self._prefill_mtp(
                    chunk, ids, i, real, tokens, last and draft, sampling
                )
            i += real
        return Prefilled(token, result, drafts)

    def _prefill_mtp(
        self, chunk, ids, i, real, tokens, last, sampling
    ) -> np.ndarray | None:
        """The MTP layer over a prefill chunk (its KV), then the chunk's hidden into the STATE
        block; after the prompt's last chunk, the first draft following its sampled token (the
        draft chain would write MTP KV past the prompt, where a prefill leases no blocks, so
        drafts 2 .. K start as zeros)."""
        pos0, T = chunk.pos0, len(tokens)
        mode = self.modes[f"p{T}"]
        # column c: position pos0 - 1 + c, token ids[c], hidden c of [prev | x] (prev: the
        # hidden at pos0 - 1); a sequence's first chunk starts one later (no position -1)
        shift = 0 if pos0 > 0 else 1
        n = real - shift
        if n > 0:
            mtp_tokens = np.zeros(T, np.int64)
            mtp_tokens[:n] = ids[i + shift : i + real]
            cols = np.zeros(T, np.int64)
            cols[:n] = np.arange(n) + shift
            self._mtp_pass(
                mode,
                "mtp",
                [chunk],
                [pos0 - 1 + shift],
                n,
                cols,
                mtp_tokens,
                self._prev_hidden(chunk.state_src),
                None,
            )
        # the chunk's last hidden, for the next chunk's first MTP column
        self.device.h2d_nb(
            self.buf["hidden_table"],
            np.array([real - 1, self.geometry.hidden_row(chunk.state_dst)], np.int32),
        )
        self._launch("hidden", "x")
        if not last:
            return None
        # the sampled token at pos0 + real: MTP at pos0 + real - 1 drafts pos0 + real + 1
        one = self.modes[f"d{self.decode_buckets[0]}"]
        params = np.zeros((one.columns, PARAMS), np.float32)
        if sampling is not None:
            params[:1] = sampling.at([0], [1])
        cols = np.zeros(one.columns, np.int64)
        cols[0] = real
        self._mtp_pass(
            one,
            "mtp",
            [chunk],
            [pos0 + real - 1],
            1,
            cols,
            self.buf["tok_out"],
            self.mtp_buf["mtp_zero"],
            params,
        )
        drafts = np.zeros(self.mtp, np.int64)
        drafts[0] = self.device.d2h(self.mtp_buf["mtp_tok_out"], (1,), np.int32)[0]
        return drafts

    def _score(self, request: ScoreRequest, which: np.ndarray, start: int, out: Scores):
        """Score request positions ``which`` (columns position - start of the last chunk), in
        launches of up to SCORE_ROWS positions and SCORE_PAIRS target logprobs."""
        dev = self.device
        groups, group, pairs = [], [], 0
        for k in which.tolist():
            n = len(request.targets[k])
            if group and (len(group) == SCORE_ROWS or pairs + n > SCORE_PAIRS):
                groups.append(group)
                group, pairs = [], 0
            if n > SCORE_PAIRS:
                raise ValueError(
                    f"more than {SCORE_PAIRS} prompt targets at one position"
                )
            group.append(k)
            pairs += n
        if group:
            groups.append(group)
        for group in groups:
            rows = np.zeros(SCORE_ROWS, np.int32)
            rows[: len(group)] = request.positions[group] - start
            table = np.full((SCORE_PAIRS, 2), -1, np.int32)
            spans, p = [], 0
            for r, k in enumerate(group):
                targets = np.asarray(request.targets[k], np.int32)
                table[p : p + len(targets)] = np.stack(
                    [np.full(len(targets), r, np.int32), targets], axis=1
                )
                spans.append((p, p + len(targets)))
                p += len(targets)
            top = int(request.tops[group].max())
            K = min(t for t in SCORE_TOPS if t >= top)
            dev.h2d_nb(self.buf["score_rows"], rows)
            dev.h2d_nb(self.buf["score_pairs"], table)
            self._launch("score", f"k{K}")
            lp = dev.d2h(self.buf["score_lp"], (SCORE_PAIRS,), np.float32)
            if K:
                lse = dev.d2h(self.buf["score_stats"], (SCORE_ROWS, 16), np.float32)[
                    :, 1
                ]
                tv = dev.d2h(self.buf["score_top_v"], (SCORE_ROWS, K), np.float32)
                ti = dev.d2h(self.buf["score_top_i"], (SCORE_ROWS, K), np.int32)
            for r, k in enumerate(group):
                a, b = spans[r]
                out.targets[k] = lp[a:b].copy()
                n = int(request.tops[k])
                out.top_ids[k] = ti[r, :n].copy() if n else np.zeros(0, np.int32)
                out.top_logprobs[k] = (
                    (tv[r, :n] - lse[r]).astype(np.float32)
                    if n
                    else np.zeros(0, np.float32)
                )

    def decode(
        self,
        rows: Sequence[RowCache],
        tokens: Sequence[int] | None,
        sampling: Sampling | np.ndarray | None = None,
        *,
        stage: Staging | None = None,
        scores: bool = False,
    ) -> np.ndarray | None:
        """One token of each row; the next tokens, sampled with ``sampling`` (None: greedy).
        ``tokens`` None reads them from the device: the previous decode step's samples, row
        for row. With ``stage``, launch only: the tokens (and with ``scores`` the logprobs)
        are copied into ``stage`` and its event recorded."""
        if isinstance(sampling, np.ndarray):
            sampling = Sampling(sampling)
        B = len(rows)
        bucket = min(b for b in self.decode_buckets if b >= B)
        mode = self.modes[f"d{bucket}"]
        ids = np.zeros(bucket, np.int64)
        if tokens is not None:
            ids[:B] = tokens
        params = np.zeros((bucket, PARAMS), np.float32)
        if sampling is not None:
            params[:B] = sampling.params[:B]
        source = None if tokens is not None else self.buf["tok_out"]
        self._target(mode, rows, ids, B, params, token_source=source)
        if stage is None:
            return self.device.d2h(self.buf["tok_out"], (B,), np.int32)
        dev = self.device
        dev.d2h_async(stage.tokens[:B], self.buf["tok_out"])
        if scores:
            dev.d2h_async(stage.lp[:B], self.buf["lp_out"])
            dev.d2h_async(stage.top_idx[:B], self.buf["top_idx"])
            dev.d2h_async(stage.top_lp[:B], self.buf["top_lp"])
        stage.record()
        return None

    def verify(
        self,
        rows: Sequence[RowCache],
        anchors: Sequence[int],
        drafts: np.ndarray,
        sampling: Sampling | None = None,
    ) -> Verified:
        """One speculative step of each row: verify [anchor, drafts] (anchor at row.pos0),
        accept, commit, and draft the next step."""
        return self.verify_finish(self.verify_launch(rows, anchors, drafts, sampling))

    def verify_launch(
        self,
        rows: Sequence[RowCache],
        anchors: Sequence[int] | Callable[[], np.ndarray],
        drafts: np.ndarray | Callable[[], np.ndarray],
        sampling: Sampling | None = None,
        *,
        stage: Staging | None = None,
        scores: bool = False,
        chained: bool = False,
    ) -> Launched:
        """Launch the verify pass of [anchor, drafts] for each row (anchor at row.pos0): the
        target over K + 1 columns per row, then the MTP layer at every column. ``anchors`` and
        ``drafts`` may be callables, read once the step's tables are queued. ``chained``: the
        same rows continue the last staged ``verify_finish``, whose continuation tokens are on
        the device; ``drafts`` is then a callable for their host copy, read at the finish.
        With ``stage``, the samples (and with ``scores`` their logprobs) are copied into it
        without waiting."""
        Tq, B = self.mtp + 1, len(rows)
        bucket = min(b for b in self.decode_buckets if b >= B)
        mode = self.modes[f"v{bucket}"]
        resolved: list[np.ndarray] = []

        def tokens() -> np.ndarray:
            out = np.zeros((bucket, Tq), np.int64)
            out[:B, 0] = anchors() if callable(anchors) else anchors
            resolved.append(drafts() if callable(drafts) else np.asarray(drafts))
            out[:B, 1:] = resolved[0]
            return out.reshape(-1)

        params = np.zeros((bucket, Tq, PARAMS), np.float32)
        if sampling is not None:  # column t samples position pos0 + t + 1
            params[:B] = sampling.grid(B, range(Tq))
        if chained:
            self._target(
                mode,
                rows,
                None,
                B,
                params.reshape(-1, PARAMS),
                token_source=self.mtp_buf["next_tok"],
            )
            resolved.append(drafts)
        else:
            self._target(mode, rows, tokens, B, params.reshape(-1, PARAMS))
        # the MTP layer at every verify column: token y_t, the target's hidden at the column
        cols = np.arange(bucket * Tq) + 1
        mtp_params = np.zeros((bucket, Tq, PARAMS), np.float32)
        if sampling is not None:  # column t drafts position pos0 + t + 2
            mtp_params[:B] = sampling.grid(B, range(1, Tq + 1))
        self._mtp_pass(
            mode,
            "mtp",
            rows,
            [row.pos0 for row in rows],
            Tq,
            cols,
            self.buf["tok_out"],
            self.mtp_buf["mtp_zero"],
            mtp_params.reshape(-1, PARAMS),
        )
        if stage is not None:
            n = bucket * Tq
            d2h = self.device.d2h_async
            d2h(stage.tokens[:n], self.buf["tok_out"])
            d2h(stage.mtp_tokens[:n], self.mtp_buf["mtp_tok_out"])
            if scores:
                d2h(stage.lp[:n], self.buf["lp_out"])
                d2h(stage.top_idx[:n], self.buf["top_idx"])
                d2h(stage.top_lp[:n], self.buf["top_lp"])
            stage.record()
        return Launched(rows, resolved[0], sampling, stage, mode)

    def verify_finish(self, launched: Launched) -> Verified:
        """Accept each row's drafts against the verify pass's samples, commit the accepted
        tokens and launch the next step's draft chain. With the launch's stage this waits only
        for the verify pass; drafts 2 .. K land in ``stage.drafts`` (event 1), and the next
        step's tokens on the device (for a ``chained`` launch)."""
        rows, drafts, sampling, stage, mode = (
            launched.rows,
            launched.drafts,
            launched.sampling,
            launched.stage,
            launched.mode,
        )
        Tq, B = self.mtp + 1, len(rows)
        bucket = mode.columns // Tq
        if callable(drafts):
            drafts = drafts()
        if stage is None:
            y = self.device.d2h(self.buf["tok_out"], (bucket, Tq), np.int32)[:B]
            q = self.device.d2h(self.mtp_buf["mtp_tok_out"], (bucket, Tq), np.int32)[:B]
        else:
            stage.wait()
            y = stage.tokens[: bucket * Tq].reshape(bucket, Tq)[:B].copy()
            q = stage.mtp_tokens[: bucket * Tq].reshape(bucket, Tq)[:B].copy()
        acceptance = self.speculation.accept(drafts.tolist(), y.tolist())
        accepted = np.asarray(acceptance.accepted_counts, np.int64)
        emitted = acceptance.token_rows
        self._commit(mode, rows, accepted)
        first = q[np.arange(B), accepted]
        first_cols = np.arange(B) * Tq + accepted + 1
        anchor_pos = [rows[b].pos0 + int(accepted[b]) + 1 for b in range(B)]
        scatter = None
        if stage is not None:
            scatter = self.mtp_buf["next_tok"]
            following = np.zeros((bucket, Tq), np.int32)
            following[:B, 0] = [row[-1] for row in emitted]
            following[:B, 1] = first
            self._upload(scatter, following)
        next_drafts = self._chain(
            rows, anchor_pos, first, first_cols, sampling, accepted, stage, scatter
        )
        if stage is not None:
            stage.record(1)
        columns = [b * Tq + np.arange(accepted[b] + 1) for b in range(B)]
        return Verified(emitted, accepted, next_drafts, columns)

    def _commit(
        self, mode: _Mode, rows: Sequence[RowCache], accepted: np.ndarray
    ) -> None:
        """Apply each row's first accepted + 1 tokens' GDN updates and store the hidden of its
        last one, into the STATE block of that position."""
        L, B = len(self.gdn), mode.columns // mode.per_row
        dsts = [
            row.state_at(row.pos0 + int(m))
            for row, m in zip(rows, accepted, strict=True)
        ]
        fields = self._state_fields(
            [row.state_src for row in rows], dsts
        )  # (L, rows, 4)
        bases = np.full((L, B, 4), -1, np.int32)
        bases[:, : len(rows)] = fields
        counts = np.zeros((L, B), np.int32)
        counts[:, : len(rows)] = accepted + 1
        self._upload(self.mtp_buf["commit_bases"], bases)
        self._upload(self.mtp_buf["commit_counts"], counts)
        table = np.full((B, 2), -1, np.int32)
        for b, (m, dst) in enumerate(zip(accepted, dsts, strict=True)):
            table[b] = (b * mode.per_row + int(m), self.geometry.hidden_row(dst))
        self._upload(self.buf["hidden_table"], table)
        self._launch("commit", mode.name)

    def scores(self, rows: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The last step's raw logprobs per column: the chosen token's (rows,), and the top
        list's ids and logprobs (rows, TOP_MAX), filled to each column's ``ntop``."""
        d2h = self.device.d2h
        return (
            d2h(self.buf["lp_out"], (rows,), np.float32),
            d2h(self.buf["top_idx"], (rows, TOP_MAX), np.int32),
            d2h(self.buf["top_lp"], (rows, TOP_MAX), np.float32),
        )

    def logits(self, rows: int = 1) -> np.ndarray:
        """(rows, vocab) logits of the last step (a prefill's last token, or each column)."""
        V = self.dims.vocab
        if rows == 1:
            return self.device.d2h(self.buf["logits"], (V,), np.float32)
        return self.device.d2h(self.buf["logits"], (rows, V), np.float32)

    def close(self) -> None:
        self.weights.close()


class SingleSequence:
    """One sequence over a range of a model's slabs (tests and tools): KV blocks are taken in
    order from the first slab up and STATE blocks from the last slab down as the sequence grows,
    one STATE block per 4096 positions, with the Engine's rule (read the block of pos0 - 1,
    write the block of the step's last position). With MTP, MTP KV blocks come after the KV
    blocks' slabs, and speculative steps carry their drafts."""

    def __init__(self, model: Qwen35Gaudi, slabs: range | None = None) -> None:
        self.model = model
        self.slabs = slabs or range(1, model.parent_capacity + 1)
        # with MTP: its KV blocks in a fixed slab below at most three STATE blocks
        self.mtp_slab = self.slabs[-4] if model.mtp else None
        self.reset()

    def reset(self) -> None:
        self.pos = 0
        self.pos_delta = 0
        self.state_blocks: dict[int, int] = {}
        self.drafts: np.ndarray | None = None

    def _state(self, logical: int) -> int:
        if logical not in self.state_blocks:
            slab = self.slabs[-1 - len(self.state_blocks)]
            self.state_blocks[logical] = self.model.geometry.state_block_row(slab, 0)
        return self.state_blocks[logical]

    def _row(self, n: int, lookahead: int = 0) -> RowCache:
        g, last = self.model.geometry, self.pos + n - 1
        reach = last + lookahead
        blocks = np.arange(reach // 128 + 1)
        src = -1 if self.pos == 0 else self._state((self.pos - 1) // STATE_TOKENS)
        dst = self._state(last // STATE_TOKENS)
        first = (self.pos - 1) // STATE_TOKENS if self.pos else 0
        states = tuple(self._state(k) for k in range(first, reach // STATE_TOKENS + 1))
        per_slab = g.kv_blocks_per_slab
        kv_slabs = self.slabs[0] + blocks // per_slab
        mtp_rows = None
        top = self.slabs[-1] - len(self.state_blocks)
        if self.mtp_slab is not None:
            mtp_rows = g.mtp_block_row(np.full(len(blocks), self.mtp_slab), blocks)
            top = min(top, self.mtp_slab - 1)
            if len(self.state_blocks) > 3:
                raise MemoryError("sequence exceeds its slabs")
        if kv_slabs.max() > top:
            raise MemoryError("sequence exceeds its slabs")
        rows = g.kv_block_row(kv_slabs, blocks % per_slab)
        return RowCache(self.pos, rows, src, dst, mtp_rows, first, states, self.pos_delta)

    def prefill(self, ids: Sequence[int], sampling=None, media=None) -> int:
        done = self.model.prefill(self._row(len(ids)), ids, sampling, media=media)
        self.pos += len(ids)
        # the prompt's own positions come from its rope rows; every later step rotates at the
        # token position plus this, which is what makes a decode after an image correct
        self.pos_delta = 0 if media is None else int(media[4])
        self.drafts = done.drafts
        return done.token

    def decode(self, token: int) -> int:
        out = int(self.model.decode([self.next_row()], [token])[0])
        self.advance()
        return out

    def verify(self, token: int, sampling: Sampling | None = None) -> list[int]:
        """One speculative step from ``token`` (the last emitted): the emitted tokens."""
        K = self.model.mtp
        drafts = self.drafts if self.drafts is not None else np.zeros(K, np.int64)
        row = self._row(1, lookahead=2 * K)
        done = self.model.verify([row], [token], drafts[None], sampling)
        self.pos += int(done.accepted[0]) + 1
        self.drafts = done.drafts[0]
        return done.tokens[0]

    def next_row(self) -> RowCache:
        """The cache of this sequence's next decode token (``advance`` after the step)."""
        return self._row(1)

    def advance(self, n: int = 1) -> None:
        self.pos += n

    def logits(self) -> np.ndarray:
        return self.model.logits()
