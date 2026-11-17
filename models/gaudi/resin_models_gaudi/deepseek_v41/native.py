"""DeepSeek-V4.1's native step executor (``resin._resin.Ds41Executor``), built at startup.

The model's step programs (``DeepSeekV41Gaudi._step_program``: the embedding, each layer, the DSpark injection, the
head and bounded replay's owner store) stay as they are. Here each step shape also gets an I/O program whose
sections the executor runs around them, with the executor's registers (``Ds41Executor.R_*``):

- ``upload``: the host copy ``R_IMAGE`` of the table image (``R_BYTES`` bytes) to the device's tables, then that
  copy's event (the copy is free again when it is done);
- ``tokens``: a continuation's tokens (``R_TOKBYTES`` bytes) from the previous step's samples;
- ``sample`` (decode shapes): the sampler's table (ring upload) and the sampler over every column;
  ``sample_last``: over one column (a prefill chunk's last; with ranks, the one all-gathered column);
- ``out``: the sampler's outputs to staging set ``R_STAGE``, then its event;
- ``compact {Tk}`` (prefill shapes, bounded replay): columns ``R_SKIP`` .. ``R_SKIP + R_KEPT`` of the cut layer's
  inputs to columns 0 .. of a step of ``Tk`` columns (``DeepSeekV41Gaudi._compact``).

The executor owns its pinned host copies of the table image, its staging sets and the sampler table; the
device's table buffer, sampler buffers and event slots are the model's.
"""

from __future__ import annotations

import numpy as np

from resin_models_gaudi.deepseek_v41.cache import WINDOW_GROUP
from resin_models_gaudi.deepseek_v41.model import (
    _SAMPLE_LAST,
    _SAMPLE_TABLE,
    _SAMPLED,
    _SAMPLED_BYTES,
    DECODE_BUCKETS,
    DSPARK_ROWS,
    PREFILL_BUCKETS,
    TMAX,
    Shape,
    bucket,
)
from resin_models_gaudi.glm5_3_flash.weights import NS
from resin_models_gaudi.synapse import runtime

SCORE_COLUMNS = 64  # prompt positions whose logits come back to the host at a time


def _round(nbytes: int, unit: int = 4096) -> int:
    return -(-nbytes // unit) * unit


def build_executor(runner):
    """The native executor of ``runner`` (a ``DeepSeekV41GaudiRunner`` with its cache attached): it runs every
    step."""
    from resin._resin import Ds41Executor, GaudiProgram

    m = runner.model
    E, dev = Ds41Executor, m.device
    stride = _round(max(tb.nbytes for tb in m.tables.values()))
    host = dev.host_buffer(2 * stride, np.uint8)
    sstride = _round(_SAMPLED_BYTES, 256)
    staging = dev.host_buffer(2 * sstride, np.uint8)
    sample = dev.host_buffer(_SAMPLE_TABLE, np.uint8)
    host_events, stage_events = m._host_event, m._stage_event
    # DSpark: each draft step's program and bucket, the pinned buffer its drafts come back to, and its event
    draft, draft_shapes = None, {}
    if m.stages and runner.speculative_tokens:
        for rows in range(1, DSPARK_ROWS + 1):
            ds = Shape(bucket(rows * m.block, False), False)
            draft_shapes[ds.T] = m._draft_natives.get(ds) or m._draft_native(
                ds, m._draft_program(ds)
            )
        drafts = dev.host_buffer(_round(4 * m.block * DSPARK_ROWS, 256), np.uint8)
        draft = {
            "programs": draft_shapes,
            "host": drafts,
            "event": dev.event_slots(1),
            "block": m.block,
            "noise": m.cfg.dspark_noise_token_id,
            "stages": m.stages,
        }
    # prompt logprobs: SCORE_COLUMNS scored columns' logits of every rank at a time
    V = m.d.vocab
    sc_stride = _round(4 * SCORE_COLUMNS * V)
    scores = dev.host_buffer(m.nranks * sc_stride, np.uint8)
    scoring = {"host": scores, "stride": sc_stride, "columns": SCORE_COLUMNS, "vocab": V,
               "ranks": m.nranks, "event": dev.event_slots(1)}  # fmt: skip
    shapes = {}
    for s in m.shapes:
        prog = m._program(s)
        step = m._step_program(s, prog)
        io = GaudiProgram(E.REGISTERS)
        io.section("upload")
        io.upload(m.table, (host.ctypes.data, E.R_IMAGE, stride), (0, E.R_BYTES, 1))
        io.select([(E.R_IMAGE, 2)], [0, 1])
        for k in range(2):
            io.arm()
            io.record(host_events + k)
        io.end()
        io.section("tokens")
        io.copy(
            m.table + m.tables[s].off["tok"], m.sampled["tok_out"], (0, E.R_TOKBYTES, 1)
        )
        if not s.prefill:
            io.section("sample")
            _sampler(io, m, sample, s, s.T)
        io.section("sample_last")
        # a prefill chunk's last column; with ranks, its head all-gathered that column alone (a one-column step)
        last = Shape(1, False) if s.prefill and m.nranks > 1 else s
        _sampler(io, m, sample, last, 1)
        if s.prefill:
            io.section("sample_at")  # column R of every column's logits (``last`` in the sampler's table)
            _sampler(io, m, sample, s, 1)
            io.section("scores")
            for r in range(m.nranks):
                io.download(
                    scores.ctypes.data + r * sc_stride,
                    (m.logits_all + 4 * V * r * s.T, E.R_SCORE_COL, 4 * V),
                    (0, E.R_SCORE_BYTES, 1),
                )
            io.record(scoring["event"])
        io.section("out")
        io.download(
            (staging.ctypes.data, E.R_STAGE, sstride), m.sampled_buf, _SAMPLED_BYTES
        )
        io.select([(E.R_STAGE, 2)], [0, 1])
        for k in range(2):
            io.arm()
            io.record(stage_events + k)
        io.end()
        if not s.prefill and s.T in draft_shapes:
            io.section("draft_out")  # the drafts (block, rows) of a draft step of this shape
            io.download(drafts.ctypes.data, m.io["dsp_out"], (0, E.R_TOKBYTES, 1))
            io.record(draft["event"])
        compact = {}
        if s.prefill and m.cut is not None:
            largest = bucket(m.replay, True)
            for tk in PREFILL_BUCKETS:
                if tk <= min(s.T, largest):
                    compact[tk] = name = f"compact {tk}"
                    io.section(name)
                    _compact(io, E, m, prog.cut, s.T, tk)
        io.finish()
        shapes[s.T, s.prefill] = {"step": step, "io": io, "compact": compact}
    g = runner.geometry
    layout = runner.cache_layout
    per = g.slab_rows // g.window_block_rows
    assert layout.groups[WINDOW_GROUP].blocks_per_slab == per
    return Ds41Executor(
        glue=dev.glue,
        fill=m._native,
        context=runner._context,
        layout=layout,
        shapes=shapes,
        image=(m.table, host, stride, host_events),
        staging=(
            staging,
            sstride,
            stage_events,
            (
                _SAMPLED["tok_out"],
                _SAMPLED["lp_out"],
                _SAMPLED["top_idx"],
                _SAMPLED["top_lp"],
                _SAMPLED_BYTES,
            ),
        ),  # fmt: skip
        sample=(sample, _SAMPLE_LAST),
        geometry=(g.slab_rows, g.window_block_rows, g.history_block_rows, per),
        model={
            "vocab": runner.vocab_size,
            "layers": m.n_layers,
            "cut": m.cut,
            "replay": runner.replay if m.cut is not None else 0,
            "prefill_buckets": list(PREFILL_BUCKETS),
            "decode_buckets": list(DECODE_BUCKETS),
            "tmax": TMAX,
            "one_column": m.nranks > 1,
            "max_batch_size": runner.max_batch_size,
            "max_seq_len": runner.max_seq_len,
            "draft_tokens": runner.speculative_tokens,
            "dspark_rows": DSPARK_ROWS,
            "engram_n": m.cfg.engram_max_ngram_size - 1,
            "spin_ns": int(runtime._SPIN_WAIT * 1e9),
        },
        draft=draft,
        scoring=scoring,
    )


def _sampler(io, m, sample, s: Shape, R: int) -> None:
    """The sampler's table (the executor's pinned copy, ring-uploaded) and the sampler of R columns of a step of
    shape ``s`` (``_sample_launch``)."""
    io.upload_ring(m.sample_table, sample.ctypes.data, _SAMPLE_TABLE)
    op = m._sample_ops.get((s, R))
    if op is None:
        op = m._sample_ops[s, R] = m._op(
            m.r["sample", s, R], None,
            {**m.sampled, "logits_all": m.logits_all, "samp_params": m.sample_table,
             "last": m.sample_table + _SAMPLE_LAST},
        )  # fmt: skip
    io.launch(op.recipe.handle, op.slots.tolist(), op.addresses.tolist())


def _compact(io, E, m, cut, Tf: int, Tk: int) -> None:
    """``_compact`` of a step of Tf columns into one of Tk: the streams X (NS H, T) and partial p (H, T) column by
    column, the coefficients c (24 rows of T) row by row, through the scratch buffer."""
    H = m.d.hidden
    X, c, p = cut
    sx, sp = m.replay_scratch, m.replay_scratch + 4 * NS * H * Tk
    sc = sp + 4 * H * Tk
    for src, dst, width in ((X, sx, NS * H), (p, sp, H)):
        io.copy(dst, (src, E.R_SKIP, 4 * width), (0, E.R_KEPT, 4 * width))
        io.copy(src, dst, (0, E.R_KEPT, 4 * width))
    for j in range(24):
        io.copy(sc + 4 * Tk * j, (c + 4 * Tf * j, E.R_SKIP, 4), (0, E.R_KEPT, 4))
    io.copy(c, sc, 4 * 24 * Tk)
