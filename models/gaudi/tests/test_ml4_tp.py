"""Mistral-Large-4 tensor parallel on every card (set ``RESIN_ML4_GGUF``), 4 layers. Prefill chunks of
the shapes the Engine's checkpoints cut, past the indexer's threshold: every launch runs on every rank,
logits are finite and equal across ranks, and nothing compiles after startup."""

from __future__ import annotations

import os
import shutil
import socket
from pathlib import Path

import numpy as np
import pytest

CHECKPOINT = os.environ.get("RESIN_ML4_GGUF")
RANKS = 8

pytestmark = pytest.mark.skipif(
    CHECKPOINT is None
    or not Path("/usr/include/habanalabs/synapse_api.h").exists()
    or shutil.which("tpc-clang") is None
    or len(list(Path("/sys/class/accel").glob("accel[0-9]*"))) < RANKS,
    reason=f"requires RESIN_ML4_GGUF, the Synapse SDK, tpc-clang and {RANKS} cards",
)

CHUNKS = (
    (1024, 1024, 9),
    (1024, 1024, 100),
    (1024, 1024, 128, 20),
    (1024, 1024, 256, 30),
    (1024, 1024, 700),
    (1024, 1024, 1024, 1024, 17),
)


def _chunks(columns: int) -> tuple[tuple[int, ...], ...]:
    """``CHUNKS``, then the widest launch: full, followed by one nearly full, and padded"""
    return (
        *CHUNKS,
        (columns, columns - 96, 9),
        (columns, columns // 2 + 100),
    )


def _worker(
    rank: int, port: int, cards: list[str], columns: int, out: str, fill: bool = False
) -> None:
    import torch
    import torch.distributed as dist

    from resin.models.mistral_large4.config import MistralLarge4Config
    from resin_models_gaudi.gguf.reader import GgufModel
    from resin_models_gaudi.mistral_large4 import model as M
    from resin_models_gaudi.mistral_large4 import parallel as par
    from resin_models_gaudi.mistral_large4.cache import CacheGeometry
    from resin_models_gaudi.synapse.runtime import open_device

    n = len(cards)
    par.init_gloo(rank, n, port)
    dev = open_device(par.accel_index(cards[rank]))
    par.hccl_rendezvous(dev, rank, n)
    gguf = GgufModel(CHECKPOINT)
    cfg = MistralLarge4Config.from_gguf(gguf.metadata).truncated(4)

    def fill_card(free: int) -> int:
        """Leave 64 arena blocks free on the card and return a capacity of 63."""
        block = CacheGeometry.from_config(cfg).block_bytes
        dev.malloc(free - 64 * block)
        return 63

    m = M.Ml4Gaudi(
        dev, gguf, cfg, capacity=fill_card if fill else 63, max_seq_len=8192, rank=rank, nranks=n,
        log=lambda _: None, prefill_columns=columns,
    )  # fmt: skip
    par.barrier()
    ids = np.random.default_rng(0).integers(1000, 100000, 8192)
    blocks = m.geo.block_row(np.arange(1, 1 + 64))
    totals = []
    for chunks in _chunks(columns):
        s = 0
        for c in chunks:
            T = m._launch(ids[s : s + c], np.arange(s, s + c), [M.Row(0, s, c, blocks)])
            logits = m.logits_host(T)[0 if T > M.MAX_T else c - 1]
            assert np.isfinite(logits).all(), (chunks, s)
            totals.append(float(logits.astype(np.float64).sum()))
            s += c
    mine = torch.tensor(totals, dtype=torch.float64)
    every = [torch.zeros_like(mine) for _ in range(n)]
    dist.all_gather(every, mine)
    assert all(torch.equal(mine, other) for other in every), "ranks' logits differ"
    assert m.compiles_after_start == 0
    if rank == 0:
        Path(out).write_text(str(len(totals)))
    dev.close()


@pytest.mark.parametrize(
    "columns, fill",
    [(1024, False), (2048, False), (2048, True), (4096, False), (4096, True)],
)
def test_prefill_chunks_across_ranks(tmp_path, columns, fill):
    """``fill``: the arena takes nearly all of the card's free memory."""
    from resin.process import spawn

    entries = sorted(Path("/sys/class/accel").glob("accel[0-9]*"))[:RANKS]
    cards = [os.path.basename(os.path.realpath(e / "device")) for e in entries]
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    out = tmp_path / "launches"
    spawn(
        _worker,
        args=(port, cards, columns, str(out), fill),
        devices=[f"pci:{c}" for c in cards],
    )
    assert int(out.read_text()) == sum(len(c) for c in _chunks(columns))
