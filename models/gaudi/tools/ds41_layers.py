"""DeepSeek-V4.1 backbone layers on one card against a ds41_ref.py dump: a prompt's prefill as one chunk.

usage: ds41_layers.py CHECKPOINT REF.npz [--layers A-B] [--ranks N] [--save OUT.npz]

Layers A..B run in turn with every head and the whole expert width (one rank), each loaded, run and freed.
Their input is the reference's streams before layer A (the embedding for A = 0) with the previous layer's
pre-mix, both from the dump; after that they chain as the model does. Engram layers look their hashed rows up
on the host. Each layer's outputs are compared with the dump. The prompt must fit one window page (128 tokens),
and a range that starts after a KV owner needs that owner in the range.
Each rank takes the first free card that HABANA_VISIBLE_MODULES allows.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from resin.models.deepseek_v41 import DeepSeekV41Config
from resin_models_gaudi.deepseek_v41 import layers as L
from resin_models_gaudi.deepseek_v41.rope import cs_table, layer_frequencies
from resin_models_gaudi.deepseek_v41.weights import (
    RankSlices,
    engram_rows,
    prepare_compressed,
    prepare_engram,
    prepare_layer,
    reader,
)
from resin_models_gaudi.synapse.graph import RecipeCache
from resin_models_gaudi.synapse.runtime import open_device

PAGE_ROWS = 129  # a latent page: 128 token rows and their scales
INDEX_ROWS = 64  # an index page: 128 keys of 256 bytes, two per row
SCORES = 4096


class Bound:
    """Device copies of named arrays, and graph launches that bind them by name."""

    def __init__(self, device, recipes) -> None:
        self.device, self.recipes = device, recipes
        self.addr: dict[str, int] = {}
        self.size: dict[str, int] = {}

    def _alloc(self, name: str, nbytes: int) -> None:
        if name in self.addr and self.size[name] < nbytes:
            self.drop(name)
        if name not in self.addr:
            self.addr[name], self.size[name] = (
                self.device.malloc(max(nbytes, 256)),
                max(nbytes, 256),
            )

    def put(self, name: str, array: np.ndarray) -> None:
        a = np.ascontiguousarray(array)
        self._alloc(name, a.nbytes)
        self.device.h2d(self.addr[name], a)

    def drop(self, name: str) -> None:
        self.device.free(self.addr.pop(name))
        self.size.pop(name)

    def swap(self, a: str, b: str) -> None:
        self.addr[a], self.addr[b] = self.addr[b], self.addr[a]
        self.size[a], self.size[b] = self.size[b], self.size[a]

    def get(self, name: str, shape, dtype) -> np.ndarray:
        return self.device.d2h(self.addr[name], shape, dtype)

    def run(self, graph):
        if os.environ.get("DS41_TRACE"):
            print(f"  launch {graph.name}", flush=True)
        recipe = self.recipes.compile(graph)
        for name, t in recipe.persistent.items():
            if name not in self.addr:
                self._alloc(name, t.nbytes)
                self.device.h2d(self.addr[name], np.zeros(self.size[name], np.uint8))
            if self.size[name] < t.nbytes:
                raise ValueError(
                    f"{graph.name}: {name} needs {t.nbytes} bytes, the buffer has {self.size[name]}"
                )
            recipe.bind(name, self.addr[name])
        recipe.launch()
        self.device.sync()
        graph.destroy()
        return recipe


def rel(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64).reshape(np.shape(a))
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def parse(text: str) -> range:
    a, _, b = text.partition("-")
    return range(int(a), int(b or a) + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("checkpoint")
    parser.add_argument(
        "ref", help="a ds41_ref.py dump, or a file of token ids (no comparisons)"
    )
    parser.add_argument("--layers", default="0")
    parser.add_argument("--save", help="write each layer's streams to this .npz")
    parser.add_argument(
        "--ranks",
        type=int,
        default=1,
        help="tensor-parallel ranks, one card each",
    )
    parser.add_argument(
        "--time",
        default="",
        help="keep every layer's weights on the cards and time the forward at these token counts (e.g. 126,1): "
        "a warm pass, then a timed pass with a sync after each launch",
    )
    args = parser.parse_args()
    if args.ranks == 1:
        run(0, 1, args)
        return
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=run, args=(r, args.ranks, args)) for r in range(args.ranks)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
    if any(proc.exitcode for proc in procs):
        raise SystemExit("a rank failed")


def run(rank: int, nranks: int, args) -> None:
    import time
    from collections import defaultdict

    say = print if rank == 0 else (lambda *a, **k: None)
    cfg = DeepSeekV41Config.from_hugging_face(
        json.load(open(os.path.join(args.checkpoint, "config.json")))
    )
    if args.ref.endswith(".npz"):
        ref = np.load(args.ref)
        prompt = ref["prompt"]
    else:
        prompt = np.array([int(x) for x in open(args.ref).read().split()])
        ref = {"argmax": None}
    H = cfg.hidden_size
    if len(prompt) > 128:
        raise SystemExit("the prompt must fit one window page")
    layers = parse(args.layers)
    src = reader(args.checkpoint)
    sl = RankSlices.of(cfg, rank, nranks)
    device = open_device()
    say(f"rank {rank}: accel{device.card}", flush=True)
    if nranks > 1:
        from resin_models_gaudi.mistral_large4.parallel import (
            hccl_rendezvous,
            init_gloo,
        )

        init_gloo(rank, nranks, 29641)
        hccl_rendezvous(device, rank, nranks)
    bound = Bound(
        device,
        RecipeCache(device, os.path.expanduser("~/.cache/resin/ds41-layer-recipes")),
    )
    vocab = sl.vocab.stop - sl.vocab.start
    dims = L.Dims.of(cfg, nranks, vocab)
    nh = dims.heads
    timing = [int(t) for t in args.time.split(",") if t]

    # the arena: each layer's window page, then each KV owner's latent page and index page
    window = {layer: PAGE_ROWS * layer for layer in range(cfg.num_hidden_layers)}
    base = PAGE_ROWS * cfg.num_hidden_layers
    latent, index = {}, {}
    for owner in cfg.kv_source_layers:
        latent[owner], index[owner] = base, base + PAGE_ROWS
        base += PAGE_ROWS + INDEX_ROWS
    arena_rows = 1 << int(np.ceil(np.log2(base)))

    def weights_of(layer: int) -> dict[str, np.ndarray]:
        w = prepare_layer(src, cfg, layer, sl)
        if cfg.compress_ratios[layer]:
            w.update(prepare_compressed(src, cfg, layer))
        if layer in cfg.engram_layer_ids:
            w.update(prepare_engram(src, layer))
        return w

    resident: dict[int, list[str]] = {}
    if timing:  # every layer's weights on the cards under w{layer}.{name}
        for layer in layers:
            names = []
            for name, array in weights_of(layer).items():
                bound.put(f"w{layer}.{name}", array)
                names.append(name)
            resident[layer] = names
            say(f"layer {layer} resident", flush=True)
        if layers[-1] == cfg.num_hidden_layers - 1:
            from resin_models_gaudi.qwen3_5.checkpoint import kperm

            bound.put("head", kperm(src.read("head.weight", sl.vocab)))
            bound.put(
                "norm",
                (src.read("norm.weight").astype(np.uint32) << 16).view(np.float32),
            )
    hashes = None
    engram_cache: dict[tuple[int, int], np.ndarray] = {}

    def forward(ids: np.ndarray, clock=None) -> dict[str, np.ndarray]:
        """The prefill of ``ids`` from position 0 through the layers (and the head after 39)."""
        nonlocal hashes
        T = len(ids)
        pos = np.arange(T)

        repeat = 10

        def launch(kind: str, graph) -> None:
            """Bind and launch; when timing, launch the bound recipe ``repeat`` more times back to back (each writes
            the same values) and keep the average: the device's time without the host's binding."""
            recipe = bound.run(graph)
            if clock is not None:
                t0 = time.perf_counter()
                for _ in range(repeat):
                    recipe.launch()
                device.sync()
                clock[kind] += (time.perf_counter() - t0) / repeat

        def reduce(name: str) -> None:
            if nranks > 1:
                if (
                    clock is not None
                ):  # on a scratch buffer: repeated sums in place would grow the data
                    scratch = device.malloc(H * T * 4)
                    t0 = time.perf_counter()
                    for _ in range(repeat):
                        device.allreduce(scratch, H * T)
                    device.sync()
                    clock["allreduce"] += (time.perf_counter() - t0) / repeat
                    device.free(scratch)
                device.allreduce(bound.addr[name], H * T)
                device.sync()

        c = np.zeros(
            (24, T), np.float32
        )  # (T, 24) in graph order: one row per coefficient
        c[8:24] = np.eye(4, dtype=np.float32).reshape(-1, 1)
        first = layers[0]
        if first == 0:
            emb = src.read("embed.weight")[ids]
            X = np.repeat(
                ((emb.astype(np.uint32) << 16).view(np.float32))[:, None, :], 4, 1
            )
            c[0] = 1.0
        else:
            X = ref[f"x{first - 1}"].reshape(T, 4, H)
            c[:4] = ref[f"p{first - 1}"].T
        common = {
            "X": X.reshape(T, 4 * H), "c": c, "p": np.zeros((1, T, H), np.float32),
            "arena": np.zeros((arena_rows // 1024, 1024, 512), np.uint8), "ix_none": np.zeros((T, 128), np.uint16),
            "one": np.ones(1, np.float32), "dep": np.zeros(64, np.float32), "meta": np.array([0, T, 0, 0], np.int32),
            "rope_b1": np.ones(1, np.float32), "rope_s1": np.zeros(1, np.float32),
            f"rope_b{nh}": np.ones(nh, np.float32), f"rope_s{nh}": np.zeros(nh, np.float32),
            f"rope_b{L.INDEX_HEADS}": np.ones(L.INDEX_HEADS, np.float32),
            f"rope_s{L.INDEX_HEADS}": np.zeros(L.INDEX_HEADS, np.float32),
            "ix_score": np.zeros((T, SCORES), np.float32),
        }  # fmt: skip
        for name in ("X", "c", "p", "X_out", "c_out", "p_out", "X_eg"):
            if name in bound.addr:
                bound.drop(name)
        for name, array in common.items():
            bound.put(name, array)
        saved = {}
        for layer in layers:
            ratio = cfg.compress_ratios[layer]
            owner, source = (
                layer in cfg.kv_source_layers,
                layer in cfg.index_source_layers,
            )
            engram = layer in cfg.engram_layer_ids
            if timing:
                names = resident[layer]
                for name in names:
                    bound.addr[name], bound.size[name] = (
                        bound.addr[f"w{layer}.{name}"],
                        bound.size[f"w{layer}.{name}"],
                    )
            else:
                weights = weights_of(layer)
                names = list(weights)
                for name, array in weights.items():
                    bound.put(name, array)
            freqs = layer_frequencies(cfg, ratio)
            rows = np.full((T, 8), -1, np.int32)
            rows[:, 0], rows[:, 3], rows[:, 4] = (
                window[layer] + pos,
                window[layer] + 128,
                pos,
            )
            tables = {
                "rows": rows, "slots": np.array([[0, 0, 0, 0, T, 0, 0, L.WINDOW]], np.int32),
                "lat_pages": np.array([[window[layer]]], np.int32), "rope_pages": np.zeros((1, 1), np.int32),
                "cs": cs_table(pos, freqs), "cs_inv": cs_table(pos, freqs, True), f"rope_lp{T}": np.zeros(T, np.float32),
            }  # fmt: skip
            G = max(T // ratio, 1) if ratio else 0
            if ratio:
                kv_owner = cfg.kv_owner(layer)
                n = T // ratio
                groups = np.zeros((G, 4), np.int32)
                groups[:n, 0], groups[:n, 1] = (
                    np.arange(n) * ratio,
                    np.arange(n) * ratio + ratio - 1,
                )
                store = np.full((G, 8), -1, np.int32)
                store[:n, 0] = latent[kv_owner] + np.arange(n)
                store[:n, 2] = 2 * index[kv_owner] + np.arange(n)
                store[:n, 3], store[:n, 4] = latent[kv_owner] + 128, np.arange(n)
                loff = np.zeros(64, np.int32)
                loff[1] = -1
                open_last = np.where((pos == T - 1) & (T % ratio != 0), 0, -1).astype(
                    np.int32
                )
                tables.update({
                    "cmp_groups": groups, "cmp_rows": open_last, "cmp_store": store,
                    "cs_g": cs_table(np.arange(G) * ratio, freqs), f"rope_lp{G}": np.zeros(G, np.float32),
                    "cmp_state": np.zeros((1, 1024), np.float32),
                    "ix_slots": np.array([[0, 0, 0, 0, n, 0, ratio, 0]], np.int32),
                    "ix_pages": np.array([[index[kv_owner]]], np.int32),
                    "ix_vis": np.minimum((pos + 1) // ratio, n).astype(np.int32),
                    "cmp_blocks": np.full((1, T), latent[kv_owner], np.int32), "cmp_loff": loff,
                })  # fmt: skip
            for name, array in tables.items():
                bound.put(name, array)
            if engram:
                from tokenizers import Tokenizer

                from resin.models.deepseek_v41.engram import EngramHash

                key = (layer, T)
                if key not in engram_cache:
                    if hashes is None:
                        hashes = EngramHash.build(
                            cfg,
                            Tokenizer.from_file(
                                os.path.join(args.checkpoint, "tokenizer.json")
                            ),
                        )
                    hashed = hashes.ids(hashes.compress(ids), pos)[
                        :, cfg.engram_layer_ids.index(layer)
                    ]
                    engram_cache[key] = engram_rows(
                        src, layer, hashed.reshape(-1)
                    ).reshape(T, -1)
                bound.put("eg_emb", engram_cache[key])
                launch("engram", L.engram_graph(device, dims, T))
                bound.swap("X", "X_eg")
                if clock is None and f"l{layer}.engram_out" in getattr(
                    ref, "files", ()
                ):
                    err = rel(
                        bound.get("X", (T, 4 * H), np.float32),
                        ref[f"l{layer}.engram_out"],
                    )
                    say(f"layer {layer}: streams after Engram {err:.2e}", flush=True)
            if not ratio:
                launch(
                    "attention (window)",
                    L.attention_window_graph(
                        device, dims, T, 1, 128, arena_rows // 1024, residual=not engram
                    ),
                )
            else:
                launch("attention (compressed, first)", L.attention_pre_graph(
                    device, dims, T, 1, 128, arena_rows // 1024, ratio=ratio, owner=owner, source=source, groups=G,
                    residual=not engram))  # fmt: skip
                if source:
                    launch(
                        "indexer scores",
                        L.index_score_graph(
                            device, dims, T, 1, 128, arena_rows // 1024, SCORES
                        ),
                    )
                    launch(
                        "indexer top-512", L.index_select_graph(device, dims, T, SCORES)
                    )
                launch(
                    "attention (compressed, last)",
                    L.attention_out_graph(device, dims, T, arena_rows // 1024, 1),
                )
            if not engram:
                bound.swap("X", "X_out")
            bound.swap("c", "c_out")
            reduce("p_out")
            bound.swap("p", "p_out")
            if clock is None:
                attn = bound.get("p", (T, H), np.float32)
            launch("MoE", L.ffn_graph(device, dims, T))
            reduce("p_out")
            bound.swap("X", "X_out")
            bound.swap("c", "c_out")
            bound.swap("p", "p_out")
            if clock is None:
                ffn = bound.get("p", (T, H), np.float32)
                Xs = bound.get("X", (T, 4, H), np.float32)
                cc = bound.get("c", (24, T), np.float32)
                comb = cc[8:24].T.reshape(-1, 4, 4)
                streams = cc[4:8].T[:, :, None] * ffn[:, None, :] + np.einsum(
                    "tij,tih->tjh", comb, Xs
                )
                kinds = f"ratio {ratio}{', owner' if owner else ''}{', index' if source else ''}"
                parts = [f"layer {layer} ({kinds}):"]
                for label, got, key in (
                    ("attention", attn, f"l{layer}.attn_out"),
                    ("MoE", ffn, f"l{layer}.ffn_out"),
                ):
                    if key in getattr(ref, "files", ()):
                        parts.append(f"{label} {rel(got, ref[key]):.2e}")
                if f"x{layer}" in getattr(ref, "files", ()):
                    want = ref[f"x{layer}"]
                    parts.append(f"streams {rel(streams[-len(want) :], want):.2e}")
                say("  ".join(parts), flush=True)
                saved[f"x{layer}"] = streams
            if not timing:
                for name in names:
                    bound.drop(name)
        if layers[-1] == cfg.num_hidden_layers - 1:
            if not timing:
                from resin_models_gaudi.qwen3_5.checkpoint import kperm

                bound.put("head", kperm(src.read("head.weight", sl.vocab)))
                bound.put(
                    "norm",
                    (src.read("norm.weight").astype(np.uint32) << 16).view(np.float32),
                )
            if "logits" in bound.addr:
                bound.drop("logits")
            launch("head", L.head_graph(device, dims, T))
            if nranks > 1:
                t0 = time.perf_counter()
                every = device.malloc(nranks * T * vocab * 4)
                device.allgather(bound.addr["logits"], every, T * vocab)
                device.sync()
                if clock is not None:
                    clock["allgather (logits)"] += time.perf_counter() - t0
                parts = device.d2h(every, (nranks, T, vocab), np.float32)
                device.free(every)
                saved["logits"] = np.concatenate(list(parts), axis=1)[
                    :, : cfg.vocab_size
                ]
            else:
                saved["logits"] = bound.get("logits", (T, vocab), np.float32)
        return saved

    if not timing:
        saved = forward(prompt)
        if "logits" in saved:
            argmax = saved["logits"].argmax(-1)
            T = len(prompt)
            if ref["argmax"] is not None:
                want = ref["argmax"]
                last = ref["logits"][: min(T, int(ref["last"]))]  # the prompt's rows
                corr = [
                    float(np.corrcoef(saved["logits"][t], last[t])[0, 1])
                    for t in range(T)
                ]
                say(f"argmax equal at {int((argmax == want).sum())} of {T} prompt positions; "
                    f"logits correlation min {min(corr):.4f} mean {np.mean(corr):.4f}", flush=True)  # fmt: skip
            say("wright argmax", argmax.tolist())
        if args.save and rank == 0:
            np.savez(args.save, **saved)
        return
    for T in timing:
        ids = prompt[:T]
        forward(ids)  # warm: compiles every recipe
        clock: dict[str, float] = defaultdict(float)
        t0 = time.perf_counter()
        forward(ids, clock)
        total = time.perf_counter() - t0
        busy = sum(clock.values())
        say(f"--- {T} tokens, {len(layers)} layers, {nranks} ranks: device time per launch kind, each launch "
            f"repeated back to back (harness wall time {total:.1f} s)", flush=True)  # fmt: skip
        for kind, seconds in sorted(clock.items(), key=lambda kv: -kv[1]):
            say(
                f"  {kind:32s} {seconds * 1e3:9.2f} ms  ({seconds / busy:5.1%})",
                flush=True,
            )
        say(
            f"  {'device total':32s} {busy * 1e3:9.2f} ms: {T / busy:.1f} tok/s",
            flush=True,
        )


if __name__ == "__main__":
    main()
