"""Unrolled tiled index chain (index_chain_graph) == one-pass full selection, bit-exact.

The chain unrolls B tiles of gather+dequant+GEMM+acc+merge into ONE recipe, the acc writing each tile straight
into the double-buffered merge buffer (no host copy), the running top-512 carried device-resident. This gate
compares its ix_list/ix_lslots against a full (B*2048, T) score buffer + index_select_graph (the same oracle the
b7f405c tiled-merge gate uses), on the SAME arena/queries/weights. RED control: perturbing one query must change
the selection. Covers the in-place merge-buffer write (section_rmw) and the cross-tile running set.
"""
from __future__ import annotations
import os, struct
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.path.exists("/usr/include/habanalabs/synapse_api.h"), reason="needs SDK")


@pytest.fixture(scope="module")
def device():
    from resin_models_gaudi.synapse.runtime import SynapseError, open_device
    card = os.environ.get("RESIN_GAUDI_CARD")
    try:
        opened = open_device(None if card is None else int(card))
    except SynapseError as e:
        pytest.skip(f"no card: {e}")
    yield opened


@pytest.fixture(scope="module")
def recipes(device, tmp_path_factory):
    from resin_models_gaudi.synapse.graph import RecipeCache
    return RecipeCache(device, tmp_path_factory.mktemp("recipes"))


def _malloc_upload(device, a):
    addr = device.malloc(max(a.nbytes, 256))
    device.h2d(addr, np.ascontiguousarray(a))
    return addr


@pytest.mark.parametrize("B", [1, 2, 4, 8], ids=["B1", "B2", "B4", "B8"])
def test_unrolled_chain_matches_full(device, recipes, B):
    from resin_models_gaudi.deepseek_v41 import layers as L
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F8, F32, I32, U8, BF16

    d = L.Dims(hidden=256, heads=2, head_dim=512, q_lora=64, o_lora=64, groups=1, expert=64,
               experts=4, top=2, vocab=1024, eps=1e-20, hc_iters=2, hc_eps=1e-6, limit=10.0,
               route_scale=1.5, ranks=1)
    rng = np.random.default_rng(7)
    T, qb, S, C, run = 16, 1, 16, 2048, L.INDEX_RUN
    K = run + C
    KT = B * C                      # total scored keys
    ID, IH = L.INDEX_DIM, L.INDEX_HEADS
    W = IH * T
    PR, NB, PAIR, CMP = 1024, 16, 97, 0
    pages = KT // 128
    # per-column visible counts: everything, most, a middle amount, non-multiples of 64
    pattern = np.array([KT, 5000, 900, 700, 3000, 1500, KT - 1, 4096], np.int32)
    vis = np.minimum(np.resize(pattern, T), KT).astype(np.int32)
    pos = vis - 1
    blk = np.zeros((T, 2), np.int32)
    blk[:, 0] = pos
    blk[:, 1] = vis
    # arena: fp4 index keys, page p at lat0 = p*PAIR
    ixkeys = rng.standard_normal((KT, ID)).astype(np.float32)  # similar norms: query direction ranks the top-512
    rows = np.full((8, KT), -1, np.int32)
    rows[0] = (np.arange(KT) // 128) * PAIR
    rows[1] = np.arange(KT) % 128
    cmp_blocks = np.full((T, pages), -1, np.int32)
    for c in range(T):
        npg = -(-int(vis[c]) // 128)
        cmp_blocks[c, :npg] = np.arange(npg) * PAIR
    loff = np.zeros(64, np.int32)
    # queries: safe fp8 codes (avoid 0x78-0x7F / 0xF8-0xFF poison), scale 1; positive head weights
    qb8 = rng.integers(0, 0x78, (W, ID), dtype=np.uint8)
    neg = rng.integers(0, 2, (W, ID))
    qb8 = np.where(neg, qb8 | 0x80, qb8).astype(np.uint8)
    ix_q8 = np.ascontiguousarray(qb8)              # (W, ID) numpy = device (ID, W)
    ix_qs = np.ones(W, np.float32)
    ix_w = rng.uniform(0.2, 1.0, (T, IH)).astype(np.float32)

    # ---- store the fp4 keys into the arena
    gstore = Graph(device, "t_chain_store")
    arena_t = gstore.input("arena", (512, PR, NB), U8)
    sink_t = gstore.input("sink", (64, 1), I32)
    gstore.node("rg_ds_store_f4_f32",
                [arena_t, gstore.input("rows", (8, KT), I32), gstore.input("lat", (512, KT), F32),
                 gstore.input("ix", (ID, KT), F32)], [sink_t])
    arena = np.zeros((NB, PR, 512), np.uint8)
    store_addr = {n: _malloc_upload(device, a) for n, a in {
        "arena": arena, "sink": np.zeros((1, 64), np.int32), "rows": rows,
        "lat": np.zeros((KT, 512), np.float32), "ix": np.ascontiguousarray(ixkeys),
    }.items()}
    rs = recipes.compile(gstore)
    for n in rs.persistent:
        rs.bind(n, store_addr[n])
    rs.launch(); device.sync()
    arena = device.d2h(store_addr["arena"], (NB, PR, 512), np.uint8)
    for a in store_addr.values():
        device.free(a)
    gstore.destroy()

    # ---- gslot (chain and reference), the chain graph, the per-tile score graph, the full selection
    gslot_c = L.index_gslot_graph(device, d, T, S, 1, 1, T, qb, 1, B, C, run, 128)
    gslot_r = L.index_gslot_graph(device, d, T, S, 1, 1, T, qb, 1, B, C, run, 128)
    gchain = L.index_chain_graph(device, d, T, S, C, NB, run, B, query_block=qb, parts=1, cpages=pages)
    gscore = L.index_score_graph(device, d, T, S, C, NB, C, query_block=qb, parts=1, cpages=pages)
    gfull = L.index_select_graph(device, d, T, KT)

    common = {
        "arena": arena, "cmp_blocks": cmp_blocks, "cmp_loff": loff,
        "ix_q8": ix_q8, "ix_qs": ix_qs, "ix_w": ix_w, "ix_vis": vis,
    }
    addr = {n: _malloc_upload(device, a) for n, a in common.items()}
    addr["blk"] = _malloc_upload(device, blk)
    addr["dep"] = _malloc_upload(device, np.zeros(64, np.float32))
    addr["ix_sink.p0"] = _malloc_upload(device, np.zeros((1, 64), np.int32))
    addr["ix_sink"] = addr["ix_sink.p0"]
    snk = _malloc_upload(device, np.zeros((B, 64), np.int32))
    rec_c = _malloc_upload(device, np.zeros((B * S, 8), np.int32))
    gb_c = _malloc_upload(device, np.zeros((B, 64), np.int32))
    rec_r = _malloc_upload(device, np.zeros((B * S, 8), np.int32))
    gb_r = _malloc_upload(device, np.zeros((B, 64), np.int32))
    rinit = _malloc_upload(device, np.zeros((T, run), np.float32))
    pinit = _malloc_upload(device, np.zeros((T, run), np.int32))
    rout = _malloc_upload(device, np.zeros((T, run), np.float32))
    pout = _malloc_upload(device, np.zeros((T, run), np.int32))
    lst = _malloc_upload(device, np.zeros((T, run), np.int32))
    lsl = _malloc_upload(device, np.zeros((T, 8), np.int32))
    tile = _malloc_upload(device, np.zeros((T, C), np.float32))
    big = _malloc_upload(device, np.zeros((T, KT), np.float32))
    lst_r = _malloc_upload(device, np.zeros((T, run), np.int32))
    lsl_r = _malloc_upload(device, np.zeros((T, 8), np.int32))
    thr_r = _malloc_upload(device, np.zeros((T, 64), np.int32))

    def launch(g, bind):
        rec = recipes.compile(g)
        for n in rec.persistent:
            rec.bind(n, bind[n])
        rec.launch(); device.sync()
        return rec

    try:
        # gslot builds the records + gbase on device
        launch(gslot_c, {"blk": addr["blk"], "ix_rec": rec_c, "ix_gbase": gb_c})
        launch(gslot_r, {"blk": addr["blk"], "ix_rec": rec_r, "ix_gbase": gb_r})
        # ---- the unrolled chain: ONE launch
        bind = {**addr, "ix_rinit": rinit, "ix_pinit": pinit, "ix_rout": rout, "ix_pout": pout, "ix_tile": tile,
                "ix_list": lst, "ix_lslots": lsl}
        for i in range(B):
            bind[f"ix_s{i}.p0"] = rec_c + 4 * 8 * S * i
            bind[f"ix_gb{i}"] = gb_c + 256 * i
            bind[f"ix_snk{i}.0"] = snk + 256 * i
        launch(gchain, bind)
        got = device.d2h(lst, (T, run), np.int32)
        got_slots = device.d2h(lsl, (T, 8), np.int32)
        if os.environ.get("CHAIN_DBG"):
            tl = device.d2h(tile, (T, C), np.float32)
            rc = device.d2h(rec_c, (B * S, 8), np.int32)
            rr = device.d2h(rec_r, (B * S, 8), np.int32)
            print("DBG chain last-tile score[0,:6]", tl[0, :6], "nonzero", np.count_nonzero(tl))
            print("DBG rec_c==rec_r", np.array_equal(rc, rr), "rec_c[0]", rc[0].tolist())
            globals()["_dbg_tl"] = tl
        # ---- reference: per-tile score into a full (KT, T) buffer, then one-pass selection
        for i in range(B):
            launch(gscore, {**addr, "ix_slots": rec_r + 4 * 8 * S * i, "ix_score": tile})
            tl = device.d2h(tile, (T, C), np.float32)
            bg = device.d2h(big, (T, KT), np.float32)
            bg[:, i * C:(i + 1) * C] = tl
            device.h2d(big, np.ascontiguousarray(bg))
        launch(gfull, {"ix_score": big, "ix_vis": addr["ix_vis"], "ix_list": lst_r,
                       "ix_lslots": lsl_r, "ix_thr": thr_r})
        want = device.d2h(lst_r, (T, run), np.int32)
        want_slots = device.d2h(lsl_r, (T, 8), np.int32)
        # ---- host-copy tiled merge (the b7f405c / model mechanism) on the SAME scores
        gm = L.index_merge_graph(device, d, T, C, run, heads=d.heads)
        mg = [_malloc_upload(device, np.zeros((T, K), np.float32)) for _ in range(2)]
        pp = [_malloc_upload(device, np.zeros((T, run), np.int32)) for _ in range(2)]
        basebuf = _malloc_upload(device, np.zeros(1, np.int32))
        lst_h = _malloc_upload(device, np.zeros((T, run), np.int32))
        lsl_h = _malloc_upload(device, np.zeros((T, 8), np.int32))
        thr_h = _malloc_upload(device, np.zeros((T, 64), np.int32))
        bgfull = device.d2h(big, (T, KT), np.float32)
        for g in range(B):
            row0 = 0 if g == 0 else run
            marr = np.zeros((T, K), np.float32)
            # carry the running rows from the previous merge's sout buffer
            prev = device.d2h(mg[1 - g % 2], (T, K), np.float32)
            marr[:, :row0] = prev[:, :row0]
            marr[:, row0:row0 + C] = bgfull[:, g * C:(g + 1) * C]
            device.h2d(mg[g % 2], np.ascontiguousarray(marr))
            device.h2d(basebuf, np.array([g * C], np.int32))
            launch(gm, {"ix_merge": mg[g % 2], "ix_vis": addr["ix_vis"], "ix_pos": pp[g % 2],
                        "ix_gbase": basebuf, "ix_thr": thr_h, "ix_lslots": lsl_h, "ix_list": lst_h,
                        "ix_sout": mg[1 - g % 2], "ix_pos_out": pp[1 - g % 2]})
        host = device.d2h(lst_h, (T, run), np.int32)
        for a in (*mg, *pp, basebuf, lst_h, lsl_h, thr_h):
            device.free(a)
        gm.destroy()
        if os.environ.get("CHAIN_DBG"):
            dh = sum(len(set(got[c, :int(want_slots[c, 4])]) ^ set(host[c, :int(want_slots[c, 4])])) for c in range(T))
            print("DBG chain-vs-hostcopy total setdiff:", dh)
        if os.environ.get("CHAIN_DBG"):
            bg = device.d2h(big, (T, KT), np.float32)
            tl = globals().get("_dbg_tl")
            print("DBG last-tile chain==ref:", np.array_equal(tl, bg[:, (B - 1) * C:B * C]))
            c = 0
            n = int(want_slots[c, 4])
            a, b = got[c, :n].tolist(), want[c, :n].tolist()
            da, db = sorted(set(a) - set(b)), sorted(set(b) - set(a))
            print("DBG col0 only_chain", [(p, float(bg[c, p])) for p in da[:8]])
            print("DBG col0 only_ref  ", [(p, float(bg[c, p])) for p in db[:8]])
        for c in range(T):
            n = int(want_slots[c, 4])
            assert n == min(int(vis[c]), run), (c, n, int(vis[c]))
            assert int(got_slots[c, 4]) == n, (c, got_slots[c, 4], n)
            a, b = got[c, :n].tolist(), want[c, :n].tolist()
            assert a == b, (B, c, a[:12], b[:12], len(set(a) ^ set(b)))
            assert len(set(a)) == n, ("a key listed twice", c)
        print(f"\nB={B}: unrolled chain == full selection, {T} columns bit-exact")
        # RED control: same shape, different data — perturb queries AND head weights, recompute the reference,
        # and require the chain to track it bit-exactly (a chain that baked the old inputs would not).
        big_orig = device.d2h(big, (T, KT), np.float32)
        q8b = device.d2h(addr["ix_q8"], (W, ID), np.uint8)
        fresh = rng.integers(0, 0x78, (W, ID), dtype=np.uint8)
        q8b[:] = np.where(rng.integers(0, 2, (W, ID)), fresh | 0x80, fresh)
        device.h2d(addr["ix_q8"], np.ascontiguousarray(q8b))
        w2 = rng.uniform(0.2, 1.0, (T, IH)).astype(np.float32)
        device.h2d(addr["ix_w"], np.ascontiguousarray(w2))
        device.sync()
        launch(gchain, bind)
        got2 = device.d2h(lst, (T, run), np.int32)
        for i in range(B):
            launch(gscore, {**addr, "ix_slots": rec_r + 4 * 8 * S * i, "ix_score": tile})
            tl = device.d2h(tile, (T, C), np.float32)
            bg = device.d2h(big, (T, KT), np.float32)
            bg[:, i * C:(i + 1) * C] = tl
            device.h2d(big, np.ascontiguousarray(bg))
        launch(gfull, {"ix_score": big, "ix_vis": addr["ix_vis"], "ix_list": lst_r,
                       "ix_lslots": lsl_r, "ix_thr": thr_r})
        want2 = device.d2h(lst_r, (T, run), np.int32)
        big_new = device.d2h(big, (T, KT), np.float32)
        assert not np.array_equal(big_orig, big_new), "RED: perturbed inputs left the scores unchanged"
        for c in range(T):
            n = int(want_slots[c, 4])
            assert got2[c, :n].tolist() == want2[c, :n].tolist(), ("RED chain!=ref", B, c)
        print(f"B={B}: RED control OK (scores changed; chain tracks reference under perturbed inputs)")
        if os.environ.get("CHAIN_BREAKDOWN"):
            import time as _t
            for _stop in ("gemm", "acc", "merge"):
                _g = L.index_chain_graph(device, d, T, S, C, NB, run, B, query_block=qb, parts=1,
                                         cpages=pages, stop=_stop)
                _rec = recipes.compile(_g)
                for _n in _rec.persistent:
                    _rec.bind(_n, bind[_n])
                device.sync()
                _t0 = _t.perf_counter_ns()
                _rec.launch(); device.sync()
                _t1 = _t.perf_counter_ns()
                print(f"BREAKDOWN stop={_stop}: {(_t1 - _t0) / 1e3:.1f} us", flush=True)
                _g.destroy()
    finally:
        for a in set([rec_c, gb_c, rec_r, gb_r, rinit, pinit, rout, pout, snk, lst, lsl, tile, big, lst_r, lsl_r, thr_r,
                      *addr.values()]):
            device.free(a)
        for g in (gslot_c, gslot_r, gchain, gscore, gfull):
            g.destroy()


@pytest.mark.parametrize("G", [1, 2, 4], ids=["G1", "G2", "G4"])
def test_chain_carry_matches_full(device, recipes, G):
    """G-tile bucketed walk: two recipes of G tiles carry ONE device-resident running top-512 (recipe 0 fresh,
    recipe 1 reads recipe 0's ix_rout/ix_pout) == one-pass full selection over B=2G tiles. This is the Part C
    mechanism: ceil(B/G) launches, O(1) host metadata, no (kmax, T) array."""
    from resin_models_gaudi.deepseek_v41 import layers as L
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F8, F32, I32, U8

    d = L.Dims(hidden=256, heads=2, head_dim=512, q_lora=64, o_lora=64, groups=1, expert=64,
               experts=4, top=2, vocab=1024, eps=1e-20, hc_iters=2, hc_eps=1e-6, limit=10.0,
               route_scale=1.5, ranks=1)
    rng = np.random.default_rng(21)
    B = 2 * G
    T, qb, S, C, run = 16, 1, 16, 2048, L.INDEX_RUN
    KT = B * C
    ID, IH = L.INDEX_DIM, L.INDEX_HEADS
    W = IH * T
    PR, NB, PAIR = 1024, 16, 97
    pages = KT // 128
    pattern = np.array([KT, 5000, 900, 700, 3000, 1500, KT - 1, 4096], np.int32)
    vis = np.minimum(np.resize(pattern, T), KT).astype(np.int32)
    pos = vis - 1
    blk = np.zeros((T, 2), np.int32)
    blk[:, 0] = pos
    blk[:, 1] = vis
    ixkeys = rng.standard_normal((KT, ID)).astype(np.float32)
    rows = np.full((8, KT), -1, np.int32)
    rows[0] = (np.arange(KT) // 128) * PAIR
    rows[1] = np.arange(KT) % 128
    cmp_blocks = np.full((T, pages), -1, np.int32)
    for c in range(T):
        npg = -(-int(vis[c]) // 128)
        cmp_blocks[c, :npg] = np.arange(npg) * PAIR
    loff = np.zeros(64, np.int32)
    qb8 = rng.integers(0, 0x78, (W, ID), dtype=np.uint8)
    qb8 = np.where(rng.integers(0, 2, (W, ID)), qb8 | 0x80, qb8).astype(np.uint8)
    ix_w = rng.uniform(0.2, 1.0, (T, IH)).astype(np.float32)

    gstore = Graph(device, "t_carry_store")
    at = gstore.input("arena", (512, PR, NB), U8)
    gstore.node("rg_ds_store_f4_f32",
                [at, gstore.input("rows", (8, KT), I32), gstore.input("lat", (512, KT), F32),
                 gstore.input("ix", (ID, KT), F32)], [gstore.input("sink", (64, 1), I32)])
    sa = {n: _malloc_upload(device, a) for n, a in {
        "arena": np.zeros((NB, PR, 512), np.uint8), "sink": np.zeros((1, 64), np.int32), "rows": rows,
        "lat": np.zeros((KT, 512), np.float32), "ix": np.ascontiguousarray(ixkeys)}.items()}
    rs = recipes.compile(gstore)
    for n in rs.persistent:
        rs.bind(n, sa[n])
    rs.launch(); device.sync()
    arena = device.d2h(sa["arena"], (NB, PR, 512), np.uint8)
    for a in sa.values():
        device.free(a)
    gstore.destroy()

    gslot = L.index_gslot_graph(device, d, T, S, 1, 1, T, qb, 1, B, C, run, 128)
    g0 = L.index_chain_graph(device, d, T, S, C, NB, run, G, query_block=qb, cpages=pages, carry=False)
    g1 = L.index_chain_graph(device, d, T, S, C, NB, run, G, query_block=qb, cpages=pages, carry=True)
    gfull = L.index_select_graph(device, d, T, KT)
    common = {"arena": arena, "cmp_blocks": cmp_blocks, "cmp_loff": loff,
              "ix_q8": np.ascontiguousarray(qb8), "ix_qs": np.ones(W, np.float32), "ix_w": ix_w, "ix_vis": vis}
    addr = {n: _malloc_upload(device, a) for n, a in common.items()}
    addr["blk"] = _malloc_upload(device, blk)
    addr["dep"] = _malloc_upload(device, np.zeros(64, np.float32))
    rec = _malloc_upload(device, np.zeros((B * S, 8), np.int32))
    gb = _malloc_upload(device, np.zeros((B, 64), np.int32))
    addr["ix_sink"] = addr["ix_sink.p0"] = _malloc_upload(device, np.zeros((1, 64), np.int32))
    rbuf = [_malloc_upload(device, np.zeros((T, run), np.float32)) for _ in range(2)]
    pbuf = [_malloc_upload(device, np.zeros((T, run), np.int32)) for _ in range(2)]
    lst = _malloc_upload(device, np.zeros((T, run), np.int32))
    lsl = _malloc_upload(device, np.zeros((T, 8), np.int32))
    big = _malloc_upload(device, np.zeros((T, KT), np.float32))
    tile = _malloc_upload(device, np.zeros((T, C), np.float32))
    lst_r = _malloc_upload(device, np.zeros((T, run), np.int32))
    lsl_r = _malloc_upload(device, np.zeros((T, 8), np.int32))
    thr_r = _malloc_upload(device, np.zeros((T, 64), np.int32))
    gscore = L.index_score_graph(device, d, T, S, C, NB, C, query_block=qb, parts=1, cpages=pages)

    def launch(g, bind):
        rc = recipes.compile(g)
        for n in rc.persistent:
            rc.bind(n, bind[n])
        rc.launch(); device.sync()

    try:
        launch(gslot, {"blk": addr["blk"], "ix_rec": rec, "ix_gbase": gb})
        # recipe k processes tiles [k*G, k*G+G); it reads rbuf[k%2] and writes rbuf[(k+1)%2]
        for k, g in enumerate((g0, g1)):
            bind = {**addr, "ix_rinit": rbuf[k % 2], "ix_pinit": pbuf[k % 2],
                    "ix_rout": rbuf[(k + 1) % 2], "ix_pout": pbuf[(k + 1) % 2],
                    "ix_list": lst, "ix_lslots": lsl}
            for i in range(G):
                t = k * G + i
                bind[f"ix_s{i}.p0"] = rec + 4 * 8 * S * t
                bind[f"ix_gb{i}"] = gb + 256 * t
            launch(g, bind)
        got = device.d2h(lst, (T, run), np.int32)
        got_slots = device.d2h(lsl, (T, 8), np.int32)
        for i in range(B):
            launch(gscore, {**addr, "ix_slots": rec + 4 * 8 * S * i, "ix_score": tile})
            tl = device.d2h(tile, (T, C), np.float32)
            bg = device.d2h(big, (T, KT), np.float32)
            bg[:, i * C:(i + 1) * C] = tl
            device.h2d(big, np.ascontiguousarray(bg))
        launch(gfull, {"ix_score": big, "ix_vis": addr["ix_vis"], "ix_list": lst_r,
                       "ix_lslots": lsl_r, "ix_thr": thr_r})
        want = device.d2h(lst_r, (T, run), np.int32)
        want_slots = device.d2h(lsl_r, (T, 8), np.int32)
        for c in range(T):
            n = int(want_slots[c, 4])
            assert n == min(int(vis[c]), run), (c, n, int(vis[c]))
            assert int(got_slots[c, 4]) == n, (c, got_slots[c, 4], n)
            a, b = got[c, :n].tolist(), want[c, :n].tolist()
            assert a == b, (G, c, a[:12], b[:12], len(set(a) ^ set(b)))
        print(f"\ncarry G={G} B={B}: two-recipe device-resident carry == full selection, {T} cols bit-exact")
    finally:
        for a in set([rec, gb, *rbuf, *pbuf, lst, lsl, big, tile, lst_r, lsl_r, thr_r, *addr.values()]):
            device.free(a)
        for g in (gslot, g0, g1, gscore, gfull):
            g.destroy()


@pytest.mark.parametrize("T", [1, 4], ids=["T1", "T4"])
def test_chain_padded_stride_small_decode(device, recipes, T):
    """The model's small-decode wiring: blocks=T<8 query blocks, but the gslot record stride is rounded to 8
    (ix_rec_stride) so each tile's slot-table section base is 256-aligned. The chain reads only the first T slots
    of each stride-8 region (the canonical tile records); the gslot's tail columns are padding it never reads.
    == one-pass full selection."""
    from resin_models_gaudi.deepseek_v41 import layers as L
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F8, F32, I32, U8

    d = L.Dims(hidden=256, heads=2, head_dim=512, q_lora=64, o_lora=64, groups=1, expert=64,
               experts=4, top=2, vocab=1024, eps=1e-20, hc_iters=2, hc_eps=1e-6, limit=10.0,
               route_scale=1.5, ranks=1)
    rng = np.random.default_rng(41)
    B, stride = 4, 8
    qb, S, C, run = 1, T, 2048, L.INDEX_RUN
    KT = B * C
    ID, IH = L.INDEX_DIM, L.INDEX_HEADS
    W = IH * T
    PR, NB, PAIR = 1024, 16, 97
    pages = KT // 128
    vis = np.full(T, KT, np.int32)
    pos = vis - 1
    blk = np.zeros((T, 2), np.int32)
    blk[:, 0] = pos
    blk[:, 1] = vis
    ixkeys = rng.standard_normal((KT, ID)).astype(np.float32)
    rows = np.full((8, KT), -1, np.int32)
    rows[0] = (np.arange(KT) // 128) * PAIR
    rows[1] = np.arange(KT) % 128
    cmp_blocks = np.full((T, pages), -1, np.int32)
    for c in range(T):
        npg = -(-int(vis[c]) // 128)
        cmp_blocks[c, :npg] = np.arange(npg) * PAIR
    loff = np.zeros(64, np.int32)
    qb8 = rng.integers(0, 0x78, (W, ID), dtype=np.uint8)
    qb8 = np.where(rng.integers(0, 2, (W, ID)), qb8 | 0x80, qb8).astype(np.uint8)
    ix_w = rng.uniform(0.2, 1.0, (T, IH)).astype(np.float32)

    gstore = Graph(device, "t_pad_store")
    at = gstore.input("arena", (512, PR, NB), U8)
    gstore.node("rg_ds_store_f4_f32",
                [at, gstore.input("rows", (8, KT), I32), gstore.input("lat", (512, KT), F32),
                 gstore.input("ix", (ID, KT), F32)], [gstore.input("sink", (64, 1), I32)])
    sa = {n: _malloc_upload(device, a) for n, a in {
        "arena": np.zeros((NB, PR, 512), np.uint8), "sink": np.zeros((1, 64), np.int32), "rows": rows,
        "lat": np.zeros((KT, 512), np.float32), "ix": np.ascontiguousarray(ixkeys)}.items()}
    rs = recipes.compile(gstore)
    for n in rs.persistent:
        rs.bind(n, sa[n])
    rs.launch(); device.sync()
    arena = device.d2h(sa["arena"], (NB, PR, 512), np.uint8)
    for a in sa.values():
        device.free(a)
    gstore.destroy()

    # gslot stride 8 (nblk=T), chain slots=T: the model's ix_rec_stride wiring
    gslot = L.index_gslot_graph(device, d, T, stride, 1, 1, T, qb, 1, B, C, run, 128)
    gchain = L.index_chain_graph(device, d, T, S, C, NB, run, B, query_block=qb, cpages=pages)
    gscore = L.index_score_graph(device, d, T, S, C, NB, C, query_block=qb, parts=1, cpages=pages)
    gfull = L.index_select_graph(device, d, T, KT)
    common = {"arena": arena, "cmp_blocks": cmp_blocks, "cmp_loff": loff,
              "ix_q8": np.ascontiguousarray(qb8), "ix_qs": np.ones(W, np.float32), "ix_w": ix_w, "ix_vis": vis}
    addr = {n: _malloc_upload(device, a) for n, a in common.items()}
    addr["blk"] = _malloc_upload(device, blk)
    addr["dep"] = _malloc_upload(device, np.zeros(64, np.float32))
    addr["ix_sink"] = addr["ix_sink.p0"] = _malloc_upload(device, np.zeros((1, 64), np.int32))
    rec = _malloc_upload(device, np.zeros((B * stride, 8), np.int32))
    gb = _malloc_upload(device, np.zeros((B, 64), np.int32))
    rinit = _malloc_upload(device, np.zeros((T, run), np.float32))
    pinit = _malloc_upload(device, np.zeros((T, run), np.int32))
    rout = _malloc_upload(device, np.zeros((T, run), np.float32))
    pout = _malloc_upload(device, np.zeros((T, run), np.int32))
    lst = _malloc_upload(device, np.zeros((T, run), np.int32))
    lsl = _malloc_upload(device, np.zeros((T, 8), np.int32))
    tile = _malloc_upload(device, np.zeros((T, C), np.float32))
    big = _malloc_upload(device, np.zeros((T, KT), np.float32))
    lst_r = _malloc_upload(device, np.zeros((T, run), np.int32))
    lsl_r = _malloc_upload(device, np.zeros((T, 8), np.int32))
    thr_r = _malloc_upload(device, np.zeros((T, 64), np.int32))

    def launch(g, bind):
        rc = recipes.compile(g)
        for n in rc.persistent:
            rc.bind(n, bind[n])
        rc.launch(); device.sync()

    try:
        launch(gslot, {"blk": addr["blk"], "ix_rec": rec, "ix_gbase": gb})
        bind = {**addr, "ix_rinit": rinit, "ix_pinit": pinit, "ix_rout": rout, "ix_pout": pout,
                "ix_list": lst, "ix_lslots": lsl}
        for i in range(B):
            bind[f"ix_s{i}.p0"] = rec + 4 * 8 * stride * i
            bind[f"ix_gb{i}"] = gb + 256 * i
        launch(gchain, bind)
        got = device.d2h(lst, (T, run), np.int32)
        got_slots = device.d2h(lsl, (T, 8), np.int32)
        for i in range(B):
            launch(gscore, {**addr, "ix_slots": rec + 4 * 8 * stride * i, "ix_score": tile})
            tl = device.d2h(tile, (T, C), np.float32)
            bg = device.d2h(big, (T, KT), np.float32)
            bg[:, i * C:(i + 1) * C] = tl
            device.h2d(big, np.ascontiguousarray(bg))
        launch(gfull, {"ix_score": big, "ix_vis": addr["ix_vis"], "ix_list": lst_r,
                       "ix_lslots": lsl_r, "ix_thr": thr_r})
        want = device.d2h(lst_r, (T, run), np.int32)
        want_slots = device.d2h(lsl_r, (T, 8), np.int32)
        for c in range(T):
            n = int(want_slots[c, 4])
            assert int(got_slots[c, 4]) == n, (c, got_slots[c, 4], n)
            assert got[c, :n].tolist() == want[c, :n].tolist(), (T, c, got[c, :12], want[c, :12])
        print(f"\npadded-stride T={T} stride={stride}: chain == full selection, bit-exact")
    finally:
        for a in set([rec, gb, rinit, pinit, rout, pout, lst, lsl, tile, big, lst_r, lsl_r, thr_r,
                      *addr.values()]):
            device.free(a)
        for g in (gslot, gchain, gscore, gfull):
            g.destroy()


def test_chain_running_blocks_matches_full(device, recipes):
    """Part A: the bmax chain keeps a running top-2048 BLOCK list while streaming block-maxima tiles (no full
    (kb,T) cand_bmax published). Its cand_list/cand_slots == ixbl over the full block maxima, bit-exact."""
    from resin_models_gaudi.deepseek_v41 import layers as L
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F8, F32, I32, U8

    d = L.Dims(hidden=256, heads=2, head_dim=512, q_lora=64, o_lora=64, groups=1, expert=64,
               experts=4, top=2, vocab=1024, eps=1e-20, hc_iters=2, hc_eps=1e-6, limit=10.0,
               route_scale=1.5, ranks=1)
    rng = np.random.default_rng(51)
    B, top = 4, L.INDEX_BLOCKS
    T, qb, S, C, run = 16, 1, 16, 2048, L.INDEX_RUN
    KT = B * C
    kb = KT // 8
    ID, IH = L.INDEX_DIM, L.INDEX_HEADS
    W = IH * T
    PR, NB, PAIR = 1024, 16, 97
    pages = KT // 128
    pattern = np.array([KT, 5000, 900, 700, 3000, 1500, KT - 1, 4096], np.int32)
    vis = np.minimum(np.resize(pattern, T), KT).astype(np.int32)
    pos = vis - 1
    blk = np.zeros((T, 2), np.int32)
    blk[:, 0] = pos
    blk[:, 1] = vis
    visb = np.minimum(-(-vis // 8), kb).astype(np.int32)
    ixkeys = rng.standard_normal((KT, ID)).astype(np.float32)
    rows = np.full((8, KT), -1, np.int32)
    rows[0] = (np.arange(KT) // 128) * PAIR
    rows[1] = np.arange(KT) % 128
    cmp_blocks = np.full((T, pages), -1, np.int32)
    for c in range(T):
        npg = -(-int(vis[c]) // 128)
        cmp_blocks[c, :npg] = np.arange(npg) * PAIR
    loff = np.zeros(64, np.int32)
    qb8 = rng.integers(0, 0x78, (W, ID), dtype=np.uint8)
    qb8 = np.where(rng.integers(0, 2, (W, ID)), qb8 | 0x80, qb8).astype(np.uint8)
    ix_w = rng.uniform(0.2, 1.0, (T, IH)).astype(np.float32)

    gstore = Graph(device, "t_rb_store")
    at = gstore.input("arena", (512, PR, NB), U8)
    gstore.node("rg_ds_store_f4_f32",
                [at, gstore.input("rows", (8, KT), I32), gstore.input("lat", (512, KT), F32),
                 gstore.input("ix", (ID, KT), F32)], [gstore.input("sink", (64, 1), I32)])
    sa = {n: _malloc_upload(device, a) for n, a in {
        "arena": np.zeros((NB, PR, 512), np.uint8), "sink": np.zeros((1, 64), np.int32), "rows": rows,
        "lat": np.zeros((KT, 512), np.float32), "ix": np.ascontiguousarray(ixkeys)}.items()}
    rs = recipes.compile(gstore)
    for n in rs.persistent:
        rs.bind(n, sa[n])
    rs.launch(); device.sync()
    arena = device.d2h(sa["arena"], (NB, PR, 512), np.uint8)
    for a in sa.values():
        device.free(a)
    gstore.destroy()

    gslot = L.index_gslot_graph(device, d, T, S, 1, 1, T, qb, 1, B, C, run, 128)
    gchain = L.index_chain_graph(device, d, T, S, C, NB, run, B, query_block=qb, cpages=pages, kind="bmax")
    gscore = L.index_score_graph(device, d, T, S, C, NB, C, query_block=qb, parts=1, cpages=pages)
    gblk = L.index_blocks_graph(device, d, T, C)
    gsel = L.index_block_select_graph(device, d, T, kb, top, True)
    gfull = L.index_select_graph(device, d, T, KT)
    common = {"arena": arena, "cmp_blocks": cmp_blocks, "cmp_loff": loff,
              "ix_q8": np.ascontiguousarray(qb8), "ix_qs": np.ones(W, np.float32), "ix_w": ix_w, "ix_vis": vis}
    addr = {n: _malloc_upload(device, a) for n, a in common.items()}
    addr["blk"] = _malloc_upload(device, blk)
    addr["dep"] = _malloc_upload(device, np.zeros(64, np.float32))
    addr["ix_sink"] = addr["ix_sink.p0"] = _malloc_upload(device, np.zeros((1, 64), np.int32))
    rec = _malloc_upload(device, np.zeros((B * S, 8), np.int32))
    gb = _malloc_upload(device, np.zeros((B, 64), np.int32))
    bb = _malloc_upload(device, np.zeros((B, 64), np.int32))
    device.h2d(bb, np.ascontiguousarray(np.array([[i * (C // 8)] + [0] * 63 for i in range(B)], np.int32)))
    rinit = _malloc_upload(device, np.zeros((T, run), np.float32))
    pinit = _malloc_upload(device, np.zeros((T, run), np.int32))
    rout = _malloc_upload(device, np.zeros((T, run), np.float32))
    pout = _malloc_upload(device, np.zeros((T, run), np.int32))
    brinit = _malloc_upload(device, np.zeros((T, top), np.float32))
    bpinit = _malloc_upload(device, np.zeros((T, top), np.int32))
    brout = _malloc_upload(device, np.zeros((T, top), np.float32))
    bpout = _malloc_upload(device, np.zeros((T, top), np.int32))
    lst = _malloc_upload(device, np.zeros((T, run), np.int32))
    lsl = _malloc_upload(device, np.zeros((T, 8), np.int32))
    clist = _malloc_upload(device, np.zeros((T, top), np.int32))
    cslots = _malloc_upload(device, np.zeros((T, 8), np.int32))
    cthr = _malloc_upload(device, np.zeros((T, 64), np.int32))
    visb_a = _malloc_upload(device, visb)
    tile = _malloc_upload(device, np.zeros((T, C), np.float32))
    big = _malloc_upload(device, np.zeros((T, KT), np.float32))
    bmaxfull = _malloc_upload(device, np.zeros((T, kb), np.float32))
    lst_r = _malloc_upload(device, np.zeros((T, run), np.int32))
    lsl_r = _malloc_upload(device, np.zeros((T, 8), np.int32))
    thr_r = _malloc_upload(device, np.zeros((T, 64), np.int32))
    clist_r = _malloc_upload(device, np.zeros((T, top), np.int32))
    cslots_r = _malloc_upload(device, np.zeros((T, 8), np.int32))

    def launch(g, bind):
        rc = recipes.compile(g)
        for n in rc.persistent:
            rc.bind(n, bind[n])
        rc.launch(); device.sync()

    try:
        launch(gslot, {"blk": addr["blk"], "ix_rec": rec, "ix_gbase": gb})
        bind = {**addr, "ix_rinit": rinit, "ix_pinit": pinit, "ix_rout": rout, "ix_pout": pout,
                "ix_list": lst, "ix_lslots": lsl, "ix_visb": visb_a, "cand_list": clist,
                "cand_slots": cslots, "cand_thr": cthr, "ix_brinit": brinit, "ix_bpinit": bpinit,
                "ix_brout": brout, "ix_bpout": bpout}
        for i in range(B):
            bind[f"ix_s{i}.p0"] = rec + 4 * 8 * S * i
            bind[f"ix_gb{i}"] = gb + 256 * i
            bind[f"ix_gbb{i}"] = bb + 256 * i
        launch(gchain, bind)
        got_cl = device.d2h(clist, (T, top), np.int32)
        got_cs = device.d2h(cslots, (T, 8), np.int32)
        got_lst = device.d2h(lst, (T, run), np.int32)
        # reference: per-tile scores -> per-tile block maxima -> full cand_bmax -> ixbl
        for i in range(B):
            launch(gscore, {**addr, "ix_slots": rec + 4 * 8 * S * i, "ix_score": tile})
            launch(gblk, {"ix_tile": tile, "ix_vis": addr["ix_vis"], "ix_gbase": gb + 256 * i,
                          "ix_bmax": bmaxfull + 4 * T * (C // 8) * i})
            tl = device.d2h(tile, (T, C), np.float32)
            bg = device.d2h(big, (T, KT), np.float32)
            bg[:, i * C:(i + 1) * C] = tl
            device.h2d(big, np.ascontiguousarray(bg))
        launch(gsel, {"cand_bmax": bmaxfull, "ix_visb": visb_a, "cand_thr": thr_r,
                      "cand_list": clist_r, "cand_slots": cslots_r})
        want_cl = device.d2h(clist_r, (T, top), np.int32)
        want_cs = device.d2h(cslots_r, (T, 8), np.int32)
        launch(gfull, {"ix_score": big, "ix_vis": addr["ix_vis"], "ix_list": lst_r,
                       "ix_lslots": lsl_r, "ix_thr": thr_r})
        want_lst = device.d2h(lst_r, (T, run), np.int32)
        for c in range(T):
            n = int(want_cs[c, 4])
            assert n == min(int(visb[c]), top), (c, n, int(visb[c]))
            assert int(got_cs[c, 4]) == n, (c, got_cs[c, 4], n)
            assert got_cl[c, :n].tolist() == want_cl[c, :n].tolist(), (
                "blocks", c, "setdiff", sorted(set(got_cl[c, :n].tolist()) ^ set(want_cl[c, :n].tolist()))[:20],
                "order_only", sorted(got_cl[c, :n].tolist()) == sorted(want_cl[c, :n].tolist()))
            nk = int(want_cs[c, 4])  # key list check below uses full selection
        for c in range(T):
            n = min(int(vis[c]), run)
            assert got_lst[c, :n].tolist() == want_lst[c, :n].tolist(), ("keys", c)
        print(f"\nrunning-blocks B={B}: cand_list == full block selection AND ix_list == full, {T} cols bit-exact")
    finally:
        for a in set([rec, gb, bb, rinit, pinit, rout, pout, brinit, bpinit, brout, bpout, lst, lsl,
                      clist, cslots, cthr, visb_a, tile, big, bmaxfull, lst_r, lsl_r, thr_r, clist_r,
                      cslots_r, *addr.values()]):
            device.free(a)
        for g in (gslot, gchain, gscore, gblk, gsel, gfull):
            g.destroy()


def test_chain_bmax_mask_matches_full(device, recipes):
    """Candidate-source ("bmax") running top-2048 blocks feeds the consumer ("mask") chain: the mask chain reads
    cand_list/cand_slots (no full cand_bmax) and == one-pass selection over the scores masked to those blocks."""
    from resin_models_gaudi.deepseek_v41 import layers as L
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F8, F32, I32, U8

    d = L.Dims(hidden=256, heads=2, head_dim=512, q_lora=64, o_lora=64, groups=1, expert=64,
               experts=4, top=2, vocab=1024, eps=1e-20, hc_iters=2, hc_eps=1e-6, limit=10.0,
               route_scale=1.5, ranks=1)
    rng = np.random.default_rng(31)
    B, top = 16, L.INDEX_BLOCKS
    T, qb, S, C, run = 16, 1, 16, 2048, L.INDEX_RUN
    KT = B * C
    kb = KT // 8
    ID, IH = L.INDEX_DIM, L.INDEX_HEADS
    W = IH * T
    PR, NB, PAIR = 1024, 16, 97
    pages = KT // 128
    vis = np.full(T, KT, np.int32)
    pos = vis - 1
    blk = np.zeros((T, 2), np.int32)
    blk[:, 0] = pos
    blk[:, 1] = vis
    visb = np.full(T, kb, np.int32)
    ixkeys = rng.standard_normal((KT, ID)).astype(np.float32)
    rows = np.full((8, KT), -1, np.int32)
    rows[0] = (np.arange(KT) // 128) * PAIR
    rows[1] = np.arange(KT) % 128
    cmp_blocks = np.full((T, pages), -1, np.int32)
    for c in range(T):
        npg = -(-int(vis[c]) // 128)
        cmp_blocks[c, :npg] = np.arange(npg) * PAIR
    loff = np.zeros(64, np.int32)
    qb8 = rng.integers(0, 0x78, (W, ID), dtype=np.uint8)
    qb8 = np.where(rng.integers(0, 2, (W, ID)), qb8 | 0x80, qb8).astype(np.uint8)
    ix_w = rng.uniform(0.2, 1.0, (T, IH)).astype(np.float32)

    gstore = Graph(device, "t_bm_store")
    at = gstore.input("arena", (512, PR, NB), U8)
    gstore.node("rg_ds_store_f4_f32",
                [at, gstore.input("rows", (8, KT), I32), gstore.input("lat", (512, KT), F32),
                 gstore.input("ix", (ID, KT), F32)], [gstore.input("sink", (64, 1), I32)])
    sa = {n: _malloc_upload(device, a) for n, a in {
        "arena": np.zeros((NB, PR, 512), np.uint8), "sink": np.zeros((1, 64), np.int32), "rows": rows,
        "lat": np.zeros((KT, 512), np.float32), "ix": np.ascontiguousarray(ixkeys)}.items()}
    rs = recipes.compile(gstore)
    for n in rs.persistent:
        rs.bind(n, sa[n])
    rs.launch(); device.sync()
    arena = device.d2h(sa["arena"], (NB, PR, 512), np.uint8)
    for a in sa.values():
        device.free(a)
    gstore.destroy()

    gslot = L.index_gslot_graph(device, d, T, S, 1, 1, T, qb, 1, B, C, run, 128)
    gbmax = L.index_chain_graph(device, d, T, S, C, NB, run, B, query_block=qb, cpages=pages, kind="bmax")
    gmask = L.index_chain_graph(device, d, T, S, C, NB, run, B, query_block=qb, cpages=pages, kind="mask")
    gscore = L.index_score_graph(device, d, T, S, C, NB, C, query_block=qb, parts=1, cpages=pages)
    gfull = L.index_select_graph(device, d, T, KT)
    common = {"arena": arena, "cmp_blocks": cmp_blocks, "cmp_loff": loff,
              "ix_q8": np.ascontiguousarray(qb8), "ix_qs": np.ones(W, np.float32), "ix_w": ix_w, "ix_vis": vis}
    addr = {n: _malloc_upload(device, a) for n, a in common.items()}
    addr["blk"] = _malloc_upload(device, blk)
    addr["dep"] = _malloc_upload(device, np.zeros(64, np.float32))
    addr["ix_sink"] = addr["ix_sink.p0"] = _malloc_upload(device, np.zeros((1, 64), np.int32))
    rec = _malloc_upload(device, np.zeros((B * S, 8), np.int32))
    gb = _malloc_upload(device, np.zeros((B, 64), np.int32))
    bb = _malloc_upload(device, np.zeros((B, 64), np.int32))
    device.h2d(bb, np.ascontiguousarray(np.array([[i * (C // 8)] + [0] * 63 for i in range(B)], np.int32)))
    rinit = _malloc_upload(device, np.zeros((T, run), np.float32))
    pinit = _malloc_upload(device, np.zeros((T, run), np.int32))
    rout = _malloc_upload(device, np.zeros((T, run), np.float32))
    pout = _malloc_upload(device, np.zeros((T, run), np.int32))
    brinit = _malloc_upload(device, np.zeros((T, top), np.float32))
    bpinit = _malloc_upload(device, np.zeros((T, top), np.int32))
    brout = _malloc_upload(device, np.zeros((T, top), np.float32))
    bpout = _malloc_upload(device, np.zeros((T, top), np.int32))
    lst = _malloc_upload(device, np.zeros((T, run), np.int32))
    lsl = _malloc_upload(device, np.zeros((T, 8), np.int32))
    clist = _malloc_upload(device, np.zeros((T, top), np.int32))
    cslots = _malloc_upload(device, np.zeros((T, 8), np.int32))
    cthr = _malloc_upload(device, np.zeros((T, 64), np.int32))
    visb_a = _malloc_upload(device, visb)
    tile = _malloc_upload(device, np.zeros((T, C), np.float32))
    big = _malloc_upload(device, np.zeros((T, KT), np.float32))
    lst_r = _malloc_upload(device, np.zeros((T, run), np.int32))
    lsl_r = _malloc_upload(device, np.zeros((T, 8), np.int32))
    thr_r = _malloc_upload(device, np.zeros((T, 64), np.int32))

    def launch(g, bind):
        rc = recipes.compile(g)
        for n in rc.persistent:
            rc.bind(n, bind[n])
        rc.launch(); device.sync()

    try:
        launch(gslot, {"blk": addr["blk"], "ix_rec": rec, "ix_gbase": gb})
        # full scores + one-pass full selection (the oracle)
        for i in range(B):
            launch(gscore, {**addr, "ix_slots": rec + 4 * 8 * S * i, "ix_score": tile})
            tl = device.d2h(tile, (T, C), np.float32)
            bg = device.d2h(big, (T, KT), np.float32)
            bg[:, i * C:(i + 1) * C] = tl
            device.h2d(big, np.ascontiguousarray(bg))
        launch(gfull, {"ix_score": big, "ix_vis": addr["ix_vis"], "ix_list": lst_r,
                       "ix_lslots": lsl_r, "ix_thr": thr_r})
        want = device.d2h(lst_r, (T, run), np.int32)
        want_slots = device.d2h(lsl_r, (T, 8), np.int32)
        # bmax chain: running top-2048 blocks -> cand_list; key selection == full
        bbind = {**addr, "ix_rinit": rinit, "ix_pinit": pinit, "ix_rout": rout, "ix_pout": pout,
                 "ix_list": lst, "ix_lslots": lsl, "ix_visb": visb_a, "cand_list": clist,
                 "cand_slots": cslots, "cand_thr": cthr, "ix_brinit": brinit, "ix_bpinit": bpinit,
                 "ix_brout": brout, "ix_bpout": bpout}
        for i in range(B):
            bbind[f"ix_s{i}.p0"] = rec + 4 * 8 * S * i
            bbind[f"ix_gb{i}"] = gb + 256 * i
            bbind[f"ix_gbb{i}"] = bb + 256 * i
        launch(gbmax, bbind)
        got = device.d2h(lst, (T, run), np.int32)
        for c in range(T):
            n = int(want_slots[c, 4])
            assert got[c, :n].tolist() == want[c, :n].tolist(), ("bmax chain != full", c)
        # the running cand_list must == the natural-order top-2048 block selection over the full block maxima:
        # strict positives (max>0) in ascending block index, then zero ties (max==0) in ascending block index.
        # (Block maxima are mostly zero, so the tie cut at 2048 is order-defined; the running merge's natural row
        # order is the defined tie-break, distinct from the permuted ds_ix_bmax/ixbl order.)
        bg0 = device.d2h(big, (T, KT), np.float32)
        bm_np = bg0.reshape(T, KT // 8, 8).max(axis=2)
        bm_np[:, (KT - 1) // 8] = 3.0e38
        cl = device.d2h(clist, (T, top), np.int32)
        cs = device.d2h(cslots, (T, 8), np.int32)
        ref_np = np.zeros((T, top), np.int32)
        for c in range(T):
            posb = np.flatnonzero(bm_np[c] > 0).tolist()
            zerb = np.flatnonzero(bm_np[c] == 0).tolist()
            ref_np[c, :top] = (posb + zerb)[:top]
            n = int(cs[c, 4])
            assert n == min(int(visb[c]), top), (c, n, int(visb[c]))
            assert cl[c, :n].tolist() == ref_np[c, :n].tolist(), ("bmax cand_list != natural ref", c,
                "setdiff", sorted(set(cl[c, :n].tolist()) ^ set(ref_np[c, :n].tolist()))[:20])
        # mask chain: reads cand_list/cand_slots, masks the tile scores to those blocks, running merge
        mbind = {**addr, "ix_rinit": rinit, "ix_pinit": pinit, "ix_rout": rout, "ix_pout": pout,
                 "ix_list": lst, "ix_lslots": lsl, "cand_list": clist, "cand_slots": cslots}
        for i in range(B):
            mbind[f"ix_s{i}.p0"] = rec + 4 * 8 * S * i
            mbind[f"ix_gb{i}"] = gb + 256 * i
        launch(gmask, mbind)
        gotm = device.d2h(lst, (T, run), np.int32)
        gotm_slots = device.d2h(lsl, (T, 8), np.int32)
        # oracle: mask the full scores to the cand_list blocks (membership), then one-pass selection
        bg = device.d2h(big, (T, KT), np.float32)
        for c in range(T):
            n = int(cs[c, 4])
            keep = np.zeros(KT // 8, bool)
            keep[cl[c, :n]] = True
            bg[c, ~np.repeat(keep, 8)] = -3.0e38
        device.h2d(big, np.ascontiguousarray(bg))
        launch(gfull, {"ix_score": big, "ix_vis": addr["ix_vis"], "ix_list": lst_r,
                       "ix_lslots": lsl_r, "ix_thr": thr_r})
        wantm = device.d2h(lst_r, (T, run), np.int32)
        wantm_slots = device.d2h(lsl_r, (T, 8), np.int32)
        for c in range(T):
            n = int(wantm_slots[c, 4])
            assert int(gotm_slots[c, 4]) == n, (c, gotm_slots[c, 4], n)
            assert gotm[c, :n].tolist() == wantm[c, :n].tolist(), (
                "mask chain != masked full", c,
                "setdiff", sorted(set(gotm[c, :n].tolist()) ^ set(wantm[c, :n].tolist()))[:20],
                "order_only", sorted(gotm[c, :n].tolist()) == sorted(wantm[c, :n].tolist()))
        print(f"\nbmax+mask B={B}: source selection == full; consumer mask(cand_list) == masked full, bit-exact")
    finally:
        for a in set([rec, gb, bb, rinit, pinit, rout, pout, brinit, bpinit, brout, bpout, lst, lsl,
                      clist, cslots, cthr, visb_a, tile, big, lst_r, lsl_r, thr_r, *addr.values()]):
            device.free(a)
        for g in (gslot, gbmax, gmask, gscore, gfull):
            g.destroy()


@pytest.mark.parametrize("B", [2, 4], ids=["pB2", "pB4"])
def test_unrolled_chain_prefill(device, recipes, B):
    """Prefill query-block slots (qb=8, parts=2, ds_ix_qacc_o): the chain == one-pass full selection."""
    from resin_models_gaudi.deepseek_v41 import layers as L
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F8, F32, I32, U8

    d = L.Dims(hidden=256, heads=2, head_dim=512, q_lora=64, o_lora=64, groups=1, expert=64,
               experts=4, top=2, vocab=1024, eps=1e-20, hc_iters=2, hc_eps=1e-6, limit=10.0,
               route_scale=1.5, ranks=1)
    rng = np.random.default_rng(11)
    T, qb, C, run = 64, 8, 2048, L.INDEX_RUN
    nblk = T // qb
    S, P = nblk, 1
    Sp = S // P
    K = run + C
    KT = B * C
    ID, IH = L.INDEX_DIM, L.INDEX_HEADS
    W = IH * T
    PR, NB, PAIR = 1024, 16, 97
    pages = KT // 128
    pos = np.arange(T) + (KT - T - 50)
    vis = (pos + 1).astype(np.int32)
    blk = np.zeros((T, 2), np.int32)
    for b in range(nblk):
        blk[b, 0] = pos[b * qb]
        blk[b, 1] = vis[b * qb + qb - 1]
    ixkeys = rng.standard_normal((KT, ID)).astype(np.float32)  # similar norms: query direction ranks the top-512
    rows = np.full((8, KT), -1, np.int32)
    rows[0] = (np.arange(KT) // 128) * PAIR
    rows[1] = np.arange(KT) % 128
    cmp_blocks = np.full((T, pages), -1, np.int32)
    for c in range(T):
        npg = -(-int(vis[c]) // 128)
        cmp_blocks[c, :npg] = np.arange(npg) * PAIR
    loff = np.zeros(64, np.int32)
    qb8 = rng.integers(0, 0x78, (W, ID), dtype=np.uint8)
    qb8 = np.where(rng.integers(0, 2, (W, ID)), qb8 | 0x80, qb8).astype(np.uint8)
    ix_w = rng.uniform(0.2, 1.0, (T, IH)).astype(np.float32)

    gstore = Graph(device, "t_chainp_store")
    at = gstore.input("arena", (512, PR, NB), U8)
    gstore.node("rg_ds_store_f4_f32",
                [at, gstore.input("rows", (8, KT), I32), gstore.input("lat", (512, KT), F32),
                 gstore.input("ix", (ID, KT), F32)], [gstore.input("sink", (64, 1), I32)])
    sa = {n: _malloc_upload(device, a) for n, a in {
        "arena": np.zeros((NB, PR, 512), np.uint8), "sink": np.zeros((1, 64), np.int32), "rows": rows,
        "lat": np.zeros((KT, 512), np.float32), "ix": np.ascontiguousarray(ixkeys)}.items()}
    rs = recipes.compile(gstore)
    for n in rs.persistent:
        rs.bind(n, sa[n])
    rs.launch(); device.sync()
    arena = device.d2h(sa["arena"], (NB, PR, 512), np.uint8)
    for a in sa.values():
        device.free(a)
    gstore.destroy()

    gslot = L.index_gslot_graph(device, d, T, S, 1, 1, nblk, qb, 1, B, C, run, 128)
    gchain = L.index_chain_graph(device, d, T, S, C, NB, run, B, query_block=qb, parts=P, cpages=pages)
    gscore = L.index_score_graph(device, d, T, S, C, NB, C, query_block=qb, parts=P, cpages=pages)
    gfull = L.index_select_graph(device, d, T, KT)
    common = {"arena": arena, "cmp_blocks": cmp_blocks, "cmp_loff": loff,
              "ix_q8": np.ascontiguousarray(qb8), "ix_qs": np.ones(W, np.float32), "ix_w": ix_w, "ix_vis": vis}
    addr = {n: _malloc_upload(device, a) for n, a in common.items()}
    addr["blk"] = _malloc_upload(device, blk)
    addr["dep"] = _malloc_upload(device, np.zeros(64, np.float32))
    rec = _malloc_upload(device, np.zeros((B * S, 8), np.int32))
    gb = _malloc_upload(device, np.zeros((B, 64), np.int32))
    rinit = _malloc_upload(device, np.zeros((T, run), np.float32))
    pinit = _malloc_upload(device, np.zeros((T, run), np.int32))
    rout = _malloc_upload(device, np.zeros((T, run), np.float32))
    pout = _malloc_upload(device, np.zeros((T, run), np.int32))
    lst = _malloc_upload(device, np.zeros((T, run), np.int32))
    lsl = _malloc_upload(device, np.zeros((T, 8), np.int32))
    tile = _malloc_upload(device, np.zeros((T, C), np.float32))
    big = _malloc_upload(device, np.zeros((T, KT), np.float32))
    lst_r = _malloc_upload(device, np.zeros((T, run), np.int32))
    lsl_r = _malloc_upload(device, np.zeros((T, 8), np.int32))
    thr_r = _malloc_upload(device, np.zeros((T, 64), np.int32))
    snk = _malloc_upload(device, np.zeros((B * P, 64), np.int32))

    def launch(g, bind):
        rc = recipes.compile(g)
        for n in rc.persistent:
            rc.bind(n, bind[n])
        rc.launch(); device.sync()

    try:
        launch(gslot, {"blk": addr["blk"], "ix_rec": rec, "ix_gbase": gb})
        bind = {**addr, "ix_rinit": rinit, "ix_pinit": pinit, "ix_rout": rout, "ix_pout": pout, "ix_list": lst, "ix_lslots": lsl}
        for i in range(B):
            for p in range(P):
                bind[f"ix_s{i}.p{p}"] = rec + 4 * 8 * S * i + 4 * 8 * Sp * p
                bind[f"ix_sink.p{p}"] = snk + 256 * (i * P + p)
            bind[f"ix_gb{i}"] = gb + 256 * i
        launch(gchain, bind)
        got = device.d2h(lst, (T, run), np.int32)
        got_slots = device.d2h(lsl, (T, 8), np.int32)
        for i in range(B):
            sb = {**addr, "ix_score": tile, "ix_slots": rec + 4 * 8 * S * i,
                  "ix_sink": snk + 256 * i}
            launch(gscore, sb)
            tl = device.d2h(tile, (T, C), np.float32)
            bg = device.d2h(big, (T, KT), np.float32)
            bg[:, i * C:(i + 1) * C] = tl
            device.h2d(big, np.ascontiguousarray(bg))
        launch(gfull, {"ix_score": big, "ix_vis": addr["ix_vis"], "ix_list": lst_r,
                       "ix_lslots": lsl_r, "ix_thr": thr_r})
        want = device.d2h(lst_r, (T, run), np.int32)
        want_slots = device.d2h(lsl_r, (T, 8), np.int32)
        for c in range(T):
            n = int(want_slots[c, 4])
            assert n == min(int(vis[c]), run), (c, n, int(vis[c]))
            assert int(got_slots[c, 4]) == n, (c, got_slots[c, 4], n)
            a, b = got[c, :n].tolist(), want[c, :n].tolist()
            assert a == b, (B, c, a[:12], b[:12], len(set(a) ^ set(b)))
        print(f"\nprefill B={B} qb={qb} parts={P}: chain == full selection, {T} columns bit-exact")
    finally:
        for a in set([rec, gb, rinit, pinit, rout, pout, snk, lst, lsl, tile, big, lst_r, lsl_r, thr_r, *addr.values()]):
            device.free(a)
        for g in (gslot, gchain, gscore, gfull):
            g.destroy()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))