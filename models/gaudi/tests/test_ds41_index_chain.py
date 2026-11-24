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
        bind = {**addr, "ix_rinit": rinit, "ix_pinit": pinit, "ix_tile": tile,
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
    finally:
        for a in set([rec_c, gb_c, rec_r, gb_r, rinit, pinit, snk, lst, lsl, tile, big, lst_r, lsl_r, thr_r,
                      *addr.values()]):
            device.free(a)
        for g in (gslot_c, gslot_r, gchain, gscore, gfull):
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
        bind = {**addr, "ix_rinit": rinit, "ix_pinit": pinit, "ix_list": lst, "ix_lslots": lsl}
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
        for a in set([rec, gb, rinit, pinit, snk, lst, lsl, tile, big, lst_r, lsl_r, thr_r, *addr.values()]):
            device.free(a)
        for g in (gslot, gchain, gscore, gfull):
            g.destroy()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))