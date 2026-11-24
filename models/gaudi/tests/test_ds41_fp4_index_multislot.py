"""Multi-slot indexer fp4 gate: ds_store_f4 -> ds_ix_gather_f4 (Sp live slots + padding) -> mx4_decs
(per-key normalized fp8 + sk).  The effective value fp8_value(dec)*sk must equal the checkpoint fp4
reference for every (slot, key); dead keys zero.  Covers the multi-slot gather/decode layout (Sp>1)
that the single-slot gate does not."""
from __future__ import annotations
import os, struct
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.path.exists("/usr/include/habanalabs/synapse_api.h"), reason="needs SDK")

from test_ds41_fp4_kv import fp4_ref, E2M1_SIGNED, e4m3_value
from test_ds41_kernels import fp8_value


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


def run(device, recipes, g, arrays, outs):
    addr = {}
    try:
        for n, a in arrays.items():
            addr[n] = device.malloc(max(a.nbytes, 256))
            device.h2d(addr[n], np.ascontiguousarray(a))
        r = recipes.compile(g)
        for n in r.persistent:
            r.bind(n, addr[n])
        r.launch(); device.sync()
        return {n: device.d2h(addr[n], arrays[n].shape, arrays[n].dtype) for n in outs}
    finally:
        for a in addr.values():
            device.free(a)


def test_multislot_index_gather_decode(device, recipes):
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F8, F32, I32, U8

    rng = np.random.default_rng(303)
    T = 16          # stored keys
    C = 2048        # keys per slot (composition shape)
    Sp = 4          # slots per part
    CMP = 300
    PR, NB = 1024, 1
    ix = (rng.standard_normal((T, 128)) * np.exp2(rng.integers(-4, 4, (T, 4)).repeat(32, 1))).astype(np.float32)

    g = Graph(device, "t_dbg_ix_ms")
    arena = g.input("arena", (512, PR, NB), U8)
    sink = g.input("sink", (64, 1), I32)
    rows = np.full((T, 8), -1, np.int32)
    rows[:, 0] = CMP
    rows[:, 1] = np.arange(T)
    g.node("rg_ds_store_f4_f32",
           [arena, g.input("rows", (8, T), I32), g.input("lat", (512, T), F32), g.input("ix", (128, T), F32)],
           [sink])
    codes = g.input("codes", (64, C, Sp), U8)
    scales = g.input("scales", (4, C, Sp), U8)
    sk = g.input("sk", (C, Sp), F32)
    ktab = g.tensor((C, Sp), I32)
    ids = g.tensor((C, Sp), I32)
    g.node("rg_ds_ix_gather_f4_f32",
           [arena, g.input("cmp_blocks", (16, 1), I32), g.input("slots", (8, Sp), I32), sink,
            g.input("loff", (64,), I32)],
           [codes, scales, sk, ktab, ids], struct.pack("i", 80))
    codes2 = g.tensor((64, C * Sp), U8)
    scales2 = g.tensor((4, C * Sp), U8)
    ktab2 = g.tensor((C * Sp,), I32)
    ids2 = g.tensor((C * Sp,), I32)
    dec2 = g.tensor((128, C * Sp), F8)
    g.node("reshape", [codes], [codes2])
    g.node("reshape", [scales], [scales2])
    g.node("reshape", [ktab], [ktab2])
    g.node("reshape", [ids], [ids2])
    g.node("rg_mx4_decs_f32", [codes2, scales2, ids2, ktab2], [dec2], struct.pack("ii", 0, 0))
    dec = g.input("dec", (128, C, Sp), F8)
    g.node("reshape", [dec2], [dec])

    # all Sp slots live, same key range (key0=0, klen=T, col=0, pbase=0, ratio=2)
    slots = np.zeros((Sp, 8), np.int32)
    slots[:, 0] = 0
    slots[:, 3] = 0
    slots[:, 4] = T
    slots[:, 5] = 0
    slots[:, 6] = 2
    slots[:, 7] = 0
    cmp_blocks = np.full((1, 16), -1, np.int32); cmp_blocks[0, 0] = CMP
    out = run(device, recipes, g, {
        "arena": np.zeros((NB, PR, 512), np.uint8), "sink": np.zeros((1, 64), np.int32),
        "rows": rows, "lat": np.zeros((T, 512), np.float32), "ix": np.ascontiguousarray(ix),
        "cmp_blocks": cmp_blocks, "slots": slots, "loff": np.zeros(64, np.int32),
        "codes": np.zeros((Sp, C, 64), np.uint8), "scales": np.zeros((Sp, C, 4), np.uint8),
        "sk": np.zeros((Sp, C), np.float32),
        "dec": np.zeros((Sp, C, 128), np.uint8)},
        ["dec", "codes", "scales", "sk"])
    dec = out["dec"]          # numpy (Sp, C, 128): dec[s, c, d] = device dec[d, c, s]
    skv = out["sk"]           # (Sp, C): sk[s, c] = 2^k
    want = fp4_ref(ix, 32, False)   # (T, 128)
    print("\n=== multi-slot index gather+decode (normalized fp8 * sk) ===")
    bad = 0
    for s in range(Sp):
        for c in range(T):
            got = fp8_value(dec[s, c]) * skv[s, c]
            if not np.array_equal(got, want[c]):
                bad += 1
                if bad <= 5:
                    print(f"  slot {s} key {c}: max_abs={np.abs(got-want[c]).max():.3e}")
    # dead keys (c>=T) must be zero
    deadnz = sum(int(np.count_nonzero(fp8_value(dec[s, c]) * skv[s, c])) for s in range(Sp) for c in range(T, C))
    print(f"  live-key mismatches: {bad}/{Sp*T}   dead-key nonzero: {deadnz}")
    assert bad == 0, f"{bad} live (slot,key) decoded wrong"
    assert deadnz == 0, "dead keys not zero"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))