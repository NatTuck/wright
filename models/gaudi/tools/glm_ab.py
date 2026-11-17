"""GLM-5.3-Flash scenarios through Resin's Engine, for A/B runs of two builds (e.g. the Python step path and the
native executor): every request's tokens, logprobs and top lists, and every runner call's kind and shape, as JSON.
The Engine loop, recorder and comparison are ``ds41_ab``'s.

usage (run with the venv whose resin you test; card jobs through the card lock script):

    .venv/bin/python glm_ab.py --layers 8 --of 4 --out A.json      # one card as rank 0 of 4
    .venv/bin/python glm_ab.py --ranks 4 --out B.json              # the checkpoint on 4 cards
    .venv/bin/python glm_ab.py --ranks 4 --mtp 3 --out C.json      # MTP drafts
    .venv/bin/python glm_ab.py --ranks 4 --drafter DIR --mtp 7 ... # DFlash 2 drafts
    python glm_ab.py --compare A.json B.json

``--layers N`` loads the first N layers only (fast builds; numerics of a shorter model, the same tables and
programs). ``--of N`` on one card: stand for rank ``--rank`` of N with a one-rank communicator (each all-reduce
leaves the rank's partial, each all-gather fills the rank's part). With ``--ranks N`` every rank runs the same
plans (Gloo groups and the socket step channel, as ``wright`` serves); rank 0 writes the JSON. Set
RESIN_GLM_DRAFT_POLICY=always or never for comparable runs with drafts (``auto`` depends on step times).
Two runs are comparable when their call shapes agree: bucket-dependent numerics make outputs of different shapes
differ.
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ds41_ab  # noqa: E402

CHECKPOINT = Path(os.environ.get("GLM_CHECKPOINT", "GLM-5.3-Flash"))
DRAFTER = Path("GLM-5.3-Flash-DFlash2")


def scenarios(text: list[int], draft_tokens: int):
    """(name, [(prompt, params kwargs)], together): requests submitted together or one after the other. Prompt
    lengths cross the prefill chunking (1024-token launches up to 3072 tokens, 4096 past it), the indexer (on past
    2051 positions) and the 4096-token STATE blocks (the prefix cache's unit)."""
    greedy = dict(temperature=0.0)
    scored = dict(temperature=0.0, logprobs=True, top_logprobs=3)
    sampled = dict(
        temperature=0.8, top_p=0.9, top_k=40, seed=7, logprobs=True, top_logprobs=2
    )
    out = [
        ("greedy alone", [(text[0:37], dict(greedy, max_output_tokens=24)),
                          (text[100:350], dict(greedy, max_output_tokens=24)),
                          (text[300:1400], dict(greedy, max_output_tokens=24)),
                          (text[400:2500], dict(greedy, max_output_tokens=24)),
                          (text[0:3073], dict(greedy, max_output_tokens=16)),
                          (text[500:4600], dict(greedy, max_output_tokens=16)),
                          (text[0:9000], dict(greedy, max_output_tokens=16))], False),
        ("three together", [(text[i * 50: i * 50 + 700], dict(scored, max_output_tokens=32)) for i in range(3)],
         True),
        ("eight together", [(text[i * 211: i * 211 + 400 + 37 * i], dict(greedy, max_output_tokens=20))
                            for i in range(8)], True),
        ("long rows together", [(text[i * 300: i * 300 + 2100 + 500 * i], dict(scored, max_output_tokens=20))
                                for i in range(5)], True),
        ("sampled", [(text[900:1900], dict(sampled, max_output_tokens=24)),
                     (text[2000:2100], dict(sampled, seed=11, max_output_tokens=24))], True),
        ("prefix hit", [(text[0:4889], dict(greedy, max_output_tokens=16)),
                        (text[0:4889], dict(greedy, max_output_tokens=8)),
                        (text[0:4500], dict(greedy, max_output_tokens=8))], False),
        ("long decode", [(text[3000:4000], dict(greedy, max_output_tokens=300))], False),
        ("wide", [(text[i * 37: i * 37 + 64], dict(greedy, max_output_tokens=12)) for i in range(40)], True),
        ("widest", [(text[i * 41: i * 41 + 48], dict(greedy, max_output_tokens=10)) for i in range(64)], True),
        ("prompt logprobs", [(text[0:300], dict(greedy, max_output_tokens=2, prompt_logprobs=True,
                                                 prompt_top_logprobs=3)),
                             (text[500:2600], dict(greedy, max_output_tokens=2, prompt_logprobs=True,
                                                   prompt_top_logprobs=0))], False),
        ("long context", [(text[0:16500], dict(greedy, max_output_tokens=16))], False),
        # prefill time alone: one request at a time, one token each (not in a default run: --only)
        ("prefill 2K", [(text[i * 97: i * 97 + 2048], dict(greedy, max_output_tokens=1)) for i in range(12)],
         False),
        ("prefill 8K", [(text[i * 97: i * 97 + 8192], dict(greedy, max_output_tokens=1)) for i in range(6)],
         False),
    ]  # fmt: skip
    return out


def one_rank_of(rank: int, nranks: int) -> None:
    """Make the runner's ``open_rank`` stand one card for rank ``rank`` of ``nranks`` (one-rank communicator)."""
    from ds41_fake import _Ranks

    import resin_models_gaudi.glm5_3_flash.runner as runner_module

    def open_rank(groups, card, model):
        from resin_models_gaudi.synapse.runtime import open_device

        device = open_device(card)
        device.ccl_init(device.ccl_unique_id(), 1, 0)
        return device, rank, nranks

    runner_module.ranks = _Ranks(runner_module.ranks, open_rank)
    # the partial and logit all-gathers' destinations: zero where a one-rank communicator writes nothing
    model_module = runner_module.Glm53Gaudi
    attach = model_module.attach

    def attach_zeroed(self, *args, **kwargs):
        attach(self, *args, **kwargs)
        d, dev = self.d, self.device
        gathered = 4 * d.hidden * d.gather_tokens * self.nranks
        for address, nbytes in (
            (self.pa_all, gathered),
            (self.pb_all, gathered),
            (self.logits_all, 4 * d.vocab * 64 * self.nranks),
        ):
            dev.memset32(address, 0, nbytes // 4)
        dev.sync()

    model_module.attach = attach_zeroed


def build(args, rank: int, groups, native_ranks=None):
    from resin_models_gaudi.glm5_3_flash.runner import Glm53FlashGaudiRunner

    if args.ranks == 1 and args.of > 1:
        one_rank_of(args.rank, args.of)
    cards = os.environ.get("VT_HPU_CARDS")
    card = int(cards.split(",")[rank]) if cards else None  # accel indices
    extra = {} if native_ranks is None else {"native_ranks": native_ranks}
    return Glm53FlashGaudiRunner(
        str(args.checkpoint), groups=groups, card=card, max_batch_size=64, max_seq_len=args.max_seq_len,
        parent_capacity=args.slabs, num_layers=args.layers, mtp=args.mtp, drafter=args.drafter, **extra,
    )  # fmt: skip


def _patch() -> None:
    """``ds41_ab``'s Engine loop over this model's runner and scenarios"""
    ds41_ab.build, ds41_ab.scenarios = build, scenarios
    ds41_ab.TIMED = ("prefill 2K", "prefill 8K")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument(
        "--layers", type=int, default=None, help="the first N layers only"
    )
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument(
        "--of",
        type=int,
        default=4,
        help="one card: the tensor-parallel ranks it stands in for",
    )
    parser.add_argument(
        "--native-ranks", action="store_true", help="the native rank group (no Gloo)"
    )
    parser.add_argument(
        "--rank", type=int, default=0, help="one card: stand for this rank of --of"
    )
    parser.add_argument("--slabs", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=20480)
    parser.add_argument(
        "--mtp",
        type=int,
        default=0,
        help="MTP drafts, or with --drafter the drafts a step verifies",
    )
    parser.add_argument(
        "--drafter", default=None, help=f"a DFlash 2 drafter directory (e.g. {DRAFTER})"
    )
    parser.add_argument(
        "--fixed",
        type=int,
        help="accept this many drafts of every row (benchmark mode)",
    )
    parser.add_argument("--only", nargs="*", help="run these scenarios only")
    parser.add_argument("--out")
    parser.add_argument("--compare", nargs=2)
    args = parser.parse_args()
    args.draft_tokens = args.mtp
    if args.compare:
        raise SystemExit(ds41_ab.compare(*args.compare))
    _patch()
    if args.ranks == 1:
        ds41_ab.run(0, args)
        return
    import tempfile

    from resin.process import spawn

    cards = os.environ["VT_HPU_CARDS"].split(",")[: args.ranks]
    bdf = [
        os.path.basename(os.path.realpath(f"/sys/class/accel/accel{i}/device"))
        for i in cards
    ]
    port = 20000 + secrets.randbelow(20000)
    path = os.path.join(tempfile.mkdtemp(prefix="glm-ab-"), "group")
    spawn(
        _worker,
        args=(args, port, path, secrets.token_bytes(16)),
        devices=[f"pci:{b}" for b in bdf],
    )


def _worker(rank: int, args, port: int, path: str, token: bytes) -> None:
    _patch()
    ds41_ab.run(rank, args, port, path, token)


if __name__ == "__main__":
    main()
