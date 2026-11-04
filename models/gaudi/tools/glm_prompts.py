"""The GLM-5.3-Flash oracle's prompt set: token ids of a short chat prompt and a long one that crosses the DSA
indexer's 2048-key budget, written to OUT/prompts.json for glm_ref_hf.py and the model tests.

  python models/gaudi/tools/glm_prompts.py TOKENIZER.json TEXT.txt OUT
"""

import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

CHAT = "[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>{}<|assistant|>"
QUESTION = (
    "What is the capital of France? Answer in one short sentence, and then name one river "
    "that flows through that city."
)
LONG_TOKENS = 2300


def main(tokenizer: str, text: str, out: str) -> None:
    tok = Tokenizer.from_file(tokenizer)
    encode = lambda s: tok.encode(s, add_special_tokens=False).ids  # noqa: E731
    long_ids = encode(CHAT.format(Path(text).read_text()))[:LONG_TOKENS]
    prompts = {"short": encode(CHAT.format(QUESTION)), "long": long_ids}
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "prompts.json").write_text(json.dumps(prompts))
    print({k: len(v) for k, v in prompts.items()}, tok.decode(prompts["short"][:12]))


if __name__ == "__main__":
    main(*sys.argv[1:])
