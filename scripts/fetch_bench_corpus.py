"""Fetch the pinned texts for the real-text benchmark and cut them into passages.

uv run --no-default-groups --group gaudi scripts/fetch_bench_corpus.py --output bench/corpus

Writes english.jsonl and coding.jsonl, one {text, instruction, source} passage per line, for
scripts/bench.py --prompt-file. Each download is checked against its pinned hash. The text is
not stored in this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import tarfile
import urllib.request
from pathlib import Path

# Public domain in the United States. Passages are cut from inside each book's own text, without
# the Project Gutenberg header and footer.
ENGLISH = {
    "https://www.gutenberg.org/cache/epub/1342/pg1342.txt": "3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75",
    "https://www.gutenberg.org/cache/epub/2701/pg2701.txt": "b5e88b71a1cda068e9ab1d94344c9d932338ba6da1a66def97a1aecd9cfcc87d",
    "https://www.gutenberg.org/cache/epub/1400/pg1400.txt": "9a637118af8e953e9764ec603d9b0a032883384d465acac2e27966a80cf1c6f8",
    "https://www.gutenberg.org/cache/epub/98/pg98.txt": "d54c2b80d40a40b982cd88852c6180bb944d95acdb028af3d0e01a1750681784",
    "https://www.gutenberg.org/cache/epub/2600/pg2600.txt": "2d5bb2ad5f422765e714617e21fa31bbaf8958aa79682c86fca6660fcc5d1b2b",
}
# The Python Software Foundation License allows this use.
CODING = (
    "https://github.com/python/cpython/archive/refs/tags/v3.13.0.tar.gz",
    "d2c5dbd73111a80ee654489eb0f52b663b38ef9ce9e9323312515d941d0dfaa2",
)

ENGLISH_INSTRUCTION = "Continue this text at length, in the same style."
CODING_INSTRUCTION = "Continue this source code at length, in the same style. Write only code."
# About 4 characters per token: 16 rows at a context of 64K, plus one 256K passage.
PASSAGE_CHARACTERS = 320_000
LONG_PASSAGE_CHARACTERS = 1_200_000
PASSAGES = 16


def download(url: str, sha256: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        raise SystemExit(f"{url}: expected sha256 {sha256}, got {digest}")
    return data


def book_text(data: bytes) -> str:
    text = data.decode("utf-8-sig")
    start = re.search(r"\*\*\* ?START OF .*\*\*\*", text)
    end = re.search(r"\*\*\* ?END OF .*\*\*\*", text)
    if start is None or end is None:
        raise SystemExit("a book has no start and end markers")
    return text[start.end() : end.start()].strip()


def source_text(archive: bytes) -> str:
    """The standard library's Python files, in a fixed order."""
    parts = []
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in sorted(tar.getmembers(), key=lambda member: member.name):
            if re.fullmatch(r"[^/]+/Lib/(?!test/)[^/]+(/[^/]+)*\.py", member.name):
                body = tar.extractfile(member)
                if body is not None:
                    parts.append(body.read().decode("utf-8", errors="replace"))
    return "\n\n".join(parts)


def passages(text: str, instruction: str, source: str) -> list[dict]:
    needed = PASSAGE_CHARACTERS * PASSAGES + LONG_PASSAGE_CHARACTERS
    if len(text) < needed:
        raise SystemExit(f"{source}: {len(text)} characters, and {needed} are needed")
    rows = [
        {
            "text": text[index * PASSAGE_CHARACTERS : (index + 1) * PASSAGE_CHARACTERS],
            "instruction": instruction,
            "source": source,
        }
        for index in range(PASSAGES)
    ]
    rows.append(
        {
            "text": text[PASSAGES * PASSAGE_CHARACTERS : needed],
            "instruction": instruction,
            "source": source,
        }
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("bench/corpus"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    english = "\n\n".join(book_text(download(*item)) for item in ENGLISH.items())
    coding = source_text(download(*CODING))
    for name, text, instruction in (
        ("english", english, ENGLISH_INSTRUCTION),
        ("coding", coding, CODING_INSTRUCTION),
    ):
        rows = passages(text, instruction, name)
        with (args.output / f"{name}.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        print(f"{name}: {len(rows)} passages, {len(text)} characters")


if __name__ == "__main__":
    main()
