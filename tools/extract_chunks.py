#!/usr/bin/env python3
"""Build the Chatterbox input corpus out of openings FAM actually wrote.

The Chatterbox benchmark must be fed the text FAM would really hand a voice,
not sentences invented to fill a test. Invented text gets chosen - without
anyone meaning to - for being tidy, and tidy text is short, which would make
the voice look faster than it is.

So the corpus is extracted from preserved experiment results: every recorded
opening is run through the *production* chunk rule (`first_chunk_ready`) at the
threshold that produced it, and what comes out is exactly the string the
pipeline would submit.

    python tools/preserve_run.py --latest --as warm_first_token
    python tools/extract_chunks.py experiments/results/warm_first_token \\
        --out experiments/chunks/first_chunks.json

Chunks are bucketed by word count so the benchmark can answer "does longer text
start slower", and duplicates are dropped - two arms often open identically,
and timing the same string twice measures nothing new.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments.harness import first_chunk_ready                # noqa: E402

#: Word-count buckets. The benchmark reports startup latency per bucket, which
#: is how "does length matter" gets a yes or no instead of a scatter plot.
BUCKETS = ((0, 15, "short"), (15, 30, "medium"), (30, 60, "long"), (60, 10_000, "very long"))

#: Openings are written under a heading naming the arm, then a fenced or
#: indented block. Both shapes are accepted; anything else is skipped loudly.
_HEADING = re.compile(r"^#{2,4}\s+(.+?)\s*$")


def bucket_for(words: int) -> str:
    for low, high, name in BUCKETS:
        if low <= words < high:
            return name
    return "very long"


def openings_from_markdown(text: str) -> list[tuple[str, str]]:
    """(label, opening) pairs from a preserved openings-by-arm file."""
    out, label, buffer, in_fence = [], None, [], False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            if in_fence:
                body = "\n".join(buffer).strip()
                if body:
                    out.append((label or "unlabelled", body))
                buffer = []
            in_fence = not in_fence
            continue
        if in_fence:
            buffer.append(line)
            continue
        heading = _HEADING.match(line)
        if heading:
            label = heading.group(1)
    return out


def extract(results_dir: pathlib.Path, words: int) -> list[dict]:
    sources = sorted(results_dir.glob("*.md"))
    if not sources:
        raise SystemExit(f"no markdown in {results_dir} - preserve a run first")

    seen, chunks = set(), []
    skipped = 0
    for path in sources:
        for label, opening in openings_from_markdown(path.read_text(encoding="utf-8")):
            chunk = first_chunk_ready(opening, words)
            if not chunk:
                # The opening never reached a sentence end past the threshold.
                # That is a real pipeline case, but it is not a chunk.
                skipped += 1
                continue
            key = chunk.strip()
            if key in seen:
                continue
            seen.add(key)
            count = len(key.split())
            chunks.append({
                "text": key,
                "words": count,
                "chars": len(key),
                "bucket": bucket_for(count),
                "source": f"{path.name}:{label}",
            })
    if skipped:
        print(f"  {skipped} opening(s) never reached a {words}-word sentence end")
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_dir", help="a folder under experiments/results/")
    parser.add_argument("--words", type=int, default=25,
                        help="chunk threshold; 25 is what the runs used")
    parser.add_argument("--out", default="experiments/chunks/first_chunks.json")
    args = parser.parse_args()

    chunks = extract(pathlib.Path(args.results_dir), args.words)
    if not chunks:
        raise SystemExit("no chunks extracted; nothing written")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "source": str(args.results_dir),
        "chunk_words": args.words,
        "note": ("Extracted from real FAM openings with the production chunk "
                 "rule. Nothing here was written for the benchmark."),
        "chunks": chunks,
    }, indent=2), encoding="utf-8")

    print(f"wrote {len(chunks)} chunks -> {out}")
    for _, _, name in BUCKETS:
        mine = [c for c in chunks if c["bucket"] == name]
        if mine:
            lo = min(c["words"] for c in mine)
            hi = max(c["words"] for c in mine)
            print(f"  {name:<10} {len(mine):>3}  ({lo}-{hi} words)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
