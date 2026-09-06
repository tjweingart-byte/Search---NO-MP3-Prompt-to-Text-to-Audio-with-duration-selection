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

#: Bucket names, shortest first. The boundaries are **not** fixed: the chunk
#: rule guarantees every real chunk clears the word threshold, so a fixed
#: "short = under 15 words" bucket is empty by construction and the benchmark
#: would answer "does length matter" across two buckets while claiming three.
#: Boundaries are therefore terciles of the corpus itself, and the real word
#: range is printed and stored beside every name, so "short" is never mistaken
#: for a length FAM does not actually produce.
BUCKET_NAMES = ("short", "medium", "long")


def bucket_edges(counts: list[int]) -> list[int]:
    """Tercile boundaries for these word counts, as [lo, first, second, hi]."""
    ordered = sorted(counts)
    if not ordered:
        return []
    third = len(ordered) // 3
    if third == 0:                      # too few to split three ways
        return [ordered[0], ordered[-1]]
    return [ordered[0], ordered[third], ordered[2 * third], ordered[-1]]


def assign_buckets(chunks: list[dict]) -> dict:
    """Label each chunk short/medium/long by tercile; return the word ranges."""
    counts = [c["words"] for c in chunks]
    if not counts:
        return {}
    edges = bucket_edges(counts)
    if len(edges) < 4:
        for chunk in chunks:
            chunk["bucket"] = "medium"
        return {"medium": (min(counts), max(counts))}
    _, first, second, _ = edges
    for chunk in chunks:
        words = chunk["words"]
        chunk["bucket"] = ("short" if words < first
                           else "medium" if words < second else "long")
    ranges = {}
    for name in BUCKET_NAMES:
        mine = [c["words"] for c in chunks if c["bucket"] == name]
        if mine:
            ranges[name] = (min(mine), max(mine))
    return ranges


#: `report.openings_by_arm_markdown` writes each recorded chunk as a numbered
#: line under an arm heading and a bold query line:
#:
#:     ## A-control
#:
#:     **how does a heat pump work**
#:
#:       1. (28w) A heat pump does not make heat. It moves heat that ...
#:
#: There are no code fences. The first version of this parser looked only for
#: fenced blocks, found none, and reported "no chunks extracted" - which read
#: like the run had no openings rather than like the parser had the wrong
#: shape. Parsing is now pinned to the generator by a round-trip test.
_ARM = re.compile(r"^##\s+(?!#)(.+?)\s*$")
_QUERY = re.compile(r"^\*\*(.+?)\*\*\s*$")
_CHUNK = re.compile(r"^\s*(\d+)\.\s+\((\d+)w([^)]*)\)\s+(.*)$")

#: Written by the report when a trial produced no chunk at all. It is a marker,
#: not text, and must never reach a voice.
NO_CHUNK = "(no chunk)"

#: `harness._SENTENCE_CHARS`. A chunk is only ever cut at one of these, so a
#: chunk that does not end in one did not come from the chunk rule.
SENTENCE_END = ".!?"

#: Closing punctuation the rule keeps after the sentence end.
TRAILING = "\"')]}\u201d\u2019"


def ends_complete(text: str) -> bool:
    """Did this chunk end at a sentence boundary?

    `first_chunk_ready` cuts only at `.`, `!` or `?`, so a real chunk always
    ends at one. This is the empirical form of that guarantee: it is checked
    against the text rather than assumed from the code.
    """
    stripped = text.rstrip().rstrip(TRAILING)
    return bool(stripped) and stripped[-1] in SENTENCE_END


def openings_from_markdown(text: str) -> list[dict]:
    """Every recorded chunk in an openings-by-arm file, with its provenance.

    **A chunk can span several markdown lines.** `first_chunk_ready` returns
    `buffer[:index + 1].strip()`, which strips only the ends - so any newline
    the model wrote inside its opening survives into the recorded chunk, and
    the report writes that chunk inline. A model that breaks a paragraph after
    its first sentence therefore produces a row like:

          1. (40w) Monza gave us one for the history books this weekend, and if
        you were watching you already know why.

        The rest of it continues here.

    Reading only the first line gave 23 words where the report said 40. So a
    row runs from its numbered line until the next structural marker - another
    numbered line, a query, an arm heading, or the end - and the lines between
    are rejoined exactly as the model wrote them, newlines included, because
    that is the string FAM would hand to a voice.

    The recorded text is **already** the first speakable chunk; the rule ran
    when the trial ran. Re-running it here would at best be a no-op and at
    worst cut it at an earlier sentence end.

    The word count the report printed is checked against the reconstruction by
    the caller. That check is what makes this parse trustworthy rather than
    plausible: if the reconstruction is wrong, nothing is written.
    """
    lines = text.splitlines()
    out, arm, query = [], None, None
    index = 0
    while index < len(lines):
        line = lines[index]

        heading = _ARM.match(line)
        if heading:
            arm, query = heading.group(1), None
            index += 1
            continue

        asked = _QUERY.match(line)
        if asked:
            query = asked.group(1)
            index += 1
            continue

        chunk = _CHUNK.match(line)
        if not chunk:
            index += 1
            continue

        number, stated, flags, first = chunk.groups()
        body = [first]
        index += 1
        while index < len(lines):
            following = lines[index]
            if (_CHUNK.match(following) or _QUERY.match(following)
                    or _ARM.match(following)):
                break
            body.append(following)
            index += 1

        recovered = "\n".join(body).strip()
        if not recovered or recovered == NO_CHUNK:
            continue
        out.append({
            "text": recovered,
            "stated_words": int(stated),
            "truncated": "truncated" in flags,
            "arm": arm or "unlabelled",
            "query": query or "(unknown)",
            "trial": int(number),
            "lines": len(body),
        })
    return out


def extract(results_dir: pathlib.Path, words: int) -> list[dict]:
    """Real first chunks from the preserved openings, taken as recorded."""
    sources = sorted(results_dir.rglob("*openings*.md"))
    if not sources:
        # Be specific about which thing is missing. "No markdown" was wrong
        # advice here: report.md is markdown, and it has no openings in it.
        available = sorted(p.name for p in results_dir.rglob("*.md"))
        raise SystemExit(
            f"no openings file in {results_dir}\n"
            f"  found: {', '.join(available) or '(nothing)'}\n"
            "  Expected a file whose name contains 'openings', written by\n"
            "  tools/preserve_run.py from the run's artifacts.")

    seen, chunks, short, miscounted = {}, [], [], []
    for path in sources:
        for row in openings_from_markdown(path.read_text(encoding="utf-8")):
            text = row["text"]
            count = len(text.split())
            if count != row["stated_words"]:
                miscounted.append((row["stated_words"], count, text[:60]))
            if count < words:
                # Under a `words`-word rule every recorded chunk should clear
                # `words`: first_chunk_ready returns None otherwise, and the
                # report writes "(no chunk)". So a short chunk is not routine -
                # it is either a truncated response or an unexplained one, and
                # both are named below rather than counted away.
                short.append({**row, "words": count})
                continue
            if text in seen:
                seen[text]["also_from"].append(f"{row['arm']}/{row['trial']}")
                continue
            entry = {
                "text": text,
                "words": count,
                "chars": len(text),
                "bucket": None,          # assigned once the corpus is complete
                "truncated_response": row["truncated"],
                # >1 means the model wrote a newline inside its own chunk.
                "lines": row.get("lines", 1),
                # Measured from the text, not inferred from the response.
                "ends_complete": ends_complete(text),
                "source": f"{path.name}:{row['arm']}:{row['query']}:{row['trial']}",
                "also_from": [],
            }
            seen[text] = entry
            chunks.append(entry)

    if short:
        _report_short(short, words)
    incomplete = [c for c in chunks if not c["ends_complete"]]
    if incomplete:
        raise SystemExit(
            f"{len(incomplete)} chunk(s) do not end at a sentence boundary, "
            "which the chunk rule makes impossible.\n"
            f"  {incomplete[0]['source']}: {incomplete[0]['text'][-60:]!r}\n"
            "  Either the parse is wrong or these are not chunk-rule output. "
            "Nothing was written.")
    if miscounted:
        stated, got, sample = miscounted[0]
        raise SystemExit(
            f"parse mismatch on {len(miscounted)} row(s): the report says "
            f"{stated} words, the reconstructed text has {got}\n  {sample!r}\n"
            "  The parser is reading these rows wrongly. Nothing was written.")
    return chunks


def _report_short(short: list[dict], words: int) -> None:
    """Name every sub-threshold chunk and say which are explained.

    Dropping these quietly is how a corpus ends up unrepresentative without
    anyone noticing. A truncated response explains a short chunk; nothing else
    does, so anything unexplained is called out as a defect to investigate
    rather than a rounding error.
    """
    truncated = [r for r in short if r["truncated"]]
    unexplained = [r for r in short if not r["truncated"]]

    print(f"\n  {len(short)} recorded chunk(s) below the {words}-word rule, "
          "not used in the corpus:")
    for row in sorted(short, key=lambda r: r["words"]):
        why = "response hit its token cap" if row["truncated"] else "UNEXPLAINED"
        print(f"    {row['words']:>3}w  {row['arm']}/{row['query'][:32]}"
              f"/{row['trial']}  - {why}")
    if truncated:
        print(f"\n  {len(truncated)} are explained: the response stopped at "
              "max_tokens, so the text never reached a sentence end past "
              f"{words} words.")
    if unexplained:
        print(f"\n  {len(unexplained)} are NOT explained. Under a {words}-word "
              "rule first_chunk_ready returns None rather than a short chunk, "
              "and the report writes '(no chunk)'. A short chunk that is not "
              "truncated means the run used a different threshold, or this "
              "parser is still wrong. Worth checking before trusting the "
              "corpus.")
    print()


def audit(results_dir: pathlib.Path, words: int) -> int:
    """What `truncated` actually means, joined from the run's own two files.

    `truncated` in a trial record is `stop_reason == "max_tokens"` on the
    **final message** - a property of the whole response, read after the
    stream finished. It says nothing about the first chunk, which was emitted
    much earlier.

    This joins the recorded chunk text (openings file) to the recorded
    stop_reason (trials.jsonl) on arm/query/trial, and counts the four cases
    that matter, so the distinction is measured rather than argued.
    """
    sources = sorted(results_dir.rglob("*openings*.md"))
    if not sources:
        raise SystemExit(f"no openings file in {results_dir}")

    rows = []
    for path in sources:
        rows.extend(openings_from_markdown(path.read_text(encoding="utf-8")))

    # (no chunk) rows are skipped by the parser, so count them separately.
    no_chunk = 0
    for path in sources:
        no_chunk += path.read_text(encoding="utf-8").count(f") {NO_CHUNK}")

    stop_reasons: dict = {}
    trials_path = results_dir / "trials.jsonl"
    by_key: dict = {}
    if trials_path.exists():
        for line in trials_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                trial = json.loads(line)
            except json.JSONDecodeError:
                continue
            reason = trial.get("stop_reason") or "(not recorded)"
            stop_reasons[reason] = stop_reasons.get(reason, 0) + 1
            by_key[(trial.get("arm"), trial.get("query"), trial.get("trial"))] = trial

    capped = [r for r in rows if r["truncated"]]
    complete = [r for r in rows if ends_complete(r["text"])]
    capped_and_complete = [r for r in capped if ends_complete(r["text"])]
    capped_and_incomplete = [r for r in capped if not ends_complete(r["text"])]
    short = [r for r in rows if len(r["text"].split()) < words]

    print(f"\naudit: what 'truncated' means in {results_dir}\n")
    print(f"  recorded chunks (non-empty)        {len(rows)}")
    print(f"  rows with no chunk at all          {no_chunk}")
    share = f"  ({len(capped) / len(rows) * 100:.0f}% of chunks)" if rows else ""
    print(f"  responses flagged truncated        {len(capped)}{share}")
    print()
    print(f"  chunks ending at a sentence end    {len(complete)} of {len(rows)}")
    print(f"  truncated response, complete chunk {len(capped_and_complete)}")
    print(f"  truncated response, CUT chunk      {len(capped_and_incomplete)}")
    print(f"  chunks below the {words}-word rule       {len(short)}")

    if stop_reasons:
        print(f"\n  stop_reason across {sum(stop_reasons.values())} trials in "
              "trials.jsonl")
        for reason, count in sorted(stop_reasons.items(), key=lambda kv: -kv[1]):
            print(f"    {reason:<16} {count}")
    else:
        print(f"\n  (no trials.jsonl at {trials_path}; chunk-side counts only)")

    print()
    if capped_and_incomplete:
        print("  VERDICT  some first chunks really were cut mid-sentence. Those "
              "are not valid benchmark input.")
        for row in capped_and_incomplete[:5]:
            print(f"    {row['arm']}/{row['query'][:30]}/{row['trial']}: "
                  f"...{row['text'][-50:]!r}")
        return 1
    print("  VERDICT  every recorded chunk ends at a sentence boundary. "
          "'truncated' marks the response, which ran on past the chunk and was")
    print("           cut at max_tokens later. The chunks themselves are intact.")
    print()
    return 0


def verify(corpus_path: pathlib.Path, examples: int = 3) -> int:
    """Prove the corpus exists and show what is in it. Free; reads one file.

    Printed before any generation is run, because "the benchmark had real
    input" is exactly the kind of claim this project has learned not to take
    on trust.
    """
    if not corpus_path.exists():
        print(f"\n  MISSING  no chunk corpus at {corpus_path}\n")
        print("  python tools/preserve_run.py --latest --as warm_first_token")
        print("  python tools/extract_chunks.py experiments/results/warm_first_token\n")
        return 1

    data = json.loads(corpus_path.read_text(encoding="utf-8"))
    chunks = data.get("chunks") or []
    if not chunks:
        print(f"\n  EMPTY  {corpus_path} holds no chunks\n")
        return 1

    ranges = data.get("bucket_ranges") or {}
    words = [c["words"] for c in chunks]
    truncated = [c for c in chunks if c.get("truncated_response")]
    arms = sorted({c["source"].split(":")[1] for c in chunks if ":" in c["source"]})
    topics = sorted({c["source"].split(":")[2] for c in chunks
                     if c["source"].count(":") >= 2})

    print(f"\nchunk corpus: {corpus_path}")
    print(f"  from        {data.get('source')}")
    print(f"  minimum     {data.get('min_words')} words (the run's chunk rule)")
    print(f"  chunks      {len(chunks)} distinct")
    print(f"  words       {min(words)}-{max(words)}, median "
          f"{sorted(words)[len(words) // 2]}")
    print(f"  arms        {len(arms)}: {', '.join(arms)}")
    print(f"  topics      {len(topics)}")
    multiline = [c for c in chunks if c.get("lines", 1) > 1]
    intact = [c for c in chunks if c.get("ends_complete", True)]
    if truncated:
        print(f"  capped      {len(truncated)} of {len(chunks)} came from a "
              "RESPONSE that later hit max_tokens")
        print("              (a property of the whole response, not of the "
              "chunk - see --audit)")
    else:
        print("  capped      none - no chunk came from a capped response")
    print(f"  chunk text  {len(intact)} of {len(chunks)} end at a sentence "
          "boundary" + (" - all intact" if len(intact) == len(chunks) else ""))
    if multiline:
        print(f"  multi-line  {len(multiline)} contain a newline the model "
              "wrote; sent to the voice as recorded")

    print(f"\n  {'bucket':<8}{'n':>4}{'words':>12}")
    for name in BUCKET_NAMES:
        mine = [c for c in chunks if c["bucket"] == name]
        if not mine:
            continue
        low, high = ranges.get(name, (min(c["words"] for c in mine),
                                      max(c["words"] for c in mine)))
        print(f"  {name:<8}{len(mine):>4}{f'{low}-{high}':>12}")

    print("\n  examples (real text, taken verbatim from the run)")
    for name in BUCKET_NAMES:
        mine = [c for c in chunks if c["bucket"] == name]
        for chunk in mine[:examples]:
            body = chunk["text"]
            if len(body) > 150:
                body = body[:150] + "..."
            print(f"\n    [{name}, {chunk['words']}w] {chunk['source']}")
            print(f"    {body}")
    print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_dir", nargs="?",
                        help="a folder under experiments/results/")
    parser.add_argument("--words", type=int, default=25,
                        help="minimum words to keep; 25 is the rule the runs used")
    parser.add_argument("--out", default="experiments/chunks/first_chunks.json")
    parser.add_argument("--verify", action="store_true",
                        help="show what is in an existing corpus; extract nothing")
    parser.add_argument("--audit", action="store_true",
                        help="what 'truncated' means in this run; extract nothing")
    args = parser.parse_args()

    if args.verify:
        return verify(pathlib.Path(args.out))
    if args.audit:
        if not args.results_dir:
            raise SystemExit("--audit needs the results folder")
        return audit(pathlib.Path(args.results_dir), args.words)

    if not args.results_dir:
        raise SystemExit("give a results folder, or --verify an existing corpus")
    chunks = extract(pathlib.Path(args.results_dir), args.words)
    ranges = assign_buckets(chunks) if chunks else {}
    if not chunks:
        raise SystemExit(
            "no chunks extracted; nothing written\n"
            "  The openings file was found and parsed, but nothing in it "
            f"survived the {args.words}-word minimum.\n"
            "  Lower it with --words, or check the file has recorded chunks.")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "source": str(args.results_dir),
        "min_words": args.words,
        "note": ("The real first speakable chunks recorded by the run, taken "
                 "verbatim. The chunk rule was applied when the trial ran; it "
                 "is not applied again here. Nothing was written for the "
                 "benchmark."),
        "bucket_ranges": {k: list(v) for k, v in ranges.items()},
        "chunks": chunks,
    }, indent=2), encoding="utf-8")

    print(f"wrote {len(chunks)} chunks -> {out}")
    for name in BUCKET_NAMES:
        mine = [c for c in chunks if c["bucket"] == name]
        if mine:
            lo, hi = ranges[name]
            print(f"  {name:<8} {len(mine):>3}  ({lo}-{hi} words)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
