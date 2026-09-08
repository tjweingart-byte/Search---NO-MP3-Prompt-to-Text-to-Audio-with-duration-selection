#!/usr/bin/env python3
"""What a real run's chunks were, and what the assembler would have made of them.

The Phase 6 assembly policy is bounded by the only Chatterbox-on-4090 data in
this repository - 28, 33 and 41 words - and that is not much. This closes the
gap from the other end: point it at a Phase 5 or Phase 6 `results.json` and it
reports the measured distribution, fits generation time against word count from
that run's own timings, and replays the assembler over the same sentences to
show what it would have produced.

Phase 5's per-chunk data is not in this repository, so nothing here is
guesswork about it - the file is read or the tool says it cannot find one.

    python3 tools/fit_chunk_policy.py experiments/results/streaming_4090_<stamp>/results.json
    python3 tools/fit_chunk_policy.py <phase5>/results.json --compare <phase6>/results.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments.speech_assembler import (AssemblyPolicy,        # noqa: E402
                                          SpeechAssembler)


def sentences_from(run: dict) -> list:
    """The raw sentences a run saw, whichever phase wrote the file.

    Phase 5 synthesised one sentence per call, so its `chunks` *are* the
    sentences. Phase 6 records them separately because they are no longer the
    same thing.
    """
    if run.get("raw_sentences"):
        return [row["text"] for row in run["raw_sentences"]]
    return [row["text"] for row in run.get("chunks", []) if row.get("text")]


def timings_from(run: dict) -> list:
    """(words, generate_seconds) for every synthesis the run actually did."""
    out = []
    for row in run.get("chunks", []):
        words = row.get("words")
        seconds = row.get("generate_seconds")
        if words and seconds:
            out.append((int(words), float(seconds)))
    return out


def fit(points: list) -> dict:
    """Least squares of generate-time against word count, with its own caveats."""
    if len(points) < 3:
        return {"points": len(points),
                "note": "too few syntheses to fit anything"}
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return {"points": len(points),
                "note": "every chunk was the same length; nothing to fit"}
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    ss = sum((y - my) ** 2 for y in ys)
    sr = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    return {
        "points": len(points),
        "slope_seconds_per_word": slope,
        "intercept_seconds": intercept,
        "r_squared": 1 - sr / ss if ss else None,
        "word_range": [min(xs), max(xs)],
        "note": ("a negative intercept means no measurable fixed cost per "
                 "invocation over this range - batching would not buy compute"
                 if intercept < 0 else
                 f"a fixed cost of about {intercept:.2f}s per invocation, which "
                 "is what batching would amortise"),
    }


def distribution(words: list) -> dict:
    if not words:
        return {"chunks": 0}
    return {
        "chunks": len(words),
        "total_words": sum(words),
        "median_words": statistics.median(words),
        "mean_words": statistics.mean(words),
        "min_words": min(words), "max_words": max(words),
        "under_5_words": sum(1 for w in words if w < 5),
        "under_10_words": sum(1 for w in words if w < 10),
        "under_18_words": sum(1 for w in words if w < 18),
        "histogram": {label: sum(1 for w in words if low <= w < high)
                      for label, low, high in
                      (("0-4", 0, 5), ("5-9", 5, 10), ("10-17", 10, 18),
                       ("18-27", 18, 28), ("28-41", 28, 42),
                       ("42+", 42, 10 ** 6))},
    }


def replay(sentences: list, policy: AssemblyPolicy) -> dict:
    """What the assembler would have produced from the same sentences.

    No clock and no headroom: this is the word-threshold behaviour alone. The
    timer and headroom rules only ever release *earlier*, so this is the upper
    bound on chunk size and the lower bound on call count. The opening chunk
    has no size rule either way - it is whatever the first complete sentence
    was - so a short first entry here is the policy working, not a miss.
    """
    assembler = SpeechAssembler(policy=policy, clock=lambda: 0.0)
    released = []
    for sentence in sentences:
        released += assembler.offer(sentence)
    released += assembler.flush()
    witness = assembler.spoken_matches_source(released)
    out = distribution([c.words for c in released])
    out["text_integrity"] = witness or "exact - no text lost, added or reordered"
    out["release_reasons"] = {reason: sum(1 for c in released
                                          if c.reason == reason)
                              for reason in sorted({c.reason for c in released})}
    out["first_chunk_words"] = released[0].words if released else None
    out["first_chunk_sentences"] = released[0].sentences if released else None
    return out


def report(path: pathlib.Path, policy: AssemblyPolicy) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {"file": str(path), "label": data.get("label"),
           "stub": data.get("stub", False), "runs": {}}
    for run in data.get("runs", []):
        sentences = sentences_from(run)
        words = [len(s.split()) for s in sentences]
        out["runs"][run.get("run", "?")] = {
            "as_run": distribution([row.get("words") or 0
                                    for row in run.get("chunks", [])]),
            "sentences": distribution(words),
            "generate_fit": fit(timings_from(run)),
            "assembler_would_produce": replay(sentences, policy),
        }
    return out


def _print(section: str, body: dict, indent: str = "  ") -> None:
    print(f"\n{section}")
    for key, value in body.items():
        if isinstance(value, dict):
            print(f"{indent}{key}:")
            width = max((len(str(k)) for k in value), default=0) + 2
            for inner, count in value.items():
                print(f"{indent}  {str(inner):<{width}}{count}")
        elif isinstance(value, float):
            print(f"{indent}{key:<28}{value:.4f}")
        else:
            print(f"{indent}{key:<28}{value}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", help="a results.json from Phase 5 or 6")
    parser.add_argument("--compare", default="",
                        help="a second results.json to put beside it")
    # No first-chunk knob: the opening has no size rule to sweep.
    parser.add_argument("--min-words", type=int, dest="min_words")
    parser.add_argument("--target-words", type=int, dest="target_words")
    parser.add_argument("--max-words", type=int, dest="max_words")
    parser.add_argument("--json", default="", help="write the report here too")
    args = parser.parse_args(argv)

    given = {name: getattr(args, name) for name in
             ("min_words", "target_words", "max_words")
             if getattr(args, name, None)}
    policy = AssemblyPolicy(**given)
    print(f"\npolicy under test: {vars(policy)}")

    reports = []
    for name in [args.results] + ([args.compare] if args.compare else []):
        path = pathlib.Path(name)
        if not path.exists():
            raise SystemExit(f"no results file at {path}")
        found = report(path, policy)
        reports.append(found)
        print(f"\n{'=' * 72}\n{path}\n  {found['label']}")
        if found["stub"]:
            print("  THIS IS A STUB FILE - not a measurement.")
        for run, body in found["runs"].items():
            print(f"\n--- run: {run} ---")
            _print("as run (one entry per synthesis)", body["as_run"])
            _print("generate time vs words, fitted from this run",
                   body["generate_fit"])
            _print("what the assembler would produce",
                   body["assembler_would_produce"])

    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps(reports, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {args.json}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
