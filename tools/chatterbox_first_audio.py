#!/usr/bin/env python3
"""Chatterbox: first speakable text in -> first playable audio out.

Answers one product question and refuses to drift off it: once FAM has a
sentence worth speaking, how long before sound starts? Total synthesis time is
recorded but is not the headline, because a listener never waits for it.

Three transports, and the difference between them is the point:

    --local              Chatterbox Turbo in this process. Free. Needs a GPU
                         (or Apple mps); CPU is refused, not silently used.
    --endpoint URL       A pod someone already started. This tool never starts
                         one.
    --simulate           A fake with a fixed synthetic delay. Free, no model,
                         no network - proves the instrument before it is aimed
                         at anything that costs money. It is labelled
                         SIMULATED everywhere it appears.

Chatterbox is one-shot, so several of the five requested marks collapse onto
one instant. The probe records them separately and lets them come out equal;
`collapsed_marks` on every row says which, and why.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments import chatterbox_probe as probe                # noqa: E402
from experiments.adapters import chatterbox_impl as impl         # noqa: E402

DEFAULT_CHUNKS = "experiments/chunks/first_chunks.json"

#: The simulated engine's pretend speed. Roughly the realtime factor the
#: recovered 4090 benchmarks imply, so the shape of the output is plausible -
#: but it is a stand-in and every row it produces says so.
SIMULATED_REALTIME = 12.0
SIMULATED_WPM = 150.0


def load_chunks(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"no chunk corpus at {path}\n"
            "  Build one from real openings:\n"
            "    python tools/preserve_run.py --latest --as warm_first_token\n"
            "    python tools/extract_chunks.py experiments/results/warm_first_token")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["chunks"]


def simulate(text: str) -> probe.FirstAudio:
    """No model, no network. Proves the timing harness, measures nothing real."""
    words = len(text.split())
    audio_seconds = words / SIMULATED_WPM * 60.0
    generate = audio_seconds / SIMULATED_REALTIME
    start = time.perf_counter()
    time.sleep(generate)
    elapsed = time.perf_counter() - start

    result = probe.FirstAudio(text=text, words=words, chars=len(text),
                              transport="simulated", device="none")
    result.dispatch = 0.0
    result.stream_begin = elapsed
    result.first_audio_bytes = elapsed
    result.playable = elapsed
    result.complete = elapsed
    result.sample_rate = 24000
    result.audio_seconds = audio_seconds
    result.collapsed = ["stream_begin", "first_audio_bytes"]
    result.detail = {"SIMULATED": True, "one_shot": True,
                     "why_collapsed": "Simulated one-shot engine."}
    return result


def run(chunks, transport, trials, endpoint=None, device=None) -> list[dict]:
    rows = []
    for trial in range(1, trials + 1):
        for chunk in chunks:
            text = chunk["text"]
            try:
                if transport == "simulate":
                    measured = simulate(text)
                elif transport == "local":
                    measured = probe.measure_in_process(text, device=device)
                else:
                    measured = probe.measure_http(endpoint, text)
            except Exception as exc:
                rows.append({"trial": trial, "bucket": chunk["bucket"],
                             "words": chunk["words"], "ok": False,
                             "error": f"{type(exc).__name__}: {exc}"})
                print(f"  trial {trial} {chunk['bucket']:<10} FAILED: {exc}")
                continue
            row = {"trial": trial, "bucket": chunk["bucket"], "ok": True,
                   "source": chunk.get("source"), **measured.as_dict()}
            rows.append(row)
            print(f"  trial {trial} {chunk['bucket']:<10} "
                  f"{chunk['words']:>3}w  first playable "
                  f"{measured.to_first_playable:.3f}s")
    return rows


def summarise(rows) -> dict:
    ok = [r for r in rows if r.get("ok")]
    by_bucket = {}
    for row in ok:
        by_bucket.setdefault(row["bucket"], []).append(row)
    out = {"trials": len(rows), "ok": len(ok), "buckets": {}}
    for name, mine in by_bucket.items():
        played = sorted(r["first_playable_seconds"] for r in mine)
        out["buckets"][name] = {
            "n": len(mine),
            "words_median": statistics.median(r["words"] for r in mine),
            "first_playable_p50": statistics.median(played),
            "first_playable_min": played[0],
            "first_playable_max": played[-1],
            "audio_seconds_p50": statistics.median(
                r["audio_seconds"] for r in mine if r.get("audio_seconds")),
        }
    collapsed = sorted({m for r in ok for m in r.get("collapsed_marks", [])})
    out["collapsed_marks"] = collapsed
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--local", action="store_true", help="in-process, free, needs a GPU")
    where.add_argument("--endpoint", help="an already-running Chatterbox endpoint")
    where.add_argument("--simulate", action="store_true", help="fake engine; proves the harness")
    parser.add_argument("--chunks", default=DEFAULT_CHUNKS)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--device", help="cuda / mps; never cpu unless named")
    parser.add_argument("--out", help="write rows and summary here as JSON")
    args = parser.parse_args()

    chunks = load_chunks(pathlib.Path(args.chunks))
    transport = "simulate" if args.simulate else ("local" if args.local else "http")

    if transport == "local":
        resolved, explicit = impl.resolve_device(args.device)
        if resolved == "cpu" and not explicit:
            raise SystemExit(
                "This machine has no cuda or mps device. Chatterbox on CPU is "
                "slower than realtime, so timing it would measure the machine "
                "rather than the model.\n"
                "  Run with --device cpu only if you mean to time the CPU.")
        print(f"device: {resolved}")
    if transport == "simulate":
        print("SIMULATED: no model, no network. These numbers are not measurements.")

    print(f"{len(chunks)} chunks x {args.trials} trials = "
          f"{len(chunks) * args.trials} synthesises\n")
    rows = run(chunks, transport, args.trials, args.endpoint, args.device)
    summary = summarise(rows)

    print(f"\n{'bucket':<12}{'n':>4}{'words':>8}{'first playable p50':>22}")
    for name, stats in summary["buckets"].items():
        print(f"{name:<12}{stats['n']:>4}{stats['words_median']:>8.0f}"
              f"{stats['first_playable_p50']:>21.3f}s")
    if summary["collapsed_marks"]:
        print(f"\ncollapsed marks: {', '.join(summary['collapsed_marks'])}"
              "  (one-shot engine / non-streaming contract)")

    if args.out:
        path = pathlib.Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"transport": transport, "simulated": transport == "simulate",
             "summary": summary, "rows": rows}, indent=2), encoding="utf-8")
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
