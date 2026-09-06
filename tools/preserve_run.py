#!/usr/bin/env python3
"""Copy a run out of `experiments/runs/` and into the repository, for keeps.

`experiments/runs/` is git-ignored: it holds whole episodes of audio and raw
API records, and it is per-machine. That is the right default and it has now
cost this project three handovers, because the run that answers a question
lives on one laptop and the analysis happens somewhere else. A measurement
nobody can reread is a measurement that has to be paid for twice.

This writes the durable part - the per-trial numbers, the report, and the
generated openings - under `experiments/results/<slug>/`, which *is* tracked.
It never copies audio, and it re-runs the redaction check on everything it
writes rather than trusting that the run was clean when it was made.

    python tools/preserve_run.py <run-id> [--as <slug>]
    python tools/preserve_run.py --latest
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments import redact, store                        # noqa: E402

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "experiments" / "results"

#: Copied verbatim if present. Audio is deliberately not on this list.
ARTIFACT_SUFFIXES = (".md", ".json", ".csv", ".txt")

#: Per-trial fields worth keeping forever. Everything else in a trial record is
#: either large, per-machine, or reconstructible from these.
KEEP_METRICS = (
    "seg_dispatch_to_headers",
    "seg_headers_to_first_token",
    "seg_dispatch_to_first_token",
    "seg_first_token_to_25_words",
    "seg_dispatch_to_boundary",
    "first_chunk_seconds",
    "generate_seconds",
    "search_seconds",
    "synthesis_seconds",
    "connection_reused",
    "reuse_client",
    "phase_connect",
    "phase_tls",
    "phase_local_setup",
    "phase_upload",
    "phase_wait_for_headers",
    "input_tokens",
    "output_tokens",
    "stop_reason",
    "truncated",
    "cost_usd",
)


def preserve(run, slug: str | None = None) -> pathlib.Path:
    spec = run.spec_dict() or {}
    slug = slug or run.id
    target = RESULTS_DIR / slug
    target.mkdir(parents=True, exist_ok=True)

    trials = run.trials()
    kept = []
    for trial in trials:
        metrics = trial.get("metrics") or {}
        row = {
            "arm": trial.get("arm"),
            "query": trial.get("query"),
            "trial": trial.get("index"),
            "ok": trial.get("ok"),
        }
        for key in KEEP_METRICS:
            if metrics.get(key) is not None:
                row[key] = metrics[key]
        kept.append(row)

    with (target / "trials.jsonl").open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(redact.scrub(row)) + "\n")

    (target / "spec.json").write_text(
        json.dumps(redact.scrub(spec), indent=2), encoding="utf-8")

    report = run.report()
    if report:
        (target / "report.md").write_text(redact.scrub_text(report), encoding="utf-8")

    copied = []
    art = run.artifacts_dir
    if art.is_dir():
        for path in sorted(art.iterdir()):
            if path.suffix.lower() not in ARTIFACT_SUFFIXES:
                continue
            (target / path.name).write_text(
                redact.scrub_text(path.read_text(encoding="utf-8", errors="replace")),
                encoding="utf-8")
            copied.append(path.name)

    # Prove it, do not assume it: re-read every file actually written.
    dirty = [p.name for p in sorted(target.iterdir())
             if p.is_file() and redact.looks_like_secret(
                 p.read_text(encoding="utf-8", errors="replace"))]
    if dirty:
        raise SystemExit(
            f"refusing to preserve: {', '.join(dirty)} still looks like it holds "
            "a credential. Nothing was committed; inspect the run first.")

    print(f"preserved {run.id} -> {target.relative_to(RESULTS_DIR.parent.parent)}")
    print(f"  {len(kept)} trials, {len(report.splitlines())} report lines"
          + (f", artifacts: {', '.join(copied)}" if copied else ""))
    print("  redaction re-checked on every written file: clean")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_id", nargs="?", help="run directory name")
    parser.add_argument("--latest", action="store_true", help="the newest run")
    parser.add_argument("--as", dest="slug", help="folder name under experiments/results/")
    args = parser.parse_args()

    if args.latest:
        runs = store.list_runs()
        if not runs:
            raise SystemExit("no runs on this machine")
        run = runs[0]
    elif args.run_id:
        run = store.load(args.run_id)
        if run is None:
            raise SystemExit(f"no run called {args.run_id}")
    else:
        raise SystemExit("give a run id, or --latest")

    preserve(run, args.slug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
