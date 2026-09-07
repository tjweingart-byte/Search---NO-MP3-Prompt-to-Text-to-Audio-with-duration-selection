#!/usr/bin/env python3
"""Verify a Phase 1 stage-split run and apply the pre-registered criteria.

The criteria were fixed in PHASE1_STAGE_SPLIT.md *before* the run, so the
verdict is arithmetic rather than judgement:

    T3 > 60% of total        -> incremental Chatterbox is a model-level rewrite
    Flow + HiFiGAN dominate  -> the existing finalize/cache_source are most of it
    roughly even             -> report the arithmetic, no verdict

It also prints the numbers the analysis needs, so they can be transcribed
rather than described: per-stage medians and shares, the per-token cost and
whether it is linear enough to project from, both chunking outcomes with their
error text, the seam ratios, and what Perth's detector said.

    python tools/verify_stage_split.py experiments/results/chatterbox_stage_split.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

#: From PHASE1_STAGE_SPLIT.md, fixed before the run.
T3_REWRITE_THRESHOLD = 0.60
#: Above this, the mean per-token cost is not a safe basis for projection.
PER_TOKEN_SPREAD_LIMIT = 1.5


def load(path: pathlib.Path) -> dict:
    if not path.exists():
        raise SystemExit(f"no file at {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} is not valid JSON: {exc}")
    for key in ("stages", "chunked", "watermark", "summary"):
        if key not in data:
            raise SystemExit(f"{path} is not a stage-split run (no {key!r})")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path")
    args = parser.parse_args()
    path = pathlib.Path(args.path)
    data = load(path)

    stages = data["stages"]
    print(f"\nfile        {path}")
    print(f"label       {data.get('label')}")
    print(f"device      {data.get('device')}")
    print(f"production  {data.get('is_production_latency')}")
    print(f"rows        {len(stages)}")
    cold = data.get("cold_start") or {}
    print(f"cold start  load {cold.get('load_seconds')}s "
          f"(excluded: {cold.get('excluded_from_trials')})")
    print(f"chunk_tokens {data.get('chunk_tokens')}")

    problems = []
    if data.get("is_production_latency"):
        problems.append("is_production_latency is true")
    if not stages:
        problems.append("no stage rows")

    print(f"\n{'bucket':<8}{'words':>6}{'total':>9}{'T3':>9}{'Flow':>9}"
          f"{'HiFi':>9}{'wmark':>9}{'tokens':>8}{'T3 %':>8}")
    shares = []
    for bucket, stat in (data["summary"].get("buckets") or {}).items():
        total = stat.get("total") or 0.0
        share = (stat.get("t3") or 0.0) / total if total else 0.0
        shares.append(share)
        print(f"{bucket:<8}{stat.get('words') or 0:>6.0f}{total:>8.3f}s"
              f"{stat.get('t3') or 0:>8.3f}s{stat.get('flow') or 0:>8.3f}s"
              f"{stat.get('hift') or 0:>8.3f}s{stat.get('watermark') or 0:>8.3f}s"
              f"{stat.get('tokens') or 0:>8.0f}{share * 100:>7.1f}%")

    spread = data["summary"].get("seconds_per_token_spread") or {}
    if spread:
        ratio = spread.get("max_over_min")
        print(f"\nper-token   {spread.get('median', 0) * 1000:.2f} ms  "
              f"(max/min {ratio:.2f})")
        if ratio and ratio > PER_TOKEN_SPREAD_LIMIT:
            print("            NOT linear across lengths - any projection from "
                  "the mean must be discarded")

    print("\nchunking")
    for row in data["chunked"]:
        if row.get("ok"):
            print(f"  {row['mode']:<18} first {row.get('first_chunk_seconds')}  "
                  f"worst seam {row.get('worst_seam_ratio')}  "
                  f"max diff {row.get('max_abs_diff')}")
        else:
            print(f"  {row['mode']:<18} FAILED  {row.get('error')}")

    print("\nwatermark")
    for row in data["watermark"]:
        print(f"  {row.get('bucket', '?'):<8} whole {row.get('whole_detected')}")
        print(f"           per-chunk joined {row.get('per_chunk_detected')}")
        print(f"           each chunk {row.get('per_chunk_each_detected')}")
        print(f"           seams {row.get('per_chunk_seam_ratios')}")

    print("\nverdict against the pre-registered criteria")
    if not shares:
        print("  no bucket summary; cannot apply the criteria")
        return 1
    worst = max(shares)
    mean = statistics.fmean(shares)
    print(f"  T3 share: mean {mean * 100:.1f}%, max {worst * 100:.1f}%")
    if mean > T3_REWRITE_THRESHOLD:
        print(f"  T3 exceeds the {T3_REWRITE_THRESHOLD * 100:.0f}% threshold -> "
              "incremental Chatterbox is a MODEL-LEVEL REWRITE.")
        print("  Flow and HiFiGAN streaming primitives cannot help what they "
              "do not own: the")
        print(f"  most they could remove is {(1 - mean) * 100:.1f}% of the wait.")
    else:
        print("  T3 is below the threshold; Flow + HiFiGAN carry enough of the "
              "cost to matter.")

    if problems:
        print("\nproblems")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
