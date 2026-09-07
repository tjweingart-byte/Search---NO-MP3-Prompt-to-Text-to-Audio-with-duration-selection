#!/usr/bin/env python3
"""Two first-audio runs, verified and compared on identical definitions.

Verification first, because a comparison of a truncated file is worse than no
comparison: it looks finished. Every count is read from the file, and the
expected trial count is derived from the corpus size the run recorded rather
than assumed.

    python tools/compare_runs.py \\
        experiments/results/chatterbox_local_mps.json \\
        experiments/results/chatterbox_runpod_4090.json

Nothing here infers a number that is not in the files. Where a field is absent
it prints a dash and says so, rather than deriving a plausible substitute.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

BUCKETS = ("short", "medium", "long")


def load(path: pathlib.Path) -> dict:
    if not path.exists():
        raise SystemExit(f"no file at {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} is not valid JSON: {exc}")
    if "rows" not in data or "summary" not in data:
        raise SystemExit(f"{path} is not a first-audio run (no rows/summary)")
    return data


def percentile(values: list[float], fraction: float) -> float | None:
    """Nearest-rank percentile. Explicit, so the tails are not a library's guess."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def verify(name: str, data: dict, path: pathlib.Path) -> list[str]:
    """Structural checks. Returns the problems; empty means sound."""
    problems = []
    rows = data.get("rows") or []
    ok_rows = [r for r in rows if r.get("ok")]
    failed = [r for r in rows if not r.get("ok")]

    print(f"\n{name}")
    print(f"  file            {path}")
    print(f"  label           {data.get('label')}")
    print(f"  transport       {data.get('transport')}")
    print(f"  rows            {len(rows)}  ({len(ok_rows)} ok, {len(failed)} failed)")

    if data.get("simulated"):
        problems.append("this run is SIMULATED and is not a measurement")
    if data.get("smoke_run"):
        problems.append("this is a smoke run, not the full sweep")
    if data.get("is_production_latency"):
        problems.append("is_production_latency is true; these are dev benchmarks")

    devices = sorted({r.get("device") for r in ok_rows if r.get("device")})
    print(f"  devices         {', '.join(devices) or '(none recorded)'}")
    if len(devices) > 1:
        problems.append(f"rows span more than one device: {devices}")

    trials = sorted({r.get("trial") for r in ok_rows if r.get("trial")})
    chunks = len({r.get("source") for r in ok_rows if r.get("source")})
    if trials and chunks:
        expected = chunks * len(trials)
        print(f"  chunks x trials {chunks} x {len(trials)} = {expected} expected")
        if len(rows) != expected:
            problems.append(
                f"{len(rows)} rows but {expected} expected "
                f"({chunks} distinct chunks x {len(trials)} trials)")
    else:
        print("  chunks x trials (source/trial not recorded; count unchecked)")

    for bucket in BUCKETS:
        mine = [r for r in ok_rows if r.get("bucket") == bucket]
        if not mine:
            problems.append(f"bucket {bucket!r} has no rows")

    cold = (data.get("summary") or {}).get("cold_start")
    if cold:
        print(f"  cold start      load {cold.get('load_seconds', float('nan')):.2f}s, "
              f"warmup {cold.get('warmup_seconds', float('nan')):.2f}s "
              f"(excluded: {cold.get('excluded_from_trials')})")
    else:
        print("  cold start      not recorded")

    collapsed = (data.get("summary") or {}).get("collapsed_marks") or []
    print(f"  collapsed marks {', '.join(collapsed) or 'none'}")

    print(f"  verdict         {'OK' if not problems else 'PROBLEMS'}")
    for problem in problems:
        print(f"    - {problem}")
    return problems


def stats_for(rows: list[dict], key: str) -> dict | None:
    values = [r[key] for r in rows if r.get(key) is not None]
    if not values:
        return None
    return {
        "n": len(values),
        "p50": statistics.median(values),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
        "iqr": (percentile(values, 0.75) - percentile(values, 0.25)),
    }


def slope(rows: list[dict]) -> tuple[float, float, float] | None:
    """Least-squares seconds-per-word, plus the correlation. Measured, not assumed."""
    pairs = [(r["words"], r["first_playable_seconds"]) for r in rows
             if r.get("words") and r.get("first_playable_seconds") is not None]
    if len(pairs) < 3:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    if sxx == 0:
        return None
    beta = sxy / sxx
    syy = sum((y - mean_y) ** 2 for y in ys)
    r = sxy / math.sqrt(sxx * syy) if syy else float("nan")
    return beta, mean_y - beta * mean_x, r


def fmt(value, places=3, unit="s"):
    return f"{value:.{places}f}{unit}" if value is not None else "—"


def compare(a_name: str, a: dict, b_name: str, b: dict) -> None:
    a_rows = [r for r in a["rows"] if r.get("ok")]
    b_rows = [r for r in b["rows"] if r.get("ok")]

    print(f"\n\nfirst playable audio, by bucket   ({a_name} vs {b_name})\n")
    print(f"  {'bucket':<8}{'words':>6}{'n':>5}"
          f"{a_name[:9]+' p50':>16}{b_name[:9]+' p50':>16}{'speedup':>10}")
    for bucket in BUCKETS:
        ar = [r for r in a_rows if r.get("bucket") == bucket]
        br = [r for r in b_rows if r.get("bucket") == bucket]
        a_stat = stats_for(ar, "first_playable_seconds")
        b_stat = stats_for(br, "first_playable_seconds")
        words = [r["words"] for r in ar + br if r.get("words")]
        speed = (a_stat["p50"] / b_stat["p50"]
                 if a_stat and b_stat and b_stat["p50"] else None)
        print(f"  {bucket:<8}{statistics.median(words) if words else 0:>6.0f}"
              f"{(a_stat['n'] if a_stat else 0):>5}"
              f"{fmt(a_stat['p50']) if a_stat else '—':>16}"
              f"{fmt(b_stat['p50']) if b_stat else '—':>16}"
              f"{(f'{speed:.1f}x' if speed else '—'):>10}")

    for label, rows in ((a_name, a_rows), (b_name, b_rows)):
        print(f"\n\ndistribution and tails - {label}\n")
        print(f"  {'bucket':<8}{'n':>5}{'p50':>9}{'p90':>9}{'p95':>9}"
              f"{'min':>9}{'max':>9}{'IQR':>9}")
        for bucket in BUCKETS:
            mine = [r for r in rows if r.get("bucket") == bucket]
            stat = stats_for(mine, "first_playable_seconds")
            if not stat:
                print(f"  {bucket:<8}{'—':>5}")
                continue
            print(f"  {bucket:<8}{stat['n']:>5}{fmt(stat['p50']):>9}"
                  f"{fmt(stat['p90']):>9}{fmt(stat['p95']):>9}"
                  f"{fmt(stat['min']):>9}{fmt(stat['max']):>9}"
                  f"{fmt(stat['iqr']):>9}")

        print(f"\n  {'bucket':<8}{'model p50':>12}{'delivery p50':>14}"
              f"{'audio p50':>12}{'realtime':>10}")
        for bucket in BUCKETS:
            mine = [r for r in rows if r.get("bucket") == bucket]
            model = stats_for(mine, "model_seconds")
            delivery = stats_for(mine, "delivery_seconds")
            audio = stats_for(mine, "audio_seconds")
            rtf = (audio["p50"] / model["p50"]
                   if audio and model and model["p50"] else None)
            print(f"  {bucket:<8}{fmt(model['p50']) if model else '—':>12}"
                  f"{fmt(delivery['p50'], 4) if delivery else '—':>14}"
                  f"{fmt(audio['p50']) if audio else '—':>12}"
                  f"{(f'{rtf:.1f}x' if rtf else '—'):>10}")

        line = slope(rows)
        if line:
            beta, intercept, r = line
            print(f"\n  words -> latency   {beta * 1000:.1f} ms per word, "
                  f"intercept {intercept:.2f}s, r = {r:.3f}")
        else:
            print("\n  words -> latency   not computable from this file")

    print("\n\ncold start (excluded from every figure above)\n")
    for label, data in ((a_name, a), (b_name, b)):
        cold = (data.get("summary") or {}).get("cold_start")
        if not cold:
            print(f"  {label:<10} not recorded")
            continue
        print(f"  {label:<10} load {cold.get('load_seconds', 0):.2f}s  "
              f"warmup {cold.get('warmup_seconds', 0):.2f}s  "
              f"total {cold.get('total_cold_seconds', 0):.2f}s")

    print("\n\nanomalies\n")
    found = False
    for label, rows, data in ((a_name, a_rows, a), (b_name, b_rows, b)):
        failed = [r for r in data["rows"] if not r.get("ok")]
        if failed:
            found = True
            print(f"  {label}: {len(failed)} failed row(s); first: "
                  f"{failed[0].get('error')}")
        nonzero = [r for r in rows if (r.get("delivery_seconds") or 0) > 0.001]
        if nonzero:
            found = True
            print(f"  {label}: {len(nonzero)} row(s) with delivery > 1 ms "
                  "(in-process delivery should be ~0)")
        backwards = [r for r in rows
                     if r.get("model_seconds") is not None
                     and r.get("first_playable_seconds") is not None
                     and r["first_playable_seconds"] < r["model_seconds"] - 1e-6]
        if backwards:
            found = True
            print(f"  {label}: {len(backwards)} row(s) playable before the model "
                  "finished, which is impossible; check the clock")
    marks_a = set((a.get("summary") or {}).get("collapsed_marks") or [])
    marks_b = set((b.get("summary") or {}).get("collapsed_marks") or [])
    if marks_a != marks_b:
        found = True
        print(f"  collapsed marks differ: {a_name} {sorted(marks_a)} vs "
              f"{b_name} {sorted(marks_b)}")
    if not found:
        print("  none")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("first")
    parser.add_argument("second")
    args = parser.parse_args()

    first, second = pathlib.Path(args.first), pathlib.Path(args.second)
    a, b = load(first), load(second)
    a_name = "MPS" if "mps" in first.name.lower() else first.stem[:10]
    b_name = "4090" if "4090" in second.name.lower() else second.stem[:10]

    problems = verify(a_name, a, first) + verify(b_name, b, second)
    compare(a_name, a, b_name, b)
    if problems:
        print("Verification found problems above. Treat the comparison as "
              "provisional until they are explained.\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
