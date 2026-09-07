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


#: Every result carries where it ran, because "Chatterbox is Xs" without the
#: device is the kind of number that ends up in a slide and then in a plan.
LABEL_SIMULATED = "SIMULATED - NOT A CHATTERBOX MEASUREMENT"


def banner(device: str) -> str:
    """What these numbers are, said before they appear and stored beside them."""
    if device == "mps":
        return ("LOCAL MPS / DEVELOPMENT BENCHMARK - Apple silicon, not the "
                "production deployment target. Not production latency.")
    if device == "cpu":
        return ("LOCAL CPU / DEVELOPMENT BENCHMARK - slower than realtime by "
                "design. Not production latency.")
    return f"LOCAL {device.upper()} / DEVELOPMENT BENCHMARK - not production latency."


def prepare_local(device: str) -> dict:
    """Load the model and warm it, on the clock, before any trial is timed.

    Cold start is measured here and then excluded from every trial, because it
    is paid once per process and a production server pays it at boot. Folding
    it into request latency would make the first chunk look catastrophic and
    every later one look free.

    The device is re-read from the loaded model rather than trusted: Chatterbox
    Turbo's `from_pretrained` silently falls back to CPU when MPS is missing,
    which would otherwise be reported as an MPS result.
    """
    load_started = time.perf_counter()
    model, load_seconds = impl.load_model(device)
    actual = str(getattr(model, "device", device))
    if actual.split(":")[0] != device:
        raise SystemExit(
            f"asked for device {device!r} but the model loaded on {actual!r}.\n"
            "  Chatterbox falls back to CPU without raising. Refusing to label "
            "a CPU run as something else.")
    warm_started = time.perf_counter()
    impl.warm_up(model, device)
    return {
        "device": device,
        "load_seconds": load_seconds,
        "warmup_seconds": time.perf_counter() - warm_started,
        "total_cold_seconds": time.perf_counter() - load_started,
        "excluded_from_trials": True,
    }


def preflight(chunks_path: pathlib.Path, device: str | None) -> int:
    """Everything that can fail, checked before anything slow starts.

    A benchmark that dies twenty minutes in - on a missing package, an absent
    corpus, or a device that quietly became CPU - has cost more than the run
    was worth. None of these checks loads a model or downloads a weight.
    """
    ok = True

    def check(label, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'ok  ' if good else 'FAIL'}  {label}"
              + (f"  {detail}" if detail else ""))

    print("\npreflight\n")

    # The accelerator that must work is the one asked for. This used to check
    # MPS unconditionally, which failed a perfectly good CUDA pod on two lines
    # about Apple silicon - a preflight that cries wolf gets read past, which
    # is the opposite of the point.
    resolved, explicit = impl.resolve_device(device)
    devices = impl.available_devices()

    try:
        import torch
        check("torch importable", True, torch.__version__)
    except ImportError:
        check("torch importable", False,
              "pip install -r experiments/requirements-chatterbox.txt")
        torch = None

    # `resolve_device` honours an explicitly named device without asking the
    # machine whether it has one - right for the runner, wrong here: naming
    # --device mps on a box with no Metal passed until this consulted
    # available_devices().
    check(f"accelerator {resolved!r} exists on this machine",
          devices.get(resolved, False),
          "available: " + ", ".join(k for k, v in devices.items() if v))
    check("device is not CPU", resolved != "cpu" or explicit, resolved)

    if torch is not None and resolved == "cuda" and devices.get("cuda"):
        try:
            name = torch.cuda.get_device_name(0)
        except Exception as exc:                   # pragma: no cover - defensive
            name = f"(could not read: {exc})"
        print(f"  info  cuda device  {name}")
    # Anything about the accelerators NOT being used is information, never a
    # gate. MPS on a Linux box is absent by definition, not broken.
    others = [k for k in ("cuda", "mps") if k != resolved]
    print("  info  other accelerators  "
          + ", ".join(f"{k}={devices.get(k, False)}" for k in others))

    try:
        import importlib
        importlib.import_module(impl.TURBO_MODULE)
        check(f"{impl.TURBO_MODULE} importable", True)
    except ImportError as exc:
        check(f"{impl.TURBO_MODULE} importable", False, str(exc))

    # Importing the module is not enough. from_pretrained instantiates
    # perth.PerthImplicitWatermarker, and perth sets that to None when its own
    # import fails - so the failure only appears after a 4 GB download, as a
    # TypeError with no mention of the real cause. Check the attribute itself.
    try:
        import perth
        ok_perth = getattr(perth, "PerthImplicitWatermarker", None) is not None
        check("perth watermarker is loadable", ok_perth,
              "" if ok_perth else "it is None - run tools/diagnose_chatterbox.py")
    except ImportError as exc:
        check("perth watermarker is loadable", False, str(exc))

    if chunks_path.exists():
        try:
            chunks = json.loads(chunks_path.read_text(encoding="utf-8"))["chunks"]
        except Exception as exc:
            check("chunk corpus readable", False, str(exc))
            chunks = []
        buckets = sorted({c["bucket"] for c in chunks})
        check("chunk corpus present", bool(chunks), f"{len(chunks)} chunks")
        # Buckets are terciles of the corpus, so three of them means the corpus
        # was big enough to split - not that any fixed length was present.
        check("corpus splits into three length buckets",
              {"short", "medium", "long"} <= set(buckets), ", ".join(buckets))
    else:
        check("chunk corpus present", False, f"{chunks_path} does not exist")
        print("        python tools/preserve_run.py --latest --as warm_first_token")
        print("        python tools/extract_chunks.py experiments/results/warm_first_token")

    print()
    if ok:
        print("  preflight ok - the benchmark can run.\n")
        return 0
    print("  preflight failed - fix the above before running.\n")
    return 1


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
    result.model_seconds = elapsed
    result.delivery_seconds = 0.0
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
        def med(key):
            values = [r[key] for r in mine if r.get(key) is not None]
            return statistics.median(values) if values else None

        audio = med("audio_seconds")
        model = med("model_seconds")
        out["buckets"][name] = {
            "n": len(mine),
            "words_median": statistics.median(r["words"] for r in mine),
            "first_playable_p50": statistics.median(played),
            "first_playable_min": played[0],
            "first_playable_max": played[-1],
            "audio_seconds_p50": audio,
            # The two halves the experiment exists to tell apart.
            "model_seconds_p50": model,
            "delivery_seconds_p50": med("delivery_seconds"),
            "realtime_factor_p50": (audio / model) if audio and model else None,
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
    parser.add_argument("--max-chunks", type=int, dest="max_chunks",
                        help="use only the first N chunks of each bucket; for a "
                             "smoke run that proves the device before the sweep")
    parser.add_argument("--device", help="cuda / mps; never cpu unless named")
    parser.add_argument("--out", help="write rows and summary here as JSON")
    parser.add_argument("--preflight", action="store_true",
                        help="check everything and run nothing; free and fast")
    args = parser.parse_args()

    if args.preflight:
        return preflight(pathlib.Path(args.chunks), args.device)

    chunks = load_chunks(pathlib.Path(args.chunks))
    if args.max_chunks:
        # Evenly across buckets, so a smoke run still exercises the short and
        # the long case rather than whichever happened to be first.
        trimmed, seen = [], {}
        for chunk in chunks:
            bucket = chunk["bucket"]
            if seen.get(bucket, 0) < args.max_chunks:
                seen[bucket] = seen.get(bucket, 0) + 1
                trimmed.append(chunk)
        chunks = trimmed
        print(f"SMOKE RUN: {len(chunks)} of the corpus, "
              f"{args.max_chunks} per bucket. Not a result.")
    transport = "simulate" if args.simulate else ("local" if args.local else "http")

    cold = None
    if transport == "local":
        resolved, explicit = impl.resolve_device(args.device)
        if resolved == "cpu" and not explicit:
            raise SystemExit(
                "This machine has no cuda or mps device. Chatterbox on CPU is "
                "slower than realtime, so timing it would measure the machine "
                "rather than the model.\n"
                "  Run with --device cpu only if you mean to time the CPU.")
        cold = prepare_local(resolved)
        print(banner(cold["device"]))
    if transport == "simulate":
        print(LABEL_SIMULATED + ": no model, no network.")

    print(f"{len(chunks)} chunks x {args.trials} trials = "
          f"{len(chunks) * args.trials} synthesises\n")
    rows = run(chunks, transport, args.trials, args.endpoint, args.device)
    summary = summarise(rows)
    if cold:
        summary["cold_start"] = cold
        print(f"\ncold start (excluded from every number below): "
              f"model load {cold['load_seconds']:.1f}s, "
              f"warmup generate {cold['warmup_seconds']:.2f}s")

    def fmt(value, unit="s"):
        return f"{value:.3f}{unit}" if value is not None else "—"

    print(f"\n{'bucket':<11}{'n':>4}{'words':>7}{'model':>10}{'delivery':>10}"
          f"{'playable':>10}{'audio':>9}{'xRT':>7}")
    for name, stats in summary["buckets"].items():
        rtf = stats.get("realtime_factor_p50")
        print(f"{name:<11}{stats['n']:>4}{stats['words_median']:>7.0f}"
              f"{fmt(stats.get('model_seconds_p50')):>10}"
              f"{fmt(stats.get('delivery_seconds_p50')):>10}"
              f"{fmt(stats['first_playable_p50']):>10}"
              f"{fmt(stats.get('audio_seconds_p50')):>9}"
              f"{(f'{rtf:.1f}x' if rtf else '—'):>7}")
    if summary["collapsed_marks"]:
        print(f"\ncollapsed marks: {', '.join(summary['collapsed_marks'])}"
              "  (one-shot engine / non-streaming contract)")

    if args.out:
        path = pathlib.Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"label": LABEL_SIMULATED if transport == "simulate" else (
                banner(cold["device"]) if cold else "REMOTE ENDPOINT"),
             "smoke_run": bool(args.max_chunks),
             "transport": transport, "simulated": transport == "simulate",
             "is_production_latency": False,
             "summary": summary, "rows": rows}, indent=2), encoding="utf-8")
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
