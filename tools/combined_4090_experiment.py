#!/usr/bin/env python3
"""Chatterbox Base on its own, and the whole FAM request, in one 4090 session.

Two experiments, one pod, results kept apart:

**A. Chatterbox Base alone.** Cold model load, the first generation after that
load, then warm generations across the short / medium / long buckets of the
real first-chunk corpus. Audio duration, wall-clock, realtime factor, GPU
memory. Written to `bench_chatterbox_base.json`. **Turbo's numbers are never
touched**; this lands in a directory of its own.

**B. The whole request.** `user request -> Exa -> Claude -> Chatterbox Base ->
playable audio`, on one monotonic clock, run cold and then immediately again
warm. Written to `pipeline.json`.

The real integrations are used. If Exa or Claude is not wired the run stops and
says what is missing rather than substituting a stub, because a stubbed stage
in a latency waterfall is a number someone will later quote.

A and B run as **separate processes** so each gets a genuine cold model load -
there is only one cold load per process, and both experiments need one.

    python3 tools/combined_4090_experiment.py --preflight --device cuda \\
        --reference experiments/references/working/reference_1.wav

    STAMP=$(date -u +%Y%m%dT%H%M%SZ)
    OUT=experiments/results/combined_4090_$STAMP
    python3 tools/combined_4090_experiment.py --experiment a --out "$OUT" ...
    python3 tools/combined_4090_experiment.py --experiment b --out "$OUT" ...
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments import cost as cost_model                      # noqa: E402
from experiments import pipeline_probe as probe                 # noqa: E402
from experiments import voice_bakeoff as bake                   # noqa: E402
from experiments import voice_identity as identity              # noqa: E402

#: The realistic request. Chosen to need today's facts, so Exa is genuinely
#: exercised rather than answered from what the model already knows.
DEFAULT_QUERY = "What happened in the markets this week?"

#: The validated first-chunk corpus - the same one every earlier Chatterbox
#: run used, so "short" and "long" mean here what they meant there.
CORPUS = "experiments/chunks/first_chunks.json"

BUCKETS = ("short", "medium", "long")

#: The first speakable chunk rule the benchmark corpus was built under.
FIRST_CHUNK_WORDS = 25

#: Episode length asked of Claude. Three minutes is FAM's default.
MINUTES = 3.0

LABEL = "RTX 4090 / DEVELOPMENT BENCHMARK - not production latency"


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------
def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f"  {detail}" if detail else ""))
    return bool(ok)


def verify_exa() -> tuple:
    """A real search, not a check that a variable is set.

    "A key is set" is not "the key works" - the rule this project learned the
    expensive way. One cheap call settles it. Costs about $0.005.
    """
    if not os.environ.get("EXA_API_KEY", "").strip():
        return False, "EXA_API_KEY is not set in this shell"
    try:
        from experiments.adapters.exa_impl import run_search

        reply = asyncio.run(run_search("what happened in the markets this week",
                                       num_results=2, packet_sources=1,
                                       highlights_per_source=1))
        packet = reply.get("context") or ""
        return bool(packet.strip()), (f"{len(packet)} chars, "
                                      f"{reply.get('results_returned')} results")
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def verify_claude() -> tuple:
    """One eight-token completion. Same reasoning as Exa. Costs under $0.001."""
    try:
        from config import settings

        if not settings.anthropic_api_key:
            return False, "ANTHROPIC_API_KEY is not set in this shell"
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        client.messages.create(model=settings.model, max_tokens=8,
                               messages=[{"role": "user", "content": "Say OK."}])
        return True, f"{settings.model} answered"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def load_corpus(path: pathlib.Path, per_bucket: int) -> list:
    """A few real chunks from each length bucket."""
    data = json.loads(path.read_text(encoding="utf-8"))
    chunks = data.get("chunks") or []
    picked, seen = [], {}
    for chunk in chunks:
        bucket = chunk.get("bucket")
        if bucket in BUCKETS and seen.get(bucket, 0) < per_bucket:
            seen[bucket] = seen.get(bucket, 0) + 1
            picked.append(chunk)
    missing = [b for b in BUCKETS if not seen.get(b)]
    if missing:
        raise SystemExit(f"corpus at {path} has no {', '.join(missing)} chunks")
    return picked


def preflight(args) -> int:
    print("\npreflight - free except two tiny API calls (about $0.006 total)\n")
    ok = True

    gpu = ""
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True,
                             timeout=20).stdout.strip().splitlines()[0]
    except Exception as exc:
        gpu = f"(no nvidia-smi: {exc})"
    ok &= check("GPU is an RTX 4090", "4090" in gpu, gpu)

    try:
        import torch

        ok &= check("torch sees CUDA", torch.cuda.is_available(), torch.__version__)
    except ImportError:
        ok &= check("torch importable", False,
                    "pip install -r experiments/requirements-chatterbox.txt")

    try:
        from chatterbox.tts import ChatterboxTTS                # noqa: F401

        ok &= check("chatterbox.tts importable", True)
    except Exception as exc:
        ok &= check("chatterbox.tts importable", False, str(exc))

    try:
        import perth

        ok &= check("perth watermarker loadable",
                    getattr(perth, "PerthImplicitWatermarker", None) is not None,
                    "watermarking is required; run tools/diagnose_chatterbox.py")
    except Exception as exc:
        ok &= check("perth importable", False, str(exc))

    reference = pathlib.Path(args.reference)
    ok &= check("reference voice present", reference.exists(), str(reference))

    corpus = pathlib.Path(args.corpus)
    if corpus.exists():
        try:
            picked = load_corpus(corpus, args.per_bucket)
            ok &= check("first-chunk corpus has all three buckets", True,
                        f"{len(picked)} chunks selected")
        except SystemExit as exc:
            ok &= check("first-chunk corpus usable", False, str(exc))
    else:
        ok &= check("first-chunk corpus present", False,
                    f"{corpus} is git-ignored; pack it with tools/pack_for_pod.py")

    try:
        from experiments.generate import ClaudeGenerator             # noqa: F401
        from experiments.harness import first_chunk_ready            # noqa: F401

        ok &= check("the production model stage imports", True,
                    "experiments/generate.py -> script_generator")
    except Exception as exc:
        ok &= check("the production model stage imports", False,
                    f"{type(exc).__name__}: {exc}")

    exa_ok, exa_detail = verify_exa()
    ok &= check("Exa answers a real search", exa_ok, exa_detail)
    claude_ok, claude_detail = verify_claude()
    ok &= check("Claude answers a real request", claude_ok, claude_detail)

    print()
    if ok:
        print("  preflight ok - the combined experiment can run.\n")
        return 0
    print("  preflight FAILED. Nothing was generated and no long run started.\n")
    if not exa_ok:
        print("    Exa:    export EXA_API_KEY='...'  in this shell")
    if not claude_ok:
        print("    Claude: export ANTHROPIC_API_KEY='...'  in this shell")
    print()
    return 1


# --------------------------------------------------------------------------
# shared
# --------------------------------------------------------------------------
def load_model(device: str):
    """The cold load, timed. Returns (model, seconds, gpu-before, gpu-after)."""
    import torch

    from chatterbox.tts import ChatterboxTTS

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    before = probe.gpu_memory()
    started = time.perf_counter()
    model = ChatterboxTTS.from_pretrained(device=device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    actual = str(getattr(model, "device", device))
    if actual.split(":")[0] != device.split(":")[0]:
        raise SystemExit(f"asked for {device!r}, model loaded on {actual!r}")
    return model, seconds, before, probe.gpu_memory()


def synthesise(model, text: str, reference: pathlib.Path):
    """One generation. Returns (samples, sample rate, seconds)."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        wav = model.generate(text, audio_prompt_path=str(reference),
                             **identity.GENERATION)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    samples = wav.squeeze(0).detach().cpu().numpy()
    del wav
    return samples, int(getattr(model, "sr", 24000)), seconds


# --------------------------------------------------------------------------
# experiment A: Chatterbox Base alone
# --------------------------------------------------------------------------
def experiment_a(args, out: pathlib.Path) -> dict:
    reference = pathlib.Path(args.reference)
    chunks = load_corpus(pathlib.Path(args.corpus), args.per_bucket)

    print("A. Chatterbox Base alone\n")
    model, cold_load, before, after = load_model(args.device)
    print(f"  cold model load    {cold_load:.2f}s")

    _, _, first_generation = synthesise(model, chunks[0]["text"], reference)
    print(f"  first generation   {first_generation:.2f}s   "
          f"(kept separate - it is not a warm number)")

    print("\n  warm generations")
    rows = []
    for chunk in chunks:
        for trial in range(args.warm_trials):
            samples, rate, seconds = synthesise(model, chunk["text"], reference)
            audio_seconds = len(samples) / rate
            rows.append({
                "bucket": chunk["bucket"], "words": chunk.get("words")
                or len(chunk["text"].split()), "trial": trial,
                "generate_seconds": seconds, "audio_seconds": audio_seconds,
                "realtime_factor": audio_seconds / seconds if seconds else None,
                "gpu": probe.gpu_memory(),
            })
            print(f"    {chunk['bucket']:<8}t{trial}  {seconds:6.3f}s for "
                  f"{audio_seconds:5.1f}s audio  "
                  f"({audio_seconds / seconds:5.2f}x realtime)")
            del samples

    summary = {}
    for bucket in BUCKETS:
        mine = [r for r in rows if r["bucket"] == bucket]
        if mine:
            summary[bucket] = {
                "n": len(mine),
                "words_mean": statistics.mean(r["words"] for r in mine),
                "generate_p50": statistics.median(r["generate_seconds"] for r in mine),
                "audio_p50": statistics.median(r["audio_seconds"] for r in mine),
                "realtime_p50": statistics.median(r["realtime_factor"] for r in mine),
            }

    result = {
        "label": LABEL, "is_production_latency": False,
        "engine": "Chatterbox Base (chatterbox.tts.ChatterboxTTS)",
        "device": args.device, "reference": reference.name,
        "generation_settings": identity.GENERATION,
        "watermarking": "applied by chatterbox; not bypassed",
        "cold_load_seconds": cold_load,
        "first_generation_seconds": first_generation,
        "gpu_before_load": before, "gpu_after_load": after,
        "gpu_peak": probe.gpu_memory(),
        "trials": rows, "by_bucket": summary,
        "finished": time.time(),
    }
    (out / "bench_chatterbox_base.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


# --------------------------------------------------------------------------
# experiment B: the whole request
# --------------------------------------------------------------------------
def run_pipeline(model, reference: pathlib.Path, query: str, out: pathlib.Path,
                 run_label: str, log: probe.EventLog) -> dict:
    """One realistic request end to end, on the clock `log` already holds."""
    from experiments.adapters.exa_impl import run_search
    from experiments.generate import ClaudeGenerator
    from experiments.harness import first_chunk_ready

    log.cannot("first_exa_result",
               "exa_impl makes one blocking search_and_contents call, so there "
               "is no observable point between request and complete")
    log.cannot("first_playable_audio_mid_generation",
               "Chatterbox Base is one-shot - generate() returns a finished "
               "waveform and the package yields nothing - so the first playable "
               "moment is generation-complete for the first chunk")
    log.notes.append(f"run={run_label}; query={query!r}")

    log.mark("request_start")
    log.mark("exa_request_start")
    reply = asyncio.run(run_search(query))
    packet = reply.get("context") or ""
    log.mark("exa_complete", {"packet_chars": len(packet),
                              "results": reply.get("results_returned"),
                              "sources": reply.get("sources")})

    generator = ClaudeGenerator()
    text, first_chunk = "", None

    async def stream() -> None:
        nonlocal text, first_chunk
        log.mark("claude_request_start")
        async for delta in generator.stream(query, MINUTES, context=packet,
                                            search=False):
            if not text:
                log.mark("claude_first_token")
            text += delta
            if first_chunk is None:
                candidate = first_chunk_ready(text, FIRST_CHUNK_WORDS)
                if candidate:
                    first_chunk = candidate
                    log.mark("first_speakable_chunk",
                             {"words": len(candidate.split())})

    asyncio.run(stream())
    log.mark("claude_complete", {"chars": len(text)})
    usage = generator.usage()

    # If the script never produced a sentence ending past the word floor, the
    # chunk rule did not fire. Say so rather than speaking a raw buffer, which
    # is the production fallback and can be mid-sentence (PRODUCT_ISSUES P8).
    chunk = first_chunk
    if chunk is None:
        log.cannot("first_speakable_chunk",
                   "the chunk rule never fired on this script; the whole "
                   "script was synthesised instead")
        chunk = text.strip()

    log.mark("tts_start", {"words": len(chunk.split())})
    samples, rate, _ = synthesise(model, chunk, reference)
    log.mark("first_playable_audio")
    log.mark("audio_complete", {"audio_seconds": len(samples) / rate})

    audio_path = out / "audio" / f"{run_label}.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    bake.write_wav(audio_path, samples, rate)
    (out / "audio" / f"{run_label}.txt").write_text(
        f"QUERY\n{query}\n\nSPOKEN CHUNK\n{chunk}\n\nFULL SCRIPT\n{text}\n",
        encoding="utf-8")

    exa_cost = float(reply.get("cost") or cost_model.EXA_COST_PER_SEARCH)
    claude_cost = cost_model.model_cost(usage.get("model", ""),
                                        usage.get("input_tokens", 0),
                                        usage.get("output_tokens", 0))
    return {
        "run": run_label, "query": query,
        "script_chars": len(text), "spoken_chunk": chunk,
        "spoken_chunk_words": len(chunk.split()),
        "audio_seconds": len(samples) / rate, "audio_file": audio_path.name,
        "exa": {k: reply.get(k) for k in
                ("sources", "results_returned", "packet_chars", "search_type",
                 "remote_seconds")},
        "claude_usage": usage,
        "cost_usd": {"exa": exa_cost, "anthropic": claude_cost,
                     "total": exa_cost + claude_cost},
        "gpu": probe.gpu_memory(),
        "headlines": probe.headlines(log),
        "waterfall": probe.waterfall(log, probe.PIPELINE_SEGMENTS),
        "log": log.as_dict(),
    }


def experiment_b(args, out: pathlib.Path) -> dict:
    reference = pathlib.Path(args.reference)

    print("B. Full FAM request - cold\n")
    cold_log = probe.EventLog()
    cold_log.mark("process_start")
    cold_log.mark("model_load_start")
    model, cold_load, before, after = load_model(args.device)
    cold_log.mark("model_load_complete", {"seconds": cold_load})
    print(f"  cold model load    {cold_load:.2f}s")
    cold = run_pipeline(model, reference, args.query, out, "cold", cold_log)
    _report(cold)

    print("\nB. Full FAM request - warm, immediately after\n")
    warm = run_pipeline(model, reference, args.query, out, "warm",
                        probe.EventLog())
    _report(warm)

    result = {
        "label": LABEL, "is_production_latency": False,
        "device": args.device, "reference": reference.name,
        "generation_settings": identity.GENERATION,
        "query": args.query, "minutes": MINUTES,
        "first_chunk_words": FIRST_CHUNK_WORDS,
        "cold_model_load_seconds": cold_load,
        "gpu_before_load": before, "gpu_after_load": after,
        "gpu_peak": probe.gpu_memory(),
        "cold": cold, "warm": warm,
        "cost_usd_total": cold["cost_usd"]["total"] + warm["cost_usd"]["total"],
        "finished": time.time(),
    }
    (out / "pipeline.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8")
    (out / "events.jsonl").write_text("\n".join(
        json.dumps({"run": run["run"], **event})
        for run in (cold, warm) for event in run["log"]["events"]) + "\n",
        encoding="utf-8")
    return result


def _report(run: dict) -> None:
    for name, value in run["headlines"].items():
        print(f"    {name:<34}"
              + (f"{value:7.3f}s" if value is not None else "      -"))


# --------------------------------------------------------------------------
# analysis, written from whatever sections exist
# --------------------------------------------------------------------------
def analyse(out: pathlib.Path) -> int:
    bench_path, pipeline_path = (out / "bench_chatterbox_base.json",
                                 out / "pipeline.json")
    bench = (json.loads(bench_path.read_text(encoding="utf-8"))
             if bench_path.exists() else None)
    pipeline = (json.loads(pipeline_path.read_text(encoding="utf-8"))
                if pipeline_path.exists() else None)
    if bench is None and pipeline is None:
        raise SystemExit(f"nothing to analyse in {out}")

    results = {"label": LABEL, "is_production_latency": False,
               "chatterbox_alone": bench, "pipeline": pipeline,
               "written": time.time()}
    (out / "results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")
    (out / "ANALYSIS.md").write_text(_analysis(bench, pipeline),
                                     encoding="utf-8")
    print(f"\nwrote {out}/results.json and {out}/ANALYSIS.md")
    return 0


def _analysis(bench, pipeline) -> str:
    lines = ["# Chatterbox Base, and the whole FAM request, on an RTX 4090", "",
             f"**{LABEL}.** One machine, one request at a time, no concurrency "
             "and no queueing. These are development numbers.", ""]

    if bench:
        lines += ["## A. Chatterbox Base alone", "",
                  f"- reference voice: `{bench['reference']}`",
                  f"- settings: `{bench['generation_settings']}`",
                  f"- watermarking: {bench['watermarking']}",
                  f"- **cold model load: {bench['cold_load_seconds']:.2f}s**",
                  f"- **first generation after load: "
                  f"{bench['first_generation_seconds']:.2f}s** - kept separate; "
                  "it is not the warm number", "",
                  "| bucket | words | generate p50 | audio p50 | realtime |",
                  "|---|---|---|---|---|"]
        for bucket, stat in bench["by_bucket"].items():
            lines.append(f"| {bucket} | {stat['words_mean']:.0f} | "
                         f"{stat['generate_p50']:.3f}s | {stat['audio_p50']:.2f}s "
                         f"| {stat['realtime_p50']:.2f}x |")
        gpu = bench.get("gpu_peak") or {}
        if gpu.get("available"):
            lines += ["", f"GPU: {gpu.get('name')}, peak allocated "
                          f"{gpu.get('max_allocated_mb', 0):.0f} MB of "
                          f"{gpu.get('total_mb', 0):.0f} MB."]
        lines.append("")

    if pipeline:
        lines += ["## B. The whole request", "",
                  f"- query: {pipeline['query']!r}",
                  f"- episode length asked of Claude: {pipeline['minutes']} min",
                  f"- first speakable chunk rule: first sentence ending past "
                  f"{pipeline['first_chunk_words']} words",
                  f"- cold model load, measured inside the cold run: "
                  f"**{pipeline['cold_model_load_seconds']:.2f}s**",
                  f"- API cost for both runs: "
                  f"**${pipeline['cost_usd_total']:.4f}**", ""]
        for key, title in (("cold", "Cold"), ("warm", "Warm, immediately after")):
            run = pipeline[key]
            lines += [f"### {title}", "",
                      "| stage | seconds | share of the measured wait |",
                      "|---|---|---|"]
            for row in run["waterfall"]:
                seconds = "-" if row["seconds"] is None else f"{row['seconds']:.3f}s"
                share = "-" if row["share"] is None else f"{row['share'] * 100:.1f}%"
                lines.append(f"| {row['label']} | {seconds} | {share} |")
            lines += ["", "| measure | seconds |", "|---|---|"]
            for name, value in run["headlines"].items():
                lines.append(f"| `{name}` | "
                             + ("-" if value is None else f"{value:.3f}s") + " |")
            lines += ["", f"Spoken chunk: {run['spoken_chunk_words']} words, "
                          f"{run['audio_seconds']:.1f}s of audio "
                          f"(`audio/{run['audio_file']}`).", ""]
            if run["log"]["unavailable"]:
                lines += ["**Not measurable in this run:**", ""]
                for name, why in run["log"]["unavailable"].items():
                    lines.append(f"- `{name}` - {why}")
                lines.append("")

    lines += ["## Where the listener waits", "",
              "Read the cold waterfall top to bottom. The two numbers the "
              "product is judged on are `search_to_first_listen` (the "
              "one-sentence spec) and `search_to_complete_audio`. "
              "`cold_start_to_first_listen` adds the model load, which "
              "production pays once per process and a listener never pays "
              "again.", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", choices=("a", "b", "analyse"),
                        help="a: Chatterbox alone. b: the whole request. "
                             "analyse: write results.json and ANALYSIS.md.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reference", default="",
                        help="the chosen FAM reference voice, e.g. "
                             "experiments/references/working/reference_1.wav")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--corpus", default=CORPUS)
    parser.add_argument("--per-bucket", type=int, default=2, dest="per_bucket")
    parser.add_argument("--warm-trials", type=int, default=3, dest="warm_trials")
    parser.add_argument("--out", default="",
                        help="results directory; defaults to a new timestamped "
                             "one under experiments/results")
    parser.add_argument("--preflight", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.preflight:
        if not args.reference:
            raise SystemExit("--reference is required: name the reference voice "
                             "chosen by the Phase 3 identity bake-off")
        return preflight(args)

    if not args.experiment:
        raise SystemExit("pass --preflight, or --experiment a|b|analyse")

    if args.out:
        out = pathlib.Path(args.out)
    else:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        out = pathlib.Path("experiments/results") / f"combined_4090_{stamp}"
    out.mkdir(parents=True, exist_ok=True)

    if args.experiment == "analyse":
        return analyse(out)

    if not args.reference:
        raise SystemExit("--reference is required: name the reference voice "
                         "chosen by the Phase 3 identity bake-off")
    if not pathlib.Path(args.reference).exists():
        raise SystemExit(f"no reference voice at {args.reference}")

    existing = out / ("bench_chatterbox_base.json" if args.experiment == "a"
                      else "pipeline.json")
    if existing.exists():
        raise SystemExit(f"{existing} already exists. Nothing is overwritten - "
                         "use a new --out directory.")

    print(f"\nresults -> {out}\n")
    if args.experiment == "a":
        experiment_a(args, out)
    else:
        experiment_b(args, out)
    analyse(out)
    print("\n  Copy this directory to your Mac before terminating the pod.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
