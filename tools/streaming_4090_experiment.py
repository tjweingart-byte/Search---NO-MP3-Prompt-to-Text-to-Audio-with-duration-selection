#!/usr/bin/env python3
"""The real FAM pipeline on a 4090: Claude and Chatterbox running at once.

The previous combined run measured a **sequential** pipeline. It detected the
first speakable chunk at about 2.4s and then waited for `claude_complete`
before starting TTS. That is not FAM's architecture: `pipeline.py` pumps
`script_generator.stream_sentences` into a bounded queue and speaks each
sentence while the model is still writing.

This runs that shape for real - the same Exa call, the same Claude model, the
same Chatterbox Base weights, the same reference voice - and **asserts** the
overlap rather than assuming it. A run whose first TTS started at or after
Claude finished is a failed run, and says so with a non-zero exit.

    python3 tools/streaming_4090_experiment.py --preflight --device cuda \\
        --reference experiments/references/working/reference_3.wav

    python3 tools/streaming_4090_experiment.py --device cuda \\
        --reference experiments/references/working/reference_3.wav
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments import concurrent_pipeline as cp                # noqa: E402
from experiments import cost as cost_model                       # noqa: E402
from experiments import pipeline_probe as probe                  # noqa: E402
from experiments import voice_bakeoff as bake                    # noqa: E402
from experiments import voice_identity as identity               # noqa: E402
from tools.combined_4090_experiment import (LABEL, check,        # noqa: E402
                                            load_model, verify_claude,
                                            verify_exa)

DEFAULT_QUERY = "What happened in the markets this week?"
MINUTES = 3

#: A stub run exercises the wiring and measures nothing. It says so in every
#: artefact it writes, and its output directory has to be named for it.
STUB_LABEL = ("STUB - NO MODEL, NO API CALLS, NO GPU. Proves the wiring and "
              "the event ordering. NOT A RESULT.")


# --------------------------------------------------------------------------
# the model stage, with a first-token mark and nothing else changed
# --------------------------------------------------------------------------
class _TimedStream:
    """Production's stream, with one mark on the first text delta.

    `stream_sentences` yields sentences, so time-to-first-*token* is invisible
    from outside it. Rather than reimplement the chunker to see it - which
    would measure a copy of production instead of production - the client is
    wrapped and the real generator is left alone.
    """

    def __init__(self, inner, log):
        self._inner, self._log = inner, log

    @property
    def text_stream(self):
        async def deltas():
            first = True
            async for delta in self._inner.text_stream:
                if first:
                    self._log.mark("claude_ttft", {"chars": len(delta)})
                    first = False
                yield delta
        return deltas()

    async def get_final_message(self):
        return await self._inner.get_final_message()


class _TimedStreamContext:
    def __init__(self, inner, log):
        self._inner, self._log = inner, log

    async def __aenter__(self):
        return _TimedStream(await self._inner.__aenter__(), self._log)

    async def __aexit__(self, *exc):
        return await self._inner.__aexit__(*exc)


class _TimedMessages:
    def __init__(self, inner, log):
        self._inner, self._log = inner, log

    def stream(self, **kwargs):
        return _TimedStreamContext(self._inner.stream(**kwargs), self._log)


class TimedClient:
    def __init__(self, inner, log):
        self._inner = inner
        self.messages = _TimedMessages(inner.messages, log)

    async def close(self):
        await self._inner.close()


def build_generator(log, model_name: str = ""):
    """Production's ScriptGenerator, with the experiment's settings override.

    The override is a local `dataclasses.replace` restored in the caller's
    `finally`, exactly as `experiments/generate.py` does it. The cache is off:
    a cached script would return in milliseconds and look like a win.
    """
    import script_generator
    from anthropic_client import build_async_client
    from config import settings
    from script_generator import ScriptGenerator

    patched = dataclasses.replace(settings, search_mode="never",
                                  cache_enabled=False,
                                  model=model_name or settings.model)
    original = script_generator.settings
    script_generator.settings = patched
    generator = ScriptGenerator()
    generator.client = TimedClient(build_async_client(patched.anthropic_api_key),
                                   log)
    return generator, script_generator, original, patched


# --------------------------------------------------------------------------
# the three stages, real or stubbed
# --------------------------------------------------------------------------
class RealStages:
    """Exa, Claude and Chatterbox, exactly as the product uses them."""

    stub = False

    def __init__(self, model, reference: pathlib.Path, args):
        self.model, self.reference, self.args = model, reference, args

    async def search(self, query: str) -> dict:
        from experiments.adapters.exa_impl import run_search

        return await run_search(query)

    def sentences(self, query: str, packet: str, log):
        """Production's chunker, over production's request, unmodified."""
        from experiments.generate import with_packet
        from script_generator import plan_episode

        generator, module, original, patched = build_generator(log, self.args.model)
        plan = plan_episode(with_packet(query, packet), MINUTES, search=False)
        log.mark("claude_start", {"model": patched.model,
                                  "max_words": plan.max_words})
        return generator.stream_sentences(plan), generator, module, original

    async def synth(self, text: str):
        import torch

        def blocking():
            with torch.inference_mode():
                wav = self.model.generate(text,
                                          audio_prompt_path=str(self.reference),
                                          **identity.GENERATION)
            # .cpu() synchronises, so the thread returns only once the audio
            # exists rather than once the kernels are queued.
            return (wav.squeeze(0).detach().cpu().numpy(),
                    int(getattr(self.model, "sr", 24000)))

        # A thread, exactly as tts.py hands Piper's blocking call to one. This
        # is what lets the Claude stream keep advancing during synthesis.
        return await asyncio.to_thread(blocking)


class StubStages:
    """The same code path with nothing real behind it.

    For proving the wiring on a laptop. Everything it writes is stamped
    `stub: true` and carries a label saying so, because a demo mode that
    announced itself quietly once cost this project a whole session.
    """

    stub = True
    SENTENCES = [
        "The oldest working clock in Europe has no face.",
        "It was built to ring, not to be read.",
        "Salisbury Cathedral has kept it turning since about thirteen eighty-six.",
        "Nobody there needed to know the minute; they needed to know when to pray.",
        "The face came later, when knowing the hour became somebody's business.",
    ]

    def __init__(self, model, reference, args):
        self.args = args

    async def search(self, query: str) -> dict:
        await asyncio.sleep(0.4)
        return {"context": "SOURCE 1\nTitle: stub\n", "sources": [],
                "results_returned": 0, "packet_chars": 24,
                "search_type": "stub", "remote_seconds": 0.4, "cost": 0.0}

    def sentences(self, query: str, packet: str, log):
        async def stream():
            await asyncio.sleep(0.3)
            log.mark("claude_ttft", {"chars": 1})
            for sentence in self.SENTENCES:
                await asyncio.sleep(0.25)
                yield sentence

        log.mark("claude_start", {"model": "stub"})
        return stream(), None, None, None

    async def synth(self, text: str):
        rate = 24000
        await asyncio.sleep(0.35)
        return [0] * int(rate * 0.42 * len(text.split())), rate


# --------------------------------------------------------------------------
# one request, start to finish, with the two halves overlapping
# --------------------------------------------------------------------------
async def run_once(stages, out: pathlib.Path, label: str, args,
                   log: probe.EventLog) -> dict:
    query = args.query
    log.cannot("first_exa_result",
               "exa_impl makes one blocking search_and_contents call, so there "
               "is no observable point between request and complete")
    log.cannot("first_playable_audio_mid_chunk",
               "Chatterbox Base is one-shot - generate() returns a finished "
               "waveform - so the first playable moment is the first chunk's "
               "generation-complete, not a point inside it")

    log.mark("request_start")
    log.mark("exa_start")
    reply = await stages.search(query)
    packet = reply.get("context") or ""
    log.mark("exa_complete", {"packet_chars": len(packet),
                              "results": reply.get("results_returned"),
                              "sources": reply.get("sources")})

    sentences, generator, module, original = stages.sentences(query, packet, log)
    try:
        run = await cp.run_concurrent(sentences, stages.synth, log,
                                      queue_depth=args.queue_depth,
                                      max_chunks=args.max_chunks)
    finally:
        if module is not None:
            module.settings = original
        if generator is not None:
            await generator.client.close()

    problems = cp.concurrency_problems(run)
    written = write_audio(run, out, label)
    summary = cp.summarise(run)
    playback = cp.playback_analysis(run)

    exa_cost = float(reply.get("cost") or cost_model.EXA_COST_PER_SEARCH)
    return {
        "run": label, "query": query, "stub": stages.stub,
        "concurrency_problems": problems,
        "concurrent": not problems,
        "summary": summary,
        "playback": playback,
        "chunks": [c.to_dict() for c in run.chunks],
        "first_emitted_chunk": run.first_emitted,
        "full_script": run.full_script,
        "queue_depth_series": run.queue_depth,
        "peak_queue_depth": run.peak_queue_depth,
        "backpressure_seconds": run.backpressure_seconds,
        "audio": written,
        "exa": {k: reply.get(k) for k in
                ("sources", "results_returned", "packet_chars", "search_type",
                 "remote_seconds")},
        "cost_usd": {"exa": exa_cost,
                     "anthropic_note": "usage is not exposed by "
                                       "stream_sentences; Exa cost only"},
        "gpu": probe.gpu_memory(),
        "log": log.as_dict(),
    }


def write_audio(run: cp.PipelineRun, out: pathlib.Path, label: str) -> dict:
    """Every chunk in order, plus one file to listen to.

    The concatenation lays production's own `SENTENCE_GAP` between chunks, so
    what you hear is what the app would have played rather than a tighter edit
    of it.
    """
    import numpy as np

    folder = out / "audio" / label
    folder.mkdir(parents=True, exist_ok=True)
    names = []
    for chunk, samples in zip(run.chunks, run.samples):
        name = f"chunk_{chunk.index:02d}.wav"
        bake.write_wav(folder / name, samples, run.sample_rate)
        names.append(name)

    gap = np.zeros(int(cp.SENTENCE_GAP * run.sample_rate), dtype=np.float32)
    pieces = []
    for index, samples in enumerate(run.samples):
        if index:
            pieces.append(gap)
        pieces.append(np.asarray(samples, dtype=np.float32))
    episode = (np.concatenate(pieces) if pieces
               else np.zeros(0, dtype=np.float32))
    bake.write_wav(out / "audio" / f"{label}_episode.wav", episode,
                   run.sample_rate)

    (out / "audio" / f"{label}_script.txt").write_text(
        "\n\n".join(f"[{c.index:02d}] {c.text}" for c in run.chunks) + "\n",
        encoding="utf-8")
    return {"chunk_files": names, "folder": f"audio/{label}",
            "episode": f"audio/{label}_episode.wav",
            "episode_seconds": len(episode) / run.sample_rate
            if run.sample_rate else 0.0}


# --------------------------------------------------------------------------
def preflight(args) -> int:
    print("\npreflight - free except two tiny API calls (about $0.006 total)\n")
    ok = True

    import subprocess

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
        from chatterbox.tts import ChatterboxTTS                 # noqa: F401

        ok &= check("chatterbox.tts importable", True)
    except Exception as exc:
        ok &= check("chatterbox.tts importable", False, str(exc))

    try:
        import perth

        ok &= check("perth watermarker loadable",
                    getattr(perth, "PerthImplicitWatermarker", None) is not None,
                    "watermarking is required and is not bypassed")
    except Exception as exc:
        ok &= check("perth importable", False, str(exc))

    reference = pathlib.Path(args.reference)
    ok &= check("reference voice present", reference.exists(), str(reference))

    try:
        from script_generator import ScriptGenerator

        ok &= check("production chunker importable",
                    hasattr(ScriptGenerator, "stream_sentences"),
                    "script_generator.ScriptGenerator.stream_sentences")
    except Exception as exc:
        ok &= check("production chunker importable", False,
                    f"{type(exc).__name__}: {exc}")

    ok &= check("queue and gap come from production",
                cp.QUEUE_DEPTH > 0 and cp.SENTENCE_GAP > 0,
                f"QUEUE_DEPTH={cp.QUEUE_DEPTH}, SENTENCE_GAP={cp.SENTENCE_GAP}s")

    exa_ok, exa_detail = verify_exa()
    ok &= check("Exa answers a real search", exa_ok, exa_detail)
    claude_ok, claude_detail = verify_claude()
    ok &= check("Claude answers a real request", claude_ok, claude_detail)

    print()
    if ok:
        print("  preflight ok - the streaming experiment can run.\n")
        return 0
    print("  preflight FAILED. Nothing was generated and no long run started.\n")
    if not exa_ok:
        print("    Exa:    export EXA_API_KEY='...'  in this shell")
    if not claude_ok:
        print("    Claude: export ANTHROPIC_API_KEY='...'  in this shell")
    print()
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reference", default="",
                        help="the selected FAM reference voice, e.g. "
                             "experiments/references/working/reference_3.wav")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--model", default="",
                        help="override the configured Claude model; the "
                             "default is production's")
    parser.add_argument("--queue-depth", type=int, default=cp.QUEUE_DEPTH,
                        dest="queue_depth",
                        help=f"FIFO depth; production uses {cp.QUEUE_DEPTH}")
    parser.add_argument("--max-chunks", type=int, default=0, dest="max_chunks",
                        help="stop after N chunks; 0 means the whole episode")
    parser.add_argument("--runs", type=int, default=2,
                        help="cold, then warm. 1 runs cold only.")
    parser.add_argument("--out", default="")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="run the whole path with no model, no GPU and no "
                             "API calls, to prove the wiring and the event "
                             "ordering. Writes into a directory named _STUB "
                             "and stamps every artefact NOT A RESULT.")
    return parser


async def _main(args, out: pathlib.Path) -> dict:
    reference = pathlib.Path(args.reference)
    stub = bool(args.dry_run)
    results = {
        "label": STUB_LABEL if stub else LABEL,
        "stub": stub,
        "is_production_latency": False,
        "experiment": "concurrent streaming pipeline - Claude and Chatterbox "
                      "overlapping, as pipeline.py does it",
        "device": "none (stub)" if stub else args.device,
        "reference": "none (stub)" if stub else reference.name,
        "generation_settings": identity.GENERATION,
        "queue_depth": args.queue_depth,
        "sentence_gap_seconds": cp.SENTENCE_GAP,
        "chunker": "stubbed sentence stream" if stub else
                   "script_generator.ScriptGenerator.stream_sentences "
                   "(production, unmodified)",
        "query": args.query, "minutes": MINUTES,
        "started": time.time(), "runs": [],
    }

    labels = ["cold", "warm"][:max(1, args.runs)]
    model = None
    for index, label in enumerate(labels):
        log = probe.EventLog()
        if index == 0 and not stub:
            log.mark("process_start")
            log.mark("model_load_start")
            model, cold_load, before, after = load_model(args.device)
            log.mark("model_load_complete", {"seconds": cold_load})
            results["cold_model_load_seconds"] = cold_load
            results["gpu_before_load"], results["gpu_after_load"] = before, after
            print(f"  cold model load    {cold_load:.2f}s\n")
        stages = (StubStages if stub else RealStages)(model, reference, args)
        print(f"--- run: {label} ---\n")
        record = await run_once(stages, out, label, args, log)
        _report(record)
        results["runs"].append(record)

    results["gpu_peak"] = probe.gpu_memory()
    results["finished"] = time.time()
    return results


def _report(record: dict) -> None:
    summary, playback = record["summary"], record["playback"]
    print(f"\n    chunks {summary['chunks']}, "
          f"{summary['audio_seconds']:.1f}s of audio, "
          f"{summary['realtime_factor_overall']:.2f}x realtime overall")
    for name in ("exa_latency", "claude_ttft", "claude_to_first_chunk",
                 "first_chunk_tts_seconds", "search_to_first_listen",
                 "claude_total", "search_to_complete_audio",
                 "overlap_seconds"):
        value = summary.get(name)
        print(f"    {name:<28}" + (f"{value:7.3f}s" if value is not None
                                   else "      -"))
    print(f"    peak queue depth            {summary['peak_queue_depth']:>7}")
    print(f"    backpressure                "
          f"{summary['backpressure_seconds']:7.3f}s")
    print(f"    kept ahead of playback      {str(playback['kept_ahead']):>7}"
          + (f"   (stalled {playback['stall_seconds']:.2f}s across "
             f"{len(playback['underruns'])} chunk(s))"
             if playback["underruns"] else ""))
    if record["concurrency_problems"]:
        print("\n    CONCURRENCY ASSERTION FAILED:")
        for problem in record["concurrency_problems"]:
            print(f"      - {problem}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if not args.dry_run and not args.reference:
        raise SystemExit("--reference is required: name the selected FAM "
                         "reference voice (reference_3)")
    if args.preflight:
        return preflight(args)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if args.out:
        out = pathlib.Path(args.out)
    elif args.dry_run:
        out = pathlib.Path("experiments/results") / f"streaming_STUB_{stamp}"
    else:
        out = pathlib.Path("experiments/results") / f"streaming_4090_{stamp}"
    # A stub run may only ever land somewhere named for what it is. Announcing
    # a demo mode is not enough if the thing it writes outlives the run.
    if args.dry_run and "STUB" not in out.name.upper():
        raise SystemExit(f"a --dry-run must write to a directory whose name "
                         f"contains STUB; {out.name!r} does not.")
    if not args.dry_run and "STUB" in out.name.upper():
        raise SystemExit(f"{out.name!r} is a stub directory; a real run may "
                         "not write into it.")
    if not args.dry_run and not pathlib.Path(args.reference).exists():
        raise SystemExit(f"no reference voice at {args.reference}")
    if (out / "results.json").exists():
        raise SystemExit(f"{out}/results.json already exists. Nothing is "
                         "overwritten - use a new --out directory.")
    out.mkdir(parents=True, exist_ok=True)
    print(f"\nresults -> {out}\n")
    if args.dry_run:
        print(f"  {STUB_LABEL}\n")

    results = asyncio.run(_main(args, out))

    # Everything is written before the run is allowed to fail, so a failed
    # assertion still leaves the evidence for why.
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str),
                                      encoding="utf-8")
    (out / "events.jsonl").write_text("\n".join(
        json.dumps({"run": record["run"], **event})
        for record in results["runs"]
        for event in record["log"]["events"]) + "\n", encoding="utf-8")
    (out / "ANALYSIS.md").write_text(_analysis(results), encoding="utf-8")

    print(f"\nwrote {out}/")
    for name in ("results.json", "ANALYSIS.md", "events.jsonl", "audio/"):
        print(f"  {name}")

    if results["stub"]:
        print(f"\n  {STUB_LABEL}")

    failed = [r for r in results["runs"] if r["concurrency_problems"]]
    if failed:
        print("\n  THIS RUN DID NOT OVERLAP. The results are written; the "
              "exit code is non-zero on purpose.\n")
        return 2
    if results["stub"]:
        print("\n  Overlap asserted and held on the stub. The wiring and the "
              "event ordering are proved; nothing was measured.\n")
    else:
        print("\n  Overlap asserted and held. Copy this directory to your Mac "
              "before terminating the pod.\n")
    return 0


def _analysis(results: dict) -> str:
    lines = [
        "# The FAM pipeline, stubbed: wiring and event ordering only"
        if results["stub"] else
        "# The real FAM pipeline on an RTX 4090: Claude and Chatterbox at once",
        "",
        (f"**{STUB_LABEL}**" if results["stub"] else f"**{LABEL}.**")
        + " One machine, one request at a time, no concurrency "
        "between requests and no queueing. These are development numbers.", "",
        f"- chunker: `{results['chunker']}`",
        f"- FIFO depth: {results['queue_depth']} (production's `QUEUE_DEPTH`)",
        f"- silence between chunks: {results['sentence_gap_seconds']}s "
        "(production's `SENTENCE_GAP`)",
        f"- reference voice: `{results['reference']}`",
        f"- settings: `{results['generation_settings']}`",
        f"- query: {results['query']!r}",
        ("- cold model load: not applicable to a stub run"
         if results["stub"] else
         f"- cold model load: "
         f"**{results.get('cold_model_load_seconds', float('nan')):.2f}s**"), "",
        "## Did it actually overlap?", "",
    ]
    for record in results["runs"]:
        summary = record["summary"]
        if record["concurrent"]:
            lines.append(
                f"- **{record['run']}: yes.** First TTS started "
                f"{summary['overlap_seconds']:.2f}s before Claude finished, on "
                f"the first chunk the chunker emitted "
                f"({record['chunks'][0]['words']} words).")
        else:
            lines.append(f"- **{record['run']}: NO.** "
                         + "; ".join(record["concurrency_problems"]))
    lines.append("")

    for record in results["runs"]:
        summary, playback = record["summary"], record["playback"]
        lines += [f"## Run: {record['run']}", "",
                  "| measure | seconds |", "|---|---|"]
        for name in ("exa_latency", "claude_ttft", "claude_to_first_chunk",
                     "first_chunk_tts_seconds", "search_to_first_listen",
                     "claude_total", "search_to_complete_audio",
                     "overlap_seconds", "backpressure_seconds"):
            value = summary.get(name)
            lines.append(f"| `{name}` | "
                         + ("-" if value is None else f"{value:.3f}s") + " |")
        lines += ["",
                  f"{summary['chunks']} chunks, {summary['words']} words, "
                  f"{summary['audio_seconds']:.1f}s of audio, "
                  f"{summary['realtime_factor_overall']:.2f}x realtime overall. "
                  f"Peak queue depth {summary['peak_queue_depth']}.", ""]
        if playback["kept_ahead"]:
            lines.append(f"**Synthesis kept ahead of playback** with "
                         f"{playback['headroom_seconds']:.1f}s of headroom at "
                         "the end. No dead air.")
        else:
            lines.append(
                f"**Synthesis fell behind playback**: "
                f"{playback['stall_seconds']:.2f}s of dead air across "
                f"{len(playback['underruns'])} chunk(s), worst "
                f"{playback['max_stall_seconds']:.2f}s.")
        lines += ["", "| chunk | words | ready | tts start | tts done | "
                      "audio | gen | realtime | queued |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for chunk in record["chunks"]:
            lines.append(
                f"| {chunk['index']:02d} | {chunk['words']} | "
                f"{chunk['ready_at']:.2f}s | {chunk['tts_start']:.2f}s | "
                f"{chunk['tts_complete']:.2f}s | {chunk['audio_seconds']:.2f}s "
                f"| {chunk['generate_seconds']:.2f}s | "
                f"{chunk['realtime_factor']:.2f}x | "
                f"{chunk['waited_in_queue']:.2f}s |")
        lines += ["", f"Listen: `{record['audio']['episode']}` "
                      f"({record['audio']['episode_seconds']:.1f}s), chunks in "
                      f"`{record['audio']['folder']}/`.", ""]
        if record["log"]["unavailable"]:
            lines += ["**Not measurable in this run:**", ""]
            for name, why in record["log"]["unavailable"].items():
                lines.append(f"- `{name}` - {why}")
            lines.append("")

    lines += ["## Reading this against the previous run", "",
              "The combined run measured the same three stages in sequence: "
              "the first speakable chunk was *detected* early and then waited "
              "for `claude_complete`. Compare `search_to_first_listen` between "
              "the two. The difference is what the architecture is worth, and "
              "it costs no extra compute - only a different order.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
