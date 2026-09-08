#!/usr/bin/env python3
"""Phase 6: Claude decoupled from Chatterbox, and speech-sized chunks.

Phase 5 proved the overlap and then reported `claude_total = 66.668s` with
`backpressure = 64.273s` inside it. That number is not Claude's. The TTS queue
was bounded at 4 and sat directly under the Claude reader, so when Chatterbox
fell behind, the reader stopped pulling.

Phase 6 puts a cheap character-bounded script buffer between them and an
assembler after it, so:

* Claude is read to completion regardless of Chatterbox's backlog, and
  `claude_stream_seconds` finally means what it says;
* the opening chunk still goes out the instant a speakable thought exists;
* later sentences are batched into speech-sized chunks instead of one
  synthesis call per fragment.

    python3 tools/phase6_experiment.py --preflight --device cuda \\
        --reference experiments/references/working/reference_3.wav

    python3 tools/phase6_experiment.py --device cuda \\
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

from experiments import cost as cost_model                       # noqa: E402
from experiments import decoupled_pipeline as dp                 # noqa: E402
from experiments import pipeline_probe as probe                  # noqa: E402
from experiments import voice_bakeoff as bake                    # noqa: E402
from experiments import voice_identity as identity               # noqa: E402
from experiments.speech_assembler import AssemblyPolicy          # noqa: E402
from tools.combined_4090_experiment import LABEL, load_model     # noqa: E402
from tools.streaming_4090_experiment import (RealStages,         # noqa: E402
                                             STUB_LABEL,
                                             StubStages,
                                             preflight)

#: Phase 5's warm figures, quoted by the user and used only as the comparison
#: column. They are not re-measured here and are labelled as reported.
PHASE5_WARM = {
    "exa_latency": 0.464, "claude_ttft": 1.039,
    "claude_to_first_speech_chunk": 1.435,
    "first_chunk_tts_seconds": 2.757,
    "search_to_first_listen": 4.659,
    "claude_stream_seconds_reported": 66.668,
    "claude_reader_blocked_seconds": 64.273,
    "search_to_complete_audio": 95.093,
    "peak_tts_queue_depth": 4, "playback_stalls": 0,
    "note": "as reported from the Phase 5 warm run; the raw results.json was "
            "not available in this repository, so nothing here was re-derived "
            "from it",
}

#: The primary success criterion, stated as a number so the report cannot fudge it.
FIRST_LISTEN_BUDGET = 5.0


class StubStagesLong(StubStages):
    """The stub, lengthened and calibrated so it exercises what it claims to.

    A five-sentence stub with an instant voice never fills a queue of four, so
    it would "prove" the decoupling holds in the one case where nothing tests
    it. This one is long enough to saturate, and its voice follows the measured
    4090 curve rather than a made-up constant:

        generate(words) = 0.1086 * words - 0.81   (least squares, R2 0.991)
        audio(words)    = words / 2.5             (TARGET_WPM = 150)

    Both divided by `SPEED`, so the *ratios* - and therefore the queue
    behaviour, the headroom and the stalls - are the real ones while the run
    takes seconds. It is still a stub and still says so on every artefact; the
    calibration makes it a better rehearsal, never a measurement.
    """

    SENTENCES = StubStages.SENTENCES * 6
    #: Wall-clock divisor. Ratios are preserved; only the clock is compressed.
    SPEED = 10.0
    #: The fit goes negative below about 7 words, where nothing was measured.
    #: A floor keeps the rehearsal from claiming free synthesis.
    MIN_GENERATE = 0.30

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
                await asyncio.sleep(0.25 / self.SPEED)
                yield sentence

        log.mark("claude_start", {"model": "stub"})
        return stream(), None, None, None

    async def synth(self, text: str):
        words = len(text.split())
        generate = max(self.MIN_GENERATE, 0.10856 * words - 0.8113) / self.SPEED
        rate = 24000
        await asyncio.sleep(generate)
        return [0] * int(rate * (words / 2.5) / self.SPEED), rate


async def run_once(stages, out: pathlib.Path, label: str, args,
                   log: probe.EventLog, policy: AssemblyPolicy) -> dict:
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
        run = await dp.run_decoupled(sentences, stages.synth, log,
                                     policy=policy,
                                     queue_depth=args.queue_depth,
                                     buffer_chars=args.buffer_chars,
                                     coupled=args.coupled)
    finally:
        if module is not None:
            module.settings = original
        if generator is not None:
            await generator.client.close()

    problems = dp.phase6_problems(run)
    written = write_audio(run, out, label)
    timing = dp.timing_report(run)
    exa_cost = float(reply.get("cost") or cost_model.EXA_COST_PER_SEARCH)
    return {
        "run": label, "query": query, "stub": stages.stub,
        "coupled_fixture": args.coupled,
        "passed": not problems, "problems": problems,
        "timing": timing,
        "decoupling": dp.decoupling_evidence(run),
        "playback": dp.playback_report(run),
        "chunking": dp.chunk_report(run),
        "policy": vars(policy),
        "chunks": [s.to_dict() for s in run.chunks],
        "raw_sentences": [{"index": index, "words": len(s.split()),
                           "characters": len(s), "text": s}
                          for index, s in enumerate(run.sentences)],
        "script_buffer_series": run.script_buffer_series,
        "tts_queue_series": run.tts_queue_series,
        "audio": written,
        "exa": {k: reply.get(k) for k in
                ("sources", "results_returned", "packet_chars", "search_type",
                 "remote_seconds")},
        "cost_usd": {"exa": exa_cost,
                     "anthropic_note": "stream_sentences returns no usage "
                                       "block; Exa cost only"},
        "gpu": probe.gpu_memory(),
        "log": run.log.as_dict(),
    }


def write_audio(run: dp.DecoupledRun, out: pathlib.Path, label: str) -> dict:
    import numpy as np

    folder = out / "audio" / label
    folder.mkdir(parents=True, exist_ok=True)
    names = []
    for spoken, samples in zip(run.chunks, run.samples):
        name = f"chunk_{spoken.chunk.index:02d}.wav"
        bake.write_wav(folder / name, samples, run.sample_rate)
        names.append(name)

    gap = np.zeros(int(run.playback.gap * run.sample_rate), dtype=np.float32)
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
        "\n\n".join(f"[{s.chunk.index:02d} {s.chunk.words}w {s.chunk.reason}] "
                    f"{s.chunk.text}" for s in run.chunks) + "\n",
        encoding="utf-8")
    return {"chunk_files": names, "folder": f"audio/{label}",
            "episode": f"audio/{label}_episode.wav",
            "episode_seconds": len(episode) / run.sample_rate
            if run.sample_rate else 0.0}


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reference", default="",
                        help="the selected FAM reference voice, "
                             "experiments/references/working/reference_3.wav")
    parser.add_argument("--query",
                        default="What happened in the markets this week?")
    parser.add_argument("--model", default="")
    parser.add_argument("--queue-depth", type=int, default=dp.QUEUE_DEPTH,
                        dest="queue_depth")
    parser.add_argument("--buffer-chars", type=int, default=dp.SCRIPT_BUFFER_CHARS,
                        dest="buffer_chars")
    parser.add_argument("--first-min-words", type=int, dest="first_min_words")
    parser.add_argument("--min-words", type=int, dest="min_words")
    parser.add_argument("--target-words", type=int, dest="target_words")
    parser.add_argument("--max-words", type=int, dest="max_words")
    parser.add_argument("--max-chunks", type=int, default=0, dest="max_chunks")
    parser.add_argument("--runs", type=int, default=2,
                        help="cold, then warm. 1 runs cold only.")
    parser.add_argument("--out", default="")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="the whole path with no model, no GPU and no API "
                             "calls. Writes to a directory named _STUB and "
                             "stamps every artefact NOT A RESULT.")
    parser.add_argument("--coupled", action="store_true",
                        help="rebuild Phase 5's coupling deliberately, so the "
                             "decoupling assertion can be seen to reject it. "
                             "A fixture, not an option: the run is expected "
                             "to FAIL.")
    return parser


def policy_from(args) -> AssemblyPolicy:
    given = {name: getattr(args, name) for name in
             ("first_min_words", "min_words", "target_words", "max_words")
             if getattr(args, name, None)}
    return AssemblyPolicy(**given)


async def _main(args, out: pathlib.Path) -> dict:
    reference = pathlib.Path(args.reference)
    stub = bool(args.dry_run)
    policy = policy_from(args)
    if stub:
        # The stub compresses wall-clock by `SPEED`, so the policy's two
        # wall-clock rules have to be compressed with it or they fire on a
        # timeline ten times shorter than the one they were set for. The word
        # thresholds are unitless and stay exactly as they will run.
        policy = dataclasses.replace(
            policy,
            max_wait_seconds=policy.max_wait_seconds / StubStagesLong.SPEED,
            headroom_floor_seconds=(policy.headroom_floor_seconds
                                    / StubStagesLong.SPEED))
    results = {
        "label": STUB_LABEL if stub else LABEL,
        "stub": stub, "is_production_latency": False,
        "experiment": "Phase 6 - decoupled script buffer, adaptive speech "
                      "packetisation, Claude measured independently",
        "coupled_fixture": args.coupled,
        "device": "none (stub)" if stub else args.device,
        "reference": "none (stub)" if stub else reference.name,
        "generation_settings": identity.GENERATION,
        "policy": vars(policy),
        "policy_time_rules_scaled_for_stub": stub,
        "queue_depth": args.queue_depth,
        "script_buffer_chars": args.buffer_chars,
        "sentence_gap_seconds": dp.SENTENCE_GAP,
        "chunker": "stubbed sentence stream" if stub else
                   "script_generator.ScriptGenerator.stream_sentences "
                   "(production, unmodified) -> experiments.speech_assembler",
        "first_listen_budget_seconds": FIRST_LISTEN_BUDGET,
        "phase5_warm_reported": PHASE5_WARM,
        "query": args.query, "started": time.time(), "runs": [],
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
            print(f"  Chatterbox cold model load  {cold_load:.2f}s   "
                  "(infrastructure, not per-request latency)\n")
        stages = (StubStagesLong if stub else RealStages)(model, reference, args)
        print(f"--- run: {label} ---\n")
        record = await run_once(stages, out, label, args, log, policy)
        _report(record)
        results["runs"].append(record)

    results["gpu_peak"] = probe.gpu_memory()
    results["finished"] = time.time()
    return results


def _report(record: dict) -> None:
    timing, chunking = record["timing"], record["chunking"]
    playback, decoupling = record["playback"], record["decoupling"]
    for name in ("exa_latency", "claude_ttft", "claude_to_first_sentence",
                 "claude_to_first_speech_chunk", "first_chunk_tts_seconds",
                 "search_to_first_listen", "claude_stream_seconds",
                 "claude_reader_blocked_seconds", "assembler_blocked_seconds",
                 "search_to_complete_audio", "overlap_seconds"):
        value = timing.get(name)
        print(f"    {name:<32}" + (f"{value:8.3f}s" if value is not None
                                   else "       -"))
    print(f"    TTS invocations                 {chunking['tts_invocations']:>8}"
          f"  (from {chunking['raw_sentences']} sentences)")
    print(f"    median words per chunk          "
          f"{chunking['median_words']:>8.0f}  "
          f"(min {chunking['min_words']}, max {chunking['max_words']})")
    print(f"    chunks under 10 words           "
          f"{chunking['chunks_under_10_words']:>8}")
    print(f"    playback stalls                 {playback['playback_stalls']:>8}"
          f"  ({playback['total_stall_seconds']:.2f}s)")
    minimum = playback.get("minimum_playback_headroom")
    print(f"    minimum headroom                "
          + (f"{minimum:8.2f}s" if minimum is not None else "       -"))
    print(f"    decoupling                      {decoupling['verdict']}")
    if record["problems"]:
        print("\n    ASSERTIONS FAILED:")
        for problem in record["problems"]:
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
        out = pathlib.Path("experiments/results") / f"phase6_STUB_{stamp}"
    else:
        out = pathlib.Path("experiments/results") / f"phase6_4090_{stamp}"
    if args.dry_run and "STUB" not in out.name.upper():
        raise SystemExit("a --dry-run must write to a directory whose name "
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
    if args.coupled:
        print("  COUPLED FIXTURE: Phase 5's shape, rebuilt on purpose. This "
              "run is EXPECTED TO FAIL its decoupling assertion.\n")

    results = asyncio.run(_main(args, out))

    (out / "results.json").write_text(json.dumps(results, indent=2, default=str),
                                      encoding="utf-8")
    (out / "events.jsonl").write_text("\n".join(
        json.dumps({"run": record["run"], **event})
        for record in results["runs"]
        for event in record["log"]["events"]) + "\n", encoding="utf-8")
    (out / "chunks.json").write_text(json.dumps(
        {record["run"]: {"policy": record["policy"],
                         "raw_sentences": record["raw_sentences"],
                         "chunks": record["chunks"]}
         for record in results["runs"]}, indent=2, default=str),
        encoding="utf-8")
    (out / "ANALYSIS.md").write_text(_analysis(results), encoding="utf-8")

    print(f"\nwrote {out}/")
    for name in ("ANALYSIS.md", "results.json", "events.jsonl", "chunks.json",
                 "audio/"):
        print(f"  {name}")
    if results["stub"]:
        print(f"\n  {STUB_LABEL}")

    failed = [r for r in results["runs"] if r["problems"]]
    if failed and not args.coupled:
        print("\n  PHASE 6 ASSERTIONS FAILED. The results are written; the "
              "exit code is non-zero on purpose.\n")
        return 2
    if args.coupled:
        print("\n  Coupled fixture complete. It was supposed to fail, and it "
              + ("did." if failed else "DID NOT - the assertion has no teeth.")
              + "\n")
        return 0 if failed else 3
    print("\n  All Phase 6 assertions held.\n")
    return 0


# --------------------------------------------------------------------------
def _verdicts(results: dict) -> list:
    """The plain-English answers, before any table."""
    warm = next((r for r in results["runs"] if r["run"] == "warm"), None)
    run = warm or results["runs"][0]
    timing, chunking = run["timing"], run["chunking"]
    playback, decoupling = run["playback"], run["decoupling"]
    first = timing.get("search_to_first_listen")
    phase5 = results["phase5_warm_reported"]
    lines = []

    if first is None:
        lines.append("1. **Search to first listen: not measured.**")
    else:
        delta = first - phase5["search_to_first_listen"]
        verdict = "YES" if first <= results["first_listen_budget_seconds"] else "NO"
        lines.append(
            f"1. **Search to first listen ({run['run']}): {first:.3f}s - "
            f"{verdict}**, against a {results['first_listen_budget_seconds']}s "
            f"budget. Phase 5 reported {phase5['search_to_first_listen']:.3f}s, "
            f"so this is {abs(delta):.3f}s "
            f"{'slower' if delta > 0 else 'faster'}. Single request, so API "
            "variance is not separated from architecture here.")

    blocked = timing["claude_reader_blocked_seconds"]
    lines.append(
        f"2. **Claude decoupled from TTS backpressure: "
        f"{'YES' if blocked <= dp.BLOCKED_EPSILON else 'NO'}.** The reader was "
        f"blocked for {blocked:.3f}s (Phase 5: "
        f"{phase5['claude_reader_blocked_seconds']:.3f}s). "
        f"{decoupling['verdict']}.")

    lines.append(
        f"3. **Claude actually took "
        f"{timing['claude_stream_seconds']:.3f}s** to stream the script, with "
        f"{timing['claude_local_processing_seconds']:.3f}s of that our own "
        f"processing and {blocked:.3f}s our own blocking. Phase 5's reported "
        f"{phase5['claude_stream_seconds_reported']:.3f}s was "
        f"{phase5['claude_reader_blocked_seconds']:.3f}s backpressure.")

    stalls = playback["playback_stalls"]
    lines.append(
        f"4. **Playback stalls: {stalls}** "
        f"({playback['total_stall_seconds']:.2f}s of dead air). Minimum "
        f"headroom {playback['minimum_playback_headroom']:.2f}s, median "
        f"{playback['median_playback_headroom']:.2f}s, maximum "
        f"{playback['maximum_playback_headroom']:.2f}s.")

    lines.append(
        f"5. **Batching: {chunking['raw_sentences']} sentences became "
        f"{chunking['tts_invocations']} TTS calls**, median "
        f"{chunking['median_words']:.0f} words (min {chunking['min_words']}, "
        f"max {chunking['max_words']}); {chunking['chunks_under_5_words']} under "
        f"five words, {chunking['chunks_under_10_words']} under ten. Aggregate "
        f"{chunking['aggregate_realtime_factor']:.2f}x realtime over "
        f"{chunking['total_tts_compute_seconds']:.1f}s of compute for "
        f"{chunking['audio_seconds']:.1f}s of audio.")

    improved = (blocked <= dp.BLOCKED_EPSILON and stalls == 0 and not run["problems"]
                and chunking["tts_invocations"] < chunking["raw_sentences"])
    lines.append(
        f"6. **Better than Phase 5: {'YES' if improved else 'NOT PROVEN'}** on "
        "the evidence above - decoupling held, playback did not stall, batching "
        "reduced the call count, and every assertion passed."
        if improved else
        "6. **Better than Phase 5: NOT PROVEN.** Read the failed assertions and "
        "the numbers above before claiming anything.")
    return lines


def _analysis(results: dict) -> str:
    stub = results["stub"]
    head = ("# Phase 6, stubbed: wiring and assertions only" if stub else
            "# Phase 6 - Claude decoupled from Chatterbox, on an RTX 4090")
    lines = [head, "",
             (f"**{STUB_LABEL}**" if stub else f"**{results['label']}.**")
             + " One machine, one request at a time. Not production latency.",
             ""]
    if results.get("coupled_fixture"):
        lines += ["**This is the coupled fixture** - Phase 5's shape rebuilt on "
                  "purpose so the decoupling assertion can be seen to reject "
                  "it. It is supposed to fail.", ""]
    lines += ["## Executive result", ""] + _verdicts(results) + [""]

    lines += ["## Architecture", "",
              "```",
              "Claude stream",
              "  -> reader            never touches the TTS queue",
              f"  -> script buffer     bounded at {results['script_buffer_chars']:,} characters",
              "  -> assembler         whole sentences -> speech-sized chunks",
              f"  -> TTS work queue    bounded at {results['queue_depth']}",
              "  -> Chatterbox Base",
              "  -> playback model",
              "```", "",
              f"- chunker: `{results['chunker']}`",
              f"- policy: `{results['policy']}`",
              f"- silence between chunks: {results['sentence_gap_seconds']}s "
              "(production's `SENTENCE_GAP`)",
              f"- reference voice: `{results['reference']}`",
              ("- Chatterbox cold model load: not applicable to a stub run"
               if stub else
               f"- Chatterbox cold model load: "
               f"**{results.get('cold_model_load_seconds', float('nan')):.2f}s** "
               "- infrastructure, paid once per process, never by a listener "
               "on a warm server"), ""]

    for record in results["runs"]:
        timing = record["timing"]
        lines += [f"## Run: {record['run']}", "",
                  "### Timing waterfall", "",
                  "| measure | seconds | what it is |", "|---|---|---|"]
        rows = [
            ("exa_latency", "the retrieval round trip"),
            ("claude_ttft", "Claude's first text delta"),
            ("claude_to_first_sentence", "first complete sentence out of the chunker"),
            ("claude_to_first_speech_chunk", "first assembled chunk released"),
            ("first_chunk_tts_seconds", "Chatterbox on the opening chunk"),
            ("search_to_first_listen", "**the number the product is judged on**"),
            ("claude_stream_seconds", "Claude's own stream, nothing downstream holding it"),
            ("claude_local_processing_seconds", "our work inside the reader loop"),
            ("claude_reader_blocked_seconds", "time our architecture stopped reading Claude"),
            ("assembler_blocked_seconds", "downstream blocking - allowed, not Claude's"),
            ("tts_total_seconds", "all Chatterbox compute"),
            ("overlap_seconds", "TTS running while Claude still wrote"),
            ("search_to_complete_audio", "last sample rendered"),
        ]
        for name, meaning in rows:
            value = timing.get(name)
            lines.append(f"| `{name}` | "
                         + ("-" if value is None else f"{value:.3f}s")
                         + f" | {meaning} |")
        lines += ["",
                  f"Peak script buffer {timing['peak_script_buffer_chars']:,} "
                  f"characters of {results['script_buffer_chars']:,}; peak TTS "
                  f"queue depth {timing['peak_tts_queue_depth']} of "
                  f"{results['queue_depth']}.", ""]

        playback = record["playback"]
        lines += ["### The listener's timeline", "",
                  f"- playback stalls: **{playback['playback_stalls']}** "
                  f"({playback['total_stall_seconds']:.2f}s)",
                  f"- minimum headroom: {playback['minimum_playback_headroom']:.2f}s",
                  f"- median headroom: {playback['median_playback_headroom']:.2f}s",
                  f"- maximum headroom: {playback['maximum_playback_headroom']:.2f}s",
                  f"- headroom at Claude complete: "
                  f"{playback['headroom_at_claude_complete']:.2f}s"
                  if playback.get("headroom_at_claude_complete") is not None
                  else "- headroom at Claude complete: -",
                  f"- headroom at final synthesis: "
                  f"{playback['headroom_at_final_tts']:.2f}s", ""]
        if playback["stalls"]:
            lines += ["| chunk | needed at | ready at | stall |",
                      "|---|---|---|---|"]
            for stall in playback["stalls"]:
                lines.append(f"| {stall['index']:02d} | "
                             f"{stall['needed_at']:.2f}s | "
                             f"{stall['ready_at']:.2f}s | "
                             f"{stall['stall_seconds']:.2f}s |")
            lines.append("")

        chunking = record["chunking"]
        lines += ["### Chunk quality and efficiency", "",
                  "| | Phase 5 (reported) | Phase 6 |", "|---|---|---|",
                  f"| TTS invocations | not reported | "
                  f"{chunking['tts_invocations']} |",
                  f"| raw sentences | not reported | "
                  f"{chunking['raw_sentences']} |",
                  f"| median words per call | not reported | "
                  f"{chunking['median_words']:.0f} |",
                  f"| min / max words | not reported | "
                  f"{chunking['min_words']} / {chunking['max_words']} |",
                  f"| under 5 words | not reported | "
                  f"{chunking['chunks_under_5_words']} |",
                  f"| under 10 words | not reported | "
                  f"{chunking['chunks_under_10_words']} |",
                  f"| mean generation | not reported | "
                  f"{chunking['mean_generate_seconds']:.3f}s |",
                  f"| total TTS compute | not reported | "
                  f"{chunking['total_tts_compute_seconds']:.1f}s |",
                  f"| audio produced | not reported | "
                  f"{chunking['audio_seconds']:.1f}s |",
                  f"| aggregate realtime | not reported | "
                  f"{chunking['aggregate_realtime_factor']:.2f}x |",
                  f"| playback stalls | "
                  f"{results['phase5_warm_reported']['playback_stalls']} | "
                  f"{playback['playback_stalls']} |",
                  f"| search to first listen | "
                  f"{results['phase5_warm_reported']['search_to_first_listen']:.3f}s | "
                  + ("-" if timing['search_to_first_listen'] is None
                     else f"{timing['search_to_first_listen']:.3f}s") + " |",
                  f"| Claude, as measured | "
                  f"{results['phase5_warm_reported']['claude_stream_seconds_reported']:.3f}s "
                  "(contaminated) | "
                  f"{timing['claude_stream_seconds']:.3f}s (clean) |",
                  f"| search to complete audio | "
                  f"{results['phase5_warm_reported']['search_to_complete_audio']:.3f}s | "
                  + ("-" if timing['search_to_complete_audio'] is None
                     else f"{timing['search_to_complete_audio']:.3f}s") + " |",
                  "",
                  "Why chunks were released:", ""]
        for reason, count in chunking["release_reasons"].items():
            lines.append(f"- {reason}: {count}")
        lines += ["",
                  "**Phase 5's per-chunk figures are not in this repository** - "
                  "only its headline numbers were reported. The comparison "
                  "columns above are filled where a figure was quoted and left "
                  "as *not reported* everywhere else, rather than reconstructed."
                  "", "",
                  f"Listen: `{record['audio']['episode']}` "
                  f"({record['audio']['episode_seconds']:.1f}s); chunks in "
                  f"`{record['audio']['folder']}/`.", ""]
        if record["problems"]:
            lines += ["### Assertions that failed", ""]
            lines += [f"- {problem}" for problem in record["problems"]] + [""]
        if record["log"]["unavailable"]:
            lines += ["### Not measurable in this run", ""]
            for name, why in record["log"]["unavailable"].items():
                lines.append(f"- `{name}` - {why}")
            lines.append("")

    lines += ["## What this does not settle", "",
              "- One request per condition. First-listen differences of a few "
              "hundred milliseconds are inside API variance, not evidence.",
              "- Chatterbox's T3 stage is a Python autoregressive loop holding "
              "the GIL, so the Claude stream still shares a process with it. "
              "`claude_stream_seconds` is clean of *queue* blocking, not of "
              "interpreter contention.",
              "- The assembly policy's word thresholds are bounded by the only "
              "4090 data in this repository, which covers 28-41 words. Nothing "
              "below 28 or above 41 has been measured; "
              "`tools/fit_chunk_policy.py` re-derives them from a real run.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
