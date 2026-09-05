"""Run the trials, on one clock, and write them down as they happen.

Design decisions worth knowing before changing anything here:

**Sequential by default.** Trials run one at a time so they do not time each
other's queueing. `tools/compare_search.py` reached the same conclusion; a
parallel sweep measures contention and calls it latency.

**Every stage is measured on the orchestrator's clock**, whichever machine did
the work. See `timeline.py` for why remote self-reported times are recorded
beside that number rather than instead of it.

**A failed trial is data.** It is written to `trials.jsonl` with its error and
excluded from the statistics, and the report says how many failed. Dropping it
silently would turn a broken arm into a fast one.

**Nothing is cached.** The generator disables the script cache for the duration
of a run; a cache hit returns in milliseconds and would look like a result.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from experiments import cost as cost_mod
from experiments import registry
from experiments.adapters.base import InfrastructureRequired
from experiments.fakes import SIMULATED
from experiments.generate import build_generator
from experiments.spec import Arm, ExperimentSpec
from experiments.timeline import Timeline

_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s")


#: Arm params the generators understand. Everything else in `params` belongs
#: to an adapter, so it is not forwarded and cannot be silently misread.
GENERATOR_OPTIONS = ("thinking", "effort", "first_sentence_directive", "http_trace")


def _generator_for(arm: Arm):
    """Which model call this arm makes, and how it is configured.

    `production` (the default) uses the app's own request shape, so the result
    says something about what ships. `benchmark` reproduces the manual Exa
    benchmark's short opening call byte for byte. `tuned` is that same call
    with the request settings under experiment.
    """
    options = {k: arm.params[k] for k in GENERATOR_OPTIONS if k in arm.params}
    return build_generator(arm.params.get("generator", "production"), **options)


#: The manual Exa benchmark's chunk rule: the first sentence that leaves at
#: least this many words. Below about this length a chunk is too short to give
#: the voice natural intonation; much above it and sound starts later than it
#: needs to. Kept as the default for pipeline arms so the repeated-trial runs
#: stay comparable with the hand-measured one.
BENCHMARK_FIRST_CHUNK_WORDS = 25

_SENTENCE_CHARS = ".!?"


def first_chunk_ready(buffer: str, words: int = 0) -> Optional[str]:
    """The first speakable chunk, under whichever policy is in force.

    `words=0` is what the shipped pipeline does: speak the first complete
    sentence, however short.

    A positive number is the rule from `exa_claude_benchmark.py`: the first
    sentence *ending* that leaves at least that many words. Note it is the
    whole run up to that sentence end, not a truncation - a chunk handed to a
    voice mid-clause sounds wrong, so the boundary always wins over the count.
    A short opening sentence is therefore not enough on its own; the scan
    continues to the next ending.

    This is the knob a first-chunk-size experiment turns: a smaller chunk
    starts sound sooner but gives the voice less to work with.
    """
    if words <= 0:
        match = _SENTENCE_END.search(buffer)
        return buffer[: match.end()].strip() if match else None
    for index, char in enumerate(buffer):
        if char in _SENTENCE_CHARS:
            candidate = buffer[: index + 1].strip()
            if len(candidate.split()) >= words:
                return candidate
    return None


@dataclass
class TrialResult:
    """One trial: what happened, when, and whether it worked."""

    arm: str
    query: str
    index: int
    ok: bool = True
    error: Optional[str] = None
    simulated: bool = False
    #: threshold -> the opening that threshold would have spoken. Saved so the
    #: tradeoff can be judged on the writing, not only on the clock.
    threshold_texts: dict = field(default_factory=dict)
    #: The opening the model actually wrote. Saved so arms can be compared on
    #: what they said, not only on how fast they said it - a latency win that
    #: costs quality is not a win.
    first_chunk_text: str = ""
    timeline: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    cost: float = 0.0
    artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "arm": self.arm,
            "query": self.query,
            "index": self.index,
            "ok": self.ok,
            "error": self.error,
            "simulated": self.simulated,
            "metrics": {k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in self.metrics.items()},
            "usage": self.usage,
            "cost": round(self.cost, 6),
            "artifacts": list(self.artifacts),
            "first_chunk_text": self.first_chunk_text,
            "threshold_texts": dict(self.threshold_texts),
            "timeline": self.timeline,
            "recorded_at": time.time(),
        }


class Harness:
    """Runs a spec. Writes each trial to the store the moment it completes."""

    def __init__(
        self,
        spec: ExperimentSpec,
        run=None,
        generator_factory: Optional[Callable[[Arm], object]] = None,
        search_factory: Optional[Callable[[str], object]] = None,
        tts_factory: Optional[Callable[[str, Optional[str]], object]] = None,
        save_audio: bool = False,
    ) -> None:
        self.spec = spec
        self.run = run
        self.generator_factory = generator_factory or _generator_for
        self.search_factory = search_factory or registry.search_adapter
        self.tts_factory = tts_factory or registry.tts_adapter
        self.save_audio = save_audio
        self.results: list[TrialResult] = []

    # -- one trial ------------------------------------------------------
    async def run_trial(self, arm: Arm, query: str, index: int) -> TrialResult:
        result = TrialResult(arm=arm.name, query=query, index=index)
        timeline = Timeline()
        token_stream = None
        generator = None
        search = self.search_factory(arm.search)
        tts = self.tts_factory(arm.tts, arm.params.get("voice"))
        generator = self.generator_factory(arm)

        try:
            # 1. Retrieval, when it is a separable step at all.
            retrieved = await search.search(query, timeline, **arm.params)
            server_side = not getattr(search, "separable", True)

            # 2. The model, streamed, watching for the two moments that matter.
            chunk_words = int(arm.params.get("first_chunk_words", 0) or 0)
            buffer = ""
            first_chunk: Optional[str] = None
            full: list[str] = []
            # Held by name so the outer `finally` can close it. Breaking out
            # of an `async for` does NOT close an async generator: it is left
            # suspended at its `yield`, inside the `async with` holding the
            # HTTP response. Finalised later from the loop's async-generator
            # finaliser - a different task - it raises out of httpcore and
            # prints a traceback after the sweep has already reported success.
            #
            # It is closed after every checkpoint has been marked and after the
            # timeline has been snapshotted, so teardown cannot appear in any
            # reported number - `generate`, `synthesis` or `first_audio`.
            # Observation only: every threshold it watches is reached at or
            # before the 25-word boundary this trial already breaks on, so it
            # asks for nothing extra and changes nothing about the run.
            probe = None
            thresholds = arm.params.get("chunk_thresholds")
            if thresholds:
                from experiments.chunk_probe import ChunkProbe

                probe = ChunkProbe(thresholds)

            token_stream = generator.stream(
                query,
                self.spec.minutes,
                context=retrieved.context,
                model=arm.model,
                search=(arm.search == "anthropic_web_search"),
                max_searches=int(arm.params.get("max_searches", 3) or 3),
            )
            with timeline.span("generate", host="anthropic-api", adapter="claude") as stage:
                async for delta in token_stream:
                    if timeline.at("first_token") is None:
                        timeline.mark("first_token")
                        stage.detail["first_token_at"] = timeline.at("first_token")
                        if probe is not None:
                            # Zeroed at the first token, so every threshold is
                            # reported as a delay after text started arriving.
                            probe.start()
                    buffer += delta
                    full.append(delta)
                    if probe is not None:
                        probe.observe(buffer)
                    # The word count crosses 25 before the sentence that
                    # contains it closes, so these are two different moments
                    # and the gap between them is what the chunk rule costs.
                    if timeline.at("words_25") is None and len(buffer.split()) >= 25:
                        timeline.mark("words_25")
                    if first_chunk is None:
                        candidate = first_chunk_ready(buffer, chunk_words)
                        if candidate:
                            first_chunk = candidate
                            timeline.mark("first_chunk")
                            # Sound could start here, so stop timing the model
                            # for the purposes of first audio and go speak.
                            break
                stage.detail["streamed_chars"] = len(buffer)
            if first_chunk is None:
                first_chunk = buffer.strip()
                if first_chunk:
                    timeline.mark("first_chunk")

            # 3. Speech for that first chunk only: the moment sound reaches a
            #    listener, which is the number this product is judged on.
            audio = None
            if first_chunk and arm.tts != "none":
                audio = await tts.synth(first_chunk, timeline, **arm.params)
                timeline.mark("first_audio")
                if self.save_audio and self.run is not None and audio.pcm:
                    name = f"{arm.name}-{index}-{abs(hash(query)) % 10000}.pcm"
                    path = self.run.write_artifact(name, audio.pcm)
                    result.artifacts.append(str(path.name))

            # 4. Record.
            usage = generator.usage() if hasattr(generator, "usage") else {}
            result.simulated = bool(usage.get(SIMULATED)) or getattr(
                search.available(), "reason", ""
            ) == SIMULATED
            result.usage = usage
            result.cost = retrieved.cost + (audio.cost if audio else 0.0)
            if usage.get("input_tokens"):
                result.cost += cost_mod.model_cost(
                    usage.get("model") or "claude-sonnet-5",
                    usage.get("input_tokens", 0),
                    usage.get("output_tokens", 0),
                )
            result.first_chunk_text = first_chunk or ""
            timing = (generator.usage() or {}).get("timing") if hasattr(generator, "usage") else None
            result.metrics = {
                "first_token": timeline.at("first_token"),
                "words_25": timeline.at("words_25"),
                "first_chunk": timeline.at("first_chunk"),
                "first_audio": timeline.at("first_audio"),
                "search_seconds": _stage_duration(timeline, "search"),
                "generate_seconds": _stage_duration(timeline, "generate"),
                "synthesis_seconds": _stage_duration(timeline, "synthesis"),
                "audio_seconds": audio.audio_seconds if audio else 0.0,
                "realtime_factor": audio.realtime_factor if audio else None,
                "first_chunk_words": len(first_chunk.split()) if first_chunk else 0,
                "first_chunk_chars": len(first_chunk) if first_chunk else 0,
                # How long the 25-word mark waited for its sentence to close.
                "boundary_wait": (
                    timeline.at("first_chunk") - timeline.at("words_25")
                    if timeline.at("first_chunk") is not None
                    and timeline.at("words_25") is not None else None
                ),
                "sources": len(retrieved.sources),
                "searches": retrieved.searches or usage.get("searches", 0),
                "server_side_search": server_side,
            }
            if probe is not None:
                result.metrics.update(probe.metrics())
                result.threshold_texts = probe.texts()
            result.metrics.update(_segments(timeline, timing))
            if timing:
                result.usage = {**result.usage, "timing": timing}
        except InfrastructureRequired:
            raise  # never swallowed: this one needs a person, not a retry
        except Exception as exc:
            result.ok = False
            result.error = f"{type(exc).__name__}: {str(exc)[:200]}"
        finally:
            # Snapshot first, then tear down: nothing below this line may
            # influence a measurement.
            result.timeline = timeline.to_dict()
            await _aclose(token_stream)
            # Draining the stream is what makes a total generation time
            # knowable at all: the trial stops reading at the first chunk, so
            # only teardown sees the response end. Read after the close.
            if generator is not None and hasattr(generator, "usage"):
                complete = generator.usage()
                if complete:
                    result.usage = {**result.usage, **complete}
                    if complete.get("stream_seconds") is not None:
                        result.metrics["total_generation_seconds"] = complete["stream_seconds"]
                    # The dispatch checkpoints only exist once the generator has
                    # finished, because its `finally` is where they are written.
                    # Computing the segments earlier read an empty dict and
                    # silently produced a row of blanks.
                    if complete.get("timing"):
                        result.metrics.update(_segments(timeline, complete["timing"]))

        return result

    # -- the sweep ------------------------------------------------------
    async def run_all(self, progress: Optional[Callable[[TrialResult], None]] = None) -> list[TrialResult]:
        """Every arm x query x trial, in order, recorded as it goes."""
        for index in range(1, self.spec.trials + 1):
            for query in self.spec.queries:
                for arm in self.spec.arms:
                    result = await self.run_trial(arm, query, index)
                    self.results.append(result)
                    if self.run is not None:
                        self.run.append_trial(result.to_dict())
                    if progress:
                        progress(result)
        return self.results


def _segments(timeline: Timeline, timing: Optional[dict]) -> dict:
    """The wait, cut into pieces that have different causes.

    Five segments, all measured from the request's dispatch or between two
    checkpoints on the same `perf_counter` clock:

        dispatch -> first token          connection, queueing, prefill, thinking
        first token -> 25 words          how fast tokens arrive
        25 words -> sentence boundary    what the chunk rule costs
        dispatch -> sentence boundary    the number a listener would feel
        dispatch -> completion           the whole response

    `dispatch_to_stream_open` splits the first segment again: everything before
    the response headers came back is transport, not the model. That is the
    line between a slow network and a slow model, and it is the reason this
    exists.

    Returns an empty-valued shape rather than nothing when the generator did
    not report timing, so a missing segment reads as missing instead of being
    quietly absent from the table.
    """
    out: dict = {
        "seg_dispatch_to_first_token": None,
        "seg_first_token_to_25_words": None,
        "seg_25_words_to_boundary": None,
        "seg_dispatch_to_boundary": None,
        "seg_dispatch_to_complete": None,
        "dispatch_to_stream_open": None,
        "harness_first_token_lag": None,
    }
    words_25 = timeline.absolute("words_25")
    boundary = timeline.absolute("first_chunk")
    first_token_mark = timeline.absolute("first_token")

    if first_token_mark is not None and words_25 is not None:
        out["seg_first_token_to_25_words"] = words_25 - first_token_mark
    if words_25 is not None and boundary is not None:
        out["seg_25_words_to_boundary"] = boundary - words_25

    if not timing:
        return out

    # Phase timings, when the run carried the HTTP tracer. Copied through
    # verbatim: a phase that was not observed stays absent rather than zero.
    for key, value in timing.items():
        if key.startswith("phase_") or key in ("http_trace", "connection_reused",
                                               "events_seen"):
            out[key] = value

    dispatch = timing.get("dispatch_perf")
    first_text = timing.get("first_text_perf")
    out["dispatch_to_stream_open"] = timing.get("dispatch_to_stream_open")
    out["seg_dispatch_to_first_token"] = timing.get("dispatch_to_first_text")
    out["seg_dispatch_to_complete"] = timing.get("dispatch_to_complete")
    if dispatch is not None and boundary is not None:
        out["seg_dispatch_to_boundary"] = boundary - dispatch
    # The generator sees the first token as it leaves the SDK; the harness sees
    # it one `yield` later. The difference is this engine's own overhead, and
    # it is recorded so it can be subtracted rather than assumed to be zero.
    if first_text is not None and first_token_mark is not None:
        out["harness_first_token_lag"] = first_token_mark - first_text
    return out


async def _aclose(stream) -> None:
    """Close an async generator, tolerating one never created or already done.

    `aclose()` here triggers the generator's own `finally`, which is where
    `generate._drain` reads out whatever is left of the HTTP response - the
    only teardown that leaves httpcore's stream clean. A generator that ran to
    completion is already closed and this is a no-op.
    """
    if stream is None:
        return
    aclose = getattr(stream, "aclose", None)
    if aclose is None:
        return
    await aclose()


def _stage_duration(timeline: Timeline, name: str) -> Optional[float]:
    total = sum(s.duration for s in timeline.stages if s.name == name)
    return total if any(s.name == name for s in timeline.stages) else None
