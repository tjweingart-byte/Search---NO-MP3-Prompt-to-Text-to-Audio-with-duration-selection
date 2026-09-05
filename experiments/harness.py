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


def _generator_for(arm: Arm):
    """Which model call this arm makes.

    `production` (the default) uses the app's own request shape, so the result
    says something about what ships. `benchmark` reproduces the manual Exa
    benchmark's short opening call, so a run can be checked against the numbers
    measured by hand.
    """
    return build_generator(arm.params.get("generator", "production"))


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
            # The stream is held by name so it can be closed deterministically
            # below. Breaking out of an `async for` does NOT close an async
            # generator: it is left suspended at its `yield`, inside the
            # `async with` that holds the HTTP response open. Python finalises
            # it later, from the event loop's async-generator finaliser - a
            # different task from this one - and httpx/httpcore bind their
            # anyio cancel scopes to the entering task, so that teardown raises
            # "Attempted to exit cancel scope in a different task" and prints a
            # GeneratorExit traceback after the sweep has already finished.
            token_stream = generator.stream(
                query,
                self.spec.minutes,
                context=retrieved.context,
                model=arm.model,
                search=(arm.search == "anthropic_web_search"),
                max_searches=int(arm.params.get("max_searches", 3) or 3),
            )
            try:
                with timeline.span("generate", host="anthropic-api", adapter="claude") as stage:
                    async for delta in token_stream:
                        if timeline.at("first_token") is None:
                            timeline.mark("first_token")
                            stage.detail["first_token_at"] = timeline.at("first_token")
                        buffer += delta
                        full.append(delta)
                        if first_chunk is None:
                            candidate = first_chunk_ready(buffer, chunk_words)
                            if candidate:
                                first_chunk = candidate
                                timeline.mark("first_chunk")
                                # Sound could start here, so stop timing the model
                                # for the purposes of first audio and go speak.
                                break
                    stage.detail["streamed_chars"] = len(buffer)
            finally:
                # Outside the span on purpose: teardown is not model latency, so
                # closing here cannot inflate `generate_seconds`. Inside this
                # task on purpose: it is the task that opened the connection.
                await _aclose(token_stream)

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
            result.metrics = {
                "first_token": timeline.at("first_token"),
                "first_chunk": timeline.at("first_chunk"),
                "first_audio": timeline.at("first_audio"),
                "search_seconds": _stage_duration(timeline, "search"),
                "generate_seconds": _stage_duration(timeline, "generate"),
                "synthesis_seconds": _stage_duration(timeline, "synthesis"),
                "audio_seconds": audio.audio_seconds if audio else 0.0,
                "realtime_factor": audio.realtime_factor if audio else None,
                "first_chunk_words": len(first_chunk.split()) if first_chunk else 0,
                "sources": len(retrieved.sources),
                "searches": retrieved.searches or usage.get("searches", 0),
                "server_side_search": server_side,
            }
        except InfrastructureRequired:
            raise  # never swallowed: this one needs a person, not a retry
        except Exception as exc:
            result.ok = False
            result.error = f"{type(exc).__name__}: {str(exc)[:200]}"
        finally:
            result.timeline = timeline.to_dict()

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


async def _aclose(stream) -> None:
    """Close an async generator, tolerating one that is already finished.

    A generator that ran to completion is already closed and `aclose()` on it
    is a no-op; one broken out of early is closed here rather than whenever the
    garbage collector gets to it.
    """
    aclose = getattr(stream, "aclose", None)
    if aclose is None:
        return
    await aclose()


def _stage_duration(timeline: Timeline, name: str) -> Optional[float]:
    total = sum(s.duration for s in timeline.stages if s.name == name)
    return total if any(s.name == name for s in timeline.stages) else None
