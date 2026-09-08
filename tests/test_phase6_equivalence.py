"""Legacy against Phase 6, on the same fakes, before anything can route to it.

`_start_phase6` and `_speak_phase6` are unreachable in this build: nothing calls
them and `STREAMING_PIPELINE` does not select them. This file is the evidence
that they could be selected - the user-visible contract (duration, ordering,
text integrity, accounting, tail) has to survive the change of execution
strategy, even though the synthesis boundaries deliberately do not.

Both paths run through one harness that mirrors `stream_pcm`'s orchestration,
so the only difference between the two columns is the start/speak pair. A test
below checks that harness against the real `stream_pcm` for legacy, so the
comparison is not against a convenient reimplementation.
"""
from __future__ import annotations

import asyncio
import dataclasses
import inspect
import os
import re
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pipeline as pipeline_module
from audio_utils import PaceController, pcm_duration
from config import settings
from pipeline import (MAX_TOPUPS, OVERRUN_GRACE, TOPUP_THRESHOLD,
                      GenerationStats, PodcastPipeline)
from script_generator import ScriptNotes, plan_episode
from speech_assembly import AssembledChunk, AssemblyPolicy
from tts import DebugEngine

from tests.test_pipeline import FakeGenerator

ENGINE = DebugEngine()


async def _close(pump, budget: float = 0.5) -> float:
    """Close a pump without inheriting legacy's hang. Returns seconds taken.

    A bare `await pump.close()` never returns when the producer is blocked on a
    full queue (see the test below), and `wait_for` does not surface it either
    - `close` swallows the cancellation and returns normally - so the only
    honest detector is the clock.
    """
    started = time.perf_counter()
    try:
        await asyncio.wait_for(asyncio.shield(pump.close()), budget)
    except asyncio.TimeoutError:
        pass
    return time.perf_counter() - started


@dataclasses.dataclass
class Episode:
    """Everything the two paths must agree about."""

    seconds: float
    words: int
    script: list
    sentences: int
    truncated: bool
    topups: int
    tail_seconds: float
    first_chunk: str


async def _run(path: str, minutes: int, ratio: float, allow_topups: bool = False,
               generator=None) -> Episode:
    """`stream_pcm`'s orchestration, with the start/speak pair swappable.

    Deliberately a mirror rather than a call: `stream_pcm` selects `_start`
    unconditionally, and Step 4 is not allowed to change that.
    """
    plan = plan_episode("what is the nasdaq", minutes)
    pipeline = PodcastPipeline(generator=generator or FakeGenerator(ratio),
                               engine=ENGINE, cache=None)
    stats = GenerationStats()
    stats.plan_seconds = plan.target_seconds
    pace = PaceController(target_seconds=float(plan.target_seconds),
                          total_words=plan.word_budget,
                          sample_rate=ENGINE.sample_rate)
    notes = ScriptNotes()
    first_chunk = ""

    def start(sentences):
        nonlocal first_chunk
        if path == "legacy":
            return pipeline._start(sentences)
        pump = pipeline._start_phase6(sentences)
        return pump

    async def speak(pump):
        nonlocal first_chunk
        speaker = (pipeline._speak if path == "legacy" else pipeline._speak_phase6)
        async for audio in speaker(pump, pace, stats):
            if not first_chunk and stats.script:
                first_chunk = (stats.script[0] if path == "legacy"
                               else " ".join(stats.script))
            yield audio

    total = 0
    async for audio in speak(start(pipeline.generator.stream_sentences(plan, notes))):
        total += len(audio)

    while (allow_topups and not stats.truncated
           and pace.remaining_seconds > TOPUP_THRESHOLD
           and stats.topups < MAX_TOPUPS):
        stats.topups += 1
        words_needed = int(pace.remaining_seconds / 60.0 * settings.target_wpm)
        before = pace.spoken_words
        extra = start(pipeline.generator.top_up(plan, " ".join(stats.script),
                                                words_needed))
        async for audio in speak(extra):
            total += len(audio)
        if pace.spoken_words == before:
            break

    body = pcm_duration(total, ENGINE.sample_rate)
    async for audio in pipeline._finish(pace, stats):
        total += len(audio)
    seconds = pcm_duration(total, ENGINE.sample_rate)
    return Episode(seconds=seconds, words=stats.words, script=list(stats.script),
                   sentences=stats.sentences, truncated=stats.truncated,
                   topups=stats.topups, tail_seconds=seconds - body,
                   first_chunk=first_chunk)


def run(path, minutes, ratio, **kwargs) -> Episode:
    return asyncio.run(_run(path, minutes, ratio, **kwargs))


def both(minutes, ratio, **kwargs):
    return run("legacy", minutes, ratio, **kwargs), run("phase6", minutes, ratio, **kwargs)


# ==========================================================================
# The harness is honest
# ==========================================================================
@pytest.mark.parametrize("minutes, ratio", [(1, 1.0), (3, 0.5), (5, 1.6)])
def test_the_harness_reproduces_the_real_stream_pcm_for_legacy(minutes, ratio):
    """If the harness drifted from `stream_pcm`, every comparison below would
    be against a convenient reimplementation rather than production."""
    async def real():
        plan = plan_episode("what is the nasdaq", minutes)
        pipeline = PodcastPipeline(generator=FakeGenerator(ratio), engine=ENGINE,
                                   cache=None)
        stats = GenerationStats()
        total = 0
        async for chunk in pipeline.stream_pcm(plan, stats):
            total += len(chunk)
        return pcm_duration(total, ENGINE.sample_rate), stats

    seconds, stats = asyncio.run(real())
    harness = run("legacy", minutes, ratio)
    assert harness.seconds == pytest.approx(seconds, abs=0.01)
    assert harness.words == stats.words
    assert harness.script == stats.script
    assert harness.truncated == stats.truncated


# ==========================================================================
# The grid
# ==========================================================================
GRID = [(minutes, ratio) for minutes in (1, 3, 5)
        for ratio in (0.5, 1.0, 1.6)]


@pytest.mark.parametrize("minutes, ratio", GRID)
def test_the_spoken_text_is_the_same_script_in_the_same_order(minutes, ratio):
    """Ordering and integrity are user-visible; chunk boundaries are not.

    Not equality: Phase 6 lays `SENTENCE_GAP` once per chunk rather than once
    per sentence, so it spends less of the budget on silence and can reach one
    more sentence before the episode is full. What it speaks is always a
    continuation of what legacy spoke, never a divergence from it.
    """
    legacy, phase6 = both(minutes, ratio)
    shared = min(len(legacy.script), len(phase6.script))
    assert phase6.script[:shared] == legacy.script[:shared]
    assert len(phase6.script) >= len(legacy.script)


@pytest.mark.parametrize("minutes, ratio", GRID)
def test_the_accounting_agrees_with_what_was_spoken(minutes, ratio):
    """`stats.words` and `stats.sentences` must describe `stats.script` on both
    paths - that is what the cache, the pace controller and analytics read."""
    for episode in both(minutes, ratio):
        assert episode.sentences == len(episode.script)
        assert episode.words == sum(len(s.split()) for s in episode.script)


@pytest.mark.parametrize("minutes, ratio", GRID)
def test_phase6_never_speaks_less_than_legacy_and_never_much_more(minutes, ratio):
    """The bound on the difference: at most what the reclaimed gaps buy, which
    cannot exceed one assembled chunk."""
    legacy, phase6 = both(minutes, ratio)
    assert phase6.words >= legacy.words
    assert phase6.words - legacy.words <= AssemblyPolicy().max_words


@pytest.mark.parametrize("minutes, ratio", GRID)
def test_the_duration_contract_holds_on_both_paths(minutes, ratio):
    plan = plan_episode("q", minutes)
    legacy, phase6 = both(minutes, ratio)
    assert phase6.seconds <= plan.target_seconds + OVERRUN_GRACE
    assert legacy.seconds <= plan.target_seconds + OVERRUN_GRACE


@pytest.mark.parametrize("minutes, ratio", GRID)
def test_the_audio_length_matches_within_the_gap_difference(minutes, ratio):
    """The one measured difference, bounded rather than waved away.

    Legacy lays `SENTENCE_GAP` after every sentence; Phase 6 lays it after every
    chunk, because a chunk is one utterance and its internal pauses come from
    the voice rather than from inserted silence. So Phase 6 can be shorter by up
    to (sentences - chunks) x SENTENCE_GAP. The pace controller absorbs it
    whenever there is material left to absorb it with.
    """
    legacy, phase6 = both(minutes, ratio)
    slack = legacy.sentences * pipeline_module.SENTENCE_GAP
    assert abs(phase6.seconds - legacy.seconds) <= slack


@pytest.mark.parametrize("minutes, ratio", GRID)
def test_a_full_script_still_lands_on_the_requested_length(minutes, ratio):
    if ratio < 1.0:
        pytest.skip("a short script is expected to end early on both paths")
    plan = plan_episode("q", minutes)
    _, phase6 = both(minutes, ratio)
    assert plan.target_seconds - phase6.seconds <= 1.0, (
        f"Phase 6 landed {plan.target_seconds - phase6.seconds:.2f}s short")


@pytest.mark.parametrize("minutes", [1, 3, 5])
def test_tail_silence_is_the_same_on_both_paths(minutes):
    legacy, phase6 = both(minutes, 0.5)
    assert phase6.tail_seconds == pytest.approx(legacy.tail_seconds, abs=0.35)
    assert phase6.tail_seconds <= pipeline_module.MAX_TAIL_SILENCE + 0.01


# ==========================================================================
# Sentence-shape cases
# ==========================================================================
class ScriptedGenerator:
    """Exactly these sentences, in this order, for both paths."""

    def __init__(self, sentences):
        self.sentences = list(sentences)

    async def stream_sentences(self, plan, notes=None):
        for sentence in self.sentences:
            await asyncio.sleep(0)
            yield sentence

    async def top_up(self, plan, spoken_so_far, words_needed):
        for sentence in self.sentences:
            await asyncio.sleep(0)
            yield sentence


def sized(words: int, marker: str = "w") -> str:
    return " ".join(f"{marker}{i}" for i in range(words - 1)) + " end."


def scripted(sentences, minutes=3):
    return (run("legacy", minutes, 1.0, generator=ScriptedGenerator(sentences)),
            run("phase6", minutes, 1.0, generator=ScriptedGenerator(sentences)))


def _same_prefix(legacy, phase6):
    shared = min(len(legacy.script), len(phase6.script))
    assert phase6.script[:shared] == legacy.script[:shared]


def test_a_short_first_sentence():
    script = ["It was built to ring."] + [sized(12) for _ in range(20)]
    legacy, phase6 = scripted(script)
    _same_prefix(legacy, phase6)
    assert phase6.script[0] == "It was built to ring."


def test_a_long_first_sentence():
    script = [sized(60)] + [sized(12) for _ in range(20)]
    legacy, phase6 = scripted(script)
    _same_prefix(legacy, phase6)
    assert phase6.script[0] == script[0]


def test_a_long_sentence_later_in_the_script():
    script = [sized(9)] + [sized(12) for _ in range(6)] + [sized(90)] \
        + [sized(12) for _ in range(6)]
    legacy, phase6 = scripted(script)
    _same_prefix(legacy, phase6)


def test_a_sentence_longer_than_a_whole_one_minute_episode():
    """It must terminate on both paths, not hang and not be split."""
    legacy, phase6 = scripted([sized(9), sized(400), sized(12)], minutes=1)
    _same_prefix(legacy, phase6)
    assert phase6.truncated and legacy.truncated
    assert phase6.seconds <= 60 + OVERRUN_GRACE
    assert sized(400) not in phase6.script


# ==========================================================================
# Top-ups
# ==========================================================================
@pytest.mark.parametrize("minutes", [1, 3])
def test_topups_off_is_the_default_on_both_paths(minutes):
    legacy, phase6 = both(minutes, 0.5, allow_topups=False)
    assert legacy.topups == 0 and phase6.topups == 0


@pytest.mark.parametrize("minutes", [3, 5])
def test_topups_on_behave_the_same_on_both_paths(minutes):
    plan = plan_episode("q", minutes)
    legacy, phase6 = both(minutes, 0.5, allow_topups=True)
    assert phase6.topups == legacy.topups > 0
    assert phase6.seconds <= plan.target_seconds + OVERRUN_GRACE
    _same_prefix(legacy, phase6)


# ==========================================================================
# answer_first: two streams, two assemblers
# ==========================================================================
def test_two_phase6_pumps_never_share_assembler_state():
    """`_answer_first` runs an instant and a researched stream at once. One
    assembler across both would interleave two scripts into one chunk."""
    async def main():
        pipeline = PodcastPipeline(generator=None, engine=ENGINE, cache=None)
        instant = pipeline._start_phase6(
            ScriptedGenerator([sized(9, "a"), sized(9, "a")]).stream_sentences(None))
        research = pipeline._start_phase6(
            ScriptedGenerator([sized(9, "b"), sized(9, "b")]).stream_sentences(None))
        got = {"instant": [], "research": []}
        for name, pump in (("instant", instant), ("research", research)):
            while True:
                item = await pump.next()
                if item is None:
                    break
                got[name].append(item)
            await _close(pump)
        return got

    got = asyncio.run(main())
    instant_text = " ".join(c.text for c in got["instant"])
    research_text = " ".join(c.text for c in got["research"])
    assert "a0" in instant_text and "b0" not in instant_text
    assert "b0" in research_text and "a0" not in research_text
    for chunk in got["instant"] + got["research"]:
        assert not ("a0" in chunk.text and "b0" in chunk.text)


def test_the_answer_first_share_is_unchanged_by_assembly():
    assert 0.0 < settings.answer_first_share <= 1.0
    assert plan_episode("q", 3).target_seconds * settings.answer_first_share == 90.0


# ==========================================================================
# The first chunk
# ==========================================================================
def test_the_first_complete_thought_is_the_first_thing_synthesised():
    """No batching delay: a five-word opening reaches the work queue alone."""
    async def main():
        pipeline = PodcastPipeline(generator=None, engine=ENGINE, cache=None)
        pump = pipeline._start_phase6(ScriptedGenerator(
            ["It was built to ring."] + [sized(20) for _ in range(6)]
        ).stream_sentences(None))
        first = await pump.next()
        await _close(pump)
        return first

    first = asyncio.run(main())
    assert isinstance(first, AssembledChunk)
    assert first.index == 0 and first.sentences == 1
    assert first.text == "It was built to ring." and first.words == 5
    assert first.reason.startswith("first chunk")
    assert first.held_seconds == pytest.approx(0.0, abs=0.05)


def test_later_chunks_do_batch():
    """The opening is alone; what follows is not - otherwise Phase 6 would be
    legacy with extra machinery."""
    async def main():
        pipeline = PodcastPipeline(generator=None, engine=ENGINE, cache=None)
        pump = pipeline._start_phase6(ScriptedGenerator(
            ["It was built to ring."] + [sized(12) for _ in range(12)]
        ).stream_sentences(None))
        chunks = []
        while True:
            item = await pump.next()
            if item is None:
                break
            chunks.append(item)
        await _close(pump)
        return chunks

    chunks = asyncio.run(main())
    assert chunks[0].sentences == 1
    assert any(c.sentences > 1 for c in chunks[1:])
    assert len(chunks) < 13


# ==========================================================================
# Decoupling
# ==========================================================================
def test_a_slow_consumer_cannot_stop_the_phase6_reader():
    """The property `_start` does not have. With a consumer far slower than the
    model, the reader must still drain the whole stream."""
    read = []

    class Counting(ScriptedGenerator):
        async def stream_sentences(self, plan, notes=None):
            for sentence in self.sentences:
                await asyncio.sleep(0)
                read.append(sentence)
                yield sentence

    async def main(path):
        read.clear()
        pipeline = PodcastPipeline(generator=None, engine=ENGINE, cache=None)
        source = Counting([sized(12) for _ in range(60)]).stream_sentences(None)
        pump = (pipeline._start(source) if path == "legacy"
                else pipeline._start_phase6(source))
        await pump.next()                      # take one item, then stall
        await asyncio.sleep(0.3)
        drained = len(read)
        # Legacy's close hangs here - see `test_closing_a_blocked_pump...` -
        # so it is bounded rather than awaited. Phase 6 returns immediately.
        await _close(pump)
        return drained

    assert asyncio.run(main("phase6")) == 60, "the Phase 6 reader was blocked"
    assert asyncio.run(main("legacy")) < 60, (
        "legacy did not block, so this test proves nothing about the difference")


def test_closing_a_blocked_pump_hangs_on_legacy_and_not_on_phase6():
    """A latent production defect, characterised rather than fixed here.

    `_Pump.close` cancels the producer, but `_start`'s producer ends with
    `finally: await queue.put(None)`. Cancellation is delivered once; that
    `finally` then awaits a full queue nobody will drain, swallowing the
    cancellation, so `await self.task` never returns. `wait_for` does not
    reveal it either - the swallowed cancellation makes `close` return
    normally - which is why this measures the clock.

    Reachable wherever `_speak` leaves early with the producer blocked: a
    truncated episode, a failure mid-stream, and `_answer_first`, which closes
    its instant pump early by design.

    Step 4 may not change the shipped path, so `_start` keeps the defect and it
    is reported. `_start_phase6` does not reproduce it: its producer re-raises
    `CancelledError` instead of sending a sentinel nobody is waiting for. That
    divergence is deliberate, and it is the one behavioural difference in this
    step that is an improvement rather than a cost.
    """
    async def probe(path):
        pipeline = PodcastPipeline(generator=None, engine=ENGINE, cache=None)
        source = ScriptedGenerator([sized(12) for _ in range(60)]
                                   ).stream_sentences(None)
        pump = (pipeline._start(source) if path == "legacy"
                else pipeline._start_phase6(source))
        await pump.next()
        await asyncio.sleep(0.2)
        return await _close(pump, budget=0.4)

    assert asyncio.run(probe("legacy")) >= 0.4, "legacy stopped hanging - update this"
    assert asyncio.run(probe("phase6")) < 0.1, "Phase 6 inherited the hang"


# ==========================================================================
# Unreachability - the safety rule of this step
# ==========================================================================
def test_start_phase6_exists_and_nothing_calls_it():
    assert hasattr(PodcastPipeline, "_start_phase6")
    assert hasattr(PodcastPipeline, "_speak_phase6")
    source = inspect.getsource(pipeline_module)
    for name in ("_start_phase6", "_speak_phase6"):
        calls = re.findall(rf"self\.{name}\s*\(", source)
        assert calls == [], f"{name} is called from production code"


def test_stream_pcm_still_selects_only_the_legacy_pair():
    source = inspect.getsource(PodcastPipeline.stream_pcm)
    assert "self._start(" in source
    assert "_start_phase6" not in source and "_speak_phase6" not in source


def test_answer_first_still_selects_only_the_legacy_pair():
    source = inspect.getsource(PodcastPipeline._answer_first)
    assert "self._start(" in source
    assert "phase6" not in source


def test_the_flag_still_selects_nothing():
    """`STREAMING_PIPELINE=phase6` must remain inert in this step."""
    import importlib

    import config

    os.environ["STREAMING_PIPELINE"] = "phase6"
    os.environ["FAM_IGNORE_DOTENV"] = "1"
    try:
        importlib.reload(config)
        assert config.settings.streaming_pipeline == "phase6"
        source = inspect.getsource(pipeline_module)
        assert "streaming_pipeline" not in source
        legacy, _ = both(1, 1.0)
        assert legacy.seconds == run("legacy", 1, 1.0).seconds
    finally:
        os.environ.pop("STREAMING_PIPELINE", None)
        importlib.reload(config)


def test_legacy_start_is_byte_for_byte_what_trunk_shipped():
    """The safety rule of this step, checked rather than promised."""
    import subprocess

    def method(source: str, name: str) -> str:
        match = re.search(
            rf"\n    def {name}\(.*?(?=\n    async def |\n    def |\nclass |\Z)",
            source, re.S)
        return match.group(0)

    trunk = subprocess.run(
        ["git", "show", "762279b:pipeline.py"],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).stdout
    if not trunk:
        pytest.skip("trunk revision not available in this checkout")
    current = open(pipeline_module.__file__).read()
    assert method(current, "_start") == method(trunk, "_start")
