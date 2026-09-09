"""`STREAMING_PIPELINE`, exercised through the real `stream_pcm` entry point.

Every earlier Phase 6 test drove `_start_phase6` and `_speak_phase6` by hand.
These do not: they set the flag, call the method the API calls, and check what
came out. That is the difference between "the parts work" and "the product can
be switched".

Legacy is the default and the rollback path, so most of what is asserted here
is that nothing moved when the flag is absent.
"""
from __future__ import annotations

import asyncio
import dataclasses
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pipeline as pipeline_module
from audio_utils import pcm_duration
from config import Settings, settings
from pipeline import (MAX_TAIL_SILENCE, MAX_TOPUPS, OVERRUN_GRACE,
                      GenerationStats, PodcastPipeline)
from script_generator import plan_episode
from speech_assembly import AssembledChunk, AssemblyPolicy
from tts import DebugEngine

from tests.test_pipeline import FakeGenerator

ENGINE = DebugEngine()


@pytest.fixture
def flag(monkeypatch):
    """Set `settings.streaming_pipeline` as a running server would see it.

    `Settings` is frozen and read at import, so the object `pipeline` holds is
    replaced - the same mechanism `_answer_first` uses for its plans, and the
    one that keeps `__post_init__` validation in play.
    """
    def choose(value=None):
        replaced = (settings if value is None
                    else dataclasses.replace(settings, streaming_pipeline=value))
        monkeypatch.setattr(pipeline_module, "settings", replaced)
        return replaced
    return choose


def sized(words: int, marker: str = "w") -> str:
    return " ".join(f"{marker}{i}" for i in range(words - 1)) + " end."


def episode(minutes: int = 3, ratio: float = 1.0, generator=None, cache=None):
    """One real `stream_pcm` call. Returns (seconds, stats)."""
    async def main():
        plan = plan_episode("what is the nasdaq", minutes)
        pipe = PodcastPipeline(generator=generator or FakeGenerator(ratio),
                               engine=ENGINE, cache=cache)
        stats = GenerationStats()
        total = 0
        async for chunk in pipe.stream_pcm(plan, stats):
            total += len(chunk)
        return pcm_duration(total, ENGINE.sample_rate), stats

    return asyncio.run(asyncio.wait_for(main(), 30))


def live_tasks(base: set) -> list:
    """What is still running beyond the baseline, named for the failure text."""
    return [getattr(t.get_coro(), "__qualname__", repr(t.get_coro()))
            for t in asyncio.all_tasks()
            if t not in base and t is not asyncio.current_task()]


# ==========================================================================
# 1-4. The selector itself
# ==========================================================================
def test_a_fresh_deployment_with_nothing_set_runs_phase6():
    """The production default, asserted at every layer it passes through.

    Reversed from `..._is_legacy`, and the reversal is the change: an
    installation that has never heard of this setting gets the validated
    architecture. Reaching the older one takes naming it.
    """
    from config import DEFAULT_PIPELINE

    assert "STREAMING_PIPELINE" not in os.environ, (
        "the suite is not clean; this test would be reading the machine")
    assert DEFAULT_PIPELINE == "phase6"
    assert settings.streaming_pipeline == "phase6"
    assert pipeline_module.settings.streaming_pipeline == "phase6"
    assert PodcastPipeline(generator=None, engine=ENGINE,
                           cache=None)._phase6() is True


def test_an_unset_flag_routes_to_the_phase6_pair(flag):
    """Not just the setting - the pump a real request is actually served by."""
    flag(None)
    assert _pump_item_type() is AssembledChunk


def test_legacy_routes_to_the_legacy_pair(flag):
    flag("legacy")
    assert _pump_item_type() is str


def test_phase6_routes_to_the_phase6_pair(flag):
    """The pump carries assembled chunks, which only `_start_phase6` makes."""
    from speech_assembly import AssembledChunk

    flag("phase6")
    assert _pump_item_type() is AssembledChunk


def test_switching_back_to_legacy_restores_legacy_behaviour(flag):
    """Rollback. The same request, the same numbers as before the switch."""
    flag("legacy")
    before = episode(3, 1.0)
    flag("phase6")
    switched = episode(3, 1.0)
    flag("legacy")
    after = episode(3, 1.0)

    assert after[0] == pytest.approx(before[0], abs=0.01)
    assert after[1].script == before[1].script
    assert after[1].words == before[1].words
    assert after[1].truncated == before[1].truncated
    assert switched[1].words != before[1].words, "the switch did nothing"


async def _sentences(items):
    for item in items:
        await asyncio.sleep(0)
        yield item


def _pump_item_type():
    """What one pump yields. Built inside the loop: it starts a task."""
    async def main():
        pipe = PodcastPipeline(generator=None, engine=ENGINE, cache=None)
        pump = pipe._pump_for(_sentences(["One two three four five."]))
        first = await pump.next()
        await pump.close()
        return type(first)

    return _bounded(main(), 5)


# ==========================================================================
# 5, 11. The grid, through the real entry point
# ==========================================================================
GRID = [(m, r) for m in (1, 3, 5) for r in (0.5, 1.0, 1.6)]


@pytest.mark.parametrize("minutes, ratio", GRID)
def test_the_duration_ceiling_holds_on_both_paths(flag, minutes, ratio):
    plan = plan_episode("q", minutes)
    for value in ("legacy", "phase6"):
        flag(value)
        seconds, _ = episode(minutes, ratio)
        assert seconds <= plan.target_seconds + OVERRUN_GRACE, value


@pytest.mark.parametrize("minutes, ratio", GRID)
def test_the_spoken_text_is_the_same_script_in_the_same_order(flag, minutes, ratio):
    flag("legacy")
    _, legacy = episode(minutes, ratio)
    flag("phase6")
    _, phase6 = episode(minutes, ratio)
    shared = min(len(legacy.script), len(phase6.script))
    assert phase6.script[:shared] == legacy.script[:shared]
    assert len(phase6.script) >= len(legacy.script)


@pytest.mark.parametrize("minutes, ratio", GRID)
def test_the_accounting_describes_what_was_spoken(flag, minutes, ratio):
    for value in ("legacy", "phase6"):
        flag(value)
        _, stats = episode(minutes, ratio)
        assert stats.sentences == len(stats.script)
        assert stats.words == sum(len(s.split()) for s in stats.script)


def test_the_three_minute_surplus_is_characterised_not_removed(flag):
    """The one difference worth naming.

    Phase 6 lays `SENTENCE_GAP` once per chunk rather than once per sentence,
    so it spends less of the budget on silence and reaches further into the
    script. At three minutes that is about ten more words. It is inside the
    ceiling, it is a consequence of batching rather than a bug, and it is a
    listening question for later - so it is pinned here rather than tuned away
    to force numerical identity with legacy.
    """
    plan = plan_episode("q", 3)
    flag("legacy")
    legacy_seconds, legacy = episode(3, 1.0)
    flag("phase6")
    phase6_seconds, phase6 = episode(3, 1.0)

    assert legacy.words == 442 and legacy.sentences == 51
    assert phase6.words == 452 and phase6.sentences == 52
    assert phase6.words - legacy.words == 10
    assert phase6_seconds <= plan.target_seconds + OVERRUN_GRACE
    assert legacy_seconds <= plan.target_seconds + OVERRUN_GRACE
    assert phase6.words - legacy.words <= AssemblyPolicy().max_words


@pytest.mark.parametrize("minutes", [1, 3, 5])
def test_tail_silence_is_bounded_on_both_paths(flag, minutes):
    for value in ("legacy", "phase6"):
        flag(value)
        seconds, stats = episode(minutes, 0.5)
        plan = plan_episode("q", minutes)
        assert seconds < plan.target_seconds
        assert plan.target_seconds - seconds > MAX_TAIL_SILENCE - 0.01, value


# ==========================================================================
# 6. The first complete thought
# ==========================================================================
def test_the_first_complete_thought_is_spoken_alone_under_phase6(flag):
    """Not batched with the sentence after it, through the real entry point:
    `stats.script[0]` is the opening sentence and nothing else."""
    class Opening:
        async def stream_sentences(self, plan, notes=None):
            yield "It was built to ring."
            for _ in range(30):
                await asyncio.sleep(0)
                yield sized(20)

        async def top_up(self, plan, spoken_so_far, words_needed):
            async for s in self.stream_sentences(plan):
                yield s

    for value in ("legacy", "phase6"):
        flag(value)
        _, stats = episode(3, generator=Opening())
        assert stats.script[0] == "It was built to ring.", value
        assert stats.script[1] != "It was built to ring."


# ==========================================================================
# 7. Decoupling, through stream_pcm
# ==========================================================================
def test_the_reader_runs_ahead_under_phase6_and_not_under_legacy(flag):
    """The architectural difference, observed on the real path: with a slow
    voice, Phase 6 reads the whole script while legacy stalls behind its
    queue."""
    read = []

    class Counted:
        async def stream_sentences(self, plan, notes=None):
            for _ in range(60):
                await asyncio.sleep(0)
                read.append(1)
                yield sized(12)

        async def top_up(self, plan, spoken_so_far, words_needed):
            async for s in self.stream_sentences(plan):
                yield s

    class Slow(DebugEngine):
        async def synth(self, text, wpm, voice=None):
            await asyncio.sleep(0.02)
            return await DebugEngine.synth(self, text, wpm, voice)

    async def measure():
        plan = plan_episode("q", 5)
        pipe = PodcastPipeline(generator=Counted(), engine=Slow(), cache=None)
        stats = GenerationStats()
        stream = pipe.stream_pcm(plan, stats)
        await stream.__anext__()
        await asyncio.sleep(0.25)
        seen = len(read)
        await stream.aclose()
        return seen

    read.clear()
    flag("legacy")
    legacy_seen = asyncio.run(asyncio.wait_for(measure(), 20))
    read.clear()
    flag("phase6")
    phase6_seen = asyncio.run(asyncio.wait_for(measure(), 20))

    assert phase6_seen > legacy_seen, (
        f"phase6 read {phase6_seen}, legacy read {legacy_seen} - not decoupled")


# ==========================================================================
# 8, 10. Teardown, cancellation and disconnect
# ==========================================================================
class Endless:
    async def stream_sentences(self, plan, notes=None):
        for _ in range(4000):
            await asyncio.sleep(0)
            yield sized(12)

    async def top_up(self, plan, spoken_so_far, words_needed):
        async for s in self.stream_sentences(plan):
            yield s


@pytest.mark.parametrize("value", ["legacy", "phase6"])
def test_a_truncated_episode_leaves_no_task_behind(flag, value):
    flag(value)

    async def main():
        base = set(asyncio.all_tasks())
        plan = plan_episode("q", 1)
        pipe = PodcastPipeline(generator=Endless(), engine=ENGINE, cache=None)
        stats = GenerationStats()
        total = 0
        async for chunk in pipe.stream_pcm(plan, stats):
            total += len(chunk)
        for _ in range(5):
            await asyncio.sleep(0)
        return total, stats.truncated, live_tasks(base)

    total, truncated, alive = _bounded(main())
    assert total > 0 and truncated
    assert alive == [], f"{value} left tasks running: {alive}"


@pytest.mark.parametrize("value", ["legacy", "phase6"])
def test_a_listener_disconnecting_tears_the_request_down(flag, value):
    """`app.py` abandons the generator when the client goes away. Whatever the
    architecture, nothing may keep running afterwards."""
    flag(value)

    async def main():
        base = set(asyncio.all_tasks())
        plan = plan_episode("q", 5)
        pipe = PodcastPipeline(generator=Endless(), engine=ENGINE, cache=None)
        stats = GenerationStats()
        stream = pipe.stream_pcm(plan, stats)
        await stream.__anext__()
        await asyncio.sleep(0.05)
        await stream.aclose()
        # `aclose()` itself schedules an `async_generator_athrow` task; a few
        # extra turns let it retire so it is not counted as a pipeline leak.
        for _ in range(20):
            await asyncio.sleep(0)
        return live_tasks(base)

    assert _bounded(main()) == [], f"{value} left tasks after a disconnect"


@pytest.mark.parametrize("value", ["legacy", "phase6"])
def test_cancelling_the_request_task_tears_it_down(flag, value):
    flag(value)

    async def main():
        base = set(asyncio.all_tasks())
        plan = plan_episode("q", 5)
        pipe = PodcastPipeline(generator=Endless(), engine=ENGINE, cache=None)
        stats = GenerationStats()

        async def serve():
            async for _ in pipe.stream_pcm(plan, stats):
                pass

        task = asyncio.create_task(serve())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        for _ in range(5):
            await asyncio.sleep(0)
        return live_tasks(base)

    assert _bounded(main()) == [], f"{value} left tasks after cancellation"


@pytest.mark.parametrize("value", ["legacy", "phase6"])
def test_a_generator_failure_still_reaches_the_caller(flag, value):
    flag(value)

    class Failing:
        async def stream_sentences(self, plan, notes=None):
            yield sized(10)
            raise RuntimeError("the model call failed")

    async def main():
        base = set(asyncio.all_tasks())
        plan = plan_episode("q", 3)
        pipe = PodcastPipeline(generator=Failing(), engine=ENGINE, cache=None)
        with pytest.raises(RuntimeError, match="the model call failed"):
            async for _ in pipe.stream_pcm(plan, GenerationStats()):
                pass
        for _ in range(5):
            await asyncio.sleep(0)
        return live_tasks(base)

    assert _bounded(main()) == []


def _bounded(coro, timeout: float = 20.0):
    """Run on a loop the test closes itself, so a leak fails rather than hangs."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(asyncio.wait_for(coro, timeout))
    finally:
        for task in asyncio.all_tasks(loop):
            task.cancel()
        loop.close()


# ==========================================================================
# 9. answer_first
# ==========================================================================
def test_answer_first_under_phase6_keeps_its_two_streams_apart(flag):
    """Two pumps, so two buffers and two assemblers. No chunk may hold text
    from both halves, and the instant half must come first.

    Asks for the cover explicitly. It is no longer the default - it follows
    RESEARCH_BACKEND, and Exa needs no covering - so a test about the cover has
    to turn it on rather than inherit it.
    """
    flag("phase6")
    import dataclasses

    import pipeline as _pipeline
    _pipeline.settings = dataclasses.replace(_pipeline.settings,
                                             answer_first=True)

    class TwoHalves:
        async def stream_sentences(self, plan, notes=None):
            marker = "b" if getattr(plan, "role", "") == "continuation" else "a"
            if marker == "b":
                for _ in range(4):
                    await asyncio.sleep(0)
            for _ in range(12):
                await asyncio.sleep(0)
                yield sized(12, marker)

        async def top_up(self, plan, spoken_so_far, words_needed):
            async for s in self.stream_sentences(plan):
                yield s

    async def main():
        plan = plan_episode("q", 3)
        plan = dataclasses.replace(plan, search=True)
        pipe = PodcastPipeline(generator=TwoHalves(), engine=ENGINE, cache=None)
        stats = GenerationStats()
        async for _ in pipe.stream_pcm(plan, stats):
            pass
        return stats

    stats = _bounded(main())
    assert stats.answered_first
    for sentence in stats.script:
        assert not ("a0" in sentence and "b0" in sentence), (
            f"the two halves were assembled into one chunk: {sentence!r}")
    assert stats.script[0].startswith("a0"), "the instant half did not go first"
    assert any(s.startswith("b0") for s in stats.script), "research never spoke"


def test_answer_first_builds_a_separate_pump_for_each_half():
    import inspect

    source = inspect.getsource(PodcastPipeline._answer_first)
    assert source.count("self._pump_for(") == 2


# ==========================================================================
# Top-ups and cache, unchanged by the selector
# ==========================================================================
@pytest.mark.parametrize("value", ["legacy", "phase6"])
def test_topups_still_fill_a_short_episode(flag, monkeypatch, value):
    flag(value)
    monkeypatch.setattr(pipeline_module, "settings",
                        dataclasses.replace(pipeline_module.settings,
                                            allow_topups=True))
    plan = plan_episode("q", 3)
    seconds, stats = episode(3, 0.5)
    assert 0 < stats.topups <= MAX_TOPUPS
    assert seconds <= plan.target_seconds + OVERRUN_GRACE
    assert seconds > 98.0, "the top-up added nothing"


@pytest.mark.parametrize("value", ["legacy", "phase6"])
def test_a_cached_script_replays_through_the_selected_path(flag, value):
    from cache import MemoryScriptCache

    flag(value)
    store = MemoryScriptCache()
    first = episode(3, 1.0, cache=store)
    second = episode(3, 1.0, cache=store)
    assert second[1].cache == "hit"
    assert second[1].script == first[1].script
    assert second[0] == pytest.approx(first[0], abs=0.5)


# ==========================================================================
# The engine boundary: what actually reaches the voice, and when
# ==========================================================================
class Recording(DebugEngine):
    """Records every synthesis request, in order, with its arrival time.

    A stand-in for Chatterbox at the one place that matters: the boundary
    where Phase 6 hands text to the voice. It needs no GPU, so the invariant
    is checked on every run rather than only on a rented card.
    """

    def __init__(self):
        self.calls = []

    async def synth(self, text, wpm, voice=None):
        self.calls.append({"text": text, "wpm": wpm,
                           "at": asyncio.get_running_loop().time()})
        return await DebugEngine.synth(self, text, wpm, voice)


def test_the_first_complete_thought_reaches_the_engine_un_batched(flag):
    """The Phase 6 first-chunk rule, asserted where the voice sees it: call
    one is the opening sentence alone - not merged with the next, not held for
    a word count, not delayed by a batching threshold."""
    class Opening:
        async def stream_sentences(self, plan, notes=None):
            yield "It was built to ring."
            for _ in range(30):
                await asyncio.sleep(0)
                yield sized(20)

        async def top_up(self, plan, spoken_so_far, words_needed):
            async for s in self.stream_sentences(plan):
                yield s

    async def main():
        engine = Recording()
        plan = plan_episode("q", 3)
        pipe = PodcastPipeline(generator=Opening(), engine=engine, cache=None)
        async for _ in pipe.stream_pcm(plan, GenerationStats()):
            pass
        return engine.calls

    flag("phase6")
    calls = _bounded(main())
    assert calls[0]["text"] == "It was built to ring."
    assert len(calls) < 31, "every sentence was sent on its own; nothing batched"
    assert any(len(c["text"].split()) > 20 for c in calls[1:]), (
        "no chunk after the first was assembled")


def test_the_first_call_is_the_first_thing_the_engine_is_asked_for(flag):
    """No warm-up utterance, no preamble, nothing ahead of the listener's
    opening sentence in the queue."""
    class Opening:
        async def stream_sentences(self, plan, notes=None):
            yield "The clock has no face."
            for _ in range(10):
                await asyncio.sleep(0)
                yield sized(20)

        async def top_up(self, plan, spoken_so_far, words_needed):
            async for s in self.stream_sentences(plan):
                yield s

    async def main():
        engine = Recording()
        pipe = PodcastPipeline(generator=Opening(), engine=engine, cache=None)
        async for _ in pipe.stream_pcm(plan_episode("q", 3), GenerationStats()):
            pass
        return engine.calls

    for value in ("legacy", "phase6"):
        flag(value)
        calls = _bounded(main())
        assert calls[0]["text"] == "The clock has no face.", value


def test_a_rate_ignoring_engine_still_honours_the_duration_ceiling(flag):
    """Chatterbox has no speaking-rate control, so pacing cannot help it.
    Length has to hold on the budget and the trim alone - which is the trade
    accepted when Chatterbox became the production voice."""
    class Deaf(DebugEngine):
        """Ignores wpm entirely, as Chatterbox does."""

        async def synth(self, text, wpm, voice=None):
            return await DebugEngine.synth(self, text, 150.0, voice)

    async def main(minutes):
        plan = plan_episode("q", minutes)
        pipe = PodcastPipeline(generator=FakeGenerator(1.6), engine=Deaf(),
                               cache=None)
        stats = GenerationStats()
        total = 0
        async for chunk in pipe.stream_pcm(plan, stats):
            total += len(chunk)
        return pcm_duration(total, Deaf().sample_rate), plan.target_seconds

    for value in ("legacy", "phase6"):
        flag(value)
        for minutes in (1, 3, 5):
            seconds, target = _bounded(main(minutes))
            assert seconds <= target + OVERRUN_GRACE, (
                f"{value} at {minutes} min ran to {seconds:.2f}s over {target}s")
