"""Shutting down the Phase 6 task graph, proved by counting tasks.

Step 5 fixed `_start`'s sentinel deadlock. `_start_phase6` had a separate
lifecycle defect: its producer drove the assembler's timer with
`asyncio.wait_for(buffer.get(), tick)`, and `wait_for` can absorb the outer
cancellation it is meant to propagate - it cancels its inner task and, when
that inner task finishes at the same moment, returns a result instead of
re-raising. The producer then continued its loop having eaten the cancel, and
`_speak_phase6` was left with a producer that would not die.

The episode itself always finished; the process hung at loop shutdown, waiting
on an orphan. That is why it looked like a truncation hang.

Every test here asserts on `asyncio.all_tasks()`, because a teardown fix that
leaves something running is not a fix. None of them can hang the suite: each is
bounded by `wait_for` at the outermost level only.
"""
from __future__ import annotations

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_utils import PaceController
from pipeline import GenerationStats, PodcastPipeline
from script_generator import ScriptNotes, plan_episode
from tts import DebugEngine

ENGINE = DebugEngine()


def sized(words: int) -> str:
    return " ".join(f"w{i}" for i in range(words - 1)) + " end."


class Script:
    """A generator with a fixed number of sentences, yielding cooperatively."""

    def __init__(self, count: int, fail_after: int = 0):
        self.count, self.fail_after = count, fail_after

    async def stream_sentences(self, plan, notes=None):
        for index in range(self.count):
            await asyncio.sleep(0)
            if self.fail_after and index == self.fail_after:
                raise RuntimeError("the model call failed")
            yield sized(12)


def run_bounded(coro, timeout: float = 10.0):
    """Run on a loop we close ourselves, without draining what is left.

    `asyncio.run` finishes by cancelling the remaining tasks and awaiting
    them - and the defect under test is a task that absorbs cancellation, so
    that await never returns and a failing test would hang the suite rather
    than fail it. Here the loop is closed regardless, and the leak is reported
    by the assertion instead.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(asyncio.wait_for(coro, timeout))
    finally:
        for task in asyncio.all_tasks(loop):
            task.cancel()
        loop.close()


def baseline() -> set:
    """The tasks that already existed. Anything beyond this is Phase 6's."""
    return set(asyncio.all_tasks())


async def survivors(base: set) -> list:
    """What Phase 6 left running, recorded and then forcibly cleaned up.

    The cleanup is not the fix and hides nothing: the list is captured first
    and asserted on. It exists because a leaked task stops `asyncio.run` from
    closing its loop, so without it a failing test would hang the suite
    instead of failing it.
    """
    for _ in range(5):
        await asyncio.sleep(0)
    alive = [t for t in asyncio.all_tasks()
             if t not in base and t is not asyncio.current_task()]
    for task in alive:
        task.cancel()
    if alive:
        # Bounded: a task that absorbs cancellation will not finish, which is
        # the defect. `run_bounded` closes the loop either way.
        await asyncio.wait(alive, timeout=0.2)
    return [t.get_coro().__qualname__ for t in alive]


def episode(minutes: int, count: int, fail_after: int = 0, stop_after: int = 0):
    """Run one Phase 6 episode and report what survived it."""
    async def main():
        base = baseline()
        pipe = PodcastPipeline(generator=Script(count, fail_after),
                               engine=ENGINE, cache=None)
        plan = plan_episode("q", minutes)
        stats = GenerationStats()
        pace = PaceController(target_seconds=float(plan.target_seconds),
                              total_words=plan.word_budget,
                              sample_rate=ENGINE.sample_rate)
        pump = pipe._start_phase6(
            pipe.generator.stream_sentences(plan, ScriptNotes()))
        total = 0
        speaker = pipe._speak_phase6(pump, pace, stats)
        try:
            async for audio in speaker:
                total += len(audio)
                if stop_after and total >= stop_after:
                    break
        finally:
            await speaker.aclose()
        return total, stats, pump, await survivors(base)

    return run_bounded(main())


# --------------------------------------------------------------------------
# 1-2. The two normal exits
# --------------------------------------------------------------------------
def test_a_truncated_episode_leaves_nothing_running():
    """The reproduction. Pre-fix the producer survives and the loop cannot
    close, so this fails on the task count rather than by hanging."""
    total, stats, pump, alive = episode(1, count=400)
    assert stats.truncated and total > 0
    assert pump.task.done(), "the producer outlived the episode"
    assert alive == [], f"tasks still running: {alive}"


def test_a_completed_episode_leaves_nothing_running():
    total, stats, pump, alive = episode(5, count=20)
    assert not stats.truncated and total > 0
    assert pump.task.done()
    assert alive == []


# --------------------------------------------------------------------------
# 3. Producer failure
# --------------------------------------------------------------------------
def test_a_producer_exception_propagates_and_leaves_nothing_running():
    async def main():
        base = baseline()
        pipe = PodcastPipeline(generator=Script(50, fail_after=3),
                               engine=ENGINE, cache=None)
        plan = plan_episode("q", 3)
        stats = GenerationStats()
        pace = PaceController(target_seconds=180.0, total_words=450,
                              sample_rate=ENGINE.sample_rate)
        pump = pipe._start_phase6(
            pipe.generator.stream_sentences(plan, ScriptNotes()))
        with pytest.raises(RuntimeError, match="the model call failed"):
            async for _ in pipe._speak_phase6(pump, pace, stats):
                pass
        done = pump.task.done()
        return done, await survivors(base)

    done, alive = run_bounded(main())
    assert done and alive == []


# --------------------------------------------------------------------------
# 4. Consumer walks away mid-episode
# --------------------------------------------------------------------------
def test_a_consumer_that_stops_early_tears_down_the_whole_graph():
    """A listener closing the tab. Every child - reader, buffered get,
    producer - has to go with it."""
    total, stats, pump, alive = episode(5, count=400, stop_after=200_000)
    assert total > 0
    assert pump.task.done()
    assert alive == []


def test_cancelling_the_consuming_task_tears_down_the_whole_graph():
    async def main():
        base = baseline()
        pipe = PodcastPipeline(generator=Script(400), engine=ENGINE, cache=None)
        plan = plan_episode("q", 5)
        stats = GenerationStats()
        pace = PaceController(target_seconds=300.0, total_words=750,
                              sample_rate=ENGINE.sample_rate)
        pump = pipe._start_phase6(
            pipe.generator.stream_sentences(plan, ScriptNotes()))

        async def consume():
            async for _ in pipe._speak_phase6(pump, pace, stats):
                pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        done = pump.task.done()
        return done, await survivors(base)

    done, alive = run_bounded(main())
    assert done, "the producer outlived a cancelled consumer"
    assert alive == []


# --------------------------------------------------------------------------
# 5-6. Closing the pump directly, and twice
# --------------------------------------------------------------------------
def test_closing_a_phase6_pump_directly_is_prompt_and_complete():
    async def main():
        base = baseline()
        pipe = PodcastPipeline(generator=Script(400), engine=ENGINE, cache=None)
        pump = pipe._start_phase6(
            pipe.generator.stream_sentences(plan_episode("q", 5), ScriptNotes()))
        await pump.next()
        await asyncio.sleep(0.05)
        await pump.close()
        done = pump.task.done()
        return done, await survivors(base)

    done, alive = run_bounded(main(), 5)
    assert done and alive == []


def test_closing_a_phase6_pump_twice_is_safe():
    async def main():
        base = baseline()
        pipe = PodcastPipeline(generator=Script(400), engine=ENGINE, cache=None)
        pump = pipe._start_phase6(
            pipe.generator.stream_sentences(plan_episode("q", 5), ScriptNotes()))
        await pump.next()
        await pump.close()
        await pump.close()
        done = pump.task.done()
        return done, await survivors(base)

    done, alive = run_bounded(main(), 5)
    assert done and alive == []


# --------------------------------------------------------------------------
# The invariants teardown must not have cost
# --------------------------------------------------------------------------
def test_the_first_chunk_still_leaves_alone_and_immediately():
    async def main():
        base = baseline()
        pipe = PodcastPipeline(generator=None, engine=ENGINE, cache=None)

        async def opening():
            yield "It was built to ring."
            for _ in range(30):
                await asyncio.sleep(0)
                yield sized(20)

        pump = pipe._start_phase6(opening())
        first = await pump.next()
        await pump.close()
        return first, await survivors(base)

    first, alive = run_bounded(main(), 5)
    assert first.text == "It was built to ring." and first.sentences == 1
    assert first.words == 5 and first.reason.startswith("first chunk")
    assert alive == []


def test_the_reader_still_runs_ahead_of_a_slow_consumer():
    read = []

    async def counted():
        for _ in range(60):
            await asyncio.sleep(0)
            read.append(1)
            yield sized(12)

    async def main():
        base = baseline()
        pipe = PodcastPipeline(generator=None, engine=ENGINE, cache=None)
        pump = pipe._start_phase6(counted())
        await pump.next()
        await asyncio.sleep(0.3)
        drained = len(read)
        await pump.close()
        return drained, await survivors(base)

    drained, alive = run_bounded(main(), 5)
    assert drained == 60, "the Claude reader was blocked by the work queue"
    assert alive == []
