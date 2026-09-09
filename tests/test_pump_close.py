"""Closing a sentence pump must never hang, whatever the producer is doing.

`_Pump.close()` cancels the producer and awaits it. `_start`'s producer ended
with `finally: await queue.put(None)`, so on the close path the cancellation
was delivered, the `finally` ran, and that put suspended on a queue nobody was
draining - swallowing the cancellation and blocking forever. `close()` then
never returned.

Reachable wherever `_speak` leaves early with the producer still mid-stream: a
truncated episode, a failure part-way through, and every `_answer_first`
handover, which closes its instant pump early by design.

Every close below is bounded by the clock. `wait_for` alone would not reveal
the bug - `close()` swallows the timeout cancellation and returns normally - so
these measure elapsed time and inspect the task, never the return.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_utils import PaceController
from pipeline import GenerationStats, PodcastPipeline
from script_generator import ScriptNotes, plan_episode
from tts import DebugEngine

ENGINE = DebugEngine()
#: Longer than any correct close, far shorter than a hang.
BUDGET = 0.5


async def timed_close(pump) -> float:
    """Close, bounded. Returns seconds taken; a hang returns >= BUDGET."""
    started = time.perf_counter()
    try:
        await asyncio.wait_for(asyncio.shield(pump.close()), BUDGET)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass
    return time.perf_counter() - started


def sized(words: int) -> str:
    return " ".join(f"w{i}" for i in range(words - 1)) + " end."


async def slow_stream(count: int = 200, delay: float = 0.0):
    for _ in range(count):
        await asyncio.sleep(delay)
        yield sized(12)


def pipeline() -> PodcastPipeline:
    return PodcastPipeline(generator=None, engine=ENGINE, cache=None)


# --------------------------------------------------------------------------
# 1. The defect itself
# --------------------------------------------------------------------------
def test_closing_while_the_producer_is_blocked_terminates_promptly():
    """The regression test. Against the pre-fix producer this never returns."""
    async def main():
        pump = pipeline()._start(slow_stream())
        await pump.next()                    # take one, leave the queue full
        await asyncio.sleep(0.05)
        elapsed = await timed_close(pump)
        return elapsed, pump.task.done()

    elapsed, done = asyncio.run(main())
    assert elapsed < BUDGET, f"close took {elapsed:.2f}s - it hung"
    assert done, "the producer task outlived close()"


def test_no_producer_task_survives_a_close():
    """Not fixed by leaving an orphan running: after close the task is
    finished, and nothing of it is left pending on the loop."""
    async def main():
        pump = pipeline()._start(slow_stream())
        await pump.next()
        await asyncio.sleep(0.05)
        await timed_close(pump)
        await asyncio.sleep(0)               # let the loop settle
        return pump.task.done(), pump.task.cancelled(), [
            t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    done, cancelled, alive = asyncio.run(main())
    assert done and cancelled
    assert alive == [], f"tasks still running after close: {alive}"


# --------------------------------------------------------------------------
# 2-3. The paths that already worked must keep working
# --------------------------------------------------------------------------
def test_closing_a_normally_draining_pump_terminates_promptly():
    async def main():
        pump = pipeline()._start(slow_stream(count=6))
        seen = []
        for _ in range(3):
            seen.append(await pump.next())
        elapsed = await timed_close(pump)
        return seen, elapsed, pump.task.done()

    seen, elapsed, done = asyncio.run(main())
    assert len(seen) == 3 and all(isinstance(s, str) for s in seen)
    assert elapsed < BUDGET and done


def test_closing_after_the_producer_finished_is_safe():
    """The sentinel still arrives on a stream that ran to the end."""
    async def main():
        pump = pipeline()._start(slow_stream(count=3))
        drained = []
        while True:
            item = await pump.next()
            if item is None:
                break
            drained.append(item)
        elapsed = await timed_close(pump)
        return drained, elapsed, pump.task.done()

    drained, elapsed, done = asyncio.run(main())
    assert len(drained) == 3, "the sentinel must still end a completed stream"
    assert elapsed < BUDGET and done


# --------------------------------------------------------------------------
# 4. Exceptions still reach the consumer
# --------------------------------------------------------------------------
def test_a_producer_exception_still_reaches_the_consumer():
    async def broken():
        yield sized(10)
        raise RuntimeError("the model call failed")

    async def main():
        pump = pipeline()._start(broken())
        first = await pump.next()
        second = await pump.next()
        elapsed = await timed_close(pump)
        return first, second, elapsed, pump.task.done()

    first, second, elapsed, done = asyncio.run(main())
    assert isinstance(first, str)
    assert isinstance(second, RuntimeError) and "model call failed" in str(second)
    assert elapsed < BUDGET and done


def test_the_speaking_loop_still_raises_a_producer_failure():
    """End to end: `_speak` re-raises what the producer put on the queue, and
    its own `finally: await pump.close()` does not swallow or hang on it."""
    async def broken(plan, notes=None):
        yield sized(10)
        raise RuntimeError("the model call failed")

    class Failing:
        async def stream_sentences(self, plan, notes=None):
            async for s in broken(plan, notes):
                yield s

    async def main():
        pipe = PodcastPipeline(generator=Failing(), engine=ENGINE, cache=None)
        plan = plan_episode("q", 3)
        stats = GenerationStats()
        pace = PaceController(target_seconds=180.0, total_words=450,
                              sample_rate=ENGINE.sample_rate)
        pump = pipe._start(pipe.generator.stream_sentences(plan, ScriptNotes()))
        with pytest.raises(RuntimeError, match="model call failed"):
            async for _ in pipe._speak(pump, pace, stats):
                pass
        return pump.task.done()

    assert asyncio.run(asyncio.wait_for(main(), 5))


# --------------------------------------------------------------------------
# 5. Closing twice
# --------------------------------------------------------------------------
def test_closing_twice_is_safe():
    async def main():
        pump = pipeline()._start(slow_stream())
        await pump.next()
        await asyncio.sleep(0.05)
        first = await timed_close(pump)
        second = await timed_close(pump)
        return first, second, pump.task.done()

    first, second, done = asyncio.run(main())
    assert first < BUDGET and second < BUDGET and done


# --------------------------------------------------------------------------
# 6-7. The two real paths that reach the defect
# --------------------------------------------------------------------------
class Endless:
    """A generator with far more material than any episode can use, so the
    producer is always still mid-stream when the consumer gives up."""

    async def stream_sentences(self, plan, notes=None):
        for _ in range(4000):
            await asyncio.sleep(0)
            yield sized(12)

    async def top_up(self, plan, spoken_so_far, words_needed, notes=None):
        async for s in self.stream_sentences(plan):
            yield s


def test_a_truncated_episode_does_not_hang():
    """`_speak` breaks on truncation and closes a pump whose producer still
    has thousands of sentences to give."""
    async def main():
        pipe = PodcastPipeline(generator=Endless(), engine=ENGINE, cache=None)
        plan = plan_episode("q", 1)
        stats = GenerationStats()
        total = 0
        async for chunk in pipe.stream_pcm(plan, stats):
            total += len(chunk)
        return stats.truncated, total

    truncated, total = asyncio.run(asyncio.wait_for(main(), 10))
    assert truncated and total > 0


def test_the_answer_first_handover_does_not_hang():
    """`_answer_first` closes its instant pump the moment research is ready.
    That producer is mid-stream by construction."""
    class Instant(Endless):
        async def stream_sentences(self, plan, notes=None):
            if getattr(plan, "role", "") == "continuation":
                for _ in range(30):
                    await asyncio.sleep(0)
                    yield sized(12)
                return
            async for s in Endless.stream_sentences(self, plan, notes):
                yield s

    async def main():
        pipe = PodcastPipeline(generator=Instant(), engine=ENGINE, cache=None)
        plan = plan_episode("q", 3)
        plan = type(plan)(**{**plan.__dict__, "search": True})
        stats = GenerationStats()
        pace = PaceController(target_seconds=180.0, total_words=450,
                              sample_rate=ENGINE.sample_rate)
        total = 0
        async for chunk in pipe._answer_first(plan, pace, stats, ScriptNotes()):
            total += len(chunk)
        return stats.answered_first, total

    answered, total = asyncio.run(asyncio.wait_for(main(), 10))
    assert answered and total > 0
