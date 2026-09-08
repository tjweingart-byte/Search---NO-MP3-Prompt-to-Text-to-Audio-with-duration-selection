"""The buffer that stops synthesis backpressure reaching the Claude reader.

`pipeline._start` puts sentences into an `asyncio.Queue(maxsize=QUEUE_DEPTH)`
that the synthesiser drains, so a slow voice stops the reader. The Phase 6
RTX 4090 run measured what that costs: a 12.5s Claude stream reported as 66.7s,
64.3s of it our own backpressure. These tests hold the replacement to the
property that matters - the reader keeps going however slow the consumer is.

Nothing here needs an API key, a voice engine or a network.
"""
from __future__ import annotations

import asyncio
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from script_buffer import (DEFAULT_MAX_CHARACTERS, ScriptBuffer,
                           assemble_chunks)
from speech_assembly import AssemblyPolicy

SCRIPT = [
    "The oldest working clock in Europe has no face, and for six hundred years "
    "nobody thought that was strange.",
    "It was built to ring.",
    "Not to be read.",
    "Salisbury Cathedral has kept it turning since about thirteen eighty-six, "
    "through a civil war and two restorations.",
    "Nobody there needed the minute.",
    "They needed to know when to pray.",
    "So what changed?",
    "Clocks grew faces when somebody else started owning your hours, which is a "
    "sentence about factories more than it is about horology.",
    "The face is the invoice.",
    "It still is.",
]


async def _stream(script=SCRIPT, delay=0.005):
    for sentence in script:
        await asyncio.sleep(delay)
        yield sentence


def _collect(stream=None, consume=0.0, **kwargs) -> list:
    async def run():
        out = []
        async for chunk in assemble_chunks(stream or _stream(), **kwargs):
            if consume:
                await asyncio.sleep(consume)      # a slow voice
            out.append(chunk)
        return out
    return asyncio.run(run())


# --------------------------------------------------------------------------
# the property the whole module exists for
# --------------------------------------------------------------------------
def test_a_slow_consumer_never_stops_the_reader():
    """Synthesis an order of magnitude slower than the model. The reader must
    finish the stream anyway, and the buffer must record no blocking."""
    buffer = ScriptBuffer()
    chunks = _collect(stream=_stream(SCRIPT * 3, delay=0.001), consume=0.15,
                      buffer=buffer)
    assert chunks
    assert buffer.blocked_seconds == 0.0
    assert 0 < buffer.peak_characters < DEFAULT_MAX_CHARACTERS


def test_the_reader_gets_ahead_of_a_slow_consumer():
    """The observable sign of decoupling: text piles up in the buffer while the
    consumer is still on an early chunk. A coupled reader could not."""
    buffer = ScriptBuffer()
    _collect(stream=_stream(SCRIPT * 3, delay=0.001), consume=0.12, buffer=buffer)
    assert buffer.peak_sentences > 1, "the reader never got ahead; not decoupled"


def test_the_bound_is_characters_and_blocking_is_recorded_if_it_is_reached():
    """The bound is real. A buffer small enough to reach it must say so rather
    than silently stalling the reader - that is the failure being designed out."""
    buffer = ScriptBuffer(max_characters=120)
    chunks = _collect(stream=_stream(SCRIPT * 2, delay=0.0), consume=0.02,
                      buffer=buffer)
    assert chunks
    assert buffer.blocked_seconds > 0
    assert buffer.peak_characters <= 120 + max(len(s) for s in SCRIPT)


def test_the_default_bound_holds_many_episodes():
    """A three-minute FAM episode is about 2,700 characters."""
    assert DEFAULT_MAX_CHARACTERS >= 20 * 2700


def test_one_sentence_longer_than_the_bound_is_still_accepted():
    """A sentence is never rejected for its length; the bound only decides when
    a further one waits."""
    async def run():
        buffer = ScriptBuffer(max_characters=10)
        await buffer.put("a sentence far longer than ten characters.")
        buffer.close()
        first = await buffer.get()
        assert first.startswith("a sentence")
        assert await buffer.get() is None
        assert buffer.blocked_seconds == 0.0
    asyncio.run(run())


# --------------------------------------------------------------------------
# what comes out
# --------------------------------------------------------------------------
def test_the_first_chunk_is_the_first_complete_sentence_alone():
    chunks = _collect(stream=_stream(["It was built to ring."] + SCRIPT))
    assert chunks[0].text == "It was built to ring."
    assert chunks[0].sentences == 1 and chunks[0].words == 5
    assert chunks[0].reason.startswith("first chunk")


def test_batching_reduces_the_call_count_without_touching_the_text():
    chunks = _collect()
    assert len(chunks) < len(SCRIPT)
    assert " ".join(c.text for c in chunks).split() == " ".join(SCRIPT).split()


def test_chunks_arrive_in_order_and_exactly_once():
    chunks = _collect()
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_every_short_chunk_has_a_reason_that_is_not_a_batching_rule():
    """Short chunks are legitimate as the opening, the final flush, or under low
    headroom. What must never happen is a short chunk from a batching rule."""
    chunks = _collect()
    for chunk in chunks:
        if chunk.words < AssemblyPolicy().min_words:
            assert (chunk.reason.startswith("first chunk")
                    or chunk.reason == "end of script"
                    or "headroom" in chunk.reason), chunk


def test_a_headroom_source_can_force_early_release():
    starved = _collect(headroom=lambda: 0.5)
    assert any("headroom" in c.reason for c in starved[1:]), \
        [c.reason for c in starved]
    assert len(starved) > len(_collect())


def test_an_empty_stream_produces_nothing_rather_than_an_empty_chunk():
    async def nothing():
        return
        yield  # pragma: no cover
    assert _collect(stream=nothing()) == []


def test_blank_sentences_are_dropped_not_spoken():
    chunks = _collect(stream=_stream(["", "   ", "A real sentence here."]))
    assert len(chunks) == 1 and chunks[0].text == "A real sentence here."


# --------------------------------------------------------------------------
# failure and cancellation
# --------------------------------------------------------------------------
def test_a_producer_failure_reaches_the_consumer():
    """`pipeline._start` surfaces a producer exception rather than swallowing
    it, and so must this."""
    async def broken():
        yield "The first sentence arrives."
        raise RuntimeError("the model call failed")

    with pytest.raises(RuntimeError, match="the model call failed"):
        _collect(stream=broken())


def test_abandoning_the_consumer_stops_the_reader():
    """A cancelled request must not leave a reader running against a dead
    stream."""
    started, finished = [], []

    async def counted():
        for index, sentence in enumerate(SCRIPT * 4):
            started.append(index)
            await asyncio.sleep(0.005)
            yield sentence
        finished.append(True)

    async def run():
        stream = assemble_chunks(counted())
        async for _ in stream:
            break
        await stream.aclose()
        await asyncio.sleep(0.15)

    asyncio.run(run())
    assert not finished, "the reader ran to completion after the consumer left"
    assert len(started) < len(SCRIPT) * 4


def test_a_policy_can_be_passed_through():
    chunks = _collect(policy=AssemblyPolicy(min_words=4, target_words=6,
                                            max_words=10))
    assert max(c.words for c in chunks) <= 10 or any(
        c.sentences == 1 for c in chunks)
