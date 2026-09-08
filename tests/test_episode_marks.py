"""What a real episode reports about itself.

Phase 6's numbers came from an event log in the experiment harness. Production
had first-audio and little else, so the same request served to a listener could
not answer "did the first synthesis hold one sentence" or "was Claude ever
blocked". These are the marks that close that, and the rules they follow:
recorded not inferred, first-time-only, and never read back by the pipeline.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as appmod
import dataclasses
import pipeline as pipeline_mod
from episode_marks import EpisodeMarks, TimedClient
from pipeline import GenerationStats, PodcastPipeline
from script_generator import plan_episode
from tts import DebugEngine

from tests.test_pipeline import FakeGenerator

ENGINE = DebugEngine()


def sized(words: int) -> str:
    return " ".join(f"w{i}" for i in range(words - 1)) + " end."


class Opening:
    def __init__(self, first: str = "It was built to ring."):
        self.first = first

    async def stream_sentences(self, plan, notes=None):
        yield self.first
        for _ in range(30):
            await asyncio.sleep(0)
            yield sized(20)

    async def top_up(self, plan, spoken_so_far, words_needed):
        async for sentence in self.stream_sentences(plan):
            yield sentence


def episode(pipeline_value: str, generator=None, minutes: int = 3):
    original = pipeline_mod.settings
    pipeline_mod.settings = dataclasses.replace(original,
                                                streaming_pipeline=pipeline_value)
    try:
        async def main():
            plan = plan_episode("q", minutes)
            pipe = PodcastPipeline(generator=generator or Opening(),
                                   engine=ENGINE, cache=None)
            stats = GenerationStats()
            async for _ in pipe.stream_pcm(plan, stats):
                pass
            return stats

        return asyncio.run(asyncio.wait_for(main(), 20))
    finally:
        pipeline_mod.settings = original


# --------------------------------------------------------------------------
# the rules the marks follow
# --------------------------------------------------------------------------
def test_a_mark_records_the_first_time_and_never_moves():
    """Every mark names a *first*. A later sentence overwriting one would
    quietly turn it into a last."""
    marks = EpisodeMarks()
    first = marks.mark("first_sentence")
    marks.mark("first_sentence")
    assert marks.at("first_sentence") == first


def test_a_span_across_a_missing_mark_is_absent_not_zero():
    marks = EpisodeMarks()
    marks.mark("claude_start")
    assert marks.span("claude_start", "never_happened") is None
    assert marks.summary()["claude_ttft"] is None


def test_instrumentation_is_never_read_back_by_the_pipeline():
    """Recording must not be able to change what a listener hears."""
    import inspect

    source = inspect.getsource(pipeline_mod)
    for forbidden in ("marks.at(", "marks.span(", "marks.summary(",
                      "marks.first_chunk"):
        assert forbidden not in source, f"pipeline reads {forbidden}"


# --------------------------------------------------------------------------
# what a real episode reports
# --------------------------------------------------------------------------
@pytest.mark.parametrize("value", ["legacy", "phase6"])
def test_the_episode_reports_its_own_timeline(value):
    summary = episode(value).marks.summary()
    for name in ("claude_to_first_sentence", "first_sentence_to_synthesis",
                 "first_synthesis_seconds", "speaking_total"):
        assert summary[name] is not None and summary[name] >= 0, name
    assert summary["chunks"] > 0


@pytest.mark.parametrize("value", ["legacy", "phase6"])
def test_the_marks_are_ordered_the_way_the_request_runs(value):
    events = episode(value).marks.events
    # `claude_complete` is deliberately absent from this list: a truncated
    # episode stops before the sentinel, and most episodes are truncated.
    order = ["claude_start", "first_sentence", "first_tts_start",
             "first_tts_complete", "speaking_complete"]
    times = [events.get(name) for name in order]
    assert None not in times, dict(zip(order, times))
    assert times == sorted(times), dict(zip(order, times))


def test_the_first_synthesis_held_exactly_one_sentence_under_phase6():
    """The first-chunk invariant, reported by the episode rather than argued
    from the code - which is what the real run has to be able to show."""
    summary = episode("phase6").marks.summary()
    assert summary["first_chunk_sentences"] == 1
    assert summary["first_chunk_words"] == 5
    assert summary["chunk_words"][0] == 5


def test_later_chunks_are_larger_than_the_first_under_phase6():
    words = episode("phase6").marks.summary()["chunk_words"]
    assert len(words) > 1
    assert max(words[1:]) > words[0], "nothing was assembled after the opening"


def test_every_synthesis_is_recorded_with_its_size_and_cost():
    chunks = episode("phase6").marks.chunks
    assert chunks and all(c.words > 0 for c in chunks)
    assert all(c.generate_seconds >= 0 for c in chunks)
    assert all(c.audio_seconds > 0 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_the_decoupling_verdict_comes_from_the_two_instants_that_make_it():
    """`claude_decoupled` is `first_tts_start < claude_complete`, and nothing
    else.

    It used to be `backlog_at_claude_complete > 0` - whether the bounded work
    queue happened to be non-empty when speaking stopped. That is a different
    question, and on a real 4090 run it reported COUPLED for a healthy episode
    whose consumer had simply drained the queue. The backlog is still recorded,
    as corroboration; it is not the verdict.
    """
    marks = episode("phase6").marks
    summary = marks.summary()
    assert summary["claude_decoupled"] is (
        marks.events["first_tts_start"] < marks.events["claude_complete"])
    assert summary["backlog_at_claude_complete"] is not None


def test_an_unmeasured_run_is_unknown_rather_than_coupled():
    """"Not measured" and "measured and false" must not look alike: one is a
    gap in the instrumentation, the other is a product failure."""
    from episode_marks import EpisodeMarks

    marks = EpisodeMarks()
    assert marks.summary()["claude_decoupled"] is None
    marks.mark("first_tts_start")
    assert marks.summary()["claude_decoupled"] is None, (
        "one instant is not a verdict")
    marks.mark("claude_complete")
    assert marks.summary()["claude_decoupled"] is True


def test_a_queue_drained_at_the_end_is_not_reported_as_coupled():
    """The exact 4090 failure: backlog zero on a decoupled run."""
    from episode_marks import EpisodeMarks

    marks = EpisodeMarks()
    marks.mark("first_tts_start")
    marks.mark("claude_complete")
    marks.backlog_at_claude_complete = 0
    assert marks.summary()["claude_decoupled"] is True


def test_legacy_does_not_claim_a_backlog_it_cannot_measure():
    """`_speak` has no separate reader, so there is no backlog to report and
    it stays absent rather than reporting a misleading zero."""
    summary = episode("legacy").marks.summary()
    assert summary["backlog_at_claude_complete"] is None


def test_only_the_decoupled_pipeline_can_say_when_claude_finished():
    """The difference between the two architectures, in one measurement.

    Both mark the model's completion in the producer now. On a truncated
    episode - which is most of them - only Phase 6 has one to report, and the
    reason is the whole point:

    * legacy's queue is bounded at QUEUE_DEPTH, so the reader is backpressured
      by the voice. When truncation cancels it, Claude is *still writing*.
      There is no completion to record because it never completed.
    * Phase 6 puts a character-bounded buffer between the reader and the voice,
      so the reader runs to the end of the model stream regardless of how far
      behind synthesis is. It finishes, and says when.

    So the absence in legacy and the presence in Phase 6 are the same fact seen
    from two sides, and it is stronger evidence than the verdict flag: legacy
    cannot even be asked the question.
    """
    assert episode("legacy").marks.summary()["claude_total"] is None
    assert episode("phase6").marks.summary()["claude_total"] is not None


def test_a_truncated_episode_still_records_when_claude_finished():
    """Reversed, and the reversal is the point.

    This used to assert `claude_total is None` on a truncated episode, on the
    reasoning that Claude did not complete - we stopped it. That was true of
    where the mark was taken, not of Claude: the mark was set by the *consumer*
    on receiving the queue sentinel, which is the end of speaking, and a
    truncated episode breaks before the sentinel arrives. Most episodes
    truncate, so the model's completion was usually unrecorded and the
    decoupling verdict had nothing to stand on.

    The mark now belongs to the producer, which is the only place that knows
    the stream ended. Truncation stops the speaking, not the reading, so the
    model's finish is a real instant and it is recorded.
    """
    summary = episode("phase6").marks.summary()
    assert summary["claude_total"] is not None
    assert summary["speaking_total"] is not None


# --------------------------------------------------------------------------
# time to first token
# --------------------------------------------------------------------------
def test_the_client_wrapper_marks_the_first_delta_once():
    marks = EpisodeMarks()

    class Inner:
        @property
        def text_stream(self):
            async def gen():
                for delta in ("Hello", " there", "."):
                    yield delta
            return gen()

    from episode_marks import TimedStream

    async def drain():
        return [d async for d in TimedStream(Inner(), marks).text_stream]

    assert asyncio.run(drain()) == ["Hello", " there", "."]
    assert marks.at("claude_first_token") is not None


def test_the_wrapper_is_transparent_to_everything_else():
    """It observes. It must not become a thing the request has to go through."""
    class Inner:
        messages = object()
        beta = "passthrough"

    wrapped = TimedClient(Inner(), EpisodeMarks())
    assert wrapped.beta == "passthrough"


def test_a_generator_without_a_client_is_left_alone():
    """Test generators have no client; wrapping must not require one."""
    stats = episode("phase6")
    assert stats.marks.at("claude_first_token") is None


# --------------------------------------------------------------------------
# what reaches the operator
# --------------------------------------------------------------------------
def test_the_marks_reach_the_response_headers(monkeypatch):
    monkeypatch.setattr(appmod, "SCRIPT_CACHE", None)
    monkeypatch.setattr(appmod, "DEMO_MODE", False)
    monkeypatch.setattr(appmod, "_rate_limit", lambda request: None)
    monkeypatch.setattr(
        appmod, "_make_pipeline",
        lambda voice=None: PodcastPipeline(generator=FakeGenerator(1.0),
                                           engine=ENGINE, cache=None,
                                           voice=voice))
    client = TestClient(appmod.app)
    with client.stream("GET", "/api/audio?q=x&minutes=1&fmt=pcm") as response:
        headers = dict(response.headers)
        for _ in response.iter_bytes():
            pass

    summary = json.loads(headers["x-episode-marks"])
    assert summary["chunks"] > 0
    assert summary["first_chunk_sentences"] >= 1
    assert summary["claude_to_first_sentence"] is not None
