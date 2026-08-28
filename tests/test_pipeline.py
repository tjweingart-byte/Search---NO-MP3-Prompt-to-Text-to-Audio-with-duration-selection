"""Tests that run with no API key and no TTS engine installed.

Claude is replaced by a scripted generator and speech by the debug engine, so
the thing under test is the part that actually tends to break: length control.
"""
from __future__ import annotations

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_utils import PaceController, pcm_duration, silence, streaming_wav_header  # noqa: E402
from config import settings  # noqa: E402
from pipeline import GenerationStats, PodcastPipeline  # noqa: E402
from script_generator import clean_for_speech, plan_episode  # noqa: E402
from tts import DebugEngine  # noqa: E402

LOREM = (
    "The question turns out to have a surprisingly practical answer. "
    "Researchers have studied it for decades without full agreement. "
    "Here is what the evidence currently supports. "
)


class FakeGenerator:
    """Emits sentences totalling roughly `ratio` of the planned word budget."""

    def __init__(self, ratio: float = 1.0):
        self.ratio = ratio

    async def _emit(self, target):
        emitted = 0
        while emitted < target:
            for sentence in LOREM.strip().split(". "):
                sentence = sentence.rstrip(".") + "."
                yield sentence
                emitted += len(sentence.split())
                await asyncio.sleep(0)
                if emitted >= target:
                    break

    async def stream_sentences(self, plan):
        async for s in self._emit(int(plan.word_budget * self.ratio)):
            yield s

    async def top_up(self, plan, spoken_so_far, words_needed):
        async for s in self._emit(words_needed):
            yield s


@pytest.mark.parametrize("minutes", [1, 3, 5, 10])
def test_word_budget_scales_with_duration(minutes):
    plan = plan_episode("anything", minutes)
    assert plan.target_seconds == minutes * 60
    assert plan.word_budget == int(round(minutes * settings.target_wpm))
    assert plan.min_words < plan.word_budget < plan.max_words


def test_duration_is_clamped_to_the_supported_range():
    assert plan_episode("q", 0).minutes == 1
    assert plan_episode("q", 99).minutes == 10


def test_streaming_wav_header_is_a_live_stream_header():
    header = streaming_wav_header()
    assert len(header) == 44
    assert header[:4] == b"RIFF" and header[8:12] == b"WAVE"
    assert header[4:8] == b"\xff\xff\xff\xff"  # unknown length


def test_pace_controller_speeds_up_when_running_behind():
    pace = PaceController(target_seconds=60.0, total_words=150)
    baseline = pace.next_wpm()
    pace.observe(int(50 * settings.bytes_per_second), 50)  # 50s spent, 100 words left
    assert pace.next_wpm() > baseline
    assert pace.next_wpm() <= settings.max_wpm


def test_pace_controller_slows_down_when_running_ahead():
    pace = PaceController(target_seconds=600.0, total_words=1500)
    pace.observe(int(10 * settings.bytes_per_second), 500)
    assert pace.next_wpm() >= settings.min_wpm
    assert pace.next_wpm() < settings.target_wpm


@pytest.mark.parametrize("minutes", [1, 4, 10])
def test_audio_lands_on_the_requested_duration(minutes):
    plan = plan_episode("a question", minutes)
    pipeline = PodcastPipeline(generator=FakeGenerator(), engine=DebugEngine())
    stats = GenerationStats()

    async def run():
        total = 0
        async for chunk in pipeline.stream_pcm(plan, stats):
            total += len(chunk)
        return total

    total_bytes = asyncio.run(run())
    seconds = pcm_duration(total_bytes)
    tolerance = max(1.0, plan.target_seconds * settings.duration_tolerance)
    assert abs(seconds - plan.target_seconds) <= tolerance, (
        f"{minutes} min episode produced {seconds:.1f}s of audio"
    )
    assert stats.words > 0 and stats.sentences > 0


@pytest.mark.parametrize("ratio", [0.7, 1.3])
def test_a_short_or_long_script_still_lands_on_time(ratio):
    """The pacing controller must absorb a model that misses the budget."""
    plan = plan_episode("a question", 5)
    pipeline = PodcastPipeline(generator=FakeGenerator(ratio), engine=DebugEngine())
    stats = GenerationStats()

    async def run():
        async for _ in pipeline.stream_pcm(plan, stats):
            pass

    asyncio.run(run())
    assert abs(stats.audio_seconds - plan.target_seconds) <= 3.0
    if ratio < 1:
        assert stats.topups >= 1, "a short script should have been topped up"
    else:
        assert stats.truncated, "a long script should have been trimmed"


def test_wav_stream_starts_with_exactly_one_header():
    plan = plan_episode("a question", 1)
    pipeline = PodcastPipeline(generator=FakeGenerator(), engine=DebugEngine())

    async def run():
        chunks = []
        async for chunk in pipeline.stream_wav(plan, GenerationStats()):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(run())
    assert chunks[0] == streaming_wav_header()
    assert b"RIFF" not in b"".join(chunks[1:])


def test_generator_errors_propagate_instead_of_producing_silence():
    class Broken:
        async def stream_sentences(self, plan):
            raise RuntimeError("upstream is down")
            yield ""  # pragma: no cover

        async def top_up(self, plan, spoken_so_far, words_needed):
            yield ""  # pragma: no cover

    plan = plan_episode("a question", 1)
    pipeline = PodcastPipeline(generator=Broken(), engine=DebugEngine())

    async def run():
        async for _ in pipeline.stream_pcm(plan, GenerationStats()):
            pass

    with pytest.raises(RuntimeError, match="upstream is down"):
        asyncio.run(run())


def test_speech_cleanup_removes_unspeakable_markup():
    dirty = "**Host:** Welcome. [intro music] See the *chart* below (music fades)."
    clean = clean_for_speech(dirty)
    for junk in ("*", "[", "]", "Host:", "music"):
        assert junk not in clean


def test_silence_is_exact():
    assert pcm_duration(len(silence(2.5))) == pytest.approx(2.5, abs=0.01)
