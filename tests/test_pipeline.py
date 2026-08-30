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
from cache import (  # noqa: E402
    MemoryScriptCache,
    SqliteScriptCache,
    cache_key,
    is_shareable,
    normalize_query,
    ttl_for,
)

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
        async for s in self._emit(int(plan.body_budget * self.ratio)):
            yield s

    async def cold_open(self, plan):
        yield "Here is what happened, and why it mattered."

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
    pipeline = PodcastPipeline(generator=FakeGenerator(), engine=DebugEngine(), cache=None)
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
    pipeline = PodcastPipeline(generator=FakeGenerator(ratio), engine=DebugEngine(), cache=None)
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
    pipeline = PodcastPipeline(generator=FakeGenerator(), engine=DebugEngine(), cache=None)

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
    pipeline = PodcastPipeline(generator=Broken(), engine=DebugEngine(), cache=None)

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


# --------------------------------------------------------------------------
# Shared script cache
# --------------------------------------------------------------------------


class CountingGenerator(FakeGenerator):
    """Counts how many times the model was actually asked to write."""

    def __init__(self, ratio: float = 1.0):
        super().__init__(ratio)
        self.calls = 0

    async def stream_sentences(self, plan):
        self.calls += 1
        async for s in super().stream_sentences(plan):
            yield s


def _run_episode(pipeline, plan, stats):
    async def run():
        total = 0
        async for chunk in pipeline.stream_pcm(plan, stats):
            total += len(chunk)
        return total

    return asyncio.run(run())


def test_a_second_listener_reuses_the_first_listeners_script():
    """The whole point: two different people, one model call."""
    cache = MemoryScriptCache()
    gen = CountingGenerator()
    plan = plan_episode("recap of week 5 of the NFL season", 3)

    first = GenerationStats()
    bytes_first = _run_episode(
        PodcastPipeline(generator=gen, engine=DebugEngine(), cache=cache), plan, first
    )

    # A different person, phrasing it differently.
    plan2 = plan_episode("a recap of week 5 of the NFL season, please!", 3)
    second = GenerationStats()
    bytes_second = _run_episode(
        PodcastPipeline(generator=gen, engine=DebugEngine(), cache=cache), plan2, second
    )

    assert first.cache == "miss" and second.cache == "hit"
    assert gen.calls == 1, "the second listener should not have cost a model call"
    assert second.script == first.script
    # Identical script through an identical controller means identical audio.
    assert bytes_second == bytes_first


def test_cache_key_separates_durations_and_matches_phrasings():
    # Lexical normalisation collapses punctuation, word order and filler.
    same = {
        cache_key("Give me a recap of week 5 of the NFL season", 3),
        cache_key("a recap of week 5 of the NFL season, please!", 3),
        cache_key("NFL season week 5 - recap", 3),
    }
    assert len(same) == 1, "equivalent phrasings must share one entry"
    assert cache_key("NFL week 5 recap", 3) != cache_key("NFL week 5 recap", 4)
    assert cache_key("NFL week 5 recap", 3) != cache_key("NBA week 5 recap", 3)


def test_lexical_keys_miss_across_a_real_synonym():
    """A documented limit: this is what the semantic key layer exists to fix."""
    assert cache_key("NFL week 5 recap", 3) != cache_key("NFL season week 5 recap", 3)
    # Supplying a canonical label collapses them, which is what the small-model
    # canonicaliser does when CACHE_SEMANTIC_KEY is enabled.
    label = "nfl season week 5 recap"
    assert cache_key("NFL week 5 recap", 3, label) == cache_key(
        "give me a recap of week five of the NFL season", 3, label
    )


def test_normalization_drops_filler_but_keeps_topic():
    assert normalize_query("What is the offside rule") == normalize_query("offside rule")
    assert normalize_query("apples") != normalize_query("oranges")


def test_volatile_queries_get_a_short_ttl():
    assert ttl_for("latest news on the election") < ttl_for("why is the sky blue")
    assert ttl_for("today's scores") <= settings.cache_ttl_volatile


def test_personal_queries_are_never_shared():
    assert not is_shareable("summarize my medical results")
    assert not is_shareable("what should I do about our mortgage")
    assert not is_shareable("what is known about alice@example.com")
    assert is_shareable("recap of week 5 of the NFL season")


def test_ordinary_phrasing_is_not_mistaken_for_a_personal_query():
    """'give me a recap' is not personal - treating it so kills the hit rate."""
    for q in [
        "give me a recap of week 5 of the NFL season",
        "tell me why the sky is blue",
        "I want a briefing on the election",
        "can we get a summary of the moon landing",
    ]:
        assert is_shareable(q), q


def test_personal_queries_bypass_the_cache_entirely():
    cache = MemoryScriptCache()
    gen = CountingGenerator()
    plan = plan_episode("a briefing on my lab results", 1)
    for _ in range(2):
        stats = GenerationStats()
        _run_episode(PodcastPipeline(generator=gen, engine=DebugEngine(), cache=cache), plan, stats)
        assert stats.cache == "miss"
    assert gen.calls == 2, "personal queries must always be regenerated"


def test_expired_entries_are_not_served(tmp_path):
    cache = SqliteScriptCache(str(tmp_path / "c.db"))
    cache.put("k", ["One sentence."], ttl=-1, query="q")
    assert cache.get("k") is None
    cache.put("k2", ["One sentence."], ttl=60, query="q")
    assert cache.get("k2") == ["One sentence."]


def test_sqlite_cache_is_visible_to_another_process_instance(tmp_path):
    """Different uvicorn workers must see each other's entries."""
    path = str(tmp_path / "shared.db")
    SqliteScriptCache(path).put("k", ["Shared across workers."], ttl=300, query="q")
    assert SqliteScriptCache(path).get("k") == ["Shared across workers."]


def test_a_broken_cache_never_breaks_an_episode(tmp_path):
    cache = SqliteScriptCache(str(tmp_path / "c.db"))
    cache.path = "/nonexistent/directory/c.db"
    cache._local = type(cache._local)()  # force a fresh, failing connection
    plan = plan_episode("a question", 1)
    stats = GenerationStats()
    _run_episode(PodcastPipeline(generator=FakeGenerator(), engine=DebugEngine(), cache=cache), plan, stats)
    assert stats.audio_seconds > 0, "a cache failure must degrade to normal generation"
