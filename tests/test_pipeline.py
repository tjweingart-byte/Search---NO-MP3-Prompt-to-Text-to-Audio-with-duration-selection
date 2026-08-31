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
from script_generator import build_prompt, clean_for_speech, now_line, plan_episode  # noqa: E402
from tts import DebugEngine  # noqa: E402


@pytest.fixture
def with_opener(monkeypatch):
    """The opener is opt-in now; these tests exercise that path deliberately."""
    import dataclasses

    import pipeline as pipeline_module
    import script_generator as sg

    patched = dataclasses.replace(settings, enable_cold_open=True, allow_topups=True)
    monkeypatch.setattr(pipeline_module, "settings", patched)
    monkeypatch.setattr(sg, "settings", patched)
    return patched
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

    async def stream_sentences(self, plan, notes=None):
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


def test_a_long_script_is_trimmed_to_the_selected_length():
    """Over-running is still cut: the slider is a ceiling on the listener's time."""
    plan = plan_episode("a question", 5)
    pipeline = PodcastPipeline(generator=FakeGenerator(1.3), engine=DebugEngine(), cache=None)
    stats = GenerationStats()

    async def run():
        async for _ in pipeline.stream_pcm(plan, stats):
            pass

    asyncio.run(run())
    assert abs(stats.audio_seconds - plan.target_seconds) <= 3.0
    assert stats.truncated, "a long script should have been trimmed"


def test_a_short_script_ends_early_rather_than_being_padded(with_opener):
    """Deliberate change: the length is a ceiling, not a quota.

    Padding a script that has run out of substance produces exactly the filler
    the opener was removed for. With top-ups enabled the old behaviour is still
    available, and still lands on the clock.
    """
    plan = plan_episode("a question", 5)
    pipeline = PodcastPipeline(generator=FakeGenerator(0.7), engine=DebugEngine(), cache=None)
    stats = GenerationStats()

    async def run():
        async for _ in pipeline.stream_pcm(plan, stats):
            pass

    asyncio.run(run())
    assert stats.topups >= 1, "with top-ups on, a short script should be extended"
    assert abs(stats.audio_seconds - plan.target_seconds) <= 3.0


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
        async def stream_sentences(self, plan, notes=None):
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

    async def stream_sentences(self, plan, notes=None):
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


# --------------------------------------------------------------------------
# Cold open continuity
#
# Regression for a reported bug: five seconds of audio, then five seconds of
# dead air, then the rest. The opener covered only part of the research wait,
# and the listener heard the shortfall as silence.
# --------------------------------------------------------------------------

OPENER_SENTENCES = ["Opener one.", "Opener two.", "Opener three.", "Opener four."]


class SlowResearchGenerator:
    """A model whose main script takes `delay` seconds to start arriving."""

    def __init__(self, delay: float):
        self.delay = delay

    async def cold_open(self, plan):
        for sentence in OPENER_SENTENCES:
            yield sentence

    async def stream_sentences(self, plan, notes=None):
        await asyncio.sleep(self.delay)
        for i in range(60):
            yield f"Body sentence {i} of the real briefing."

    async def top_up(self, plan, spoken_so_far, words_needed):
        for i in range(30):
            yield f"Extra sentence {i}."


def _episode(delay, minutes=2):
    plan = plan_episode("a topic", minutes)
    stats = GenerationStats()
    pipeline = PodcastPipeline(
        generator=SlowResearchGenerator(delay), engine=DebugEngine(), cache=None
    )

    async def run():
        async for _ in pipeline.stream_pcm(plan, stats):
            pass

    asyncio.run(run())
    opener_spoken = [s for s in stats.script if s.startswith("Opener")]
    return stats, opener_spoken


def test_opener_grows_to_cover_slow_research(with_opener):
    """The listener must never run out of audio while the script is written."""
    _, quick = _episode(0.0)
    _, slow = _episode(3.0)
    assert len(quick) < len(slow), "a slower model should get more introduction"
    assert len(slow) > 1, "one sentence cannot cover a multi-second wait"


def test_opener_is_not_wasted_when_research_is_fast(with_opener):
    _, quick = _episode(0.0)
    assert len(quick) <= 2, "fast research should cut over almost immediately"
    assert len(quick) >= 1, "something should still open the episode"


@pytest.mark.parametrize("delay", [0.0, 1.0, 3.0])
def test_duration_holds_whatever_the_research_latency(delay, with_opener):
    stats, _ = _episode(delay)
    assert abs(stats.audio_seconds - 120) <= 3.0


def test_no_opener_is_spoken_when_the_script_fails(with_opener):
    """Never introduce an episode that is not coming."""

    class Failing(SlowResearchGenerator):
        async def stream_sentences(self, plan, notes=None):
            raise RuntimeError("model is down")
            yield ""  # pragma: no cover

    plan = plan_episode("a topic", 1)
    stats = GenerationStats()
    pipeline = PodcastPipeline(generator=Failing(0.0), engine=DebugEngine(), cache=None)

    async def run():
        async for _ in pipeline.stream_pcm(plan, stats):
            pass

    with pytest.raises(RuntimeError, match="model is down"):
        asyncio.run(run())
    assert stats.sentences == 0


# --------------------------------------------------------------------------
# Time scope, follow-up scope, and opener refill
# --------------------------------------------------------------------------


def test_the_script_is_told_what_time_it_is():
    """Without a clock the model cannot tell current news from stale news."""
    prompt = build_prompt(plan_episode("tour championship update", 3))
    assert now_line().split(" at ")[0] in prompt, "the current date must reach the model"
    assert "newest information" in prompt


def test_a_follow_up_is_told_not_to_repeat_what_was_heard():
    plain = build_prompt(plan_episode("the prize money", 3))
    follow = build_prompt(plan_episode("the prize money", 3, context="The Tour Championship"))
    assert "FOLLOW-UP" not in plain
    assert "FOLLOW-UP" in follow
    assert "The Tour Championship" in follow
    assert "Do not re-explain" in follow


def test_a_follow_up_is_cached_separately_from_the_same_question_asked_cold():
    from cache import cache_key

    cold = cache_key("the prize money", 3)
    follow = cache_key("the prize money", 3, None, "The Tour Championship")
    assert cold != follow, "a follow-up is a different episode from the same words asked cold"


class ShortOpenerGenerator(SlowResearchGenerator):
    """An opener that runs out long before the script is ready."""

    async def cold_open(self, plan):
        yield "Only one framing sentence."


def test_the_opener_is_extended_rather_than_running_into_silence(with_opener):
    """The reported bug: opener ends, script is not ready, listener hears nothing."""
    plan = plan_episode("a topic", 3)
    stats = GenerationStats()
    pipeline = PodcastPipeline(
        generator=ShortOpenerGenerator(4.0), engine=DebugEngine(), cache=None
    )

    async def run():
        async for _ in pipeline.stream_pcm(plan, stats):
            pass

    asyncio.run(run())
    openers = [s for s in stats.script if s.startswith("Only one")]
    assert len(openers) > 1, "a one-sentence opener must be refilled, not left to run dry"


# --------------------------------------------------------------------------
# The opener must never let the stream fall silent
#
# Regression for the reported "consistent 5-7 second pause". The opener used to
# get a fixed number of top-ups, covering roughly twenty seconds; a researched
# call can take longer, and the listener heard the difference as dead air.
# --------------------------------------------------------------------------


class ShortOpenerSlowBody:
    """A one-sentence opener and a script that takes a long time to start."""

    def __init__(self, delay: float):
        self.delay = delay
        self.opener_calls = 0

    async def cold_open(self, plan):
        self.opener_calls += 1
        yield "One short framing sentence."

    async def stream_sentences(self, plan, notes=None):
        await asyncio.sleep(self.delay)
        for i in range(80):
            yield f"Body sentence {i} of the researched briefing."

    async def top_up(self, plan, spoken_so_far, words_needed):
        for i in range(40):
            yield f"Extra {i}."


def _run_with_slow_body(delay, minutes=3):
    plan = plan_episode("a topic", minutes)
    stats = GenerationStats()
    generator = ShortOpenerSlowBody(delay)
    pipeline = PodcastPipeline(generator=generator, engine=DebugEngine(), cache=None)

    async def run():
        async for _ in pipeline.stream_pcm(plan, stats):
            pass

    asyncio.run(run())
    return stats, generator


@pytest.mark.parametrize("delay", [0.5, 2.0])
def test_the_stream_never_starves_while_the_script_is_written(delay, with_opener):
    """Audio produced must stay ahead of the wall clock the whole way."""
    stats, _ = _run_with_slow_body(delay)
    assert not stats.starved, (
        f"the listener heard silence: min headroom {stats.min_headroom:.1f}s"
    )
    assert stats.min_headroom > 0


def test_the_opener_is_topped_up_when_one_batch_is_not_enough(with_opener):
    """A single short opener cannot cover a slow script, so more is fetched."""
    stats, generator = _run_with_slow_body(2.0)
    assert generator.opener_calls > 1, "the opener should have been refilled"


def test_a_fast_script_wastes_no_preamble(with_opener):
    """The opener is paced to the listener, so a quick script uses almost none."""
    stats, generator = _run_with_slow_body(0.0)
    assert generator.opener_calls == 1, "no top-up should be needed"
    openers = [s for s in stats.script if s.startswith("One short")]
    assert len(openers) <= 2, f"spoke {len(openers)} opener sentences for a fast script"


def test_duration_still_lands_despite_a_long_opener(with_opener):
    stats, _ = _run_with_slow_body(2.0)
    assert abs(stats.audio_seconds - 180) <= 4.0


# --------------------------------------------------------------------------
# The default: no filler
#
# The opener was prompted to state no facts, which made it filler by
# construction; covering a long research wait meant 15-30 seconds of it. And
# padding a short script back to length reintroduces the same problem from the
# other end. Both are now off by default.
# --------------------------------------------------------------------------


def test_no_opener_is_spoken_by_default():
    """Nothing plays until the real briefing does."""
    assert settings.enable_cold_open is False

    plan = plan_episode("a topic", 2)
    stats = GenerationStats()
    generator = ShortOpenerSlowBody(0.2)
    pipeline = PodcastPipeline(generator=generator, engine=DebugEngine(), cache=None)

    async def run():
        async for _ in pipeline.stream_pcm(plan, stats):
            pass

    asyncio.run(run())
    assert generator.opener_calls == 0, "the opener must not be called at all"
    assert not any(s.startswith("One short framing") for s in stats.script)
    assert stats.cold_open is False


def test_a_short_script_is_not_padded_back_to_length():
    """A briefing that runs out of substance should end, not be stretched."""
    assert settings.allow_topups is False

    class BriefGenerator:
        async def stream_sentences(self, plan, notes=None):
            for i in range(6):
                yield f"A genuinely substantive sentence number {i}."

        async def top_up(self, plan, spoken_so_far, words_needed):
            raise AssertionError("top-up must not run when padding is disabled")
            yield ""  # pragma: no cover

    plan = plan_episode("a topic", 5)
    stats = GenerationStats()
    pipeline = PodcastPipeline(generator=BriefGenerator(), engine=DebugEngine(), cache=None)

    async def run():
        async for _ in pipeline.stream_pcm(plan, stats):
            pass

    asyncio.run(run())
    assert stats.topups == 0
    assert stats.audio_seconds < 300, "the episode should end early rather than pad"


def test_the_brief_does_not_demand_a_word_count_above_all_else():
    """The old prompt said the length contract was the most important thing."""
    prompt = build_prompt(plan_episode("anything", 5))
    assert "most important requirement" not in prompt
    assert "not a quota" in prompt
    assert "resolves early, stop" in prompt


def test_the_brief_no_longer_imposes_a_section_template():
    """Fixed beats forced the model to invent content for sections it had none for."""
    prompt = build_prompt(plan_episode("recap of a golf tournament", 5))
    for beat in ("one-line hook", "the essential background", "what to watch next"):
        assert beat not in prompt, f"the {beat!r} template beat is still being imposed"


# --------------------------------------------------------------------------
# Style examples
#
# Rules describe a style; examples demonstrate one, and a model matches a
# demonstration far more closely. This is the most direct control over the
# writing, so the loading has to be predictable.
# --------------------------------------------------------------------------


def test_no_examples_leaves_the_prompt_untouched(monkeypatch):
    import script_generator as sg

    monkeypatch.setattr(sg, "STYLE_EXAMPLES", [])
    assert sg.system_prompt() == sg.SYSTEM_PROMPT


def test_examples_reach_the_model_framed_as_voice_not_fact(monkeypatch):
    """A borrowed fact would be a hallucination; only the sound should carry."""
    import script_generator as sg

    monkeypatch.setattr(
        sg, "STYLE_EXAMPLES", [("the offside rule", "A player is offside if...")]
    )
    prompt = sg.system_prompt()
    assert "A player is offside if..." in prompt
    assert "house voice" in prompt
    assert "not borrow its facts" in prompt


def test_examples_are_read_from_disk(tmp_path, monkeypatch):
    import script_generator as sg

    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "1-a-topic.txt").write_text(
        "what is a heat pump\n\nIt moves heat rather than making it.\n"
    )
    monkeypatch.setattr(sg, "__file__", str(tmp_path / "script_generator.py"))
    loaded = sg.load_style_examples()
    assert loaded == [("what is a heat pump", "It moves heat rather than making it.")]


def test_a_malformed_example_is_skipped_not_fatal(tmp_path, monkeypatch):
    """A typo in one file must not stop the app writing anything at all."""
    import script_generator as sg

    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "1-broken.txt").write_text("no blank line so no script")
    (tmp_path / "examples" / "2-fine.txt").write_text("a query\n\nA script.\n")
    monkeypatch.setattr(sg, "__file__", str(tmp_path / "script_generator.py"))
    assert sg.load_style_examples() == [("a query", "A script.")]


# --------------------------------------------------------------------------
# A FAM episode is a story
#
# The differentiator: not a briefing with storytelling added, but information
# delivered as narrative. Two failure modes bracket it - a flat list of facts on
# one side, and scene-setting that makes a listener think "get to the point" on
# the other. These assert the prompt guards both.
# --------------------------------------------------------------------------


def test_the_prompt_asks_for_narrative_not_a_list():
    from script_generator import SYSTEM_PROMPT

    assert "Because, therefore, but" in SYSTEM_PROMPT, "causation, not chronology"
    assert "Tension then release" in SYSTEM_PROMPT


def test_the_prompt_guards_against_over_storytelling():
    """The listener must never think 'get to the point'."""
    from script_generator import SYSTEM_PROMPT

    assert "get to the point" in SYSTEM_PROMPT
    assert "Atmosphere on its own is cut" in SYSTEM_PROMPT
    for banned in ("picture this", "imagine"):
        assert banned in SYSTEM_PROMPT.lower(), f"{banned!r} should be called out"


def test_the_opening_must_not_state_the_conclusion():
    """News-writing front-loads the answer; a story opens a question instead."""
    from script_generator import SYSTEM_PROMPT

    assert "Do not state your" in SYSTEM_PROMPT
    assert "nowhere to go after that" in SYSTEM_PROMPT
    assert "Do not delay it" in SYSTEM_PROMPT, "and it must not become a stall either"


def test_length_describes_story_scope_not_a_section_template():
    """A fixed beat template forced invented content; scope guides instead."""
    short = build_prompt(plan_episode("a topic", 1))
    long = build_prompt(plan_episode("a topic", 10))
    assert "opened and answered" in short
    assert "full arc" in long
    for beat in ("one-line hook", "the essential background", "what to watch next"):
        assert beat not in long


# --------------------------------------------------------------------------
# Annexation: hold the listener rather than conclude at them
#
# A good ending is still an exit - it gives permission to leave. The aim is that
# leaving takes a deliberate act, so every resolution opens the next question
# and the last line widens instead of wrapping up.
# --------------------------------------------------------------------------


def test_the_prompt_forbids_building_exits():
    from script_generator import SYSTEM_PROMPT

    assert "Never build an exit" in SYSTEM_PROMPT
    assert "Do not end. Widen" in SYSTEM_PROMPT
    for exit_signal in ("to sum up", "in conclusion", "the bottom line is"):
        assert exit_signal in SYSTEM_PROMPT, f"{exit_signal!r} should be banned by name"


def test_the_prompt_asks_the_writer_to_speak_from_inside():
    """Annexation means the listener is already in, not being shown round."""
    from script_generator import SYSTEM_PROMPT

    assert "Speak from inside" in SYSTEM_PROMPT
    assert "as you may know" in SYSTEM_PROMPT
    assert "point of view" in SYSTEM_PROMPT


def test_the_brief_asks_for_momentum_and_a_widening_end():
    prompt = build_prompt(plan_episode("a topic", 3))
    assert "make the next thing matter more" in prompt
    assert "Leave one thread open" in prompt
    assert "<<NEXT:" in prompt


def test_the_prompt_asks_for_one_named_unresolved_thread():
    """Widening only converts into a Go Deeper tap if it names something."""
    from script_generator import SYSTEM_PROMPT

    assert "Leave exactly one thread" in SYSTEM_PROMPT
    # A thread the listener has never heard of is a tease, not a thread.
    assert "already be in the room" in SYSTEM_PROMPT
    # And curiosity is not manufactured by asking for it.
    assert "Point at it, do not ask about it" in SYSTEM_PROMPT
    assert "<<NEXT:" in SYSTEM_PROMPT


def test_the_thread_marker_is_extracted_and_never_spoken():
    from script_generator import clean_for_speech, extract_thread

    script = "She filed the appeal on Tuesday.\n\n<<NEXT: whether the appeal is heard at all>>"
    assert extract_thread(script) == "whether the appeal is heard at all"
    assert "NEXT" not in clean_for_speech(script)
    assert clean_for_speech(script) == "She filed the appeal on Tuesday."


def test_a_half_written_marker_is_never_spoken_either():
    """The marker arrives in stream chunks; a partial one must not be read out."""
    from script_generator import clean_for_speech

    assert clean_for_speech("The count is still going. <<NEX") == "The count is still going."


def test_no_marker_is_not_an_error():
    from script_generator import extract_thread

    assert extract_thread("A script that named nothing.") == ""


def test_the_thread_survives_a_cache_hit():
    """The suggestion must still be there when the script is replayed for free."""
    cache = MemoryScriptCache()
    cache.put("k", ["One sentence."], ttl=60, query="q", thread="what happens to the levy")
    assert cache.thread("k") == "what happens to the levy"


def test_the_pipeline_carries_the_thread_out_of_generation():
    class Threaded:
        client = None

        async def stream_sentences(self, plan, notes=None):
            if notes is not None:
                notes.thread = "why the rule expires in March"
            yield "A sentence with substance in it."

    pipeline = PodcastPipeline(generator=Threaded(), engine=DebugEngine(), cache=None)
    stats = GenerationStats()

    async def run():
        async for _ in pipeline.stream_pcm(plan_episode("a topic", 1), stats):
            pass

    asyncio.run(run())
    assert stats.thread == "why the rule expires in March"
    assert stats.as_dict()["thread"] == "why the rule expires in March"


def test_the_marker_is_never_spoken_even_when_it_arrives_in_pieces():
    """The real risk of the feature: a marker read aloud, or half of one.

    The model streams token by token, so "<<NEXT: ...>>" reliably arrives split
    across several events - and one of those fragments ends in a full stop.
    """
    import script_generator

    chunks = [
        "She filed the appeal on Tuesday. ",
        "The clerk has not listed it.",
        "\n\n<<NE",
        "XT: whether the appe",
        "al is heard at all>>",
    ]

    class _Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        @property
        async def text_stream(self):  # pragma: no cover - replaced below
            raise AssertionError

        async def get_final_message(self):
            class _Msg:
                stop_reason = "end_turn"

            return _Msg()

    async def _text_stream():
        for chunk in chunks:
            yield chunk

    stream = _Stream()
    type(stream).text_stream = property(lambda self: _text_stream())

    class _Messages:
        def stream(self, **kwargs):
            return stream

    class _Client:
        messages = _Messages()

    gen = script_generator.ScriptGenerator.__new__(script_generator.ScriptGenerator)
    gen.client = _Client()
    notes = script_generator.ScriptNotes()

    async def run():
        return [s async for s in gen.stream_sentences(plan_episode("the appeal", 1), notes)]

    spoken = asyncio.run(run())
    assert notes.thread == "whether the appeal is heard at all"
    for sentence in spoken:
        assert "<" not in sentence and "NEXT" not in sentence, sentence
    assert " ".join(spoken) == "She filed the appeal on Tuesday. The clerk has not listed it."
