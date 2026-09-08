"""The duration contract, characterised on the shipped path and held for Phase 6.

A 1-, 3- or 5-minute request has to keep meaning what the product promises.
Streaming and chunking are execution strategies; they must not quietly redefine
the length of an episode.

Part one measures what production does today, on a deterministic generator and
the debug engine, and pins it as the compatibility baseline. Part two holds
`speech_assembly.fit_to_budget` - the only new duration-relevant code - to that
same baseline, at the boundaries rather than in comfortable cases.

Nothing here modifies production. `_start` is untouched, the flag is unwired,
and every legacy assertion below is a description of behaviour that already
shipped.
"""
from __future__ import annotations

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_utils import PaceController, pcm_duration
from config import settings
from pipeline import (MAX_TAIL_SILENCE, OVERRUN_GRACE, SENTENCE_GAP,
                      GenerationStats, PodcastPipeline)
from script_generator import count_words, plan_episode
from speech_assembly import AssemblyPolicy, SpeechAssembler, fit_to_budget
from tts import DebugEngine

from tests.test_pipeline import FakeGenerator

#: `DebugEngine.synth` returns exactly `words / (wpm / 60)` seconds of audio,
#: which is the estimator `_speak_one` uses. So these measurements are exact
#: rather than approximate, and a drift of even a tenth of a second is real.
ENGINE = DebugEngine()


def episode(minutes: int, ratio: float = 1.0, **kwargs):
    """One legacy episode, start to finish. Returns (plan, stats, seconds)."""
    async def run():
        plan = plan_episode("what is the nasdaq", minutes)
        pipeline = PodcastPipeline(generator=FakeGenerator(ratio), engine=ENGINE,
                                   cache=None, **kwargs)
        stats = GenerationStats()
        total = 0
        async for chunk in pipeline.stream_pcm(plan, stats):
            total += len(chunk)
        return plan, stats, pcm_duration(total, ENGINE.sample_rate)
    return asyncio.run(run())


def sentences_of(words_each: int, count: int) -> list:
    return [" ".join(f"w{i}" for i in range(words_each - 1)) + " end."
            for _ in range(count)]


def legacy_decision(words: int, remaining: float, wpm: float) -> bool:
    """`pipeline._speak_one`'s fit test, quoted rather than reimplemented."""
    return not (words / (wpm / 60.0) + SENTENCE_GAP > remaining + OVERRUN_GRACE)


# ==========================================================================
# 1. The legacy baseline, measured
# ==========================================================================
@pytest.mark.parametrize("minutes", [1, 3, 5])
def test_a_full_script_lands_on_the_requested_length(minutes):
    """The headline contract. A script with enough material fills the slot and
    is cut at a sentence boundary, never past it."""
    plan, stats, seconds = episode(minutes, ratio=1.0)
    assert plan.target_seconds == minutes * 60
    assert seconds <= plan.target_seconds + OVERRUN_GRACE
    assert plan.target_seconds - seconds <= 1.0, (
        f"{minutes} min landed {plan.target_seconds - seconds:.2f}s short")
    assert stats.truncated, "a full script is trimmed at the last boundary"


@pytest.mark.parametrize("minutes", [1, 3, 5])
def test_an_over_long_script_is_cut_rather_than_gabbled(minutes):
    """1.6x the budget must produce the same length as 1.0x, not a longer one."""
    _, _, at_budget = episode(minutes, ratio=1.0)
    plan, stats, over = episode(minutes, ratio=1.6)
    assert over == pytest.approx(at_budget, abs=0.01)
    assert over <= plan.target_seconds + OVERRUN_GRACE
    assert stats.truncated


@pytest.mark.parametrize("minutes", [1, 3, 5])
def test_a_short_script_ends_early_rather_than_being_padded_to_length(minutes):
    """"Duration is a ceiling, not a quota." Half the material must not become
    minutes of silence - only the bounded tail."""
    plan, stats, seconds = episode(minutes, ratio=0.5)
    assert seconds < plan.target_seconds - MAX_TAIL_SILENCE
    assert not stats.truncated
    assert settings.allow_topups is False, "the baseline assumes topups are off"


@pytest.mark.parametrize("minutes, ratio", [(m, r) for m in (1, 3, 5)
                                            for r in (0.5, 1.0, 1.6)])
def test_no_configuration_overshoots_the_requested_length(minutes, ratio):
    """The bound that matters, stated once over the whole grid."""
    plan, _, seconds = episode(minutes, ratio)
    assert seconds <= plan.target_seconds + OVERRUN_GRACE


def test_the_measured_baseline_is_recorded_so_a_change_is_visible():
    """Characterisation. These are not targets - they are what production did
    when Phase 6 work began, so a later change cannot pass unnoticed."""
    measured = {(minutes, ratio): round(episode(minutes, ratio)[2], 2)
                for minutes in (1, 3, 5) for ratio in (0.5, 1.0)}
    assert measured == {
        (1, 0.5): 37.97, (1, 1.0): 60.00,
        (3, 0.5): 98.98, (3, 1.0): 180.00,
        (5, 0.5): 162.85, (5, 1.0): 300.00,
    }, f"the legacy duration behaviour moved: {measured}"


# ==========================================================================
# 2. Chunk-boundary fitting
# ==========================================================================
def test_a_chunk_is_cut_at_the_last_boundary_that_fits():
    """Not accepted whole, not rejected whole."""
    sentences = sentences_of(10, 3)                 # 4s each at 150 wpm
    fit = fit_to_budget(sentences, remaining_seconds=8.5, wpm=150.0,
                        gap=SENTENCE_GAP, grace=OVERRUN_GRACE)
    assert len(fit.spoken) == 2 and len(fit.remainder) == 1
    assert fit.truncated is True


def test_nothing_is_lost_or_duplicated_by_the_cut():
    sentences = sentences_of(9, 6)
    for remaining in (0.0, 1.0, 4.0, 12.0, 60.0):
        fit = fit_to_budget(sentences, remaining, 150.0, SENTENCE_GAP,
                            OVERRUN_GRACE)
        assert fit.spoken + fit.remainder == sentences, remaining


def test_a_sentence_is_never_split_to_make_it_fit():
    """A boundary is the smallest unit. Cutting inside a sentence damages
    meaning, and cutting at a boundary is why the pipeline works in sentences."""
    sentences = ["A single very long sentence " + "word " * 60 + "ends here."]
    fit = fit_to_budget(sentences, remaining_seconds=2.0, wpm=150.0,
                        gap=SENTENCE_GAP, grace=OVERRUN_GRACE)
    assert fit.spoken == [] and fit.remainder == sentences
    assert fit.truncated is True


def test_a_whole_chunk_that_fits_is_kept_whole():
    sentences = sentences_of(10, 3)
    fit = fit_to_budget(sentences, remaining_seconds=60.0, wpm=150.0,
                        gap=SENTENCE_GAP, grace=OVERRUN_GRACE)
    assert fit.spoken == sentences and fit.remainder == []
    assert fit.truncated is False


# ==========================================================================
# 3. Overshoot: the same bound as legacy, at the boundary
# ==========================================================================
@pytest.mark.parametrize("delta, expected", [
    # `remaining` is set to exactly what the sentence needs, less the grace,
    # plus `delta`. So delta is how much budget there is beyond the limit.
    (-0.01, False),   # a hundredth of a second short: dropped
    (0.0, True),      # exactly at the limit: legacy tests `>`, so it is spoken
    (+0.01, True),    # a hundredth over: comfortably spoken
])
def test_the_boundary_is_decided_exactly_as_legacy_decides_it(delta, expected):
    words = 25
    wpm = 150.0
    needed = words / (wpm / 60.0) + SENTENCE_GAP
    remaining = needed - OVERRUN_GRACE + delta
    sentence = " ".join(f"w{i}" for i in range(words - 1)) + " end."

    fit = fit_to_budget([sentence], remaining, wpm, SENTENCE_GAP, OVERRUN_GRACE)
    assert bool(fit.spoken) is expected
    assert bool(fit.spoken) is legacy_decision(words, remaining, wpm), (
        "the chunk fit disagreed with pipeline._speak_one at the boundary")


def test_batching_does_not_widen_the_allowed_overshoot():
    """The failure this whole test file exists to prevent: a chunk-level accept
    would let the episode run over by `OVERRUN_GRACE + one whole chunk`."""
    sentences = sentences_of(20, 4)                 # 8s each at 150 wpm
    remaining = 9.0
    fit = fit_to_budget(sentences, remaining, 150.0, SENTENCE_GAP, OVERRUN_GRACE)

    assert fit.estimated_seconds <= remaining + OVERRUN_GRACE
    whole_chunk = sum(count_words(s) / 2.5 + SENTENCE_GAP for s in sentences)
    assert fit.estimated_seconds < whole_chunk, (
        "accepting the chunk whole would have overshot by nearly three chunks")


@pytest.mark.parametrize("remaining", [0.0, 0.3, 1.0, 3.7, 8.0, 15.0, 40.0])
def test_the_overshoot_bound_holds_at_every_budget(remaining):
    fit = fit_to_budget(sentences_of(12, 5), remaining, 150.0, SENTENCE_GAP,
                        OVERRUN_GRACE)
    assert fit.estimated_seconds <= remaining + OVERRUN_GRACE + 1e-9


def test_only_the_last_kept_sentence_may_cross_the_budget():
    """Every earlier sentence must have fitted outright, exactly as legacy."""
    sentences = sentences_of(10, 5)
    fit = fit_to_budget(sentences, 9.0, 150.0, SENTENCE_GAP, OVERRUN_GRACE)
    without_last = _estimate_of(fit.spoken[:-1])
    assert without_last <= 9.0


def _estimate_of(sentences: list, wpm: float = 150.0) -> float:
    return sum(count_words(s) / (wpm / 60.0) + SENTENCE_GAP for s in sentences)


def test_one_sentence_chunks_reproduce_the_legacy_decision_sequence():
    """The equivalence claim, stated as a whole episode rather than one call.

    Fed one sentence at a time - the legacy grouping - the fit must make the
    same accept/stop decision as `_speak_one` at every step, including where it
    stops. If this drifts, batching is not the only thing that changed.
    """
    sentences = sentences_of(11, 40)
    remaining, wpm = 60.0, 150.0

    ours, theirs = [], []
    budget = remaining
    for sentence in sentences:
        fit = fit_to_budget([sentence], budget, wpm, SENTENCE_GAP, OVERRUN_GRACE)
        if fit.truncated:
            break
        ours.append(sentence)
        budget = max(0.0, budget - fit.estimated_seconds)

    budget = remaining
    for sentence in sentences:
        words = count_words(sentence)
        if not legacy_decision(words, budget, wpm):
            break
        theirs.append(sentence)
        budget = max(0.0, budget - (words / (wpm / 60.0) + SENTENCE_GAP))

    assert ours == theirs
    assert 0 < len(ours) < len(sentences), "the episode neither stopped at once nor ran to the end"


def test_a_chunked_walk_never_speaks_more_than_the_one_at_a_time_walk():
    """Grouping may cost a sentence at the end - it must never gain one."""
    sentences = sentences_of(11, 40)
    remaining, wpm = 60.0, 150.0

    one_at_a_time, budget = [], remaining
    for sentence in sentences:
        fit = fit_to_budget([sentence], budget, wpm, SENTENCE_GAP, OVERRUN_GRACE)
        if fit.truncated:
            break
        one_at_a_time.append(sentence)
        budget = max(0.0, budget - fit.estimated_seconds)

    chunked, budget, index = [], remaining, 0
    while index < len(sentences):
        fit = fit_to_budget(sentences[index:index + 3], budget, wpm,
                            SENTENCE_GAP, OVERRUN_GRACE)
        chunked += fit.spoken
        budget = max(0.0, budget - fit.estimated_seconds)
        if fit.truncated:
            break
        index += 3

    assert len(chunked) <= len(one_at_a_time)
    assert chunked == one_at_a_time[:len(chunked)]


def test_a_fixed_rate_chunk_never_fits_more_than_legacy_would():
    """Legacy re-plans the rate between sentences and may speed up to fit one
    more. A chunk is one call at one rate, so it fits the same or fewer - the
    safe direction for a duration contract, and worth pinning."""
    sentences = sentences_of(15, 6)
    slow = fit_to_budget(sentences, 20.0, settings.min_wpm, SENTENCE_GAP,
                         OVERRUN_GRACE)
    fast = fit_to_budget(sentences, 20.0, settings.max_wpm, SENTENCE_GAP,
                         OVERRUN_GRACE)
    assert len(slow.spoken) <= len(fast.spoken)


# ==========================================================================
# 4. Accounting stays semantically the same
# ==========================================================================
def test_words_and_script_match_what_legacy_would_have_recorded():
    """`pace.observe` reads the word count, the cache stores the sentences, and
    analytics reads both. A chunk must contribute exactly what its sentences
    would have contributed one at a time."""
    sentences = sentences_of(11, 4)
    fit = fit_to_budget(sentences, 12.0, 150.0, SENTENCE_GAP, OVERRUN_GRACE)

    assert fit.words == sum(count_words(s) for s in fit.spoken)
    assert fit.text.split() == " ".join(fit.spoken).split()
    # The cache stores a list of sentences, not one blob: `_replay` feeds them
    # back through the speaking path, so the unit has to survive.
    assert fit.spoken == sentences[:len(fit.spoken)]


def test_the_script_a_chunk_contributes_is_still_a_list_of_sentences():
    """`stats.script` is replayed by `pipeline._replay` and re-chunked. If a
    chunk were stored as one joined string the replay would speak it as one
    sentence and the pacing would differ from the original."""
    fit = fit_to_budget(sentences_of(9, 3), 30.0, 150.0, SENTENCE_GAP,
                        OVERRUN_GRACE)
    assert isinstance(fit.spoken, list) and len(fit.spoken) == 3
    assert all(s.endswith(".") for s in fit.spoken)


def test_an_empty_fit_contributes_nothing_to_the_accounting():
    fit = fit_to_budget(sentences_of(40, 2), 0.0, 150.0, SENTENCE_GAP,
                        OVERRUN_GRACE)
    assert fit.spoken == [] and fit.words == 0 and fit.text == ""
    assert fit.estimated_seconds == 0.0


# ==========================================================================
# 5. The long-sentence edge case
# ==========================================================================
def test_a_sentence_longer_than_the_whole_budget_terminates_the_episode():
    """It must not be spoken, must not be split, and must not be retried."""
    plan = plan_episode("q", 1)
    long_one = " ".join(f"w{i}" for i in range(400)) + " end."
    fit = fit_to_budget([long_one], plan.target_seconds, 150.0, SENTENCE_GAP,
                        OVERRUN_GRACE)
    assert fit.spoken == [] and fit.truncated is True
    assert fit.remainder == [long_one]


def test_re_offering_the_remainder_makes_no_progress_and_says_so():
    """The infinite-loop guard. A caller that keeps handing back the remainder
    must see `spoken` stay empty and `truncated` stay True rather than the
    sentence being nibbled at - so the only correct move is to stop."""
    sentences = [" ".join(f"w{i}" for i in range(400)) + " end."]
    for _ in range(3):
        fit = fit_to_budget(sentences, 5.0, 150.0, SENTENCE_GAP, OVERRUN_GRACE)
        assert fit.spoken == [] and fit.truncated
        sentences = fit.remainder
    assert sentences  # unchanged, and never silently dropped


def test_a_long_sentence_first_in_a_chunk_does_not_block_the_rest():
    """Order is preserved, so a giant opening sentence truncates the episode
    rather than being skipped over to reach shorter ones. That is legacy's
    behaviour and must not change."""
    giant = " ".join(f"w{i}" for i in range(400)) + " end."
    sentences = [giant] + sentences_of(8, 2)
    fit = fit_to_budget(sentences, 10.0, 150.0, SENTENCE_GAP, OVERRUN_GRACE)
    assert fit.spoken == []
    assert fit.remainder == sentences


@pytest.mark.parametrize("minutes", [1, 3, 5])
def test_the_real_pipeline_terminates_on_an_over_long_script(minutes):
    """End to end on the legacy path: the episode ends, it does not hang."""
    plan, stats, seconds = episode(minutes, ratio=3.0)
    assert stats.truncated and seconds <= plan.target_seconds + OVERRUN_GRACE


# ==========================================================================
# 6. Tail silence
# ==========================================================================
def test_tail_silence_is_bounded_and_is_what_legacy_emits():
    async def run(spoken_seconds, sentences):
        pipeline = PodcastPipeline(generator=None, engine=ENGINE, cache=None)
        pace = PaceController(target_seconds=60.0, total_words=150,
                              sample_rate=ENGINE.sample_rate)
        pace.observe(int(spoken_seconds * ENGINE.sample_rate * 2), 100)
        stats = GenerationStats()
        stats.sentences = sentences
        total = 0
        async for chunk in pipeline._finish(pace, stats):
            total += len(chunk)
        return pcm_duration(total, ENGINE.sample_rate)

    assert asyncio.run(run(20.0, 9)) == pytest.approx(MAX_TAIL_SILENCE, abs=0.01)
    assert asyncio.run(run(57.0, 9)) == pytest.approx(3.0, abs=0.01)
    assert asyncio.run(run(59.99, 9)) == 0.0


def test_an_empty_script_gets_no_tail_at_all():
    """Padding an empty episode manufactures something that looks valid to
    every layer above - which is how a failed script once reached listeners as
    "it generated, but I hear nothing"."""
    async def run():
        pipeline = PodcastPipeline(generator=None, engine=ENGINE, cache=None)
        pace = PaceController(target_seconds=60.0, total_words=150,
                              sample_rate=ENGINE.sample_rate)
        stats = GenerationStats()          # sentences == 0
        return sum([len(c) async for c in pipeline._finish(pace, stats)] or [0])

    assert asyncio.run(run()) == 0


def test_assembly_cannot_change_how_much_tail_is_owed():
    """The tail is computed from elapsed audio, not from how the text was
    grouped. Two groupings of the same sentences owe the same tail."""
    sentences = sentences_of(10, 6)
    one_at_a_time = _estimate_of(sentences)
    as_chunks = _estimate_of(sentences[:3]) + _estimate_of(sentences[3:])
    assert one_at_a_time == pytest.approx(as_chunks, abs=1e-9)


# ==========================================================================
# 7. Top-ups
# ==========================================================================
def test_topups_fill_a_short_episode_when_enabled(monkeypatch):
    """The legacy semantics this must not disturb.

    `Settings` is frozen, so the flag is swapped by replacing the object
    `pipeline` reads - which is also how `experiments` and `_answer_first`
    override settings, and it keeps `__post_init__` validation in play.
    """
    import dataclasses

    import config
    import pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "settings",
                        dataclasses.replace(config.settings, allow_topups=True))
    plan, stats, seconds = episode(3, ratio=0.5)
    assert stats.topups > 0
    assert seconds > 98.98, "the top-up added nothing"
    assert seconds <= plan.target_seconds + OVERRUN_GRACE
    assert config.settings.allow_topups is False, "the default must stay off"


def test_a_topup_is_subject_to_the_same_fit_as_the_body():
    """A top-up arrives when little time is left, which is exactly where a
    chunk-level accept would overshoot. It gets the same treatment."""
    remaining = 4.0
    fit = fit_to_budget(sentences_of(20, 3), remaining, 150.0, SENTENCE_GAP,
                        OVERRUN_GRACE)
    assert fit.estimated_seconds <= remaining + OVERRUN_GRACE
    assert fit.truncated


def test_a_topup_into_no_remaining_time_speaks_nothing():
    fit = fit_to_budget(sentences_of(15, 2), 0.0, 150.0, SENTENCE_GAP,
                        OVERRUN_GRACE)
    assert fit.spoken == [] and fit.truncated


# ==========================================================================
# 8. answer_first
# ==========================================================================
def test_answer_first_still_divides_the_episode_by_its_share():
    assert 0.0 < settings.answer_first_share <= 1.0
    plan = plan_episode("q", 3)
    ceiling = plan.target_seconds * settings.answer_first_share
    assert ceiling == pytest.approx(90.0)


def test_two_streams_need_two_assemblers_and_do_not_share_state():
    """`_answer_first` runs an instant and a researched stream at once. One
    assembler across both would interleave two scripts into one chunk, which is
    a text-integrity failure, not just a pacing one."""
    instant = SpeechAssembler(policy=AssemblyPolicy())
    research = SpeechAssembler(policy=AssemblyPolicy())

    instant.offer("The durable half of the answer starts here.")
    research.offer("The researched half starts here instead.")

    assert instant.seen != research.seen
    assert instant.released == 1 and research.released == 1
    assert instant.spoken_matches_source([]) is not None   # independent witness


def test_each_stream_accounts_for_only_its_own_words():
    """The handover shares one PaceController, so the two streams must not
    double-count: what each contributes is exactly what it spoke."""
    instant = sentences_of(10, 2)
    research = sentences_of(12, 3)
    a = fit_to_budget(instant, 30.0, 150.0, SENTENCE_GAP, OVERRUN_GRACE)
    b = fit_to_budget(research, 30.0 - a.estimated_seconds, 150.0, SENTENCE_GAP,
                      OVERRUN_GRACE)
    assert a.words == sum(count_words(s) for s in instant)
    assert a.estimated_seconds + b.estimated_seconds <= 30.0 + OVERRUN_GRACE


def test_the_handover_ceiling_is_a_time_not_a_chunk_count():
    """Whatever assembly does to grouping, the instant half's ceiling stays the
    share of the episode it always was."""
    plan = plan_episode("q", 5)
    ceiling = plan.target_seconds * settings.answer_first_share
    fit = fit_to_budget(sentences_of(30, 20), ceiling, 150.0, SENTENCE_GAP,
                        OVERRUN_GRACE)
    assert fit.estimated_seconds <= ceiling + OVERRUN_GRACE


# ==========================================================================
# 9. The first-chunk invariant, under duration pressure
# ==========================================================================
def test_the_assembler_never_consults_a_duration_budget():
    """Duration enforcement belongs to the fit, which happens after release.
    If the assembler learned about time it could hold the opening, and the
    first-chunk latency invariant would be gone."""
    import inspect
    import re

    import speech_assembly

    source = inspect.getsource(speech_assembly.SpeechAssembler)
    for name in ("remaining_seconds", "target_seconds", "budget", "pace",
                 "PaceController"):
        assert not re.search(rf"\b{name}\b", source), (
            f"the assembler learned about {name}")


def test_a_complete_first_thought_is_released_however_little_time_remains():
    """The regression condition: duration enforcement must never hold a
    complete first speakable thought merely to wait for more text."""
    assembler = SpeechAssembler(policy=AssemblyPolicy())
    released = assembler.offer("It was built to ring.")
    assert len(released) == 1 and released[0].words == 5


def test_a_first_thought_that_fits_is_still_spoken_at_a_tight_budget():
    """Released immediately by the assembler, and then accepted by the fit -
    so nothing between the model and the voice delays it."""
    first = "It was built to ring."
    released = SpeechAssembler(policy=AssemblyPolicy()).offer(first)
    fit = fit_to_budget([c.text for c in released], remaining_seconds=2.0,
                        wpm=150.0, gap=SENTENCE_GAP, grace=OVERRUN_GRACE)
    assert fit.spoken == [first] and fit.truncated is False


def test_a_first_thought_too_long_for_the_budget_is_refused_not_held():
    """The one case where the opening does not get spoken, it is *dropped* by
    the duration contract - not queued up waiting for more text."""
    giant = " ".join(f"w{i}" for i in range(400)) + " end."
    released = SpeechAssembler(policy=AssemblyPolicy()).offer(giant)
    assert len(released) == 1, "the assembler still released it immediately"
    fit = fit_to_budget([released[0].text], 5.0, 150.0, SENTENCE_GAP,
                        OVERRUN_GRACE)
    assert fit.spoken == [] and fit.truncated is True
