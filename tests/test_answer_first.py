"""Answer first, research underneath.

A question that needs today's facts costs 10-25 seconds of searching before a
word can be written. Rather than make the listener wait, or cover the wait with
filler, both halves start at once: one with no tools that begins writing
immediately, and one with search that is still reading. The first is spoken
while the second works.

This is the shape the cold open had. The difference is the whole point, and
these tests are mostly about that difference: the cover is now the answer.
"""
from __future__ import annotations

import asyncio
import dataclasses
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pipeline as pipeline_mod  # noqa: E402
import script_generator as sg  # noqa: E402
from pipeline import GenerationStats, PodcastPipeline  # noqa: E402
from script_generator import ROLE_BRIEFS, build_prompt, plan_episode  # noqa: E402
from tts import DebugEngine  # noqa: E402


class TwoHalves:
    """A generator whose researched half is slow, like a real one."""

    def __init__(self, research_delay: float = 0.4, research_fails: bool = False):
        self.research_delay = research_delay
        self.research_fails = research_fails
        self.roles: list[str] = []
        self.searched: list[bool] = []

    async def stream_sentences(self, plan, notes=None):
        self.roles.append(plan.role)
        self.searched.append(plan.search)
        if plan.search:
            await asyncio.sleep(self.research_delay)
            if self.research_fails:
                raise RuntimeError("search is down")
            for i in range(30):
                yield f"Researched sentence {i} with a number in it."
        else:
            for i in range(60):
                await asyncio.sleep(0.005)
                yield f"Known sentence {i} explaining how the thing works."


def _run(pipe, plan, stats):
    async def go():
        async for _ in pipe.stream_pcm(plan, stats):
            pass
    asyncio.run(go())


def _episode(generator, minutes: int = 3) -> GenerationStats:
    """One researched episode, run to the end, with its stats."""
    stats = GenerationStats()
    _run(PodcastPipeline(generator=generator, engine=DebugEngine(), cache=None),
         plan_episode("latest news on the fed", minutes), stats)
    return stats


@pytest.fixture
def on(monkeypatch):
    patched = dataclasses.replace(pipeline_mod.settings, answer_first=True)
    monkeypatch.setattr(pipeline_mod, "settings", patched)
    return patched


# --- the two halves ---------------------------------------------------------

def test_a_researched_episode_starts_both_halves_at_once(on):
    """Serialising them would just be the wait with extra steps."""
    gen = TwoHalves()
    stats = GenerationStats()
    _run(PodcastPipeline(generator=gen, engine=DebugEngine(), cache=None),
         plan_episode("latest news on the fed", 2), stats)

    assert sorted(gen.roles) == ["continuation", "opening"]
    assert sorted(gen.searched) == [False, True], "one half must search, one must not"
    assert stats.answered_first is True


def test_an_unresearched_episode_makes_only_one_call(on):
    """The cover exists to fill a research wait, so no research means no cover.

    Reaching an unresearched episode now takes `search=False`: production
    researches every one (PROBLEMS.md 76), and the question no longer opts
    out on the listener's behalf. The rule being tested is unchanged - the
    second call is paid for by the wait, not by the episode.
    """
    gen = TwoHalves()
    _run(PodcastPipeline(generator=gen, engine=DebugEngine(), cache=None),
         plan_episode("what is the NASDAQ", 2, search=False), GenerationStats())
    assert gen.roles == [""], "an unresearched episode split itself in two"


def test_the_listener_hears_the_known_half_first_then_the_researched_one(on):
    """The handover is the feature: audio starts before research finishes."""
    gen = TwoHalves(research_delay=0.8)
    stats = GenerationStats()
    _run(PodcastPipeline(generator=gen, engine=DebugEngine(), cache=None),
         plan_episode("latest news on the fed", 3), stats)

    spoken = " ".join(stats.script)
    assert "Known sentence" in spoken, "the instant half was never spoken"
    assert "Researched sentence" in spoken, "research never took over"
    first_research = next(i for i, s in enumerate(stats.script) if "Researched" in s)
    assert first_research > 0, "the instant half did not go first"
    assert all("Known" in s for s in stats.script[:first_research]), "the halves interleaved"
    assert stats.handover_seconds > 0


def test_nothing_from_the_known_half_is_spoken_after_the_handover(on):
    """A seam the listener can hear is worse than the wait would have been."""
    gen = TwoHalves(research_delay=0.3)
    stats = GenerationStats()
    _run(PodcastPipeline(generator=gen, engine=DebugEngine(), cache=None),
         plan_episode("latest news on the fed", 3), stats)
    seen_research = False
    for sentence in stats.script:
        if "Researched" in sentence:
            seen_research = True
        elif seen_research:
            pytest.fail("the instant half spoke again after research took over")


def test_faster_research_means_less_of_the_known_half(on):
    """The instant half is cover, not a fixed preamble: it lasts exactly as
    long as the research does and not a sentence longer."""
    lengths = {}
    for delay in (0.05, 0.9):
        gen = TwoHalves(research_delay=delay)
        stats = GenerationStats()
        _run(PodcastPipeline(generator=gen, engine=DebugEngine(), cache=None),
             plan_episode("latest news on the fed", 3), stats)
        lengths[delay] = sum(1 for s in stats.script if "Known" in s)
    assert lengths[0.05] < lengths[0.9], (
        f"fast research still spoke {lengths[0.05]} known sentences vs "
        f"{lengths[0.9]} for slow research - the cover is not tracking the wait"
    )


def test_the_episode_survives_research_failing(on):
    """A dead search must degrade to the answer from knowledge, not silence."""
    gen = TwoHalves(research_delay=0.2, research_fails=True)
    stats = GenerationStats()
    pipe = PodcastPipeline(generator=gen, engine=DebugEngine(), cache=None)
    with pytest.raises(RuntimeError):
        _run(pipe, plan_episode("latest news on the fed", 2), stats)
    # What matters is that the listener was already being spoken to when it died.
    assert any("Known sentence" in s for s in stats.script), (
        "nothing had been said before research failed"
    )


def test_it_can_be_switched_off(monkeypatch):
    """ANSWER_FIRST=0 goes back to one call and an honest wait."""
    monkeypatch.setattr(pipeline_mod, "settings",
                        dataclasses.replace(pipeline_mod.settings, answer_first=False))
    gen = TwoHalves()
    _run(PodcastPipeline(generator=gen, engine=DebugEngine(), cache=None),
         plan_episode("latest news on the fed", 2), GenerationStats())
    assert gen.roles == [""]


# --- the halves are told which job is whose ---------------------------------

def test_the_opening_is_told_to_answer_not_to_stall():
    """The whole difference from the cold open. That one was told to state no
    facts; this one is told to answer."""
    brief = ROLE_BRIEFS["opening"]
    assert "Answer from" in brief
    assert "no waiting" in brief or "starting immediately" in brief
    for banned in ("state no facts", "never answer", "frame the question"):
        assert banned not in brief.lower(), f"the opening is being told to stall: {banned}"


def test_the_continuation_is_told_not_to_open_the_episode_again():
    """Two openings stacked is the failure this replaces a seam with."""
    brief = ROLE_BRIEFS["continuation"]
    assert "already playing" in brief
    assert "Do not re-introduce" in brief
    assert "contradicts" in brief, "no instruction to correct a stale opening"


def test_a_whole_episode_gets_neither_brief():
    plan = plan_episode("what is the NASDAQ", 3)
    text = build_prompt(plan)
    assert "taking over mid-episode" not in text
    assert "continue after you" not in text


@pytest.mark.parametrize("role", ["opening", "continuation"])
def test_each_half_carries_its_own_brief(role):
    plan = dataclasses.replace(plan_episode("latest news on the fed", 3), role=role)
    assert ROLE_BRIEFS[role].strip()[:40] in build_prompt(plan)


# --------------------------------------------------------------------------
# the handover must not be a silence
#
# The 4090 run starved on the researched question. `_answer_first` treated
# ANSWER_FIRST_SHARE as a deadline: the cover stopped at the ceiling and the
# next line blocked on `research.next()`, so a listener already mid-episode
# heard nothing until the search returned. The ceiling is about sharing, and
# dead air is a worse outcome than an over-long opening - especially since the
# opening is a real answer rather than filler.
# --------------------------------------------------------------------------
def test_the_cover_keeps_speaking_when_research_is_not_ready_yet(on):
    """The fix. Past the ceiling, silence is the alternative, so it keeps
    going - and records that it did, so the trade is visible."""
    generator = TwoHalves(research_delay=0.6)
    stats = _episode(generator, minutes=1)
    assert stats.cover_overran is True, "the cover stopped at the ceiling"
    assert stats.cover_seconds > 0


def test_the_cover_still_gives_way_the_moment_research_lands(on):
    """Overrunning is only ever to avoid silence. The instant research has a
    sentence, the episode moves to it."""
    generator = TwoHalves(research_delay=0.0)
    stats = _episode(generator, minutes=3)
    assert stats.handover_reason == "research ready"
    assert stats.cover_overran is False, "it overran with research already up"


def test_the_cover_cannot_eat_the_episode(on):
    """The failure the ceiling was written to prevent, which ignoring the
    ceiling would have reintroduced.

    Synthesis runs several times faster than research, so a cover with no cap
    finishes the whole episode while the search is still reading - and the
    listener gets an unresearched answer to a question that was researched
    *because* it needed today's facts.
    """
    from config import settings as cfg
    from speech_assembly import AssemblyPolicy

    generator = TwoHalves(research_delay=5.0)
    stats = _episode(generator, minutes=1)

    # The cap is checked between items, and under Phase 6 an item is a chunk
    # rather than a sentence - so the cover can overshoot by at most one chunk,
    # exactly as the duration ceiling overshoots by at most one sentence plus
    # OVERRUN_GRACE. Stated as that bound rather than as a round number, so it
    # tracks the policy instead of a measurement taken once.
    one_chunk = AssemblyPolicy().max_words / (cfg.target_wpm / 60.0)
    cap = stats.plan_seconds * cfg.answer_first_max_share
    assert stats.cover_seconds <= cap + one_chunk, (
        f"the cover reached {stats.cover_seconds:.1f}s against a {cap:.1f}s "
        f"cap and a {one_chunk:.1f}s chunk")
    assert stats.handover_reason == "cover cap reached"
    # The point of the cap: research still got said.
    assert any("Researched" in sentence for sentence in stats.script), (
        "the cover ate the episode after all")


def _short_cover(sentences: int):
    """A TwoHalves whose from-knowledge half runs out almost immediately."""
    original = TwoHalves.stream_sentences

    async def short(self, plan, notes=None):
        if plan.search:
            async for sentence in original(self, plan, notes):
                yield sentence
        else:
            for i in range(sentences):
                yield f"Known sentence {i} explaining how the thing works."

    return original, short


def test_the_cover_running_out_is_not_a_gap_while_the_buffer_holds(on, caplog):
    """The 4090 case, reproduced.

    The cover exhausted at 85.8s of audio having taken 24.7s of wall clock, so
    the listener had ~66s buffered; research's first item arrived 3.9s later.
    The run warned of silence and the listener heard none - the same run
    reported a tightest playback margin of +9.95s.

    Whether a handover is audible is not "is research ready", which is a fact
    about the producer. It is whether the wait outlasts the audio already made
    and not yet played.
    """
    import logging

    original, short = _short_cover(6)
    TwoHalves.stream_sentences = short
    try:
        with caplog.at_level(logging.INFO, logger="pipeline"):
            stats = _episode(TwoHalves(research_delay=0.4), minutes=3)
    finally:
        TwoHalves.stream_sentences = original

    assert stats.handover_reason == "instant half exhausted"
    assert stats.handover_stall_seconds > 0, "the stall was not measured"
    assert stats.handover_buffer_seconds > stats.handover_stall_seconds
    assert stats.handover_gap_seconds == 0, (
        f"reported {stats.handover_gap_seconds:.2f}s of silence the listener "
        "did not hear")
    assert not any("GAP" in r.getMessage() for r in caplog.records), (
        "warned of a silence that the buffer covered")
    assert any("no silence reached the listener" in r.getMessage()
               for r in caplog.records)


def test_a_gap_the_listener_would_hear_is_reported_as_one(on, caplog):
    """The other half of the same rule, so the warning keeps its teeth.

    One short cover sentence is a thin buffer; research five seconds away
    outlasts it. That is real silence, and it must still be named.
    """
    import logging

    original, short = _short_cover(1)
    TwoHalves.stream_sentences = short
    try:
        with caplog.at_level(logging.WARNING, logger="pipeline"):
            stats = _episode(TwoHalves(research_delay=5.0), minutes=3)
    finally:
        TwoHalves.stream_sentences = original

    assert stats.handover_reason == "instant half exhausted"
    assert stats.handover_gap_seconds > 0, "real silence went unreported"
    assert any("GAP" in r.getMessage() for r in caplog.records), caplog.text
    assert any("heard" in r.getMessage() for r in caplog.records)


def test_the_gap_is_the_stall_less_the_buffer_and_never_negative(on):
    """The arithmetic itself, so the definition cannot drift."""
    original, short = _short_cover(6)
    TwoHalves.stream_sentences = short
    try:
        stats = _episode(TwoHalves(research_delay=0.3), minutes=3)
    finally:
        TwoHalves.stream_sentences = original

    expected = max(0.0, stats.handover_stall_seconds
                   - stats.handover_buffer_seconds)
    assert stats.handover_gap_seconds == pytest.approx(expected, abs=0.01)
    assert stats.handover_gap_seconds >= 0


def test_the_share_is_a_ceiling_and_the_cap_is_above_it():
    """Two numbers that must not cross, or the cover would be capped before it
    was allowed to start."""
    from config import settings as cfg

    assert 0 < cfg.answer_first_share <= cfg.answer_first_max_share <= 1.0
