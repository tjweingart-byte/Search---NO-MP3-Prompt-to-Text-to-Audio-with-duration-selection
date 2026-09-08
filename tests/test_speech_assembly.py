"""Speech-sized chunks: the policy, and the rules it must never break.

Ported from the Phase 6 experiment, which validated this behaviour on an RTX
4090 (`phase6_4090_20260908T064113Z`: 22 sentences became 15 synthesis calls,
median 33 words, zero playback stalls). These are the invariants that result
depended on.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from speech_assembly import (AssemblyPolicy, SpeechAssembler,
                             audio_seconds_for)


class Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t
    def advance(self, seconds): self.t += seconds


def words(n: int, marker: str = "w") -> str:
    return " ".join(f"{marker}{i}" for i in range(n - 1)) + " end."


def _assembler(**overrides):
    clock = Clock()
    return SpeechAssembler(policy=AssemblyPolicy(**overrides), clock=clock), clock


# --------------------------------------------------------------------------
# the first chunk: time to first listen outranks everything
# --------------------------------------------------------------------------
@pytest.mark.parametrize("sentence", [
    "It happened.",                                        # 2 words
    "The clock has no face.",                              # 5
    "It was built to ring, not to be read.",               # 9
    "Salisbury has kept it turning since thirteen eighty-six now.",   # 8
    "Nobody there needed the minute, only the hour of prayer, and that was enough.",
])
def test_a_short_but_complete_first_sentence_is_released_immediately(sentence):
    """The regression test for the rule this module exists to hold.

    A complete natural sentence goes to the voice on the offer that produced
    it - at two words or at fifteen - and is never held for a second sentence,
    a word count, or a preferred size.
    """
    assembler, clock = _assembler()
    out = assembler.offer(sentence)
    assert len(out) == 1, f"{sentence!r} was held instead of released"
    assert out[0].index == 0 and out[0].sentences == 1
    assert out[0].text == sentence
    assert out[0].held_seconds == 0.0
    assert clock.t == 0.0            # nothing waited for anything
    assert out[0].words < AssemblyPolicy().min_words


def test_a_long_opening_sentence_also_goes_straight_out():
    assembler, _ = _assembler()
    out = assembler.offer(words(33))
    assert len(out) == 1 and out[0].index == 0
    assert out[0].reason.startswith("first chunk")
    assert out[0].held_seconds == 0.0


def test_there_is_no_first_chunk_size_knob_to_override():
    """A knob left behind with a default of zero is an invitation to raise it."""
    assert not any("first_min" in name for name in vars(AssemblyPolicy()))
    assert not any("first" in name and "word" in name
                   for name in vars(AssemblyPolicy()))


def test_the_only_first_chunk_gate_is_completeness_not_size():
    """A half-written clause is not a speakable thought. That is the whole
    safeguard, and it is semantic rather than dimensional."""
    assembler, _ = _assembler()
    assert assembler.offer("It was built to") == []
    out = assembler.offer("ring the hours.")
    assert len(out) == 1 and out[0].sentences == 2
    assert out[0].reason == "first chunk, released on the first complete thought"


def test_an_incomplete_opening_is_still_never_held_indefinitely():
    assembler, clock = _assembler()
    assert assembler.offer("A clause with no ending") == []
    clock.advance(AssemblyPolicy().max_wait_seconds + 0.01)
    out = assembler.due()
    assert out and "held long enough" in out[0].reason


def test_the_first_handoff_risk_the_missing_floor_leaves_open():
    """Kept as arithmetic because it is measured by the caller, not prevented
    here: a very short opening can fail to cover the second chunk's synthesis.
    Break-even is about 5.6 words against a target-sized follower and 10.2
    against the largest allowed."""
    policy = AssemblyPolicy()
    against_target = 0.10856 * policy.target_words - 0.8113      # ~2.23s
    against_cap = 0.10856 * policy.max_words - 0.8113            # ~4.07s
    assert audio_seconds_for(5, wpm=150) < against_target
    assert audio_seconds_for(6, wpm=150) > against_target
    assert audio_seconds_for(10, wpm=150) < against_cap
    assert audio_seconds_for(11, wpm=150) > against_cap


# --------------------------------------------------------------------------
# later chunks: batching, bounded every way
# --------------------------------------------------------------------------
def _past_first(assembler):
    assembler.offer(words(30))
    assert assembler.released == 1


def test_later_sentences_accumulate_towards_the_target():
    assembler, _ = _assembler()
    _past_first(assembler)
    assert assembler.offer(words(8)) == []
    assert assembler.offer(words(8)) == []
    out = assembler.offer(words(14))
    assert out and out[0].words >= AssemblyPolicy().target_words
    assert out[0].sentences == 3


def test_a_chunk_never_exceeds_the_cap():
    assembler, _ = _assembler()
    _past_first(assembler)
    released = []
    for _ in range(6):
        released += assembler.offer(words(20))
    released += assembler.flush()
    assert all(c.words <= AssemblyPolicy().max_words for c in released), \
        [c.words for c in released]


def test_a_single_sentence_longer_than_the_cap_is_never_split():
    """Splitting a sentence would damage meaning. It goes alone, oversized."""
    assembler, _ = _assembler()
    _past_first(assembler)
    out = assembler.offer(words(80))
    assert len(out) == 1 and out[0].sentences == 1 and out[0].words == 80


def test_text_is_never_held_indefinitely():
    assembler, clock = _assembler()
    _past_first(assembler)
    assert assembler.offer(words(4)) == []
    assert assembler.due() == []
    clock.advance(AssemblyPolicy().max_wait_seconds + 0.01)
    out = assembler.due()
    assert out and out[0].reason == "held long enough"


def test_low_headroom_beats_ideal_batching():
    """If the listener is about to catch up, latency wins."""
    assembler, _ = _assembler()
    _past_first(assembler)
    assert assembler.offer(words(6), headroom=30.0) == []
    out = assembler.offer(words(3), headroom=1.0)
    assert out and "headroom" in out[0].reason


def test_a_question_mark_past_the_minimum_ends_a_chunk():
    assembler, _ = _assembler()
    _past_first(assembler)
    assembler.offer(words(20))
    out = assembler.offer("So what changed?")
    assert out and out[0].reason == "natural beat past the minimum"


def test_a_short_final_fragment_still_ships():
    assembler, _ = _assembler()
    _past_first(assembler)
    assembler.offer("And that was that.")
    out = assembler.flush()
    assert out and out[0].reason == "end of script" and out[0].words == 4


def test_flushing_an_empty_assembler_emits_nothing():
    assembler, _ = _assembler()
    assert assembler.flush() == []


# --------------------------------------------------------------------------
# text integrity
# --------------------------------------------------------------------------
def test_the_spoken_text_is_exactly_the_sentences_offered_in_order():
    assembler, clock = _assembler()
    released = []
    for size in (30, 7, 9, 12, 4, 20, 3):
        released += assembler.offer(words(size))
        clock.advance(0.1)
    released += assembler.flush()
    assert assembler.spoken_matches_source(released) is None
    assert " ".join(c.text for c in released).split() == \
        " ".join(assembler.seen).split()


def test_lost_text_is_detected():
    assembler, _ = _assembler()
    released = assembler.offer(words(30))
    released += assembler.offer(words(9))
    released += assembler.flush()
    assert assembler.spoken_matches_source(released[:1]).startswith("text was lost")


def test_duplicated_text_is_detected():
    assembler, _ = _assembler()
    released = assembler.offer(words(30)) + assembler.flush()
    assert assembler.spoken_matches_source(
        released + released).startswith("text was duplicated")


def test_reordered_text_is_detected():
    assembler, clock = _assembler()
    released = assembler.offer(words(30))
    clock.advance(0.1)
    released += assembler.offer(words(20, "x"))
    released += assembler.flush()
    assert len(released) >= 2
    assert assembler.spoken_matches_source(list(reversed(released))) is not None


# --------------------------------------------------------------------------
# the policy, and its ties to production
# --------------------------------------------------------------------------
def test_an_incoherent_policy_is_refused_at_construction():
    with pytest.raises(ValueError, match="min <= target <= max"):
        AssemblyPolicy(min_words=40, target_words=20)
    with pytest.raises(ValueError):
        AssemblyPolicy(min_words=0)
    with pytest.raises(ValueError):
        AssemblyPolicy(target_words=60, max_words=45)


def test_the_defaults_stay_inside_the_measured_range():
    """Nothing above 41 words has been measured on a 4090. The cap must not
    invent a size on the strength of a three-point extrapolation."""
    policy = AssemblyPolicy()
    assert policy.max_words <= 45 and policy.target_words <= 41


def test_speaking_rate_comes_from_production_settings():
    """Not a copy of 150: a second source of truth for the speaking rate is
    how the planned duration and the assembler drift apart."""
    from config import settings

    assert audio_seconds_for(settings.target_wpm) == pytest.approx(60.0)


def test_word_counting_is_production_s_own():
    """The assembler must count words the way the pace controller does."""
    import speech_assembly
    from script_generator import count_words

    assert speech_assembly.count_words is count_words
