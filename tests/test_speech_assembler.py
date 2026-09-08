"""Adaptive speech packetisation: the policy, and the rules it must never break."""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments.speech_assembler import (AssemblyPolicy, SpeechAssembler,
                                          audio_seconds_for, count_words)


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
# the first chunk: latency is king
# --------------------------------------------------------------------------
def test_an_opening_sentence_past_the_floor_goes_straight_out():
    """Phase 5's first chunk was about 33 words. Nothing should hold it."""
    assembler, _ = _assembler()
    out = assembler.offer(words(33))
    assert len(out) == 1 and out[0].index == 0
    assert out[0].reason.startswith("first chunk")
    assert out[0].held_seconds == 0.0


def test_a_tiny_opening_sentence_waits_only_for_the_floor():
    """A two-word opening buys 0.8s of playback; the second chunk needs longer
    than that to synthesise, so playback would stall on the first handoff."""
    assembler, _ = _assembler()
    assert assembler.offer("It happened.") == []
    out = assembler.offer(words(12))
    assert len(out) == 1 and out[0].sentences == 2
    assert out[0].words >= AssemblyPolicy().first_min_words


def test_the_floor_is_covered_by_the_arithmetic_that_set_it():
    """audio(first chunk) must exceed generate(max chunk) or the first handoff
    stalls. 0.1086 s/word from the 4090 curve, at TARGET_WPM."""
    policy = AssemblyPolicy()
    worst_case_generate = 0.10856 * policy.max_words - 0.8113
    assert audio_seconds_for(policy.first_min_words) > worst_case_generate


def test_the_first_chunk_is_never_held_for_the_later_target():
    assembler, clock = _assembler()
    out = assembler.offer(words(14))
    assert out and out[0].words < AssemblyPolicy().target_words


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
# text integrity - assertions 5 and 6
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
    released = assembler.offer(words(30))
    released += assembler.flush()
    witness = assembler.spoken_matches_source(released + released)
    assert witness.startswith("text was duplicated")


def test_reordered_text_is_detected():
    assembler, clock = _assembler()
    released = assembler.offer(words(30))
    clock.advance(0.1)
    released += assembler.offer(words(20, "x"))
    released += assembler.flush()
    assert len(released) >= 2
    assert assembler.spoken_matches_source(list(reversed(released))) is not None


# --------------------------------------------------------------------------
# the policy itself
# --------------------------------------------------------------------------
def test_an_incoherent_policy_is_refused_at_construction():
    with pytest.raises(ValueError, match="first_min <= min <= target <= max"):
        AssemblyPolicy(min_words=40, target_words=20)
    with pytest.raises(ValueError):
        AssemblyPolicy(first_min_words=0)


def test_the_defaults_stay_inside_the_measured_range():
    """Nothing above 41 words has been measured on a 4090. The cap must not
    invent a size on the strength of a three-point extrapolation."""
    policy = AssemblyPolicy()
    assert policy.max_words <= 45
    assert policy.target_words <= 41
    assert count_words(words(policy.target_words)) == policy.target_words
