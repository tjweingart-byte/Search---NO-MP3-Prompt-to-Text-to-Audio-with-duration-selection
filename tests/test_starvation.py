"""Once playback starts, the listener must not hear silence.

The RTX 4090 run starved: playback began and then ran out. Nothing in the suite
caught it, because the mechanism that prevents it had never been exercised
where it actually runs.

The margin is thin by design, and the arithmetic is worth stating because it is
what makes this a real risk rather than a theoretical one. The first chunk is
deliberately the first complete thought and nothing more - roughly 9 words, or
3.6s of audio at TARGET_WPM. Chatterbox generates at about 4.6x realtime, so a
45-word chunk (the assembler's cap) takes about 3.9s to synthesise, and no
bytes flow while it does. 3.9s of synthesis against 3.6s of buffer is silence.

`AssemblyPolicy.headroom_floor_seconds` exists exactly for this: below it,
batching stops mattering and whatever is pending ships immediately. It was dead
code in production - `_start_phase6` called `offer()` and `due()` without ever
passing headroom, so the rule could not fire. These tests are about the wiring,
not the rule, because the rule was already tested and already correct.
"""
from __future__ import annotations

import asyncio
import dataclasses
import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pipeline as pipeline_mod  # noqa: E402
from audio_utils import PaceController  # noqa: E402
from pipeline import GenerationStats, PodcastPipeline  # noqa: E402
from speech_assembly import AssemblyPolicy, SpeechAssembler  # noqa: E402
from tts import DebugEngine  # noqa: E402


# --------------------------------------------------------------------------
# the wiring that was missing
# --------------------------------------------------------------------------
def test_the_producer_passes_headroom_to_both_assembler_paths():
    """The defect itself: `offer()` and `due()` were called bare.

    Read from the source rather than behaviour because the failure is an
    omission - a call that looks right and silently disables a rule - and an
    omission is invisible from the outside until a listener hears it.
    """
    source = inspect.getsource(pipeline_mod.PodcastPipeline._start_phase6)
    assert "assembler.due(left)" in source, "the timer path lost its headroom"
    assert "assembler.offer(sentence, left)" in source, (
        "the sentence path lost its headroom")
    assert "assembler.due()" not in source
    assert "assembler.offer(sentence)" not in source


def _pace(seconds_of_audio: float = 0.0) -> PaceController:
    from config import settings as cfg

    pace = PaceController(target_seconds=180, total_words=450)
    pace.emitted_bytes = int(seconds_of_audio * cfg.sample_rate
                             * cfg.sample_width)
    return pace


def test_the_probe_measures_audio_made_against_wall_clock():
    """Headroom is what the listener has left to play, and it must be read
    live - a value captured when the last chunk finished is already stale."""
    stats = GenerationStats()
    empty = PodcastPipeline._headroom_probe(_pace(0.0), stats)()
    full = PodcastPipeline._headroom_probe(_pace(60.0), stats)()
    assert full - empty == pytest.approx(60.0, abs=0.5), (
        "a minute of audio must be a minute of headroom")


def test_the_probe_falls_as_wall_clock_passes():
    """Nothing new synthesised means the listener is catching up."""
    import time

    probe = PodcastPipeline._headroom_probe(_pace(10.0), GenerationStats())
    before = probe()
    time.sleep(0.05)
    assert probe() < before


# --------------------------------------------------------------------------
# what the rule does once it is wired
# --------------------------------------------------------------------------
def test_a_thin_buffer_ships_the_pending_text_immediately():
    """The 4090 case: one short sentence pending, buffer nearly empty. It must
    go now, not wait for the 18-word minimum."""
    assembler = SpeechAssembler(policy=AssemblyPolicy())
    assembler.offer("A first complete thought.")          # releases: opening
    chunks = assembler.offer("Four more words here.", headroom=1.0)
    assert chunks, "a thin buffer must not batch"
    assert chunks[0].words < AssemblyPolicy().min_words
    assert "headroom" in chunks[0].reason


def test_a_comfortable_buffer_still_batches():
    """The rule must not become 'never batch'. Batching is what keeps the cost
    per word down, and it is safe whenever the listener is well ahead."""
    assembler = SpeechAssembler(policy=AssemblyPolicy())
    assembler.offer("A first complete thought.")
    assert not assembler.offer("Four more words here.", headroom=30.0), (
        "a comfortable buffer batched nothing")


def test_the_cap_is_never_reached_on_a_thin_buffer():
    """A 45-word chunk takes ~3.9s to synthesise at 4.6x realtime. That is
    longer than the first chunk's own audio, so reaching the cap while the
    buffer is thin is the starvation, stated as a size."""
    assembler = SpeechAssembler(policy=AssemblyPolicy())
    assembler.offer("A first complete thought.")
    for _ in range(6):
        chunks = assembler.offer("Six more words go in here.", headroom=2.0)
        if chunks:
            assert chunks[0].words <= AssemblyPolicy().min_words, (
                f"batched to {chunks[0].words} words with 2s of buffer left")
            return
    pytest.fail("nothing was released at all while the buffer was thin")


# --------------------------------------------------------------------------
# end to end, through the real producer
# --------------------------------------------------------------------------
class Slow:
    """A model that writes steadily, as a real one does."""

    def __init__(self, sentences: int = 24, delay: float = 0.01):
        self.sentences, self.delay = sentences, delay

    async def stream_sentences(self, plan, notes=None):
        for i in range(self.sentences):
            await asyncio.sleep(self.delay)
            yield f"Sentence {i} carrying a few words of real content."

    async def top_up(self, plan, spoken_so_far, words_needed):
        return
        yield ""  # pragma: no cover


def _episode(pipeline_value: str, minutes: int = 1) -> GenerationStats:
    from script_generator import plan_episode

    original = pipeline_mod.settings
    pipeline_mod.settings = dataclasses.replace(
        original, streaming_pipeline=pipeline_value)
    try:
        async def main():
            stats = GenerationStats()
            pipe = PodcastPipeline(generator=Slow(), engine=DebugEngine(),
                                   cache=None)
            async for _ in pipe.stream_pcm(plan_episode("q", minutes), stats):
                pass
            return stats

        return asyncio.run(asyncio.wait_for(main(), 20))
    finally:
        pipeline_mod.settings = original


def test_an_episode_reports_its_own_starvation():
    """`stats.starved` is the server's own verdict, and it must survive into
    the record - a starved episode that reports nothing is the silent failure
    this project treats as worse than a crash."""
    stats = _episode("phase6")
    assert "starved" in stats.as_dict()
    assert "min_headroom" in stats.as_dict()


def test_the_first_chunk_is_still_the_first_complete_thought():
    """The invariant the fix must not have bought its safety with.

    Everything above makes chunks ship *sooner*. None of it may make the first
    one ship later, or larger, or behind a word floor.
    """
    stats = _episode("phase6")
    first = stats.marks.chunks[0]
    assert first.sentences == 1, "the opening was batched"
    assert first.words < AssemblyPolicy().min_words or first.words < 20


def test_headroom_never_delays_a_chunk_only_hastens_it():
    """The rule is one-directional by construction: it can add a reason to
    release, never a reason to hold."""
    policy = AssemblyPolicy()
    for headroom in (None, 0.0, 1.0, 2.9, 3.0, 100.0):
        a = SpeechAssembler(policy=policy)
        b = SpeechAssembler(policy=policy)
        a.offer("A first complete thought.")
        b.offer("A first complete thought.")
        with_headroom = a.offer("Some more words follow along here.", headroom)
        without = b.offer("Some more words follow along here.", None)
        assert len(with_headroom) >= len(without), (
            f"headroom={headroom} held text back that would otherwise ship")
