"""The generate-while-you-listen pipeline.

    Claude tokens -> sentences -> TTS -> raw PCM -> HTTP response -> speakers

Nothing is written to disk and nothing is encoded. The bytes leaving the TTS
engine are the bytes the browser plays.

Hitting the requested duration takes three mechanisms, because no single one is
enough on its own:

* **Budget** - the script is commissioned at the right word count up front.
* **Pacing** - the speaking rate is re-planned before every sentence, so small
  misses are absorbed invisibly (clamped to a range a listener accepts).
* **Trim / top-up** - a script that is too long is cut at a sentence boundary;
  one that is too short is extended with a second, smaller Claude request.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from audio_utils import PaceController, silence, streaming_wav_header
from config import settings
from script_generator import EpisodePlan, ScriptGenerator, count_words
from tts import TTSEngine, build_engine

log = logging.getLogger(__name__)

# How many sentences may sit synthesised-and-waiting. Small on purpose: this is
# the whole memory budget of a 10-minute episode.
QUEUE_DEPTH = 4
# Silence inserted between sentences so the delivery does not sound rushed.
SENTENCE_GAP = 0.12
# A sentence may overshoot the remaining time by this much before it is cut.
OVERRUN_GRACE = 0.6
# Dead air worth going back to Claude for.
TOPUP_THRESHOLD = 4.0
# Cap the number of extra requests, so a model that keeps under-writing cannot
# turn one episode into an unbounded fan-out of API calls.
MAX_TOPUPS = 2
# Residual gap after the last top-up, closed with room tone rather than a cut.
MAX_TAIL_SILENCE = 6.0


@dataclass
class GenerationStats:
    """Everything the UI needs to show, and the tests need to assert on."""

    plan_seconds: int = 0
    audio_seconds: float = 0.0
    words: int = 0
    sentences: int = 0
    engine: str = ""
    sample_rate: int = settings.sample_rate
    truncated: bool = False
    topups: int = 0
    script: list[str] = field(default_factory=list)

    @property
    def drift(self) -> float:
        return self.audio_seconds - self.plan_seconds

    def as_dict(self) -> dict:
        return {
            "requested_seconds": self.plan_seconds,
            "audio_seconds": round(self.audio_seconds, 2),
            "drift_seconds": round(self.drift, 2),
            "words": self.words,
            "sentences": self.sentences,
            "engine": self.engine,
            "truncated": self.truncated,
            "topups": self.topups,
        }


class PodcastPipeline:
    def __init__(
        self,
        generator: Optional[ScriptGenerator] = None,
        engine: Optional[TTSEngine] = None,
    ):
        self.generator = generator or ScriptGenerator()
        self.engine = engine or build_engine()

    async def _speak(
        self,
        sentences: AsyncIterator[str],
        pace: PaceController,
        stats: GenerationStats,
    ) -> AsyncIterator[bytes]:
        """Synthesise a stream of sentences, staying inside the time budget.

        Claude and the TTS engine are decoupled by a bounded queue so the slower
        of the two never blocks the other: the model writes ahead while the
        current sentence is still being spoken, and the queue depth caps memory
        at a few seconds of audio however long the episode is.
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_DEPTH)
        gap = silence(SENTENCE_GAP, self.engine.sample_rate)

        async def produce() -> None:
            try:
                async for sentence in sentences:
                    await queue.put(sentence)
            except Exception as exc:  # surfaced to the consumer, never swallowed
                await queue.put(exc)
            finally:
                await queue.put(None)

        producer = asyncio.create_task(produce())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item

                sentence: str = item
                words = count_words(sentence)
                # Keep the controller's view of "words left" honest: the model
                # rarely hits the budget exactly, so grow the total when it
                # overshoots rather than sprinting through the remainder.
                pace.total_words = max(pace.total_words, pace.spoken_words + words)

                wpm = pace.next_wpm()
                # Will this sentence fit in the time that is left? Speeding up
                # is already clamped to a rate a listener accepts, so an
                # over-long script has to be cut rather than gabbled. Cutting at
                # a sentence boundary is why the pipeline works in sentences.
                estimated = words / (wpm / 60.0) + SENTENCE_GAP
                if estimated > pace.remaining_seconds + OVERRUN_GRACE:
                    stats.truncated = True
                    break

                pcm = await self.engine.synth(sentence, wpm)
                if not pcm:
                    continue
                pace.observe(len(pcm) + len(gap), words)
                stats.sentences += 1
                stats.words += words
                stats.script.append(sentence)
                yield pcm
                yield gap
        finally:
            producer.cancel()
            try:
                await producer
            except BaseException:
                pass

    async def stream_pcm(
        self, plan: EpisodePlan, stats: Optional[GenerationStats] = None
    ) -> AsyncIterator[bytes]:
        """Yield raw PCM for the whole episode, starting as soon as possible."""
        stats = stats if stats is not None else GenerationStats()
        stats.plan_seconds = plan.target_seconds
        stats.engine = self.engine.name
        stats.sample_rate = self.engine.sample_rate

        pace = PaceController(
            target_seconds=float(plan.target_seconds),
            total_words=plan.word_budget,
            sample_rate=self.engine.sample_rate,
        )

        async for chunk in self._speak(self.generator.stream_sentences(plan), pace, stats):
            yield chunk

        # The model under-wrote. Rather than pad minutes of silence, buy more
        # script: a top-up request is small, cheap and arrives while the
        # listener is still hearing the material already generated.
        while (
            not stats.truncated
            and pace.remaining_seconds > TOPUP_THRESHOLD
            and stats.topups < MAX_TOPUPS
        ):
            stats.topups += 1
            words_needed = int(pace.remaining_seconds / 60.0 * settings.target_wpm)
            log.info("topping up %d words for %.1fs of dead air", words_needed, pace.remaining_seconds)
            before = pace.spoken_words
            extra = self.generator.top_up(plan, " ".join(stats.script), words_needed)
            async for chunk in self._speak(extra, pace, stats):
                yield chunk
            if pace.spoken_words == before:
                break  # the top-up produced nothing; stop asking

        # Close any residual gap with room tone. A second or two of quiet at the
        # end reads as the episode finishing; a hard cut reads as a bug.
        shortfall = min(pace.remaining_seconds, MAX_TAIL_SILENCE)
        if shortfall > 0.05:
            pad = silence(shortfall, self.engine.sample_rate)
            pace.observe(len(pad), 0)
            yield pad

        stats.audio_seconds = pace.elapsed

    async def stream_wav(
        self, plan: EpisodePlan, stats: Optional[GenerationStats] = None
    ) -> AsyncIterator[bytes]:
        """Same stream, prefixed with a live WAV header for <audio> playback."""
        yield streaming_wav_header(sample_rate=self.engine.sample_rate)
        async for chunk in self.stream_pcm(plan, stats):
            yield chunk
