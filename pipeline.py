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
from cache import ScriptCache, build_cache, cache_key, canonical_key, is_shareable, ttl_for
from config import settings
from script_generator import EpisodePlan, ScriptGenerator, count_words
from tts import TTSEngine, build_engine

log = logging.getLogger(__name__)

# How many sentences may sit synthesised-and-waiting. Small on purpose: this is
# the whole memory budget of a 10-minute episode.
#: Sentinel meaning "use the configured cache" - see PodcastPipeline.__init__.
AUTO = "auto"

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
class _Pump:
    """A sentence stream that is already running in the background."""

    queue: asyncio.Queue
    task: asyncio.Task
    #: Items pulled off the queue early by prime(), consumed before it.
    pending: list = field(default_factory=list)
    primed: bool = False

    async def prime(self) -> object:
        """Wait for this stream's first item, without consuming it.

        Used to answer "is there more audio ready to follow?" before committing
        to play something that would otherwise run into silence.
        """
        if not self.primed:
            self.pending.append(await self.queue.get())
            self.primed = True
        return self.pending[0] if self.pending else None

    def ready(self) -> bool:
        """Is there an item available right now, without waiting?"""
        return bool(self.pending) or not self.queue.empty()

    async def next(self) -> object:
        if self.pending:
            return self.pending.pop(0)
        return await self.queue.get()

    async def close(self) -> None:
        self.task.cancel()
        try:
            await self.task
        except BaseException:
            pass


async def _replay(sentences: list[str]) -> AsyncIterator[str]:
    """Feed a cached script back through the normal speaking path."""
    for sentence in sentences:
        yield sentence


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
    #: "hit" | "miss" | "off" - whether this episode reused a shared script.
    cache: str = "off"
    #: Whether a fast-model opener covered the research latency.
    cold_open: bool = False
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
            "cache": self.cache,
            "cold_open": self.cold_open,
        }


class PodcastPipeline:
    def __init__(
        self,
        generator: Optional[ScriptGenerator] = None,
        engine: Optional[TTSEngine] = None,
        cache: ScriptCache | None | str = AUTO,
    ):
        """`cache` takes a store, or AUTO to build the configured one, or None
        to disable caching.

        The explicit AUTO sentinel exists because `cache=None` previously meant
        "build the default", so passing None to switch caching *off* silently
        turned it on. That misread caused two separate test failures before it
        was noticed; a caller saying None now unambiguously gets no cache.
        """
        self.generator = generator or ScriptGenerator()
        self.engine = engine or build_engine()
        self.cache = build_cache() if cache is AUTO else cache

    def _start(self, sentences: AsyncIterator[str]) -> "_Pump":
        """Begin consuming a sentence stream *now*, into a bounded queue.

        Starting is separated from speaking so two model calls can be in flight
        at once: the researched main script begins the moment the request
        arrives, while the cold open is what actually reaches the speakers
        first. The queue depth caps memory at a few seconds of audio however
        long the episode is.
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_DEPTH)

        async def produce() -> None:
            try:
                async for sentence in sentences:
                    await queue.put(sentence)
            except Exception as exc:  # surfaced to the consumer, never swallowed
                await queue.put(exc)
            finally:
                await queue.put(None)

        return _Pump(queue, asyncio.create_task(produce()))

    async def _speak(
        self,
        pump: "_Pump",
        pace: PaceController,
        stats: GenerationStats,
        fatal: bool = True,
    ) -> AsyncIterator[bytes]:
        """Synthesise an already-running sentence stream inside the time budget.

        `fatal=False` means a failure in this stream is logged and skipped
        rather than ending the episode - used for the optional cold open.
        """
        try:
            while True:
                item = await pump.next()
                if item is None:
                    break
                if isinstance(item, Exception):
                    if fatal:
                        raise item
                    log.warning("optional stream failed; continuing", exc_info=item)
                    break
                async for chunk in self._speak_one(item, pace, stats):
                    yield chunk
                if stats.truncated:
                    break
        finally:
            await pump.close()

    async def _speak_one(
        self, sentence: str, pace: PaceController, stats: GenerationStats
    ) -> AsyncIterator[bytes]:
        """Synthesise one sentence, or stop the episode if it no longer fits."""
        gap = silence(SENTENCE_GAP, self.engine.sample_rate)
        words = count_words(sentence)
        # Keep the controller's view of "words left" honest: the model rarely
        # hits the budget exactly, so grow the total when it overshoots rather
        # than sprinting through the remainder.
        pace.total_words = max(pace.total_words, pace.spoken_words + words)

        wpm = pace.next_wpm()
        # Will this sentence fit in the time that is left? Speeding up is
        # already clamped to a rate a listener accepts, so an over-long script
        # has to be cut rather than gabbled. Cutting at a sentence boundary is
        # why the pipeline works in sentences.
        estimated = words / (wpm / 60.0) + SENTENCE_GAP
        if estimated > pace.remaining_seconds + OVERRUN_GRACE:
            stats.truncated = True
            return

        pcm = await self.engine.synth(sentence, wpm)
        if not pcm:
            return
        pace.observe(len(pcm) + len(gap), words)
        stats.sentences += 1
        stats.words += words
        stats.script.append(sentence)
        yield pcm
        yield gap

    async def _run_cold_open(
        self,
        opener: "_Pump",
        body: "_Pump",
        pace: PaceController,
        stats: GenerationStats,
    ) -> AsyncIterator[bytes]:
        """Speak framing sentences for exactly as long as the body needs.

        The cold open exists to cover research latency. A single fixed sentence
        only covers a few seconds, so if research runs longer the listener gets
        the worst possible outcome - speech, dead air, speech - which sounds
        broken in a way that simply waiting does not.

        So the opener is written as several short framing sentences and released
        one at a time, and before each one we ask whether the main script has
        arrived yet. The moment it has, the opener stops mid-flow and the real
        briefing takes over seamlessly. Slow research gets more introduction;
        fast research gets almost none. Unused sentences are discarded.

        Two guards: nothing is spoken if the body has already failed (there is
        no point introducing an episode that will not come), and the opener will
        not run past `max_seconds` of stalling, after which a gap is preferable
        to endless preamble.
        """
        spoken_any = False
        # One task watches for the body's first sentence; its completion is the
        # signal to stop introducing and start briefing.
        waiting = asyncio.create_task(body.prime())
        try:
            # Give the body a moment to fail before committing to an opener.
            # Credential and model errors surface almost instantly, and playing
            # an introduction to an episode that then dies is worse than a
            # clean error - it is the silent-failure bug all over again.
            await asyncio.wait({waiting}, timeout=settings.cold_open_grace)

            if waiting.done():
                try:
                    first = waiting.result()
                except Exception:  # pragma: no cover - defensive
                    first = None
                if first is None or isinstance(first, Exception):
                    # The script failed or came back empty. Say nothing.
                    return

            # At least one framing sentence always plays, because the main
            # script is written on the understanding that the episode has
            # already been opened - without it the briefing starts mid-thought.
            # Beyond the first, more are spoken only while the script is still
            # being written.
            spoken_count = 0
            deadline = pace.elapsed + settings.cold_open_max_seconds
            while (spoken_count == 0 or not waiting.done()) and pace.elapsed < deadline:
                item = await opener.next()
                if item is None or isinstance(item, Exception):
                    if isinstance(item, Exception):
                        log.warning("cold open failed; continuing", exc_info=item)
                    break

                async for chunk in self._speak_one(item, pace, stats):
                    spoken_any = True
                    yield chunk
                spoken_count += 1
                if stats.truncated:
                    break

            stats.cold_open = spoken_any
        finally:
            await opener.close()
            if not waiting.done():
                # Leave the body stream intact; only stop watching it here.
                await asyncio.wait({waiting})

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

        # --- Cache: has anyone already asked for this? --------------------
        shareable = is_shareable(plan.query)
        key = ""
        if self.cache and shareable:
            canonical = None
            if settings.cache_semantic_key:
                canonical = await canonical_key(plan.query, self.generator.client)
            key = cache_key(plan.query, plan.minutes, canonical)
        if self.cache and shareable:
            cached = self.cache.get(key)
            if cached:
                stats.cache = "hit"
                log.info("cache hit for %r (%d min)", plan.query, plan.minutes)
                # Replaying the same sentences through the same controller
                # reproduces the episode exactly - and costs zero API tokens.
                async for chunk in self._speak(self._start(_replay(cached)), pace, stats):
                    yield chunk
                async for chunk in self._finish(pace, stats):
                    yield chunk
                return
        stats.cache = "miss" if self.cache else "off"

        # --- Generate ------------------------------------------------------
        # Start the researched call FIRST so web search is already running
        # while the cold open is being written and spoken.
        body = self._start(self.generator.stream_sentences(plan))
        cold_open = getattr(self.generator, "cold_open", None)
        if settings.enable_cold_open and plan.reserved_words and cold_open:
            opener = self._start(cold_open(plan))

            async for chunk in self._run_cold_open(opener, body, pace, stats):
                yield chunk

        async for chunk in self._speak(body, pace, stats):
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
            extra = self._start(self.generator.top_up(plan, " ".join(stats.script), words_needed))
            async for chunk in self._speak(extra, pace, stats):
                yield chunk
            if pace.spoken_words == before:
                break  # the top-up produced nothing; stop asking

        if self.cache and shareable and stats.script:
            ttl = ttl_for(plan.query)
            self.cache.put(key, stats.script, ttl, plan.query)
            log.info("cached %d sentences for %r (ttl %ds)", len(stats.script), plan.query, ttl)

        async for chunk in self._finish(pace, stats):
            yield chunk

    async def _finish(
        self, pace: PaceController, stats: GenerationStats
    ) -> AsyncIterator[bytes]:
        """Close any residual gap with room tone.

        A second or two of quiet at the end reads as the episode finishing; a
        hard cut reads as a bug.
        """
        # Never pad an episode that has no speech in it. Doing so manufactures
        # a few seconds of silence that looks like a valid episode to every
        # layer above, which is how an empty script reached listeners as
        # "it generated something but I hear nothing".
        if stats.sentences == 0:
            stats.audio_seconds = pace.elapsed
            return
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
