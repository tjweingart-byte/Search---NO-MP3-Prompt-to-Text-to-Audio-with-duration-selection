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

import time

from audio_utils import PaceController, pcm_duration, silence, streaming_wav_header
from cache import ScriptCache, build_cache, cache_key, canonical_key, is_shareable, ttl_for
from config import settings
from script_generator import EpisodePlan, ScriptGenerator, ScriptNotes, count_words
from tts import TTSEngine, build_engine

log = logging.getLogger(__name__)

# How many sentences may sit synthesised-and-waiting. Small on purpose: this is
# the whole memory budget of a 10-minute episode.
#: Sentinel meaning "use the configured cache" - see PodcastPipeline.__init__.
AUTO = "auto"


class NotCached(Exception):
    """A replay-only request found nothing in the cache.

    Explore is built on the promise that it never spends a model call. If that
    promise lived only in the interface it would be one refactor away from
    being broken silently and expensively, so the pipeline refuses instead.
    """

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
# Keep this much unspoken opener in hand: comfortably longer than one API call
# (~1s) so a fill never interrupts speech, but low enough that we are not
# re-fetching after every sentence.
OPENER_BUFFER_TARGET = 6.0

# How far ahead of the listener the opener keeps the stream. Enough that a slow
# script cannot cause silence; small enough that a fast one wastes no preamble.
OPENER_HEADROOM_TARGET = 8.0

# Hard ceiling on opener fetches, so a wedged script call cannot fan out into
# unbounded API calls. The real limit is COLD_OPEN_MAX_SECONDS.
MAX_OPENER_FILLS = 12
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

    async def peek(self) -> None:
        """Wait until an item is available, without consuming it."""
        if self.pending:
            return
        item = await self.queue.get()
        self.pending.append(item)

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


def _buffered_seconds(sentences: list[str]) -> float:
    """Roughly how long the unspoken opener sentences would take to say."""
    words = sum(count_words(s) for s in sentences)
    return words / settings.target_wpm * 60.0


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
    voice: str = ""
    sample_rate: int = settings.sample_rate
    truncated: bool = False
    topups: int = 0
    #: "hit" | "miss" | "off" - whether this episode reused a shared script.
    cache: str = "off"
    #: Whether a fast-model opener covered the research latency.
    cold_open: bool = False
    #: When generation began, for audio-produced vs wall-clock comparisons.
    started_at: float = field(default_factory=time.perf_counter)
    #: Total seconds spent inside the speech engine.
    synth_seconds: float = 0.0
    #: Smallest margin between audio produced and wall clock. Negative means
    #: the listener heard silence.
    min_headroom: float = 999.0
    #: True if the stream ever fell behind realtime.
    starved: bool = False
    #: Wall clock at which the first audio left the pipeline.
    first_audio_at: float = 0.0
    #: How many times the opener was topped up while waiting for the script.
    opener_fills: int = 0
    script: list[str] = field(default_factory=list)
    #: The thread the episode left open, phrased as the follow-up a listener
    #: would ask for. Drives the one-tap suggestion in Go Deeper; empty when
    #: the model named none.
    thread: str = ""

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
            "voice": self.voice,
            "truncated": self.truncated,
            "topups": self.topups,
            "cache": self.cache,
            "cold_open": self.cold_open,
            "synth_seconds": round(self.synth_seconds, 2),
            "min_headroom": round(self.min_headroom, 1) if self.min_headroom < 999 else None,
            "starved": self.starved,
            "first_audio_at": round(self.first_audio_at, 2),
            "opener_fills": self.opener_fills,
            "thread": self.thread,
        }


class PodcastPipeline:
    def __init__(
        self,
        generator: Optional[ScriptGenerator] = None,
        engine: Optional[TTSEngine] = None,
        cache: ScriptCache | None | str = AUTO,
        voice: Optional[str] = None,
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
        #: Passed to the engine on every sentence. The script is unaffected by
        #: it, which is why the script cache deliberately ignores voice.
        self.voice = voice

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

        started = time.perf_counter()
        pcm = await self.engine.synth(sentence, wpm, self.voice)
        synth_seconds = time.perf_counter() - started
        if not pcm:
            return

        # Compare audio produced against wall clock consumed. A listener hears
        # silence exactly when the second overtakes the first, so this is the
        # number that matters, and it is logged for every sentence.
        audio_seconds = pcm_duration(len(pcm), self.engine.sample_rate)
        stats.synth_seconds += synth_seconds
        elapsed_wall = time.perf_counter() - stats.started_at
        headroom = pace.elapsed + audio_seconds - elapsed_wall
        if headroom < stats.min_headroom:
            stats.min_headroom = headroom
        log.debug(
            "sentence %d: %.2fs audio in %.2fs (%.0fx realtime), headroom %.1fs",
            stats.sentences + 1, audio_seconds, synth_seconds,
            audio_seconds / synth_seconds if synth_seconds else 0, headroom,
        )
        if headroom < 0 and not stats.starved:
            stats.starved = True
            log.warning(
                "STARVED after %.1fs: only %.1fs of audio made in %.1fs of wall clock. "
                "The listener hears silence here. Synthesis so far: %.1fs.",
                elapsed_wall, pace.elapsed + audio_seconds, elapsed_wall, stats.synth_seconds,
            )
        pace.observe(len(pcm) + len(gap), words)
        stats.sentences += 1
        stats.words += words
        stats.script.append(sentence)
        if not stats.first_audio_at:
            stats.first_audio_at = time.perf_counter() - stats.started_at
            log.info("first audio ready after %.2fs", stats.first_audio_at)
        yield pcm
        yield gap

    async def _run_cold_open(
        self,
        plan: EpisodePlan,
        cold_open,
        opener: "_Pump",
        body: "_Pump",
        pace: PaceController,
        stats: GenerationStats,
    ) -> AsyncIterator[bytes]:
        """Keep talking until the researched script arrives.

        The opener exists to cover research latency. Its old failure was running
        out: a fixed number of refills covered about twenty seconds, and a
        researched call can take longer than that, so the listener heard the
        difference as dead air.

        Two changes make that structurally impossible up to a hard ceiling:

        * **The budget is time, not a refill count.** More material is fetched
          for as long as the script is still coming.
        * **Refills are started before the current batch runs out**, not after.
          Fetching only once the last sentence has been spoken leaves a hole the
          width of an API call, every time.

        Past `COLD_OPEN_MAX_SECONDS` it stops regardless: at some point a gap is
        better than talking indefinitely about nothing.
        """
        spoken_any = False
        fills = 0
        buffer: list[str] = []
        exhausted = False
        waiting = asyncio.create_task(body.prime())
        pumps = [opener]
        try:
            # Let the script fail before committing to an opener: playing an
            # introduction to an episode that then dies is the silent-failure
            # bug all over again.
            await asyncio.wait({waiting}, timeout=settings.cold_open_grace)
            if waiting.done():
                try:
                    first = waiting.result()
                except Exception:  # pragma: no cover - defensive
                    first = None
                if first is None or isinstance(first, Exception):
                    return

            deadline = time.perf_counter() + settings.cold_open_max_seconds
            spoken_count = 0

            # The control law: keep the listener a fixed distance ahead of
            # themselves, and no further.
            #
            # Audio is synthesised many times faster than it is heard, so a
            # wall-clock budget lets the opener run away - an earlier version
            # produced 75 seconds of preamble to cover a 30 second wait.
            # Instead, speak only while the audio produced is less than
            # OPENER_HEADROOM_TARGET seconds ahead of the wall clock. That is
            # exactly the buffer needed to never fall silent, and it paces the
            # opener to roughly real time, so a fast script wastes almost none
            # of it.
            while True:
                if waiting.done() and spoken_count > 0:
                    break
                if time.perf_counter() >= deadline:
                    log.warning("opener hit its %.0fs ceiling; the script is very slow",
                                settings.cold_open_max_seconds)
                    break

                wall = time.perf_counter() - stats.started_at
                headroom = pace.elapsed - wall

                if spoken_count > 0 and headroom >= OPENER_HEADROOM_TARGET:
                    # Comfortably ahead: stop talking and wait for the script.
                    await asyncio.wait({waiting}, timeout=0.25)
                    continue

                while opener.ready():
                    item = await opener.next()
                    if item is None:
                        exhausted = True
                        break
                    if isinstance(item, Exception):
                        log.warning("cold open failed; continuing", exc_info=item)
                        exhausted = True
                        break
                    buffer.append(item)

                # Top up on seconds of speech held, and start the next fill
                # before the buffer drains so the API call overlaps with speech.
                held = _buffered_seconds(buffer)
                if (
                    held < OPENER_BUFFER_TARGET
                    and not waiting.done()
                    and fills < MAX_OPENER_FILLS
                    and time.perf_counter() < deadline
                    and (exhausted or not opener.ready())
                ):
                    fills += 1
                    log.info(
                        "opener fill %d (%.1fs held, %.1fs spoken, %.1fs headroom)",
                        fills, held, pace.elapsed, headroom,
                    )
                    opener = self._start(cold_open(plan))
                    pumps.append(opener)
                    exhausted = False

                if not buffer:
                    if waiting.done():
                        break
                    await asyncio.wait(
                        {waiting, asyncio.ensure_future(opener.peek())},
                        timeout=0.5,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    continue

                sentence = buffer.pop(0)
                async for chunk in self._speak_one(sentence, pace, stats):
                    spoken_any = True
                    yield chunk
                spoken_count += 1
                if stats.truncated:
                    break

            stats.cold_open = spoken_any
            stats.opener_fills = fills
        finally:
            for pump in pumps:
                await pump.close()
            if not waiting.done():
                await asyncio.wait({waiting})

    async def _cache_key(self, plan: EpisodePlan) -> str:
        """Where this episode lives in the shared cache. "" when caching is off."""
        if not self.cache:
            return ""
        canonical = None
        if settings.cache_semantic_key:
            canonical = await canonical_key(plan.query, self.generator.client)
        return cache_key(plan.query, plan.minutes, canonical, plan.context, plan.search)

    async def thread_for(self, plan: EpisodePlan) -> str:
        """The go-deeper thread of an episode that has already been generated.

        Read out of the cache, so it costs nothing and needs no second call.
        The thread is only known once the script has been written, which is
        after the audio response headers have gone out - hence a separate
        lookup rather than a header on /api/audio.
        """
        if not self.cache or not is_shareable(plan.query):
            return ""
        return self.cache.thread(await self._cache_key(plan))

    async def stream_pcm(
        self, plan: EpisodePlan, stats: Optional[GenerationStats] = None
    ) -> AsyncIterator[bytes]:
        """Yield raw PCM for the whole episode, starting as soon as possible."""
        stats = stats if stats is not None else GenerationStats()
        stats.plan_seconds = plan.target_seconds
        stats.engine = self.engine.name
        stats.voice = self.voice or ""
        stats.sample_rate = self.engine.sample_rate

        pace = PaceController(
            target_seconds=float(plan.target_seconds),
            total_words=plan.word_budget,
            sample_rate=self.engine.sample_rate,
        )

        # --- Cache: has anyone already asked for this? --------------------
        shareable = is_shareable(plan.query)
        key = await self._cache_key(plan) if shareable else ""
        if self.cache and shareable:
            cached = self.cache.get(key)
            if cached:
                stats.cache = "hit"
                stats.thread = self.cache.thread(key)
                log.info("cache hit for %r (%d min)", plan.query, plan.minutes)
                # Replaying the same sentences through the same controller
                # reproduces the episode exactly - and costs zero API tokens.
                async for chunk in self._speak(self._start(_replay(cached)), pace, stats):
                    yield chunk
                async for chunk in self._finish(pace, stats):
                    yield chunk
                return
        if plan.cached_only:
            # Nothing to replay, and generating is exactly what this request
            # promised not to do.
            raise NotCached(
                "That episode is no longer in the cache. Explore only replays "
                "episodes other listeners have already generated."
            )

        stats.cache = "miss" if self.cache else "off"

        # --- Generate ------------------------------------------------------
        # Start the researched call FIRST so web search is already running
        # while the cold open is being written and spoken.
        notes = ScriptNotes()
        body = self._start(self.generator.stream_sentences(plan, notes))
        cold_open = getattr(self.generator, "cold_open", None)
        if settings.enable_cold_open and plan.reserved_words and cold_open:
            opener = self._start(cold_open(plan))
            async for chunk in self._run_cold_open(
                plan, cold_open, opener, body, pace, stats
            ):
                yield chunk

        async for chunk in self._speak(body, pace, stats):
            yield chunk

        # The model under-wrote. Rather than pad minutes of silence, buy more
        # script: a top-up request is small, cheap and arrives while the
        # listener is still hearing the material already generated.
        while (
            settings.allow_topups
            and not stats.truncated
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

        stats.thread = notes.thread

        if self.cache and shareable and stats.script:
            ttl = ttl_for(plan.query)
            self.cache.put(key, stats.script, ttl, plan.query, stats.thread, plan.minutes)
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
