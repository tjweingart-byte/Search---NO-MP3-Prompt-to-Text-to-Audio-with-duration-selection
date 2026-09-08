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
import dataclasses
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

import time

from audio_utils import PaceController, pcm_duration, silence, streaming_wav_header
from cache import (ScriptCache, build_cache, cache_key, canonical_key, is_shareable,
                   key_bucket, ttl_for)
from episode_marks import EpisodeMarks, TimedClient
from config import settings
from script_buffer import ASSEMBLER_TICK, ScriptBuffer
from script_generator import EpisodePlan, ScriptGenerator, ScriptNotes, count_words
from speech_assembly import (AssembledChunk, AssemblyPolicy,
                             SpeechAssembler, fit_to_budget)
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
    #: "exact" | "near" | "" - *how* a hit was found. A near hit replayed an
    #: episode written for a differently-worded question, which is worth being
    #: able to see: it is the one kind of hit that can be wrong.
    match: str = ""
    #: Cosine of a near hit, 0.0 otherwise.
    match_score: float = 0.0
    answered_first: bool = False
    handover_seconds: float = 0.0
    cache: str = "off"
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
    #: Named instants and per-synthesis records for this episode. Written to,
    #: never read back: instrumentation must not be able to change what a
    #: listener hears.
    marks: EpisodeMarks = field(default_factory=EpisodeMarks)
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
            "answered_first": self.answered_first,
            "handover_seconds": round(self.handover_seconds, 2),
            "cache": self.cache,
            "match": self.match,
            "match_score": round(self.match_score, 3),
            "synth_seconds": round(self.synth_seconds, 2),
            "min_headroom": round(self.min_headroom, 1) if self.min_headroom < 999 else None,
            "starved": self.starved,
            "first_audio_at": round(self.first_audio_at, 2),
            "thread": self.thread,
        }


class PodcastPipeline:
    def __init__(
        self,
        generator: Optional[ScriptGenerator] = None,
        engine: Optional[TTSEngine] = None,
        cache: ScriptCache | None | str = AUTO,
        voice: Optional[str] = None,
        cache_writes: bool = True,
    ):
        """`cache` takes a store, or AUTO to build the configured one, or None
        to disable caching.

        `cache_writes=False` reads the cache but never adds to it. That is not
        a tuning knob - it is what demo mode needs. Without credentials the
        writer is a canned sample script that describes how the audio pipeline
        works, and caching it stores that text under whatever the listener
        actually asked, where Explore and every other listener will later be
        served it as a real episode. Reads must stay on, because replaying is
        the one thing that needs no credentials at all.

        The explicit AUTO sentinel exists because `cache=None` previously meant
        "build the default", so passing None to switch caching *off* silently
        turned it on. That misread caused two separate test failures before it
        was noticed; a caller saying None now unambiguously gets no cache.
        """
        self.generator = generator or ScriptGenerator()
        self.engine = engine or build_engine()
        self.cache = build_cache() if cache is AUTO else cache
        self.cache_writes = cache_writes
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
            except asyncio.CancelledError:
                # `close()` cancelled us. The sentinel is deliberately NOT sent
                # here, and this must not be a `finally`: on the close path
                # nobody is draining, so a blocking put on a full queue would
                # suspend, swallow the cancellation that `close()` just
                # delivered, and `close()` would wait for a task that can never
                # finish. Re-raising ends the task, which is what `close()`
                # is waiting for. Nothing is lost - the sentinel only tells a
                # consumer the stream ended, and there is no consumer left.
                raise
            except Exception as exc:  # surfaced to the consumer, never swallowed
                await queue.put(exc)
            else:
                await queue.put(None)

        return _Pump(queue, asyncio.create_task(produce()))

    def _start_phase6(
        self,
        sentences: AsyncIterator[str],
        policy: Optional[AssemblyPolicy] = None,
    ) -> "_Pump":
        """`_start`, with the reader decoupled and the sentences assembled.

        **Unreachable in this build.** Nothing calls it; `STREAMING_PIPELINE`
        does not select it. It exists so the path can be measured against
        `_start` on fakes before anything is allowed to route to it.

            Claude stream
              -> reader          its own task, never touches this queue
              -> ScriptBuffer    bounded by CHARACTERS, ~20 episodes
              -> SpeechAssembler whole sentences -> speech-sized chunks
              -> this queue      bounded at QUEUE_DEPTH, as `_start`'s is
              -> _speak_phase6

        `_start` puts sentences straight onto the bounded queue, so a slow
        voice fills it and stops the reader - the Phase 6 4090 run reported a
        12.5s Claude stream as 66.7s, 64.3s of it that backpressure. Here the
        queue holds *chunks* and sits below the buffer, so filling it suspends
        the assembler and never the reader.

        Returns the same `_Pump` the rest of the pipeline expects. Its items
        are `AssembledChunk` rather than `str`, which is why the companion
        `_speak_phase6` exists: `_speak_one` synthesises one item per call and
        appends it to `stats.script`, so handing it a chunk would both coarsen
        the duration check and put a multi-sentence blob in the cache.
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_DEPTH)


        async def produce() -> None:
            """Reader, buffer and assembler, with the reader owned explicitly.

            `script_buffer.assemble_chunks` composes the same three pieces and
            is the tested primitive, but it owns its reader inside an async
            generator - and cancelling a task that is iterating a generator
            which owns another task does not unwind reliably. Here the reader
            is a task this coroutine holds and cancels itself, and every await
            is a `sleep` or a `wait_for`, both of which always cancel.
            """
            buffer = ScriptBuffer()
            assembler = SpeechAssembler(policy=policy or AssemblyPolicy())

            async def read() -> None:
                try:
                    async for sentence in sentences:
                        if sentence and sentence.strip():
                            await buffer.put(sentence)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    buffer.fail(exc)
                finally:
                    buffer.close()

            reader = asyncio.create_task(read())
            #: The pending `buffer.get()`, held across ticks and owned here.
            #:
            #: This used to be `await asyncio.wait_for(buffer.get(), tick)`,
            #: and that was the teardown defect: `wait_for` cancels its inner
            #: task on timeout, and when the inner task completes in the same
            #: turn it returns the result and *drops* the outer cancellation
            #: it was supposed to propagate. The producer then carried on
            #: round its loop having eaten the cancel `close()` sent, so it
            #: could never be shut down. `asyncio.wait` never cancels what it
            #: waits on, so an outer cancel passes straight through.
            waiting: Optional[asyncio.Task] = None
            try:
                while True:
                    if waiting is None:
                        waiting = asyncio.ensure_future(buffer.get())
                    done, _ = await asyncio.wait({waiting},
                                                 timeout=ASSEMBLER_TICK)
                    if not done:
                        # Nothing new: run the assembler's timer and headroom
                        # rules so text is never held indefinitely.
                        for chunk in assembler.due():
                            await queue.put(chunk)
                        continue
                    sentence, waiting = waiting.result(), None
                    if sentence is None:
                        for chunk in assembler.flush():
                            await queue.put(chunk)
                        break
                    for chunk in assembler.offer(sentence):
                        await queue.put(chunk)
                await queue.put(None)
            except asyncio.CancelledError:
                # Closed early. No sentinel: nobody is draining, and sending
                # one would only be another chance to block.
                raise
            except Exception as exc:  # surfaced to the consumer, never swallowed
                await queue.put(exc)
            finally:
                # Everything this coroutine started, it ends. The reader owns
                # nothing else, and the pending get is cancelled here rather
                # than left for the loop to finalise.
                if waiting is not None:
                    waiting.cancel()
                reader.cancel()

        return _Pump(queue, asyncio.create_task(produce()))


    async def _speak_phase6(
        self,
        pump: "_Pump",
        pace: PaceController,
        stats: GenerationStats,
        fatal: bool = True,
    ) -> AsyncIterator[bytes]:
        """`_speak` for a pump of assembled chunks. Unreachable in this build.

        Deliberately a copy of `_speak`'s loop rather than a refactor of it:
        changing `_speak` would change the shipped path, which this step is
        not allowed to do. The two converge when Phase 6 is selectable.
        """
        try:
            while True:
                item = await pump.next()
                if item is None:
                    stats.marks.mark("claude_complete")
                    # Waiting work at the moment Claude finished. Greater than
                    # zero means synthesis was behind and the model finished
                    # anyway, which is the decoupling seen rather than claimed.
                    stats.marks.backlog_at_claude_complete = pump.queue.qsize()
                    break
                if isinstance(item, Exception):
                    if fatal:
                        raise item
                    log.warning("optional stream failed; continuing", exc_info=item)
                    break
                stats.marks.mark("first_sentence")
                async for chunk in self._speak_chunk(item, pace, stats):
                    yield chunk
                if stats.truncated:
                    break
        finally:
            stats.marks.mark("speaking_complete")
            if stats.marks.backlog_at_claude_complete is None:
                # Truncation stops the loop before the sentinel arrives, which
                # is most episodes. The queue depth at the moment speaking
                # ended answers the same question: was synthesis behind?
                stats.marks.backlog_at_claude_complete = pump.queue.qsize()
            # `_Pump.close()`, the same as `_speak`: it cancels the producer
            # *and* awaits it, so nothing this method started outlives it.
            await pump.close()

    async def _speak_chunk(
        self, chunk: AssembledChunk, pace: PaceController, stats: GenerationStats
    ) -> AsyncIterator[bytes]:
        """Synthesise one assembled chunk, cut to what still fits.

        `_speak_one` asks "does this sentence fit" once. A chunk is several
        sentences in one synthesis call, so the same question is asked for each
        of them first, by `fit_to_budget`, and the chunk is cut at the last
        boundary that fits. Only the final spoken sentence may cross the
        budget, and by at most `OVERRUN_GRACE` - the same bound `_speak_one`
        allows, rather than that plus a whole chunk.

        Accounting is per sentence even though synthesis is per chunk:
        `stats.script` stays a list of sentences, because the cache stores it
        and `_replay` feeds it back through the speaking path.
        """
        gap = silence(SENTENCE_GAP, self.engine.sample_rate)
        # Keep the controller's view of "words left" honest, as `_speak_one`
        # does: the model rarely hits the budget exactly.
        pace.total_words = max(pace.total_words, pace.spoken_words + chunk.words)

        wpm = pace.next_wpm()
        fit = fit_to_budget(chunk.parts, pace.remaining_seconds, wpm,
                            SENTENCE_GAP, OVERRUN_GRACE)
        if fit.truncated:
            stats.truncated = True
        if not fit.spoken:
            return

        tts_start = stats.marks.mark("first_tts_start")
        started = time.perf_counter()
        pcm = await self.engine.synth(fit.text, wpm, self.voice)
        synth_seconds = time.perf_counter() - started
        tts_done = stats.marks.mark("first_tts_complete")
        if not pcm:
            return

        audio_seconds = pcm_duration(len(pcm), self.engine.sample_rate)
        stats.marks.add_chunk(fit.text, len(fit.spoken),
                              tts_done - synth_seconds, tts_done, audio_seconds)
        stats.synth_seconds += synth_seconds
        elapsed_wall = time.perf_counter() - stats.started_at
        headroom = pace.elapsed + audio_seconds - elapsed_wall
        if headroom < stats.min_headroom:
            stats.min_headroom = headroom
        log.debug(
            "chunk %d: %d sentence(s), %.2fs audio in %.2fs (%.0fx realtime), "
            "headroom %.1fs", chunk.index, len(fit.spoken), audio_seconds,
            synth_seconds, audio_seconds / synth_seconds if synth_seconds else 0,
            headroom,
        )
        if headroom < 0 and not stats.starved:
            stats.starved = True
            log.warning(
                "STARVED after %.1fs: only %.1fs of audio made in %.1fs of wall clock. "
                "The listener hears silence here. Synthesis so far: %.1fs.",
                elapsed_wall, pace.elapsed + audio_seconds, elapsed_wall, stats.synth_seconds,
            )
        pace.observe(len(pcm) + len(gap), fit.words)
        stats.sentences += len(fit.spoken)
        stats.words += fit.words
        stats.script.extend(fit.spoken)
        if not stats.first_audio_at:
            stats.first_audio_at = time.perf_counter() - stats.started_at
            log.info("first audio ready after %.2fs", stats.first_audio_at)
        yield pcm
        yield gap

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
                    stats.marks.mark("claude_complete")
                    break
                if isinstance(item, Exception):
                    if fatal:
                        raise item
                    log.warning("optional stream failed; continuing", exc_info=item)
                    break
                stats.marks.mark("first_sentence")
                async for chunk in self._speak_one(item, pace, stats):
                    yield chunk
                if stats.truncated:
                    break
        finally:
            stats.marks.mark("speaking_complete")
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

        stats.marks.mark("first_tts_start")
        started = time.perf_counter()
        pcm = await self.engine.synth(sentence, wpm, self.voice)
        synth_seconds = time.perf_counter() - started
        tts_done = stats.marks.mark("first_tts_complete")
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
        stats.marks.add_chunk(sentence, 1, tts_done - synth_seconds, tts_done,
                              pcm_duration(len(pcm), self.engine.sample_rate))
        pace.observe(len(pcm) + len(gap), words)
        stats.sentences += 1
        stats.words += words
        stats.script.append(sentence)
        if not stats.first_audio_at:
            stats.first_audio_at = time.perf_counter() - stats.started_at
            log.info("first audio ready after %.2fs", stats.first_audio_at)
        yield pcm
        yield gap

    # ---- which streaming architecture this request uses -------------------
    #
    # One decision, read from `settings.streaming_pipeline` at request time
    # and applied at every point a pump is made or spoken. Default is
    # `legacy`, so an installation that has never heard of this setting
    # behaves exactly as it always has, and rolling back is one environment
    # variable and a restart.
    #
    # The two architectures are not blended: a pump made by `_start_phase6`
    # carries `AssembledChunk` and must be spoken by `_speak_phase6`, so the
    # three helpers below always agree with each other.

    def _phase6(self) -> bool:
        return settings.streaming_pipeline == "phase6"

    def _pump_for(self, sentences: AsyncIterator[str]) -> "_Pump":
        """Start a sentence stream under whichever architecture is selected.

        Each call builds its own pump, and under Phase 6 its own script buffer
        and assembler with it - which is what keeps `_answer_first`'s two
        concurrent streams from ever sharing assembler state or interleaving
        their text into one chunk.
        """
        return self._start_phase6(sentences) if self._phase6() else self._start(sentences)

    def _speak_pump(self, pump: "_Pump", pace: PaceController,
                    stats: GenerationStats, fatal: bool = True) -> AsyncIterator[bytes]:
        return (self._speak_phase6(pump, pace, stats, fatal) if self._phase6()
                else self._speak(pump, pace, stats, fatal))

    def _speak_item(self, item, pace: PaceController,
                    stats: GenerationStats) -> AsyncIterator[bytes]:
        """One pulled item: a sentence under legacy, a chunk under Phase 6."""
        return (self._speak_chunk(item, pace, stats) if self._phase6()
                else self._speak_one(item, pace, stats))

    async def _answer_first(
        self,
        plan: EpisodePlan,
        pace: PaceController,
        stats: GenerationStats,
        notes: ScriptNotes,
    ) -> AsyncIterator[bytes]:
        """Answer immediately from knowledge; let research take over underneath.

        The listener asked something that needs today's facts, and researching
        it costs 10-25 seconds before a word can be written. Both halves start
        at once: one with no tools, which begins writing straight away, and one
        with web search, which is still reading. The first is spoken while the
        second works, and the moment the researched half has a sentence ready
        the episode moves to it.

        This is the shape the cold open had and the content it lacked. The
        opener was told to state no facts, so the seconds it covered were
        worthless and there were only five of them. Here the cover *is* the
        answer - the durable half of it - written by the same model at full
        length, so a listener who never reaches the handover has still been
        told something true.

        The two halves are divided by **content, not by text**. The opening
        cannot know what the research will find and the research cannot know
        the opening's words, so neither is asked to: the opening takes what
        does not change week to week, the continuation takes what is current
        and is told to correct the opening in passing if its sources disagree.
        """
        instant_plan = dataclasses.replace(plan, search=False, role="opening")
        research_plan = dataclasses.replace(plan, search=True, role="continuation")

        stats.marks.mark("claude_start")
        instant = self._pump_for(
            self.generator.stream_sentences(instant_plan, ScriptNotes()))
        research = self._pump_for(
            self.generator.stream_sentences(research_plan, notes))
        stats.answered_first = True
        handover = time.perf_counter()

        try:
            # Speak the instant half one sentence at a time, checking after each
            # whether research has arrived. Checking between sentences rather
            # than mid-sentence is what makes the handover inaudible.
            # The instant half may cover at most this much of the episode. Past
            # it, the researched half is owed the remainder - see
            # answer_first_share in config.py for why this ceiling exists.
            cover_ceiling = plan.target_seconds * settings.answer_first_share
            while not research.ready() and pace.elapsed < cover_ceiling:
                item = await instant.next()
                if item is None or isinstance(item, Exception):
                    # The instant half ended or failed before research landed.
                    # Nothing to cover with; wait for the researched half, which
                    # is the episode either way.
                    if isinstance(item, Exception):
                        log.warning("instant half failed; waiting for research",
                                    exc_info=item)
                    break
                async for chunk in self._speak_item(item, pace, stats):
                    yield chunk
                if stats.truncated:
                    return
        finally:
            await instant.close()

        stats.handover_seconds = time.perf_counter() - handover
        if not research.ready():
            log.info("answered from knowledge for %.0fs; now waiting on research",
                     pace.elapsed)
        log.info("research took over after %.1fs of answering from knowledge",
                 stats.handover_seconds)
        async for chunk in self._speak_pump(research, pace, stats):
            yield chunk

    async def _cache_key(self, plan: EpisodePlan) -> str:
        """Where this episode lives in the shared cache. "" when caching is off."""
        if not self.cache:
            return ""
        # An episode built on someone's own document, photo or link is theirs.
        # No key means no read, no write, and therefore nothing that could be
        # served to another listener or surface in Explore.
        if plan.attachments:
            return ""
        canonical = None
        if settings.cache_semantic_key:
            canonical = await canonical_key(plan.query, self.generator.client)
        return cache_key(plan.query, plan.minutes, canonical, plan.context, plan.search)

    def _bucket(self, plan: EpisodePlan) -> str:
        """The set of entries this episode could stand in for.

        Empty when the episode is nobody else's business - an attachment makes
        it personal, and a personal episode must not be findable by anyone,
        including by being near something.
        """
        if not self.cache or plan.attachments or not settings.cache_vector:
            return ""
        return key_bucket(plan.minutes, plan.context, plan.search)

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
        bucket = self._bucket(plan) if shareable else ""
        if self.cache and shareable:
            cached = self.cache.get(key)
            if cached:
                stats.match = "exact"
            elif bucket and not plan.cached_only:
                # Nobody has asked this in these words. Someone may have asked
                # it in different ones - which is most of what the cache misses,
                # since the key is an exact token set and people do not phrase
                # questions the same way twice.
                #
                # Not for `cached_only`: Explore offers a specific episode that
                # a listener has already seen the title of, and handing them a
                # near neighbour instead would be answering a question they did
                # not tap. A replay surface has to replay.
                near = self.cache.nearest(bucket, plan.query)
                if near:
                    cached = self.cache.get(near[0])
                    if cached:
                        key = near[0]
                        stats.match, stats.match_score = "near", near[1]
                        log.info("near cache hit %.3f for %r", near[1], plan.query)
            if cached:
                stats.cache = "hit"
                stats.thread = self.cache.thread(key)
                log.info("cache %s hit for %r (%d min)", stats.match, plan.query, plan.minutes)
                # Replaying the same sentences through the same controller
                # reproduces the episode exactly - and costs zero API tokens.
                async for chunk in self._speak_pump(
                        self._pump_for(_replay(cached)), pace, stats):
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

        # Time-to-first-token is invisible from outside `stream_sentences`,
        # which yields whole sentences. Wrapping the client observes the first
        # delta and changes nothing about the request - the same wrapper Phase
        # 6 measured with. Guarded because a test generator has no client.
        client = getattr(self.generator, "client", None)
        if client is not None and not isinstance(client, TimedClient):
            self.generator.client = TimedClient(client, stats.marks)

        # --- Generate ------------------------------------------------------
        # Nothing is spoken until the real script arrives. The opener that used
        # to cover this wait is gone: see PROBLEMS.md 55.
        notes = ScriptNotes()

        if plan.search and settings.answer_first:
            async for chunk in self._answer_first(plan, pace, stats, notes):
                yield chunk
        else:
            stats.marks.mark("claude_start")
            body = self._pump_for(self.generator.stream_sentences(plan, notes))
            async for chunk in self._speak_pump(body, pace, stats):
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
            extra = self._pump_for(
                self.generator.top_up(plan, " ".join(stats.script), words_needed))
            async for chunk in self._speak_pump(extra, pace, stats):
                yield chunk
            if pace.spoken_words == before:
                break  # the top-up produced nothing; stop asking

        stats.thread = notes.thread

        if self.cache and self.cache_writes and shareable and stats.script:
            ttl = ttl_for(plan.query)
            self.cache.put(key, stats.script, ttl, plan.query, stats.thread,
                           plan.minutes, bucket)
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
