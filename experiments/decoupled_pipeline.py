"""Three stages that cannot stall each other, and the proof that they do not.

Phase 5 wired production's sentence stream straight into a bounded TTS queue.
When Chatterbox fell behind, the queue filled and the *reader* stopped pulling
from Claude - so the measured `claude_total` of 66.7s carried 64.3s of our own
backpressure inside it. That number says nothing about Claude.

Phase 6 puts a cheap text buffer in between:

    Claude stream
      -> reader          (never touches the TTS queue)
      -> script buffer   (bounded by CHARACTERS - text is nearly free)
      -> assembler       (whole sentences -> speech-sized chunks)
      -> TTS work queue  (bounded, may block the assembler, never the reader)
      -> Chatterbox
      -> playback model

The assembler may block on a full TTS queue. That is fine and intended: it is
downstream of the buffer, so Claude keeps being read regardless. The reader
blocks only on the script buffer, and `SCRIPT_BUFFER_CHARS` is about twenty
times a three-minute script, so in practice it never does - and if it ever
did, `reader_blocked_seconds` measures it and the run fails.

Nothing here imports torch, numpy or the Anthropic SDK. The TTS stage is a
callable, so the whole thing runs against a stub - which is how the decoupling
is proved before a GPU is rented, including by building the coupled version
deliberately and watching the assertion catch it.
"""
from __future__ import annotations

import asyncio
import pathlib
import statistics
import sys
from dataclasses import dataclass, field
from typing import AsyncIterator, Awaitable, Callable, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments.pipeline_probe import EventLog                  # noqa: E402
from experiments.speech_assembler import (AssembledChunk,        # noqa: E402
                                          AssemblyPolicy,
                                          SpeechAssembler,
                                          audio_seconds_for)

try:  # pragma: no cover - production's own numbers where importable
    from pipeline import QUEUE_DEPTH, SENTENCE_GAP
except Exception:  # pragma: no cover
    QUEUE_DEPTH, SENTENCE_GAP = 4, 0.12

#: The script buffer's bound, in characters. A three-minute FAM episode is
#: about 2,700 characters, so this is roughly twenty of them - a real bound,
#: and one no realistic episode can reach. Text is the cheap half; this is
#: the whole point of separating it from the audio half.
SCRIPT_BUFFER_CHARS = 64_000

#: How often the assembler wakes with no new sentence, to run its timer and
#: headroom rules. Small enough to be invisible against `max_wait_seconds`.
ASSEMBLER_TICK = 0.05

#: Reader blocking below this is scheduler noise, not architecture.
BLOCKED_EPSILON = 0.05

Synth = Callable[[str], Awaitable[tuple]]


# --------------------------------------------------------------------------
@dataclass
class SpokenChunk:
    """One assembled chunk, from assembly through synthesis to playback."""

    chunk: AssembledChunk
    queued_at: float = 0.0
    tts_start: float = 0.0
    tts_complete: float = 0.0
    audio_seconds: float = 0.0
    sample_rate: int = 0
    #: When the listener would reach this chunk, and whether it was ready.
    plays_from: float = 0.0
    stall_seconds: float = 0.0
    headroom_after: float = 0.0

    @property
    def generate_seconds(self) -> float:
        return self.tts_complete - self.tts_start

    @property
    def realtime_factor(self) -> Optional[float]:
        seconds = self.generate_seconds
        return self.audio_seconds / seconds if seconds > 0 else None

    def to_dict(self) -> dict:
        out = self.chunk.to_dict()
        out.update({
            "queued_at": self.queued_at, "tts_start": self.tts_start,
            "tts_complete": self.tts_complete,
            "audio_seconds": self.audio_seconds,
            "generate_seconds": self.generate_seconds,
            "realtime_factor": self.realtime_factor,
            "queue_wait_seconds": self.tts_start - self.queued_at,
            "plays_from": self.plays_from, "stall_seconds": self.stall_seconds,
            "headroom_after": self.headroom_after,
        })
        return out


@dataclass
class Playback:
    """The listener's timeline, maintained as audio is produced.

    `plays_until` is the wall-clock moment the audio produced so far would
    finish playing, so `plays_until - now` is live headroom - which the
    assembler reads when deciding whether batching still matters.
    """

    gap: float = SENTENCE_GAP
    started_at: Optional[float] = None
    plays_until: float = 0.0
    stalls: list = field(default_factory=list)
    series: list = field(default_factory=list)

    def add(self, index: int, ready_at: float, audio_seconds: float) -> tuple:
        if self.started_at is None:
            self.started_at, plays_from, stall = ready_at, ready_at, 0.0
        else:
            # Sampled before this chunk is folded in: how much buffered audio
            # was left when it arrived. Negative is an underrun, and sampling
            # after the fold would make that impossible to see.
            self.series.append({"at": ready_at,
                                "headroom": self.plays_until - ready_at,
                                "note": f"chunk {index:02d} arrived"})
            earliest = self.plays_until + self.gap
            plays_from = max(earliest, ready_at)
            stall = max(0.0, ready_at - earliest)
            if stall > 0:
                self.stalls.append({"index": index, "stall_seconds": stall,
                                    "needed_at": earliest, "ready_at": ready_at})
        self.plays_until = plays_from + audio_seconds
        return plays_from, stall

    def headroom(self, now: float) -> Optional[float]:
        """None before playback starts - there is nothing to be ahead of yet."""
        return None if self.started_at is None else self.plays_until - now

    def sample(self, now: float, note: str) -> None:
        headroom = self.headroom(now)
        if headroom is not None:
            self.series.append({"at": now, "headroom": headroom, "note": note})


@dataclass
class DecoupledRun:
    """Everything one request produced, and every way it could have gone wrong."""

    log: EventLog
    policy: AssemblyPolicy
    coupled: bool = False
    sentences: list = field(default_factory=list)
    chunks: list = field(default_factory=list)
    samples: list = field(default_factory=list)
    sample_rate: int = 0
    playback: Playback = field(default_factory=Playback)
    script_buffer_series: list = field(default_factory=list)
    tts_queue_series: list = field(default_factory=list)
    #: (3) Time the Claude reader spent unable to advance because of us.
    reader_blocked_seconds: float = 0.0
    #: (2) Time inside the reader loop that was our processing, not the network.
    reader_local_seconds: float = 0.0
    #: Time the assembler spent blocked on a full TTS queue. Expected and fine.
    assembler_blocked_seconds: float = 0.0
    tts_backlog_at_claude_complete: int = 0
    queue_depth: int = QUEUE_DEPTH
    assembler_witness: Optional[str] = None

    @property
    def peak_script_buffer(self) -> int:
        return max((r["chars"] for r in self.script_buffer_series), default=0)

    @property
    def peak_tts_queue(self) -> int:
        return max((r["depth"] for r in self.tts_queue_series), default=0)


# --------------------------------------------------------------------------
async def run_decoupled(sentences: AsyncIterator[str], synth: Synth,
                        log: EventLog, policy: Optional[AssemblyPolicy] = None,
                        queue_depth: int = QUEUE_DEPTH,
                        buffer_chars: int = SCRIPT_BUFFER_CHARS,
                        gap: float = SENTENCE_GAP,
                        coupled: bool = False) -> DecoupledRun:
    """Read, assemble and speak, with the three stages decoupled.

    `coupled=True` reproduces Phase 5 deliberately - the reader writes straight
    onto the TTS queue, one sentence per synthesis - so the decoupling
    assertion can be shown to reject it. It is a test fixture, not an option.
    """
    policy = policy or AssemblyPolicy()
    run = DecoupledRun(log=log, policy=policy, coupled=coupled,
                       queue_depth=queue_depth)
    run.playback.gap = gap

    assembler = SpeechAssembler(policy=policy, clock=log.now)
    buffer: asyncio.Queue = asyncio.Queue()
    work: asyncio.Queue = asyncio.Queue(maxsize=queue_depth)
    buffered_chars = 0
    space = asyncio.Event()
    space.set()
    queued = 0

    def sample_buffer(note: str) -> None:
        run.script_buffer_series.append(
            {"at": log.now(), "chars": buffered_chars,
             "sentences": buffer.qsize(), "event": note})

    def sample_queue(note: str) -> None:
        run.tts_queue_series.append({"at": log.now(), "depth": queued,
                                     "event": note})

    async def put_work(spoken: SpokenChunk) -> None:
        nonlocal queued
        if work.full():
            blocked = log.now()
            await work.put(spoken)
            waited = log.now() - blocked
            if coupled:
                # In coupled mode this blocking IS the reader blocking, which
                # is the whole thing Phase 6 exists to remove.
                run.reader_blocked_seconds += waited
            else:
                run.assembler_blocked_seconds += waited
        else:
            await work.put(spoken)
        queued += 1
        sample_queue("enqueue")

    # ---- stage 1: read Claude, and never touch the TTS queue -------------
    async def read() -> None:
        nonlocal buffered_chars
        index = 0
        async for sentence in sentences:
            sentence = (sentence or "").strip()
            if not sentence:
                continue
            started = log.mark(f"raw_sentence:{index:02d}",
                               {"words": len(sentence.split()),
                                "characters": len(sentence)})
            if index == 0:
                log.mark("first_sentence_ready",
                         {"words": len(sentence.split())})
            run.sentences.append(sentence)

            if coupled:
                # Phase 5's shape, kept only so the assertion can reject it:
                # one sentence, one synthesis, straight onto the TTS queue.
                assembler.seen.append(sentence)
                chunk = AssembledChunk(
                    index=index, text=sentence, sentences=1,
                    words=len(sentence.split()), characters=len(sentence),
                    ready_at=log.now(), reason="coupled: one sentence per call",
                    first_sentence_at=started)
                log.mark(f"speech_chunk_ready:{index:02d}",
                         {"words": chunk.words, "reason": chunk.reason})
                if index == 0:
                    log.mark("first_speech_chunk_ready", {"words": chunk.words})
                await put_work(SpokenChunk(chunk=chunk, queued_at=log.now()))
            else:
                # The only thing that can hold the reader up: a text buffer
                # bounded at about twenty episodes. If this ever waits, it is
                # measured and the run fails.
                while buffered_chars + len(sentence) > buffer_chars:
                    space.clear()
                    blocked = log.now()
                    await space.wait()
                    run.reader_blocked_seconds += log.now() - blocked
                buffered_chars += len(sentence)
                await buffer.put(sentence)
                sample_buffer("append")

            run.reader_local_seconds += log.now() - started
            index += 1

        log.mark("claude_complete", {"sentences": index})
        run.tts_backlog_at_claude_complete = queued
        run.playback.sample(log.now(), "claude_complete")
        await (work.put(None) if coupled else buffer.put(None))

    # ---- stage 2: assemble sentences into speech-sized chunks ------------
    async def assemble() -> None:
        nonlocal buffered_chars
        ended = False
        while not ended:
            try:
                sentence = await asyncio.wait_for(buffer.get(), ASSEMBLER_TICK)
            except asyncio.TimeoutError:
                for chunk in assembler.due(run.playback.headroom(log.now())):
                    await release(chunk)
                continue
            if sentence is None:
                ended = True
                for chunk in assembler.flush():
                    await release(chunk)
                break
            buffered_chars -= len(sentence)
            if buffered_chars + 1 <= buffer_chars:
                space.set()
            sample_buffer("consume")
            for chunk in assembler.offer(sentence,
                                         run.playback.headroom(log.now())):
                await release(chunk)
        await work.put(None)

    async def release(chunk: AssembledChunk) -> None:
        log.mark(f"speech_chunk_ready:{chunk.index:02d}",
                 {"words": chunk.words, "characters": chunk.characters,
                  "sentences": chunk.sentences, "reason": chunk.reason})
        if chunk.index == 0:
            log.mark("first_speech_chunk_ready", {"words": chunk.words})
        await put_work(SpokenChunk(chunk=chunk, queued_at=log.now()))

    # ---- stage 3: synthesise -------------------------------------------
    async def speak() -> None:
        nonlocal queued
        while True:
            spoken = await work.get()
            if spoken is None:
                break
            queued -= 1
            sample_queue("dequeue")
            index = spoken.chunk.index
            spoken.tts_start = log.mark(f"tts_start:{index:02d}",
                                        {"words": spoken.chunk.words})
            if index == 0:
                log.mark("first_tts_start", {"words": spoken.chunk.words})
            samples, rate = await synth(spoken.chunk.text)
            spoken.tts_complete = log.mark(f"tts_complete:{index:02d}")
            spoken.sample_rate = rate
            spoken.audio_seconds = len(samples) / rate if rate else 0.0
            if index == 0:
                log.mark("first_tts_complete")
                # One-shot synthesis: the first playable moment is this one.
                log.mark("first_playable_audio")
                log.mark("playback_start")
            spoken.plays_from, spoken.stall_seconds = run.playback.add(
                index, spoken.tts_complete, spoken.audio_seconds)
            spoken.headroom_after = run.playback.headroom(log.now())
            run.playback.sample(log.now(), f"chunk {index:02d} buffered")
            run.chunks.append(spoken)
            run.samples.append(samples)
            run.sample_rate = rate
        log.mark("final_audio_complete", {"chunks": len(run.chunks)})

    stages = (read, speak) if coupled else (read, assemble, speak)
    tasks = [asyncio.create_task(coro()) for coro in stages]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        raise

    run.assembler_witness = assembler.spoken_matches_source(
        [s.chunk for s in run.chunks])
    return run


# --------------------------------------------------------------------------
# the assertions
# --------------------------------------------------------------------------
def phase6_problems(run: DecoupledRun) -> list:
    """Empty means every Phase 6 requirement held. Anything else is a failure."""
    problems, log = [], run.log
    first_start = log.at("first_tts_start")
    claude_done = log.at("claude_complete")

    # 1 - genuine overlap
    if first_start is None:
        problems.append("no TTS ever started")
    elif claude_done is None:
        problems.append("Claude never finished; nothing to compare against")
    elif first_start >= claude_done and len(run.chunks) > 1:
        problems.append(
            f"first_tts_start {first_start:.3f}s is at or after claude_complete "
            f"{claude_done:.3f}s - sequential, not concurrent")

    # 2 and 3 - the first payload is the first assembled chunk, not the script
    if not run.chunks:
        problems.append("no chunks were synthesised")
    else:
        first = run.chunks[0].chunk
        if first.index != 0:
            problems.append("the first synthesis was not chunk 0")
        if len(run.chunks) > 1:
            whole = " ".join(run.sentences).split()
            if first.text.split() == whole:
                problems.append(
                    "the first synthesis was the whole final script - the run "
                    "concatenated before speaking instead of speaking as it went")
        for position, spoken in enumerate(run.chunks):
            if spoken.chunk.index != position:
                problems.append(f"chunks were spoken out of order at {position}")
                break

    # 4 - the central Phase 6 assertion
    if run.reader_blocked_seconds > BLOCKED_EPSILON:
        problems.append(
            f"the Claude reader was blocked for "
            f"{run.reader_blocked_seconds:.3f}s by our own queues - TTS "
            "saturation is still reaching upstream")

    # 5 and 6 - text integrity, end to end
    if run.assembler_witness:
        problems.append(run.assembler_witness)
    spoken_once = [s.chunk.index for s in run.chunks]
    if len(set(spoken_once)) != len(spoken_once):
        problems.append("a chunk was synthesised more than once")

    return problems


def assert_phase6(run: DecoupledRun) -> None:
    problems = phase6_problems(run)
    if problems:
        raise AssertionError("; ".join(problems))


def decoupling_evidence(run: DecoupledRun) -> dict:
    """Did the run actually exercise the decoupling, or merely not violate it?

    If TTS never fell behind there was nothing to be decoupled *from*, and
    saying the architecture is proved would be overclaiming.
    """
    log = run.log
    claude_done, final = log.at("claude_complete"), log.at("final_audio_complete")
    # The last synthesis always finishes after the reader does, so "TTS
    # outlived Claude" proves nothing on its own. The decoupling is only
    # exercised if the TTS queue actually saturated, or Claude finished with a
    # backlog waiting - the situations where a coupled build would have stalled.
    exercised = (run.tts_backlog_at_claude_complete > 0
                 or run.peak_tts_queue >= run.queue_depth)
    return {
        "reader_blocked_seconds": run.reader_blocked_seconds,
        "assembler_blocked_seconds": run.assembler_blocked_seconds,
        "tts_backlog_at_claude_complete": run.tts_backlog_at_claude_complete,
        "tts_outlived_claude_by": (None if claude_done is None or final is None
                                   else final - claude_done),
        "exercised": exercised,
        "peak_tts_queue_depth": run.peak_tts_queue,
        "tts_queue_saturated": run.peak_tts_queue >= run.queue_depth,
        "overlap_applicable": len(run.chunks) > 1,
        "verdict": ("Claude finished while TTS still had a backlog, without "
                    "the reader ever blocking"
                    if exercised and run.reader_blocked_seconds <= BLOCKED_EPSILON
                    else "TTS never fell behind, so the decoupling was not "
                         "exercised by this run"
                    if not exercised else
                    "the reader was blocked; see the failed assertion"),
    }


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def first_handoff(run: DecoupledRun) -> dict:
    """Did the opening chunk buy enough playback to cover the second one?

    The first chunk has no word floor - time to first listen outranks
    everything - so the constraint the floor used to enforce is measured here
    instead. One TTS worker means chunk 2 is synthesised while chunk 1 plays:

        audio(chunk 1)  >=  generate(chunk 2)

    A short opening is a deliberate trade, not a bug. This says what it cost.
    """
    if len(run.chunks) < 2:
        return {"applicable": False,
                "why": "fewer than two chunks; there was no handoff"}
    first, second = run.chunks[0], run.chunks[1]
    cover = first.audio_seconds
    needed = second.generate_seconds
    return {
        "applicable": True,
        "first_chunk_words": first.chunk.words,
        "first_chunk_audio_seconds": cover,
        "second_chunk_words": second.chunk.words,
        "second_chunk_generate_seconds": needed,
        "margin_seconds": cover - needed,
        "covered": cover >= needed,
        "note": ("the opening covered the second chunk's synthesis"
                 if cover >= needed else
                 f"the opening was {needed - cover:.2f}s short of covering the "
                 "second chunk - the cost of releasing it without a word floor"),
    }


def playback_report(run: DecoupledRun) -> dict:
    if not run.chunks:
        return {"chunks": 0, "stalls": [], "total_stall_seconds": 0.0,
                "first_handoff": first_handoff(run)}
    headrooms = [row["headroom"] for row in run.playback.series]
    at_claude = [row["headroom"] for row in run.playback.series
                 if row["note"] == "claude_complete"]
    audio = sum(s.audio_seconds for s in run.chunks)
    return {
        "chunks": len(run.chunks),
        "sentence_gap_seconds": run.playback.gap,
        "audio_seconds": audio,
        "playback_start": run.playback.started_at,
        "playback_stalls": len(run.playback.stalls),
        "stalls": run.playback.stalls,
        "total_stall_seconds": sum(s["stall_seconds"]
                                   for s in run.playback.stalls),
        "minimum_playback_headroom": min(headrooms) if headrooms else None,
        "median_playback_headroom": (statistics.median(headrooms)
                                     if headrooms else None),
        "maximum_playback_headroom": max(headrooms) if headrooms else None,
        "headroom_at_claude_complete": at_claude[0] if at_claude else None,
        "headroom_at_final_tts": run.chunks[-1].headroom_after,
        "headroom_series": run.playback.series,
        "listening_finishes_at": run.playback.plays_until,
        "first_handoff": first_handoff(run),
    }


def chunk_report(run: DecoupledRun) -> dict:
    """The efficiency question: did assembly kill the pathological calls?"""
    if not run.chunks:
        return {"tts_invocations": 0}
    words = [s.chunk.words for s in run.chunks]
    generate = [s.generate_seconds for s in run.chunks]
    audio = sum(s.audio_seconds for s in run.chunks)
    total_generate = sum(generate)
    return {
        "tts_invocations": len(run.chunks),
        "raw_sentences": len(run.sentences),
        "median_words": statistics.median(words),
        "min_words": min(words), "max_words": max(words),
        "chunks_under_5_words": sum(1 for w in words if w < 5),
        "chunks_under_10_words": sum(1 for w in words if w < 10),
        "mean_generate_seconds": statistics.mean(generate),
        "total_tts_compute_seconds": total_generate,
        "audio_seconds": audio,
        "aggregate_realtime_factor": (audio / total_generate
                                      if total_generate else None),
        "release_reasons": {reason: sum(1 for s in run.chunks
                                        if s.chunk.reason == reason)
                            for reason in
                            sorted({s.chunk.reason for s in run.chunks})},
        # The opening carries no word floor, so it can buy little playback and
        # leave the assembler in its emergency path for a while - shipping
        # short chunks to protect continuity. That is the trade working as
        # intended, and it is worth seeing rather than only its outcome.
        **_headroom_recovery(run),
    }


def _headroom_recovery(run: DecoupledRun) -> dict:
    forced = [index for index, spoken in enumerate(run.chunks)
              if "headroom" in spoken.chunk.reason]
    return {
        "chunks_forced_by_headroom": len(forced),
        "headroom_recovered_after_chunk": (max(forced) if forced else None),
        "words_while_recovering": [run.chunks[i].chunk.words for i in forced],
    }


def timing_report(run: DecoupledRun) -> dict:
    """The five concepts the report must never let anyone confuse."""
    log = run.log
    return {
        "exa_latency": log.span("exa_start", "exa_complete"),
        "claude_ttft": log.span("claude_start", "claude_ttft"),
        "claude_to_first_sentence": log.span("claude_start",
                                             "first_sentence_ready"),
        "claude_to_first_speech_chunk": log.span("claude_start",
                                                 "first_speech_chunk_ready"),
        "first_chunk_tts_seconds": log.span("first_tts_start",
                                            "first_tts_complete"),
        # 1 - what Claude actually took, with nothing downstream holding it.
        "claude_stream_seconds": log.span("claude_start", "claude_complete"),
        # 2 - our own work inside the reader loop.
        "claude_local_processing_seconds": run.reader_local_seconds,
        # 3 - time the reader could not advance because of our architecture.
        "claude_reader_blocked_seconds": run.reader_blocked_seconds,
        # Downstream blocking, which is allowed and is not Claude's problem.
        "assembler_blocked_seconds": run.assembler_blocked_seconds,
        # 4 - the voice.
        "tts_total_seconds": sum(s.generate_seconds for s in run.chunks),
        "search_to_first_listen": log.span("request_start",
                                           "first_playable_audio"),
        "search_to_complete_audio": log.span("request_start",
                                             "final_audio_complete"),
        "overlap_seconds": (None if log.at("claude_complete") is None
                            or log.at("first_tts_start") is None
                            else log.at("claude_complete")
                            - log.at("first_tts_start")),
        "peak_script_buffer_chars": run.peak_script_buffer,
        "peak_tts_queue_depth": run.peak_tts_queue,
        "planned_audio_seconds": sum(audio_seconds_for(s.chunk.words)
                                     for s in run.chunks),
    }
