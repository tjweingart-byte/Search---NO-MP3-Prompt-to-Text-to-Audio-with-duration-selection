"""Claude and Chatterbox running at the same time, and the proof that they did.

The previous combined run measured a sequential pipeline: it *detected* the
first speakable chunk at about 2.4s and then waited for `claude_complete`
before starting TTS. That is not FAM's architecture. Production
(`pipeline.py`) pumps `script_generator.stream_sentences` into a bounded queue
and speaks each sentence as it lands, while Claude is still writing.

This harness measures that shape. It reuses the real chunker rather than
reimplementing it - the sentence boundary, `clean_for_speech`, and the
hold-back of the trailing `<<NEXT:` marker are production's, so what is
measured is what ships.

**The concurrency is asserted, not assumed.** `concurrency_problems()` fails a
run whose first TTS started at or after Claude finished, and fails one whose
first synthesis text is not the first chunk the chunker emitted. A sequential
pipeline that happens to be fast would otherwise pass for a concurrent one.

Nothing here imports torch, numpy or the Anthropic SDK. The TTS stage is a
callable, so the whole harness runs on a laptop against a stub - which is how
its concurrency is proved before a GPU is rented.
"""
from __future__ import annotations

import asyncio
import pathlib
import sys
from dataclasses import dataclass, field
from typing import AsyncIterator, Awaitable, Callable, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments.pipeline_probe import EventLog                  # noqa: E402

#: Production's own numbers, imported rather than copied. `pipeline.py` bounds
#: the sentence queue at 4 - a memory budget, not a tuning knob - and lays
#: 0.12s of silence between sentences so delivery does not sound rushed.
try:  # pragma: no cover - the app's constants when they are importable
    from pipeline import QUEUE_DEPTH, SENTENCE_GAP
except Exception:  # pragma: no cover - a bare harness still runs
    QUEUE_DEPTH, SENTENCE_GAP = 4, 0.12


#: A synthesiser: text in, (samples, sample_rate) out. Awaitable, because the
#: real one hands a blocking torch call to a thread exactly as `tts.py` does
#: for Piper - which is what lets Claude keep streaming while it runs.
Synth = Callable[[str], Awaitable[tuple]]


@dataclass
class ChunkRecord:
    """One speakable chunk, from the moment it existed to the moment it spoke."""

    index: int
    text: str
    words: int
    ready_at: float
    tts_start: float = 0.0
    tts_complete: float = 0.0
    audio_seconds: float = 0.0
    sample_rate: int = 0

    @property
    def generate_seconds(self) -> float:
        return self.tts_complete - self.tts_start

    @property
    def realtime_factor(self) -> Optional[float]:
        seconds = self.generate_seconds
        return self.audio_seconds / seconds if seconds > 0 else None

    @property
    def waited_in_queue(self) -> float:
        """How long this chunk sat written but unspoken."""
        return self.tts_start - self.ready_at

    def to_dict(self) -> dict:
        return {
            "index": self.index, "text": self.text, "words": self.words,
            "ready_at": self.ready_at, "tts_start": self.tts_start,
            "tts_complete": self.tts_complete,
            "audio_seconds": self.audio_seconds,
            "generate_seconds": self.generate_seconds,
            "realtime_factor": self.realtime_factor,
            "waited_in_queue": self.waited_in_queue,
        }


@dataclass
class PipelineRun:
    """Everything one concurrent request produced, and when."""

    log: EventLog
    chunks: list = field(default_factory=list)
    #: The first chunk as the chunker emitted it, kept separately so the
    #: assertion can compare against it rather than against itself.
    first_emitted: str = ""
    full_script: str = ""
    queue_depth: list = field(default_factory=list)
    #: Seconds the producer spent blocked on a full queue. Production applies
    #: the same backpressure; separating it keeps `claude_complete` readable.
    backpressure_seconds: float = 0.0
    samples: list = field(default_factory=list)
    sample_rate: int = 0

    @property
    def peak_queue_depth(self) -> int:
        return max((row["depth"] for row in self.queue_depth), default=0)


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------
async def run_concurrent(sentences: AsyncIterator[str], synth: Synth,
                         log: EventLog, queue_depth: int = QUEUE_DEPTH,
                         max_chunks: int = 0) -> PipelineRun:
    """Stream chunks into a FIFO queue while a worker synthesises them.

    The producer never waits for the consumer except through the bounded queue,
    and the consumer never waits for the producer to finish. That is the whole
    point: the first chunk reaches the voice while the model is still writing.
    """
    run = PipelineRun(log=log)
    queue: asyncio.Queue = asyncio.Queue(maxsize=queue_depth or 0)
    depth = 0

    def record(event: str) -> None:
        # Sampled, not marked: one named event per enqueue would bury the
        # named marks it sits between.
        run.queue_depth.append({"at": log.now(), "depth": depth,
                                "event": event})

    async def produce() -> None:
        nonlocal depth
        index = 0
        async for sentence in sentences:
            if not sentence:
                continue
            at = log.mark(f"chunk_ready:{index:02d}",
                          {"words": len(sentence.split())})
            if index == 0:
                log.mark("first_speakable_chunk_ready", {"words": len(sentence.split())})
                run.first_emitted = sentence
            run.full_script += (" " if run.full_script else "") + sentence
            chunk = ChunkRecord(index=index, text=sentence,
                                words=len(sentence.split()), ready_at=at)
            # Blocked time is measured, not assumed away: a full queue throttles
            # the model, and that is production behaviour worth seeing.
            if queue.full():
                blocked = log.mark(f"producer_blocked:{index:02d}")
                await queue.put(chunk)
                run.backpressure_seconds += log.mark(
                    f"producer_resumed:{index:02d}") - blocked
            else:
                await queue.put(chunk)
            depth += 1
            record("enqueue")
            index += 1
            if max_chunks and index >= max_chunks:
                await sentences.aclose()
                break
        log.mark("claude_complete", {"chunks": index,
                                     "chars": len(run.full_script)})
        await queue.put(None)

    async def consume() -> None:
        nonlocal depth
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            depth -= 1
            record("dequeue")
            chunk.tts_start = log.mark(f"tts_start:{chunk.index:02d}")
            if chunk.index == 0:
                log.mark("first_tts_start", {"words": chunk.words})
            samples, rate = await synth(chunk.text)
            chunk.tts_complete = log.mark(f"tts_complete:{chunk.index:02d}")
            chunk.sample_rate = rate
            chunk.audio_seconds = len(samples) / rate if rate else 0.0
            if chunk.index == 0:
                log.mark("first_tts_complete")
                # One-shot synthesis: the first playable moment is the moment
                # the first chunk finished. Phase 1 established this; it is the
                # finding, not a limitation of the probe.
                log.mark("first_playable_audio")
            run.chunks.append(chunk)
            run.samples.append(samples)
            run.sample_rate = rate
        log.mark("final_audio_complete", {"chunks": len(run.chunks)})

    producer = asyncio.create_task(produce())
    consumer = asyncio.create_task(consume())
    try:
        await asyncio.gather(producer, consumer)
    except BaseException:
        # A failure in either half would otherwise leave the other blocked on a
        # queue nobody will drain, turning a crash into a hang.
        for task in (producer, consumer):
            task.cancel()
        raise
    return run


# --------------------------------------------------------------------------
# the assertions - the point of the experiment, not decoration
# --------------------------------------------------------------------------
def concurrency_problems(run: PipelineRun) -> list:
    """Empty means the pipeline really did overlap. Anything else is a failure.

    Written as a list rather than a bare assert so a run reports every way it
    fell short at once, and so the same check can be a test and a gate.
    """
    problems = []
    log = run.log
    first_start = log.at("first_tts_start")
    claude_done = log.at("claude_complete")

    if first_start is None:
        problems.append("no TTS ever started")
    elif claude_done is None:
        problems.append("Claude never finished; nothing to compare against")
    elif first_start >= claude_done:
        problems.append(
            f"first_tts_start {first_start:.3f}s is at or after claude_complete "
            f"{claude_done:.3f}s - this is a sequential pipeline, not a "
            "concurrent one")

    if not run.chunks:
        problems.append("no chunks were synthesised")
    elif not run.first_emitted:
        problems.append("the chunker emitted nothing to compare against")
    elif run.chunks[0].text != run.first_emitted:
        problems.append(
            "the first synthesis was not the first emitted chunk: "
            f"{run.chunks[0].text[:60]!r} vs {run.first_emitted[:60]!r}")
    elif len(run.chunks) > 1 and run.chunks[0].text.strip() == run.full_script.strip():
        problems.append(
            "the first synthesis was the whole final script - the run "
            "concatenated before speaking instead of speaking as it went")

    for index, chunk in enumerate(run.chunks):
        if chunk.index != index:
            problems.append(f"chunks arrived out of order at position {index}")
            break

    return problems


def assert_concurrent(run: PipelineRun) -> None:
    problems = concurrency_problems(run)
    if problems:
        raise AssertionError("; ".join(problems))


# --------------------------------------------------------------------------
# what a listener would have experienced
# --------------------------------------------------------------------------
def playback_analysis(run: PipelineRun, gap: float = SENTENCE_GAP) -> dict:
    """Would the audio have played without stalling?

    Playback starts the moment the first chunk is ready and then runs in real
    time. A chunk that finishes synthesising after the listener has arrived at
    it is an underrun - dead air - and is the number that decides whether this
    architecture works at all.
    """
    if not run.chunks:
        return {"chunks": 0, "underruns": [], "stall_seconds": 0.0,
                "kept_ahead": None}

    cursor = run.chunks[0].tts_complete
    underruns, stall = [], 0.0
    for chunk in run.chunks:
        if chunk.tts_complete > cursor + 1e-9:
            short = chunk.tts_complete - cursor
            underruns.append({"index": chunk.index, "stall_seconds": short,
                              "needed_at": cursor,
                              "ready_at": chunk.tts_complete})
            stall += short
            cursor = chunk.tts_complete
        cursor += chunk.audio_seconds + gap

    audio = sum(c.audio_seconds for c in run.chunks)
    return {
        "chunks": len(run.chunks),
        "sentence_gap_seconds": gap,
        "audio_seconds": audio,
        "underruns": underruns,
        "stall_seconds": stall,
        "max_stall_seconds": max((u["stall_seconds"] for u in underruns),
                                 default=0.0),
        "kept_ahead": not underruns,
        "listening_finishes_at": cursor - gap,
        "synthesis_finishes_at": run.chunks[-1].tts_complete,
        "headroom_seconds": (cursor - gap) - run.chunks[-1].tts_complete,
    }


def summarise(run: PipelineRun) -> dict:
    """The numbers, from marks only. A stage never measured stays absent."""
    log = run.log
    audio = sum(c.audio_seconds for c in run.chunks)
    generate = sum(c.generate_seconds for c in run.chunks)
    return {
        "chunks": len(run.chunks),
        "words": sum(c.words for c in run.chunks),
        "audio_seconds": audio,
        "tts_seconds_total": generate,
        "realtime_factor_overall": audio / generate if generate else None,
        "peak_queue_depth": run.peak_queue_depth,
        "backpressure_seconds": run.backpressure_seconds,
        "exa_latency": log.span("exa_start", "exa_complete"),
        "claude_ttft": log.span("claude_start", "claude_ttft"),
        "claude_to_first_chunk": log.span("claude_start",
                                          "first_speakable_chunk_ready"),
        "claude_total": log.span("claude_start", "claude_complete"),
        "first_chunk_tts_seconds": log.span("first_tts_start",
                                            "first_tts_complete"),
        "overlap_seconds": (
            None if log.at("claude_complete") is None
            or log.at("first_tts_start") is None
            else log.at("claude_complete") - log.at("first_tts_start")),
        "search_to_first_listen": log.span("request_start",
                                           "first_playable_audio"),
        "search_to_complete_audio": log.span("request_start",
                                             "final_audio_complete"),
    }
