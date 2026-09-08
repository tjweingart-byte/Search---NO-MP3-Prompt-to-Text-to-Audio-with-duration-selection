"""One clock across a real episode, so the production path can be measured.

Phase 6 produced its numbers - 2.992s search-to-first-listen, warm, on a 4090 -
from an `EventLog` that lived in the experiment harness. The production path
had none of it, so the same request served to a real listener could report
first-audio and little else.

This is that discipline, kept to what `pipeline` can honestly observe:

* **Marks are recorded, never inferred.** A stage that was not reached has no
  mark, and a span across a missing mark is `None` rather than a plausible
  number. A gap gets investigated; a number gets believed.
* **It measures, it never decides.** Nothing here is read back by the pipeline,
  so instrumentation cannot change what a listener hears. Recording is a
  dictionary write on an already-running request.
* **One origin.** Every mark is seconds since the episode began, so the figures
  add up instead of being several studies to reconcile.

The chunk records are the second half: what actually reached the voice, in
order, with its size. That is how "the first synthesis was one complete
sentence" stops being an architectural claim and becomes an observation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChunkMark:
    """One synthesis call: what was sent, how big, and how long it took."""

    index: int
    sentences: int
    words: int
    characters: int
    started_at: float
    finished_at: float
    audio_seconds: float

    @property
    def generate_seconds(self) -> float:
        return self.finished_at - self.started_at

    @property
    def realtime_factor(self) -> Optional[float]:
        seconds = self.generate_seconds
        return self.audio_seconds / seconds if seconds > 0 else None

    def to_dict(self) -> dict:
        return {"index": self.index, "sentences": self.sentences,
                "words": self.words, "characters": self.characters,
                "started_at": round(self.started_at, 4),
                "generate_seconds": round(self.generate_seconds, 4),
                "audio_seconds": round(self.audio_seconds, 3),
                "realtime_factor": (round(self.realtime_factor, 2)
                                    if self.realtime_factor else None)}


@dataclass
class EpisodeMarks:
    """Named instants on one clock, plus what reached the voice."""

    started: float = field(default_factory=time.perf_counter)
    events: dict = field(default_factory=dict)
    chunks: list = field(default_factory=list)
    #: Items waiting in the sentence queue when Claude finished. Greater than
    #: zero means synthesis was behind and Claude finished anyway - the
    #: decoupling, observed rather than asserted.
    backlog_at_claude_complete: Optional[int] = None

    def mark(self, name: str) -> float:
        """Record `name` the first time it happens. Later calls are ignored.

        First-time-only on purpose: every mark here names a *first* - the first
        token, the first sentence, the first synthesis - and a later sentence
        overwriting them would quietly turn them into lasts.
        """
        at = time.perf_counter() - self.started
        self.events.setdefault(name, at)
        return at

    def at(self, name: str) -> Optional[float]:
        return self.events.get(name)

    def span(self, start: str, end: str) -> Optional[float]:
        first, last = self.at(start), self.at(end)
        return None if first is None or last is None else last - first

    def add_chunk(self, text: str, sentences: int, started_at: float,
                  finished_at: float, audio_seconds: float) -> ChunkMark:
        chunk = ChunkMark(index=len(self.chunks), sentences=sentences,
                          words=len(text.split()), characters=len(text),
                          started_at=started_at, finished_at=finished_at,
                          audio_seconds=audio_seconds)
        self.chunks.append(chunk)
        return chunk

    # ---- what the report is for -----------------------------------------
    @property
    def first_chunk(self) -> Optional[ChunkMark]:
        return self.chunks[0] if self.chunks else None

    def summary(self) -> dict:
        """The measurements, from marks only. Absent beats guessed."""
        first = self.first_chunk
        return {
            "claude_ttft": self.span("claude_start", "claude_first_token"),
            "claude_to_first_sentence": self.span("claude_start",
                                                  "first_sentence"),
            "first_sentence_to_synthesis": self.span("first_sentence",
                                                     "first_tts_start"),
            "first_synthesis_seconds": self.span("first_tts_start",
                                                 "first_tts_complete"),
            "first_pcm": self.at("first_tts_complete"),
            # Absent on a truncated episode: Claude did not finish, we
            # stopped it. `speaking_total` is the one that always exists.
            "claude_total": self.span("claude_start", "claude_complete"),
            "speaking_total": self.span("claude_start", "speaking_complete"),
            "search_to_first_pcm": self.at("first_tts_complete"),
            # The first-chunk invariant, as an observation rather than a claim.
            "first_chunk_sentences": first.sentences if first else None,
            "first_chunk_words": first.words if first else None,
            "chunks": len(self.chunks),
            "chunk_words": [c.words for c in self.chunks],
            "backlog_at_claude_complete": self.backlog_at_claude_complete,
            "claude_decoupled": (None if self.backlog_at_claude_complete is None
                                 else self.backlog_at_claude_complete > 0),
        }

    def to_dict(self) -> dict:
        return {"events": {k: round(v, 4) for k, v in self.events.items()},
                "chunks": [c.to_dict() for c in self.chunks],
                "summary": {k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in self.summary().items()}}


class TimedStream:
    """Production's model stream, with one mark on the first text delta.

    `stream_sentences` yields sentences, so time-to-first-*token* is invisible
    from outside it. Rather than reimplement the chunker to see it - which
    would measure a copy of production instead of production - the client is
    wrapped and the real generator is left exactly as it is. This is the same
    wrapper Phase 6 measured with.
    """

    def __init__(self, inner, marks: EpisodeMarks):
        self._inner, self._marks = inner, marks

    @property
    def text_stream(self):
        async def deltas():
            async for delta in self._inner.text_stream:
                self._marks.mark("claude_first_token")
                yield delta
        return deltas()

    async def get_final_message(self):
        return await self._inner.get_final_message()

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _TimedContext:
    def __init__(self, inner, marks):
        self._inner, self._marks = inner, marks

    async def __aenter__(self):
        return TimedStream(await self._inner.__aenter__(), self._marks)

    async def __aexit__(self, *exc):
        return await self._inner.__aexit__(*exc)


class _TimedMessages:
    def __init__(self, inner, marks):
        self._inner, self._marks = inner, marks

    def stream(self, **kwargs):
        return _TimedContext(self._inner.stream(**kwargs), self._marks)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class TimedClient:
    """Wraps an Anthropic client so the first delta is marked. Nothing else."""

    def __init__(self, inner, marks: EpisodeMarks):
        self._inner = inner
        self.messages = _TimedMessages(inner.messages, marks)

    def __getattr__(self, name):
        return getattr(self._inner, name)
