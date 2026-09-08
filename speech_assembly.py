"""Sentences in, speech-sized chunks out. Nothing else.

`script_generator.stream_sentences` yields one sentence at a time, and
`pipeline.py` currently sends each of them to the voice on its own. On a slow
engine that is one synthesis call per fragment - a two-word sentence becomes a
two-word invocation - which fragments delivery and multiplies queue pressure.

This decides *how many* of those sentences travel together. It is deliberately
not a second text-processing system: the sentence boundary, `clean_for_speech`
and the `<<NEXT:` hold-back all stay in `script_generator`, so the spoken text
is exactly the concatenation of the sentences production already produces.
Nothing here splits a sentence, reorders one, or rewrites a word.

## The first chunk has no size rule at all

**Time to first listen is the highest-priority metric**, so the first complete,
speakable thought goes to the voice on the offer that produced it - at five
words or at thirty-five. The minimum, target and cap below apply only *after*
that first release. There is deliberately no first-chunk word knob, not even
one defaulting to zero: a knob left behind is an invitation to raise it.

The one gate on the opening is semantic rather than dimensional: the text must
end on terminal punctuation, the same shape `script_generator._SENTENCE_END`
looks for. A half-written clause is not a speakable thought. In practice
`stream_sentences` only ever yields complete sentences, so this fires
approximately never; it exists for the degenerate stream, and even then the
wait timer and the end-of-script flush still release the text rather than
holding it.

### The trade that choice makes, stated rather than designed out

One synthesis worker means chunk 2 is produced while chunk 1 plays, so
`audio(chunk 1) >= generate(chunk 2)` or playback stalls on the very first
handoff. At `settings.target_wpm` the break-even is narrow - roughly six words
against a target-sized follower, eleven against the largest allowed - so the
band where a short opening actually stalls is small. It is not zero. The
caller measures it rather than this module preventing it.

## Where the later thresholds come from

Measured Chatterbox Base on an RTX 4090 (28 words / 2.181s, 33 / 2.848s,
41 / 3.610s): least squares gives 0.1086 s/word with a **negative** intercept,
R2 0.991. No measurable fixed cost per invocation across that range, and cost
per word rises slightly with length - so batching does not buy throughput. It
buys unfragmented delivery and fewer queue events. The fit must not be
extrapolated below 28 words, where it goes negative and nothing was measured.

Validated in the Phase 6 RTX 4090 run (`phase6_4090_20260908T064113Z`):
22 sentences became 15 synthesis calls, median 33 words, zero playback stalls.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from config import settings
from script_generator import count_words

#: A complete thought ends on terminal punctuation, optionally inside a closing
#: quote or bracket. The character class is `script_generator._SENTENCE_END`'s -
#: the same boundary, asked as a question about the end of a string rather than
#: used to split one.
_COMPLETE_THOUGHT = re.compile(r"""[.!?]["')\]]*$""")


def audio_seconds_for(words: int, wpm: float | None = None) -> float:
    """How long `words` will take to speak, at the planned rate."""
    rate = float(wpm if wpm is not None else settings.target_wpm)
    return words * 60.0 / rate if rate else 0.0


@dataclass(frozen=True)
class AssemblyPolicy:
    """Every number here is a decision with a reason. See the module docstring."""

    #: Below this a chunk is a fragment; above it a natural beat may end one.
    min_words: int = 18
    #: Where later chunks aim. Near the bottom of the measured range, because
    #: cost per word rises with length rather than falling.
    target_words: int = 28
    #: Hard cap, just past the top of the measured range. Beyond 41 words
    #: nothing has been measured, so the cap stays near the evidence.
    max_words: int = 45
    #: Text is never held longer than this, whatever size it has reached.
    max_wait_seconds: float = 2.0
    #: Playback headroom below which batching stops mattering and the pending
    #: text ships immediately, however short.
    headroom_floor_seconds: float = 3.0
    #: Natural beats. A chunk past `min_words` ends here by preference.
    prefer_break_after: tuple = ("?", "!")
    #: The opening chunk's only gate, and it is about meaning, not size.
    first_chunk_needs_terminal_punctuation: bool = True

    def __post_init__(self) -> None:
        if not (0 < self.min_words <= self.target_words <= self.max_words):
            raise ValueError(
                "policy must satisfy 0 < min <= target <= max; got "
                f"{self.min_words}, {self.target_words}, {self.max_words}")


@dataclass
class AssembledChunk:
    """One synthesis payload, and why the assembler let it go."""

    index: int
    text: str
    #: The sentences this chunk was built from, in order. Kept because the
    #: duration fit works at sentence boundaries and because `stats.script`
    #: stores sentences - joining them here and splitting them again later
    #: would be a second text-processing system, and a lossy one.
    parts: list
    sentences: int
    words: int
    characters: int
    ready_at: float
    reason: str
    first_sentence_at: float

    @property
    def held_seconds(self) -> float:
        """How long the oldest sentence in this chunk waited to be spoken for."""
        return self.ready_at - self.first_sentence_at

    def to_dict(self) -> dict:
        return {"index": self.index, "text": self.text, "parts": list(self.parts),
                "sentences": self.sentences, "words": self.words,
                "characters": self.characters, "ready_at": self.ready_at,
                "reason": self.reason, "held_seconds": self.held_seconds,
                "audio_seconds_planned": audio_seconds_for(self.words)}


@dataclass
class BudgetFit:
    """What of an assembled chunk still fits the episode's remaining time."""

    #: The sentences that may be spoken, in order. Possibly empty.
    spoken: list
    #: What did not fit, in order. `spoken + remainder` is always the input.
    remainder: list
    #: True when something was dropped - the episode ends here, as in legacy.
    truncated: bool
    #: Estimated seconds the spoken half will take, by the same formula
    #: `pipeline._speak_one` uses.
    estimated_seconds: float

    @property
    def text(self) -> str:
        return " ".join(self.spoken)

    @property
    def words(self) -> int:
        return count_words(self.text)


def fit_to_budget(sentences: list, remaining_seconds: float, wpm: float,
                  gap: float, grace: float) -> BudgetFit:
    """The largest run of complete sentences that legacy would have spoken.

    Batching creates a decision legacy never had to make. `pipeline._speak_one`
    asks the question once per sentence:

        estimated = words / (wpm / 60) + SENTENCE_GAP
        if estimated > pace.remaining_seconds + OVERRUN_GRACE: stop

    A chunk is several sentences in one synthesis call, so the same question has
    to be asked *before* the call, for each sentence in turn. Accepting or
    rejecting the whole chunk would turn the allowed overshoot into
    `OVERRUN_GRACE + one chunk`, or throw away sentences that fit. Neither is
    the contract. So the chunk is cut at the last sentence boundary that fits,
    and the rest is handed back.

    Only the final kept sentence can cross the budget, and by at most `grace` -
    exactly as in legacy. Sentences are never split: a boundary is the smallest
    unit here, because cutting inside one damages meaning and because cutting at
    a sentence boundary is why the pipeline works in sentences at all.

    `gap` and `grace` are required rather than defaulted: they belong to
    `pipeline` (`SENTENCE_GAP`, `OVERRUN_GRACE`) and a copy of them here would
    be a second source of truth for the duration contract.

    Note what this cannot do that legacy can. Legacy re-plans the speaking rate
    between sentences via `PaceController.next_wpm`, so it may speed up as time
    runs short and fit one more sentence. A chunk is one synthesis call at one
    rate, so this walks at a fixed `wpm`. The effect is conservative - it fits
    the same or fewer words, never more - which is the safe direction for a
    duration contract.
    """
    budget = max(0.0, remaining_seconds)
    per_second = max(wpm, 1.0) / 60.0
    spoken: list = []
    for index, sentence in enumerate(sentences):
        estimated = count_words(sentence) / per_second + gap
        if estimated > budget + grace:
            return BudgetFit(spoken=spoken, remainder=list(sentences[index:]),
                             truncated=True,
                             estimated_seconds=_estimate(spoken, per_second, gap))
        spoken.append(sentence)
        budget = max(0.0, budget - estimated)
    return BudgetFit(spoken=spoken, remainder=[], truncated=False,
                     estimated_seconds=_estimate(spoken, per_second, gap))


def _estimate(sentences: list, per_second: float, gap: float) -> float:
    return sum(count_words(s) / per_second + gap for s in sentences)


@dataclass
class SpeechAssembler:
    """Whole sentences in, speech-sized chunks out. Never splits, never reorders.

    `offer()` takes each sentence as `stream_sentences` produces it and returns
    the chunks that became ready. `due()` is the timer and headroom path, called
    when no sentence arrived, so text is never held indefinitely. `flush()` ends
    the stream and releases whatever is left, however short.
    """

    policy: AssemblyPolicy = field(default_factory=AssemblyPolicy)
    #: Injected so tests need not sleep and a caller can share its own clock.
    clock: Callable[[], float] = time.monotonic
    pending: list = field(default_factory=list)
    released: int = 0
    first_pending_at: Optional[float] = None
    #: Every sentence accepted, in order - the text-integrity witness.
    seen: list = field(default_factory=list)

    # ---- state -----------------------------------------------------------
    @property
    def pending_words(self) -> int:
        return sum(count_words(s) for s in self.pending)

    @property
    def pending_characters(self) -> int:
        return sum(len(s) for s in self.pending)

    def _floor(self) -> int:
        """The size below which a chunk will not be shipped early.

        One for the opening: it has no size rule, so anything speakable goes
        now rather than being merged past the cap with what follows.
        """
        return 1 if self.released == 0 else self.policy.min_words

    def _is_complete_thought(self, text: str) -> bool:
        text = text.strip()
        if not text.split():
            return False
        if not self.policy.first_chunk_needs_terminal_punctuation:
            return True
        return bool(_COMPLETE_THOUGHT.search(text))

    # ---- the decision ----------------------------------------------------
    def _reason_to_release(self, headroom: Optional[float]) -> Optional[str]:
        if not self.pending:
            return None
        words = self.pending_words
        policy = self.policy

        if self.released == 0:
            # The latency path. No floor, no target, no preferred size: the
            # first complete speakable thought goes now. The only question
            # asked is whether it is complete.
            if self._is_complete_thought(" ".join(self.pending)):
                return "first chunk, released on the first complete thought"
            # Not complete. Batching rules still do not apply to the opening,
            # but the safety rules must, or a degenerate stream could hold the
            # first chunk forever.
            if headroom is not None and headroom <= policy.headroom_floor_seconds:
                return f"first chunk, headroom {headroom:.2f}s at the floor"
            if (self.first_pending_at is not None
                    and self.clock() - self.first_pending_at
                    >= policy.max_wait_seconds):
                return "first chunk, held long enough without a complete thought"
            return None

        if words >= policy.max_words:
            return "at the cap"
        if headroom is not None and headroom <= policy.headroom_floor_seconds:
            # Batching stops mattering when the listener is about to catch up.
            return f"headroom {headroom:.2f}s at or below the floor"
        if words >= policy.target_words:
            return "target reached"
        if words >= policy.min_words and self.pending[-1].rstrip().endswith(
                tuple(policy.prefer_break_after)):
            return "natural beat past the minimum"
        if (self.first_pending_at is not None
                and self.clock() - self.first_pending_at >= policy.max_wait_seconds):
            return "held long enough"
        return None

    def _emit(self, reason: str) -> AssembledChunk:
        text = " ".join(self.pending)
        now = self.clock()
        chunk = AssembledChunk(
            index=self.released, text=text, parts=list(self.pending),
            sentences=len(self.pending),
            words=count_words(text), characters=len(text), ready_at=now,
            reason=reason,
            first_sentence_at=self.first_pending_at
            if self.first_pending_at is not None else now)
        self.pending = []
        self.first_pending_at = None
        self.released += 1
        return chunk

    # ---- the interface ---------------------------------------------------
    def offer(self, sentence: str, headroom: Optional[float] = None) -> list:
        """Accept one sentence; return the chunks that became ready (0, 1 or 2)."""
        sentence = (sentence or "").strip()
        if not sentence:
            return []
        self.seen.append(sentence)
        out = []

        # Adding this sentence would blow the cap and what is pending can stand
        # alone: ship it first rather than merging past a size nothing has been
        # measured at.
        if (self.pending
                and self.pending_words + count_words(sentence) > self.policy.max_words
                and self.pending_words >= self._floor()):
            out.append(self._emit("cap would be exceeded by the next sentence"))

        if not self.pending:
            self.first_pending_at = self.clock()
        self.pending.append(sentence)

        reason = self._reason_to_release(headroom)
        if reason:
            out.append(self._emit(reason))
        return out

    def due(self, headroom: Optional[float] = None) -> list:
        """The timer and headroom path, with no new sentence."""
        reason = self._reason_to_release(headroom)
        return [self._emit(reason)] if reason else []

    def flush(self) -> list:
        """End of stream. A short final fragment ships; nothing is ever held."""
        return [self._emit("end of script")] if self.pending else []

    # ---- the witness -----------------------------------------------------
    def spoken_matches_source(self, chunks: list) -> Optional[str]:
        """None if the chunks are exactly the sentences offered, in order.

        No text lost, duplicated, reordered or spoken twice between the chunker
        and the voice. Compared on whitespace-normalised words, because joining
        sentences inserts a single space and nothing else.
        """
        spoken = " ".join(c.text for c in chunks).split()
        source = " ".join(self.seen).split()
        if spoken == source:
            return None
        if len(spoken) < len(source):
            return f"text was lost: {len(source)} words in, {len(spoken)} out"
        if len(spoken) > len(source):
            return (f"text was duplicated: {len(source)} words in, "
                    f"{len(spoken)} out")
        for index, (a, b) in enumerate(zip(spoken, source)):
            if a != b:
                return (f"text diverged at word {index}: spoken {a!r}, "
                        f"source {b!r} - reordered or rewritten")
        return "text differs from the source"
