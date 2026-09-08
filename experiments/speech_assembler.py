"""Adaptive speech packetisation: sentences in, speech-sized chunks out.

Phase 5 sent every sentence production emitted straight to Chatterbox, so a
two-word sentence became its own synthesis call. This assembles them instead -
in order, whole sentences only, never splitting one and never reordering.

**It is not a second text-processing system.** Production's
`script_generator.stream_sentences` has already done the semantic work:
sentence boundaries that keep quotes and brackets attached, `clean_for_speech`,
and the hold-back of the `<<NEXT:` marker. This only decides *how many* of
those sentences travel together, so the spoken text is byte-identical to what
production would have spoken - which is assertion 6.

## Where the numbers come from

The 4090 curve (`experiments/results/chatterbox_mps_vs_4090/ANALYSIS.md`), the
only measured Chatterbox-on-4090 data in this repository:

| words | generate |
|---|---|
| 28 | 2.181s |
| 33 | 2.848s |
| 41 | 3.610s |

Least squares over those three: **0.1086 s/word, intercept -0.81s, R2 0.991**.

A *negative* intercept means there is **no measurable fixed per-invocation
overhead between 28 and 41 words** - cost is essentially proportional to
length, and cost per word actually rises slightly (78 -> 88 ms/word). So
batching does **not** buy compute efficiency in the measured range. The case
for it is delivery quality, fewer pathological calls, and less queue churn -
not throughput. That is worth saying plainly, because the opposite is the
intuitive assumption.

**The fit must not be extrapolated below 28 words** - it goes negative at 7 -
and there is no 4090 measurement of a tiny chunk anywhere in this repository,
because the benchmark corpus was built with a 25-word floor. Small-chunk cost
is an open question, and `tools/fit_chunk_policy.py` answers it from a real run.

## The first chunk is a latency path, not a batch

**No word floor. None.** The first complete, speakable thought
`stream_sentences` produces goes to Chatterbox immediately, whether it is
eight words or twenty. Time to first listen is the highest-priority metric and
nothing about batching outranks it. The assembler's minimum, target and cap
apply only *after* that first release.

The one safeguard kept is semantic, not dimensional: the pending text must end
on terminal punctuation - `.`, `!` or `?`, optionally followed by a closing
quote or bracket, the same shape production's `_SENTENCE_END` looks for. A
half-written clause is not a speakable thought, and shipping one to save
milliseconds would buy a fragment. In practice `stream_sentences` only ever
yields complete sentences, so this fires approximately never; it exists for the
degenerate stream, and even then the wait timer and end-of-script flush still
release the text rather than holding it.

### The risk this accepts, and how it is reported rather than prevented

An earlier draft held the first chunk to twelve words, derived from a real
constraint: one TTS worker means chunk 2 is synthesised while chunk 1 plays, so

    audio(chunk 1)  >=  generate(chunk 2)

At `TARGET_WPM = 150` the break-even is about **5.6 words** against a
target-sized follower (28 words, ~2.23s to synthesise) and about **10.2 words**
against the largest one allowed (45 words, ~4.07s). So the band where a short
opening actually stalls the first handoff is narrow - it takes an opening under
roughly six words, or under eleven if the second chunk is at the cap. Worth
knowing before treating the removed floor as a large risk: mostly it was
insurance against a case that rarely arises.

That floor is gone, deliberately. The constraint has not gone with it, so the
run **measures** it: `playback_report()["first_handoff"]` reports the opening
chunk's audio duration against the second chunk's generation time, and the
executive summary flags it when the cover was not there. A risk that is
measured and named is a decision; a risk that is silently designed out is a
different product.

Read against Phase 5: its first chunk took 2.757s to synthesise, which the
curve puts at about 33 words. Nothing here would have changed it, so this
policy still predicts **no change to Phase 5's first-listen latency** - and now
it cannot lengthen it either, because there is nothing left to wait for.
"""
from __future__ import annotations

import time
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

#: A complete thought ends on terminal punctuation, optionally inside a closing
#: quote or bracket. The character class is production's, from
#: `script_generator._SENTENCE_END` - the same boundary, asked as a question
#: about the end of a string rather than used to split one.
_COMPLETE_THOUGHT = re.compile(r"""[.!?]["')\]]*$""")

#: FAM's planned speaking rate, from `config.settings.target_wpm`. Imported
#: rather than copied where the app is importable.
try:  # pragma: no cover - the app's number when it is available
    from config import settings as _settings

    TARGET_WPM = float(_settings.target_wpm)
except Exception:  # pragma: no cover
    TARGET_WPM = 150.0


def audio_seconds_for(words: int, wpm: float = TARGET_WPM) -> float:
    """How long `words` will take to speak, at the planned rate."""
    return words * 60.0 / wpm if wpm else 0.0


@dataclass(frozen=True)
class AssemblyPolicy:
    """Every number here is a decision with a reason. See the module docstring.

    Defaults are the evidence-bounded ones; `tools/fit_chunk_policy.py` re-fits
    them against a real run rather than leaving them as somebody's guess.
    """

    #: The opening chunk has no size rule at all - see the module docstring.
    #: Its only gate is that the text is a complete thought, and that gate is
    #: semantic. There is deliberately no word knob here: a floor left behind
    #: with a default of zero is an invitation to raise it again.
    first_chunk_needs_terminal_punctuation: bool = True
    #: Below this, a chunk is a fragment. Above it, a boundary may end a chunk.
    min_words: int = 18
    #: Where later chunks aim. Near the bottom of the measured range, because
    #: cost per word rises with length rather than falling.
    target_words: int = 28
    #: Hard cap. The top of the measured range; beyond 41 words nothing has
    #: been measured on a 4090, so the cap stays inside the evidence.
    max_words: int = 45
    #: Text is never held longer than this, whatever its size.
    max_wait_seconds: float = 2.0
    #: Projected playback headroom below which batching stops mattering and
    #: the pending text ships immediately, however short.
    headroom_floor_seconds: float = 3.0
    #: Natural beats. A chunk already past `min_words` ends here by preference.
    prefer_break_after: tuple = ("?", "!")

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
    sentences: int
    words: int
    characters: int
    ready_at: float
    reason: str
    first_sentence_at: float

    #: How long the oldest sentence in this chunk waited to be spoken for.
    @property
    def held_seconds(self) -> float:
        return self.ready_at - self.first_sentence_at

    def to_dict(self) -> dict:
        return {"index": self.index, "text": self.text,
                "sentences": self.sentences, "words": self.words,
                "characters": self.characters, "ready_at": self.ready_at,
                "reason": self.reason, "held_seconds": self.held_seconds,
                "audio_seconds_planned": audio_seconds_for(self.words)}


def count_words(text: str) -> int:
    return len(text.split())


@dataclass
class SpeechAssembler:
    """Whole sentences in, speech-sized chunks out. Never splits, never reorders.

    `offer()` is called with each sentence as production emits it and returns
    the chunks that became ready. `due()` is the timer and headroom path -
    called even when no sentence arrived, so text is never held indefinitely.
    `flush()` ends the stream and releases whatever is left, however short.
    """

    policy: AssemblyPolicy = field(default_factory=AssemblyPolicy)
    #: Injected so tests do not sleep and the runner shares the run's clock.
    clock: Callable[[], float] = time.perf_counter
    pending: list = field(default_factory=list)
    released: int = 0
    first_pending_at: Optional[float] = None
    #: Every sentence ever accepted, in order - the text-integrity witness.
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

        One for the opening chunk: it has no size rule, so anything speakable
        goes now rather than being merged past the cap with what follows.
        """
        return 1 if self.released == 0 else self.policy.min_words

    def _is_complete_thought(self, text: str) -> bool:
        """Ends on terminal punctuation, and has something in it.

        The only gate on the first chunk, and it is about meaning rather than
        size: a half-written clause is not a speakable thought.
        """
        text = text.strip()
        if not text.split():
            return False
        return bool(_COMPLETE_THOUGHT.search(text)
                    or not self.policy.first_chunk_needs_terminal_punctuation)

    # ---- the decision ----------------------------------------------------
    def _reason_to_release(self, headroom: Optional[float]) -> Optional[str]:
        if not self.pending:
            return None
        words = self.pending_words
        policy = self.policy

        if self.released == 0:
            # The latency path. No word floor, no target, no preferred size:
            # the first complete speakable thought goes now, at eight words or
            # at twenty. The only question asked is whether it is complete.
            if self._is_complete_thought(" ".join(self.pending)):
                return "first chunk, released on the first complete thought"
            # Not complete. Batching rules still do not apply to the opening,
            # but the safety rules below must, or a degenerate stream could
            # hold the first chunk forever.
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
        chunk = AssembledChunk(
            index=self.released, text=text, sentences=len(self.pending),
            words=count_words(text), characters=len(text),
            ready_at=self.clock(), reason=reason,
            first_sentence_at=self.first_pending_at
            if self.first_pending_at is not None else self.clock())
        self.pending = []
        self.first_pending_at = None
        self.released += 1
        return chunk

    # ---- the interface ---------------------------------------------------
    def offer(self, sentence: str, headroom: Optional[float] = None) -> list:
        """Accept one sentence; return the chunks that became ready (0, 1 or 2)."""
        sentence = sentence.strip()
        if not sentence:
            return []
        self.seen.append(sentence)
        out = []

        # Adding this sentence would blow the cap, and what is already pending
        # is big enough to stand alone: ship it first rather than merging past
        # a size nothing has been measured at.
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

        Assertions 5 and 6: no text lost, duplicated, reordered or spoken
        twice between the chunker and the voice. Compared on whitespace-
        normalised words, because joining sentences inserts a single space and
        nothing else.
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
