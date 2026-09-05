"""When could FAM have started speaking, if the rule asked for fewer words?

The production rule is "the first sentence ending that leaves at least 25
words". This watches the same stream for the same rule at 5, 10, 15, 20 and 25,
recording when each threshold's word count is reached, when its sentence
boundary arrives, and what the resulting opening would have been.

Three things make it an observation rather than an experiment:

* **One stream answers all of them.** A boundary with 25 words is also a
  boundary with 5, so the thresholds are monotonic and every one of them occurs
  at or before the 25-word boundary the trial already breaks on. Nothing extra
  is requested and nothing is asked twice.
* **It calls the production function.** Each threshold goes through
  `harness.first_chunk_ready` - the same code the app uses - with a different
  number. A reimplementation could drift; this cannot.
* **The trial still breaks at 25.** The lower thresholds are recorded, never
  acted on, so the run remains the verified control.

The probe runs inside the streaming loop, so its own cost is measured and
reported. If it ever grew expensive enough to move the numbers it is watching,
`probe_seconds` would say so.
"""
from __future__ import annotations

import time
from typing import Optional

#: The thresholds worth asking about. 25 is the production rule and is included
#: so the run re-derives it by the same path as the others - if the probe's 25
#: disagreed with the trial's own first chunk, something is wrong.
DEFAULT_THRESHOLDS = (5, 10, 15, 20, 25)


class ChunkProbe:
    """Records, per threshold, when the words arrived and when a sentence closed."""

    def __init__(self, thresholds=DEFAULT_THRESHOLDS) -> None:
        self.thresholds = tuple(sorted(set(int(t) for t in thresholds)))
        #: threshold -> seconds since the probe's origin
        self.words_at: dict[int, float] = {}
        self.boundary_at: dict[int, float] = {}
        #: threshold -> the opening that would have been spoken
        self.boundary_text: dict[int, str] = {}
        self.probe_seconds = 0.0
        self._origin: Optional[float] = None

    def start(self) -> None:
        self._origin = time.perf_counter()

    def _now(self) -> float:
        return time.perf_counter() - (self._origin or time.perf_counter())

    def observe(self, buffer: str) -> None:
        """Called once per streamed delta. Cheap, and only for what is still open."""
        started = time.perf_counter()
        try:
            # Imported here so this module does not import the harness at load
            # time; the harness imports it.
            from experiments.harness import first_chunk_ready

            word_count = None
            for threshold in self.thresholds:
                if threshold not in self.words_at:
                    if word_count is None:
                        word_count = len(buffer.split())
                    if word_count >= threshold:
                        self.words_at[threshold] = self._now()
                if threshold in self.boundary_at:
                    continue
                # Only look for a boundary once the words are there; before that
                # the rule cannot possibly be satisfied.
                if threshold not in self.words_at:
                    continue
                candidate = first_chunk_ready(buffer, threshold)
                if candidate:
                    self.boundary_at[threshold] = self._now()
                    self.boundary_text[threshold] = candidate
        finally:
            self.probe_seconds += time.perf_counter() - started

    @property
    def complete(self) -> bool:
        return all(t in self.boundary_at for t in self.thresholds)

    def monotonic(self) -> bool:
        """A lower threshold can never close later than a higher one.

        If this is ever False the rule or the probe is wrong, and the report
        says so rather than presenting the numbers as if they held.
        """
        seen = [self.boundary_at[t] for t in self.thresholds if t in self.boundary_at]
        return seen == sorted(seen)

    def metrics(self) -> dict:
        """Flat metrics, one set per threshold, plus the probe's own cost."""
        out: dict = {
            "probe_seconds": self.probe_seconds,
            "probe_monotonic": self.monotonic(),
            "probe_thresholds": list(self.thresholds),
        }
        for threshold in self.thresholds:
            out[f"words_{threshold}_at"] = self.words_at.get(threshold)
            out[f"boundary_{threshold}_at"] = self.boundary_at.get(threshold)
            text = self.boundary_text.get(threshold)
            out[f"boundary_{threshold}_words"] = len(text.split()) if text else None
            out[f"boundary_{threshold}_chars"] = len(text) if text else None
            # What the rule cost at this threshold: words present, sentence not
            # yet closed.
            words_at = self.words_at.get(threshold)
            boundary_at = self.boundary_at.get(threshold)
            out[f"boundary_wait_{threshold}"] = (
                boundary_at - words_at
                if words_at is not None and boundary_at is not None else None
            )
        return out

    def texts(self) -> dict:
        """The candidate openings, for reading rather than timing."""
        return {str(t): self.boundary_text.get(t) for t in self.thresholds}
