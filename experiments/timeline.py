"""Where did the time go, across however many machines it went to.

A FAM pipeline experiment may touch three services that share no clock: Exa's
API, Anthropic's API, and a Chatterbox GPU somewhere else again. Comparing
their internal timestamps would be comparing three different wall clocks, so
this does not try.

Everything here is measured on **one** clock - the orchestrator's
`perf_counter` - and every stage is the interval this machine waited. That
number is what a listener actually experiences, and it is the only one that is
sound to add up.

A remote stage may additionally *report* how long it believed it spent
(`remote_ms`). That is recorded beside the measured wall time and never
substituted for it, which makes the difference between them visible:

    stage        wall    remote   network+queue
    chatterbox   820ms   540ms    280ms

That third column is the cost of the stage being on another machine, and it is
the thing a single-machine benchmark cannot see.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Iterator, Optional

#: Where a stage ran. Free-form, but keep it stable across trials or the
#: comparison between arms becomes noise.
LOCAL = "local"


@dataclass
class Stage:
    """One measured interval on the orchestrator's clock."""

    name: str
    #: Seconds from the start of the trial to the moment this stage began.
    start: float
    #: Seconds from the start of the trial to the moment it ended.
    end: float
    #: Which machine or service did the work.
    host: str = LOCAL
    #: What the remote side said it spent, in seconds. None for local stages
    #: and for remotes that do not report. Never replaces `duration`.
    remote_seconds: Optional[float] = None
    #: Anything worth keeping: bytes returned, token counts, result counts.
    detail: dict = field(default_factory=dict)
    #: Set when the stage raised. A failed stage is still a recorded stage.
    error: Optional[str] = None

    @property
    def duration(self) -> float:
        """Wall time this machine waited for the stage."""
        return self.end - self.start

    @property
    def overhead(self) -> Optional[float]:
        """Wall time minus what the remote claimed: network, queueing, retries."""
        if self.remote_seconds is None:
            return None
        return self.duration - self.remote_seconds

    def to_dict(self) -> dict:
        out = asdict(self)
        out["duration"] = round(self.duration, 6)
        if self.overhead is not None:
            out["overhead"] = round(self.overhead, 6)
        return out


class Timeline:
    """Records ordered, named stages for a single trial.

    Stages may be nested or overlapping (a pipeline that starts synthesising
    the first chunk while the model is still writing is exactly the overlap
    this product is built on), so the summary reports each stage's own
    duration and does not assume they tile the total.
    """

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self.stages: list[Stage] = []
        self._marks: dict[str, float] = {}

    def _now(self) -> float:
        return time.perf_counter() - self._t0

    @contextmanager
    def span(
        self,
        name: str,
        host: str = LOCAL,
        **detail,
    ) -> Iterator[Stage]:
        """Time a block. The stage is recorded even if the block raises.

        The yielded Stage is writable, so the body can attach `remote_seconds`
        or extra detail once it knows them::

            with tl.span("chatterbox", host="runpod") as st:
                reply = await post(...)
                st.remote_seconds = reply["gpu_seconds"]
        """
        stage = Stage(name=name, start=self._now(), end=self._now(), host=host, detail=dict(detail))
        self.stages.append(stage)
        try:
            yield stage
        except Exception as exc:
            stage.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            stage.end = self._now()

    def mark(self, name: str) -> float:
        """Record an instant - "the first speakable chunk existed at t".

        Marks are the checkpoints between stages. `since` turns two of them
        into a duration without inventing a span that wraps them.
        """
        at = self._now()
        self._marks[name] = at
        return at

    def since(self, name: str) -> Optional[float]:
        """Seconds elapsed since a mark, or None if it was never made."""
        if name not in self._marks:
            return None
        return self._now() - self._marks[name]

    def at(self, name: str) -> Optional[float]:
        """When a mark happened, relative to the start of the trial."""
        return self._marks.get(name)

    @property
    def elapsed(self) -> float:
        return self._now()

    def bottleneck(self) -> Optional[Stage]:
        """The stage that took longest. The headline answer, for one trial.

        Across trials the harness takes the median per stage instead, because
        one trial's slowest stage is not evidence of anything.
        """
        if not self.stages:
            return None
        return max(self.stages, key=lambda s: s.duration)

    def to_dict(self) -> dict:
        return {
            "elapsed": round(self.elapsed, 6),
            "stages": [s.to_dict() for s in self.stages],
            "marks": {k: round(v, 6) for k, v in self._marks.items()},
        }
