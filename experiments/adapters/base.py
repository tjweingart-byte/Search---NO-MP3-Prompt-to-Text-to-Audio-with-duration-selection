"""What every adapter must be, and what it must say when it cannot run.

Two rules hold for all of them.

**Unavailable is a sentence, not an exception.** `available()` returns a reason
a person can act on - "EXA_API_KEY is not set", "no Chatterbox endpoint is
configured" - so the planner can refuse a run *before* spending anything,
rather than failing on trial 7 of 20.

**Nothing here provisions anything.** An adapter may talk to infrastructure
that already exists. It may not create it, start it, or cause it to bill. An
adapter that needs infrastructure which is not running says so and stops.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from experiments.timeline import Timeline


class InfrastructureRequired(RuntimeError):
    """Raised when an experiment needs infrastructure that is not running.

    Carries what is needed and what it would cost, because the answer to this
    is always a human decision. The harness never resolves it by provisioning.
    """

    def __init__(self, adapter: str, what: str, how: str = "") -> None:
        self.adapter = adapter
        self.what = what
        self.how = how
        message = f"{adapter}: {what}"
        if how:
            message += f"\n  {how}"
        super().__init__(message)


@dataclass
class Availability:
    """Whether an adapter can run here, and if not, why not."""

    ok: bool
    reason: str = ""
    #: True when the *only* thing missing is paid infrastructure a human must
    #: authorise. The planner reports these separately from plain misconfiguration.
    needs_approval: bool = False
    #: What to do about it, in one line.
    remedy: str = ""

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "needs_approval": self.needs_approval,
            "remedy": self.remedy,
        }


@dataclass
class SearchResult:
    """What a retrieval stage produced."""

    #: Text to hand the model as context. Empty for server-side search, where
    #: retrieval is not separable from generation.
    context: str = ""
    #: Distinct domains, for a person to judge. Never scored by machine.
    sources: list[str] = field(default_factory=list)
    #: How many searches actually ran, against whatever cap was set.
    searches: int = 0
    #: Estimated dollars for this retrieval alone.
    cost: float = 0.0
    #: What the service said it spent, if it says.
    remote_seconds: Optional[float] = None
    detail: dict = field(default_factory=dict)


@dataclass
class SynthResult:
    """What a speech stage produced."""

    #: Raw PCM. No files, no MP3 - the product constraint holds in the lab too.
    pcm: bytes = b""
    sample_rate: int = 22050
    #: Seconds of audio produced.
    audio_seconds: float = 0.0
    cost: float = 0.0
    remote_seconds: Optional[float] = None
    detail: dict = field(default_factory=dict)

    @property
    def realtime_factor(self) -> Optional[float]:
        """How many seconds of audio per second of compute.

        Piper does about 330x. A voice below 1x cannot ship at any price, and
        that is invisible to a listening test.
        """
        wall = self.detail.get("wall_seconds")
        if not wall:
            return None
        return self.audio_seconds / wall if wall else None


@runtime_checkable
class SearchAdapter(Protocol):
    id: str
    label: str

    def available(self) -> Availability: ...

    async def search(self, query: str, timeline: Timeline, **params) -> SearchResult: ...


@runtime_checkable
class TTSAdapter(Protocol):
    id: str
    label: str

    def available(self) -> Availability: ...

    async def synth(self, text: str, timeline: Timeline, **params) -> SynthResult: ...
