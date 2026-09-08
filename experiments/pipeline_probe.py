"""One clock across the whole FAM request, from typing to sound.

Every earlier experiment measured one stage. This one measures the thing the
listener actually experiences - Exa, then Claude, then Chatterbox - on a single
monotonic clock, so the numbers add up instead of being three separate studies
that have to be reconciled by hand.

**Marks are recorded, never inferred.** A stage that cannot be observed is
written down as unavailable with the reason, because a plausible number in a
waterfall is worse than a gap: the gap gets investigated and the number gets
believed. Two are known unavailable before the run:

* **first Exa result.** `exa_impl` makes one blocking `search_and_contents`
  call, so there is no intermediate point between request and complete.
* **first playable audio, mid-generation.** Chatterbox Base is one-shot
  (`tts.py:208-271`, no yield anywhere in the package), so the first playable
  moment *is* generation-complete for that chunk. This was established in
  Phase 1 and is the finding, not a limitation of the probe.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EventLog:
    """Named marks on one monotonic clock, plus what could not be marked."""

    started: float = field(default_factory=time.perf_counter)
    wall_started: float = field(default_factory=time.time)
    events: list = field(default_factory=list)
    unavailable: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def mark(self, name: str, detail: Optional[dict] = None) -> float:
        at = time.perf_counter() - self.started
        self.events.append({"name": name, "at": at, "detail": detail or {}})
        return at

    def now(self) -> float:
        """The clock, without leaving a mark.

        For sampling something continuous - queue depth - where one named event
        per sample would bury the named ones it sits between.
        """
        return time.perf_counter() - self.started

    def cannot(self, name: str, why: str) -> None:
        """Declare a stage unmeasurable, with the reason, before anyone asks."""
        self.unavailable[name] = why

    def at(self, name: str) -> Optional[float]:
        for event in self.events:
            if event["name"] == name:
                return event["at"]
        return None

    def span(self, start: str, end: str) -> Optional[float]:
        first, last = self.at(start), self.at(end)
        return None if first is None or last is None else last - first

    def as_dict(self) -> dict:
        return {
            "wall_started": self.wall_started,
            "events": self.events,
            "unavailable": self.unavailable,
            "notes": self.notes,
        }


def gpu_memory() -> dict:
    """What the card is holding, if torch will say. Absent beats guessed."""
    out: dict = {}
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}
        out["available"] = True
        out["allocated_mb"] = torch.cuda.memory_allocated() / 1024 ** 2
        out["reserved_mb"] = torch.cuda.memory_reserved() / 1024 ** 2
        out["max_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024 ** 2
        out["name"] = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        out["free_mb"] = free / 1024 ** 2
        out["total_mb"] = total / 1024 ** 2
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def waterfall(log: EventLog, segments: list) -> list:
    """Rows of (label, seconds, share) - where the wait actually goes.

    Built from spans between recorded marks only. A segment whose endpoints
    were not both marked appears with `seconds: None` and is excluded from the
    share, rather than being quietly dropped or filled in.
    """
    rows, measured = [], 0.0
    for label, start, end in segments:
        seconds = log.span(start, end)
        rows.append({"label": label, "from": start, "to": end,
                     "seconds": seconds})
        if seconds is not None:
            measured += seconds
    for row in rows:
        row["share"] = (row["seconds"] / measured
                        if row["seconds"] is not None and measured else None)
    return rows


#: The user-facing pipeline, in order. Names must match the marks the runner
#: records; a typo shows up as a None row rather than a wrong number.
PIPELINE_SEGMENTS = [
    # Paid once per process, never by a listener on a warm server. It is a row
    # of its own so it can be read and then set aside, rather than hiding
    # inside a request number.
    ("Chatterbox cold model load", "model_load_start", "model_load_complete"),
    ("Exa retrieval", "exa_request_start", "exa_complete"),
    ("prompt assembly", "exa_complete", "claude_request_start"),
    ("Claude time to first token", "claude_request_start", "claude_first_token"),
    ("Claude remaining generation", "claude_first_token", "claude_complete"),
    ("Chatterbox synthesis", "tts_start", "audio_complete"),
]

#: What the listener waits for, end to end. `search_to_first_listen` and
#: `search_to_complete_audio` are the two numbers the product is judged on.
HEADLINES = [
    ("exa_latency", "exa_request_start", "exa_complete"),
    ("claude_ttft", "claude_request_start", "claude_first_token"),
    ("claude_total", "claude_request_start", "claude_complete"),
    ("tts_generation", "tts_start", "audio_complete"),
    ("search_to_first_listen", "request_start", "first_playable_audio"),
    ("search_to_complete_audio", "request_start", "audio_complete"),
    # Cold only: process start, through the model load, to sound. None on a
    # warm run, where there is no load to include.
    ("cold_start_to_first_listen", "process_start", "first_playable_audio"),
]


def headlines(log: EventLog) -> dict:
    return {name: log.span(start, end) for name, start, end in HEADLINES}
