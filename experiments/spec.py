"""What an experiment *is*, as data.

An English sentence becomes one of these, you read it, and only then does
anything run. Keeping the spec separate from the running of it is what makes
the plan reviewable, the run reproducible, and two runs comparable.

A spec is JSON on disk. It never holds a credential - adapters read those from
the environment at run time and the store scrubs whatever reaches it anyway.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

#: The ordered checkpoints of the full product path. An arm reports the subset
#: it can actually distinguish; see `Arm.stage_note`.
PIPELINE_STAGES = [
    "search",              # retrieval, when it is a separable step
    "generate",            # model call up to the first token
    "first_chunk",         # ...up to the first speakable chunk
    "synthesis",           # chunk -> PCM
    "first_audio",         # ...to the first byte a listener could hear
]

COMPONENT = "component"
PIPELINE = "pipeline"


@dataclass
class Arm:
    """One thing being compared: a named combination of adapters and settings.

    Two arms differing in more than one dimension is a valid experiment but a
    weak one, so `dimensions_vs` reports how many things actually differ and
    the report says so out loud.
    """

    name: str
    search: str = "none"          # registry id: none | anthropic_web_search | exa
    tts: str = "none"             # registry id: none | piper | chatterbox
    model: Optional[str] = None   # None = whatever config.settings says
    #: Knobs the adapters read: first_chunk_words, packet_bytes, max_searches...
    params: dict = field(default_factory=dict)

    def dimensions_vs(self, other: "Arm") -> list[str]:
        differs = []
        for key in ("search", "tts", "model"):
            if getattr(self, key) != getattr(other, key):
                differs.append(key)
        for key in set(self.params) | set(other.params):
            if self.params.get(key) != other.params.get(key):
                differs.append(f"params.{key}")
        return differs

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Arm":
        return cls(
            name=data["name"],
            search=data.get("search", "none"),
            tts=data.get("tts", "none"),
            model=data.get("model"),
            params=dict(data.get("params") or {}),
        )


@dataclass
class ExperimentSpec:
    """A complete, runnable, reviewable experiment."""

    name: str
    arms: list[Arm]
    queries: list[str]
    trials: int = 5
    kind: str = PIPELINE
    minutes: float = 3.0
    #: Which checkpoints to report. Defaults to the full pipeline.
    stages: list[str] = field(default_factory=lambda: list(PIPELINE_STAGES))
    #: Reserved for V2. Recorded so today's runs stay comparable with
    #: tomorrow's, and asserted to be 1 until the load model exists.
    concurrency: int = 1
    seed: Optional[int] = 1234
    notes: str = ""

    # -- validation -----------------------------------------------------
    def validate(self) -> list[str]:
        """Everything wrong with this spec, as sentences. Empty means runnable."""
        problems = []
        if not self.name.strip():
            problems.append("The experiment needs a name.")
        if not self.arms:
            problems.append("An experiment needs at least one arm.")
        if len({a.name for a in self.arms}) != len(self.arms):
            problems.append("Two arms share a name; results could not be told apart.")
        if not self.queries:
            problems.append("An experiment needs at least one query.")
        if self.trials < 1:
            problems.append("Trials must be at least 1.")
        if self.trials < 3 and len(self.arms) > 1:
            problems.append(
                f"{self.trials} trials cannot support a comparison between arms; "
                "use at least 3, and 10 if you want a usable confidence interval."
            )
        if self.kind not in (COMPONENT, PIPELINE):
            problems.append(f"Unknown kind {self.kind!r}; expected {COMPONENT} or {PIPELINE}.")
        if self.concurrency != 1:
            problems.append(
                "Concurrency above 1 is not implemented in V1: the numbers would "
                "look like latency but measure queueing. Leave it at 1."
            )
        unknown = [s for s in self.stages if s not in PIPELINE_STAGES]
        if unknown:
            problems.append(f"Unknown stage(s): {', '.join(unknown)}.")
        return problems

    @property
    def total_trials(self) -> int:
        return len(self.arms) * len(self.queries) * self.trials

    def slug(self) -> str:
        s = re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-")
        return s[:48] or "experiment"

    # -- serialisation --------------------------------------------------
    def to_dict(self) -> dict:
        data = asdict(self)
        data["arms"] = [a.to_dict() for a in self.arms]
        return data

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentSpec":
        return cls(
            name=data["name"],
            arms=[Arm.from_dict(a) for a in data["arms"]],
            queries=list(data["queries"]),
            trials=int(data.get("trials", 5)),
            kind=data.get("kind", PIPELINE),
            minutes=float(data.get("minutes", 3.0)),
            stages=list(data.get("stages") or PIPELINE_STAGES),
            concurrency=int(data.get("concurrency", 1)),
            seed=data.get("seed", 1234),
            notes=data.get("notes", ""),
        )

    @classmethod
    def from_json(cls, text: str) -> "ExperimentSpec":
        return cls.from_dict(json.loads(text))
