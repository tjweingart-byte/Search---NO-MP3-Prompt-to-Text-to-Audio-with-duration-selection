"""Names to adapters, and one honest answer about what can run.

The planner asks this module "can the whole spec run here?" before anything
spends. It answers with every problem at once - a run blocked on both a missing
Exa key and a missing GPU should say so in one breath, not one failure at a time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from experiments.adapters.base import Availability
from experiments.adapters.search import SEARCH_ADAPTERS
from experiments.adapters.tts import TTS_ADAPTERS


def search_adapter(name: str):
    if name not in SEARCH_ADAPTERS:
        raise KeyError(f"Unknown search adapter {name!r}. Known: {', '.join(sorted(SEARCH_ADAPTERS))}")
    return SEARCH_ADAPTERS[name]


def tts_adapter(name: str, voice: str | None = None):
    if name not in TTS_ADAPTERS:
        raise KeyError(f"Unknown TTS adapter {name!r}. Known: {', '.join(sorted(TTS_ADAPTERS))}")
    adapter = TTS_ADAPTERS[name]
    if name == "piper" and voice is not None:
        from experiments.adapters.tts import LocalTTS

        return LocalTTS(voice=voice)
    return adapter


@dataclass
class Preflight:
    """Whether a spec can run, and everything stopping it."""

    ok: bool
    blockers: list[str] = field(default_factory=list)
    #: Blockers that are specifically "paid infrastructure is not running".
    #: Separated because the answer to these is a human decision to spend money.
    approvals: list[str] = field(default_factory=list)
    available: dict = field(default_factory=dict)

    def render(self) -> str:
        if self.ok:
            return "  all adapters available"
        lines = []
        for item in self.blockers:
            lines.append(f"  BLOCKED  {item}")
        for item in self.approvals:
            lines.append(f"  APPROVAL REQUIRED  {item}")
        return "\n".join(lines)


def preflight(spec) -> Preflight:
    """Check every adapter the spec names, before a single trial runs."""
    blockers: list[str] = []
    approvals: list[str] = []
    available: dict = {}

    for arm in spec.arms:
        for kind, name in (("search", arm.search), ("tts", arm.tts)):
            try:
                adapter = (
                    search_adapter(name)
                    if kind == "search"
                    else tts_adapter(name, arm.params.get("voice"))
                )
            except KeyError as exc:
                blockers.append(f"arm {arm.name!r}: {exc}")
                continue
            state: Availability = adapter.available()
            available[f"{arm.name}.{kind}"] = state.to_dict()
            if state.ok:
                continue
            message = f"arm {arm.name!r} needs {kind} {name!r}: {state.reason}"
            if state.remedy:
                message += f"\n             -> {state.remedy}"
            (approvals if state.needs_approval else blockers).append(message)

    return Preflight(
        ok=not blockers and not approvals,
        blockers=blockers,
        approvals=approvals,
        available=available,
    )
