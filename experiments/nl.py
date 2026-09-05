"""Plain English in, a spec you can read out.

This is a small keyword parser, not a language model, and that is the point: it
is predictable, testable, and it never runs anything. It produces a spec plus a
list of the assumptions it made, both of which are printed before the cost
prompt, so a misreading is caught by eye rather than by spending.

Anything it could not work out it says out loud in `assumptions`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from experiments.spec import Arm, ExperimentSpec, COMPONENT, PIPELINE

DEFAULT_QUERIES = [
    "why do tides happen",
    "what changed in the news today",
    "how does a heat pump work",
]

_ALIASES = {
    "exa": ("search", "exa"),
    "anthropic web search": ("search", "anthropic_web_search"),
    "current search": ("search", "anthropic_web_search"),
    "existing search": ("search", "anthropic_web_search"),
    "web search": ("search", "anthropic_web_search"),
    "no search": ("search", "none"),
    # Longest alias first: "chatterbox local" must not match plain "chatterbox".
    "chatterbox local": ("tts", "chatterbox_local"),
    "local chatterbox": ("tts", "chatterbox_local"),
    "chatterbox": ("tts", "chatterbox"),
    "piper": ("tts", "piper"),
}


@dataclass
class Parsed:
    spec: ExperimentSpec
    assumptions: list[str] = field(default_factory=list)


def compile_request(text: str) -> Parsed:
    """Turn a sentence into a reviewable experiment spec."""
    low = text.lower()
    assumptions: list[str] = []

    # -- trials ---------------------------------------------------------
    trials = 5
    match = re.search(r"(\d+)\s*(?:times|trials|runs|reps|iterations)", low)
    if match:
        trials = max(1, int(match.group(1)))
    else:
        assumptions.append(f"No trial count given; using {trials}.")

    # -- episode length -------------------------------------------------
    minutes = 3.0
    match = re.search(r"(\d+(?:\.\d+)?)\s*[- ]?min", low)
    if match:
        minutes = float(match.group(1))
    else:
        assumptions.append(f"No episode length given; using {minutes:g} minutes.")

    # -- which components are named -------------------------------------
    # Ordered by where each appears in the sentence, so "Piper vs Chatterbox"
    # produces arms in that order rather than in whatever order this file
    # happens to list its aliases.
    hits: dict[str, list[tuple[int, str]]] = {"search": [], "tts": []}
    # Longest alias first, and each match blanks out the span it consumed, so
    # "chatterbox local" cannot also register a plain "chatterbox" arm inside
    # itself. Without the masking, one phrase produced two contradictory arms.
    scratch = low
    for alias in sorted(_ALIASES, key=len, reverse=True):
        kind, value = _ALIASES[alias]
        at = scratch.find(alias)
        if at >= 0 and all(v != value for _, v in hits[kind]):
            hits[kind].append((at, value))
            scratch = scratch[:at] + (" " * len(alias)) + scratch[at + len(alias):]
    named = {kind: [v for _, v in sorted(items)] for kind, items in hits.items()}

    full = any(w in low for w in ("pipeline", "end to end", "end-to-end", "full stack"))
    comparing = any(w in low for w in (" vs ", " versus ", "compare", "against", " or "))

    # -- build the arms --------------------------------------------------
    arms: list[Arm] = []
    if comparing and (len(named["search"]) > 1 or len(named["tts"]) > 1):
        # A comparison: vary the dimension that has two candidates, hold the rest.
        if len(named["search"]) > 1:
            other_tts = named["tts"][0] if named["tts"] else "none"
            for value in named["search"]:
                arms.append(Arm(name=value, search=value, tts=other_tts))
            if not named["tts"]:
                assumptions.append("Comparing search only; no speech stage (tts=none).")
        else:
            other_search = named["search"][0] if named["search"] else "none"
            for value in named["tts"]:
                arms.append(Arm(name=value, search=other_search, tts=value))
            if not named["search"]:
                assumptions.append("Comparing voices only; no search stage.")
    else:
        search = named["search"][0] if named["search"] else ("anthropic_web_search" if full else "none")
        tts = named["tts"][0] if named["tts"] else ("piper" if full else "none")
        label = "+".join(x for x in (search, tts) if x != "none") or "baseline"
        arms.append(Arm(name=label, search=search, tts=tts))
        if full and not named["search"]:
            assumptions.append("Full pipeline requested; using current production search.")
        if full and not named["tts"]:
            assumptions.append("Full pipeline requested; using Piper for speech.")

    kind = PIPELINE if (full or any(a.tts != "none" and a.search != "none" for a in arms)) else COMPONENT

    # A single named arm with no control is usually not what is wanted: a
    # pipeline timing with nothing beside it cannot say whether it is good.
    is_production_stack = (arms[0].search == "anthropic_web_search" and arms[0].tts == "piper")
    if len(arms) == 1 and kind == PIPELINE and not is_production_stack:
        arms.append(Arm(name="baseline", search="anthropic_web_search", tts="piper"))
        assumptions.append(
            "Added the current production stack as a control arm; a pipeline "
            "timing with nothing to compare against cannot say whether it is good."
        )

    # -- first-chunk sweep ------------------------------------------------
    chunk = re.search(r"first[- ]chunk[^.\d]*(\d+)", low)
    if chunk:
        words = int(chunk.group(1))
        for arm in arms:
            arm.params["first_chunk_words"] = words
        assumptions.append(f"First chunk fixed at {words} words.")

    name = _name_for(text, arms)
    spec = ExperimentSpec(
        name=name,
        arms=arms,
        queries=list(DEFAULT_QUERIES),
        trials=trials,
        kind=kind,
        minutes=minutes,
        notes=f"Compiled from: {text.strip()[:300]}",
    )
    assumptions.append(f"Using the {len(DEFAULT_QUERIES)} default queries; "
                       f"override with --queries <file>.")
    if trials < 10 and len(arms) > 1:
        assumptions.append("Fewer than 10 trials per arm rarely resolves a latency "
                           "difference; consider --trials 10.")
    return Parsed(spec=spec, assumptions=assumptions)


def _name_for(text: str, arms: list[Arm]) -> str:
    if len(arms) > 1:
        return " vs ".join(a.name for a in arms)
    return arms[0].name if arms else re.sub(r"\s+", " ", text.strip())[:48]
