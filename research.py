"""Retrieval that runs *before* Claude writes, so Claude reads rather than searches.

Two ways to research an episode, and the difference is who does the looking:

* **claude** - the model gets Anthropic's server-side `web_search` tool and
  searches while it writes. One call, no second credential, and the searching
  is inside the model's own turn.
* **exa** - Exa retrieves first, this module builds an evidence packet, and the
  packet goes into the prompt as context. Claude reads it. No tool, no
  searching inside the turn.

The second is what this module is. The call and the packet are byte-for-byte
the ones the manual benchmark measured on 2026-09-05 and the experiment layer
then repeated - `search_and_contents(type="fast", num_results=8,
highlights=True)`, top 3 results, 2 highlights each, formatted
`SOURCE n / Title: / Key evidence:`. Keeping them identical is the point: the
numbers already measured stay comparable, and a change of packet shape is a
deliberate act rather than a drift.

**Where this is allowed to cost time.** A retrieval in front of the first word
is the one cost this product refuses (the one-sentence spec, and PROBLEMS.md
§55 on the cold open). It is affordable here for one reason only: a researched
episode already runs two calls at once. `_answer_first` speaks the
from-knowledge half immediately while the researched half works underneath, and
this retrieval happens on the researched half - so it is covered by an answer
rather than by filler or by silence. With `ANSWER_FIRST=0` there is nothing
covering it, and the wait is real; that is a deliberate trade of that setting,
not of this module.

**It does not block the event loop.** The experiment ran `exa_py` synchronously
because trials were pinned to one at a time and a thread hand-off would have
added its own time to a number meant to be Exa's. Production is the opposite
case: the cover is speaking, the assembler is batching, and a synchronous HTTP
call on the loop would stop both. So the call goes to a worker thread, exactly
as speech synthesis does.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from config import RESEARCH_BACKENDS, settings

log = logging.getLogger("research")

#: The manual benchmark's settings, as defaults. Changing one here silently
#: breaks comparability with the hand-measured run; override per request or by
#: configuration instead.
DEFAULT_SEARCH_TYPE = "fast"

#: Exa's published rate, used only when a response reports no cost of its own.
COST_PER_SEARCH = 0.005


class ResearchUnavailable(RuntimeError):
    """The configured backend cannot run, and says which part is missing."""


@dataclass
class Packet:
    """What retrieval produced, and what it cost to produce it.

    `context` is the only part Claude sees. The rest exists so a person can
    judge the sources and so a run can be costed - neither is ever scored by
    machine, and neither reaches the prompt.
    """

    context: str = ""
    sources: list = field(default_factory=list)
    searches: int = 0
    seconds: float = 0.0
    cost: float = 0.0
    results_returned: int = 0
    backend: str = ""

    def __bool__(self) -> bool:
        return bool(self.context.strip())

    def as_dict(self) -> dict:
        return {
            "backend": self.backend,
            "sources": list(self.sources),
            "searches": self.searches,
            "seconds": round(self.seconds, 3),
            "cost": round(self.cost, 4),
            "results_returned": self.results_returned,
            "packet_chars": len(self.context),
        }


# --------------------------------------------------------------------------
# Exa
# --------------------------------------------------------------------------
def _client():
    """Build the Exa client, or say precisely what is missing.

    Imported here rather than at module scope so that `exa_py` is an optional
    dependency: a deployment on the `claude` backend must not need it, and the
    test suite must run without it.
    """
    try:
        from exa_py import Exa
    except ImportError as exc:
        raise ResearchUnavailable(
            "exa_py is not installed. `pip install -r requirements-exa.txt`"
        ) from exc
    key = (os.environ.get("EXA_API_KEY") or "").strip()
    if not key:
        raise ResearchUnavailable(
            "EXA_API_KEY is not set, so Exa cannot retrieve anything. Set it, "
            "or set RESEARCH_BACKEND=claude to let the model search instead.")
    return Exa(key)


def diagnose() -> tuple[bool, str]:
    """Why Exa can or cannot serve, in one sentence.

    The same shape as `ChatterboxEngine.diagnose`, and for the same reason:
    something that reports readiness must name what is wrong, not just answer
    no. Does not perform a retrieval - that would cost money on every health
    check - so it reports *configured*, and `retrieve` is what proves it works.
    """
    try:
        import exa_py  # noqa: F401
    except ImportError:
        return False, "exa_py is not installed"
    if not (os.environ.get("EXA_API_KEY") or "").strip():
        return False, "EXA_API_KEY is not set"
    return True, "exa_py installed and EXA_API_KEY present"


def available() -> bool:
    return diagnose()[0]


def build_packet(results, packet_sources: int, highlights_per_source: int) -> str:
    """The evidence packet, byte-for-byte as the manual benchmark built it.

    Its own function so a packet-size experiment can vary the two numbers, and
    so a test can check the shape without calling Exa or holding a credential.
    """
    parts: list[str] = []
    for index, result in enumerate(list(results)[:packet_sources], 1):
        parts.append(f"SOURCE {index}")
        parts.append(f"Title: {getattr(result, 'title', '') or ''}")
        highlights = getattr(result, "highlights", None)
        if highlights:
            parts.append("Key evidence:")
            for highlight in list(highlights)[:highlights_per_source]:
                parts.append(highlight)
        parts.append("")
    return "\n".join(parts)


def domains(results) -> list:
    """Distinct hosts, for a person to judge. Never scored by machine."""
    seen: list = []
    for result in results:
        url = getattr(result, "url", "") or ""
        if "//" in url:
            host = url.split("//", 1)[1].split("/", 1)[0]
            if host and host not in seen:
                seen.append(host)
    return seen


def _retrieve_blocking(query: str, num_results: int, packet_sources: int,
                       highlights_per_source: int, search_type: str) -> Packet:
    """The validated call, unchanged, plus the packet and the accounting."""
    client = _client()
    started = time.perf_counter()
    reply = client.search_and_contents(
        query,
        type=search_type,
        num_results=num_results,
        highlights=True,
    )
    elapsed = time.perf_counter() - started

    results = list(getattr(reply, "results", []) or [])
    cost = getattr(getattr(reply, "cost_dollars", None), "total", None)
    return Packet(
        context=build_packet(results, packet_sources, highlights_per_source),
        sources=domains(results),
        searches=1,
        seconds=elapsed,
        cost=float(cost) if cost is not None else COST_PER_SEARCH,
        results_returned=len(results),
        backend="exa",
    )


async def retrieve(query: str, backend: Optional[str] = None,
                   **overrides: Any) -> Packet:
    """Research `query` with the configured backend.

    Returns an empty packet for the `claude` backend - not an error. There is
    nothing to retrieve there because the model does its own searching inside
    the turn, and an empty packet is exactly what tells `build_prompt` to leave
    the tool attached and add no evidence block.

    Raises `ResearchUnavailable` when the *configured* backend cannot run. It
    does not fall back to the other one: a deployment that asked for Exa and
    silently got the model's own search would be measuring one thing while
    believing another, and this project has paid for that shape more than once.
    """
    chosen = (backend or settings.research_backend or "").strip().lower()
    if chosen not in RESEARCH_BACKENDS:
        raise ResearchUnavailable(
            f"RESEARCH_BACKEND={chosen!r} is not a backend. Use one of: "
            f"{', '.join(RESEARCH_BACKENDS)}. Refusing rather than falling "
            "back - an unrecognised value must never quietly pick one.")
    if chosen == "claude":
        return Packet(backend="claude")

    query = (query or "").strip()
    if not query:
        raise ResearchUnavailable("nothing to research: the query is empty")

    num_results = int(overrides.get("num_results", settings.exa_num_results))
    packet_sources = int(overrides.get("packet_sources", settings.exa_packet_sources))
    highlights_per_source = int(
        overrides.get("highlights_per_source", settings.exa_highlights_per_source))
    search_type = str(overrides.get("search_type", DEFAULT_SEARCH_TYPE))

    # Off the event loop: the cover is speaking and the assembler is batching
    # while this runs, and a synchronous HTTP call on the loop would stop both.
    packet = await asyncio.to_thread(
        _retrieve_blocking, query, num_results, packet_sources,
        highlights_per_source, search_type)

    if not packet:
        # Retrieval succeeded and found nothing usable. Not an exception - the
        # episode is still answerable from knowledge - but it must not pass as
        # research, so it says so and the prompt gets no evidence block.
        log.warning("exa returned %d result(s) but no usable evidence for %r",
                    packet.results_returned, query)
    else:
        log.info("exa: %d chars from %d source(s) in %.2fs (~$%.4f) for %r",
                 len(packet.context), len(packet.sources), packet.seconds,
                 packet.cost, query)
    return packet


def report() -> dict:
    """What the server can actually do for research right now - for /api/health."""
    ok, detail = diagnose()
    return {
        "backend": settings.research_backend,
        "backends": list(RESEARCH_BACKENDS),
        "exa_configured": ok,
        "exa_detail": detail,
        # True when the configured backend cannot run. A researched episode
        # will fail rather than quietly search another way, so this is worth
        # seeing on a tab rather than discovering in a log.
        "unavailable": settings.research_backend == "exa" and not ok,
    }
