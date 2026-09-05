"""The real Exa retrieval, ported from `exa_claude_benchmark.py`.

This is the manual benchmark run on 2026-09-05, adapted to the repeated-trial
architecture rather than rewritten. The call and the packet are deliberately
unchanged, because the point of importing that file is that its numbers stay
comparable with the ones already measured by hand:

* `search_and_contents(type="fast", num_results=8, highlights=True)`
* a packet built from the **top 3** results, **2 highlights** each,
  formatted `SOURCE n / Title: / Key evidence:`

What is new is that the three numbers the benchmark hard-coded are now
parameters, because they are exactly the knobs worth sweeping:
`num_results`, `packet_sources`, `highlights_per_source`. Their defaults
reproduce the manual run.

The credential is read from the environment here and never returned, logged or
put in the result dict. The store would scrub it anyway; not passing it around
is cheaper than relying on that.
"""
from __future__ import annotations

import os
import time
from typing import Any

#: The manual benchmark's settings. Changing a default here silently breaks
#: comparability with the hand-measured run, so sweep with parameters instead.
DEFAULT_SEARCH_TYPE = "fast"
DEFAULT_NUM_RESULTS = 8
DEFAULT_PACKET_SOURCES = 3
DEFAULT_HIGHLIGHTS_PER_SOURCE = 2

#: Exa's published rate, used only when the response reports no cost of its own.
COST_PER_SEARCH = 0.005


def _client():
    """Build the Exa client, or explain precisely what is missing."""
    try:
        from exa_py import Exa
    except ImportError as exc:  # pragma: no cover - exercised by availability()
        raise RuntimeError(
            "exa_py is not installed. `pip install -r experiments/requirements.txt`"
        ) from exc
    key = os.environ.get("EXA_API_KEY", "").strip()
    if not key:
        raise RuntimeError("EXA_API_KEY is not set.")
    return Exa(key)


def build_packet(
    results,
    packet_sources: int = DEFAULT_PACKET_SOURCES,
    highlights_per_source: int = DEFAULT_HIGHLIGHTS_PER_SOURCE,
) -> str:
    """The evidence packet, byte-for-byte as the manual benchmark built it.

    Kept as its own function so a packet-size experiment can vary the two
    numbers and a test can check the shape without calling Exa.
    """
    parts: list[str] = []
    for index, result in enumerate(results[:packet_sources], 1):
        parts.append(f"SOURCE {index}")
        parts.append(f"Title: {getattr(result, 'title', '') or ''}")
        highlights = getattr(result, "highlights", None)
        if highlights:
            parts.append("Key evidence:")
            for highlight in highlights[:highlights_per_source]:
                parts.append(highlight)
        parts.append("")
    return "\n".join(parts)


def _domains(results) -> list[str]:
    """Distinct hosts, for a person to judge. Never scored by machine."""
    seen: list[str] = []
    for result in results:
        url = getattr(result, "url", "") or ""
        if "//" in url:
            host = url.split("//", 1)[1].split("/", 1)[0]
            if host and host not in seen:
                seen.append(host)
    return seen


async def run_search(query: str, **params) -> dict[str, Any]:
    """One Exa retrieval, timed the way the benchmark timed it.

    Returns the packet as `context`, plus what a person needs to judge the
    sources. The call is synchronous on purpose: trials run one at a time
    (`concurrency` is pinned to 1), so there is nothing to starve, and a
    thread hand-off would add its own time to a number meant to be Exa's.
    """
    client = _client()
    search_type = params.get("search_type", DEFAULT_SEARCH_TYPE)
    num_results = int(params.get("num_results", DEFAULT_NUM_RESULTS))
    packet_sources = int(params.get("packet_sources", DEFAULT_PACKET_SOURCES))
    highlights_per_source = int(
        params.get("highlights_per_source", DEFAULT_HIGHLIGHTS_PER_SOURCE)
    )

    started = time.perf_counter()
    reply = client.search_and_contents(
        query,
        type=search_type,
        num_results=num_results,
        highlights=True,
    )
    elapsed = time.perf_counter() - started

    results = list(getattr(reply, "results", []) or [])
    packet = build_packet(results, packet_sources, highlights_per_source)

    cost = getattr(getattr(reply, "cost_dollars", None), "total", None)
    return {
        "context": packet,
        "sources": _domains(results),
        "searches": 1,
        "remote_seconds": elapsed,
        "cost": float(cost) if cost is not None else COST_PER_SEARCH,
        "packet_chars": len(packet),
        "results_returned": len(results),
        "packet_sources": packet_sources,
        "highlights_per_source": highlights_per_source,
        "search_type": search_type,
    }
