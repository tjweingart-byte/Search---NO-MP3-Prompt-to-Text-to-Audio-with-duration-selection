"""A saved Exa evidence packet, replayed with no network call.

The point of this adapter is subtraction. The verified replication measured
search plus generation together; to isolate Claude, search has to stop varying
and stop taking time. So one real Exa packet is captured once, written to disk,
and replayed byte-for-byte on every trial of every arm.

That makes the comparison clean in two ways: no Exa latency lands in the
numbers, and every arm sees exactly the same evidence, so a difference in the
opening cannot be a difference in what the model was given.

The packet is built by `tools/capture_packet.py`, which calls Exa once with the
recovered parameters. Nothing here calls Exa.
"""
from __future__ import annotations

import json
import pathlib
from typing import Optional

from experiments.adapters.base import Availability, SearchResult
from experiments.timeline import Timeline

PACKETS_DIR = pathlib.Path(__file__).resolve().parent.parent / "packets"


def packet_path(name: str) -> pathlib.Path:
    safe = name.replace("/", "_").replace("..", "_").lstrip(".")
    if not safe.endswith(".json"):
        safe += ".json"
    return PACKETS_DIR / safe


def load_packet(name: str) -> dict:
    path = packet_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"No saved packet at {path}. Capture one with:\n"
            f"    python tools/capture_packet.py --name {name}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


class FixedPacket:
    """Replays a saved packet. Zero network, zero cost, zero variance."""

    id = "fixed_packet"
    label = "saved Exa packet (replayed, no network)"
    #: It is a real stage in the sense that it produces context, but it takes
    #: no measurable time, so it records no span - exactly like the server-side
    #: search adapter, and for the same reason: nothing to time honestly.
    separable = False
    host = "local"

    def __init__(self, name: str = "founder_ceos") -> None:
        self.name = name
        #: query -> packet name, for a run that spans several topics. Set from
        #: the arm's `packet_map` so one arm can hold every topic constant
        #: while the topic itself varies with the query.
        self.packet_map: dict = {}

    def resolve(self, query: str, params: dict) -> str:
        """Which packet this query should be answered from.

        A `packet_map` wins, then an explicit `packet`, then the default. A
        query that is mapped to nothing is an error rather than a silent
        fallback: answering one topic from another topic's evidence would look
        like a result and be nonsense.
        """
        mapping = params.get("packet_map") or self.packet_map
        if mapping:
            if query not in mapping:
                raise KeyError(
                    f"No packet mapped for query {query!r}. Mapped queries: "
                    f"{', '.join(sorted(mapping))}"
                )
            return mapping[query]
        return params.get("packet") or self.name

    def available(self) -> Availability:
        if self.packet_map:
            missing = [f"{q[:40]}... -> {n}" for q, n in sorted(self.packet_map.items())
                       if not packet_path(n).exists()]
            if missing:
                return Availability(
                    ok=False,
                    reason=f"{len(missing)} of {len(self.packet_map)} topic packets "
                           f"are missing: {'; '.join(missing[:3])}",
                    remedy="python tools/capture_packet.py --manifest "
                           "experiments/topics/<manifest>.json",
                )
            total = sum(len(json.loads(packet_path(n).read_text())["context"])
                        for n in self.packet_map.values())
            return Availability(
                ok=True,
                reason=f"{len(self.packet_map)} topic packets, {total} chars total")
        path = packet_path(self.name)
        if not path.exists():
            return Availability(
                ok=False,
                reason=f"No saved evidence packet at {path.name}.",
                remedy=f"python tools/capture_packet.py --name {self.name}   "
                       f"(one Exa call, about $0.005)",
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return Availability(ok=False, reason=f"{path.name} is not valid JSON: {exc}")
        if not data.get("context"):
            return Availability(ok=False, reason=f"{path.name} holds no packet text.")
        return Availability(ok=True, reason=f"{len(data['context'])} chars, "
                                            f"{len(data.get('sources') or [])} sources")

    async def search(self, query: str, timeline: Timeline, **params) -> SearchResult:
        name = self.resolve(query, params)
        data = load_packet(name)
        return SearchResult(
            context=data["context"],
            sources=list(data.get("sources") or []),
            searches=0,
            cost=0.0,
            detail={
                "packet": name,
                "topic": data.get("topic"),
                "category": data.get("category"),
                "packet_chars": len(data["context"]),
                "captured_at": data.get("captured_at"),
                "replayed": True,
            },
        )


def describe(name: str) -> Optional[dict]:
    """Summary of a saved packet, for the plan output."""
    path = packet_path(name)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "chars": len(data.get("context", "")),
        "sources": len(data.get("sources") or []),
        "query": data.get("query"),
        "captured_at": data.get("captured_at"),
    }
