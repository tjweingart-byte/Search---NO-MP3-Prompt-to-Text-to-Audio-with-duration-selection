"""Stand-ins that let the whole engine be exercised with no key and no GPU.

These are not test scaffolding kept at arm's length from the real thing - they
implement the same protocols the real adapters do, so a change that breaks the
harness contract breaks these too. Every timing they produce is fabricated and
obviously so: nothing here should ever be mistaken for a measurement, and the
harness records `simulated: true` on any trial that used one.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator, Optional

from experiments.adapters.base import Availability, SearchResult, SynthResult
from experiments.timeline import Timeline

SIMULATED = "simulated"


class FakeSearch:
    """A retrieval stage with a settable delay."""

    def __init__(self, delay: float = 0.01, sources: Optional[list[str]] = None,
                 adapter_id: str = "fake_search") -> None:
        self.id = adapter_id
        self.label = f"fake search ({delay}s)"
        self.delay = delay
        self.separable = True
        self.host = "fake-search-api"
        self.sources = sources or ["example.com", "example.org"]

    def available(self) -> Availability:
        return Availability(ok=True, reason=SIMULATED)

    async def search(self, query: str, timeline: Timeline, **params) -> SearchResult:
        if not self.separable:
            # Mirror the real server-side adapter: retrieval folded into the
            # model call records no span of its own. A fake that invented a
            # 0.00s "search" stage would make the report claim a measurement
            # that production cannot actually take.
            return SearchResult(context="", searches=0, cost=0.0,
                                detail={"note": "server-side; not separable"})
        with timeline.span("search", host=self.host, adapter=self.id) as stage:
            await asyncio.sleep(self.delay)
            stage.remote_seconds = self.delay * 0.8
            stage.detail["results"] = len(self.sources)
        return SearchResult(
            context=f"Context for {query}.",
            sources=list(self.sources),
            searches=1,
            cost=0.005,
            remote_seconds=self.delay * 0.8,
        )


class FakeTTS:
    """A speech stage that produces real PCM-shaped bytes at a chosen speed."""

    def __init__(self, delay: float = 0.01, adapter_id: str = "fake_tts",
                 host: str = "fake-gpu", sample_rate: int = 22050) -> None:
        self.id = adapter_id
        self.label = f"fake tts ({delay}s)"
        self.delay = delay
        self.host = host
        self.sample_rate = sample_rate

    def available(self) -> Availability:
        return Availability(ok=True, reason=SIMULATED)

    async def synth(self, text: str, timeline: Timeline, **params) -> SynthResult:
        with timeline.span("synthesis", host=self.host, adapter=self.id) as stage:
            await asyncio.sleep(self.delay)
            stage.remote_seconds = self.delay * 0.7
        # Two bytes per sample, ~150 wpm, so the duration is at least plausible.
        seconds = max(0.1, len(text.split()) / 2.5)
        pcm = b"\x00\x00" * int(self.sample_rate * seconds)
        return SynthResult(
            pcm=pcm,
            sample_rate=self.sample_rate,
            audio_seconds=seconds,
            cost=0.0,
            remote_seconds=self.delay * 0.7,
            detail={"wall_seconds": self.delay},
        )


class FakeGenerator:
    """Streams a fixed script with a settable time-to-first-token."""

    def __init__(self, first_token_delay: float = 0.02, per_token_delay: float = 0.001,
                 script: Optional[str] = None) -> None:
        self.first_token_delay = first_token_delay
        self.per_token_delay = per_token_delay
        self.script = script or (
            "The tide goes out twice a day and nobody arranged it. "
            "That rhythm is the moon pulling on water it cannot reach. "
            "Every coastline on earth keeps the same appointment. "
            "The water does not move so much as the bulge stays still while the planet turns under it."
        )
        self._usage: dict = {}

    def usage(self) -> dict:
        return dict(self._usage)

    async def stream(self, query: str, minutes: float, context: str = "",
                     model: Optional[str] = None, search: bool = False,
                     max_searches: int = 3) -> AsyncIterator[str]:
        await asyncio.sleep(self.first_token_delay)
        # A searched call cannot write until its results are back; the fake
        # models that the same way so the shape of the comparison is right.
        if search and not context:
            await asyncio.sleep(self.first_token_delay)
        words = self.script.split(" ")
        for index, word in enumerate(words):
            if index:
                await asyncio.sleep(self.per_token_delay)
            yield (" " if index else "") + word
        self._usage = {
            "model": model or "fake-model",
            "input_tokens": 1000,
            "output_tokens": len(words),
            "searches": max_searches if search else 0,
            "sources": ["example.com"] if search else [],
            SIMULATED: True,
        }
