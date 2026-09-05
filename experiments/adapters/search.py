"""Retrieval arms: the current Anthropic web search, and Exa.

The two are not the same shape, and the report says so rather than pretending.

**Anthropic web search is server-side.** It happens *inside* the generation
call, so there is no separable "search" interval to time. What can be measured
is what `tools/compare_search.py` already measures: run the same query searched
and unsearched, and the difference between the two first-token times is the
research cost with model and prompt held constant.

**Exa is a separate call**, so it has a real interval of its own, and its
results are handed to the model as context.

Both facts are declared on the adapter (`separable`), and the comparison layer
uses end-to-end first-audio - the number a listener feels - as the measure that
is sound across both shapes.
"""
from __future__ import annotations

import os

from experiments.adapters.base import Availability, SearchResult
from experiments.timeline import Timeline

#: Exa's published price per search, for the cost estimate. Not a measurement:
#: the recorded cost after a run comes from what the API reports, if it does.
EXA_COST_PER_SEARCH = 0.005


class NoSearch:
    """The control arm: answer from what the model already knows."""

    id = "none"
    label = "no search"
    separable = True
    host = "local"

    def available(self) -> Availability:
        return Availability(ok=True)

    async def search(self, query: str, timeline: Timeline, **params) -> SearchResult:
        return SearchResult(context="", searches=0, cost=0.0)


class AnthropicWebSearch:
    """The current production search. Baseline; never replaced, only compared.

    Runs no call of its own: it sets the flag the production request shape
    already understands, and the generation stage carries the cost. `search()`
    therefore returns immediately and records nothing but its own shape.
    """

    id = "anthropic_web_search"
    label = "Anthropic web search (current production)"
    #: Retrieval is folded into the model call and cannot be timed separately.
    separable = False
    host = "anthropic-api"

    def available(self) -> Availability:
        from config import settings

        if not settings.anthropic_api_key:
            return Availability(
                ok=False,
                reason="No Anthropic API key is configured.",
                remedy="python setup_key.py",
            )
        return Availability(ok=True)

    async def search(self, query: str, timeline: Timeline, **params) -> SearchResult:
        # Deliberately empty. The generation stage does the retrieving; timing
        # it here would double-count and invent an interval that does not exist.
        return SearchResult(
            context="",
            searches=0,
            cost=0.0,
            detail={"note": "server-side; measured inside the generate stage"},
        )


class ExaSearch:
    """Exa as a retrieval arm.

    **Not yet connected, and deliberately not guessed at.** The real call shape,
    query set and parameters live in `exa_claude_benchmark.py` on the machine
    where Exa was benchmarked. This class is the socket that file plugs into.

    To connect it, drop the implementation in
    `experiments/adapters/exa_impl.py` exposing::

        async def run_search(query: str, **params) -> dict

    returning at least ``{"context": str, "sources": [str], "searches": int}``
    and optionally ``{"remote_seconds": float, "cost": float}``. This adapter
    finds it, times it, and reports it. Until that file exists, `available()`
    says so and no experiment naming `exa` will start.

    Nothing here fabricates an Exa result. An adapter that returned plausible
    numbers would be worse than one that refuses.
    """

    id = "exa"
    label = "Exa (retrieval, separate call)"
    separable = True
    host = "exa-api"

    def _impl(self):
        try:
            from experiments.adapters import exa_impl  # type: ignore
        except ImportError:
            return None
        return getattr(exa_impl, "run_search", None)

    def available(self) -> Availability:
        if self._impl() is None:
            return Availability(
                ok=False,
                reason=(
                    "The Exa adapter has no implementation yet. The benchmark it "
                    "should wrap (exa_claude_benchmark.py) has not been imported."
                ),
                remedy=(
                    "Add experiments/adapters/exa_impl.py with "
                    "`async def run_search(query, **params) -> dict`, "
                    "ported from exa_claude_benchmark.py."
                ),
            )
        try:
            import exa_py  # noqa: F401
        except ImportError:
            return Availability(
                ok=False,
                reason="The exa-py package is not installed.",
                remedy="pip install -r experiments/requirements.txt",
            )
        if not os.environ.get("EXA_API_KEY"):
            return Availability(
                ok=False,
                reason="EXA_API_KEY is not set in this environment.",
                remedy="Set EXA_API_KEY (or add it to ~/.fam/env).",
            )
        return Availability(ok=True)

    async def search(self, query: str, timeline: Timeline, **params) -> SearchResult:
        run_search = self._impl()
        if run_search is None:
            raise RuntimeError(
                "ExaSearch.search called with no implementation. The planner "
                "should have refused this run; this is a bug in the harness."
            )
        with timeline.span("search", host=self.host, adapter=self.id) as stage:
            reply = await run_search(query, **params)
            stage.remote_seconds = reply.get("remote_seconds")
            stage.detail["results"] = len(reply.get("sources") or [])
        searches = int(reply.get("searches", 1) or 1)
        return SearchResult(
            context=reply.get("context", "") or "",
            sources=list(reply.get("sources") or []),
            searches=searches,
            cost=float(reply.get("cost", searches * EXA_COST_PER_SEARCH)),
            remote_seconds=reply.get("remote_seconds"),
            detail={k: v for k, v in reply.items() if k not in ("context", "sources")},
        )


SEARCH_ADAPTERS = {a.id: a for a in (NoSearch(), AnthropicWebSearch(), ExaSearch())}
