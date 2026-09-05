"""The model stage, in the shape production actually uses.

`ClaudeGenerator` builds its request with the app's own `_request_kwargs`, so
what gets timed is the production request rather than an approximation of it -
the same discipline `tools/compare_search.py` established. Nothing here writes
to the script cache, and nothing here changes production behaviour: the
settings override is a local `dataclasses.replace` restored in a `finally`.

The stage is behind a tiny protocol so the whole harness can be exercised with
no API key at all. That is not a testing convenience bolted on afterwards; it
is how this repository has always been able to check itself.
"""
from __future__ import annotations

import dataclasses
from typing import AsyncIterator, Optional, Protocol


class Generator(Protocol):
    """Streams a script, token by token, and reports what it used."""

    async def stream(
        self,
        query: str,
        minutes: float,
        context: str = "",
        model: Optional[str] = None,
        search: bool = False,
        max_searches: int = 3,
    ) -> AsyncIterator[str]: ...

    def usage(self) -> dict: ...


class ClaudeGenerator:
    """The real model call. Needs an API key; costs money."""

    def __init__(self) -> None:
        self._usage: dict = {}

    def usage(self) -> dict:
        return dict(self._usage)

    async def stream(
        self,
        query: str,
        minutes: float,
        context: str = "",
        model: Optional[str] = None,
        search: bool = False,
        max_searches: int = 3,
    ) -> AsyncIterator[str]:
        import script_generator
        from anthropic_client import build_async_client
        from config import settings
        from script_generator import ScriptGenerator, plan_episode

        patched = dataclasses.replace(
            settings,
            search_mode="always" if search else "never",
            max_web_searches=max(1, max_searches),
            model=model or settings.model,
            # An experiment must never read or write the shared script cache:
            # a cached script would return in milliseconds and look like a win.
            cache_enabled=False,
        )
        original = script_generator.settings
        script_generator.settings = patched
        try:
            plan = plan_episode(query, minutes, search=search)
            if context:
                # Retrieved context is prepended as source material. The prompt
                # itself is untouched; this is an experiment input, not a
                # production prompt change.
                plan = dataclasses.replace(
                    plan, query=f"{query}\n\nSource material:\n{context}"
                ) if dataclasses.is_dataclass(plan) else plan
            generator = ScriptGenerator()
            generator.client = build_async_client()
            kwargs = generator._request_kwargs(plan)
            kwargs["model"] = model or patched.model

            final = None
            async with generator.client.messages.stream(**kwargs) as stream:
                async for delta in stream.text_stream:
                    yield delta
                final = await stream.get_final_message()
            usage = getattr(final, "usage", None)
            self._usage = {
                "model": kwargs["model"],
                "input_tokens": getattr(usage, "input_tokens", 0) or 0,
                "output_tokens": getattr(usage, "output_tokens", 0) or 0,
                "searches": _search_count(final),
                "sources": _domains(final),
            }
        finally:
            script_generator.settings = original


def _search_count(message) -> int:
    if message is None:
        return 0
    count = 0
    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", "") in ("server_tool_use", "web_search_tool_result"):
            count += 1
    return count


def _domains(message) -> list[str]:
    if message is None:
        return []
    seen: list[str] = []
    for block in getattr(message, "content", []) or []:
        for item in getattr(block, "content", []) or []:
            url = getattr(item, "url", "") or ""
            if "//" in url:
                host = url.split("//", 1)[1].split("/", 1)[0]
                if host and host not in seen:
                    seen.append(host)
    return seen
