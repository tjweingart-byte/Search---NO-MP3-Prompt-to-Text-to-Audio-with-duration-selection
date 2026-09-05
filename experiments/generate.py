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
import time
from typing import AsyncIterator, Optional, Protocol


def with_packet(query: str, context: str) -> str:
    """Attach retrieved source material to the question.

    It goes in the *question*, matching how `exa_claude_benchmark.py` supplied
    its evidence packet, and deliberately **not** through `EpisodePlan.context`
    - that field renders as `<already_heard>` and tells the model the material
    is known and must not be re-explained, which is the exact opposite of what
    retrieved sources are for.

    Production's prompt is not modified; this is an experiment input.
    """
    if not context:
        return query
    return (
        f"{query}\n\n"
        f"EVIDENCE PACKET:\n{context}\n"
        f"Use only the supplied evidence. Do not mention sources, research, "
        f"URLs, or that you were given a packet."
    )


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
            plan = plan_episode(with_packet(query, context), minutes, search=search)
            generator = ScriptGenerator()
            generator.client = build_async_client()
            started = time.perf_counter()
            kwargs = generator._request_kwargs(plan)
            kwargs["model"] = model or patched.model

            final = None
            try:
                async with generator.client.messages.stream(**kwargs) as stream:
                    text_stream = stream.text_stream
                    try:
                        async for delta in text_stream:
                            yield delta
                        final = await stream.get_final_message()
                    finally:
                        # Drain first: see `_drain`. Doing it here also means
                        # the usage snapshot below covers the whole response,
                        # which is what was billed.
                        elapsed = await _drain_timed(text_stream, started)
                        self._usage = _usage_from(final, stream, kwargs["model"])
                        self._usage["stream_seconds"] = elapsed
            finally:
                # The client owns an httpx connection pool. One is built per
                # trial, so leaving them to the garbage collector leaks a pool
                # per trial and produces teardown noise at interpreter exit.
                await generator.client.close()
        finally:
            script_generator.settings = original


async def _drain_timed(iterator, started: float) -> float:
    """Drain, and report when the response actually ended.

    The consumer stops at the first speakable chunk, so nobody else is in a
    position to see the end of the stream. Teardown is, and it costs nothing
    extra to note the time - which is the only way a total generation time can
    be reported for a run that deliberately stops reading early.
    """
    await _drain(iterator)
    return time.perf_counter() - started


async def _drain(iterator) -> None:
    """Read whatever is left of a stream we stopped consuming early.

    This is the only teardown that leaves httpcore clean, and it took
    measuring to establish. The SDK nests async generators - `text_stream` ->
    `__aiter__` -> `__stream__` -> the raw SSE stream -> httpcore's
    `PoolByteStream.__aiter__` - and closing an async generator does **not**
    cascade to the ones it was iterating. So `aclose()` on the outermost, or
    `stream.close()`, or `response.aclose()`, all leave httpcore's generator
    suspended; it is then finalised at loop shutdown from another task and
    raises "generator didn't stop after athrow()".

    Every combination of those closes was tried against a real SDK client over
    a real socket, six runs each: all of them still produced the traceback on
    two to five runs out of six. Draining produced none, twice over. A single
    clean run proves nothing here, because the failure depends on when the
    garbage collector happens to run.

    This costs the wall time of reading a response the API is generating
    anyway. It is called after every measurement has been taken, so no timing
    this engine reports is affected by it.
    """
    try:
        async for _ in iterator:
            pass
    except Exception:
        # Teardown must not turn a completed trial into a failed one.
        pass


def _usage_from(final, stream, model: str) -> dict:
    """Token usage for a call, whether or not the stream was read to the end.

    On an early break `get_final_message()` never runs, so `final` is None. The
    SDK still holds a partial message whose usage is what has been billed so
    far; reading it keeps the recorded cost honest instead of silently zero.
    Every access is guarded, because a stream torn down before its first event
    may have no snapshot at all - in which case the usage is genuinely unknown
    and is recorded as such rather than as free.
    """
    message = final
    if message is None:
        try:
            message = getattr(stream, "current_message_snapshot", None)
        except Exception:
            message = None

    usage = getattr(message, "usage", None)
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "searches": _search_count(message),
        "sources": _domains(message),
        "complete": final is not None,
        "usage_known": bool(input_tokens or output_tokens),
    }


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


#: The system prompt from `exa_claude_benchmark.py`, kept verbatim.
BENCHMARK_SYSTEM = (
    "You are writing the opening of a FAM audio episode. "
    "Use only the supplied evidence packet. "
    "Write natural spoken narration. "
    "Start immediately with substance. "
    "Do not mention sources, research, URLs, or that you were given a packet."
)
BENCHMARK_MAX_TOKENS = 220
BENCHMARK_MODEL = "claude-sonnet-5"


class BenchmarkOpeningGenerator:
    """The model call exactly as the manual Exa benchmark made it.

    This exists so the repeated-trial engine can *reproduce* the hand-measured
    numbers rather than merely resemble them. It is not what ships: it writes
    only an opening, capped at 220 tokens, under its own short system prompt.

    Use it to check that the engine agrees with the manual run; use
    `ClaudeGenerator` - the production request shape - to decide anything about
    the product. An arm picks one with `params={"generator": "benchmark"}`.
    """

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
        from anthropic_client import build_async_client

        client = build_async_client()
        chosen = model or BENCHMARK_MODEL
        final = None
        started = time.perf_counter()
        try:
            async with client.messages.stream(
                model=chosen,
                max_tokens=BENCHMARK_MAX_TOKENS,
                system=BENCHMARK_SYSTEM,
                messages=[{
                    "role": "user",
                    "content": (
                        f"USER SEARCH:\n{query}\n\n"
                        f"EVIDENCE PACKET:\n{context}\n\n"
                        f"Begin the episode now."
                    ),
                }],
            ) as stream:
                text_stream = stream.text_stream
                try:
                    async for delta in text_stream:
                        yield delta
                    final = await stream.get_final_message()
                finally:
                    elapsed = await _drain_timed(text_stream, started)
                    self._usage = _usage_from(final, stream, chosen)
                    self._usage["generator"] = "benchmark"
                    self._usage["stream_seconds"] = elapsed
        finally:
            await client.close()


#: Added to the system prompt by the `first_chunk` arm. It asks for a *fuller*
#: opening sentence, not a shorter one: the failure mode to avoid is buying
#: latency by degrading the writing, so this shapes the first sentence rather
#: than truncating it. Everything about FAM's voice and the
#: evidence-only rule is inherited from BENCHMARK_SYSTEM unchanged.
FIRST_SENTENCE_DIRECTIVE = (
    " Your first sentence must be a complete, concrete, self-contained "
    "declarative statement drawn from the evidence - a fact with a subject and "
    "a consequence, not a scene-setter and not a question. Write it in full "
    "before you write anything else; do not open with a fragment, a clause you "
    "intend to finish later, or a phrase that only makes sense once the second "
    "sentence arrives."
)


class TunedOpeningGenerator:
    """The benchmark's call, with the request settings under experiment.

    Same model, same evidence, same 220-token ceiling and the same system
    prompt as the control - what changes is only what an arm asks for:

    ``thinking``      "adaptive" (the control's effective behaviour) or
                      "disabled". Sonnet 5 runs adaptive when `thinking` is
                      omitted, so the control is thinking on every call with
                      `display` defaulting to omitted - which is why it can
                      look like a pause before any text arrives.
    ``effort``        `output_config.effort`; the API default is "high".
    ``first_sentence_directive``
                      appends FIRST_SENTENCE_DIRECTIVE to the system prompt.

    Nothing here changes the model, the packet, the chunk rule or max_tokens,
    so a difference it produces is a difference in request settings and
    prompting alone.
    """

    def __init__(self, thinking: str = "disabled", effort: str | None = "low",
                 first_sentence_directive: bool = False, **_ignored) -> None:
        self.thinking = thinking
        self.effort = effort
        self.first_sentence_directive = first_sentence_directive
        self._usage: dict = {}

    def usage(self) -> dict:
        return dict(self._usage)

    def system_prompt(self) -> str:
        if self.first_sentence_directive:
            return BENCHMARK_SYSTEM + FIRST_SENTENCE_DIRECTIVE
        return BENCHMARK_SYSTEM

    def request_kwargs(self, model: str, query: str, context: str) -> dict:
        kwargs: dict = {
            "model": model,
            "max_tokens": BENCHMARK_MAX_TOKENS,
            "system": self.system_prompt(),
            "messages": [{
                "role": "user",
                "content": (
                    f"USER SEARCH:\n{query}\n\n"
                    f"EVIDENCE PACKET:\n{context}\n\n"
                    f"Begin the episode now."
                ),
            }],
        }
        # Omitting `thinking` is not the same as disabling it: on Sonnet 5 an
        # omitted `thinking` runs adaptive. "adaptive" here is therefore the
        # control's behaviour stated explicitly, not an extra setting.
        if self.thinking == "disabled":
            kwargs["thinking"] = {"type": "disabled"}
        elif self.thinking == "adaptive":
            kwargs["thinking"] = {"type": "adaptive"}
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        return kwargs

    async def stream(
        self,
        query: str,
        minutes: float,
        context: str = "",
        model: Optional[str] = None,
        search: bool = False,
        max_searches: int = 3,
    ) -> AsyncIterator[str]:
        from anthropic_client import build_async_client

        client = build_async_client()
        chosen = model or BENCHMARK_MODEL
        final = None
        started = time.perf_counter()
        try:
            async with client.messages.stream(
                **self.request_kwargs(chosen, query, context)
            ) as stream:
                text_stream = stream.text_stream
                try:
                    async for delta in text_stream:
                        yield delta
                    final = await stream.get_final_message()
                finally:
                    elapsed = await _drain_timed(text_stream, started)
                    self._usage = _usage_from(final, stream, chosen)
                    self._usage["stream_seconds"] = elapsed
                    self._usage.update({
                        "generator": "tuned",
                        "thinking": self.thinking,
                        "effort": self.effort,
                        "first_sentence_directive": self.first_sentence_directive,
                    })
        finally:
            await client.close()


GENERATORS = {
    "production": ClaudeGenerator,
    "benchmark": BenchmarkOpeningGenerator,
    "tuned": TunedOpeningGenerator,
}


def build_generator(name: str = "production", **options):
    """Pick a generator by name, failing loudly on an unknown one.

    `options` come from the arm's params, so an arm can say
    `{"generator": "tuned", "thinking": "disabled", "effort": "low"}`.
    Generators that take no options ignore them.
    """
    if name not in GENERATORS:
        raise KeyError(
            f"Unknown generator {name!r}. Known: {', '.join(sorted(GENERATORS))}"
        )
    factory = GENERATORS[name]
    try:
        return factory(**options)
    except TypeError:
        return factory()
