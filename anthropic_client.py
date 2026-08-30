"""One place that builds the Anthropic client, with the HTTP version pinned.

Why this exists: some networks (corporate proxies, TLS-inspecting middleboxes)
break HTTP/2 to api.anthropic.com, which surfaces as a bare
`anthropic.APIConnectionError: Connection error` with no useful detail. The same
request over HTTP/1.1 succeeds.

`anthropic` 1.x is built on **httpx2**, not httpx, and httpx2 already defaults to
`http2=False`. So HTTP/1.1 is normally the default already - but "normally the
default" is not the same as guaranteed. Passing it explicitly means the
behaviour is stated in the code, survives a future SDK or httpx2 default change,
and can be asserted in a test.

Use `build_async_client()` everywhere rather than constructing `AsyncAnthropic`
directly, so there is exactly one HTTP configuration in the project.
"""
from __future__ import annotations

import logging

import anthropic

from config import settings

log = logging.getLogger(__name__)


class Http2Unavailable(RuntimeError):
    """HTTP/2 was requested but the optional `h2` dependency is missing."""


def build_http_client() -> anthropic.DefaultAsyncHttpxClient:
    """The SDK's own client subclass, with the HTTP version pinned.

    `DefaultAsyncHttpxClient` is used rather than a raw `httpx2.AsyncClient` so
    the SDK's default timeouts, connection limits and TCP keep-alive settings
    are preserved; only the protocol version is overridden.
    """
    try:
        return anthropic.DefaultAsyncHttpxClient(http2=settings.anthropic_http2)
    except ImportError as exc:  # http2=True without the `h2` package installed
        raise Http2Unavailable(
            "ANTHROPIC_HTTP2 is enabled but the 'h2' package is not installed. "
            "Install it with `pip install h2`, or leave ANTHROPIC_HTTP2 unset to "
            "use HTTP/1.1."
        ) from exc


def build_async_client(api_key: str | None = None) -> anthropic.AsyncAnthropic:
    """An AsyncAnthropic bound to the pinned HTTP client.

    An empty/None key is not passed through: the SDK then resolves credentials
    itself from ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN or an `ant auth login`
    profile, which is the documented behaviour we want to keep.
    """
    kwargs: dict = {"http_client": build_http_client()}
    if api_key:
        kwargs["api_key"] = api_key
    log.debug("anthropic client using %s", describe_http_version())
    return anthropic.AsyncAnthropic(**kwargs)


def http2_enabled(client: anthropic.DefaultAsyncHttpxClient | None = None) -> bool:
    """Whether the transport pool will negotiate HTTP/2. Used by tests and health."""
    client = client if client is not None else build_http_client()
    try:
        return bool(getattr(client._transport._pool, "_http2", False))
    except AttributeError:  # pragma: no cover - httpx2 internals moved
        return bool(settings.anthropic_http2)


def describe_http_version() -> str:
    return "HTTP/2" if settings.anthropic_http2 else "HTTP/1.1"
