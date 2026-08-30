"""The Anthropic client must talk HTTP/1.1.

Some proxies and TLS-inspecting middleboxes break HTTP/2 to api.anthropic.com,
and the SDK reports that only as a bare `APIConnectionError: Connection error`.
httpx2 happens to default to HTTP/1.1 today, so these tests are really guarding
against that default changing underneath the project - which would reintroduce
a failure that is very hard to diagnose from the error alone.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic  # noqa: E402

from anthropic_client import (  # noqa: E402
    Http2Unavailable,
    build_async_client,
    build_http_client,
    describe_http_version,
    http2_enabled,
)
from config import settings  # noqa: E402
from script_generator import ScriptGenerator  # noqa: E402


def test_http2_is_disabled_by_default():
    assert settings.anthropic_http2 is False
    assert describe_http_version() == "HTTP/1.1"
    assert http2_enabled() is False


def test_the_transport_pool_will_not_negotiate_http2():
    """Assert on the real transport, not just the flag we passed in."""
    client = build_http_client()
    assert client._transport._pool._http2 is False


def test_the_sdk_client_uses_our_pinned_http_client():
    client = build_async_client("sk-ant-test")
    assert isinstance(client, anthropic.AsyncAnthropic)
    assert http2_enabled(client._client) is False


def test_the_script_generator_uses_the_pinned_client():
    """The path that actually fails in production must be covered."""
    generator = ScriptGenerator("sk-ant-test")
    assert http2_enabled(generator.client._client) is False


def test_sdk_defaults_are_preserved_not_replaced():
    """Only the protocol is overridden; timeouts and limits must survive."""
    pinned = build_http_client()
    stock = anthropic.DefaultAsyncHttpxClient()
    assert pinned.timeout == stock.timeout
    assert pinned.follow_redirects == stock.follow_redirects


def test_an_empty_key_is_not_forced_onto_the_client():
    """An unset key must fall through to the SDK's own credential resolution."""
    client = build_async_client("")
    assert isinstance(client, anthropic.AsyncAnthropic)


def test_enabling_http2_without_h2_fails_loudly(monkeypatch):
    """Opting in without the dependency must explain itself, not connect-error."""
    try:
        import h2  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("h2 is installed, so http2=True is valid here")

    import dataclasses

    import anthropic_client

    # Settings is frozen, so swap in a copy with the flag flipped.
    monkeypatch.setattr(
        anthropic_client, "settings", dataclasses.replace(settings, anthropic_http2=True)
    )
    with pytest.raises(Http2Unavailable, match="h2"):
        anthropic_client.build_http_client()
