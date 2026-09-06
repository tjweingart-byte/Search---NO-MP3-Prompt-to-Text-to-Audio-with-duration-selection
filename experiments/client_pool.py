"""Long-lived Anthropic clients, for the arm that reuses one.

Every generator today builds a client, makes one request and closes it, so each
trial pays DNS, TCP and TLS again. The reuse arm needs the opposite: one client
that outlives the trial, so the second request onto it finds a warm connection.

Two details this has to get right, both learned from the SDK's own defaults.

**Keep-alive.** `DEFAULT_CONNECTION_LIMITS` expires an idle connection after
5 seconds. The harness alternates arms, so the gap between two reuse-arm
requests is a whole trial of the other arm - comfortably more than 5s. Left
alone, the reuse arm would reconnect every time and the experiment would
measure nothing while looking healthy. The pool therefore pins a keep-alive
long enough to span the run, and that is part of what the arm *is*: "one
client, kept alive", not "one client" alone.

**Closing.** A pooled client is not closed by the trial that used it, so the
sweep closes them at the end. `close_all` is idempotent and safe to call when
nothing was pooled.
"""
from __future__ import annotations

from typing import Optional

#: Long enough that no gap inside a sweep can expire a connection, and short
#: enough that a leaked client cannot hold one open indefinitely.
KEEPALIVE_SECONDS = 300.0

_CLIENTS: dict = {}
_KEEPALIVE: dict = {}


def build_pooled_client(keepalive: float = KEEPALIVE_SECONDS):
    """An Anthropic client whose connections survive between requests.

    Built from the SDK's own defaults with only the keep-alive changed, so the
    request shape, timeouts and pinned HTTP version are what every other arm
    uses.
    """
    import anthropic
    import httpx2

    from anthropic._constants import DEFAULT_CONNECTION_LIMITS
    from config import settings

    limits = httpx2.Limits(
        max_connections=DEFAULT_CONNECTION_LIMITS.max_connections,
        max_keepalive_connections=DEFAULT_CONNECTION_LIMITS.max_keepalive_connections,
        keepalive_expiry=keepalive,
    )
    http_client = anthropic.DefaultAsyncHttpxClient(
        http2=settings.anthropic_http2, limits=limits
    )
    return anthropic.AsyncAnthropic(http_client=http_client)


def acquire(key: str, keepalive: float = KEEPALIVE_SECONDS):
    """The client for this key, built once and kept."""
    if key not in _CLIENTS:
        _CLIENTS[key] = build_pooled_client(keepalive)
        _KEEPALIVE[key] = keepalive
    return _CLIENTS[key]


def pooled_keys() -> list:
    return sorted(_CLIENTS)


def keepalive_for(key: str) -> Optional[float]:
    return _KEEPALIVE.get(key)


async def close_all() -> None:
    """Close every pooled client. Idempotent; never raises."""
    for key in list(_CLIENTS):
        client = _CLIENTS.pop(key, None)
        _KEEPALIVE.pop(key, None)
        if client is None:
            continue
        try:
            await client.close()
        except Exception:
            pass
