"""Per-phase HTTP timings, from httpcore's own trace extension.

The forensics run reported a single `transport` figure - dispatch to response
headers - and called it transport. That number is real but it is a *bucket*: it
contains DNS, TCP, TLS, the request upload, the server's queue and processing,
and the return trip, with no way to tell which dominated.

httpcore emits named events around each of those phases when a request carries
a `trace` extension. Subscribing to them turns most of that bucket into
measurements. What remains genuinely inferred is listed in
`experiments/TRANSPORT_FORENSICS.md`; nothing here pretends otherwise.

**This does not change the request.** `extensions` is client-side metadata that
httpx hands to the transport and never serialises onto the wire. The wrapper
delegates every call to the transport the SDK already built, so timeouts,
connection limits and the pinned HTTP version are exactly what they were.

The one caveat worth stating plainly: reaching the transport uses
`client._transport`, which is a private attribute of httpx's client. If a
future version renames it, `attach` returns None, the run continues without
phase timings, and the trial records `http_trace: unavailable` rather than
silently reporting a bucket as though it were measured.
"""
from __future__ import annotations

import time
from typing import Optional

#: httpcore prefixes each event with its logger name, so the events arrive as
#: "connection.connect_tcp.started", "http11.receive_response_headers.complete"
#: and so on. These are the ones that bound a phase we care about.
PHASES = {
    "connect": ("connection.connect_tcp.started", "connection.connect_tcp.complete"),
    "tls": ("connection.start_tls.started", "connection.start_tls.complete"),
    "upload": ("http11.send_request_headers.started", "http11.send_request_body.complete"),
    "wait_for_headers": ("http11.send_request_body.complete",
                         "http11.receive_response_headers.complete"),
}

#: Seen on a pooled connection instead of connect_tcp: the request went
#: straight to sending. Its presence without connect_tcp is how connection
#: reuse is detected rather than assumed.
FIRST_NETWORK_EVENTS = (
    "connection.connect_tcp.started",
    "http11.send_request_headers.started",
)


class TraceRecorder:
    """Collects httpcore trace events with a `perf_counter` reading each."""

    def __init__(self) -> None:
        self.events: list[tuple[str, float]] = []
        self.available = False

    async def __call__(self, name: str, info: dict) -> None:
        # Must be async: httpcore's async path rejects a plain function.
        self.events.append((name, time.perf_counter()))

    # -- reading the events back ---------------------------------------
    def first(self, name: str) -> Optional[float]:
        for event, at in self.events:
            if event == name:
                return at
        return None

    def span(self, start: str, end: str) -> Optional[float]:
        a, b = self.first(start), self.first(end)
        if a is None or b is None:
            return None
        return b - a

    @property
    def connection_reused(self) -> Optional[bool]:
        """True when no TCP connect happened, so a pooled connection served it.

        None when nothing was traced at all - "unknown" and "reused" are
        different answers and must not be collapsed.
        """
        if not self.events:
            return None
        return self.first("connection.connect_tcp.started") is None

    def first_network_at(self) -> Optional[float]:
        candidates = [self.first(name) for name in FIRST_NETWORK_EVENTS]
        actual = [c for c in candidates if c is not None]
        return min(actual) if actual else None

    def phases(self, dispatch: Optional[float] = None) -> dict:
        """Every phase that was actually observed, in seconds."""
        out: dict = {
            "http_trace": "ok" if self.events else "unavailable",
            "connection_reused": self.connection_reused,
            "events_seen": len(self.events),
        }
        for label, (start, end) in PHASES.items():
            out[f"phase_{label}"] = self.span(start, end)

        first_network = self.first_network_at()
        if dispatch is not None and first_network is not None:
            # Everything before the socket layer was touched: building the
            # request, serialising the body, acquiring a pool slot.
            out["phase_local_setup"] = first_network - dispatch
        else:
            out["phase_local_setup"] = None

        headers_at = self.first("http11.receive_response_headers.complete")
        out["headers_complete_perf"] = headers_at
        if dispatch is not None and headers_at is not None:
            out["phase_dispatch_to_headers"] = headers_at - dispatch
        else:
            out["phase_dispatch_to_headers"] = None
        return out

    def event_log(self) -> list[dict]:
        """The raw events, relative to the first one. Kept for the record."""
        if not self.events:
            return []
        origin = self.events[0][1]
        return [{"event": name, "at": round(at - origin, 6)} for name, at in self.events]


class _TracingTransport:
    """Delegates everything, and adds the trace extension on the way past."""

    def __init__(self, inner, recorder: TraceRecorder) -> None:
        self._inner = inner
        self._recorder = recorder

    async def handle_async_request(self, request):
        # Client-side only. httpx passes extensions to the transport; it never
        # writes them to the socket.
        try:
            request.extensions = {**dict(request.extensions), "trace": self._recorder}
        except Exception:
            pass
        return await self._inner.handle_async_request(request)

    async def aclose(self):
        return await self._inner.aclose()

    def __getattr__(self, name):
        return getattr(self._inner, name)


def attach(client) -> Optional[TraceRecorder]:
    """Wrap an Anthropic client's transport so phases are recorded.

    Returns the recorder, or None when the transport could not be reached -
    in which case the caller records that phases are unavailable instead of
    reporting a bucket as a measurement.
    """
    http_client = getattr(client, "_client", None)
    if http_client is None:
        return None
    inner = getattr(http_client, "_transport", None)
    if inner is None:
        return None
    recorder = TraceRecorder()
    try:
        http_client._transport = _TracingTransport(inner, recorder)
    except Exception:
        return None
    recorder.available = True
    return recorder
