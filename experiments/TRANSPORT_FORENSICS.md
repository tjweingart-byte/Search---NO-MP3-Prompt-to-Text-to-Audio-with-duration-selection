# What "transport" actually measures

The first forensics run reported one figure - dispatch to response headers -
and called it transport. It is real, but it is a **bucket**: DNS, TCP, TLS,
the request upload, the server's queue and processing, and the return trip,
with no way to tell which dominated. This is the audit of what the SDK and the
HTTP stack can and cannot separate.

## The four events, precisely

| Term | The exact moment |
|---|---|
| **dispatch** | The `perf_counter` reading immediately before `async with client.messages.stream(...)`. The client has already been built, so client construction is *not* inside it. |
| **stream open** | When `AsyncMessageStreamManager.__aenter__` returns. It awaits the API request coroutine, which for a streaming call resolves once the response **headers** are in - the body is not read. |
| **response headers available** | `http11.receive_response_headers.complete` from httpcore. This precedes stream open by the SDK's own wrapping. |
| **first text event** | The first `content_block_delta` carrying text, as yielded by `stream.text_stream`. `message_start` and `content_block_start` arrive before it and are not text. |

So `stream_open - dispatch` is everything from "about to send" to "headers are
back", and nothing about the model's writing.

## What can be measured directly

httpcore emits named events when a request carries a `trace` extension. Each
phase below is bounded by two real events, so each is a measurement:

| # | Phase | Bounded by |
|---|---|---|
| 1 | local setup + serialisation | dispatch -> first network event |
| 2 | DNS + TCP connect | `connection.connect_tcp.started/.complete` |
| 3 | TLS handshake | `connection.start_tls.started/.complete` |
| 4 | request upload | `http11.send_request_headers.started` -> `send_request_body.complete` |
| 5 | wait for response headers | `send_request_body.complete` -> `receive_response_headers.complete` |
| 6 | headers -> first text token | SDK stream open -> first text delta |

**Connection reuse is observed, not assumed.** A pooled connection emits no
`connect_tcp`, so its absence is the signal.

## What cannot be measured directly

These stay **inferred buckets** and the report labels them as such:

| Not separable | Why | Best available substitute |
|---|---|---|
| DNS vs TCP connect | httpcore resolves the host inside `connect_tcp`; there is no event between them | A separate `getaddrinfo` timing, which hits a warmed OS cache and so is a lower bound, not the real cost |
| Server processing vs return transit, inside phase 5 | One event pair spans the whole round trip | `x-envoy-upstream-service-time`, if Anthropic's edge sends it, gives upstream time and turns this into a partial measurement. Captured already; absent on some responses |
| Model prefill vs SSE delivery, inside phase 6 | The first text delta is the only observable | None. This is why phase 6 is reported as one number |
| Queueing at Anthropic vs a slow path to it | Indistinguishable from the client | Compare against a request with no model work on the same host (see below) |

## What the audit already established, without spending anything

**Connections are not reused between trials.** Every generator calls
`build_async_client()` and closes it in a `finally`, so each trial builds an
`AsyncAnthropic`, a fresh `DefaultAsyncHttpxClient` and a fresh connection
pool. Pooling is active *within* a client, which is one request, so in practice
every trial pays DNS, TCP and TLS again. Validated over a real TLS socket:
`connection_reused` was `False` on every trial.

Production does the same shape - `ScriptGenerator` builds a client per
instance - so this is worth knowing beyond the benchmark. Nothing has been
changed.

**The first trial is different, and visibly so.** In the local validation,
phase 1 was 16.7 ms on trial 1 and ~1.2 ms afterwards: imports and one-off
setup, not network. The trace shows it rather than leaving it to be guessed at.

## The smallest experiment that separates setup from Anthropic

Re-run the **same 20-trial control** with tracing on. Nothing else changes -
same model, packet, prompt, ceiling and chunk rule - and phases 2, 3 and 4 are
then direct measurements of connection and setup cost. Whatever remains in
phase 5 is Anthropic's, bounded by the round trip.

Optionally, and at no token cost, a handful of TLS handshakes to the same host
before the run gives an independent distribution for phases 2 and 3.
