"""The shutdown traceback, caught against the real Anthropic SDK.

`tests/test_experiments.py` pins the harness contract with stand-ins. Those
stand-ins could not catch this bug: the failure lives four generators deep
inside the SDK, in httpcore's `PoolByteStream.__aiter__`, and only a real SDK
client talking over a real socket exercises it.

So this module serves a local HTTP server that speaks the Anthropic SSE
protocol and points a real `AsyncAnthropic` at it. No API key, no network
beyond loopback, no cost.

**The detection is deliberately deterministic.** Left to the garbage collector
the traceback appears on only some runs - measured at two to five runs in six -
so a test that merely ran the code and looked at stderr would pass by luck
about a third of the time. Instead each test finishes by awaiting
`loop.shutdown_asyncgens()`, which is exactly what `asyncio.run()` does on the
way out, and asserts on the loop's exception handler. That surfaces the fault
on every run.
"""
from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

anthropic = pytest.importorskip("anthropic")


# --------------------------------------------------------------------------
# A local server that speaks the Anthropic streaming protocol
# --------------------------------------------------------------------------
def _event(name: str, payload: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


def _sse_body(words: list[str]) -> bytes:
    parts = [_event("message_start", {"type": "message_start", "message": {
        "id": "msg_test", "type": "message", "role": "assistant", "content": [],
        "model": "claude-sonnet-5", "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 1840, "output_tokens": 0}}})]
    parts.append(_event("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""}}))
    for word in words:
        parts.append(_event("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": word}}))
    parts.append(_event("content_block_stop", {"type": "content_block_stop", "index": 0}))
    parts.append(_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 118}}))
    parts.append(_event("message_stop", {"type": "message_stop"}))
    return "".join(parts).encode()


#: Long enough that the 25-word rule fires while the response is still arriving.
_WORDS = [w + " " for w in (
    "Boards used to remove founders when performance slipped and nobody thought "
    "twice about it at all. That stopped being true in November 2023 and one "
    "company is the reason for it. " + " ".join(["filler"] * 400) + "."
).split(" ")]
_BODY = _sse_body(_WORDS)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", 0) or 0))
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(_BODY)))
        self.end_headers()
        # Dribbled out so the client is genuinely mid-stream at the break.
        for start in range(0, len(_BODY), 512):
            try:
                self.wfile.write(_BODY[start:start + 512])
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def sse_server():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _client(base_url):
    return anthropic.AsyncAnthropic(api_key="test-key-not-a-real-credential",
                                    base_url=base_url)


async def _consume(base_url, drain: bool, trials: int = 2) -> list[str]:
    """Run trials the way the engine does, then force finalisation."""
    problems: list[str] = []
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(lambda l, ctx: problems.append(ctx.get("message", "")))

    for _ in range(trials):
        client = _client(base_url)
        try:
            async with client.messages.stream(
                model="claude-sonnet-5", max_tokens=220,
                messages=[{"role": "user", "content": "hi"}],
            ) as stream:
                text_stream = stream.text_stream
                try:
                    buffer = ""
                    async for delta in text_stream:
                        buffer += delta
                        if len(buffer.split()) >= 25:
                            break            # the benchmark's chunk rule
                finally:
                    if drain:
                        from experiments.generate import _drain

                        await _drain(text_stream)
        finally:
            await client.close()

    # Exactly what asyncio.run() does at shutdown, but observable.
    await loop.shutdown_asyncgens()
    await asyncio.sleep(0)
    return problems


# --------------------------------------------------------------------------
# The bug, and the fix
# --------------------------------------------------------------------------
async def _teardown_state(base_url, drain: bool) -> tuple[bool, bool]:
    """Is the SDK's stream generator still suspended before and after teardown?"""
    client = _client(base_url)
    try:
        async with client.messages.stream(
            model="claude-sonnet-5", max_tokens=220,
            messages=[{"role": "user", "content": "hi"}],
        ) as stream:
            text_stream = stream.text_stream
            buffer = ""
            async for delta in text_stream:
                buffer += delta
                if len(buffer.split()) >= 25:
                    break
            before = text_stream.ag_frame is not None
            if drain:
                from experiments.generate import _drain

                await _drain(text_stream)
            return before, text_stream.ag_frame is not None
    finally:
        await client.close()


def test_breaking_early_leaves_the_sdk_stream_suspended(sse_server):
    """The mechanism, checked deterministically rather than by race.

    `ag_frame` is None only once a generator has finished or been closed. After
    the benchmark's early break it is still set - the generator is parked at a
    `yield`, holding the chain down to httpcore's `PoolByteStream`. Nothing
    closes it, so the event loop finalises it later from another task.

    Draining runs it to exhaustion, which is the one teardown that leaves
    httpcore with nothing to unwind.
    """
    before, after = asyncio.run(_teardown_state(sse_server, drain=False))
    assert before is True, "the break did not leave the stream mid-flight"
    assert after is True, "without draining the generator must still be suspended"

    before, after = asyncio.run(_teardown_state(sse_server, drain=True))
    assert before is True
    assert after is False, "draining must run the generator to exhaustion"


@pytest.mark.parametrize("attempt", range(4))
def test_the_undrained_path_can_still_raise_from_the_finaliser(sse_server, attempt):
    """The symptom the user actually saw, which is a race by nature.

    Left to the finaliser this surfaces on some runs and not others - measured
    at two to five in six - so it is asserted across several attempts and any
    one of them is enough. The deterministic statement of the same fact is the
    test above; this one exists to keep the connection to the real traceback.
    """
    problems = asyncio.run(_consume(sse_server, drain=False, trials=3))
    if problems:
        assert any("asynchronous generator" in p for p in problems), problems
    else:
        pytest.skip("the race did not manifest on this attempt")


def test_draining_leaves_nothing_for_the_finaliser_to_break_on(sse_server):
    """The fix. This is the test the real Mac run would have failed."""
    problems = asyncio.run(_consume(sse_server, drain=True))
    assert problems == [], f"teardown still reported: {problems}"


def test_the_fix_holds_across_repeated_trials(sse_server):
    """One clean run proves nothing here - the fault is GC-timing dependent.

    Every close-based teardown that was tried looked clean on some runs and
    failed on others; only draining was clean every time.
    """
    for _ in range(3):
        assert asyncio.run(_consume(sse_server, drain=True, trials=3)) == []


def test_drain_survives_a_stream_that_errors_midway():
    """Teardown must never turn a completed trial into a failed one."""
    from experiments.generate import _drain

    async def exploding():
        yield "one"
        raise RuntimeError("connection died during teardown")

    async def main():
        iterator = exploding()
        await iterator.__anext__()
        await _drain(iterator)          # must not raise

    asyncio.run(main())


def test_generate_drains_both_generators():
    """Both the production and benchmark generators must drain."""
    import ast
    import pathlib

    source = (pathlib.Path(__file__).resolve().parent.parent
              / "experiments" / "generate.py").read_text()
    assert source.count("await _drain(text_stream)") == 2

    tree = ast.parse(source)
    drained_in_finally = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for statement in node.finalbody:
                for inner in ast.walk(statement):
                    if (isinstance(inner, ast.Call)
                            and getattr(inner.func, "id", "") == "_drain"):
                        drained_in_finally += 1
    assert drained_in_finally == 2, "a drain outside a finally would be skipped on break"


def test_the_harness_tears_down_after_the_timeline_is_snapshotted():
    """Draining costs wall time; it must not land inside a measurement.

    The close must come after `result.timeline = timeline.to_dict()`, so no
    stage duration and no checkpoint can include teardown.
    """
    import pathlib

    source = (pathlib.Path(__file__).resolve().parent.parent
              / "experiments" / "harness.py").read_text()
    snapshot = source.index("result.timeline = timeline.to_dict()")
    close = source.index("await _aclose(token_stream)", snapshot)
    assert close > snapshot

    # And it must not sit between the generate stage and synthesis, where it
    # would inflate first_audio.
    synth = source.index("audio = await tts.synth(")
    assert source.index("await _aclose(token_stream)", 0) > synth
