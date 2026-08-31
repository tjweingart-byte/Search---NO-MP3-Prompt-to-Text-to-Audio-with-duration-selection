"""HTTP-level tests.

These exist because of a real bug: a failure before the first audio byte used to
arrive at the browser as a successful, silent, empty episode. A streaming
response cannot change its status code once it has begun, so the only defence is
to prove there is audio before responding at all.
"""
from __future__ import annotations

import os
import sys
import time

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as appmod  # noqa: E402
import pipeline as pipeline_mod  # noqa: E402
from tests.test_pipeline import FakeGenerator  # noqa: E402
from demo_script import DemoGenerator  # noqa: E402
from tts import DebugEngine  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(appmod, "SCRIPT_CACHE", None)
    # Settings is a frozen dataclass, so disable throttling at the call site.
    monkeypatch.setattr(appmod, "_rate_limit", lambda request: None)
    return TestClient(appmod.app)


def _use(monkeypatch, generator):
    """Point the endpoint at a specific generator, bypassing demo mode."""
    monkeypatch.setattr(appmod, "DEMO_MODE", False)
    monkeypatch.setattr(
        appmod,
        "_make_pipeline",
        lambda voice=None: pipeline_mod.PodcastPipeline(
            generator=generator, engine=DebugEngine(), cache=None, voice=voice
        ),
    )


class SilentGenerator:
    """Produces no sentences at all - e.g. the model returned nothing."""

    async def stream_sentences(self, plan):
        return
        yield ""  # pragma: no cover

    async def top_up(self, plan, spoken_so_far, words_needed):
        return
        yield ""  # pragma: no cover


class FailingGenerator:
    """Fails before the first sentence - e.g. a bad API key."""

    async def stream_sentences(self, plan):
        raise RuntimeError("Could not resolve authentication method")
        yield ""  # pragma: no cover

    async def top_up(self, plan, spoken_so_far, words_needed):
        return
        yield ""  # pragma: no cover


@pytest.mark.parametrize("fmt", ["pcm", "wav"])
def test_a_failure_before_any_audio_is_an_error_not_a_silent_success(client, monkeypatch, fmt):
    _use(monkeypatch, FailingGenerator())
    res = client.get(f"/api/audio?q=anything&minutes=1&fmt={fmt}")
    assert res.status_code == 502, "an auth failure must not arrive as a playable episode"
    assert "error" in res.json()


@pytest.mark.parametrize("fmt", ["pcm", "wav"])
def test_an_empty_episode_is_an_error_not_a_silent_success(client, monkeypatch, fmt):
    _use(monkeypatch, SilentGenerator())
    res = client.get(f"/api/audio?q=anything&minutes=1&fmt={fmt}")
    assert res.status_code == 502
    # A WAV header alone is 44 bytes of no audio; it must not count as success.
    assert res.headers["content-type"].startswith("application/json")


@pytest.mark.parametrize("fmt", ["pcm", "wav"])
def test_a_working_episode_still_streams_audio(client, monkeypatch, fmt):
    _use(monkeypatch, FakeGenerator())
    res = client.get(f"/api/audio?q=anything&minutes=1&fmt={fmt}")
    assert res.status_code == 200
    assert len(res.content) > 44
    if fmt == "wav":
        assert res.content[:4] == b"RIFF"


def test_friendly_error_explains_a_missing_key():
    message = appmod.friendly_error(TypeError("Could not resolve authentication method. Expected..."))
    assert "ANTHROPIC_API_KEY" in message


def test_demo_mode_serves_playable_audio_without_credentials(client, monkeypatch):
    """The audio approach must be judgeable before an API key exists."""
    monkeypatch.setattr(appmod, "DEMO_MODE", True)
    monkeypatch.setattr(
        appmod,
        "_make_pipeline",
        lambda voice=None: pipeline_mod.PodcastPipeline(
            generator=DemoGenerator(), engine=DebugEngine(), cache=None, voice=voice
        ),
    )
    res = client.get("/api/audio?q=anything&minutes=1&fmt=wav")
    assert res.status_code == 200
    seconds = (len(res.content) - 44) / (22050 * 2)
    assert abs(seconds - 60) <= 2, f"demo episode was {seconds:.1f}s"


def test_health_reports_demo_mode(client, monkeypatch):
    monkeypatch.setattr(appmod, "DEMO_MODE", True)
    assert client.get("/api/health").json()["mode"] == "demo"


# --------------------------------------------------------------------------
# Speed is the product
#
# The promise is: type a question, hear the answer within about a second.
# Live web search put 10-25 seconds in front of the first word, which no amount
# of buffering can disguise, so it is off unless a request asks for it.
# --------------------------------------------------------------------------


def test_web_search_is_off_by_default():
    """It is the single biggest cost in time-to-first-word."""
    from script_generator import plan_episode

    assert appmod.settings.enable_web_search is False
    assert plan_episode("anything", 3).search is False


def test_search_can_be_requested_per_episode():
    from script_generator import plan_episode

    assert plan_episode("todays results", 3, search=True).search is True


def test_a_searched_episode_is_cached_separately():
    """An instant answer and a researched one are different episodes."""
    from cache import cache_key

    assert cache_key("a topic", 3, None, "", False) != cache_key("a topic", 3, None, "", True)


def test_the_default_model_is_a_fast_one():
    """Opus took 20-30s to its first sentence; that is not this product."""
    assert appmod.settings.model in {"claude-sonnet-5", "claude-haiku-4-5"}


def test_audio_is_buffered_before_playback_begins():
    """Models stream in bursts; starting on sentence one makes a stall audible."""
    assert appmod.PREROLL_SECONDS >= 1.0


def test_the_prompt_bans_preamble_openings():
    """The reported failure: 'here's what I can tell you about...'"""
    from script_generator import SYSTEM_PROMPT

    assert "Here's what I can tell you about" in SYSTEM_PROMPT
    assert "Banned openings" in SYSTEM_PROMPT


# --------------------------------------------------------------------------
# Prefetch: do the slow work during the pause the user is already taking
# --------------------------------------------------------------------------


def test_prefetch_puts_a_script_in_the_cache(client, monkeypatch):
    """The whole point: pressing play then costs no model call."""
    from cache import MemoryScriptCache, cache_key

    cache = MemoryScriptCache()
    monkeypatch.setattr(appmod, "SCRIPT_CACHE", cache)
    monkeypatch.setattr(appmod, "DEMO_MODE", False)
    monkeypatch.setattr(appmod, "ScriptGenerator", lambda *a, **k: FakeGenerator())

    res = client.post("/api/prefetch", json={"query": "a real question here", "minutes": 1})
    assert res.json()["status"] == "started"

    # No polling: the prefetch runs as a Starlette background task, which the
    # test client runs to completion before returning. If this ever needs a
    # sleep or a retry loop again, the work has become detached from the
    # request and can be garbage collected mid-flight - which is exactly the
    # bug this asserts against.
    key = cache_key("a real question here", 1, None, "", False)
    assert cache.get(key), "the prefetched script should be waiting in the cache"


def test_prefetch_reports_ready_when_it_already_has_one(client, monkeypatch):
    from cache import MemoryScriptCache, cache_key

    cache = MemoryScriptCache()
    cache.put(cache_key("a real question here", 1, None, "", False), ["Done."], 60, "q")
    monkeypatch.setattr(appmod, "SCRIPT_CACHE", cache)
    monkeypatch.setattr(appmod, "DEMO_MODE", False)

    res = client.post("/api/prefetch", json={"query": "a real question here", "minutes": 1})
    assert res.json()["status"] == "ready"


def test_prefetch_never_stores_a_personal_query(client, monkeypatch):
    from cache import MemoryScriptCache

    cache = MemoryScriptCache()
    monkeypatch.setattr(appmod, "SCRIPT_CACHE", cache)
    monkeypatch.setattr(appmod, "DEMO_MODE", False)

    res = client.post("/api/prefetch", json={"query": "summarise my lab results", "minutes": 1})
    assert res.json()["status"] == "skipped"


def test_prefetch_failure_is_invisible(client, monkeypatch):
    """A failed prefetch must never surface; the listener just takes the slow path."""
    from cache import MemoryScriptCache

    class Broken:
        async def stream_sentences(self, plan):
            raise RuntimeError("model down")
            yield ""  # pragma: no cover

    monkeypatch.setattr(appmod, "SCRIPT_CACHE", MemoryScriptCache())
    monkeypatch.setattr(appmod, "DEMO_MODE", False)
    monkeypatch.setattr(appmod, "ScriptGenerator", lambda *a, **k: Broken())

    res = client.post("/api/prefetch", json={"query": "a real question here", "minutes": 1})
    assert res.status_code == 200


def test_a_prefetch_cannot_be_garbage_collected_mid_flight(client, monkeypatch):
    """Regression: the work must be owned by the request, not detached.

    `asyncio.create_task` was used originally. The event loop keeps only a weak
    reference to a task, so a fire-and-forget prefetch could be collected
    part-way through and silently never finish - invisible in production
    because a failed prefetch is deliberately silent, and flaky in tests.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(appmod.prefetch)))
    calls = {
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "asyncio.create_task" not in calls, (
        "prefetch must not detach its work from the request; use BackgroundTasks"
    )
    assert "background.add_task" in calls


def test_a_finished_prefetch_makes_the_episode_a_cache_hit(client, monkeypatch):
    """End to end: prefetching is what removes the wait, so prove it removes it."""
    from cache import MemoryScriptCache

    import pipeline as pipeline_mod

    cache = MemoryScriptCache()
    monkeypatch.setattr(appmod, "SCRIPT_CACHE", cache)
    monkeypatch.setattr(appmod, "DEMO_MODE", False)
    monkeypatch.setattr(appmod, "ScriptGenerator", lambda *a, **k: FakeGenerator())

    client.post("/api/prefetch", json={"query": "a real question here", "minutes": 1})

    stats = {}

    def make(voice=None):
        pipe = pipeline_mod.PodcastPipeline(
            generator=FailingGenerator(), engine=DebugEngine(), cache=cache, voice=voice
        )
        stats["pipeline"] = pipe
        return pipe

    # The generator would raise if it were called; a cache hit means it is not.
    monkeypatch.setattr(appmod, "_make_pipeline", make)
    res = client.get("/api/audio?q=a real question here&minutes=1&fmt=pcm")
    assert res.status_code == 200, "a prefetched episode must play without the model"
    assert len(res.content) > 0
