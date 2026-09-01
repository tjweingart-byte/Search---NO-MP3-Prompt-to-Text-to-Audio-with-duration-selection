"""HTTP-level tests.

These exist because of a real bug: a failure before the first audio byte used to
arrive at the browser as a successful, silent, empty episode. A streaming
response cannot change its status code once it has begun, so the only defence is
to prove there is audio before responding at all.
"""
from __future__ import annotations

import os
import sys

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

    async def stream_sentences(self, plan, notes=None):
        return
        yield ""  # pragma: no cover

    async def top_up(self, plan, spoken_so_far, words_needed):
        return
        yield ""  # pragma: no cover


class FailingGenerator:
    """Fails before the first sentence - e.g. a bad API key."""

    async def stream_sentences(self, plan, notes=None):
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
    assert "Banned outright" in SYSTEM_PROMPT
    # Scene-setting is the other way a piece fails to get to the point.
    assert "picture this" in SYSTEM_PROMPT.lower()



# --- The go-deeper thread -------------------------------------------------


def test_the_thread_endpoint_serves_what_the_episode_left_open(client, monkeypatch):
    """Generating an episode leaves a follow-up suggestion behind it."""
    from cache import MemoryScriptCache

    class Threaded:
        client = None

        async def stream_sentences(self, plan, notes=None):
            if notes is not None:
                notes.thread = "whether the appeal is heard at all"
            yield "She filed the appeal on Tuesday morning."

        async def top_up(self, plan, spoken_so_far, words_needed):
            return
            yield ""  # pragma: no cover

    shared = MemoryScriptCache()
    monkeypatch.setattr(appmod, "DEMO_MODE", False)
    monkeypatch.setattr(
        appmod,
        "_make_pipeline",
        lambda voice=None: pipeline_mod.PodcastPipeline(
            generator=Threaded(), engine=DebugEngine(), cache=shared, voice=voice
        ),
    )

    assert client.get("/api/next?q=the+appeal&minutes=1").json()["thread"] == ""
    assert client.get("/api/audio?q=the+appeal&minutes=1&fmt=pcm").status_code == 200
    assert (
        client.get("/api/next?q=the+appeal&minutes=1").json()["thread"]
        == "whether the appeal is heard at all"
    )


def test_a_missing_thread_is_an_empty_string_not_an_error(client, monkeypatch):
    """No suggestion just means the blank Go Deeper field, never a failure."""
    _use(monkeypatch, FakeGenerator())
    res = client.get("/api/next?q=anything&minutes=1")
    assert res.status_code == 200
    assert res.json()["thread"] == ""


def test_the_marker_never_reaches_the_script_endpoint_as_speech(client, monkeypatch):
    class Marked:
        client = None

        async def stream_sentences(self, plan, notes=None):
            from script_generator import clean_for_speech, extract_thread

            raw = "The rule expires in March. <<NEXT: what replaces the rule>>"
            if notes is not None:
                notes.thread = extract_thread(raw)
            yield clean_for_speech(raw)

    monkeypatch.setattr(appmod, "DEMO_MODE", False)
    monkeypatch.setattr(appmod, "ScriptGenerator", lambda: Marked())
    body = client.post("/api/script", json={"query": "the rule", "minutes": 1}).json()
    assert "NEXT" not in body["script"]
    assert body["thread"] == "what replaces the rule"
