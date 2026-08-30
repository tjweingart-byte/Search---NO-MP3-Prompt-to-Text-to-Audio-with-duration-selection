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
