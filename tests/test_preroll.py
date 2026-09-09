"""The preroll gate: what it is, what it costs, and what it must never allow.

`PREROLL_SECONDS` counts **bytes of audio**, not elapsed time. At
`TARGET_WPM = 150` the shipped 1.5s is 3.75 words, so an ordinary opening
sentence satisfies it on the first chunk and it costs nothing. It bites only on
an unusually short opening - the case the Phase 6 first-chunk rule allows -
where it forces a second synthesis before the listener hears anything.
"""
from __future__ import annotations

import importlib
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as appmod
import config
import pipeline as pipeline_mod
from tts import DebugEngine

BODY = ("Salisbury Cathedral has kept it turning since about thirteen "
        "eighty-six, through a civil war and two restorations.")


class Opening:
    def __init__(self, first: str):
        self.first = first

    async def stream_sentences(self, plan, notes=None):
        import asyncio

        yield self.first
        for _ in range(40):
            await asyncio.sleep(0)
            yield BODY

    async def top_up(self, plan, spoken_so_far, words_needed, notes=None):
        async for sentence in self.stream_sentences(plan):
            yield sentence


@pytest.fixture
def serve(monkeypatch):
    """A client whose episodes begin with a chosen sentence."""
    def build(first: str, preroll: float | None = None):
        if preroll is not None:
            monkeypatch.setattr(appmod, "PREROLL_SECONDS", preroll)
        monkeypatch.setattr(appmod, "SCRIPT_CACHE", None)
        monkeypatch.setattr(appmod, "DEMO_MODE", False)
        monkeypatch.setattr(appmod, "_rate_limit", lambda request: None)
        monkeypatch.setattr(
            appmod, "_make_pipeline",
            lambda voice=None: pipeline_mod.PodcastPipeline(
                generator=Opening(first), engine=DebugEngine(), cache=None,
                voice=voice))
        return TestClient(appmod.app)
    return build


# --------------------------------------------------------------------------
# the setting
# --------------------------------------------------------------------------
def test_the_default_is_unchanged():
    assert config.settings.preroll_seconds == 1.5
    assert appmod.PREROLL_SECONDS == 1.5


def test_it_is_configurable(monkeypatch):
    monkeypatch.setenv("PREROLL_SECONDS", "0.25")
    monkeypatch.setenv("FAM_IGNORE_DOTENV", "1")
    try:
        assert importlib.reload(config).settings.preroll_seconds == 0.25
    finally:
        monkeypatch.delenv("PREROLL_SECONDS", raising=False)
        importlib.reload(config)


@pytest.mark.parametrize("value", ["0", "0.0", "-1"])
def test_zero_and_below_are_refused(monkeypatch, value):
    """Zero is not "no preroll", it is a broken contract: a streamed WAV's
    44-byte header alone satisfies a zero gate, the empty-episode guard then
    fires, and a good episode comes back as a 502."""
    monkeypatch.setenv("PREROLL_SECONDS", value)
    monkeypatch.setenv("FAM_IGNORE_DOTENV", "1")
    try:
        with pytest.raises(ValueError, match="greater than zero"):
            importlib.reload(config)
    finally:
        monkeypatch.delenv("PREROLL_SECONDS", raising=False)
        importlib.reload(config)


# --------------------------------------------------------------------------
# the gate is a quantity, not a delay
# --------------------------------------------------------------------------
def test_an_ordinary_opening_satisfies_the_shipped_preroll_on_one_chunk(serve):
    """Question A: at 1.5 seconds, is the gate already free? For a normal
    opening it is - one chunk in, and nothing waited for."""
    client = serve("The oldest working clock in Europe has no face, and for "
                   "six hundred years nobody thought that was strange.")
    with client.stream("GET", "/api/audio?q=x&minutes=3&fmt=pcm") as response:
        assert response.status_code == 200
        headers = dict(response.headers)
        for _ in response.iter_bytes():
            pass
    assert int(headers["x-chunks-primed"]) == 1
    assert float(headers["x-audio-primed-seconds"]) >= 1.5


def test_a_short_opening_forces_extra_synthesis_at_the_shipped_preroll(serve):
    """Question B: a two-word opening is 0.8s of audio, short of the 1.5s
    gate, so the response waits for a whole further sentence."""
    client = serve("It rang.")
    with client.stream("GET", "/api/audio?q=x&minutes=3&fmt=pcm") as response:
        headers = dict(response.headers)
        for _ in response.iter_bytes():
            pass
    assert int(headers["x-chunks-primed"]) > 1
    assert float(headers["x-audio-primed-seconds"]) > 1.5


def test_lowering_the_preroll_removes_the_extra_synthesis(serve):
    """The same short opening at 0.5s: one chunk, and the listener hears the
    opening rather than the opening plus the sentence after it."""
    client = serve("It rang.", preroll=0.5)
    with client.stream("GET", "/api/audio?q=x&minutes=3&fmt=pcm") as response:
        headers = dict(response.headers)
        for _ in response.iter_bytes():
            pass
    assert int(headers["x-chunks-primed"]) == 1
    assert float(headers["x-audio-primed-seconds"]) == pytest.approx(0.8, abs=0.05)


# --------------------------------------------------------------------------
# the marks
# --------------------------------------------------------------------------
def test_the_measurement_headers_are_present_and_ordered(serve):
    client = serve("The oldest working clock in Europe has no face here.")
    with client.stream("GET", "/api/audio?q=x&minutes=1&fmt=pcm") as response:
        headers = dict(response.headers)
        for _ in response.iter_bytes():
            pass
    assert headers["x-preroll-seconds"] == "1.5"
    first_pcm = float(headers["x-first-pcm-seconds"])
    satisfied = float(headers["x-preroll-satisfied-seconds"])
    assert 0 <= first_pcm <= satisfied
    assert float(headers["x-audio-primed-seconds"]) > 0
    assert int(headers["x-chunks-primed"]) >= 1


# --------------------------------------------------------------------------
# what must not change
# --------------------------------------------------------------------------
@pytest.mark.parametrize("fmt", ["pcm", "wav"])
def test_a_failed_generation_is_still_an_error_not_a_silent_episode(serve, fmt,
                                                                    monkeypatch):
    """The gate's second job. Lowering it must never turn a failure into a
    playable empty episode."""
    class Failing:
        async def stream_sentences(self, plan, notes=None):
            raise RuntimeError("Could not resolve authentication method")
            yield ""  # pragma: no cover

        async def top_up(self, plan, spoken_so_far, words_needed, notes=None):
            return
            yield ""  # pragma: no cover

    client = serve("anything", preroll=0.1)
    monkeypatch.setattr(
        appmod, "_make_pipeline",
        lambda voice=None: pipeline_mod.PodcastPipeline(
            generator=Failing(), engine=DebugEngine(), cache=None, voice=voice))
    response = client.get(f"/api/audio?q=x&minutes=1&fmt={fmt}")
    assert response.status_code == 502
    assert "error" in response.json()


@pytest.mark.parametrize("fmt", ["pcm", "wav"])
def test_an_empty_episode_is_still_an_error_at_a_low_preroll(serve, fmt,
                                                             monkeypatch):
    class Silent:
        async def stream_sentences(self, plan, notes=None):
            return
            yield ""  # pragma: no cover

        async def top_up(self, plan, spoken_so_far, words_needed, notes=None):
            return
            yield ""  # pragma: no cover

    client = serve("anything", preroll=0.1)
    monkeypatch.setattr(
        appmod, "_make_pipeline",
        lambda voice=None: pipeline_mod.PodcastPipeline(
            generator=Silent(), engine=DebugEngine(), cache=None, voice=voice))
    response = client.get(f"/api/audio?q=x&minutes=1&fmt={fmt}")
    assert response.status_code == 502


@pytest.mark.parametrize("fmt", ["pcm", "wav"])
def test_a_good_episode_still_streams_at_a_low_preroll(serve, fmt):
    client = serve("The oldest working clock in Europe has no face here.",
                   preroll=0.1)
    response = client.get(f"/api/audio?q=x&minutes=1&fmt={fmt}")
    assert response.status_code == 200 and len(response.content) > 44
    if fmt == "wav":
        assert response.content[:4] == b"RIFF"
