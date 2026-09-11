"""A development engine must never be selectable by a request.

The bug this pins: `voice=debug:tone` from a browser returned the placeholder
tone on a machine where Chatterbox was loaded and working. `DebugEngine
.available()` is unconditionally True and routing went through `ENGINES`, which
is `DEV_ENGINES`, so the query string won. `/api/health` kept reporting
`selected: chatterbox`, because that asks `build_engine()`, which never sees the
parameter - so the interface said "voice: chatterbox" while a 220 Hz sine with
a 2 Hz envelope played. That is what a listener calls humming.

A page acquires the id honestly: `list_voices` offers the placeholder whenever
the production slot is empty, so a tab opened during warm-up, or before the
reference recording was in place, keeps sending it for the rest of the session.

CLAUDE.md already names this failure for Piper - an engine reached listeners
three ways nobody chose, one of them `engine_for_voice`.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tts  # noqa: E402


class Speaking(tts.TTSEngine):
    """Stands in for Chatterbox on a machine that can actually speak."""

    name = "chatterbox"

    @classmethod
    def available(cls) -> bool:
        return True

    @classmethod
    def voices(cls):
        return [tts.Voice(id="chatterbox:reference_3", label="FAM",
                          engine="chatterbox")]

    async def synth(self, text, wpm, voice=None):
        return b""


@pytest.fixture
def speaking(monkeypatch):
    monkeypatch.setattr(tts, "PRODUCTION_ENGINES", (Speaking,))
    return Speaking


@pytest.fixture
def mute(monkeypatch):
    """A machine with no production voice at all."""
    monkeypatch.setattr(tts, "PRODUCTION_ENGINES", ())


def test_a_request_cannot_choose_the_placeholder_over_a_real_voice(speaking):
    """The bug, as a test. This returned "debug" and played a buzz."""
    assert tts.engine_for_voice("debug:tone").name == "chatterbox"


def test_no_development_engine_is_reachable_by_voice_id(speaking):
    """Not just the tone: nothing in DEV_ENGINES may be named into service."""
    for name in tts.DEV_ENGINES:
        assert tts.engine_for_voice(name + ":anything").name == "chatterbox", (
            f"a request selected the development engine {name!r}")


def test_the_production_voice_still_routes_to_the_production_engine(speaking):
    assert tts.engine_for_voice("chatterbox:reference_3").name == "chatterbox"


def test_an_unknown_or_missing_voice_falls_back_to_production(speaking):
    assert tts.engine_for_voice("nonsense:whatever").name == "chatterbox"
    assert tts.engine_for_voice(None).name == "chatterbox"
    assert tts.engine_for_voice("").name == "chatterbox"


def test_a_machine_that_cannot_speak_still_serves_the_tone(mute, monkeypatch):
    """The counterpart guard. Refusing the id must not make the app mute:
    with no production engine the placeholder is the honest answer, not an
    override, and `/api/health` reports `interim: true` beside it."""
    monkeypatch.setattr(tts.ChatterboxEngine, "_available", False)
    assert tts.engine_for_voice("debug:tone").name == tts.PLACEHOLDER_ENGINE.name
    assert tts.engine_for_voice(None).name == tts.PLACEHOLDER_ENGINE.name


def test_the_server_side_override_still_works(mute, monkeypatch):
    """`TTS_ENGINE=debug` is a decision made by whoever started the server, not
    by a query string, so it keeps working - that is the line being drawn."""
    import dataclasses
    monkeypatch.setattr(tts, "settings",
                        dataclasses.replace(tts.settings, tts_engine="debug"))
    assert tts.build_engine().name == "debug"
