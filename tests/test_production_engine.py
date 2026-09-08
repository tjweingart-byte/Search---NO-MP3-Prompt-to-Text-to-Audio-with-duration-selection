"""Production does not choose a speech engine any more.

Chatterbox is the decided production voice. Until it is integrated the slot in
`tts.PRODUCTION_ENGINES` is empty and an interim engine speaks - which the
health report says out loud, so a stand-in cannot be mistaken for the finished
thing. What has gone is the *selection*: production no longer picks between
Piper, espeak, macOS `say` and a placeholder tone depending on what the host
happens to have installed.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tts


class Fake(tts.TTSEngine):
    """Stands in for Chatterbox, to prove the slot works before it exists."""

    name = "fake-production"

    @staticmethod
    def available() -> bool:
        return True

    @classmethod
    def voices(cls) -> list:
        return [tts.Voice(id="fake:one", label="Fake", engine=cls.name,
                          detail="test double")]

    async def synth(self, text, wpm, voice=None) -> bytes:
        return b"\x00\x00"


# --------------------------------------------------------------------------
# the slot
# --------------------------------------------------------------------------
def test_the_production_slot_is_the_only_thing_production_selects_from(monkeypatch):
    """Fill it and it wins - over the interim engine and over everything the
    host machine happens to have. That is the whole Chatterbox handover."""
    monkeypatch.setattr(tts, "PRODUCTION_ENGINES", (Fake,))
    assert tts.build_engine().name == "fake-production"
    assert tts.production_engine().name == "fake-production"
    assert [v.id for v in tts.list_voices()] == ["fake:one"]
    assert tts.default_voice() == "fake:one"


def test_chatterbox_is_the_only_production_engine():
    assert [cls.name for cls in tts.PRODUCTION_ENGINES] == ["chatterbox"]
    assert tts.engine_report()["production_engines"] == ["chatterbox"]


def test_a_slot_that_cannot_run_falls_through_and_says_so(monkeypatch):
    """Chatterbox gates itself. Where it cannot run - no package, no GPU, no
    reference - the interim voice speaks and the report says so, rather than
    the server pretending to be on the production voice."""
    monkeypatch.setattr(tts.ChatterboxEngine, "_available", False)
    assert tts.production_engine() is None
    assert tts.engine_report()["interim"] is True


def test_a_filled_slot_is_not_reported_as_interim(monkeypatch):
    monkeypatch.setattr(tts, "PRODUCTION_ENGINES", (Fake,))
    report = tts.engine_report()
    assert report["interim"] is False
    assert report["production_engines"] == ["fake-production"]
    assert report["selected"] == "fake-production"


# --------------------------------------------------------------------------
# what production no longer does
# --------------------------------------------------------------------------
def test_espeak_and_say_are_not_production_voices():
    """They exist only if the host OS provides them, which is how a deployment
    ends up sounding worse than the laptop it was built on."""
    offered = {voice.engine for voice in tts.list_voices()}
    assert "espeak" not in offered and "say" not in offered


def test_there_is_no_engine_preference_ladder_left():
    """`build_engine` used to walk Piper, then say, then espeak, then debug.
    Production selecting on what happens to be installed is the thing removed."""
    import inspect

    source = inspect.getsource(tts.build_engine)
    assert "for cls in (PiperEngine, SayEngine, EspeakEngine)" not in source
    assert "PiperEngine" not in source, "the removed engine is back in the ladder"
    assert "PRODUCTION_ENGINES" in source or "production_engine()" in source


def test_piper_is_gone_from_the_runtime_entirely():
    """Not switched off, not demoted to a development engine - removed.

    It held the interim slot, and a second engine that can speak is a second
    engine that can be selected. What is left where it stood is a tone, which
    nobody can mistake for FAM.
    """
    import inspect

    assert not hasattr(tts, "PiperEngine"), "the engine class is back"
    assert "piper" not in {name.lower() for name in tts.DEV_ENGINES}
    assert "Piper" not in inspect.getsource(tts.build_engine)
    assert "Piper" not in inspect.getsource(tts.list_voices)
    assert tts.PLACEHOLDER_ENGINE is tts.DebugEngine


# --------------------------------------------------------------------------
# the development override
# --------------------------------------------------------------------------
def test_tts_engine_is_a_development_override(monkeypatch):
    """Kept because deterministic local tests need an engine that is not a
    GPU. It is documented as development-only in config and .env.example."""
    assert set(tts.DEV_ENGINES) == {"espeak", "say", "debug"}
    assert tts.build_engine("debug").name == "debug"


def test_an_unknown_engine_name_is_refused_rather_than_guessed(monkeypatch):
    with pytest.raises(tts.TTSUnavailable, match="not a known engine"):
        tts.build_engine("chatterbox")


def test_the_deterministic_test_engine_needs_no_gpu_and_no_model():
    """The local suite must keep running with no Chatterbox and no card."""
    assert tts.DebugEngine.available() is True
    assert tts.build_engine("debug").sample_rate > 0
