"""Piper: what is true of *this* engine and no other.

Piper is no longer a production voice. It is the explicitly labelled interim
occupant of the production slot - `engine_report()` reports `interim: true`
wherever it is what speaks - and it will go when Chatterbox has been heard on
real hardware.

That changes what belongs here. The guarantees this file used to hold that are
really about *any* engine running inference in a streaming server - that
synthesis does not block the event loop, that the model is loaded once rather
than per sentence, that a stale voice id never becomes silence - moved to
`tests/test_engine_contract.py`, which applies them to Chatterbox too. They
were never Piper facts; they were being asserted for the interim voice and
assumed for the production one.

What is left is genuinely Piper-only: its naming, its discovery, and its rate
control. The last of those has no counterpart at all - Chatterbox exposes no
speaking-rate control, so length is held by the budget and by trimming at a
sentence boundary instead (see `tts.ChatterboxEngine`).

**`piper` is not a test dependency.** It is an optional package for an interim
voice, so the tests that need it skip where it is absent rather than failing.
Nothing here should ever be a reason to install it.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tts  # noqa: E402
from tts import PiperEngine, _prettify_piper_name  # noqa: E402


def use_voices_dir(monkeypatch, directory, piper_model=""):
    """Settings is a frozen dataclass, so swap in a modified copy."""
    monkeypatch.setattr(
        tts,
        "settings",
        dataclasses.replace(tts.settings, voices_dir=str(directory), piper_model=piper_model),
    )


@pytest.fixture
def a_model(tmp_path, monkeypatch):
    """A voice store containing one model. No package required to look."""
    model = tmp_path / "en_US-lessac-medium.onnx"
    model.write_bytes(b"not-a-real-model")
    (tmp_path / "en_US-lessac-medium.onnx.json").write_text(
        json.dumps({"audio": {"sample_rate": 22050}})
    )
    use_voices_dir(monkeypatch, tmp_path)
    PiperEngine._loaded.clear()
    return model


# --------------------------------------------------------------------------
# naming and discovery - no package needed to do either
# --------------------------------------------------------------------------
def test_installed_voices_are_named_for_humans():
    assert _prettify_piper_name("en_US-lessac-medium") == "Lessac (US, medium)"
    assert _prettify_piper_name("en_GB-alba-medium") == "Alba (GB, medium)"


def test_models_are_found_in_the_shared_store(a_model):
    """Where setup_voices.py writes is where the engine looks. This is path
    resolution, not speech, so it holds with or without the package."""
    assert [p.stem for p in PiperEngine.installed_models()] == ["en_US-lessac-medium"]


def test_piper_is_unavailable_when_no_model_is_installed(tmp_path, monkeypatch):
    use_voices_dir(monkeypatch, tmp_path)
    assert PiperEngine.available() is False
    assert PiperEngine.voices() == []


def test_a_model_without_the_package_is_still_unavailable(a_model, monkeypatch):
    """The two halves are separate, and both are required.

    A machine can have the model files and not the package - that is exactly
    what a checkout without `pip install -r requirements.txt` looks like, and
    what a machine that never wanted the interim voice looks like. Offering the
    voice on the strength of the files alone would put a voice in the picker
    that cannot speak.
    """
    # None in sys.modules makes `import piper` raise, which is what a machine
    # without the package does - without having to uninstall anything.
    monkeypatch.setitem(sys.modules, "piper", None)
    assert PiperEngine.available() is False
    assert PiperEngine.voices() == [], "a voice that cannot speak was offered"


# --------------------------------------------------------------------------
# what needs the package
# --------------------------------------------------------------------------
@pytest.fixture
def installed_voice(a_model, monkeypatch):
    """A loadable voice. Skipped where the interim package is not installed."""
    pytest.importorskip(
        "piper", reason="piper is the interim voice; it is not a test dependency")

    class FakeChunk:
        def __init__(self, data, rate):
            self.audio_int16_bytes = data
            self.sample_rate = rate

    class FakeVoice:
        def __init__(self):
            self.calls: list = []

        def synthesize(self, text, syn_config=None, **kwargs):
            self.calls.append((text, syn_config))
            yield FakeChunk(b"\x01\x00" * len(text), 22050)

    fake = FakeVoice()
    monkeypatch.setattr(PiperEngine, "_load", classmethod(lambda cls, path: fake))
    return a_model, fake


def test_a_voice_in_the_project_is_discovered(installed_voice):
    voices = PiperEngine.voices()
    assert [v.id for v in voices] == ["piper:en_US-lessac-medium"]
    assert voices[0].engine == "piper"


def test_a_slower_rate_asks_for_longer_audio(installed_voice):
    """Piper's rate control, and the reason the duration contract has two
    halves at all.

    Chatterbox has no equivalent: `wpm` is accepted and ignored there, so an
    episode's length is held by the budget and by trimming at a sentence
    boundary rather than by speaking faster. Whichever engine is in the slot,
    the ceiling is enforced - but only this one can hit it by pacing.
    """
    model, fake = installed_voice
    engine = PiperEngine(model)
    asyncio.run(engine.synth("Words.", 120))
    asyncio.run(engine.synth("Words.", 185))
    slow, fast = fake.calls[0][1].length_scale, fake.calls[1][1].length_scale
    assert slow > fast, "a lower words-per-minute must stretch the audio"
    assert 0.6 <= fast <= 1.6 and 0.6 <= slow <= 1.6, "scale must stay listenable"


def test_the_sample_rate_comes_from_the_model_not_the_config(installed_voice):
    """The header is written before any audio exists, from this number."""
    model, _ = installed_voice
    assert PiperEngine(model).sample_rate == 22050
