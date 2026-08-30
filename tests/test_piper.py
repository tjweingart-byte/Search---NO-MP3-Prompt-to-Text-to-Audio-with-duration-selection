"""The neural voice that ships with the app.

The model files are hosted somewhere the build machine could not reach, so the
ONNX inference itself is exercised by `verify_voice.py` on a real machine. Every
other part - discovery, naming, routing, model caching, rate control, sample
rate, and the promise not to block the event loop - is tested here against a
stub, so the untested surface is as small as it can be.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tts  # noqa: E402
from tts import PiperEngine, _prettify_piper_name  # noqa: E402


def use_voices_dir(monkeypatch, directory, piper_model=""):
    """Settings is a frozen dataclass, so swap in a modified copy."""
    import dataclasses

    monkeypatch.setattr(
        tts,
        "settings",
        dataclasses.replace(tts.settings, voices_dir=str(directory), piper_model=piper_model),
    )


class FakeChunk:
    def __init__(self, data: bytes, rate: int):
        self.audio_int16_bytes = data
        self.sample_rate = rate


class FakeVoice:
    """Stands in for a loaded Piper model."""

    loads = 0

    def __init__(self, rate: int = 22050):
        self.rate = rate
        self.calls: list = []

    def synthesize(self, text, syn_config=None, **kwargs):
        self.calls.append((text, syn_config, threading.current_thread().name))
        # One byte of audio per character, so length is predictable.
        yield FakeChunk(b"\x01\x00" * len(text), self.rate)


@pytest.fixture
def installed_voice(tmp_path, monkeypatch):
    """A voices/ directory containing one model, with loading stubbed out."""
    model = tmp_path / "en_US-lessac-medium.onnx"
    model.write_bytes(b"not-a-real-model")
    (tmp_path / "en_US-lessac-medium.onnx.json").write_text(
        json.dumps({"audio": {"sample_rate": 22050}})
    )
    use_voices_dir(monkeypatch, tmp_path)

    fake = FakeVoice()
    PiperEngine._loaded.clear()
    monkeypatch.setattr(PiperEngine, "_load", classmethod(lambda cls, path: fake))
    return model, fake


def test_a_voice_in_the_project_is_discovered(installed_voice):
    voices = PiperEngine.voices()
    assert [v.id for v in voices] == ["piper:en_US-lessac-medium"]
    assert voices[0].engine == "piper"


def test_installed_voices_are_named_for_humans():
    assert _prettify_piper_name("en_US-lessac-medium") == "Lessac (US, medium)"
    assert _prettify_piper_name("en_GB-alba-medium") == "Alba (GB, medium)"


def test_synthesis_produces_pcm(installed_voice):
    model, fake = installed_voice
    engine = PiperEngine(model)
    pcm = asyncio.run(engine.synth("Hello there.", 150, "piper:en_US-lessac-medium"))
    assert pcm == b"\x01\x00" * len("Hello there.")
    assert engine.sample_rate == 22050


def test_synthesis_runs_off_the_event_loop(installed_voice):
    """Inference is blocking CPU work; on the loop it would stall every listener."""
    model, fake = installed_voice
    engine = PiperEngine(model)
    asyncio.run(engine.synth("Some words.", 150))
    _, _, thread_name = fake.calls[0]
    assert thread_name != "MainThread", "synthesis must not run on the event loop"


def test_a_slower_rate_asks_for_longer_audio(installed_voice):
    """The pacing controller hits the clock by moving length_scale."""
    model, fake = installed_voice
    engine = PiperEngine(model)
    asyncio.run(engine.synth("Words.", 120))
    asyncio.run(engine.synth("Words.", 185))
    slow, fast = fake.calls[0][1].length_scale, fake.calls[1][1].length_scale
    assert slow > fast, "a lower words-per-minute must stretch the audio"
    assert 0.6 <= fast <= 1.6 and 0.6 <= slow <= 1.6, "scale must stay listenable"


def test_the_model_is_loaded_once_not_per_sentence(tmp_path, monkeypatch):
    """Model load is ~1s. Per-sentence loading would dominate the pipeline."""
    model = tmp_path / "en_US-lessac-medium.onnx"
    model.write_bytes(b"x")
    use_voices_dir(monkeypatch, tmp_path)
    PiperEngine._loaded.clear()

    loads = {"n": 0}

    def fake_load(path):
        loads["n"] += 1
        return FakeVoice()

    import piper

    monkeypatch.setattr(piper.PiperVoice, "load", staticmethod(fake_load))
    engine = PiperEngine(model)
    for _ in range(4):
        asyncio.run(engine.synth("A sentence.", 150))
    assert loads["n"] == 1, f"model loaded {loads['n']} times; must be cached"


def test_an_unknown_piper_voice_falls_back_to_the_default(installed_voice):
    model, fake = installed_voice
    engine = PiperEngine(model)
    pcm = asyncio.run(engine.synth("Hi.", 150, "piper:not-installed"))
    assert pcm, "an uninstalled voice must not produce silence"


def test_piper_is_unavailable_when_no_model_is_installed(tmp_path, monkeypatch):
    use_voices_dir(monkeypatch, tmp_path)
    assert PiperEngine.available() is False
    assert PiperEngine.voices() == []


def test_piper_is_preferred_over_espeak_when_present(installed_voice, monkeypatch):
    """The whole point: the bundled voice wins over whatever the OS has."""
    monkeypatch.setattr(PiperEngine, "available", staticmethod(lambda: True))
    ids = [v.id for v in tts.list_voices()]
    assert ids[0].startswith("piper:"), f"piper should lead, got {ids[:2]}"
