"""What every engine FAM speaks with must do, whichever engine it is.

These guarantees used to live in `tests/test_piper.py`, where they were written
against the engine that happened to be speaking at the time. Two things were
wrong with that. They read as facts about Piper when they are facts about
*inference in a streaming server* - and they were never applied to Chatterbox,
which is the engine that will actually run. So the guarantee that matters most
here, that inference does not block the event loop, was being asserted for the
interim voice and assumed for the production one.

It matters most for Chatterbox. Phase 6's whole claim is that Claude keeps
writing while the voice is generating; if `synth` ran its inference on the loop
instead of a worker thread, the reader could not advance during it and the
decoupling would be a lie that every timing measurement would still report as
true.

Nothing here needs a GPU, a model file, an API key, or any optional package.
An engine whose package is not installed is skipped, not failed: `piper` is an
interim fallback and `chatterbox` needs a card, so neither is a test
dependency.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import pathlib
import sys
import threading
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tts  # noqa: E402
from tts import ChatterboxEngine, DebugEngine, PiperEngine  # noqa: E402


# --------------------------------------------------------------------------
# Building each engine with its inference stubbed, and a record of the thread
# that inference ran on.
# --------------------------------------------------------------------------
class Ran:
    """Where and how often the stubbed inference was called."""

    def __init__(self):
        self.threads: list[str] = []
        self.loads = 0

    @property
    def calls(self) -> int:
        return len(self.threads)


def build_debug(monkeypatch, tmp_path) -> tuple:
    return DebugEngine(), Ran()


def build_chatterbox(monkeypatch, tmp_path) -> tuple:
    """The real engine, with `chatterbox` and `torch` stubbed at the import."""
    ran = Ran()

    class FakeWav:
        def squeeze(self, axis):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            import numpy as np

            return np.zeros(2400, dtype="float32")

    class FakeModel:
        sr = 24000
        device = "cuda"

        def generate(self, text, **kwargs):
            ran.threads.append(threading.current_thread().name)
            return FakeWav()

    def from_pretrained(device):
        ran.loads += 1
        return FakeModel()

    torch = types.ModuleType("torch")
    torch.inference_mode = contextlib.nullcontext
    module = types.ModuleType("chatterbox.tts")
    module.ChatterboxTTS = type("T", (), {
        "from_pretrained": staticmethod(from_pretrained)})
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "chatterbox", types.ModuleType("chatterbox"))
    monkeypatch.setitem(sys.modules, "chatterbox.tts", module)
    monkeypatch.setattr(ChatterboxEngine, "device", classmethod(lambda cls: "cuda"))
    monkeypatch.setattr(ChatterboxEngine, "reference_path",
                        staticmethod(lambda: tmp_path / "reference_3.wav"))
    monkeypatch.setattr(ChatterboxEngine, "_available", True)
    ChatterboxEngine._loaded.clear()
    ChatterboxEngine._gate = None
    return ChatterboxEngine(), ran


def build_piper(monkeypatch, tmp_path) -> tuple:
    """Skipped where the package is absent - it is an interim fallback, and
    installing a deep-learning-free voice must not be a test dependency."""
    pytest.importorskip("piper", reason="piper is the interim voice, not a "
                                        "test dependency")
    import dataclasses

    ran = Ran()
    model = tmp_path / "en_US-lessac-medium.onnx"
    model.write_bytes(b"not-a-real-model")
    (tmp_path / "en_US-lessac-medium.onnx.json").write_text(
        json.dumps({"audio": {"sample_rate": 22050}}))
    monkeypatch.setattr(tts, "settings", dataclasses.replace(
        tts.settings, voices_dir=str(tmp_path), piper_model=""))

    class FakeChunk:
        def __init__(self, data, rate):
            self.audio_int16_bytes = data
            self.sample_rate = rate

    class FakeVoice:
        def synthesize(self, text, syn_config=None, **kwargs):
            ran.threads.append(threading.current_thread().name)
            yield FakeChunk(b"\x01\x00" * len(text), 22050)

    def fake_load(path):
        ran.loads += 1
        return FakeVoice()

    # `PiperVoice.load`, not `PiperEngine._load` - `_load` is the thing that
    # does the caching, so stubbing it would stub out what is being asserted.
    import piper

    PiperEngine._loaded.clear()
    monkeypatch.setattr(piper.PiperVoice, "load", staticmethod(fake_load))
    return PiperEngine(model), ran


BUILDERS = {"debug": build_debug, "chatterbox": build_chatterbox,
            "piper": build_piper}

#: Engines that run a model. The event-loop and model-caching contracts are
#: about inference, so they do not apply to the tone generator.
INFERENCE = ("chatterbox", "piper")


@pytest.fixture(params=sorted(BUILDERS))
def engine(request, monkeypatch, tmp_path):
    built, ran = BUILDERS[request.param](monkeypatch, tmp_path)
    built.ran = ran
    built.engine_key = request.param
    return built


@pytest.fixture(params=INFERENCE)
def inference_engine(request, monkeypatch, tmp_path):
    built, ran = BUILDERS[request.param](monkeypatch, tmp_path)
    built.ran = ran
    built.engine_key = request.param
    return built


# --------------------------------------------------------------------------
# the contract
# --------------------------------------------------------------------------
def test_synth_returns_sixteen_bit_pcm(engine):
    """No files, no containers, no headers: the stream is raw samples."""
    pcm = asyncio.run(engine.synth("Hello there, this is a sentence.", 150))
    assert pcm, f"{engine.name} produced no audio at all"
    assert len(pcm) % 2 == 0, "16-bit samples cannot have an odd byte count"


def test_the_sample_rate_is_known_before_the_first_chunk(engine):
    """The WAV header and the player's scheduler are both written from this,
    and both are sent before any audio exists."""
    rate = engine.sample_rate
    assert isinstance(rate, int) and rate > 0, f"{engine.name} rate was {rate!r}"


def test_an_unknown_voice_id_is_never_silence(engine):
    """A listener whose chosen voice was uninstalled still gets their episode."""
    pcm = asyncio.run(engine.synth("Some words here.", 150, "gone:missing"))
    assert pcm, f"{engine.name} answered a stale voice id with silence"


def test_inference_does_not_run_on_the_event_loop(inference_engine):
    """The guarantee Phase 6 rests on.

    Claude's reader and the assembler are coroutines on this loop. Inference
    run directly on it would stop them for the duration of every chunk - and
    the marks would still report a decoupled pipeline, because the timestamps
    would all be taken after the loop resumed.
    """
    asyncio.run(inference_engine.synth("A sentence to speak.", 150))
    threads = inference_engine.ran.threads
    assert threads, f"{inference_engine.name}: inference never ran"
    assert all(name != "MainThread" for name in threads), (
        f"{inference_engine.name} ran inference on the event loop, in "
        f"{threads}")


def test_the_model_is_loaded_once_not_per_sentence(inference_engine):
    """Chatterbox is ~10s cold and Piper ~1s. Per-sentence loading would
    dominate everything else in the pipeline."""
    for _ in range(4):
        asyncio.run(inference_engine.synth("A sentence.", 150))
    assert inference_engine.ran.calls == 4, "the stub did not see every call"
    assert inference_engine.ran.loads == 1, (
        f"{inference_engine.name} loaded its model "
        f"{inference_engine.ran.loads} times; it must be cached")


def test_every_production_engine_is_covered_by_this_contract():
    """A new production engine must not be able to skip these by existing.

    The interim engine is included deliberately: while it is what speaks, it
    is held to the same guarantees.
    """
    covered = set(BUILDERS)
    for cls in tts.PRODUCTION_ENGINES + (tts.INTERIM_ENGINE,):
        assert cls.name in covered, (
            f"{cls.name} can speak in production but has no contract builder "
            f"here. Add one; do not narrow the contract.")


def test_the_reference_is_a_path_the_engine_can_be_pointed_at(tmp_path,
                                                              monkeypatch):
    """Chatterbox clones a recording, so the contract has one extra term: the
    voice is per-machine state, not something baked into the image."""
    import dataclasses

    monkeypatch.setattr(tts, "settings", dataclasses.replace(
        tts.settings, chatterbox_reference=str(tmp_path / "somewhere.wav")))
    assert ChatterboxEngine.reference_path() == tmp_path / "somewhere.wav"
    assert isinstance(ChatterboxEngine.reference_path(), pathlib.Path)
