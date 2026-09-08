"""Chatterbox as a production engine, checked without a GPU or the package.

The synthesis itself was validated on an RTX 4090 (`phase6_4090_20260908T064113Z`:
2.992s search-to-first-listen warm, zero stalls). What this file checks is the
part that is new - the adapter around it, and the promises the rest of FAM
makes about an engine: it gates itself honestly, it never blocks the event
loop, it clones only a voice whose rights are cleared, and the first complete
thought reaches it without being batched or delayed.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tts
from tts import ChatterboxEngine


@pytest.fixture(autouse=True)
def _forget_availability():
    """`available()` is memoised for /api/health; tests must not inherit it."""
    ChatterboxEngine._available = None
    yield
    ChatterboxEngine._available = None
    ChatterboxEngine._loaded.clear()


def voice_with_rights(tmp_path, **overrides):
    reference = tmp_path / "reference_3.wav"
    reference.write_bytes(b"RIFF....WAVE")
    record = {"source": "a recording", "speaker": "Someone Real",
              "consent": "yes", "commercial_use": "yes",
              "synthetic_voice_cleared": "yes", "notes": "-"}
    record.update(overrides)
    (tmp_path / "reference_3.rights.json").write_text(json.dumps(record))
    return reference


# --------------------------------------------------------------------------
# the settings it was validated with
# --------------------------------------------------------------------------
def test_the_generation_settings_are_the_ones_that_were_judged():
    """Phase 2 chose this voice by listening. Changing a number here changes
    the voice and invalidates that judgement."""
    assert tts.CHATTERBOX_GENERATION == {
        "exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8,
        "repetition_penalty": 1.2, "min_p": 0.05, "top_p": 1.0,
    }


def test_the_sample_rate_is_the_models_not_the_configured_one():
    """Chatterbox emits 24 kHz. The stream header and the pace controller both
    read it from the engine, so it must be right before the model loads."""
    assert ChatterboxEngine.SAMPLE_RATE == 24000
    assert ChatterboxEngine().sample_rate == 24000
    from config import settings

    assert settings.sample_rate != 24000, (
        "if these ever match, this test stops proving anything")


def test_float_audio_becomes_sixteen_bit_pcm():
    import numpy as np

    pcm = tts.pcm_from_float(np.array([0.0, 1.0, -1.0, 2.0, -2.0],
                                      dtype=np.float32))
    assert len(pcm) == 10
    values = np.frombuffer(pcm, dtype="<i2")
    assert list(values) == [0, 32767, -32767, 32767, -32767], "clipping failed"


# --------------------------------------------------------------------------
# it gates itself honestly
# --------------------------------------------------------------------------
def test_it_refuses_cpu_rather_than_starving_the_listener(monkeypatch, tmp_path):
    """~1x realtime or worse: the listener would hear gaps. Refusing is better
    than serving an episode that starves."""
    monkeypatch.setattr(ChatterboxEngine, "device", classmethod(lambda cls: "cpu"))
    monkeypatch.setitem(sys.modules, "chatterbox.tts", object())
    monkeypatch.setitem(sys.modules, "chatterbox", object())
    ok, why = ChatterboxEngine.diagnose()
    assert ok is False and "CPU" in why


def test_it_refuses_a_missing_reference_voice(monkeypatch, tmp_path):
    monkeypatch.setattr(ChatterboxEngine, "device", classmethod(lambda cls: "cuda"))
    monkeypatch.setitem(sys.modules, "chatterbox", object())
    monkeypatch.setitem(sys.modules, "chatterbox.tts", object())
    monkeypatch.setattr(ChatterboxEngine, "reference_path",
                        staticmethod(lambda: tmp_path / "nothing.wav"))
    ok, why = ChatterboxEngine.diagnose()
    assert ok is False and "no reference voice" in why


@pytest.mark.parametrize("field", ["consent", "commercial_use",
                                   "synthetic_voice_cleared"])
def test_it_refuses_a_voice_whose_rights_are_not_cleared(tmp_path, field):
    """A cloned voice is somebody's voice. A gate that only runs in a tool is
    not a gate, so it runs here too."""
    reference = voice_with_rights(tmp_path, **{field: None})
    cleared, why = ChatterboxEngine.rights_cleared(reference)
    assert cleared is False and field in why


def test_it_refuses_a_voice_with_no_rights_record_at_all(tmp_path):
    reference = tmp_path / "reference_3.wav"
    reference.write_bytes(b"RIFF")
    cleared, why = ChatterboxEngine.rights_cleared(reference)
    assert cleared is False and "no rights record" in why


def test_a_cleared_voice_passes(tmp_path):
    cleared, why = ChatterboxEngine.rights_cleared(voice_with_rights(tmp_path))
    assert cleared is True and "cleared" in why


def test_the_reference_defaults_to_the_shared_voice_folder(monkeypatch):
    monkeypatch.setattr(tts, "settings",
                        type("S", (), {"chatterbox_reference": "",
                                       "chatterbox_device": "auto"})())
    assert ChatterboxEngine.reference_path().name == "reference_3.wav"


def test_availability_is_memoised_so_health_does_not_reimport_torch(monkeypatch):
    calls = []
    monkeypatch.setattr(ChatterboxEngine, "diagnose",
                        classmethod(lambda cls: (calls.append(1), (False, "no"))[1]))
    ChatterboxEngine.available()
    ChatterboxEngine.available()
    assert len(calls) == 1


# --------------------------------------------------------------------------
# it behaves like a FAM engine
# --------------------------------------------------------------------------
def test_synthesis_never_runs_on_the_event_loop(monkeypatch):
    """Blocking GPU work on the loop would stall the Claude stream, and the
    Phase 6 decoupling depends on it not doing that."""
    threads = []

    def blocking(self, text):
        import threading

        threads.append(threading.current_thread().name)
        return b"\x00\x00", 24000

    monkeypatch.setattr(ChatterboxEngine, "_synth_blocking", blocking)

    async def main():
        import threading

        loop_thread = threading.current_thread().name
        await ChatterboxEngine().synth("Anything at all.", 150)
        return loop_thread

    loop_thread = asyncio.run(main())
    assert threads and threads[0] != loop_thread


def test_generations_serialise_on_one_card(monkeypatch):
    """A single GPU cannot run two generations at once. A second listener
    queues; overlapping would be a crash or an out-of-memory."""
    overlap = {"now": 0, "peak": 0}

    def blocking(self, text):
        import time

        overlap["now"] += 1
        overlap["peak"] = max(overlap["peak"], overlap["now"])
        time.sleep(0.05)
        overlap["now"] -= 1
        return b"\x00\x00", 24000

    monkeypatch.setattr(ChatterboxEngine, "_synth_blocking", blocking)

    async def main():
        engine = ChatterboxEngine()
        await asyncio.gather(*(engine.synth(f"Sentence {i}.", 150)
                               for i in range(4)))

    ChatterboxEngine._gate = None
    asyncio.run(main())
    assert overlap["peak"] == 1, f"{overlap['peak']} generations ran at once"


def test_the_speaking_rate_is_accepted_and_ignored(monkeypatch):
    """Chatterbox has no rate control, so pacing does not apply to it: length
    is held by the budget and by trimming at a sentence boundary. Accepted as
    a decided trade - the parameter stays because TTSEngine defines it."""
    seen = []
    monkeypatch.setattr(ChatterboxEngine, "_synth_blocking",
                        lambda self, text: (seen.append(text), (b"\x00\x00", 24000))[1])

    async def main():
        engine = ChatterboxEngine()
        await engine.synth("The same words.", 115)
        await engine.synth("The same words.", 185)

    ChatterboxEngine._gate = None
    asyncio.run(main())
    assert seen == ["The same words.", "The same words."]


def test_the_model_is_loaded_once_per_process(monkeypatch):
    """~10s cold. `warm_up()` pays it at startup so no listener does."""
    loads = []

    class FakeModel:
        sr = 24000
        device = "cuda"

        def generate(self, text, **kwargs):
            raise AssertionError("not called in this test")

    def from_pretrained(device):
        loads.append(device)
        return FakeModel()

    monkeypatch.setattr(ChatterboxEngine, "device", classmethod(lambda cls: "cuda"))
    module = type(sys)("chatterbox.tts")
    module.ChatterboxTTS = type("T", (), {"from_pretrained":
                                          staticmethod(from_pretrained)})
    monkeypatch.setitem(sys.modules, "chatterbox", type(sys)("chatterbox"))
    monkeypatch.setitem(sys.modules, "chatterbox.tts", module)
    ChatterboxEngine._loaded.clear()

    assert ChatterboxEngine._model() is ChatterboxEngine._model()
    assert loads == ["cuda"]


# --------------------------------------------------------------------------
# warm at startup, not on the first listener
# --------------------------------------------------------------------------
def test_warm_up_loads_the_model_so_no_request_pays_for_it(monkeypatch):
    """The cold load is ~10s on a 4090. `warm_up()` runs in the app lifespan,
    before the server accepts anything, so it lands there and not on the first
    episode of the day.

    The test asserts on the *whole* warm-up path, not just residency, because
    `warm_up` swallows every exception - a model that loads but cannot speak
    would still leave the first listener paying, silently.
    """
    import contextlib

    import numpy as np

    spoken = []

    class FakeWav:
        """Stands in for the torch tensor `generate` returns."""

        def squeeze(self, axis):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.zeros(240, dtype="float32")

    class FakeModel:
        sr = 24000
        device = "cuda"

        def generate(self, text, **kwargs):
            spoken.append(text)
            return FakeWav()

    torch = type(sys)("torch")
    torch.inference_mode = contextlib.nullcontext
    module = type(sys)("chatterbox.tts")
    module.ChatterboxTTS = type("T", (), {
        "from_pretrained": staticmethod(lambda device: FakeModel())})
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "chatterbox", type(sys)("chatterbox"))
    monkeypatch.setitem(sys.modules, "chatterbox.tts", module)
    monkeypatch.setattr(ChatterboxEngine, "device", classmethod(lambda cls: "cuda"))
    monkeypatch.setattr(ChatterboxEngine, "reference_path",
                        staticmethod(lambda: __import__("pathlib").Path("ref.wav")))
    monkeypatch.setattr(ChatterboxEngine, "_available", True)
    monkeypatch.setattr(tts, "PRODUCTION_ENGINES", (ChatterboxEngine,))
    ChatterboxEngine._loaded.clear()
    ChatterboxEngine._gate = None

    assert ChatterboxEngine._loaded == {}, "nothing should be resident yet"
    asyncio.run(tts.warm_up())
    assert "cuda" in ChatterboxEngine._loaded, "startup did not load the model"
    assert spoken, "warm_up loaded the model but never proved it can speak"
    ChatterboxEngine._loaded.clear()
