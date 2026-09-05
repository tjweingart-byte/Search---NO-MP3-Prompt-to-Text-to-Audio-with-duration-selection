"""Chatterbox in-process, ported from `test_chatterbox.py`.

What the manual test established, and this keeps:

* the model is `chatterbox.tts.ChatterboxTTS`, loaded with
  `from_pretrained(device=...)`;
* synthesis is a single `model.generate(text)` returning a waveform tensor;
* the sample rate is the model's own `model.sr`, never a value we choose.

What the manual test did **not** contain, and this therefore does not claim to
have recovered: any timing, any duration arithmetic, and any realtime factor.
The test wrote a WAV and printed "Done". Those numbers are derived here from
the tensor and the orchestrator's clock, and they are new measurements rather
than a reproduction of a previous one.

Two things this adds deliberately, because a repeated-trial run makes them
matter in a way one manual run did not:

**Model loading is not synthesis.** `from_pretrained` costs seconds and happens
once; folding it into trial one would make the first trial look catastrophic
and every later one look fast. It is timed separately and reported once.

**The first generate after a load is cold.** Lazy kernel compilation makes it
slower than the rest, and averaging it in silently is how a voice gets judged
on its worst run. Each trial records whether it was cold; `warmup=True` does a
throwaway generate first. The default is `False`, which is what the manual test
did.
"""
from __future__ import annotations

import time
from typing import Any, Optional

#: The manual test's device. Apple Silicon Metal - note this is a *Mac* run,
#: not the RTX 4090; "cuda" is the Runpod equivalent.
BENCHMARK_DEVICE = "mps"

#: Loaded models, keyed by device. Loading is expensive and a sweep reuses one.
_MODELS: dict[str, Any] = {}
#: How long each device's load took, reported once per run.
_LOAD_SECONDS: dict[str, float] = {}
#: Devices that have produced at least one waveform, so "cold" is knowable.
_WARMED: set[str] = set()


def available_devices() -> dict[str, bool]:
    """What this machine can actually run on. No guessing, no silent CPU."""
    out = {"cpu": True, "cuda": False, "mps": False}
    try:
        import torch
    except ImportError:
        return out
    try:
        out["cuda"] = bool(torch.cuda.is_available())
    except Exception:
        pass
    try:
        out["mps"] = bool(torch.backends.mps.is_available())
    except Exception:
        pass
    return out


def resolve_device(requested: Optional[str] = None) -> tuple[str, bool]:
    """The device to use, and whether the caller named it explicitly.

    CPU is never chosen automatically. Chatterbox on CPU is slower than
    realtime, so a run that quietly fell back to it would report a disastrous
    number for a model that was never given a chance - the same silent-success
    failure that made the local Piper adapter refuse the debug tone.
    """
    devices = available_devices()
    if requested:
        return requested, True
    if devices["cuda"]:
        return "cuda", False
    if devices["mps"]:
        return "mps", False
    return "cpu", False


def load_model(device: str):
    """Load once per device and keep it. Returns (model, load_seconds)."""
    if device in _MODELS:
        return _MODELS[device], _LOAD_SECONDS.get(device, 0.0)
    from chatterbox.tts import ChatterboxTTS as _Chatterbox

    started = time.perf_counter()
    model = _Chatterbox.from_pretrained(device=device)
    elapsed = time.perf_counter() - started
    _MODELS[device] = model
    _LOAD_SECONDS[device] = elapsed
    return model, elapsed


def waveform_seconds(wav, sample_rate: int) -> float:
    """Seconds of audio in the tensor the model returned.

    Derived from the tensor's own frame count rather than from encoded bytes,
    so it stays exact regardless of how the audio is later packed.
    """
    try:
        frames = int(wav.shape[-1])
    except (AttributeError, IndexError, TypeError):
        frames = len(wav)
    return frames / float(sample_rate) if sample_rate else 0.0


def to_pcm16(wav) -> bytes:
    """Waveform tensor -> 16-bit little-endian PCM.

    FAM streams raw PCM and writes no audio files, so the tensor is converted
    rather than saved. `test_chatterbox.py` wrote a WAV with `torchaudio.save`;
    that was right for listening to one sample and wrong for a pipeline whose
    settled constraint is that nothing becomes a file.

    Values are clamped before scaling: a model that overshoots [-1, 1] would
    otherwise wrap around into loud noise instead of clipping.
    """
    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and isinstance(wav, torch.Tensor):
        tensor = wav.detach().to("cpu").float()
        # Take channel 0 explicitly rather than flattening. `flatten()` on a
        # (2, N) tensor concatenates the channels, which plays left then right
        # at twice the length - wrong audio produced silently. FAM is mono
        # throughout, so a multi-channel model would be a real finding, and
        # `channels()` below is what surfaces it instead of hiding it.
        while tensor.dim() > 1:
            tensor = tensor[0]
        tensor = tensor.clamp(-1.0, 1.0)
        return (tensor * 32767.0).to(torch.int16).numpy().tobytes()

    # Without torch (tests, and any caller handing us a plain sequence).
    import array

    flat = wav
    while flat and isinstance(flat[0], (list, tuple)):
        flat = flat[0]
    packed = array.array(
        "h", [int(max(-1.0, min(1.0, float(v))) * 32767.0) for v in flat]
    )
    return packed.tobytes()


def channels(wav) -> int:
    """How many channels the model returned.

    FAM streams mono. Anything else is recorded on the trial so it is visible
    rather than quietly mixed down to the first channel.
    """
    shape = getattr(wav, "shape", None)
    if shape is None or len(shape) < 2:
        return 1
    return int(shape[0])


def synthesise(text: str, device: Optional[str] = None, warmup: bool = False) -> dict:
    """One `model.generate(text)`, timed. The measurement, minus the loading.

    Returns the PCM, the model's own sample rate, the audio duration, the wall
    time of the generate call alone, and enough context to read the number
    honestly: which device ran it, whether the model was cold, and how long the
    one-off load took.
    """
    resolved, explicit = resolve_device(device)
    model, load_seconds = load_model(resolved)

    if warmup and resolved not in _WARMED:
        model.generate("Warming up.")
        _WARMED.add(resolved)

    cold = resolved not in _WARMED

    started = time.perf_counter()
    wav = model.generate(text)
    elapsed = time.perf_counter() - started
    _WARMED.add(resolved)

    sample_rate = int(getattr(model, "sr", 0) or 0)
    pcm = to_pcm16(wav)
    seconds = waveform_seconds(wav, sample_rate)

    return {
        "pcm": pcm,
        "sample_rate": sample_rate,
        "audio_seconds": seconds,
        "generate_seconds": elapsed,
        "realtime_factor": (seconds / elapsed) if elapsed else None,
        "device": resolved,
        "device_explicit": explicit,
        "cold": cold,
        "model_load_seconds": load_seconds,
        "chars": len(text),
        "channels": channels(wav),
    }


def reset() -> None:
    """Drop cached models. For tests; a sweep should never call this."""
    _MODELS.clear()
    _LOAD_SECONDS.clear()
    _WARMED.clear()
