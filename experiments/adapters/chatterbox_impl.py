"""Chatterbox Turbo, ported from the recovered Runpod benchmarks.

Source of truth: `test_turbo.py` and `fam_chunked_benchmark.py`, both run on an
RTX 4090. They correct an earlier port built from a twelve-line Mac file, which
had the wrong class and — more seriously — no CUDA synchronisation at all.

What the recovered code establishes, and this keeps exactly:

* **`from chatterbox.tts_turbo import ChatterboxTurboTTS`** - Turbo is its own
  module and class, not a checkpoint passed to the base model.
* **`from_pretrained(device="cuda")`**, timed on its own. Loading is not
  synthesis and is reported separately.
* **`torch.cuda.synchronize()` before starting the clock and again after
  `generate` returns, before stopping it.** CUDA queues work asynchronously, so
  without both fences the measured time is whatever it took to *enqueue* the
  work - far too fast, and meaningless. This is the correction that matters
  most; every CUDA number the previous port would have produced was wrong.
* **`with torch.inference_mode():`** around generate in the chunked benchmark.
* **duration = `wav.shape[-1] / model.sr`**, and **speed = duration / gen_time**.
* **`wav.cpu()` after the clock stops**, so the device-to-host copy is not
  counted as generation.
* **$0.75/hour** for the 4090, and cost = generation_seconds / 3600 * rate.
* chunked runs join with **120 ms of silence between chunks and none after the
  last**, and report **"first chunk ready in"** - which is exactly FAM's
  time-to-first-audio.

The two recovered files disagree with each other, deliberately, and both are
reproduced rather than averaged into one house style:

    test_turbo.py            one generate over a whole episode, cold,
                             no warmup, no inference_mode
    fam_chunked_benchmark.py warmup first, inference_mode, per-chunk timing

`SINGLE` and `CHUNKED` below name those two modes.
"""
from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Any, Optional, Sequence

#: `from chatterbox.tts_turbo import ChatterboxTurboTTS` - the recovered import.
TURBO_MODULE = "chatterbox.tts_turbo"
TURBO_CLASS = "ChatterboxTurboTTS"

#: The device both recovered benchmarks used.
BENCHMARK_DEVICE = "cuda"

#: Verbatim from fam_chunked_benchmark.py. The text matters only in that it
#: costs a generate; keeping it identical keeps the warmup cost identical.
WARMUP_TEXT = "This is a warmup."

#: `silence = torch.zeros(1, int(model.sr * 0.12))`
SILENCE_SECONDS = 0.12

#: `GPU_RATE = 0.75` - an RTX 4090 pod, dollars per hour.
GPU_DOLLARS_PER_HOUR = 0.75

#: The two recovered methodologies.
SINGLE = "single"
CHUNKED = "chunked"

_MODELS: dict[str, Any] = {}
_LOAD_SECONDS: dict[str, float] = {}
_WARMED: set[str] = set()


def _torch():
    """The torch module, or None. Indirected so tests can substitute a stub."""
    try:
        import torch
    except ImportError:
        return None
    return torch


def available_devices() -> dict[str, bool]:
    """What this machine can actually run on. No guessing, no silent CPU."""
    out = {"cpu": True, "cuda": False, "mps": False}
    torch = _torch()
    if torch is None:
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

    CPU is never chosen automatically: Chatterbox on CPU is slower than
    realtime, so a silent fallback would report a disastrous number for a model
    that never got a chance.
    """
    devices = available_devices()
    if requested:
        return requested, True
    if devices["cuda"]:
        return "cuda", False
    if devices["mps"]:
        return "mps", False
    return "cpu", False


def synchronize(device: str) -> None:
    """Wait for the GPU to actually finish.

    Both recovered benchmarks fence with `torch.cuda.synchronize()` on each
    side of the timed region. Without it, `perf_counter` measures how long it
    took to *queue* the kernels, which on CUDA is close to instant and produces
    a realtime factor that looks spectacular and means nothing.
    """
    if device != "cuda":
        return
    torch = _torch()
    if torch is None:
        return
    torch.cuda.synchronize()


def _inference_mode(enabled: bool):
    """`torch.inference_mode()` as the chunked benchmark uses it."""
    torch = _torch()
    if not enabled or torch is None:
        return nullcontext()
    return torch.inference_mode()


def load_model(device: str):
    """`ChatterboxTurboTTS.from_pretrained(device=...)`, timed. Cached per device."""
    if device in _MODELS:
        return _MODELS[device], _LOAD_SECONDS.get(device, 0.0)

    import importlib

    module = importlib.import_module(TURBO_MODULE)
    turbo = getattr(module, TURBO_CLASS)

    started = time.perf_counter()
    model = turbo.from_pretrained(device=device)
    elapsed = time.perf_counter() - started

    _MODELS[device] = model
    _LOAD_SECONDS[device] = elapsed
    return model, elapsed


def warm_up(model, device: str) -> None:
    """The recovered warmup: one throwaway generate, then a fence."""
    with _inference_mode(True):
        model.generate(WARMUP_TEXT)
    synchronize(device)
    _WARMED.add(device)


def waveform_seconds(wav, sample_rate: int) -> float:
    """`wav.shape[-1] / model.sr`, exactly as both benchmarks compute it."""
    try:
        frames = int(wav.shape[-1])
    except (AttributeError, IndexError, TypeError):
        frames = len(wav)
    return frames / float(sample_rate) if sample_rate else 0.0


def channels(wav) -> int:
    """How many channels the model returned. FAM streams mono."""
    shape = getattr(wav, "shape", None)
    if shape is None or len(shape) < 2:
        return 1
    return int(shape[0])


def to_cpu(wav):
    """`wav.cpu()` - always after the clock has stopped, never inside it."""
    mover = getattr(wav, "cpu", None)
    return mover() if callable(mover) else wav


def to_pcm16(wav) -> bytes:
    """Waveform tensor -> 16-bit little-endian mono PCM.

    FAM streams raw PCM and writes no audio files. The recovered benchmarks
    save WAVs with `torchaudio.save`, which is right for listening to a sample
    and wrong for a pipeline whose settled constraint is that nothing becomes a
    file. The samples are identical either way.

    Channel 0 is taken explicitly rather than flattening: `flatten()` on a
    (2, N) tensor concatenates the channels and plays left then right at twice
    the length, silently. Values are clamped so an overshoot clips instead of
    wrapping into noise.
    """
    torch = _torch()
    if torch is not None and hasattr(torch, "Tensor") and isinstance(wav, torch.Tensor):
        tensor = wav.detach().to("cpu").float()
        while tensor.dim() > 1:
            tensor = tensor[0]
        tensor = tensor.clamp(-1.0, 1.0)
        return (tensor * 32767.0).to(torch.int16).numpy().tobytes()

    import array

    flat = wav
    while flat and isinstance(flat[0], (list, tuple)):
        flat = flat[0]
    packed = array.array(
        "h", [int(max(-1.0, min(1.0, float(v))) * 32767.0) for v in flat]
    )
    return packed.tobytes()


def silence_pcm(sample_rate: int, seconds: float = SILENCE_SECONDS) -> bytes:
    """`torch.zeros(1, int(model.sr * 0.12))`, as PCM bytes.

    Between chunks only - the recovered loop appends it `if i < len(chunks)`,
    so a joined episode never ends on silence.
    """
    return b"\x00\x00" * int(sample_rate * seconds)


def gpu_cost(generation_seconds: float, rate: float = GPU_DOLLARS_PER_HOUR) -> float:
    """`(generation_time / 3600) * GPU_RATE`, on generation time alone."""
    return (generation_seconds / 3600.0) * rate


def synthesise(
    text: str,
    device: Optional[str] = None,
    warmup: bool = False,
    inference_mode: bool = True,
) -> dict:
    """One timed `generate`, fenced on both sides exactly as recovered.

        synchronize(); start = perf_counter()
        with inference_mode(): wav = model.generate(text)
        synchronize(); gen_time = perf_counter() - start
        wav = wav.cpu()

    `warmup=False, inference_mode=False` reproduces `test_turbo.py`.
    `warmup=True, inference_mode=True` reproduces `fam_chunked_benchmark.py`.
    """
    resolved, explicit = resolve_device(device)
    model, load_seconds = load_model(resolved)

    if warmup and resolved not in _WARMED:
        warm_up(model, resolved)

    cold = resolved not in _WARMED

    synchronize(resolved)
    started = time.perf_counter()
    with _inference_mode(inference_mode):
        wav = model.generate(text)
    synchronize(resolved)
    elapsed = time.perf_counter() - started
    _WARMED.add(resolved)

    # After the clock: the device-to-host copy is not generation.
    wav = to_cpu(wav)
    sample_rate = int(getattr(model, "sr", 0) or 0)
    seconds = waveform_seconds(wav, sample_rate)

    return {
        "pcm": to_pcm16(wav),
        "sample_rate": sample_rate,
        "audio_seconds": seconds,
        "generate_seconds": elapsed,
        "realtime_factor": (seconds / elapsed) if elapsed else None,
        "gpu_cost": gpu_cost(elapsed),
        "device": resolved,
        "device_explicit": explicit,
        "cold": cold,
        "inference_mode": inference_mode,
        "model_load_seconds": load_seconds,
        "chars": len(text),
        "channels": channels(wav),
    }


def synthesise_chunks(
    chunks: Sequence[str],
    device: Optional[str] = None,
    warmup: bool = True,
    inference_mode: bool = True,
) -> dict:
    """`fam_chunked_benchmark.py`, reproduced.

    Each chunk is timed on its own with a fence either side; the pieces are
    joined with 120 ms of silence between them and none trailing; and the
    headline is **first chunk ready in**, which is the moment a listener could
    have started hearing the episode.

    Note one quirk carried over deliberately: `overall_realtime` divides the
    joined duration - silences included - by the total generation time, which
    excludes them. That is what the recovered script computes, and changing it
    would make the numbers incomparable with the run already measured.
    """
    resolved, _ = resolve_device(device)
    model, load_seconds = load_model(resolved)

    if warmup and resolved not in _WARMED:
        warm_up(model, resolved)

    results: list[dict] = []
    pieces: list[bytes] = []
    sample_rate = int(getattr(model, "sr", 0) or 0)
    gap = silence_pcm(sample_rate)

    total_started = time.perf_counter()
    for index, text in enumerate(chunks, 1):
        synchronize(resolved)
        started = time.perf_counter()
        with _inference_mode(inference_mode):
            wav = model.generate(text)
        synchronize(resolved)
        elapsed = time.perf_counter() - started

        wav = to_cpu(wav)
        seconds = waveform_seconds(wav, sample_rate)
        pieces.append(to_pcm16(wav))
        if index < len(chunks):
            pieces.append(gap)

        results.append({
            "chunk": index,
            "generate_seconds": elapsed,
            "audio_seconds": seconds,
            "realtime_factor": (seconds / elapsed) if elapsed else None,
            "chars": len(text),
        })
    total_generation = time.perf_counter() - total_started

    joined = b"".join(pieces)
    joined_seconds = (len(joined) / 2) / sample_rate if sample_rate else 0.0

    return {
        "pcm": joined,
        "sample_rate": sample_rate,
        "chunks": results,
        # The FAM latency number: when the first chunk could start playing.
        "first_chunk_seconds": results[0]["generate_seconds"] if results else None,
        "total_generation_seconds": total_generation,
        "total_audio_seconds": joined_seconds,
        "overall_realtime": (joined_seconds / total_generation) if total_generation else None,
        "gpu_cost": gpu_cost(total_generation),
        "model_load_seconds": load_seconds,
        "device": resolved,
        "chunk_count": len(results),
        "silence_seconds": SILENCE_SECONDS,
    }


def reset() -> None:
    """Drop cached models. For tests; a sweep should never call this."""
    _MODELS.clear()
    _LOAD_SECONDS.clear()
    _WARMED.clear()
