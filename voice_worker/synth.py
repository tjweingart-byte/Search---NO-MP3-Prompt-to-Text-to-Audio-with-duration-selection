"""The GPU half of the split, and the only place synthesis actually happens.

`remote_voice.py` runs on the CPU host and owns the transport. This runs on the
card and owns the audio. Between them is the JSON contract in
`remote_voice.WIRE_FORMAT`, which both entrypoints in this package serve:

    voice_worker/handler.py   RunPod Serverless  (sleeps when idle)
    voice_worker/server.py    a plain HTTP POST  (an always-on pod)

## Why this wraps `ChatterboxEngine` instead of calling the model

The generation settings are the voice. `tts.CHATTERBOX_GENERATION` carries the
six numbers Phase 2 chose and Phase 6 measured, with a comment saying that
changing one invalidates every listening judgement made on it - so a worker
that reimplemented `generate()` would be a second place for them to drift, and
the drift would be inaudible until someone compared two episodes side by side.

It also inherits, for free and without a second copy to forget:

* the **rights gate** - no consent record beside the reference recording, no
  synthesis. A cloned voice is somebody's voice, on rented hardware too.
* the **CPU refusal** - Chatterbox on a CPU is slower than speech, so an
  episode would starve mid-sentence. A worker that came up on a CPU node must
  fail loudly at boot, not serve a starving episode.
* the **resident model** - loaded once per process, not once per request.

So the voice a listener hears through this worker is the voice they would hear
from the same code in-process. That is the whole point of the split: it changes
where the card is and what it costs, not what comes out of it.
"""
from __future__ import annotations

import base64
import logging
import os
import sys
import time

# The worker image ships the whole repo (as `Dockerfile.gpu` does), so the
# engine and its settings are importable rather than duplicated.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tts import ChatterboxEngine  # noqa: E402


log = logging.getLogger(__name__)

#: The only format this worker speaks. Matches `remote_voice.WIRE_FORMAT`;
#: asserted on every request rather than assumed, because the two halves are
#: deployed separately and can be different versions of this repo.
WIRE_FORMAT = "pcm_s16le"

#: How much text one request may carry. A generous ceiling on top of
#: `speech_assembly`'s 45-word cap - it exists to refuse a runaway payload, not
#: to second-guess the assembler, which is the only thing that should be
#: deciding chunk size.
MAX_CHARACTERS = 4000


class WorkerError(RuntimeError):
    """Something the caller can act on, phrased for whoever reads the log."""


def preflight() -> tuple[bool, str]:
    """Can this machine speak? Asked at boot, before any request arrives.

    "Verify, do not inspect" (PROBLEMS.md §52): this is the difference between
    "chatterbox is installed" and "this worker will produce audio", and it is
    the check that stops a GPU-less node from accepting jobs it can only fail.
    """
    return ChatterboxEngine.diagnose()


async def synthesise(payload: dict) -> dict:
    """One chunk of text in, one wire-contract reply out.

    Raises `WorkerError` with a reason for anything a caller could fix. The
    entrypoints turn that into their own error shape; neither ever answers with
    silence or with empty audio, because an empty episode that arrives as a
    success is the failure this project has lost the most time to.
    """
    if not isinstance(payload, dict):
        raise WorkerError(f"expected a JSON object, got {type(payload).__name__}")

    text = (payload.get("text") or "").strip()
    wanted_format = str(payload.get("format") or WIRE_FORMAT).lower()
    if wanted_format != WIRE_FORMAT:
        raise WorkerError(
            f"this worker speaks {WIRE_FORMAT}; {wanted_format!r} was asked for")

    # A wake-up job. It exists to boot the container and load the model, so it
    # deliberately does no synthesis and bills only the load it was sent to pay.
    if payload.get("warm") or not text:
        ready, detail = preflight()
        if not ready:
            raise WorkerError(detail)
        model_loaded = await _load()
        return {"warm": True, "ready": True, "engine": "chatterbox",
                "sample_rate": model_loaded, "detail": detail}

    if len(text) > MAX_CHARACTERS:
        raise WorkerError(
            f"{len(text)} characters exceeds the {MAX_CHARACTERS}-character "
            "ceiling for one request; the caller should be chunking this")

    ready, detail = preflight()
    if not ready:
        raise WorkerError(detail)

    engine = ChatterboxEngine()
    started = time.monotonic()
    pcm = await engine.synth(text, wpm=0.0, voice=payload.get("voice"))
    elapsed = time.monotonic() - started
    rate = engine.sample_rate

    if not pcm:
        raise WorkerError("chatterbox produced no audio for a non-empty script")

    asked = payload.get("sample_rate")
    if asked and int(asked) != int(rate):
        # Reported, never resampled. The app writes its stream header from its
        # own configured rate before the first request, so a silent correction
        # here would play at the wrong pitch on the listener's device with
        # nothing anywhere saying why.
        log.warning("caller asked for %s Hz; this model emits %s Hz", asked, rate)

    seconds = len(pcm) / float(rate * 2)
    log.info("synthesised %d words -> %.1fs audio in %.2fs (%.1fx realtime)",
             len(text.split()), seconds, elapsed,
             seconds / elapsed if elapsed else 0.0)

    return {
        "audio": base64.b64encode(pcm).decode("ascii"),
        "sample_rate": int(rate),
        "format": WIRE_FORMAT,
        "samples": len(pcm) // 2,
        "audio_seconds": round(seconds, 3),
        "synth_seconds": round(elapsed, 3),
        "engine": "chatterbox",
    }


async def _load() -> int:
    """Pay the model load now. Returns the rate the loaded model emits.

    A warm-up that only imported the package would report ready while the ~10s
    load still stood between the next listener and their first word, which is
    the same cheaper-question failure as checking that a key is set.
    """
    engine = ChatterboxEngine()
    await engine.synth("Ready.", wpm=0.0)
    return int(engine.sample_rate)
