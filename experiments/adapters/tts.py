"""Voice arms: the shipped local engines, and Chatterbox on a GPU.

Piper is the baseline and stays the default; nothing here replaces it. The
local adapter is a thin wrapper over production `tts.py`, so what gets measured
is the engine the app actually uses rather than a copy of it.

Chatterbox is remote by nature. The adapter is a **client, not a controller**:
it can POST to an endpoint that already exists, and it has no code path that
creates, starts, resumes or scales anything. If no endpoint is configured it
refuses the run and says GPU infrastructure is required. That refusal is the
feature - a benchmark that silently spins up a 4090 is a benchmark that spends
money nobody approved.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

from experiments.adapters.base import Availability, InfrastructureRequired, SynthResult
from experiments.timeline import Timeline

#: Environment variable holding an already-running Chatterbox endpoint.
CHATTERBOX_ENDPOINT_ENV = "CHATTERBOX_ENDPOINT"

#: `GPU_RATE = 0.75` from the recovered Runpod benchmarks: an RTX 4090 pod,
#: dollars per hour. Used for the pre-run estimate, and for the recorded cost
#: on generation time alone, exactly as the benchmark computes it.
GPU_DOLLARS_PER_HOUR = 0.75


class LocalTTS:
    """Whatever speech engine this machine has, through production `tts.py`.

    `voice` selects among the engines the app already exposes (piper, espeak,
    say, debug), so "Piper vs Chatterbox" is Piper exactly as it ships.
    """

    id = "piper"
    label = "Piper / local engine (current production)"
    host = "local"

    def __init__(self, voice: Optional[str] = None) -> None:
        self.voice = voice

    def _engine(self, voice: Optional[str]):
        from tts import engine_for_voice

        return engine_for_voice(voice or self.voice)

    def available(self) -> Availability:
        """Honest about *which* engine would actually speak.

        `tts.py` always has the debug tone available as a last resort, so a
        naive check reports "yes" on a machine with no real voice and the run
        would quietly measure a square wave while the report said "Piper".
        Silent success like that has cost this project more time than any real
        bug, so the debug engine is refused as a measurement arm unless it is
        asked for by name.
        """
        try:
            from tts import DebugEngine, engine_for_voice, list_voices
        except Exception as exc:
            return Availability(ok=False, reason=f"tts.py did not import: {exc}")
        voices = list_voices()
        if not voices:
            return Availability(
                ok=False,
                reason="This machine has no speech engine installed.",
                remedy="python setup_voices.py",
            )
        try:
            engine = engine_for_voice(self.voice)
        except Exception as exc:
            return Availability(ok=False, reason=f"No engine for voice {self.voice!r}: {exc}")
        if isinstance(engine, DebugEngine) and not (self.voice or "").startswith("debug"):
            return Availability(
                ok=False,
                reason=(
                    "The only speech engine on this machine is the debug tone. "
                    "Timing it would produce a number that looks like Piper and is not."
                ),
                remedy="python setup_voices.py   (or set voice=debug:tone to time the tone deliberately)",
            )
        return Availability(ok=True, reason=f"engine: {type(engine).__name__}")

    async def synth(self, text: str, timeline: Timeline, **params) -> SynthResult:
        from audio_utils import pcm_duration
        from config import settings

        voice = params.get("voice") or self.voice
        wpm = float(params.get("wpm") or settings.target_wpm)
        engine = self._engine(voice)
        started = time.perf_counter()
        with timeline.span("synthesis", host=self.host, adapter=self.id, voice=voice) as stage:
            pcm = await engine.synth(text, wpm, voice)
            stage.detail["bytes"] = len(pcm)
        wall = time.perf_counter() - started
        rate = engine.sample_rate
        seconds = pcm_duration(len(pcm), sample_rate=rate)
        return SynthResult(
            pcm=pcm,
            sample_rate=rate,
            audio_seconds=seconds,
            cost=0.0,  # local compute; the electricity is not billed to this project
            detail={"wall_seconds": wall, "engine": type(engine).__name__, "voice": voice},
        )


class ChatterboxTTS:
    """Chatterbox Turbo on a GPU that is already running, somewhere else.

    Configuration is one environment variable, `CHATTERBOX_ENDPOINT`, pointing
    at a live HTTP endpoint. There is intentionally no Runpod SDK import, no
    API key for pod control, and no create/start/stop call anywhere in this
    file. Spinning a pod up is a human action taken deliberately, outside this
    tool.

    Expected contract, so the endpoint can be anything that honours it::

        POST {endpoint}
        {"text": str, "sample_rate": int}          # rate is a hint only
        -> {"pcm_base64": str,                     # 16-bit LE mono PCM, no header
            "sample_rate": int,                    # the MODEL's rate (model.sr)
            "gpu_seconds": float,                  # optional
            "device": str,                         # optional, e.g. "cuda"
            "cold": bool}                          # optional, first generate?

    The response's `sample_rate` wins. `test_chatterbox.py` takes the rate from
    `model.sr`, so the model decides it and a caller that imposed its own would
    be resampling or mislabelling. The request's value is a hint the endpoint
    may ignore.

    `gpu_seconds` is optional and is recorded as the remote's own view of its
    time. It never replaces the wall time measured here, because the difference
    between them is the cost of the GPU being on another machine.

    A reference endpoint honouring this contract is in
    `experiments/adapters/chatterbox_server_example.py`; it wraps the same
    `synthesise()` this repo uses locally, so both arms run identical code.
    """

    id = "chatterbox"
    label = "Chatterbox Turbo (remote GPU)"
    host = "runpod-gpu"

    def endpoint(self) -> str:
        return os.environ.get(CHATTERBOX_ENDPOINT_ENV, "").strip()

    def available(self) -> Availability:
        if not self.endpoint():
            return Availability(
                ok=False,
                needs_approval=True,
                reason=(
                    "No Chatterbox endpoint is configured, so this experiment "
                    "requires GPU infrastructure that is not currently running."
                ),
                remedy=(
                    "Start a GPU pod yourself (this tool will not), then set "
                    f"{CHATTERBOX_ENDPOINT_ENV}=https://<host>/synthesise. "
                    f"An RTX 4090 pod bills roughly ${GPU_DOLLARS_PER_HOUR:.2f}/hour "
                    "while it is running."
                ),
            )
        return Availability(ok=True)

    async def synth(self, text: str, timeline: Timeline, **params) -> SynthResult:
        import base64
        import urllib.request

        url = self.endpoint()
        if not url:
            raise InfrastructureRequired(
                adapter=self.id,
                what="a Chatterbox GPU endpoint is required and none is configured",
                how=(
                    "This tool does not start, stop or pay for GPU infrastructure. "
                    f"Start a pod, then set {CHATTERBOX_ENDPOINT_ENV}."
                ),
            )

        from config import settings

        rate = int(params.get("sample_rate") or settings.sample_rate)
        payload = json.dumps({"text": text, "sample_rate": rate}).encode()
        started = time.perf_counter()
        with timeline.span("synthesis", host=self.host, adapter=self.id) as stage:
            request = urllib.request.Request(
                url, data=payload, headers={"Content-Type": "application/json"}
            )
            timeout = float(params.get("timeout") or 60)
            # Blocking call on purpose: trials run sequentially, so there is no
            # loop to starve, and asyncio's HTTP stack would add its own timing.
            with urllib.request.urlopen(request, timeout=timeout) as reply:
                body = json.loads(reply.read().decode())
            stage.remote_seconds = body.get("gpu_seconds")
            stage.detail["bytes"] = len(body.get("pcm_base64", "")) * 3 // 4
            for key in ("device", "cold"):
                if key in body:
                    stage.detail[key] = body[key]
        wall = time.perf_counter() - started

        pcm = base64.b64decode(body.get("pcm_base64", ""))
        rate = int(body.get("sample_rate", rate))
        from audio_utils import pcm_duration

        seconds = pcm_duration(len(pcm), sample_rate=rate)
        gpu_seconds = body.get("gpu_seconds")
        return SynthResult(
            pcm=pcm,
            sample_rate=rate,
            audio_seconds=seconds,
            cost=(gpu_seconds or wall) / 3600.0 * GPU_DOLLARS_PER_HOUR,
            remote_seconds=gpu_seconds,
            detail={
                "wall_seconds": wall,
                "endpoint_configured": True,
                "device": body.get("device"),
                "cold": body.get("cold"),
            },
        )


class ChatterboxLocal:
    """Chatterbox in this process, exactly as `test_chatterbox.py` ran it.

    This is the arm that needs no GPU rental: on the Mac the manual test was
    run on, `device="mps"` reproduces it. On a machine with a card it is
    `device="cuda"`, which is the same code path the Runpod endpoint runs
    behind HTTP - so local and remote arms differ in where they run, not in
    what they do.

    Nothing here provisions anything. It loads a model already on the machine.
    """

    id = "chatterbox_local"
    label = "Chatterbox Turbo (in-process)"
    host = "local"

    def __init__(self, device: str | None = None) -> None:
        self.device = device

    def available(self) -> Availability:
        from experiments.adapters import chatterbox_impl

        try:
            import importlib

            importlib.import_module(chatterbox_impl.TURBO_MODULE)
        except ImportError:
            return Availability(
                ok=False,
                reason=(
                    f"{chatterbox_impl.TURBO_MODULE} is not importable. Turbo is "
                    "its own module, not a checkpoint of the base model."
                ),
                remedy="pip install -r experiments/requirements-chatterbox.txt",
            )
        resolved, explicit = chatterbox_impl.resolve_device(self.device)
        if resolved == "cpu" and not explicit:
            return Availability(
                ok=False,
                reason=(
                    "No GPU or Metal device is available, so this would run on "
                    "CPU - slower than realtime, which measures the machine "
                    "rather than the model."
                ),
                remedy=(
                    'Run on a machine with "mps" (Apple Silicon) or "cuda", or '
                    'ask for CPU deliberately with params={"device": "cpu"}.'
                ),
            )
        return Availability(ok=True, reason=f"device: {resolved}")

    async def synth(self, text: str, timeline: Timeline, **params) -> SynthResult:
        import asyncio

        from experiments.adapters import chatterbox_impl

        device = params.get("device") or self.device
        # The chunked benchmark warms up and uses inference_mode; test_turbo.py
        # does neither. Both are reachable, and the trial records which ran.
        warmup = bool(params.get("warmup", True))
        inference_mode = bool(params.get("inference_mode", True))

        with timeline.span("synthesis", host=self.host, adapter=self.id) as stage:
            # Off the event loop: inference is blocking work, and blocking the
            # loop here would distort nothing today but would the moment
            # anything else needs to run alongside it.
            out = await asyncio.to_thread(
                chatterbox_impl.synthesise, text, device, warmup, inference_mode
            )
            stage.detail.update({
                "device": out["device"],
                "cold": out["cold"],
                "channels": out["channels"],
                "inference_mode": out["inference_mode"],
                "bytes": len(out["pcm"]),
                "model_load_seconds": round(out["model_load_seconds"], 3),
            })

        return SynthResult(
            pcm=out["pcm"],
            sample_rate=out["sample_rate"],
            audio_seconds=out["audio_seconds"],
            # On a rented card the generate time bills; on your own it does not.
            # Recorded either way, on generation time alone as the benchmark does.
            cost=out["gpu_cost"] if out["device"] == "cuda" else 0.0,
            detail={
                "wall_seconds": out["generate_seconds"],
                "device": out["device"],
                "cold": out["cold"],
                "channels": out["channels"],
                "inference_mode": out["inference_mode"],
                "model_load_seconds": out["model_load_seconds"],
                "realtime_factor": out["realtime_factor"],
                "chars": out["chars"],
            },
        )


class NoTTS:
    """Script-only arm: measure the writing without synthesising it."""

    id = "none"
    label = "no speech"
    host = "local"

    def available(self) -> Availability:
        return Availability(ok=True)

    async def synth(self, text: str, timeline: Timeline, **params) -> SynthResult:
        return SynthResult(pcm=b"", audio_seconds=0.0)


TTS_ADAPTERS = {
    a.id: a for a in (NoTTS(), LocalTTS(), ChatterboxTTS(), ChatterboxLocal())
}
