"""Text-to-speech engines that emit raw PCM.

Design rule for this project: no engine is allowed to produce a file, and no
step encodes MP3. Each engine takes a short chunk of text plus a speaking rate
and returns 16-bit PCM bytes that are written straight to the HTTP response.

The production engine, Chatterbox, is in-process and keeps one model resident
for the life of the server: it is a GPU model that costs ~10s to load, so a
process per sentence would dominate everything else. The development engines
below are subprocess-based, which is why the abstraction has both shapes.
"""
from __future__ import annotations

import asyncio
import json
import logging
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from abc import ABC, abstractmethod
from typing import Optional

import voice_store
from audio_utils import strip_wav_header
from config import settings


log = logging.getLogger(__name__)


class TTSUnavailable(RuntimeError):
    """No usable speech engine is installed."""


@dataclass(frozen=True)
class Voice:
    """A voice a listener can choose.

    `id` is prefixed with the engine that owns it ("say:Samantha"), so a voice
    id alone is enough to route a request to the right engine.
    """

    id: str
    label: str
    engine: str
    detail: str = ""

    def as_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "engine": self.engine, "detail": self.detail}


class TTSEngine(ABC):
    name = "base"
    #: Rate the engine speaks at with default settings, used to derive scales.
    nominal_wpm = 165.0

    @abstractmethod
    async def synth(self, text: str, wpm: float, voice: str | None = None) -> bytes:
        """Return raw PCM for `text` spoken at roughly `wpm`.

        `voice` is an id from `voices()`, or None for the engine's default.
        """

    @classmethod
    def voices(cls) -> list[Voice]:
        """Voices this engine can offer right now. Empty if unavailable."""
        return []

    @staticmethod
    def _voice_arg(voice: str | None, engine_name: str) -> str | None:
        """Strip the "engine:" prefix from a voice id, if present."""
        if not voice:
            return None
        prefix = engine_name + ":"
        return voice[len(prefix):] if voice.startswith(prefix) else voice

    @property
    def sample_rate(self) -> int:
        """The engine's real output rate. Callers must trust this, not config."""
        return settings.sample_rate

    @staticmethod
    async def _run(cmd: list[str], stdin_text: Optional[str] = None) -> bytes:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE if stdin_text is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        payload = stdin_text.encode("utf-8") if stdin_text is not None else None
        out, err = await proc.communicate(payload)
        if proc.returncode != 0:
            raise RuntimeError(
                f"{cmd[0]} exited {proc.returncode}: {err.decode('utf-8', 'replace')[:400]}"
            )
        return out


def _wav_sample_rate(buf: bytes) -> int | None:
    """Read the sample rate out of a WAV header, if this looks like one."""
    if len(buf) >= 28 and buf[0:4] == b"RIFF" and buf[8:12] == b"WAVE":
        import struct

        return struct.unpack("<I", buf[24:28])[0]
    return None


class EspeakEngine(TTSEngine):
    """espeak-ng: tiny, instant, robotic. Great default for latency.

    Speaking rate is a first-class flag (-s WPM), so the pacing controller maps
    onto it directly with no resampling.
    """

    name = "espeak"

    def __init__(self) -> None:
        self._rate: int | None = None

    #: A short curated list. espeak exposes dozens of accents and hundreds of
    #: variants; offering all of them is worse for a listener than offering a
    #: few that sound genuinely different from each other.
    CURATED = [
        ("en-us", "American"),
        ("en-us+f2", "American, higher"),
        ("en-gb", "British"),
        ("en-gb-x-rp", "British, received pronunciation"),
        ("en-gb-scotland", "Scottish"),
        ("en-au", "Australian"),
    ]

    @classmethod
    def voices(cls) -> list[Voice]:
        if not cls.available():
            return []
        return [
            Voice(id=f"espeak:{vid}", label=label, engine="espeak", detail="robotic, instant")
            for vid, label in cls.CURATED
        ]

    async def synth(self, text: str, wpm: float, voice: str | None = None) -> bytes:
        chosen = self._voice_arg(voice, "espeak") or settings.espeak_voice
        cmd = [
            settings.espeak_binary,
            "-v", chosen,
            "-s", str(int(round(wpm))),
            "--stdout",
        ]
        wav = await self._run(cmd, stdin_text=text)
        detected = _wav_sample_rate(wav)
        if detected:
            self._rate = detected
        return strip_wav_header(wav)

    @property
    def sample_rate(self) -> int:
        # espeak-ng emits 22050 Hz for every stock voice; the value observed on
        # the first synthesis wins if a build differs.
        return self._rate or 22050

    @staticmethod
    def available() -> bool:
        return shutil.which(settings.espeak_binary) is not None


class SayEngine(TTSEngine):
    """macOS built-in speech. Pre-installed on every Mac - nothing to download.

    `say` writes its output through CoreAudio, which wants a seekable
    destination for a WAV container, so each sentence goes to a scratch file
    that is read and deleted immediately. That is a per-sentence temporary of a
    second or two, not an episode file: no encoding happens, the episode is
    never assembled on disk, and streaming latency is unchanged.

    Rate is a direct flag (-r words per minute), so the pacing controller maps
    onto it exactly as it does for espeak.
    """

    name = "say"

    def __init__(self) -> None:
        self._rate: int | None = None

    #: Preferred first, when present. macOS ships dozens; these are the ones
    #: that read long-form prose well.
    PREFERRED = ["Samantha", "Alex", "Ava", "Tom", "Serena", "Daniel", "Karen", "Moira", "Fiona"]

    @classmethod
    @lru_cache(maxsize=1)
    def voices(cls) -> tuple:  # tuple so the cache can hold it
        if not cls.available():
            return ()
        try:
            out = subprocess.run(
                [settings.say_binary, "-v", "?"], capture_output=True, text=True, timeout=10
            ).stdout
        except Exception:  # pragma: no cover - depends on the host
            return ()
        found = {}
        for line in out.splitlines():
            # "Samantha            en_US    # Hello, my name is Samantha."
            parts = line.split()
            if len(parts) < 2 or not parts[1].startswith("en"):
                continue
            found[parts[0]] = parts[1].replace("_", "-")
        ordered = [v for v in cls.PREFERRED if v in found] + [
            v for v in sorted(found) if v not in cls.PREFERRED
        ]
        return tuple(
            Voice(id=f"say:{v}", label=v, engine="say", detail=found[v]) for v in ordered
        )

    async def synth(self, text: str, wpm: float, voice: str | None = None) -> bytes:
        import os
        import tempfile

        chosen = self._voice_arg(voice, "say") or settings.say_voice
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            cmd = [
                settings.say_binary,
                "-r", str(int(round(wpm))),
                "--data-format=LEI16@22050",
                "--file-format=WAVE",
                "-o", path,
            ]
            if chosen:
                cmd[1:1] = ["-v", chosen]
            await self._run(cmd, stdin_text=text)
            with open(path, "rb") as handle:
                wav = handle.read()
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        detected = _wav_sample_rate(wav)
        if detected:
            self._rate = detected
        return strip_wav_header(wav)

    @property
    def sample_rate(self) -> int:
        return self._rate or 22050

    @staticmethod
    def available() -> bool:
        import sys as _sys

        return _sys.platform == "darwin" and shutil.which(settings.say_binary) is not None


class DebugEngine(TTSEngine):
    """No speech: a soft tone whose length matches what the text would take.

    This exists so the timing logic, the streaming transport and the browser
    player can all be exercised on a machine with no TTS installed. It is never
    selected unless explicitly requested or nothing else is present.
    """

    name = "debug"

    #: 220 Hz carrier and a 2 Hz tremolo both complete a whole number of cycles
    #: in exactly one second, so one second of samples tiles seamlessly. Building
    #: it once and repeating it is ~50x faster than a per-sample Python loop,
    #: which otherwise dominates the test suite.
    _CYCLE_SECONDS = 1

    @staticmethod
    @lru_cache(maxsize=4)
    def _one_second(sample_rate: int) -> bytes:
        import array
        import math

        samples = array.array("h")
        for i in range(sample_rate):
            env = 0.15 * (0.6 + 0.4 * math.sin(2 * math.pi * 2.0 * i / sample_rate))
            samples.append(int(32767 * env * math.sin(2 * math.pi * 220.0 * i / sample_rate)))
        if sys.byteorder == "big":
            samples.byteswap()  # the stream is little-endian everywhere
        return samples.tobytes()

    @classmethod
    def voices(cls) -> list[Voice]:
        return [Voice(id="debug:tone", label="Placeholder tone", engine="debug",
                      detail="no speech engine installed")]

    async def synth(self, text: str, wpm: float, voice: str | None = None) -> bytes:
        words = max(1, len(text.split()))
        seconds = words / (max(wpm, 1.0) / 60.0)
        rate = settings.sample_rate
        cycle = self._one_second(rate)
        frames = int(seconds * rate)
        whole, remainder = divmod(frames, rate)
        return cycle * whole + cycle[: remainder * settings.sample_width]

    @staticmethod
    def available() -> bool:
        return True


#: The engines production is allowed to serve.
#:
#: The generation settings Phase 2 chose and Phase 6 measured. Copied from
#: `experiments/voice_identity.GENERATION` deliberately rather than imported:
#: the experiment layer is not a production dependency. Changing a number here
#: changes the voice, and invalidates every listening judgement made on it.
CHATTERBOX_GENERATION = {
    "exaggeration": 0.5,
    "cfg_weight": 0.5,
    "temperature": 0.8,
    "repetition_penalty": 1.2,
    "min_p": 0.05,
    "top_p": 1.0,
}


def pcm_from_float(samples) -> bytes:
    """Float waveform in [-1, 1] to 16-bit little-endian PCM."""
    import numpy as np

    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


class ChatterboxEngine(TTSEngine):
    """Chatterbox Base: FAM's production voice.

    A thin adapter around the call Phase 6 validated on an RTX 4090
    (`phase6_4090_20260908T064113Z`: 2.992s search-to-first-listen warm, zero
    playback stalls). Nothing about the synthesis is reinvented here - the
    settings, the `inference_mode` wrapper, the tensor-to-PCM conversion and
    the 24 kHz rate are the measured ones.

    Three things this must get right, and one it deliberately cannot:

    * **Load the model once.** The cold load is ~10s on a 4090. Loaded models
      are cached on the class for the life of the process and `warm_up()` pays
      it at startup, so no listener ever does.
    * **Do not block the event loop.** Generation is blocking GPU work and runs
      in a worker thread, which is what lets Claude keep streaming underneath -
      the Phase 6 decoupling depends on it.
    * **One generation at a time.** A single card cannot run concurrent
      generations safely, so they serialise on a semaphore. A second listener
      queues behind the first; at ~4.6x realtime that is fine for a few and is
      a real capacity limit worth knowing.
    * **`wpm` is ignored.** Chatterbox exposes no speaking-rate control, so the
      pacing half of the duration contract does not apply: length is held by
      the budget and by trimming at a sentence boundary, not by speeding the
      voice up. This is a decided trade, not an oversight - see PROBLEMS.md.
      The parameter stays in the signature because `TTSEngine` defines it.
    """

    name = "chatterbox"
    #: The rate the model emits at. Read back from the model once loaded, but
    #: needed before that for the stream header on the very first chunk.
    SAMPLE_RATE = 24000

    #: device -> loaded model. Shared by every request in the process.
    _loaded: dict = {}
    #: One card, one generation. Class-level so it is shared, and built lazily
    #: because a semaphore binds to the loop that first awaits it.
    _gate: "asyncio.Semaphore | None" = None
    #: Memoised availability, so /api/health does not re-import torch.
    _available: bool | None = None

    # -- configuration -----------------------------------------------------

    @staticmethod
    def reference_path() -> pathlib.Path:
        """The voice Chatterbox clones. Per-machine state, never in the repo."""
        configured = (settings.chatterbox_reference or "").strip()
        if configured:
            return pathlib.Path(configured).expanduser()
        return voice_store.voices_dir() / "reference_3.wav"

    @staticmethod
    def rights_path(reference: pathlib.Path) -> pathlib.Path:
        return reference.with_suffix(".rights.json")

    @classmethod
    def rights_cleared(cls, reference: pathlib.Path) -> tuple[bool, str]:
        """A cloned voice is somebody's voice. No record, no synthesis.

        The same three fields `tools/check_reference_audio.py` requires, asked
        again here because a gate that only runs in a tool is not a gate.
        """
        path = cls.rights_path(reference)
        if not path.exists():
            return False, f"no rights record at {path.name}"
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return False, f"{path.name} is not valid JSON: {exc}"
        for field_name in ("consent", "commercial_use", "synthetic_voice_cleared"):
            value = record.get(field_name)
            if str(value).strip().lower() not in ("yes", "true"):
                return False, f"{path.name} does not clear {field_name!r}"
        return True, "consent, commercial use and synthetic voice cleared"

    @classmethod
    def device(cls) -> str:
        """Where the model runs. CPU is refused, not chosen."""
        configured = (settings.chatterbox_device or "auto").strip().lower()
        if configured != "auto":
            return configured
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    # -- discovery ---------------------------------------------------------

    @classmethod
    def diagnose(cls) -> tuple[bool, str]:
        """Why this engine can or cannot serve, in one sentence."""
        try:
            import chatterbox.tts  # noqa: F401
        except Exception as exc:
            return False, f"chatterbox is not installed ({type(exc).__name__})"
        device = cls.device()
        if device == "cpu":
            # ~1x realtime or worse: the listener would hear silence. Refusing
            # is better than serving an episode that starves.
            return False, "no GPU: Chatterbox on CPU is slower than speech"
        reference = cls.reference_path()
        if not reference.exists():
            return False, f"no reference voice at {reference}"
        cleared, detail = cls.rights_cleared(reference)
        if not cleared:
            return False, detail
        return True, f"{device}, cloning {reference.name}"

    @classmethod
    def available(cls) -> bool:
        if cls._available is None:
            cls._available = cls.diagnose()[0]
        return cls._available

    @classmethod
    def voices(cls) -> list:
        if not cls.available():
            return []
        reference = cls.reference_path()
        return [Voice(id=f"chatterbox:{reference.stem}", label="FAM",
                      engine=cls.name, detail="Chatterbox")]

    # -- synthesis ---------------------------------------------------------

    @classmethod
    def _model(cls):
        device = cls.device()
        if device not in cls._loaded:
            from chatterbox.tts import ChatterboxTTS

            log.info("loading chatterbox on %s (one-time, ~10s)", device)
            model = ChatterboxTTS.from_pretrained(device=device)
            actual = str(getattr(model, "device", device))
            if actual.split(":")[0] != device.split(":")[0]:
                raise TTSUnavailable(
                    f"asked chatterbox for {device!r}, it loaded on {actual!r}")
            cls._loaded[device] = model
        return cls._loaded[device]

    def _synth_blocking(self, text: str) -> tuple[bytes, int]:
        """The validated call, unchanged, plus the tensor-to-PCM conversion."""
        import numpy as np
        import torch

        model = self._model()
        with torch.inference_mode():
            wav = model.generate(text,
                                 audio_prompt_path=str(self.reference_path()),
                                 **CHATTERBOX_GENERATION)
        samples = wav.squeeze(0).detach().cpu().numpy()
        del wav
        return pcm_from_float(samples), int(getattr(model, "sr", self.SAMPLE_RATE))

    async def synth(self, text: str, wpm: float, voice: str | None = None) -> bytes:
        """`wpm` is accepted and ignored - Chatterbox has no rate control."""
        if type(self)._gate is None:
            type(self)._gate = asyncio.Semaphore(1)
        async with type(self)._gate:
            # Off the event loop: generation is blocking GPU work, and Claude
            # has to keep streaming while it runs.
            pcm, rate = await asyncio.to_thread(self._synth_blocking, text)
        self._rate = rate
        return pcm

    @property
    def sample_rate(self) -> int:
        rate = getattr(self, "_rate", None)
        if rate:
            return int(rate)
        model = self._loaded.get(self.device())
        return int(getattr(model, "sr", self.SAMPLE_RATE)) if model else self.SAMPLE_RATE


#: Chatterbox is FAM's production voice. It gates itself on being able to run:
#: no package, no GPU, no reference voice or uncleared rights and it reports
#: unavailable, `build_engine` falls through to the interim engine, and the
#: health report says `interim: true`. So this is live everywhere it can be and
#: nowhere it cannot, which is the switch - no separate flag to forget.
#:
#: espeak and macOS `say` are deliberately absent and are no longer production
#: options at all. They exist only if the host OS happens to provide them, so
#: relying on either means the deployed app sounds different, and worse, than
#: the laptop it was built on.
#:
#: **It is now chosen, not detected.** `settings.voice_backend` names which
#: machine fills the slot - `chatterbox` for the card in this process, `remote`
#: for the same model on a GPU somewhere else - and it defaults to
#: `chatterbox`, so a deployment that says nothing behaves exactly as it did
#: before the remote backend existed. That default is the guard PROBLEMS.md §61
#: asked for by name: a rented or hosted voice must never become what every
#: listener gets merely because nothing local was installed.
def _remote_engine():
    """Imported late: remote_voice.py imports this module for its base class."""
    from remote_voice import RemoteChatterboxEngine

    return RemoteChatterboxEngine


def production_engines() -> tuple:
    """The engine classes this deployment may serve, in preference order.

    Exactly one, always. There is no falling back from the remote voice to the
    local one or the other way round: substituting a different engine means a
    listener judging one backend by another's behaviour, and an operator
    debugging a GPU that was never being asked to speak.
    """
    if settings.voice_backend == "remote":
        return (_remote_engine(),)
    return PRODUCTION_ENGINES


#: The in-process slot, and what `production_engines()` returns for every
#: backend but `remote`. Left as a module constant deliberately: it is what
#: `verify_voice.py` and the engine-contract tests name, and what they
#: substitute in order to exercise the fallback without a GPU.
PRODUCTION_ENGINES: tuple = (ChatterboxEngine,)

#: What a machine that cannot run the production engine gets instead: a tone,
#: not a voice.
#:
#: **There is no interim voice any more.** Piper held this slot and is gone -
#: package, engine class, config and dependency (PROBLEMS.md, and the commit
#: that removed it). It was removed rather than switched off because a
#: second engine that can speak is a second engine that can be *selected*, and
#: an app that quietly sounds worse than intended is the failure this project
#: has lost the most time to. A tone cannot be mistaken for FAM; a flat neural
#: voice can.
#:
#: So the honest states are exactly two: Chatterbox speaks, or nothing does and
#: everything says so - `engine_report()` reports `interim: true`, `demo.sh`
#: refuses to start quietly broken, and `build_engine` logs a warning naming
#: the reason Chatterbox was unavailable.
PLACEHOLDER_ENGINE = DebugEngine

#: Reachable only through `TTS_ENGINE`, which is a **development** override -
#: for deterministic tests and local work, never for a deployment. Production
#: ignores it entirely unless it names a production engine.
#:
#: `debug` is why this exists: the suite must run with no GPU, no model and no
#: credentials. espeak and macOS `say` remain reachable here for local work and
#: are not production voices - they exist only if the host OS happens to
#: provide them.
DEV_ENGINES = {
    "espeak": EspeakEngine,
    "say": SayEngine,
    "debug": DebugEngine,
}


def production_engine() -> TTSEngine | None:
    """The first production engine this machine can actually run, or None."""
    for cls in production_engines():
        if cls.available():
            return cls()
    return None


def build_engine(preference: str | None = None) -> TTSEngine:
    """The engine this process speaks with.

    Production does not choose. There is one production slot, filled by
    `PRODUCTION_ENGINES`; when it is empty there is no voice at all, only the
    placeholder tone, and the health report says so. `TTS_ENGINE` names a
    development engine and is honoured only because deterministic local tests
    need it - it cannot name a production engine into existence.
    """
    choice = (preference or settings.tts_engine or "auto").lower()
    if choice != "auto":
        cls = DEV_ENGINES.get(choice)
        if cls is None:
            raise TTSUnavailable(f"TTS engine '{choice}' is not a known engine")
        if not cls.available():
            raise TTSUnavailable(f"TTS engine '{choice}' is not installed or not configured")
        return cls()

    engine = production_engine()
    if engine is not None:
        return engine
    # Nothing can speak. Say why, at WARNING, every time an engine is built:
    # this used to be a flat-sounding voice nobody had chosen, which is a
    # failure that plays. A tone is a failure that is heard as one.
    for cls in production_engines():
        log.warning("%s is unavailable (%s); serving a placeholder tone, not a "
                    "voice", cls.name, cls.diagnose()[1])
    return PLACEHOLDER_ENGINE()


ENGINES = dict(DEV_ENGINES)


def list_voices() -> list[Voice]:
    """Every voice production may serve on this machine.

    espeak and `say` are gone from this list: they were never a production
    voice, and offering them in the picker made them one in practice.
    """
    voices: list[Voice] = []
    for cls in production_engines():
        voices.extend(cls.voices())
    if not voices:
        # Never empty: the picker must always have something in it, and a
        # placeholder tone that says what it is beats an empty control.
        voices.extend(PLACEHOLDER_ENGINE.voices())
    return voices


def default_voice() -> str | None:
    voices = list_voices()
    return voices[0].id if voices else None


def engine_for_voice(voice: str | None) -> TTSEngine:
    """Route a voice id to the engine that owns it.

    **A development engine is never selectable by a request.** This used to
    route on `ENGINES`, which is `DEV_ENGINES` - so `voice=debug:tone` from a
    browser returned the placeholder tone even on a machine where Chatterbox
    was loaded and working, because `DebugEngine.available()` is
    unconditionally True. `/api/health` went on reporting
    `selected: chatterbox`, since that asks `build_engine()`, which never sees
    the parameter. The interface said "voice: chatterbox" while a 220 Hz sine
    with a 2 Hz envelope played - which is exactly what a listener would
    describe as humming or buzzing.

    A page acquires that id honestly: `list_voices` offers the placeholder
    whenever the production slot is empty, so a tab opened while the voice
    file was still being placed, or during warm-up, picks it up and keeps
    sending it for the life of the session. Nothing later takes it back.

    This is the Piper failure in a new place, and CLAUDE.md already names it:
    an engine reached listeners three ways nobody chose, one of which was
    `engine_for_voice` falling back to it. Announcing was not enough then
    either. So production engines are matched first and a development id is
    refused outright while a real voice exists.

    What still works: `TTS_ENGINE=debug` selects a development engine, because
    that is a decision made on the server by whoever started it rather than by
    a query string, and `build_engine` honours it. And with no production
    engine at all, `build_engine` returns the placeholder anyway, so a machine
    that genuinely cannot speak still serves the tone rather than failing.
    """
    if voice and ":" in voice:
        name = voice.split(":", 1)[0]
        for cls in PRODUCTION_ENGINES:
            if cls.name == name:
                if cls.available():
                    return cls()
                log.warning("voice %r is unavailable; falling back", voice)
                break
        else:
            # Only worth saying when there is a real voice being passed over.
            # With none, the placeholder is the honest answer, not an override.
            if name in DEV_ENGINES and production_engine() is not None:
                log.warning(
                    "ignoring development voice %r from a request: production "
                    "speaks with %s, and an engine nobody chose is how this "
                    "app has lost the most time", voice,
                    production_engine().name)
    return build_engine()


async def warm_up() -> None:
    """Load the speech model before the first listener needs it.

    A neural voice takes a second or more to load, and that cost would otherwise
    land on the first episode of the day - exactly where it is most visible.
    Doing it at startup makes every request behave like the second one.
    """
    engine = build_engine()
    try:
        await engine.synth("Ready.", 150)
        log.info("speech engine %s warmed up", engine.name)
    except Exception as exc:  # pragma: no cover - never block startup
        log.warning("could not warm up %s: %s", engine.name, exc)
        # For a remote voice this is the *only* moment anything performs the
        # real action before a listener does, so the answer is kept rather than
        # only logged: `/api/health` reports it, and "configured but never
        # reachable" stops looking like "configured".
        if engine.name == "remote":
            import remote_voice

            remote_voice.RemoteChatterboxEngine.record_failure(str(exc))


def engine_report() -> dict:
    """What the server can actually do right now — surfaced in /api/health.

    `interim` is the honest bit, and it now means something stronger than it
    used to: not "a stand-in voice is speaking" but "no voice is speaking at
    all". With Piper gone there is nothing between Chatterbox and the tone.

    The keys are unchanged so that `/api/health` keeps its shape; only what
    fills `interim_engine` changed, from a voice to the placeholder.
    """
    selected = build_engine()
    engines = production_engines()
    report = {
        "selected": selected.name,
        "backend": settings.voice_backend,
        "production_engines": [cls.name for cls in engines],
        "interim": not engines or production_engine() is None,
        "interim_engine": PLACEHOLDER_ENGINE.name,
        "debug": True,
        "voices": [v.as_dict() for v in list_voices()],
        "default_voice": default_voice(),
    }
    if settings.voice_backend == "remote":
        # Where the card is, and whether a real call has ever succeeded against
        # it. "Configured" and "reachable" are different questions and this
        # answers both separately - reporting only the first is the cheaper
        # question PROBLEMS.md §52 is about.
        import remote_voice

        report["remote"] = remote_voice.report()
    return report
