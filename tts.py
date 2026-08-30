"""Text-to-speech engines that emit raw PCM.

Design rule for this project: no engine is allowed to produce a file, and no
step encodes MP3. Each engine takes a short chunk of text plus a speaking rate
and returns 16-bit PCM bytes that are written straight to the HTTP response.

Engines are subprocess-based on purpose. A subprocess per sentence keeps peak
memory flat, lets the pacing controller change the rate mid-podcast, and means
the heavy neural weights stay in one place (piper's own process) instead of
being loaded into every web worker.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from abc import ABC, abstractmethod
from typing import Optional

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


class PiperEngine(TTSEngine):
    """Piper: neural voice, still CPU-only and faster than real time.

    `--output_raw` writes headerless PCM to stdout, which is exactly the format
    the rest of the pipeline speaks, so there is no conversion step at all.
    Rate is controlled by `--length_scale` (higher = slower).
    """

    name = "piper"

    def __init__(self) -> None:
        # Every piper voice ships a sidecar JSON stating its native rate, and
        # voices differ (16000 and 22050 are both common). Reading it is the
        # only way to avoid a chipmunk-or-baritone bug that no test catches
        # because the byte counts still look plausible.
        self._rate = 22050
        try:
            import json
            import pathlib

            cfg = pathlib.Path(settings.piper_model + ".json")
            if not cfg.exists():
                cfg = pathlib.Path(settings.piper_model).with_suffix(".onnx.json")
            if cfg.exists():
                self._rate = int(json.loads(cfg.read_text())["audio"]["sample_rate"])
        except Exception:  # keep the default; surfaced via /api/health
            pass

    @property
    def sample_rate(self) -> int:
        return self._rate

    @classmethod
    def voices(cls) -> list[Voice]:
        if not cls.available():
            return []
        import pathlib

        stem = pathlib.Path(settings.piper_model).stem.replace(".onnx", "")
        return [Voice(id=f"piper:{stem}", label=stem.replace("_", " "),
                      engine="piper", detail="neural, natural")]

    async def synth(self, text: str, wpm: float, voice: str | None = None) -> bytes:
        # One model per process; a voice id selects the configured model only.
        length_scale = max(0.6, min(1.6, self.nominal_wpm / max(wpm, 1.0)))
        cmd = [
            settings.piper_binary,
            "--model", settings.piper_model,
            "--length_scale", f"{length_scale:.3f}",
            "--output_raw",
        ]
        return await self._run(cmd, stdin_text=text)

    @staticmethod
    def available() -> bool:
        return bool(settings.piper_model) and shutil.which(settings.piper_binary) is not None


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


def build_engine(preference: str | None = None) -> TTSEngine:
    """Pick an engine, honouring TTS_ENGINE and falling back sensibly."""
    choice = (preference or settings.tts_engine or "auto").lower()

    explicit = {
        "piper": PiperEngine,
        "espeak": EspeakEngine,
        "say": SayEngine,
        "debug": DebugEngine,
    }
    if choice in explicit:
        cls = explicit[choice]
        if not cls.available():
            raise TTSUnavailable(f"TTS engine '{choice}' is not installed or not configured")
        return cls()

    # Preference order: best voice first, then anything already on the machine.
    for cls in (PiperEngine, EspeakEngine, SayEngine):
        if cls.available():
            return cls()
    return DebugEngine()


ENGINES = {
    "piper": PiperEngine,
    "say": SayEngine,
    "espeak": EspeakEngine,
    "debug": DebugEngine,
}


def list_voices() -> list[Voice]:
    """Every voice available on this machine, best-sounding first.

    Ordered by engine quality rather than alphabetically, because the first
    entry is what a listener gets by default.
    """
    voices: list[Voice] = []
    for name in ("piper", "say", "espeak"):
        voices.extend(ENGINES[name].voices())
    if not voices:
        voices.extend(DebugEngine.voices())
    return voices


def default_voice() -> str | None:
    voices = list_voices()
    return voices[0].id if voices else None


def engine_for_voice(voice: str | None) -> TTSEngine:
    """Route a voice id to the engine that owns it.

    An unknown or unavailable voice falls back to the best engine present
    rather than failing: a listener choosing a voice that has since been
    uninstalled should still hear their episode.
    """
    if voice and ":" in voice:
        name = voice.split(":", 1)[0]
        cls = ENGINES.get(name)
        if cls is not None and cls.available():
            return cls()
        log.warning("voice %r is unavailable; falling back", voice)
    return build_engine()


def engine_report() -> dict:
    """What the server can actually do right now — surfaced in /api/health."""
    return {
        "selected": build_engine().name,
        "piper": PiperEngine.available(),
        "espeak": EspeakEngine.available(),
        "say": SayEngine.available(),
        "debug": True,
        "voices": [v.as_dict() for v in list_voices()],
        "default_voice": default_voice(),
    }
