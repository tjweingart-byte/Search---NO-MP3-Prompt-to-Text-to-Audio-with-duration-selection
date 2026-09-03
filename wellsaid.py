"""WellSaid Labs as an alternate voice, for listening tests.

Claude still writes every word. This is a speech engine and nothing else: it
receives the sentences `pipeline.py` has already produced and returns PCM, so
the script, the search behaviour, the cache and the player are untouched.

**Verified against WellSaid's published reference** rather than assumed:

    POST https://api.wellsaidlabs.com/v1/tts/stream
    X-Api-Key: <key>
    Content-Type: application/json
    Accept: audio/mpeg
    {"text": "...", "speaker_id": 35}

with roughly a 1000-character ceiling per request. Their documented output
format is MP3; whether they also serve WAV is **not** documented anywhere I
could reach, so this asks for WAV first and accepts MP3 if that is refused,
logging which one arrived. If they add or remove a format the code adapts
rather than breaking.

Three things this has to get right, none of them obvious:

* **The browser is still given raw PCM.** "No MP3" is a rule about what
  reaches the listener and what is written to disk, not about what two servers
  say to each other over a socket. MP3 is decoded here, in memory, and never
  becomes a file. If WAV is served, no decoder is needed at all.

* **The sample rate must be known before the first call.** `app.py` writes the
  stream header from `engine.sample_rate` before any audio has been requested,
  so the engine cannot wait to be told. Everything is therefore resampled to
  the app's own rate, which is fixed and knowable.

* **It must never quietly become Piper.** A fallback here would mean listening
  to a Piper episode and drawing conclusions about WellSaid. Every failure
  raises with the real reason attached.
"""
from __future__ import annotations

import array
import asyncio
import logging
import re
import shutil
import struct
import time

from audio_utils import strip_wav_header
from config import settings
from tts import TTSEngine, TTSUnavailable, Voice

log = logging.getLogger("fam.wellsaid")

#: Retried once each, with a pause. Anything else is reported as it stands.
_RETRY_STATUS = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 3

#: Below this amplitude counts as silence when tightening a join. About -36 dB;
#: low enough that no speech is mistaken for it.
_SILENCE_FLOOR = 500
#: Left either side of a trimmed join, so a word is never clipped short.
_JOIN_MARGIN_SECONDS = 0.03


class WellSaidError(RuntimeError):
    """WellSaid refused or failed. Carries the real reason, never a fallback."""


class WellSaidLocalError(WellSaidError):
    """WellSaid answered; **this machine** could not use the answer.

    A separate type because the two have opposite consequences for a key.
    WellSaid returning HTTP 200 means the key is good, whatever happens next
    locally - so a missing or broken decoder must not be allowed to condemn
    it. Matching on the words in an error message got this wrong twice; the
    distinction is now something the code states rather than something a
    caller has to infer.
    """


class WellSaidUnreachable(WellSaidError):
    """The request never arrived. Says nothing at all about the key."""


# --- text ---------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
_CLAUSE_END = re.compile(r"(?<=[;:,—–])\s+")


def chunk_text(text: str, limit: int) -> list[str]:
    """Split `text` into pieces WellSaid will accept, at natural boundaries.

    Sentences first, then clauses, then - only if a single clause is somehow
    longer than the whole limit - at a space. Never inside a word, because a
    chunk boundary is a breath the listener hears.

    In practice this rarely does anything: the pipeline already calls the
    engine once per sentence, so almost every chunk arrives well under the
    limit. It exists for the sentence that does not.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    def pack(pieces: list[str]) -> list[str]:
        out: list[str] = []
        for piece in pieces:
            if out and len(out[-1]) + 1 + len(piece) <= limit:
                out[-1] = f"{out[-1]} {piece}"
            else:
                out.append(piece)
        return out

    chunks: list[str] = []
    for sentence in pack([s.strip() for s in _SENTENCE_END.split(text) if s.strip()]):
        if len(sentence) <= limit:
            chunks.append(sentence)
            continue
        for clause in pack([c.strip() for c in _CLAUSE_END.split(sentence) if c.strip()]):
            if len(clause) <= limit:
                chunks.append(clause)
                continue
            # Last resort, and still on a word boundary.
            words, line = clause.split(), ""
            for word in words:
                if line and len(line) + 1 + len(word) > limit:
                    chunks.append(line)
                    line = word
                else:
                    line = f"{line} {word}".strip()
            if line:
                chunks.append(line)
    return chunks


# --- audio --------------------------------------------------------------

def _looks_like_wav(buf: bytes) -> bool:
    return len(buf) >= 12 and buf[0:4] == b"RIFF" and buf[8:12] == b"WAVE"


def _wav_format(buf: bytes) -> tuple[int, int]:
    """(sample_rate, channels) from a WAV's fmt chunk."""
    pos = 12
    while pos + 8 <= len(buf):
        chunk_id = buf[pos:pos + 4]
        (size,) = struct.unpack("<I", buf[pos + 4:pos + 8])
        if chunk_id == b"fmt ":
            channels, rate = struct.unpack("<HI", buf[pos + 10:pos + 16])
            return rate, max(1, channels)
        pos = pos + 8 + size + (size & 1)
    return settings.sample_rate, 1


def _to_mono(samples: array.array, channels: int) -> array.array:
    if channels <= 1:
        return samples
    return array.array("h", [
        sum(samples[i:i + channels]) // channels
        for i in range(0, len(samples) - channels + 1, channels)
    ])


def _resample(samples: array.array, source_rate: int, target_rate: int) -> array.array:
    """Linear resampling. Speech at 24k -> 22.05k is inaudible; nothing here
    justifies a filter bank."""
    if source_rate == target_rate or not samples:
        return samples
    ratio = source_rate / float(target_rate)
    out = array.array("h", bytes(2 * int(len(samples) / ratio)))
    last = len(samples) - 1
    for i in range(len(out)):
        at = i * ratio
        low = int(at)
        if low >= last:
            out[i] = samples[last]
            continue
        frac = at - low
        out[i] = int(samples[low] + (samples[low + 1] - samples[low]) * frac)
    return out


def _decode_mp3(buf: bytes, target_rate: int) -> bytes:
    """MP3 -> PCM, in memory, via ffmpeg.

    Only reached if WellSaid refuses WAV. ffmpeg is the one extra thing this
    path needs, and if it is missing the error says exactly that rather than
    producing silence.
    """
    binary = shutil.which(settings.ffmpeg_binary)
    if not binary:
        raise WellSaidLocalError(
            f"WellSaid returned MP3 and {settings.ffmpeg_binary!r} is not installed, so it "
            "cannot be decoded. Install it (macOS: brew install ffmpeg) and try again. "
            "Nothing was played rather than falling back to another voice."
        )
    import subprocess

    try:
        done = subprocess.run(
            [binary, "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
             "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(target_rate), "pipe:1"],
            input=buf, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise WellSaidLocalError(f"Could not run {binary}: {exc}") from exc
    if done.returncode != 0:
        raise WellSaidLocalError(
            f"ffmpeg could not decode WellSaid's audio: "
            f"{done.stderr.decode('utf-8', 'replace')[:300]}"
        )
    return done.stdout


def _to_pcm(buf: bytes, content_type: str, target_rate: int) -> bytes:
    """Whatever WellSaid sent -> mono 16-bit PCM at the app's rate."""
    if _looks_like_wav(buf):
        rate, channels = _wav_format(buf)
        samples = array.array("h")
        samples.frombytes(strip_wav_header(buf))
        return _resample(_to_mono(samples, channels), rate, target_rate).tobytes()
    if buf[:3] == b"ID3" or (len(buf) > 1 and buf[0] == 0xFF and buf[1] & 0xE0 == 0xE0):
        return _decode_mp3(buf, target_rate)
    raise WellSaidLocalError(
        f"WellSaid sent {len(buf)} bytes of {content_type or 'an unlabelled format'}, "
        "which is neither WAV nor MP3. The audio was not played."
    )


def _trim_join(pcm: bytes, rate: int) -> bytes:
    """Tighten the silence at both ends of a chunk before it is joined.

    Each chunk is rendered independently, so each carries its own lead-in and
    tail. Concatenated untrimmed those stack into a pause in the middle of a
    sentence - the audible seam this is here to prevent. Only used when a
    sentence needed more than one chunk.
    """
    samples = array.array("h")
    samples.frombytes(pcm)
    if not samples:
        return pcm
    start, end = 0, len(samples) - 1
    while start < end and abs(samples[start]) < _SILENCE_FLOOR:
        start += 1
    while end > start and abs(samples[end]) < _SILENCE_FLOOR:
        end -= 1
    margin = int(_JOIN_MARGIN_SECONDS * rate)
    start = max(0, start - margin)
    end = min(len(samples) - 1, end + margin)
    return samples[start:end + 1].tobytes()


# --- the engine ---------------------------------------------------------

class WellSaidEngine(TTSEngine):
    """A hosted voice, selected explicitly and never automatically.

    Deliberately absent from `build_engine()`'s automatic preference order:
    it costs money per character and needs a key, so it speaks only when a
    listener picks one of its voices by name.
    """

    name = "wellsaid"

    #: The two voices under test. `speaker_id` comes from settings so it can be
    #: changed without editing code, per the WELLSAID_*_ID variables.
    @staticmethod
    def catalogue() -> list[tuple[str, str]]:
        return [
            (settings.wellsaid_chase_j_id, "Chase J"),
            (settings.wellsaid_kai_m_id, "Kai M"),
        ]

    @classmethod
    def voices(cls) -> list[Voice]:
        if not cls.available():
            return []
        return [
            Voice(id=f"wellsaid:{speaker}", label=f"{label} (WellSaid)",
                  engine="wellsaid", detail=f"hosted, speaker {speaker}")
            for speaker, label in cls.catalogue()
        ]

    @staticmethod
    def available() -> bool:
        """A key is the whole requirement. Without one the voices are simply
        not offered, which is better than offering them and failing on tap."""
        return bool(settings.wellsaid_api_key)

    @classmethod
    def label_for(cls, speaker: str) -> str:
        for candidate, label in cls.catalogue():
            if candidate == speaker:
                return label
        return f"speaker {speaker}"

    @property
    def sample_rate(self) -> int:
        # Fixed, not discovered: app.py needs this before the first request,
        # and everything is resampled to it on arrival.
        return settings.sample_rate

    async def synth(self, text: str, wpm: float, voice: str | None = None) -> bytes:
        speaker = self._voice_arg(voice, "wellsaid") or settings.wellsaid_chase_j_id
        # Nothing to say costs nothing to say, and must not need a key to
        # establish that: the pipeline hands over whatever the model wrote,
        # and a blank line should not become a failed episode.
        chunks = chunk_text(text, settings.wellsaid_max_chars)
        if not chunks:
            return b""
        if not settings.wellsaid_api_key:
            raise TTSUnavailable(
                "WELLSAID_API_KEY is not set, so the WellSaid voices cannot speak. "
                "Run: python setup_wellsaid.py"
            )

        label = self.label_for(speaker)
        started = time.perf_counter()
        # One line per sentence is readable; two is not, and the pipeline calls
        # this once per sentence. The detail is kept for the multi-chunk case,
        # which is the one where a join could go wrong.
        detailed = len(chunks) > 1
        if detailed:
            log.info(
                "TTS provider=wellsaid voice=%r speaker_id=%s chars=%d chunks=%d - requesting",
                label, speaker, len(text), len(chunks),
            )
        # WellSaid exposes no speaking-rate control, so the pacing controller's
        # wpm cannot be honoured. Said once rather than per sentence. Length is
        # still bounded, because duration is a ceiling and over-runs are trimmed.
        if not getattr(WellSaidEngine, "_warned_about_rate", False):
            WellSaidEngine._warned_about_rate = True
            log.info("wellsaid has no rate control; requested %.0f wpm is advisory", wpm)

        pieces: list[bytes] = []
        first_audio_at: float | None = None
        for index, chunk in enumerate(chunks, start=1):
            pcm = await self._speak_one(chunk, speaker, index, len(chunks), detailed)
            if first_audio_at is None:
                first_audio_at = time.perf_counter() - started
            pieces.append(_trim_join(pcm, self.sample_rate) if len(chunks) > 1 else pcm)

        audio = b"".join(pieces)
        total = time.perf_counter() - started
        seconds = len(audio) / float(self.sample_rate * settings.sample_width)
        log.info(
            "TTS provider=wellsaid voice=%r speaker_id=%s chars=%d chunks=%d "
            "first_audio=%.2fs total=%.2fs audio=%.2fs (%.0fx realtime)",
            label, speaker, len(text), len(chunks), first_audio_at or 0.0, total, seconds,
            (seconds / total) if total else 0.0,
        )
        return audio

    async def _speak_one(self, text: str, speaker: str, index: int, of: int,
                         detailed: bool = True) -> bytes:
        """One request, with retries on the statuses worth retrying.

        Every failure is raised. There is no path from here back to Piper: an
        episode that silently changed voice mid-test is worse than no episode,
        because the conclusion drawn from it would be about the wrong engine.
        """
        import httpx

        body: dict = {"text": text, "speaker_id": self._speaker_id(speaker)}
        if settings.wellsaid_model:
            body["model"] = settings.wellsaid_model

        last: str = ""
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            started = time.perf_counter()
            try:
                async with httpx.AsyncClient(timeout=settings.wellsaid_timeout) as client:
                    response = await client.post(
                        settings.wellsaid_api_url,
                        json=body,
                        headers={
                            "X-Api-Key": settings.wellsaid_api_key,
                            "Content-Type": "application/json",
                            # WAV needs no decoder. Their documented format is
                            # MP3, so this may well be ignored - _to_pcm reads
                            # what actually arrived rather than what was asked.
                            "Accept": "audio/wav, audio/mpeg;q=0.9",
                        },
                    )
            except Exception as exc:  # noqa: BLE001 - the reason is the point
                last = f"{type(exc).__name__}: {exc}"
                log.warning("wellsaid chunk %d/%d attempt %d failed: %s",
                            index, of, attempt, last)
                if attempt < _MAX_ATTEMPTS:
                    await asyncio.sleep(attempt)
                    continue
                raise WellSaidUnreachable(f"Could not reach WellSaid: {last}") from exc

            elapsed = time.perf_counter() - started
            if response.status_code == 200:
                kind = response.headers.get("Content-Type", "")
                (log.info if detailed else log.debug)(
                    "wellsaid chunk %d/%d ok in %.2fs (%d bytes, %s)",
                    index, of, elapsed, len(response.content), kind or "unlabelled")
                return _to_pcm(response.content, kind, self.sample_rate)

            detail = response.text[:300].strip()
            last = f"HTTP {response.status_code}: {detail or '(no body)'}"
            log.warning("wellsaid chunk %d/%d attempt %d rejected - %s",
                        index, of, attempt, last)
            if response.status_code in _RETRY_STATUS and attempt < _MAX_ATTEMPTS:
                await asyncio.sleep(attempt)
                continue
            raise WellSaidError(self._explain(response.status_code, detail))

        raise WellSaidUnreachable(
            f"WellSaid failed after {_MAX_ATTEMPTS} attempts: {last}")

    @staticmethod
    def _speaker_id(speaker: str):
        """Sent as a number where it is one. Their examples show both forms;
        a numeric id is the documented shape."""
        try:
            return int(speaker)
        except (TypeError, ValueError):
            return speaker

    @staticmethod
    def _explain(status: int, detail: str) -> str:
        """Say what to do about it, not just what happened."""
        hint = {
            401: "WellSaid rejected the API key. Check it with: python setup_wellsaid.py --show",
            403: "The key was accepted but is not allowed to use this voice or endpoint.",
            404: "That speaker_id does not exist on this account.",
            429: "WellSaid is rate limiting this key.",
        }.get(status, "")
        return f"WellSaid returned HTTP {status}. {hint} {detail}".strip()
