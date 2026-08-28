"""Raw-PCM helpers.

Nothing here writes a file and nothing encodes MP3. The whole pipeline moves
16-bit little-endian PCM through memory and out over the socket.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

from config import settings


def streaming_wav_header(
    sample_rate: int = None,
    channels: int = None,
    sample_width: int = None,
) -> bytes:
    """A 44-byte RIFF header with 'unknown' sizes.

    A normal WAV header must state the total byte count, which we cannot know
    until synthesis finishes. Writing 0xFFFFFFFF is the conventional trick for
    live streams: players treat it as "read until the socket closes". This is
    what lets a plain <audio> tag start playing before the podcast exists.
    """
    sample_rate = sample_rate or settings.sample_rate
    channels = channels or settings.channels
    sample_width = sample_width or settings.sample_width

    byte_rate = sample_rate * channels * sample_width
    block_align = channels * sample_width
    unknown = 0xFFFFFFFF

    return b"".join(
        [
            b"RIFF",
            struct.pack("<I", unknown),
            b"WAVE",
            b"fmt ",
            struct.pack("<I", 16),
            struct.pack("<H", 1),  # PCM
            struct.pack("<H", channels),
            struct.pack("<I", sample_rate),
            struct.pack("<I", byte_rate),
            struct.pack("<H", block_align),
            struct.pack("<H", sample_width * 8),
            b"data",
            struct.pack("<I", unknown),
        ]
    )


def silence(seconds: float, sample_rate: int | None = None) -> bytes:
    """PCM silence, used for beats between segments and end-of-show padding."""
    rate = sample_rate or settings.sample_rate
    frames = int(max(0.0, seconds) * rate)
    return b"\x00\x00" * frames * settings.channels


def pcm_duration(num_bytes: int, sample_rate: int | None = None) -> float:
    """Seconds of audio represented by `num_bytes` of PCM."""
    rate = sample_rate or settings.sample_rate
    return num_bytes / float(rate * settings.channels * settings.sample_width)


def strip_wav_header(buf: bytes) -> bytes:
    """Return the payload of `data` chunk from a WAV blob.

    espeak-ng emits a full WAV on stdout; we only want the samples so they can
    be concatenated into one continuous stream.
    """
    if len(buf) < 12 or buf[0:4] != b"RIFF" or buf[8:12] != b"WAVE":
        return buf  # already raw
    pos = 12
    while pos + 8 <= len(buf):
        chunk_id = buf[pos : pos + 4]
        (chunk_size,) = struct.unpack("<I", buf[pos + 4 : pos + 8])
        body = pos + 8
        if chunk_id == b"data":
            if chunk_size in (0, 0xFFFFFFFF):
                return buf[body:]
            return buf[body : body + chunk_size]
        pos = body + chunk_size + (chunk_size & 1)
    return b""


@dataclass
class PaceController:
    """Keeps the finished audio on the clock while it is still being made.

    The naive approach ("count words, hope for the best") drifts because real
    TTS timing depends on punctuation, numbers and abbreviations. Instead we
    re-plan before every segment: given the audio already emitted and the words
    still unspoken, what speaking rate lands exactly on the target?

    Corrections are clamped to [min_wpm, max_wpm] so the voice never becomes a
    chipmunk or a sleepwalker; residual error is absorbed by trailing silence
    or by dropping the optional sign-off.
    """

    target_seconds: float
    total_words: int
    emitted_bytes: int = 0
    spoken_words: int = 0
    #: Sample rate of the engine feeding this controller. Timing is computed
    #: from the engine's real rate, never an assumed one - a mismatch here is
    #: the classic cause of audio that is pitch-shifted and the wrong length.
    sample_rate: int = settings.sample_rate

    @property
    def elapsed(self) -> float:
        return pcm_duration(self.emitted_bytes, self.sample_rate)

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.target_seconds - self.elapsed)

    @property
    def remaining_words(self) -> int:
        return max(0, self.total_words - self.spoken_words)

    def next_wpm(self) -> float:
        """Words-per-minute for the upcoming segment."""
        if self.remaining_words == 0:
            return settings.target_wpm
        if self.remaining_seconds <= 0.5:
            return settings.max_wpm
        ideal = self.remaining_words / (self.remaining_seconds / 60.0)
        return max(settings.min_wpm, min(settings.max_wpm, ideal))

    def observe(self, pcm_bytes: int, words: int) -> None:
        self.emitted_bytes += pcm_bytes
        self.spoken_words += words

    def drift(self) -> float:
        """Signed error in seconds: positive means we ran long."""
        return self.elapsed - self.target_seconds
