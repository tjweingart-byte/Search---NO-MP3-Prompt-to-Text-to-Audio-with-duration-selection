"""Text submitted to Chatterbox -> the first moment a listener could hear sound.

This measures one thing, and it is not "how long to synthesise the episode".
It is the gap the product actually feels: FAM has a speakable sentence, and
nothing is coming out of the speaker yet.

Five marks are requested. Only some of them are separable, and *which* ones is
itself a finding rather than an inconvenience:

    1 dispatch            text handed over / `generate` entered
    2 stream begins       response headers back (remote only)
    3 first audio bytes   the first byte that decodes to a sample
    4 playable            enough samples buffered to start playing
    5 complete            all audio for this chunk in hand

**Chatterbox Turbo is one-shot.** `model.generate(text)` returns the whole
waveform; it emits nothing partway. That is established by the recovered Runpod
code, not assumed - see CHATTERBOX_UNKNOWNS.md. So in-process, marks 2-5 are
the same instant, and the probe records them as *collapsed* rather than
inventing four numbers from one event.

Over HTTP a second collapse is imposed by the current endpoint contract, and
this one is fixable: the reference endpoint answers with JSON carrying
base64-encoded PCM. No byte of that is decodable until the closing brace
arrives, so first-audio-bytes and playable both equal complete. The probe
measures the marks separately anyway - so the collapse shows up as equal
timestamps in the data, which is the evidence for changing the contract rather
than an opinion about it.

`PLAYABLE_SECONDS` is the buffer a player wants before it starts. Anything
below it is a stutter, so "playable" is not "first byte".
"""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from typing import Optional

#: Audio in hand before playback can sensibly start. FAM streams raw PCM to the
#: browser, which will play the moment it has something; this is the small
#: cushion that stops the first word arriving as a click.
PLAYABLE_SECONDS = 0.2

#: How much of the HTTP body to take per read, so the first-byte mark is a
#: real observation and not an artefact of buffering the whole response.
READ_CHUNK_BYTES = 8192

#: 16-bit mono.
BYTES_PER_SAMPLE = 2


@dataclass
class FirstAudio:
    """One measurement, with every mark on one clock and nothing inferred."""

    text: str
    words: int
    chars: int
    transport: str                      # "in_process" | "http"
    dispatch: float = 0.0
    stream_begin: Optional[float] = None
    first_audio_bytes: Optional[float] = None
    playable: Optional[float] = None
    complete: Optional[float] = None
    audio_seconds: Optional[float] = None
    sample_rate: Optional[int] = None
    device: Optional[str] = None
    collapsed: list = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    @property
    def to_first_playable(self) -> Optional[float]:
        """The headline: submission -> the listener could start hearing it."""
        return self.playable

    def as_dict(self) -> dict:
        return {
            "words": self.words,
            "chars": self.chars,
            "transport": self.transport,
            "seg_dispatch": self.dispatch,
            "seg_stream_begin": self.stream_begin,
            "seg_first_audio_bytes": self.first_audio_bytes,
            "seg_playable": self.playable,
            "seg_complete": self.complete,
            "first_playable_seconds": self.to_first_playable,
            "audio_seconds": self.audio_seconds,
            "sample_rate": self.sample_rate,
            "device": self.device,
            "collapsed_marks": list(self.collapsed),
            **self.detail,
        }


def playable_bytes(sample_rate: int, seconds: float = PLAYABLE_SECONDS) -> int:
    """Bytes of 16-bit mono PCM that constitute `seconds` of audio."""
    return int(sample_rate * seconds) * BYTES_PER_SAMPLE


def measure_in_process(text: str, *, device: Optional[str] = None,
                       warmup: bool = True) -> FirstAudio:
    """Chatterbox Turbo in this process. No GPU rental, no endpoint, no cost.

    Timing goes through `chatterbox_impl.synthesise`, which is the audited
    port of the recovered benchmarks - CUDA fenced on both sides, `wav.cpu()`
    outside the clock. Re-implementing that fence here is exactly how this
    project has produced meaningless GPU numbers before, so it is not
    re-implemented.
    """
    from experiments.adapters import chatterbox_impl as impl

    words = len(text.split())
    start = time.perf_counter()
    out = impl.synthesise(text, device=device, warmup=warmup, inference_mode=True)
    complete = time.perf_counter() - start

    generated = out["generate_seconds"]
    rate = int(out.get("sample_rate") or 0)
    result = FirstAudio(text=text, words=words, chars=len(text),
                        transport="in_process", device=out.get("device"))
    result.dispatch = 0.0
    # One-shot: nothing exists until generate returns, and then all of it does.
    result.stream_begin = generated
    result.first_audio_bytes = generated
    result.playable = complete
    result.complete = complete
    result.sample_rate = rate
    result.audio_seconds = out.get("audio_seconds")
    result.collapsed = ["stream_begin", "first_audio_bytes"]
    result.detail = {
        "generate_seconds": generated,
        "post_generate_seconds": complete - generated,
        "pcm_bytes": len(out.get("pcm") or b""),
        "realtime_factor": out.get("realtime_factor"),
        "cold": out.get("cold"),
        "model_load_seconds": out.get("model_load_seconds"),
        "channels": out.get("channels"),
        "gpu_cost": out.get("gpu_cost"),
        "one_shot": True,
        "why_collapsed": (
            "ChatterboxTurboTTS.generate returns the complete waveform; it "
            "emits no partial audio, so there is no earlier moment to mark."
        ),
    }
    return result


def measure_http(endpoint: str, text: str, *, sample_rate: int = 24000,
                 timeout: float = 120.0) -> FirstAudio:
    """A Chatterbox endpoint that is already running, somewhere else.

    Reads the body incrementally so "first bytes back" is observed rather than
    assumed. Handles both the JSON+base64 contract the reference endpoint uses
    today and a raw-PCM streaming response, and records which one answered -
    because the difference between them is the whole question.
    """
    import urllib.request

    words = len(text.split())
    result = FirstAudio(text=text, words=words, chars=len(text), transport="http")

    payload = json.dumps({"text": text, "sample_rate": sample_rate}).encode()
    request = urllib.request.Request(
        endpoint, data=payload, headers={"Content-Type": "application/json"})

    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        result.stream_begin = time.perf_counter() - start
        content_type = (reply.headers.get("Content-Type") or "").lower()
        streaming_pcm = "application/json" not in content_type
        header_rate = reply.headers.get("X-Sample-Rate")
        rate = int(header_rate) if header_rate else sample_rate
        want = playable_bytes(rate)

        body = bytearray()
        audio_bytes = 0
        while True:
            piece = reply.read(READ_CHUNK_BYTES)
            if not piece:
                break
            now = time.perf_counter() - start
            body += piece
            if streaming_pcm:
                # Every byte is audio, so both marks are real observations.
                if result.first_audio_bytes is None:
                    result.first_audio_bytes = now
                audio_bytes += len(piece)
                if result.playable is None and audio_bytes >= want:
                    result.playable = now
        result.complete = time.perf_counter() - start

    if streaming_pcm:
        pcm = bytes(body)
        result.sample_rate = rate
        if result.playable is None:
            # The whole response was shorter than the playback cushion.
            result.playable = result.complete
            result.collapsed.append("playable")
        result.detail = {"contract": "streaming_pcm", "pcm_bytes": len(pcm)}
    else:
        reply_body = json.loads(bytes(body).decode())
        pcm = base64.b64decode(reply_body.get("pcm_base64", ""))
        rate = int(reply_body.get("sample_rate", rate))
        result.sample_rate = rate
        # Base64 inside JSON: nothing decodes until the last byte lands.
        result.first_audio_bytes = result.complete
        result.playable = result.complete
        result.collapsed = ["first_audio_bytes", "playable"]
        result.device = reply_body.get("device")
        result.detail = {
            "contract": "json_base64",
            "pcm_bytes": len(pcm),
            "gpu_seconds": reply_body.get("gpu_seconds"),
            "cold": reply_body.get("cold"),
            "why_collapsed": (
                "The response is one JSON object with base64 PCM inside it. No "
                "audio can be decoded until the final byte arrives, so first "
                "audio and playable are both the completion time. A raw-PCM "
                "streaming response would separate them."
            ),
        }

    result.dispatch = 0.0
    result.audio_seconds = len(pcm) / BYTES_PER_SAMPLE / rate if rate else None
    return result
