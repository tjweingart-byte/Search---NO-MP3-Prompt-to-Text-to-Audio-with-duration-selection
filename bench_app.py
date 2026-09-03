"""The voice bench: search, and audio, and nothing else.

A place to try a candidate voice on real FAM output without the rest of the
app in the way. One screen: ask a question or paste a paragraph, pick a voice,
listen. No myFAM, no DailyFAM, no Explore, no profile, no echoes, no mixes.

**Why this is a separate app rather than a stripped-down one.** The point of
this branch is to find a replacement for Piper and then *bring it home*, and a
branch that deletes half the repo makes that merge miserable - the voice work
arrives tangled up with hundreds of lines of removal that have nothing to do
with voices. So nothing is deleted. `app.py`, `static/index.html` and every
module they use are untouched, and this branch differs from the main one only
by files it adds.

What that buys: a new voice is a new `TTSEngine` subclass in `tts.py` plus a
line in `ENGINES`, exactly as it would be on the main branch, so the thing you
end up wanting to keep is a clean diff over shared files.

Two ways in, on purpose:

* **Ask** runs the real pipeline - Claude writes a FAM script, it is cached,
  and the audio streams sentence by sentence. This is the honest test, because
  a voice has to read *this product's* prose, not a demo sentence.
* **Speak** sends text straight to the engine. No model call, no key needed,
  and the same paragraph every time - which is what makes an A/B fair, and
  what lets someone with no API key still judge a voice.

Run it with ./bench.sh
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from audio_utils import pcm_duration, silence
from cache import build_cache
from config import settings
from demo_script import DemoGenerator
from pipeline import GenerationStats, NotCached, PodcastPipeline
from script_generator import plan_episode
from tts import (
    TTSUnavailable,
    build_engine,
    default_voice,
    engine_for_voice,
    engine_report,
    list_voices,
)

log = logging.getLogger("fam.bench")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="FAM voice bench")

SCRIPT_CACHE = build_cache()
DEMO_MODE = not settings.anthropic_api_key
#: Same as the main app: hold back the first moment of audio so a failure is
#: still an HTTP error the page can show, rather than a silent empty stream.
PREROLL_SECONDS = 1.5
WAV_HEADER_BYTES = 44
#: A paragraph, not an episode. Long enough to judge a voice, short enough
#: that a mistake costs nothing.
MAX_SPEAK_CHARS = 2000


def _wav_header(samples: int, rate: int) -> bytes:
    """A WAV header with the *real* length in it.

    `streaming_wav_header` deliberately lies about the length, because the
    main app is streaming and does not know it yet. /api/speak synthesises up
    front, so it does know - and the streaming header made a 9.6 second clip
    report itself as 27 hours, which breaks every tool that trusts the header
    (an <audio> tag, a scrub bar, anything the file is dragged into).
    """
    import struct

    data = samples * 2
    return (b"RIFF" + struct.pack("<I", 36 + data) + b"WAVE"
            + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
            + b"data" + struct.pack("<I", data))


def _strict_engine(voice: str):
    """Resolve a voice, or refuse. Never substitute another one.

    `engine_for_voice()` falls back to the best available engine, which is
    right in the product - a listener whose voice was uninstalled should still
    hear their episode - and exactly wrong here. This page exists to answer
    "how does *this* voice sound", so silently answering with a different one
    makes every judgement made on it worthless. Same lesson the hosted-voice
    experiment produced, applied to the bench itself.
    """
    if voice and voice not in {v.id for v in list_voices()}:
        raise HTTPException(
            status_code=400,
            detail=(f"No voice called {voice!r} on this machine. "
                    "Pick one from the list - nothing was played, rather than "
                    "playing a different voice and calling it this one."),
        )
    return engine_for_voice(voice or None)


def _pipeline(voice: Optional[str]) -> PodcastPipeline:
    engine = engine_for_voice(voice)
    if DEMO_MODE:
        # Reads the cache, never writes to it - a canned script must not end
        # up behind someone's real question later.
        return PodcastPipeline(generator=DemoGenerator(), engine=engine,
                               cache=SCRIPT_CACHE, voice=voice, cache_writes=False)
    return PodcastPipeline(engine=engine, cache=SCRIPT_CACHE, voice=voice)


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "mode": "demo" if DEMO_MODE else "live",
        "model": settings.model,
        "sample_rate": build_engine().sample_rate,
        "tts": engine_report(),
    }


@app.get("/api/voices")
async def voices() -> dict:
    return {"default": default_voice(), "voices": [v.as_dict() for v in list_voices()]}


@app.get("/api/audio")
async def audio(
    q: str = Query(..., max_length=500),
    minutes: int = Query(3, ge=1, le=10),
    voice: str = Query("", description="Voice id from /api/voices"),
    search: bool = Query(False),
):
    """A real FAM episode, in the chosen voice.

    Voice is not part of the script cache key, so asking the same question
    again in a different voice reuses the script: the comparison is the same
    words in two voices, and the second one costs nothing.
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="Ask a question first.")
    plan = plan_episode(q.strip(), minutes, "", search or None)
    try:
        _strict_engine(voice)          # refuse an unknown voice before generating
        pipe = _pipeline(voice or None)
    except TTSUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    stats = GenerationStats()
    rate = pipe.engine.sample_rate
    started = time.monotonic()
    source = pipe.stream_pcm(plan, stats)

    primed: list[bytes] = []
    try:
        async for chunk in source:
            primed.append(chunk)
            if sum(len(c) for c in primed) >= PREROLL_SECONDS * rate * 2:
                break
    except NotCached as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("generation failed before any audio")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if not primed:
        raise HTTPException(status_code=500, detail="The server produced no audio.")

    first_audio = time.monotonic() - started
    log.info("BENCH ask voice=%s engine=%s first_audio=%.2fs q=%r",
             voice or "(default)", pipe.engine.name, first_audio, q[:60])

    async def body():
        for chunk in primed:
            yield chunk
        async for chunk in source:
            yield chunk
        total = time.monotonic() - started
        log.info("BENCH ask done voice=%s sentences=%d audio=%.1fs wall=%.1fs synth=%.1fs",
                 voice or "(default)", stats.sentences, stats.audio_seconds,
                 total, stats.synth_seconds)

    return StreamingResponse(body(), media_type="audio/L16", headers={
        "Cache-Control": "no-store",
        "X-Accel-Buffering": "no",
        "X-Sample-Rate": str(rate),
        "X-First-Audio-Ms": str(int(first_audio * 1000)),
        "X-Engine": pipe.engine.name,
    })


@app.get("/api/speak")
async def speak(
    text: str = Query(..., max_length=MAX_SPEAK_CHARS),
    voice: str = Query("", description="Voice id from /api/voices"),
    wpm: float = Query(settings.target_wpm, ge=80, le=260),
):
    """Speak text exactly as given. No model call, no key required.

    This is the fair half of the bench: the same paragraph through every
    candidate, so what differs is the voice and nothing else. It also means a
    voice can be judged on a machine with no Anthropic key at all.
    """
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Type something to say first.")
    try:
        engine = _strict_engine(voice)
    except TTSUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    rate = engine.sample_rate
    sentences = _sentences(text)
    started = time.monotonic()

    # Synthesised up front rather than streamed, and deliberately so: this
    # endpoint exists to *measure* a voice, and the numbers worth having -
    # time to the first sentence, and how far ahead of playback it runs - are
    # only honest if nothing is overlapped with the download.
    try:
        first_at, pieces = None, []
        for sentence in sentences:
            pcm = await engine.synth(sentence, wpm, voice or None)
            if first_at is None:
                first_at = time.monotonic() - started
            pieces.append(pcm)
            pieces.append(silence(0.28, rate))
    except Exception as exc:
        log.exception("synthesis failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    audio = b"".join(pieces)
    elapsed = time.monotonic() - started
    seconds = pcm_duration(len(audio), rate)
    ratio = seconds / elapsed if elapsed else 0.0
    log.info("BENCH speak voice=%s engine=%s chars=%d sentences=%d "
             "first=%.2fs total=%.2fs audio=%.1fs (%.0fx realtime)",
             voice or "(default)", engine.name, len(text), len(sentences),
             first_at or 0.0, elapsed, seconds, ratio)

    return StreamingResponse(
        iter([_wav_header(len(audio) // 2, rate), audio]),
        media_type="audio/wav",
        headers={
            "Cache-Control": "no-store",
            "X-Sample-Rate": str(rate),
            "X-Engine": engine.name,
            # Read by the page and shown next to the play button. A voice that
            # sounds right and runs at 0.8x realtime cannot ship, and that is
            # invisible unless it is measured.
            "X-First-Audio-Ms": str(int((first_at or 0) * 1000)),
            "X-Synth-Ms": str(int(elapsed * 1000)),
            "X-Audio-Ms": str(int(seconds * 1000)),
            "X-Realtime-Factor": f"{ratio:.1f}",
        },
    )


def _sentences(text: str) -> list[str]:
    """Split on sentence ends, the same shape the pipeline feeds the engine.

    A voice judged on one long blob is not judged the way the product uses it:
    the pipeline synthesises a sentence at a time, and how a voice handles the
    start and end of each is most of what makes it sound stitched or natural.
    """
    import re

    parts = [p.strip() for p in re.split(r"(?<=[.!?…])\s+", text) if p.strip()]
    return parts or [text]


@app.get("/")
async def index():
    return FileResponse("static/bench.html")


@app.exception_handler(HTTPException)
async def http_error(_, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("bench_app:app", host=settings.host, port=settings.port, reload=False)
