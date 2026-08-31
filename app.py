"""Web app: search box in, live audio out.

Endpoints
    GET  /                 the interface
    GET  /api/health       engine and configuration report
    POST /api/script       script only (JSON), for previewing or debugging
    GET  /api/audio        the podcast, streamed as live PCM or WAV

/api/audio is a GET on purpose so it can be used directly as an <audio> src.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from collections import defaultdict
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from anthropic_client import describe_http_version, http2_enabled
from cache import MemoryScriptCache, SqliteScriptCache, build_cache
from demo_script import DemoGenerator
from config import settings
from pipeline import GenerationStats, PodcastPipeline
from script_generator import ScriptGenerator, plan_episode
import voice_store
from tts import (
    TTSUnavailable,
    build_engine,
    default_voice,
    engine_for_voice,
    engine_report,
    list_voices,
    warm_up,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("podcast")

# Prepare the shared voice store before anything asks it what it holds. On the
# first run of a new version this adopts voices an older project folder already
# downloaded; every run after, it is a no-op.
VOICE_STORE = voice_store.ensure_ready()
if VOICE_STORE["adopted"]:
    log.info(
        "reused %d voice(s) from a previous version of the app: %s",
        len(VOICE_STORE["adopted"]), ", ".join(VOICE_STORE["adopted"]),
    )
log.info("voices: %s", voice_store.describe())

@asynccontextmanager
async def lifespan(_: FastAPI):
    # Pay the voice model's load cost now rather than on the first listener.
    await warm_up()
    yield


app = FastAPI(title="Search to Podcast", version="1.0.0", lifespan=lifespan)

# A streamed WAV opens with a 44-byte header, which is not audio.
WAV_HEADER_BYTES = 44

# Hold this much audio before playing anything. Models stream in bursts, so
# starting on the very first sentence means a stall becomes an audible hole a
# second in. With a fast model this costs almost nothing: synthesis runs many
# times faster than speech, so a few seconds of audio arrives in a fraction of
# a second. It is the difference between "starts instantly" and "starts
# instantly and keeps going".
PREROLL_SECONDS = 1.5

_last_request: dict[str, float] = defaultdict(float)


def friendly_error(exc: Exception) -> str:
    """Turn an SDK failure into something the person in the browser can act on."""
    import anthropic

    if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)) or (
        isinstance(exc, TypeError) and "authentication method" in str(exc)
    ):
        return (
            "Claude rejected the credentials. Set ANTHROPIC_API_KEY in .env "
            "(or run `ant auth login`) and restart the server."
        )
    if isinstance(exc, anthropic.NotFoundError):
        return f"The model {settings.model!r} is not available to this account. Try MODEL=claude-sonnet-5."
    if isinstance(exc, anthropic.RateLimitError):
        return "Claude is rate limiting this key. Wait a moment and try again."
    if isinstance(exc, anthropic.APIConnectionError):
        return "Could not reach the Claude API. Check the server's network access."
    return f"Generation failed: {type(exc).__name__}. See the server log for details."

# One cache shared by every request this worker serves - and, with the SQLite
# backend, by every other worker on the machine too.
SCRIPT_CACHE = build_cache()

# With no credentials the app runs on a built-in sample script instead of
# refusing to start. Everything downstream of the model - streaming, pacing,
# duration matching, playback - is exercised for real; only the writer is
# canned. This is what makes the audio approach verifiable before anyone has
# an API key in place.
DEMO_MODE = not settings.anthropic_api_key


def _make_pipeline(voice: Optional[str] = None) -> PodcastPipeline:
    engine = engine_for_voice(voice)
    if DEMO_MODE:
        return PodcastPipeline(
            generator=DemoGenerator(), engine=engine, cache=None, voice=voice
        )
    return PodcastPipeline(engine=engine, cache=SCRIPT_CACHE, voice=voice)


def _cache_report() -> dict:
    if SCRIPT_CACHE is None:
        return {"enabled": False}
    report = {"enabled": True, "semantic_key": settings.cache_semantic_key}
    if isinstance(SCRIPT_CACHE, (MemoryScriptCache, SqliteScriptCache)):
        report.update(SCRIPT_CACHE.stats())
    return report


def _rate_limit(request: Request) -> None:
    """One generation per client per RATE_LIMIT_SECONDS.

    Each request holds a Claude stream and a TTS subprocess open for the whole
    episode, so an unthrottled endpoint is trivially expensive to abuse.
    """
    if settings.rate_limit_seconds <= 0:
        return
    client = request.client.host if request.client else "anonymous"
    now = time.monotonic()
    if now - _last_request[client] < settings.rate_limit_seconds:
        raise HTTPException(status_code=429, detail="Slow down a moment, then try again.")
    _last_request[client] = now


def _validated_plan(q: str, minutes: int, context: str = "", search: bool | None = None):
    q = (q or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Ask a question first.")
    if len(q) > 500:
        raise HTTPException(status_code=400, detail="Query is too long (500 characters max).")
    if not settings.min_minutes <= minutes <= settings.max_minutes:
        raise HTTPException(
            status_code=400,
            detail=f"Length must be {settings.min_minutes}-{settings.max_minutes} minutes.",
        )
    return plan_episode(q, minutes, (context or "").strip()[:300], search)


class ScriptRequest(BaseModel):
    query: str = Field(..., max_length=500)
    minutes: int = Field(..., ge=1, le=10)


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "mode": "demo" if DEMO_MODE else "live",
        "model": settings.model,
        "web_search_default": settings.enable_web_search,
        "http": {"version": describe_http_version(), "http2_negotiated": http2_enabled()},
        "api_key_configured": bool(settings.anthropic_api_key),
        "sample_rate": build_engine().sample_rate,
        "min_minutes": settings.min_minutes,
        "max_minutes": settings.max_minutes,
        "tts": engine_report(),
        "cold_open": {
            "enabled": settings.enable_cold_open,
            "model": settings.cold_open_model,
        },
        "cache": _cache_report(),
        "voice_store": VOICE_STORE["dir"],
    }


@app.get("/api/voices")
async def voices() -> dict:
    """Voices this server can speak in, best first."""
    return {
        "default": default_voice(),
        "store": VOICE_STORE["dir"],
        "voices": [v.as_dict() for v in list_voices()],
    }


@app.post("/api/script")
async def script(req: ScriptRequest, request: Request) -> dict:
    _rate_limit(request)
    plan = _validated_plan(req.query, req.minutes)
    generator = DemoGenerator() if DEMO_MODE else ScriptGenerator()
    text = " ".join([s async for s in generator.stream_sentences(plan)])
    return {
        "query": plan.query,
        "minutes": plan.minutes,
        "word_budget": plan.word_budget,
        "words": len(text.split()),
        "script": text,
    }


@app.get("/api/audio")
async def audio(
    request: Request,
    q: str = Query(..., description="What the listener asked"),
    minutes: int = Query(3, ge=1, le=10),
    fmt: str = Query("wav", pattern="^(wav|pcm)$"),
    context: str = Query("", description="Topic the listener just heard, for a follow-up"),
    voice: str = Query("", description="Voice id from /api/voices"),
    search: bool = Query(False, description="Look up live sources; adds 10-25s before audio"),
):
    """Stream the episode.

    `fmt=wav` prefixes a live-stream WAV header so a plain <audio> tag works.
    `fmt=pcm` sends bare samples for the Web Audio player, which schedules
    chunks itself and therefore starts sooner and seeks better.
    """
    _rate_limit(request)
    plan = _validated_plan(q, minutes, context, search)

    try:
        pipeline = _make_pipeline(voice or None)
    except TTSUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    stats = GenerationStats()
    started = time.monotonic()
    # The player must be told the engine's real rate, not the configured one.
    sample_rate = pipeline.engine.sample_rate

    source = pipeline.stream_wav(plan, stats) if fmt == "wav" else pipeline.stream_pcm(plan, stats)

    # Pull chunks until real audio exists BEFORE returning a response. Once the
    # first byte is sent the status code is fixed, so a failure after that point
    # can only be logged - which is how a broken API key used to arrive at the
    # browser as a successful, silent, empty episode. Priming here means such a
    # failure becomes a proper error the interface can show.
    primed: list[bytes] = []
    preroll_bytes = int(PREROLL_SECONDS * sample_rate * 2)
    try:
        async for chunk in source:
            primed.append(chunk)
            if sum(len(c) for c in primed) - WAV_HEADER_BYTES >= preroll_bytes:
                break
    except Exception as exc:
        log.exception("generation failed before any audio was produced")
        raise HTTPException(status_code=502, detail=friendly_error(exc)) from exc

    # `stats.sentences` is the honest test: silence is bytes, but it is not an
    # episode. A script that came back empty must not be served as one.
    if stats.sentences == 0 or sum(len(c) for c in primed) <= WAV_HEADER_BYTES:
        log.error("generation produced no audio for %r", plan.query)
        raise HTTPException(
            status_code=502,
            detail="The episode came back with no speech in it. Check the server "
            "log, and that ANTHROPIC_API_KEY is set and a speech engine is installed.",
        )

    async def body():
        try:
            for chunk in primed:
                yield chunk
            async for chunk in source:
                if await request.is_disconnected():
                    log.info("client disconnected; abandoning generation")
                    break
                yield chunk
        except Exception:
            # Past the first byte the status code is already sent, so this can
            # only be logged. The player detects the short stream and says so.
            log.exception("audio stream failed mid-flight")
        finally:
            log.info(
                "episode q=%r %s wall=%.1fs", plan.query, stats.as_dict(), time.monotonic() - started
            )

    media_type = "audio/wav" if fmt == "wav" else "audio/L16"
    return StreamingResponse(
        body(),
        media_type=media_type,
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",  # tell nginx not to buffer the stream
            "X-Sample-Rate": str(sample_rate),
            "X-Requested-Seconds": str(plan.target_seconds),
        },
    )


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host=settings.host, port=settings.port, reload=False)
