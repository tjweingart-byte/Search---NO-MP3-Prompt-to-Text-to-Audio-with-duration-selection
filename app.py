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
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import settings
from pipeline import GenerationStats, PodcastPipeline
from script_generator import ScriptGenerator, plan_episode
from tts import TTSUnavailable, build_engine, engine_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("podcast")

app = FastAPI(title="Search to Podcast", version="1.0.0")

_last_request: dict[str, float] = defaultdict(float)


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


def _validated_plan(q: str, minutes: int):
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
    return plan_episode(q, minutes)


class ScriptRequest(BaseModel):
    query: str = Field(..., max_length=500)
    minutes: int = Field(..., ge=1, le=10)


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "model": settings.model,
        "web_search": settings.enable_web_search,
        "api_key_configured": bool(settings.anthropic_api_key),
        "sample_rate": build_engine().sample_rate,
        "min_minutes": settings.min_minutes,
        "max_minutes": settings.max_minutes,
        "tts": engine_report(),
    }


@app.post("/api/script")
async def script(req: ScriptRequest, request: Request) -> dict:
    _rate_limit(request)
    plan = _validated_plan(req.query, req.minutes)
    text = await ScriptGenerator().full_script(plan)
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
):
    """Stream the episode.

    `fmt=wav` prefixes a live-stream WAV header so a plain <audio> tag works.
    `fmt=pcm` sends bare samples for the Web Audio player, which schedules
    chunks itself and therefore starts sooner and seeks better.
    """
    _rate_limit(request)
    plan = _validated_plan(q, minutes)

    try:
        pipeline = PodcastPipeline(engine=build_engine())
    except TTSUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    stats = GenerationStats()
    started = time.monotonic()
    # The player must be told the engine's real rate, not the configured one.
    sample_rate = pipeline.engine.sample_rate

    async def body():
        try:
            source = pipeline.stream_wav(plan, stats) if fmt == "wav" else pipeline.stream_pcm(plan, stats)
            async for chunk in source:
                if await request.is_disconnected():
                    log.info("client disconnected; abandoning generation")
                    break
                yield chunk
        except Exception:
            # The response has already begun, so an error cannot become a 500.
            # Log it and close the stream; the player stops at what it has.
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
