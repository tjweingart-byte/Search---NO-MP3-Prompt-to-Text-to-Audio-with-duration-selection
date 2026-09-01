"""Web app: search box in, live audio out.

Endpoints
    GET  /                 the interface
    GET  /api/health       engine and configuration report
    POST /api/script       script only (JSON), for previewing or debugging
    GET  /api/audio        the podcast, streamed as live PCM or WAV

/api/audio is a GET on purpose so it can be used directly as an <audio> src.
"""
from __future__ import annotations

import os
import logging
import time
from contextlib import asynccontextmanager
from collections import defaultdict, deque
from typing import Optional, Union

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from anthropic_client import build_async_client, describe_http_version, http2_enabled
from cache import MemoryScriptCache, SqliteScriptCache, build_cache, research_words
from demo_script import DemoGenerator
from config import describe_key, settings
from pipeline import GenerationStats, NotCached, PodcastPipeline
from script_generator import ScriptGenerator, ScriptNotes, plan_episode
import attachments as attachments_mod
import topics as topics_mod
import mixes as mixes_mod
import social as social_mod
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

#: Filled in at startup by _verify_credentials. "unchecked" until then.
CREDENTIALS = {"state": "unchecked", "detail": "", "key": ""}


async def _verify_credentials() -> None:
    """Ask Claude whether the key works, before anyone presses play.

    Every credential failure this project has had was discovered by a listener,
    mid-episode, as a 502 - because the app validated its *configuration* (is a
    key set?) and never the credential (does it work?). A key that is missing,
    expired, revoked, truncated on paste, or simply the wrong string all look
    identical until the first request, and by then someone is waiting for audio.

    `models.retrieve` is the cheapest possible question: it bills nothing, and
    it answers both "is this key accepted" and "can this account use this
    model" - which are the two ways this has actually failed.
    """
    CREDENTIALS["key"] = describe_key()
    if DEMO_MODE:
        CREDENTIALS.update(state="absent", detail="No API key: the canned sample script is standing in.")
        log.warning("NO API KEY - every episode will be the built-in sample script, "
                    "which does not answer what was asked.")
        return
    try:
        client = build_async_client()
        await client.models.retrieve(settings.model)
    except Exception as exc:  # noqa: BLE001 - the report matters, not the type
        CREDENTIALS.update(state="rejected", detail=friendly_error(exc))
        log.error("CREDENTIALS REJECTED - nothing will generate. %s", CREDENTIALS["detail"])
        log.error("  key in force: %s", CREDENTIALS["key"])
        log.error("  fix it and restart; the interface says the same thing on every tab.")
        return
    CREDENTIALS.update(state="ok", detail=f"{settings.model} is reachable with this key.")
    log.info("credentials OK - %s reachable (%s)", settings.model, CREDENTIALS["key"])


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Pay the voice model's load cost now rather than on the first listener.
    await warm_up()
    # Before a listener finds out the hard way.
    await _verify_credentials()
    # Expired scripts are already filtered out on read, so nothing ever deleted
    # them and the file grew for the life of the deployment. One DELETE at
    # startup is enough: entries expire on a timescale of days, not minutes.
    purged = getattr(SCRIPT_CACHE, "purge_expired", lambda: 0)()
    if purged:
        log.info("cache: dropped %d expired script(s)", purged)
    stale = ATTACHMENTS.purge_expired()
    if stale:
        log.info("attachments: dropped %d expired", stale)
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
#: Recent cheap-read timestamps per client, for the burst-tolerant limiter.
_read_hits: dict[str, deque] = defaultdict(deque)
READ_WINDOW_SECONDS = 10.0


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
ATTACHMENTS = attachments_mod.AttachmentStore(
    os.environ.get("ATTACHMENTS_PATH", "attachments.db")
)

# With no credentials the app runs on a built-in sample script instead of
# refusing to start. Everything downstream of the model - streaming, pacing,
# duration matching, playback - is exercised for real; only the writer is
# canned. This is what makes the audio approach verifiable before anyone has
# an API key in place.
DEMO_MODE = not settings.anthropic_api_key


def _make_pipeline(voice: Optional[str] = None) -> PodcastPipeline:
    engine = engine_for_voice(voice)
    if DEMO_MODE:
        # Demo mode swaps the model, not the plumbing. It used to pass
        # cache=None, which quietly made Explore impossible without
        # credentials - and Explore is the one surface that needs no
        # credentials at all, since it only ever replays. Keeping the real
        # cache also means demo mode exercises the real hit/miss path.
        # Reads yes, writes never. The canned script does not answer the
        # question it was asked, so caching it puts a briefing about the audio
        # pipeline behind someone's search - for the whole TTL, and for every
        # other listener, including after a key is finally added.
        return PodcastPipeline(
            generator=DemoGenerator(), engine=engine, cache=SCRIPT_CACHE,
            voice=voice, cache_writes=False,
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

    This belongs on the endpoints that generate, and nowhere else. It was on
    all eighteen, including the cheap cache and JSON reads - and opening a tab
    fires several of those at once, so ordinary navigation answered itself with
    "Slow down a moment, then try again." A limiter that fires on correct use
    is not protecting anything; it is the failure.
    """
    if settings.rate_limit_seconds <= 0:
        return
    client = request.client.host if request.client else "anonymous"
    now = time.monotonic()
    if now - _last_request[client] < settings.rate_limit_seconds:
        raise HTTPException(status_code=429, detail="Slow down a moment, then try again.")
    _last_request[client] = now


def _read_limit(request: Request) -> None:
    """A ceiling for the cheap endpoints: JSON reads and cache lookups.

    These cost a SQLite query and no model call, and the interface fires a
    handful of them every time a tab opens, so the limit has to allow bursts.
    It exists to bound a script hammering the server, not to pace a listener.
    """
    if settings.read_limit_per_window <= 0:
        return
    client = request.client.host if request.client else "anonymous"
    now = time.monotonic()
    hits = _read_hits[client]
    cutoff = now - READ_WINDOW_SECONDS
    while hits and hits[0] < cutoff:
        hits.popleft()
    if len(hits) >= settings.read_limit_per_window:
        raise HTTPException(status_code=429, detail="Slow down a moment, then try again.")
    hits.append(now)


def _validated_plan(q: str, minutes: int, context: str = "", search: bool | None = None,
                    cached_only: bool = False, attachments: tuple = ()):
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
    return plan_episode(q, minutes, (context or "").strip()[:300], search, cached_only,
                        attachments)


class ScriptRequest(BaseModel):
    query: str = Field(..., max_length=500)
    minutes: int = Field(..., ge=1, le=10)
    search: bool = False


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "mode": "demo" if DEMO_MODE else "live",
        "model": settings.model,
        "web_search_default": settings.enable_web_search,
        "http": {"version": describe_http_version(), "http2_negotiated": http2_enabled()},
        "api_key_configured": bool(settings.anthropic_api_key),
        # Configured is not the same as working, and only one of them matters.
        "credentials": CREDENTIALS,
        "sample_rate": build_engine().sample_rate,
        "min_minutes": settings.min_minutes,
        "max_minutes": settings.max_minutes,
        "tts": engine_report(),
        # How the listener is told what is happening while they wait. There is
        # no filler any more, so the interface has to be honest instead.
        "search_mode": settings.search_mode,
        "research_words": sorted(research_words()),
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
    notes = ScriptNotes()
    text = " ".join([s async for s in generator.stream_sentences(plan, notes)])
    return {
        "query": plan.query,
        "minutes": plan.minutes,
        "word_budget": plan.word_budget,
        "words": len(text.split()),
        "script": text,
        "thread": notes.thread,
    }


EVENTS = topics_mod.EventStore(os.environ.get("MYFAM_DB", "myfam.db"))
MIXES = mixes_mod.MixStore(os.environ.get("MIXES_DB", "mixes.db"))
SOCIAL = social_mod.SocialStore(os.environ.get("SOCIAL_DB", "social.db"))


class MixRequest(BaseModel):
    user: str = Field(..., max_length=64)
    name: Optional[str] = Field(None, max_length=mixes_mod.MAX_NAME)
    #: Bank ids as strings, or {"query": "..."} for a topic the listener typed.
    #: Validated in mixes.clean_items rather than here, so one place owns the
    #: rules and the message the listener sees.
    topic_ids: Optional[list[Union[str, dict]]] = None
    #: Public mixes appear on the listener's profile.
    public: Optional[bool] = None


def _attachments_for(user: str, ids: str) -> tuple:
    """Resolve `attach=` into stored attachments, or say which one is gone.

    Silently dropping an expired attachment would produce an episode about a
    document the listener believes was read and was not - the exact failure
    this project treats as worse than an error.
    """
    wanted = [i for i in (ids or "").split(",") if i.strip()]
    if not wanted:
        return ()
    found = ATTACHMENTS.resolve(user, wanted)
    if len(found) != len(wanted):
        raise HTTPException(
            status_code=410,
            detail="An attachment has expired. Add it again and re-run the search.",
        )
    return tuple(found)


class AttachRequest(BaseModel):
    user: str = Field("", max_length=64)
    kind: str = Field(..., pattern="^(document|image|link)$")
    name: str = Field("", max_length=300)
    data: str = Field("", description="Base64 file contents; documents and photos")
    url: str = Field("", max_length=2000)


@app.post("/api/attach")
async def attach(req: AttachRequest, request: Request) -> dict:
    """Extract a document, photo or link once, when it is added.

    Deliberately not on the generation path: reading a PDF or fetching a page
    is a round-trip, and the one thing this product will not spend is seconds
    in front of the first word. Doing it here puts the cost while someone is
    still typing.
    """
    _rate_limit(request)
    try:
        item = attachments_mod.build(req.kind, req.name, req.data, req.url)
    except attachments_mod.AttachmentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    ATTACHMENTS.put(req.user, item)
    return item.as_dict()


@app.delete("/api/attach")
async def detach(request: Request, user: str = Query("", max_length=64),
                 id: str = Query(..., max_length=64)) -> dict:
    _read_limit(request)
    return {"ok": ATTACHMENTS.delete(user, id)}


@app.get("/api/topics")
async def bank(request: Request):
    """The whole shared bank, for the mix topic picker."""
    _read_limit(request)
    return {"topics": [t.as_dict() for t in topics_mod.TOPIC_BANK]}


@app.get("/api/mixes")
async def list_mixes(request: Request, user: str = Query("", max_length=64)):
    """This listener's playFAM mixes, each with its topics resolved."""
    _read_limit(request)
    return {
        "mixes": [m.as_dict() for m in MIXES.list_for_user(user)],
        "starters": [
            {"name": name, "topic_ids": list(ids)}
            for name, ids in mixes_mod.STARTER_MIXES
        ],
    }


@app.post("/api/mixes")
async def create_mix(req: MixRequest, request: Request):
    _read_limit(request)
    try:
        mix = MIXES.create(req.user, req.name or "", req.topic_ids or [])
    except mixes_mod.MixError as exc:
        # Phrased for the listener: these are things they did, not faults.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return mix.as_dict()


@app.patch("/api/mixes/{mix_id}")
async def update_mix(mix_id: str, req: MixRequest, request: Request):
    _read_limit(request)
    try:
        mix = MIXES.update(req.user, mix_id, req.name, req.topic_ids, req.public)
    except mixes_mod.MixError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return mix.as_dict()


@app.delete("/api/mixes/{mix_id}")
async def delete_mix(mix_id: str, request: Request, user: str = Query("", max_length=64)):
    _read_limit(request)
    if not MIXES.delete(user, mix_id):
        raise HTTPException(status_code=404, detail="That mix no longer exists.")
    return {"ok": True}


class EventRequest(BaseModel):
    user: str = Field(..., max_length=64)
    kind: str = Field(..., max_length=16)
    topic_id: str = Field("", max_length=64)
    text: str = Field("", max_length=300)
    #: The follow-up predicted for the finished episode, so Go Deeper can offer it
    #: back later without a second lookup.
    thread: str = Field("", max_length=200)


@app.get("/api/myfam")
async def myfam(request: Request, user: str = Query("", max_length=64)):
    """The four myFAM sections, ranked for this listener.

    Costs no model call: the topic bank is fixed and this only orders it.
    A listener with no history still gets Trending and a starter set, with
    the personal sections honestly empty rather than filled with fakes.
    """
    _read_limit(request)
    return topics_mod.build_feed(EVENTS, user)


@app.post("/api/event")
async def record_event(req: EventRequest, request: Request):
    """Log one interaction. Playback never depends on this succeeding."""
    _read_limit(request)
    tags = ()
    if req.topic_id and req.topic_id in topics_mod.BANK_BY_ID:
        tags = topics_mod.BANK_BY_ID[req.topic_id].tags
    elif req.text:
        tags = topics_mod.tags_for_text(req.text)
    EVENTS.record(
        topics_mod.Event(req.user, req.kind, req.topic_id, req.text, tags,
                         thread=req.thread)
    )
    return {"ok": True}


class PersonRequest(BaseModel):
    user: str = Field(..., max_length=64)
    name: str = Field("", max_length=social_mod.MAX_NAME)
    handle: str = Field("", max_length=social_mod.MAX_HANDLE + 1)


class EchoRequest(BaseModel):
    user: str = Field(..., max_length=64)
    query: str = Field(..., max_length=300)
    title: str = Field("", max_length=200)
    minutes: int = Field(3, ge=1, le=10)
    thread: str = Field("", max_length=200)


@app.post("/api/me")
async def set_me(req: PersonRequest, request: Request):
    """Name and handle for this device. Not an account - see /api/profile."""
    _read_limit(request)
    try:
        return SOCIAL.set_person(req.user, req.name, req.handle)
    except social_mod.SocialError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/echo")
async def post_echo(req: EchoRequest, request: Request):
    """Push a finished episode to the people who follow this listener.

    Costs nothing to generate: an echo is a row pointing at a query whose
    script already exists, which is exactly why the social layer is cheap.
    """
    _read_limit(request)
    try:
        echo = SOCIAL.echo(req.user, req.query, req.title, req.minutes, req.thread)
    except social_mod.SocialError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return echo.as_dict()


@app.delete("/api/echo")
async def delete_echo(request: Request, user: str = Query("", max_length=64),
                      q: str = Query("", max_length=300),
                      minutes: int = Query(3, ge=1, le=10)):
    _read_limit(request)
    return {"ok": SOCIAL.unecho(user, q, minutes)}


@app.get("/api/profile")
async def profile(request: Request, user: str = Query("", max_length=64)):
    """Counts and subjects from this listener's own event log. No model call."""
    _read_limit(request)
    body = topics_mod.summary(EVENTS, user)
    person = SOCIAL.person(user)
    body["name"] = person["name"]
    body["handle"] = person["handle"]
    body["joined"] = person["joined"]
    body["mixes"] = [m.as_dict() for m in MIXES.public_for_user(user)]
    body["echoes"] = [e.as_dict(person["name"], person["handle"])
                      for e in SOCIAL.echoes_by(user, limit=12)]
    body["echo_count"] = len(SOCIAL.echoes_by(user, limit=200))
    return body


@app.get("/api/godeeper")
async def go_deeper(request: Request, user: str = Query("", max_length=64)):
    """Follow-ups predicted for the episodes this listener finished.

    Costs nothing: the model named it on the episode's trailing marker line,
    which was never spoken. The episode itself does not tease it - it simply
    ends - and the suggestion is waiting here afterwards for anyone who wants
    to keep going.
    """
    _read_limit(request)
    return {"threads": EVENTS.open_threads(user)}


@app.get("/api/explore")
async def explore(request: Request, limit: int = Query(30, ge=1, le=60),
                  user: str = Query("", max_length=64)):
    """Episodes other listeners have already generated, newest first.

    This endpoint costs nothing and, by design, can cause nothing to be
    generated: it reads finished scripts out of the cache. Everything in that
    cache passed the personal-query filter before it was written, so it is
    already safe to show someone else.
    """
    _read_limit(request)
    store = SCRIPT_CACHE if SCRIPT_CACHE is not None else build_cache()
    if store is None:
        return {"episodes": [], "reason": "The shared cache is switched off."}
    now = time.time()
    # Who echoed what. An echo does not create an episode - the script was
    # already here - it changes what the card says, from "someone asked this"
    # to "Rachel sent you this", which is a different reason to press play.
    labels = SOCIAL.recent_echoes(exclude_user=user)
    episodes = [
        {
            "query": entry["query"],
            "title": entry["query"][:1].upper() + entry["query"][1:],
            "minutes": entry["minutes"],
            "plays": entry["plays"],
            "thread": entry["thread"],
            "age_seconds": max(0.0, now - entry["created"]),
            "echoed_by": labels.get((entry["query"], entry["minutes"]), {}).get("by", ""),
        }
        for entry in store.recent(limit)
    ]
    # An echoed episode leads, because someone chose to send it.
    episodes.sort(key=lambda e: (not e["echoed_by"], e["age_seconds"]))
    return {"episodes": episodes}


@app.get("/api/next")
async def next_thread(
    request: Request,
    q: str = Query(..., description="What the listener asked"),
    minutes: int = Query(3, ge=1, le=10),
    context: str = Query("", description="Topic the listener just heard"),
    search: bool = Query(False),
):
    """The follow-up this listener is most likely to want, after this episode.

    Read from the script cache, so it costs no tokens and no time. The interface
    offers it as a one-tap suggestion in Go Deeper: an episode that ends pointed
    at something specific is only half the job if acting on it still means
    composing a question into an empty box.

    An empty thread is normal - the script may not be cached, or the model may
    not have named one - and the interface falls back to the blank field.
    """
    _read_limit(request)
    plan = _validated_plan(q, minutes, context, search)
    try:
        pipeline = _make_pipeline()
    except TTSUnavailable:
        return {"thread": ""}
    return {"thread": await pipeline.thread_for(plan)}


@app.get("/api/audio")
async def audio(
    request: Request,
    q: str = Query(..., description="What the listener asked"),
    minutes: int = Query(3, ge=1, le=10),
    fmt: str = Query("wav", pattern="^(wav|pcm)$"),
    context: str = Query("", description="Topic the listener just heard, for a follow-up"),
    voice: str = Query("", description="Voice id from /api/voices"),
    search: bool = Query(False, description="Look up live sources; adds 10-25s before audio"),
    user: str = Query("", max_length=64, description="Anonymous listener id, for myFAM"),
    cached_only: bool = Query(False, description="Replay only; never generate. Used by Explore"),
    topic_id: str = Query("", max_length=64, description="Bank topic id, when played from myFAM"),
    attach: str = Query("", max_length=400, description="Attachment ids from /api/attach"),
):
    """Stream the episode.

    `fmt=wav` prefixes a live-stream WAV header so a plain <audio> tag works.
    `fmt=pcm` sends bare samples for the Web Audio player, which schedules
    chunks itself and therefore starts sooner and seeks better.
    """
    # The pace exists to bound model spend. A replay-only request - Explore,
    # and any card played from it - provably cannot spend one, so pacing it
    # only stops someone swiping a feed at a normal speed, which is exactly
    # what the feed is for.
    (_read_limit if cached_only else _rate_limit)(request)
    plan = _validated_plan(q, minutes, context, search, cached_only,
                           _attachments_for(user, attach))

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
    except NotCached as exc:
        # Expected, not a fault: the entry expired between listing and tapping.
        # The interface drops the card and moves on.
        raise HTTPException(status_code=409, detail=str(exc)) from exc
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

    # Recorded here rather than client-side: audio is being served, so the
    # play is a fact. A dropped event costs one weak signal, never the episode.
    if user:
        EVENTS.record(
            topics_mod.Event(
                user, "play", topic_id, plan.query,
                topics_mod.BANK_BY_ID[topic_id].tags
                if topic_id in topics_mod.BANK_BY_ID
                else topics_mod.tags_for_text(plan.query),
            )
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
