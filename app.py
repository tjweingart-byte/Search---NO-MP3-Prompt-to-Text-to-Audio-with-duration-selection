"""Web app: search box in, live audio out.

Endpoints
    GET  /                 the interface
    GET  /api/health       engine and configuration report
    POST /api/script       script only (JSON), for previewing or debugging
    GET  /api/audio        the podcast, streamed as live PCM or WAV

/api/audio is a GET on purpose so it can be used directly as an <audio> src.
"""
from __future__ import annotations

import asyncio
import hmac
import os
import json
import logging
import sqlite3
import time
from contextlib import asynccontextmanager
from urllib.parse import quote
from collections import defaultdict, deque
from typing import Optional, Union

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi import Response
from fastapi.responses import (
    JSONResponse, RedirectResponse, StreamingResponse)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from anthropic_client import build_async_client, describe_http_version, http2_enabled
from cache import MemoryScriptCache, SqliteScriptCache, build_cache, research_words
import embeddings
from demo_script import DemoGenerator
import credentials
import entitlements
import messages as messages_mod
import metering
import oauth
import quotas
import saved as saved_mod
import sharing
from config import DEFAULT_PIPELINE, describe_key, key_source, settings
from research import ResearchUnavailable, report as research_report
from pipeline import GenerationStats, NotCached, PodcastPipeline
from script_generator import ScriptGenerator, ScriptNotes, plan_episode
import attachments as attachments_mod
import topics as topics_mod
import accounts as accounts_mod
from paths import PROJECT_ROOT
import mixes as mixes_mod
import preferences as prefs_mod
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
CREDENTIALS = {"state": "unchecked", "detail": "", "key": "", "source": "", "secrets": {}}


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

    A rejection now has two things to try before it is reported, and the order
    matters. **Re-read the provider first**: the commonest reason a key that
    worked yesterday is refused today is that it was rotated, and the new one
    is already sitting in the secrets manager. **Then fail over**, if a pool was
    configured. Only when both are spent is this a rejection - which is what it
    always was, said at the same place, in the same words.
    """
    credentials.prime()
    CREDENTIALS["key"] = describe_key()
    CREDENTIALS["source"] = key_source()
    CREDENTIALS["secrets"] = credentials.report()
    if DEMO_MODE:
        CREDENTIALS.update(state="absent", detail="No API key: the canned sample script is standing in.")
        log.warning("NO API KEY - every episode will be the built-in sample script, "
                    "which does not answer what was asked.")
        _say_where_a_key_could_come_from()
        return
    rotated = False
    while True:
        try:
            client = build_async_client()
            await client.models.retrieve(settings.model)
        except Exception as exc:  # noqa: BLE001 - the report matters, not the type
            detail = friendly_error(exc)
            if not rotated and credentials.refresh("the key in force was rejected"):
                # The provider answered. Start again at the top of the pool: the
                # keys behind the rejected one may have been rotated as well.
                rotated = True
                credentials.reset("ANTHROPIC_API_KEY")
                CREDENTIALS["key"] = describe_key()
                CREDENTIALS["source"] = key_source()
                continue
            if credentials.demote("ANTHROPIC_API_KEY", detail):
                CREDENTIALS["key"] = describe_key()
                continue
            CREDENTIALS.update(state="rejected", detail=detail,
                               secrets=credentials.report())
            log.error("CREDENTIALS REJECTED - nothing will generate. %s", CREDENTIALS["detail"])
            log.error("  key in force: %s", CREDENTIALS["key"])
            log.error("  it came from: %s", CREDENTIALS["source"])
            log.error("  fix it and restart; the interface says the same thing on every tab.")
            return
        break
    CREDENTIALS.update(state="ok", detail=f"{settings.model} is reachable with this key.",
                       key=describe_key(), source=key_source(),
                       secrets=credentials.report())
    log.info("credentials OK - %s reachable (%s from %s)", settings.model,
             CREDENTIALS["key"], CREDENTIALS["source"])


def _say_where_a_key_could_come_from() -> None:
    """With no key, say the thing that stops this happening on the next machine.

    A fresh pod, a fresh container and a colleague's laptop all arrive here, and
    the answer that has been given four times is "paste it again". It is worth
    one line at the exact moment somebody is about to.
    """
    report = credentials.report()
    if report["state"] == "failed":
        log.error("  %s IS set and could not be read: %s",
                  credentials.PROVIDER_VAR, report["detail"])
        log.error("  That is why there is no key. Fix the provider, not the app.")
        return
    if report["configured"]:
        log.warning("  %s is set (%s) but supplied no ANTHROPIC_API_KEY.",
                    credentials.PROVIDER_VAR, ", ".join(report["provider"]))
        return
    log.warning("  On this machine:   python setup_key.py")
    log.warning("  On every machine:  set %s, e.g.", credentials.PROVIDER_VAR)
    log.warning("    %s='cmd:aws secretsmanager get-secret-value "
                "--secret-id fam --query SecretString --output text'",
                credentials.PROVIDER_VAR)
    log.warning("  See CREDENTIALS.md. A machine that has never run FAM needs "
                "that one line and nothing typed.")


def _announce_research() -> None:
    """Say at startup whether the configured research backend can actually run.

    `exa` is the default and needs a second credential. Without it a researched
    episode fails - it does not quietly search another way - and finding that
    out on a listener's first researched question is the shape of failure this
    project has paid for most. So it is said here, once, loudly, and again on
    every /api/health.

    Not fatal. Most questions are not researched, and an app that refuses to
    start because one path is unconfigured is worse than one that starts and
    says which path is unavailable.
    """
    report = research_report()
    if not report["unavailable"]:
        log.info("research: %s (%s)", report["backend"], report["exa_detail"])
        return
    log.warning(
        "RESEARCH UNAVAILABLE: RESEARCH_BACKEND=%s but %s.", report["backend"],
        report["exa_detail"])
    log.warning(
        "  Researched episodes will FAIL rather than search another way.")
    log.warning(
        "  Set EXA_API_KEY, or set RESEARCH_BACKEND=claude to let the model "
        "search instead.")
    log.warning("  Every tab says the same thing; /api/health carries it too.")


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Pay the voice model's load cost now rather than on the first listener.
    await warm_up()
    # Before a listener finds out the hard way.
    await _verify_credentials()
    _announce_research()
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
#
# A **quantity**, not a delay: the gate below counts bytes of audio, not
# elapsed time. At TARGET_WPM this is 3.75 words, so an ordinary opening
# sentence satisfies it on the first chunk and it costs nothing at all. It
# only forces a second synthesis when the opening is very short - which is the
# case the Phase 6 first-chunk rule deliberately allows, so the two interact.
#
# Configurable since the preroll sweep, so the value can be measured rather
# than argued about. The default is unchanged, and zero is refused in
# `config.Settings.__post_init__`.
PREROLL_SECONDS = settings.preroll_seconds

_last_request: dict[str, float] = defaultdict(float)
#: Recent cheap-read timestamps per client, for the burst-tolerant limiter.
_read_hits: dict[str, deque] = defaultdict(deque)
READ_WINDOW_SECONDS = 10.0


def _ms(value: float | None) -> str:
    """A mark as milliseconds, or a dash when it never happened."""
    return "-" if value is None else f"{value * 1000:.0f}ms"


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
    if isinstance(exc, ResearchUnavailable):
        # This one already carries the remedy - "exa_py is not installed,
        # `pip install -r requirements-exa.txt`", or which key is missing.
        # Replacing that with "see the server log" throws away the one
        # sentence that would let the person fix it, which is the whole job
        # of this function. Every other component names what is wrong.
        return f"This question needed research and the backend could not run. {exc}"
    return f"Generation failed: {type(exc).__name__}. See the server log for details."

# One cache shared by every request this worker serves - and, with the SQLite
# backend, by every other worker on the machine too.
SCRIPT_CACHE = build_cache()
ATTACHMENTS = attachments_mod.AttachmentStore()

# With no credentials the app runs on a built-in sample script instead of
# refusing to start. Everything downstream of the model - streaming, pacing,
# duration matching, playback - is exercised for real; only the writer is
# canned. This is what makes the audio approach verifiable before anyone has
# an API key in place.
DEMO_MODE = not settings.anthropic_api_key


def _wake_remote_voice() -> None:
    """Fire-and-forget: start a remote GPU booting, if one is configured.

    Deliberately not awaited. The wake is worth several seconds when it lands
    and must be worth zero when it does not, so nothing here may raise, block,
    or keep a reference the request has to clean up. A no-op for every backend
    but a serverless one, where there is genuinely something asleep.
    """
    if settings.voice_backend != "remote":
        return
    try:
        import remote_voice

        task = asyncio.create_task(remote_voice.RemoteChatterboxEngine.wake())
        # Held so the loop cannot garbage-collect a running task, and dropped
        # the moment it finishes.
        _WAKES.add(task)
        task.add_done_callback(_WAKES.discard)
    except Exception as exc:  # pragma: no cover - a hint that cannot cost one
        log.debug("could not wake the remote voice: %s", exc)


#: Strong references to in-flight wake tasks. asyncio keeps only weak ones, so
#: without this a wake can be collected mid-flight and silently never sent.
_WAKES: set = set()


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
    # Near matching is the one cache setting that can serve a *wrong* episode,
    # so the health report says whether it is on and, if it is, what kind of
    # embedding is behind it. "vector matching on" reads like semantics; with
    # no model installed it is lexical, and the difference decides how much to
    # trust a hit. Reporting one without the other would be the §52 mistake in
    # a new place.
    report["near_match"] = settings.cache_vector
    if settings.cache_vector:
        report["embedding"] = embeddings.describe()
        report["threshold"] = settings.cache_vector_threshold
        report["overlap"] = settings.cache_vector_overlap
    if isinstance(SCRIPT_CACHE, (MemoryScriptCache, SqliteScriptCache)):
        report.update(SCRIPT_CACHE.stats())
    return report


def _database_report() -> list[dict]:
    """Where each database actually is, and whether it really opens.

    §52's rule applied to storage: `"status": "ok"` used to be reported while
    four of the five could be pointed anywhere or be unwritable, and no runtime
    surface named a single path. Configuration was being confirmed instead of
    readiness being verified.

    So each entry performs a real read - `SELECT count(*) FROM sqlite_master`,
    which forces the file header and schema to be parsed - and reports the path
    the store is genuinely holding rather than one re-derived here.

    It is deliberately *not* `SELECT 1`: that is a constant expression, answered
    without touching the file, so it returns happily for a path containing
    nothing but rubbish. This check was written that way first and a test caught
    it reporting a corrupt database as readable - the same mistake §52 is about,
    made inside the code meant to prevent it. `writable` is a permission check and is labelled as one; writing on
    every health poll would cost more than it tells anyone.
    """
    stores = [
        ("scripts", "CACHE_PATH", getattr(SCRIPT_CACHE, "path", "")),
        ("events", "MYFAM_DB", EVENTS.path),
        ("social", "SOCIAL_DB", SOCIAL.path),
        ("mixes", "MIXES_DB", MIXES.path),
        ("attachments", "ATTACHMENTS_PATH", ATTACHMENTS.path),
        ("accounts", "ACCOUNTS_DB", ACCOUNTS.path),
        ("preferences", "PREFS_DB", PREFS.path),
    ]
    report = []
    for name, env_var, path in stores:
        entry = {
            "name": name,
            "env_var": env_var,
            "path": path or "(in memory)",
            # Whether this machine was told where to put it, or worked it out.
            "configured": bool(os.environ.get(env_var, "").strip()),
        }
        if not path:
            entry["readable"] = True  # the memory backend has no file to open
            entry["writable"] = True
            report.append(entry)
            continue
        try:
            sqlite3.connect(path, timeout=2.0).execute(
                "SELECT count(*) FROM sqlite_master"
            ).fetchone()
            entry["readable"] = True
        except Exception as exc:
            entry["readable"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"
        target = path if os.path.exists(path) else os.path.dirname(path) or "."
        entry["writable"] = os.access(target, os.W_OK)
        entry["bytes"] = os.path.getsize(path) if os.path.exists(path) else 0
        report.append(entry)
    return report


def _limit_key(request: Request) -> str:
    """Who a limiter is pacing: the listener, not the address.

    Keying on `request.client.host` was correct on a laptop and wrong
    everywhere this actually runs. Behind the RunPod public proxy - and
    behind Render's router - every request arrives from the proxy, so the
    whole world shared one bucket: reproduced with two listeners through a
    non-loopback proxy, where the first got 200 and the second got 429
    while `X-Forwarded-For` carried the right addresses and was ignored
    (uvicorn trusts forwarded headers only from 127.0.0.1 by default). At
    RATE_LIMIT_SECONDS=3 that is one episode every three seconds for all
    listeners at once, which is not a limiter, it is an outage.

    Trusting `X-Forwarded-For` would fix the symptom and open a hole: the
    header is client-supplied, so anyone could forge a new one per request
    and never be paced at all - and this limiter guards model spend, which
    `metering.py` exists precisely because it is real.

    The session id is the right key and was already here. It is minted by
    the server, carried in an HttpOnly cookie, and cannot be set by the
    page - the same property that made it the right key for every store.
    So pacing follows the listener across proxies, across Render and
    RunPod, and across a phone changing networks mid-episode.

    The address stays as a fallback for the one case that has no session:
    the middleware mints identity for `/` and `/api/*` but skips
    `/api/health`, and minting can fail. An unpaced endpoint is worse than
    a coarsely paced one, so that case keeps the old behaviour rather than
    keeping no behaviour. The prefixes keep the two namespaces apart, so a
    session id can never collide with an address.
    """
    listener = _listener(request)
    if listener:
        return "listener:" + listener
    return "ip:" + (request.client.host if request.client else "anonymous")


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
    client = _limit_key(request)
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
    # Same key, same reason. This one is worse when it is wrong: the
    # interface fires several cheap reads whenever a tab opens, so a
    # shared 60-per-10s ceiling is spent by a handful of listeners
    # navigating normally.
    client = _limit_key(request)
    now = time.monotonic()
    hits = _read_hits[client]
    cutoff = now - READ_WINDOW_SECONDS
    while hits and hits[0] < cutoff:
        hits.popleft()
    if len(hits) >= settings.read_limit_per_window:
        raise HTTPException(status_code=429, detail="Slow down a moment, then try again.")
    hits.append(now)


def _has_password(user_id: str) -> bool:
    """Whether this account can be logged into with a password.

    The settings screen needs it to decide between "change password" and "set
    one", and `unlink_identity` needs it to know whether dropping a provider
    would lock the account. Read through the store rather than exposing the
    hash anywhere near a response.
    """
    try:
        row = ACCOUNTS._conn().execute(  # noqa: SLF001 - one field, no public reader
            "SELECT password FROM accounts WHERE user_id = ?", (user_id,)).fetchone()
    except Exception:
        log.exception("could not check for a password")
        return False
    return bool(row and row[0])


def _quota_snapshot(user: str, tier_name: str) -> dict:
    """Where this listener stands against every countable resource.

    One shape, so the settings screen, the entitlements endpoint and a refusal
    all describe the allowance the same way. A status read never refuses and
    never raises: an interface unable to say what the limit is, is worse than
    one showing a limit that is briefly stale.
    """
    if not user:
        return {}
    return {resource: QUOTAS.status(user, tier_name, resource).as_dict()
            for resource in entitlements.RESOURCES}


def _tier(request: Request) -> str:
    """Which tier this request is entitled to.

    From the resolved session, never from a parameter - the same rule the
    listener id follows, and for the same reason: a plan is worth money, so a
    client-supplied one is a client-supplied upgrade.
    """
    listener = getattr(request.state, "listener", None)
    return entitlements.normalise(listener.tier if listener else "free")


def _reserve(request: Request, resource: str):
    """Take one from this listener's allowance, or refuse with the reason.

    Returns the granted verdict, which the caller keeps so it can refund. A
    402 would be the pedantic status for "you have run out of allowance", but
    it means "payment required" in a way browsers and SDKs have never agreed
    on; 429 is what a client library already knows to back off from, and the
    `X-FAM-Quota` header carries the whole verdict - the limit, what is left,
    when it resets - so the interface can say what to do next rather than only
    that something was refused.
    """
    user = _listener(request)
    if not user:
        # No session to count against. Not an error - `carry_the_session` logs
        # why - and not a free pass either: `_rate_limit` still applies.
        return None
    try:
        return QUOTAS.reserve(user, _tier(request), resource)
    except quotas.QuotaExceeded as exc:
        raise HTTPException(status_code=429, detail=exc.verdict.message,
                            headers={"X-FAM-Quota": json.dumps(exc.verdict.as_dict())}
                            ) from exc


def _refund(verdict, user: str) -> None:
    """Give the allowance back unconditionally. For the paths where nothing
    could have been spent - the server has no voice, the request never
    started - so there is nothing to weigh."""
    if verdict is not None and user:
        QUOTAS.refund(user, verdict.resource, verdict.window)


def _refund_if_unspent(verdict, user: str, usage: metering.Usage) -> None:
    """On a **failed** request, give the allowance back if nothing was billed.

    Only ever called from an error path. A request that succeeded keeps its
    reservation whatever it cost to serve - in particular **a cache hit is a
    full episode**: the listener heard one and the GPU produced it, and only
    the Claude call was saved. An earlier version refunded whenever no model
    call had been made, which silently made every cached episode free and
    would have made the allowance unenforceable exactly as the cache warmed up.

    Among failures the rule is *was money spent*:

    * A replay whose entry expired between listing and tapping, and a server
      with no voice installed, spent nothing - charging for those would shrink
      an allowance for reasons the listener cannot see.
    * A generation that called Claude and then failed did spend, and refunding
      it would make a broken key the cheapest thing on the server and the most
      expensive thing on the invoice - the same reasoning `_record_usage`
      already applies to the ledger.

    An episode that arrives empty is therefore not refunded, and that is a bug
    to fix rather than a quota to soften.
    """
    if verdict is None or not user:
        return
    if usage.model_calls or usage.exa_searches:
        return
    QUOTAS.refund(user, verdict.resource, verdict.window)


def erase_listener(user_id: str) -> dict:
    """Delete everything FAM holds about one listener, and say what went.

    Required by App Store guideline 5.1.1(v) for any app that lets someone
    create an account, and the shape of it is a decision rather than a loop:

    * **Every per-listener store is emptied** - events, mixes, echoes, the
      profile row, preferences, attachments, quota counters, credentials,
      identities and sessions.
    * **The cost ledger is anonymised, not emptied.** What the GPU and the
      model cost in a given month is a fact about the business; a ledger with
      holes cannot be reconciled against an invoice. The link to the person
      goes and the amount stays (`metering.anonymise`).
    * **The shared script cache is untouched, and needs no decision.** It holds
      no `user_id` at all - it never has - so a script written for this
      listener is already unattributed, and other listeners' Explore feeds do
      not develop holes because somebody left.

    Returns a per-store count so the endpoint reports what it did. Each store
    is attempted independently: a failure in one must not leave the other six
    undeleted, which would be the worst outcome available here - a deletion
    that half happened and reported success.
    """
    removed: dict[str, int] = {}
    for name, store in (("events", EVENTS), ("mixes", MIXES), ("social", SOCIAL),
                        ("preferences", PREFS), ("attachments", ATTACHMENTS),
                        ("quotas", QUOTAS), ("messages", MESSAGES),
                        ("saved", SAVED), ("shares", SHARES)):
        try:
            removed[name] = store.forget(user_id)
        except Exception:
            log.exception("could not erase %s for %r", name, user_id)
            removed[name] = -1
    try:
        removed["usage_rows_anonymised"] = METER.anonymise(user_id)
    except Exception:
        log.exception("could not anonymise usage for %r", user_id)
        removed["usage_rows_anonymised"] = -1
    credentials_gone = ACCOUNTS.delete_account(user_id)
    removed["identities"] = credentials_gone["identities"]
    removed["sessions"] = credentials_gone["sessions"]
    removed["account"] = 1 if credentials_gone["account"] else 0
    return removed


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
    #: Omitted means "let the question decide" - see /api/audio.
    search: bool | None = None


def _build_report() -> dict:
    """Which code this process is actually running.

    "Is my fix deployed?" was unanswerable from outside this server, so it was
    answered by reasoning about what *should* have happened - which is the
    shape PROBLEMS.md 52 is about. Render injects RENDER_GIT_COMMIT and
    RENDER_GIT_BRANCH into every build; FAM_COMMIT covers a host that does
    not, and a checkout that has its .git is asked directly. Unknown says
    unknown rather than guessing.
    """
    commit = (os.environ.get("RENDER_GIT_COMMIT")
              or os.environ.get("FAM_COMMIT") or "").strip()
    branch = (os.environ.get("RENDER_GIT_BRANCH")
              or os.environ.get("FAM_BRANCH") or "").strip()
    source = "environment"
    if not commit:
        try:
            import subprocess

            import pathlib

            root = pathlib.Path(__file__).resolve().parent
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                text=True, timeout=5, check=True).stdout.strip()
            source = "git"
        except Exception:
            source = "unknown"
    return {"commit": commit or "unknown", "short": (commit or "unknown")[:7],
            "branch": branch or "unknown", "source": source}


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        # Which commit is serving this request. Without it, "the fix is
        # pushed" and "the fix is live" are the same sentence from outside.
        "build": _build_report(),
        "mode": "demo" if DEMO_MODE else "live",
        "model": settings.model,
        "web_search_default": settings.enable_web_search,
        "http": {"version": describe_http_version(), "http2_negotiated": http2_enabled()},
        "api_key_configured": bool(credentials.active("ANTHROPIC_API_KEY")
                                    or settings.anthropic_api_key),
        # Configured is not the same as working, and only one of them matters.
        # `credentials.secrets` says where the working one came from and whether
        # the next machine will find it without anybody typing it.
        "credentials": CREDENTIALS,
        "sample_rate": build_engine().sample_rate,
        "min_minutes": settings.min_minutes,
        "max_minutes": settings.max_minutes,
        "tts": engine_report(),
        # Who does the looking on a researched episode, and whether that
        # backend can actually run. `unavailable: true` means researched
        # episodes will fail rather than quietly search another way - worth
        # seeing on a tab rather than discovering in a log.
        "research": research_report(),
        # Which streaming architecture this process is actually running, and
        # whether that was chosen or inherited. A deployment that has been
        # rolled back to `legacy` by hand looks identical to one that has not
        # from the outside, and that is exactly the thing worth being able to
        # ask a running server.
        "streaming_pipeline": settings.streaming_pipeline,
        "streaming_pipeline_default": settings.streaming_pipeline == DEFAULT_PIPELINE,
        # How the listener is told what is happening while they wait. There is
        # no filler any more, so the interface has to be honest instead.
        "search_mode": settings.search_mode,
        # Where that value came from. An env var beats the code default
        # silently and outlives any number of pushes, so "the default was
        # changed" and "this server researches" are different claims and this
        # is the one that settles them.
        "search_mode_source": ("SEARCH_MODE env var"
                               if os.environ.get("SEARCH_MODE", "").strip()
                               else "ENABLE_WEB_SEARCH env var"
                               if os.environ.get("ENABLE_WEB_SEARCH", "").strip()
                               else "config.py default"),
        "research_words": sorted(research_words()),
        "cache": _cache_report(),
        # Every database, its resolved path, and a real read against each.
        "databases": _database_report(),
        "voice_store": VOICE_STORE["dir"],
        # The public API surface, so a client can ask rather than assume.
        "api": {"version": API_VERSION, "prefix": API_PREFIX,
                "cors_origins": _ALLOWED_ORIGINS},
        # Whether Google and Apple sign-in can actually complete on this
        # machine, per provider and with the reason when they cannot.
        # "Configured" is not "works" (PROBLEMS.md §52): a missing PyJWT and an
        # empty audience both make the button fail, and both say so here rather
        # than at the moment somebody presses it.
        "oauth": oauth.report(),
        # Whether tier limits bite, and what they are. A server running with
        # them off looks identical from the outside to one running with them
        # on, and that is exactly the thing worth being able to ask.
        "quotas": {"enforced": quotas.settings_enforcing(),
                   "tiers": entitlements.catalogue()["tiers"]},
    }


class CredentialsRequest(BaseModel):
    """Email **or** phone, plus a password.

    Both optional at the schema level and exactly one required in the handler,
    because "you must send one of these two" is not a thing a field validator
    can say clearly, and a 422 from the framework is a worse message than a
    sentence written for the person reading it.
    """

    email: str = Field("", max_length=accounts_mod.MAX_EMAIL)
    phone: str = Field("", max_length=accounts_mod.MAX_PHONE * 2)
    password: str = Field(..., max_length=accounts_mod.MAX_PASSWORD)
    #: Native clients only. See `_maybe_token` - a browser must never ask for
    #: this, because reading the token in script is precisely what the HttpOnly
    #: cookie exists to prevent.
    want_token: bool = False


class ProviderRequest(BaseModel):
    """A verified identity token from Google or Apple."""

    provider: str = Field(..., max_length=16)
    id_token: str = Field(..., max_length=8192)
    #: The raw nonce the client generated for this sign-in, if it used one.
    #: Sending it is what stops a captured token being replayed; the server
    #: accepts both the raw value and its SHA-256, because Apple is sent the
    #: hash and Google echoes the original.
    nonce: str = Field("", max_length=256)
    want_token: bool = False


class ProfileRequest(BaseModel):
    """Account settings. Every field optional and `None` means "leave it" -
    an empty string means "remove it", which is a different request."""

    display_name: Optional[str] = Field(None, max_length=accounts_mod.MAX_DISPLAY_NAME)
    email: Optional[str] = Field(None, max_length=accounts_mod.MAX_EMAIL)
    phone: Optional[str] = Field(None, max_length=accounts_mod.MAX_PHONE * 2)


class NewPasswordRequest(BaseModel):
    """Setting a first password, for an account created with Google or Apple.
    `PasswordChangeRequest` cannot serve this: there is no current password to
    prove, and asking for one would lock those accounts out of ever having
    one."""

    new: str = Field(..., max_length=accounts_mod.MAX_PASSWORD)


class PasswordChangeRequest(BaseModel):
    current: str = Field(..., max_length=accounts_mod.MAX_PASSWORD)
    new: str = Field(..., max_length=accounts_mod.MAX_PASSWORD)


@app.get("/api/auth/me")
async def auth_me(request: Request) -> dict:
    """Who the server thinks is asking. Cheap, and the only way to find out -
    the id is not in the page's reach, which is the point of the cookie."""
    listener = getattr(request.state, "listener", None)
    if listener is None:
        return {"user_id": "", "email": "", "authenticated": False}
    return listener.as_dict()


def _one_identifier(req: CredentialsRequest) -> str:
    """Which of email or phone this request is using. Exactly one."""
    if bool(req.email) == bool(req.phone):
        raise HTTPException(
            status_code=400,
            detail="Send either an email address or a phone number, not both.")
    return "email" if req.email else "phone"


def _maybe_token(request: Request, token: str, want_token: bool) -> dict:
    """The session token in the response body, but only if it was asked for.

    A native app has to be handed the token: it stores it in the Keychain and
    sends it as `Authorization: Bearer`, because iOS clears its cookie jar
    under conditions the app does not control.

    A browser must never ask. The cookie is set either way, and it is HttpOnly
    precisely so that page script cannot read it - a web client that requests
    the token has voluntarily undone that, and an XSS on that page can then
    take the session rather than merely borrow it. Which is why this is an
    explicit opt-in rather than something every response carries.
    """
    if not want_token:
        return {}
    return {"session_token": token, "expires_in": accounts_mod.SESSION_TTL}


@app.post("/api/auth/signup")
async def auth_signup(req: CredentialsRequest, request: Request) -> dict:
    """Attach an account to the identity this listener already has.

    Not "create a user": they exist already, with a history and possibly mixes
    and echoes. Signing up claims that identity rather than starting a second
    one, which is why nothing has to be migrated.
    """
    _rate_limit(request)
    user = _require_listener(request)
    try:
        if _one_identifier(req) == "email":
            listener = ACCOUNTS.sign_up(user, req.email, req.password)
        else:
            listener = ACCOUNTS.sign_up_phone(user, req.phone, req.password)
    except accounts_mod.AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # A fresh token even though the id has not changed, so that a native
    # client is handed one it can store. The old session stays valid: nothing
    # about signing up should log out the browser tab that did it.
    token = _session_token(request)
    if req.want_token and not token:
        token, _ = ACCOUNTS.new_session(listener.user_id)
        request.state.set_session = token
    return {**listener.as_dict(), **_maybe_token(request, token, req.want_token)}


@app.post("/api/auth/login")
async def auth_login(req: CredentialsRequest, request: Request) -> dict:
    """Verify credentials and move this browser onto that account's identity.

    A fresh session token is minted rather than the current one being
    repointed: reusing it would let a token captured before login keep working
    after it, which is the session-fixation bug.
    """
    _rate_limit(request)
    try:
        if _one_identifier(req) == "email":
            listener = ACCOUNTS.log_in(req.email, req.password)
        else:
            listener = ACCOUNTS.log_in_phone(req.phone, req.password)
    except accounts_mod.AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    old = _session_token(request)
    token, _user_id = ACCOUNTS.new_session(listener.user_id)
    if old:
        # The anonymous session this client was carrying is finished with.
        ACCOUNTS.end_session(old)
    request.state.set_session = token
    return {**listener.as_dict(), **_maybe_token(request, token, req.want_token)}


@app.post("/api/auth/provider")
async def auth_provider(req: ProviderRequest, request: Request) -> dict:
    """Sign in with Google or Sign in with Apple.

    The app gets an identity token from the platform SDK and posts it here;
    `oauth.verify` checks the signature, issuer, audience and nonce against the
    provider's published keys. There is no code exchange and no client secret,
    because a native app needs neither - which removes the most common way this
    is built wrong.

    The two failure modes are deliberately different status codes. 503 means
    *this server* cannot verify tokens for that provider - PyJWT is missing, or
    no audience is configured - and is an operator's problem with an operator's
    message. 401 means the token was checked and refused.
    """
    _rate_limit(request)
    provider = (req.provider or "").strip().lower()
    try:
        verified = oauth.verify(provider, req.id_token, req.nonce)
    except oauth.OAuthUnavailable as exc:
        log.error("provider sign-in unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except oauth.OAuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    try:
        listener, is_new = ACCOUNTS.sign_in_with(
            verified.provider, verified.subject, email=verified.email,
            display_name=verified.name,
            current_user_id=_require_listener(request))
    except accounts_mod.AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    old = _session_token(request)
    token, _user_id = ACCOUNTS.new_session(listener.user_id)
    if old:
        ACCOUNTS.end_session(old)
    request.state.set_session = token
    return {**listener.as_dict(), "is_new": is_new,
            # Said out loud rather than left for the client to work out from
            # the address: an Apple relay address forwards today and can be
            # switched off by its owner tomorrow, so nothing should promise to
            # reach somebody there.
            "private_relay": verified.is_private_relay,
            **_maybe_token(request, token, req.want_token)}


@app.post("/api/auth/logout")
async def auth_logout(request: Request) -> dict:
    """Drop the session. The next request mints a fresh anonymous one, so the
    app keeps working - as a different listener, with nothing of theirs."""
    _read_limit(request)
    ACCOUNTS.end_session(_session_token(request))
    request.state.set_session = ""
    return {"ok": True}


@app.post("/api/auth/password")
async def auth_password(req: PasswordChangeRequest, request: Request) -> dict:
    """Change it, and log every device out - including this one. A password
    change that leaves old sessions alive does not do what people believe."""
    _rate_limit(request)
    try:
        ACCOUNTS.change_password(_require_listener(request), req.current, req.new)
    except accounts_mod.AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    request.state.set_session = ""
    return {"ok": True}


@app.post("/api/auth/password/set")
async def auth_password_set(req: NewPasswordRequest, request: Request) -> dict:
    """Add a first password to an account that signed up with Google or Apple.

    Its own endpoint rather than a branch inside the change-password one,
    because the two have different preconditions: that one proves the current
    password, and this one is reachable exactly when there is none to prove.
    Merging them would mean a request that omits `current` is sometimes a
    legitimate first set and sometimes an attempt to skip the check.
    """
    _rate_limit(request)
    try:
        ACCOUNTS.set_password(_require_account(request), req.new)
    except accounts_mod.AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}


@app.get("/api/account")
async def account_read(request: Request) -> dict:
    """Everything the settings screen shows: who you are, how you get in,
    what tier you are on, and where else you are signed in."""
    _read_limit(request)
    user = _require_account(request)
    account = ACCOUNTS.account(user) or {}
    tier_name = entitlements.normalise(account.get("plan", "free"))
    return {
        "user_id": user,
        "email": account.get("email", ""),
        "phone": account.get("phone", ""),
        "display_name": account.get("display_name", ""),
        "created": account.get("created", 0),
        "identities": ACCOUNTS.identities_for(user),
        "has_password": bool(ACCOUNTS.account(user) and _has_password(user)),
        "sessions": ACCOUNTS.sessions_for(user, _session_token(request)),
        "entitlements": entitlements.describe(tier_name),
        # Said here as well as on /api/entitlements because a settings screen
        # that shows a plan without showing what is left of it invites the
        # question it cannot answer.
        "usage": _quota_snapshot(user, tier_name),
    }


@app.post("/api/account")
async def account_update(req: ProfileRequest, request: Request) -> dict:
    """Change the account's own details. Not preferences - those are
    `/api/preferences`, they are not credentials, and they work without an
    account at all."""
    _rate_limit(request)
    try:
        account = ACCOUNTS.update_profile(
            _require_account(request), display_name=req.display_name,
            email=req.email, phone=req.phone)
    except accounts_mod.AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return account


@app.delete("/api/account")
async def account_delete(request: Request) -> dict:
    """Delete the account and everything FAM holds about this listener.

    Required of any app that offers account creation (App Store guideline
    5.1.1(v)), and it has to be reachable *in the app* rather than through a
    support address - which is why it is an endpoint and not a mailbox.

    `erase_listener` says exactly what is removed, what is anonymised and what
    is deliberately untouched. The session is dropped afterwards, so the next
    request mints a fresh anonymous listener and the app keeps working.
    """
    _rate_limit(request)
    user = _require_account(request)
    removed = erase_listener(user)
    request.state.set_session = ""
    log.info("erased listener %r: %s", user, removed)
    return {"ok": True, "removed": removed}


@app.post("/api/account/signout-everywhere")
async def account_signout_everywhere(request: Request) -> dict:
    """Drop every other session, keeping this one.

    The thing somebody reaches for when they think a device is lost, and it is
    the reason `sessions_for` reports a count without reporting tokens.
    """
    _rate_limit(request)
    ended = ACCOUNTS.end_other_sessions(_require_account(request),
                                        _session_token(request))
    return {"ok": True, "ended": ended}


@app.delete("/api/account/identity")
async def account_unlink(request: Request,
                         provider: str = Query(..., max_length=16)) -> dict:
    """Remove one sign-in route, unless it is the only way in."""
    _rate_limit(request)
    try:
        removed = ACCOUNTS.unlink_identity(_require_account(request),
                                           provider.strip().lower())
    except accounts_mod.AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": bool(removed), "removed": removed}


class FollowRequest(BaseModel):
    """Who to follow. By handle from a search, or by the id a search returned -
    never by a listener id the client made up, which is why both come from
    somewhere the server produced."""

    handle: str = Field("", max_length=social_mod.MAX_HANDLE + 1)
    user_id: str = Field("", max_length=64)


class SendMessageRequest(BaseModel):
    to: str = Field(..., max_length=64)
    text: str = Field("", max_length=messages_mod.MAX_TEXT)
    #: An episode share carries the question and the length, which is the
    #: script cache's key - so the recipient's play is a cache hit and the
    #: share cost one row.
    query: str = Field("", max_length=messages_mod.MAX_QUERY)
    minutes: int = Field(0, ge=0, le=60)
    title: str = Field("", max_length=messages_mod.MAX_TITLE)


class SaveRequest(BaseModel):
    query: str = Field(..., max_length=saved_mod.MAX_QUERY)
    minutes: int = Field(3, ge=0, le=60)
    title: str = Field("", max_length=saved_mod.MAX_TITLE)
    source: str = Field("", max_length=40)
    folder_id: str = Field("", max_length=64)


class FolderRequest(BaseModel):
    name: str = Field(..., max_length=saved_mod.MAX_NAME)


class MoveRequest(BaseModel):
    folder_id: str = Field("", max_length=64)


class ConfirmDownloadRequest(BaseModel):
    #: What the device actually stored. The server's own figure is a
    #: deliberately generous estimate; this is the truth from the only place
    #: that knows it.
    bytes: int = Field(0, ge=0)


class ShareRequest(BaseModel):
    query: str = Field(..., max_length=sharing.MAX_QUERY)
    minutes: int = Field(3, ge=0, le=60)
    title: str = Field("", max_length=sharing.MAX_TITLE)


# --- friends --------------------------------------------------------------

@app.get("/api/friends")
async def friends_read(request: Request) -> dict:
    """Who this listener follows, who follows them, and who does both.

    Mutuals are derived rather than stored, so there is no request-and-accept
    state machine and no way for the two directions to disagree.
    """
    _read_limit(request)
    user = _require_account(request)
    return {
        "following": SOCIAL.following(user),
        "followers": SOCIAL.followers(user),
        "friends": SOCIAL.friends(user),
        "counts": SOCIAL.follow_counts(user),
    }


@app.get("/api/people")
async def people_search(request: Request,
                        q: str = Query("", max_length=64)) -> dict:
    """Find somebody by handle or name, to follow or share with.

    Only people who have chosen a handle are findable. Someone who has never
    set one is not hidden from a directory - they are not in one, which is the
    difference between a private setting and a feature nobody enabled.
    """
    _read_limit(request)
    user = _require_account(request)
    found = SOCIAL.find_people(q, exclude_user=user)
    following = {p["user_id"] for p in SOCIAL.following(user)}
    for person in found:
        person["following"] = person["user_id"] in following
    return {"people": found}


@app.post("/api/friends/follow")
async def friends_follow(req: FollowRequest, request: Request) -> dict:
    _rate_limit(request)
    user = _require_account(request)
    target = req.user_id
    if not target and req.handle:
        found = SOCIAL.find_people(req.handle, exclude_user=user, limit=5)
        exact = [p for p in found
                 if p["handle"] == req.handle.strip().lstrip("@").lower()]
        target = exact[0]["user_id"] if exact else ""
    if not target:
        raise HTTPException(status_code=404, detail="No listener by that handle.")
    try:
        changed = SOCIAL.follow(user, target)
    except social_mod.SocialError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "changed": changed,
            "counts": SOCIAL.follow_counts(user)}


@app.delete("/api/friends/follow")
async def friends_unfollow(request: Request,
                           user_id: str = Query(..., max_length=64)) -> dict:
    _rate_limit(request)
    user = _require_account(request)
    return {"ok": SOCIAL.unfollow(user, user_id),
            "counts": SOCIAL.follow_counts(user)}


# --- messages -------------------------------------------------------------

def _decorate(people: list[dict]) -> dict:
    """user_id -> what to draw. One lookup for a whole inbox rather than one
    per row."""
    out = {}
    for person in people:
        out[person["user_id"]] = {"name": person.get("name") or "",
                                  "handle": person.get("handle") or ""}
    return out


@app.get("/api/messages")
async def messages_inbox(request: Request) -> dict:
    """Every conversation, most recent first, with an unread count each."""
    _read_limit(request)
    user = _require_account(request)
    inbox = MESSAGES.inbox(user)
    known = _decorate(SOCIAL.following(user) + SOCIAL.followers(user))
    for row in inbox:
        person = known.get(row["with"]) or SOCIAL.person(row["with"])
        row["name"] = person.get("name") or "Someone"
        row["handle"] = person.get("handle") or ""
    return {"threads": inbox, "unread": MESSAGES.unread_total(user)}


@app.get("/api/messages/thread")
async def messages_thread(request: Request,
                          with_: str = Query(..., alias="with", max_length=64)) -> dict:
    """One conversation, and reading it marks it read.

    Marking on read rather than on a separate call, because the two would drift
    the moment a client crashed between them - and a thread that stays unread
    after somebody has read it is the more annoying direction.
    """
    _read_limit(request)
    user = _require_account(request)
    try:
        thread = MESSAGES.thread(user, with_)
    except messages_mod.MessageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    MESSAGES.mark_read(user, with_)
    person = SOCIAL.person(with_)
    return {"with": {"user_id": with_, "name": person.get("name") or "Someone",
                     "handle": person.get("handle") or ""},
            "messages": [m.as_dict(user) for m in thread]}


@app.post("/api/messages")
async def messages_send(req: SendMessageRequest, request: Request) -> dict:
    """Send a message, or share an episode into a conversation.

    Paced by `_read_limit` rather than `_rate_limit`: this provably makes no
    model call - it writes one row pointing at a question - and pacing it at
    one every three seconds would make a conversation unusable. The generation
    happens when the recipient taps, against their own allowance.
    """
    _read_limit(request)
    user = _require_account(request)
    kind = "episode" if req.query else "text"
    try:
        message = MESSAGES.send(user, req.to, kind=kind, text=req.text,
                                query=req.query, minutes=req.minutes,
                                title=req.title)
    except messages_mod.MessageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # A share is a signal about taste as much as an act of sending, and the
    # feed already learns from plays. Recorded for the sender only: what the
    # recipient thinks of it is not known until they press play.
    if kind == "episode":
        EVENTS.record(topics_mod.Event(
            user, "share", "", req.query, topics_mod.tags_for_text(req.query)))
    return {"ok": True, "message": message.as_dict(user)}


# --- save for later -------------------------------------------------------

@app.get("/api/saved")
async def saved_read(request: Request,
                     folder_id: Optional[str] = Query(None, max_length=64),
                     downloaded: bool = Query(False)) -> dict:
    """The shelf: folders, what is on it, and how much offline room is left."""
    _read_limit(request)
    user = _require_account(request)
    tier_name = _tier(request)
    return {
        "folders": SAVED.folders(user),
        "items": [i.as_dict() for i in SAVED.items(user, folder_id, downloaded)],
        "downloads": SAVED.download_status(user, tier_name),
    }


@app.post("/api/saved")
async def saved_save(req: SaveRequest, request: Request) -> dict:
    """Save an episode for later.

    The response carries the download status because the interface asks about
    downloading the moment something is saved - and asking a question whose
    answer is "you have no room" would be a worse popup than not asking.
    """
    _read_limit(request)
    user = _require_account(request)
    tier_name = _tier(request)
    try:
        item = SAVED.save(user, req.query, req.minutes, title=req.title,
                          source=req.source, folder_id=req.folder_id)
    except saved_mod.SavedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "item": item.as_dict(),
            "downloads": SAVED.download_status(user, tier_name)}


@app.delete("/api/saved/{item_id}")
async def saved_remove(item_id: str, request: Request) -> dict:
    _read_limit(request)
    return {"ok": SAVED.remove(_require_account(request), item_id)}


@app.post("/api/saved/{item_id}/played")
async def saved_played(item_id: str, request: Request) -> dict:
    """Note that a saved episode was played.

    Feeds the "what to clear" list, which offers the ones nobody has been back
    to rather than the oldest - the episode somebody saved first is often the
    one they are keeping on purpose. Recorded here rather than inferred from
    the event log because a download plays with the network off, so the only
    honest moment to record it is the next time the client is online.
    """
    _read_limit(request)
    SAVED.played(_require_account(request), item_id)
    return {"ok": True}


@app.post("/api/saved/{item_id}/move")
async def saved_move(item_id: str, req: MoveRequest, request: Request) -> dict:
    _read_limit(request)
    try:
        item = SAVED.move(_require_account(request), item_id, req.folder_id)
    except saved_mod.SavedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if item is None:
        raise HTTPException(status_code=404, detail="No such saved episode.")
    return {"ok": True, "item": item.as_dict()}


@app.post("/api/saved/folders")
async def saved_folder_create(req: FolderRequest, request: Request) -> dict:
    _read_limit(request)
    try:
        return {"ok": True,
                "folder": SAVED.create_folder(_require_account(request), req.name)}
    except saved_mod.SavedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/saved/folders/{folder_id}")
async def saved_folder_rename(folder_id: str, req: FolderRequest,
                              request: Request) -> dict:
    _read_limit(request)
    try:
        return {"ok": True, "folder": SAVED.rename_folder(
            _require_account(request), folder_id, req.name)}
    except saved_mod.SavedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/saved/folders/{folder_id}")
async def saved_folder_delete(folder_id: str, request: Request) -> dict:
    """Remove a folder. Its episodes are unfiled, never deleted - see
    `SavedStore.delete_folder` for why that is the only safe direction."""
    _read_limit(request)
    return {"ok": True,
            "unfiled": SAVED.delete_folder(_require_account(request), folder_id)}


# --- downloads ------------------------------------------------------------

@app.post("/api/saved/{item_id}/download")
async def saved_download(item_id: str, request: Request) -> dict:
    """Take a slot on the offline shelf.

    The server records the claim and the device holds the bytes - there is no
    file here to hand over, because the settled constraint is that nothing
    writes one. The client downloads by streaming `/api/audio` exactly as it
    would to play it, and keeps what arrives.

    A full shelf is a 409 rather than a 429: this is not a rate, it is a
    capacity, and the body names what to clear because a limit without a
    remedy is a dead end on a phone.
    """
    _read_limit(request)
    user = _require_account(request)
    try:
        item = SAVED.reserve_download(item_id=item_id, user_id=user,
                                      tier_name=_tier(request))
    except saved_mod.DownloadLimit as exc:
        raise HTTPException(status_code=409, detail=str(exc), headers={
            "X-FAM-Downloads": json.dumps(
                {"candidates": exc.candidates,
                 "status": SAVED.download_status(user, _tier(request))})
        }) from exc
    except saved_mod.SavedError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True, "item": item.as_dict(),
            "downloads": SAVED.download_status(user, _tier(request)),
            # What the client streams to fill the slot. Named here so the
            # download path and the play path cannot drift apart.
            "stream": f"/api/audio?q={quote(item.query)}&minutes={item.minutes}&fmt=pcm"}


@app.post("/api/saved/{item_id}/download/confirm")
async def saved_download_confirm(item_id: str, req: ConfirmDownloadRequest,
                                 request: Request) -> dict:
    _read_limit(request)
    item = SAVED.confirm_download(_require_account(request), item_id, req.bytes)
    if item is None:
        raise HTTPException(status_code=404, detail="No such saved episode.")
    return {"ok": True, "item": item.as_dict()}


@app.delete("/api/saved/{item_id}/download")
async def saved_download_release(item_id: str, request: Request) -> dict:
    """Give the slot back, keeping the episode saved.

    Also how a client re-syncs after its storage was evicted: release what it
    no longer holds. "I need the space" and "I am not interested" are different
    requests, and merging them loses somebody's list while they tidy their
    phone.
    """
    _read_limit(request)
    user = _require_account(request)
    released = SAVED.release_download(user, item_id)
    return {"ok": released,
            "downloads": SAVED.download_status(user, _tier(request))}


# --- sharing outside FAM --------------------------------------------------

def _share_url(share_id: str) -> tuple[str, bool]:
    """The link, and whether it names a host anybody else can reach.

    Both, because a relative link is still useful inside the app and is a
    broken promise on LinkedIn. The caller decides what to do with that; what
    it must not do is invent `localhost`.
    """
    base = settings.public_base_url
    return (f"{base}/s/{share_id}" if base else f"/s/{share_id}"), bool(base)


@app.get("/api/share/targets")
async def share_targets(request: Request) -> dict:
    """Where an episode can be sent, and what each destination can carry.

    `needs_image` is the one that changes what the client does: a story is a
    picture with a link attached, not a sentence with a URL in it, so those
    destinations take the card from `/api/share/card` instead of the text.
    """
    _read_limit(request)
    return {"targets": [
        {"key": t.key, "label": t.label, "kind": t.kind,
         "needs_image": t.needs_image, "max_chars": t.max_chars}
        for t in sharing.TARGETS]}


@app.post("/api/share")
async def share_create(req: ShareRequest, request: Request) -> dict:
    """Make a share link and the words to send with it, per destination.

    Deliberately reachable without an account: a share link is the cheapest
    route FAM has to a listener who does not have it yet, and putting a sign-up
    in front of the act of recommending it would be a strange way to grow.
    Everything *kept* still needs an account - which is the settled boundary,
    and a share is an outbound act rather than a shelf.

    Nothing is posted anywhere. FAM holds no token for any of these platforms
    and asks for none; the phone's share sheet and the platforms' own apps do
    the posting, with the person looking at it.
    """
    _read_limit(request)
    user = _listener(request)
    try:
        share = SHARES.create(user, req.query, req.minutes, req.title)
    except sharing.ShareError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    url, public = _share_url(share["id"])
    rendered = {t.key: sharing.render(
        t.key, title=share["title"], question=share["query"],
        minutes=share["minutes"], url=url) for t in sharing.TARGETS}
    return {
        "share": share, "url": url,
        # Said plainly rather than left to be discovered: without
        # PUBLIC_BASE_URL this link works inside the app and nowhere else.
        "public": public,
        "card": f"/api/share/card?share={share['id']}",
        "targets": rendered,
    }


@app.get("/api/share/card")
async def share_card(request: Request,
                     share: str = Query(..., max_length=64)) -> Response:
    """The story image, as SVG.

    Instagram and Snapchat stories cannot carry a link as text - they are
    pictures with a sticker on them - so without this the listener shares a
    screenshot of a player UI, which is not an invitation to anything.
    """
    _read_limit(request)
    record = SHARES.get(share)
    if not record:
        raise HTTPException(status_code=404, detail="No such share.")
    person = SOCIAL.person(record["user_id"]) if record["user_id"] else {}
    svg = sharing.story_card(record["title"], record["query"],
                             record["minutes"], person.get("handle") or "")
    return Response(content=svg, media_type="image/svg+xml",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.get("/s/{share_id}")
async def share_open(share_id: str, request: Request):
    """Where a shared link lands.

    A redirect into the app with the question and length in the query string,
    counting the open on the way past. The count is the only number sharing
    produces and it is the one that says whether any of this does anything.
    """
    record = SHARES.get(share_id)
    if not record:
        return RedirectResponse(url="/", status_code=302)
    SHARES.opened(share_id)
    target = (f"/?q={quote(record['query'])}&minutes={record['minutes']}"
              f"&from=share")
    return RedirectResponse(url=target, status_code=302)


@app.get("/api/entitlements")
async def entitlements_read(request: Request) -> dict:
    """What this listener may do, and how much of it is left.

    Works without an account, because an anonymous listener is on a tier too
    and needs to be told what it allows - a limit nobody can see coming is
    indistinguishable from a bug when it arrives.
    """
    _read_limit(request)
    user = _listener(request)
    tier_name = _tier(request)
    return {
        "enforced": quotas.settings_enforcing(),
        **entitlements.describe(tier_name),
        "usage": _quota_snapshot(user, tier_name),
    }


@app.get("/api/plans")
async def plans_read(request: Request) -> dict:
    """Every tier and every feature, for a pricing screen.

    Deliberately has no prices in it. A number here and a number in App Store
    Connect are two places for one fact, and the one that is wrong is always
    the one the listener is reading - so the price comes from the store's own
    product metadata, which is also the only place it can be right per country.
    """
    _read_limit(request)
    return {**entitlements.catalogue(), "current": _tier(request)}


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
    # A script is a Claude call, which is the expensive half of an episode.
    # Counted against the same allowance rather than a second one: from the
    # allowance's point of view this *is* an episode, minus the audio.
    # Kept only so the shape matches /api/audio: there is no failure path here
    # between the reservation and the response, so nothing is ever refunded.
    _reserve(request, "episode")
    plan = _validated_plan(req.query, req.minutes, "", req.search)
    generator = DemoGenerator() if DEMO_MODE else ScriptGenerator()
    notes = ScriptNotes()
    text = " ".join([s async for s in generator.stream_sentences(plan, notes)])
    # No audio, but a full script call - the same money as an episode, minus
    # the synthesis. Left out, a tool or a probe hammering this endpoint would
    # be the one kind of spend the ledger could not see.
    _record_usage(_listener(request), notes.usage, surface="script",
                  minutes=plan.minutes)
    return {
        "query": plan.query,
        "minutes": plan.minutes,
        "word_budget": plan.word_budget,
        "words": len(text.split()),
        "script": text,
        "thread": notes.thread,
    }


# Each store resolves its own path (env var, else the project root), so the
# mapping from variable to file lives in one place per store rather than
# being restated here.
# Browser origins allowed to call this server. Off unless configured, and
# deliberately not a wildcard: these requests carry the session cookie, and a
# browser refuses `*` together with credentials - so a wildcard here would look
# permissive, not work, and hide the real fix behind a setting that appeared to
# be already correct.
#
# A native app is not a browser. It sends no Origin header and is not subject
# to the same-origin policy at all, so the iOS client needs nothing here; this
# exists only for a web client served from somewhere other than this server.
_ALLOWED_ORIGINS = [o.strip() for o in settings.api_origins.split(",") if o.strip()]
if _ALLOWED_ORIGINS:
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
        # So a browser client can read the quota verdict on a 429 rather than
        # only the status code.
        expose_headers=["X-FAM-Quota", "X-Sample-Rate", "X-Requested-Seconds"],
    )
    log.info("CORS enabled for %s", ", ".join(_ALLOWED_ORIGINS))


EVENTS = topics_mod.EventStore()
MIXES = mixes_mod.MixStore()
SOCIAL = social_mod.SocialStore()
ACCOUNTS = accounts_mod.AccountStore()
PREFS = prefs_mod.PreferenceStore()
METER = metering.MeterStore()
QUOTAS = quotas.QuotaStore()
MESSAGES = messages_mod.MessageStore()
SAVED = saved_mod.SavedStore()
SHARES = sharing.ShareStore()


@app.middleware("http")
async def carry_the_session(request: Request, call_next):
    """Resolve who is asking, from a cookie the client cannot forge.

    This is where the old hole is closed. The listener id used to arrive as
    `?user=` - chosen by the browser, checked by nobody - so anyone who guessed
    one could act as that listener. It is now read from an HttpOnly session
    cookie and, when there is no cookie, minted here with `secrets`.

    Minting rather than demanding a login is the point: an anonymous listener
    still gets a real, unforgeable identity, so search, myFAM and Go Deeper
    work exactly as before for someone who has never signed up. Signing up
    later attaches an email to the id they already have, which is why no data
    has to move.

    Only paths that need identity mint one, so a monitoring poll on
    /api/health does not accumulate a session row per request.
    """
    path = request.url.path
    wants_identity = path == "/" or (
        path.startswith("/api/") and path != "/api/health"
    )
    token = _session_token(request)
    listener = ACCOUNTS.listener_for(token) if token else None
    minted = ""
    if listener is None and wants_identity:
        try:
            minted, user_id = ACCOUNTS.new_session()
            listener = accounts_mod.Listener(user_id)
        except Exception:
            # Never fail a request because a session could not be written; the
            # listener is simply anonymous-and-unrecorded for this one.
            log.exception("could not mint a session; continuing without one")
    request.state.listener = listener

    response = await call_next(request)

    # An endpoint that changes who you are (log in, log out, sign up) says so
    # here rather than building its own response.
    new_token = getattr(request.state, "set_session", None)
    if new_token is not None:
        if new_token:
            _set_session_cookie(response, request, new_token)
        else:
            response.delete_cookie(accounts_mod.COOKIE_NAME, path="/")
    elif minted:
        _set_session_cookie(response, request, minted)
    return response


#: The public API's version. Every endpoint is reachable at `/api/v1/...` as
#: well as at `/api/...`, and the prefixed form is the one a shipped app must
#: use. The reason is the whole reason a version exists: an app on somebody's
#: phone cannot be redeployed with the server, so the day an endpoint has to
#: change shape, `/api/v2` can carry the new one while `/api/v1` keeps the
#: promise made to every phone already out there. Without a prefix that day
#: forces a choice between breaking installed apps and never changing the API.
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"


@app.middleware("http")
async def version_prefix(request: Request, call_next):
    """Serve `/api/v1/x` from the same handler as `/api/x`.

    A rewrite rather than a second set of routes: two registrations of one
    endpoint is two places for a decorator to drift, and the failure would be
    a native client quietly getting different behaviour from the web one.

    Declared after `carry_the_session` so it wraps it and therefore runs
    *first* - the session middleware decides what to do from the path, and it
    has to see the real one.
    """
    path = request.scope.get("path", "")
    if path.startswith(API_PREFIX + "/") or path == API_PREFIX:
        request.scope["path"] = "/api" + path[len(API_PREFIX):]
    return await call_next(request)


def _session_token(request: Request) -> str:
    """The session token, from the cookie or from an Authorization header.

    Two carriers, one session model. The web client uses the HttpOnly cookie
    and cannot read it, which is what makes an XSS unable to walk off with
    somebody's identity. A native app has no cookie jar worth relying on -
    iOS clears `HTTPCookieStorage` under conditions the app does not control,
    and "the listener silently became a different listener" is the worst
    failure available to a product whose personalisation is an append-only log
    keyed on that id - so it holds the same token in the Keychain and sends it
    as `Authorization: Bearer`.

    The settled rule is untouched: **the id still never comes from the
    client.** A bearer token is the same server-minted, high-entropy,
    revocable, never-stored-in-the-clear string the cookie carries. What
    changes is the envelope, not the trust.

    The cookie wins when both are present. A browser attaches its cookie
    automatically, so a header alongside it is either a mistake or somebody
    testing whether one overrides the other; the answer is no.
    """
    cookie = request.cookies.get(accounts_mod.COOKIE_NAME, "")
    if cookie:
        return cookie
    header = request.headers.get("authorization", "")
    scheme, _, value = header.partition(" ")
    if scheme.lower() == "bearer" and value.strip():
        return value.strip()
    return ""


def _set_session_cookie(response, request: Request, token: str) -> None:
    """HttpOnly so page scripts cannot read it, which is what makes an XSS
    unable to walk off with someone's identity. Secure only over https, or the
    cookie would be dropped on the http://<lan-ip> address a phone uses."""
    response.set_cookie(
        accounts_mod.COOKIE_NAME,
        token,
        max_age=accounts_mod.SESSION_TTL,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
        path="/",
    )


def _surface(cached_only: bool, topic_id: str, context: str) -> str:
    """Which part of the app spent this money.

    Derived rather than passed, because the interface already says it in the
    parameters it sends. It matters for pricing: Explore replays and never
    writes a script, so an Explore-heavy listener costs a fraction of a
    search-heavy one, and a single blended per-listener number hides that.
    """
    if cached_only:
        return "explore"
    if topic_id:
        return "myfam"
    if context:
        return "godeeper"
    return "search"


def _record_usage(user: str, usage: metering.Usage, *, surface: str,
                  minutes: int = 0, audio_seconds: float = 0.0,
                  cache_hit: bool = False) -> None:
    """Append this episode to the ledger. Never fails the request.

    Called after the listener already has their audio. A metering failure that
    became a failed episode would trade a gap in the billing record - which is
    recoverable, and which `MeterStore.record` logs loudly - for a broken
    product, which is not.
    """
    usage.audio_seconds = audio_seconds
    usage.cache_hit = cache_hit
    try:
        METER.record(user, usage, plan=ACCOUNTS.plan_for(user),
                     surface=surface, minutes=minutes)
    except Exception:  # noqa: BLE001 - the episode already played
        log.exception("could not meter an episode for %r", user)


def _listener(request: Request) -> str:
    """The id every store keys on, or "" when there is no session."""
    listener = getattr(request.state, "listener", None)
    return listener.user_id if listener else ""


def _require_listener(request: Request) -> str:
    user = _listener(request)
    if not user:
        raise HTTPException(status_code=503, detail="Could not start a session.")
    return user


#: What "Skip for now" costs, in one place. Playback is never gated: search,
#: myFAM, DailyFAM's episodes, Explore and Go Deeper all work with no account,
#: because a login in front of the first word breaks the one-sentence spec and
#: that mistake has already been avoided once here (see accounts.py).
#:
#: What *is* gated is everything the server keeps for you long-term - saved
#: mixes, chosen interests and language, the weekly recap - on the product
#: decision that durable per-listener storage is what an account is for.
#:
#: The interaction log is deliberately NOT in that set. It is ambient
#: personalisation rather than a thing the listener made and can point at, and
#: gating it would mean an anonymous listener's feed could never be ranked -
#: which is the product, not an account perk.
ACCOUNT_REQUIRED = ("You need an account for this. Signing up keeps the "
                    "listening you have already done — it does not start you over.")


def _require_account(request: Request) -> str:
    """The listener id, but only if credentials are attached to it.

    401 rather than 403: the listener genuinely has an identity, it just has
    nothing proving it is theirs on another device. The interface reads the
    status and offers signup rather than printing a failure.
    """
    listener = getattr(request.state, "listener", None)
    if listener is None or not listener.user_id:
        raise HTTPException(status_code=503, detail="Could not start a session.")
    if not listener.is_authenticated:
        raise HTTPException(status_code=401, detail=ACCOUNT_REQUIRED)
    return listener.user_id


class MixRequest(BaseModel):
    # No `user` field: identity comes from the session cookie, never the body.
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
    ATTACHMENTS.put(_listener(request), item)
    return item.as_dict()


@app.delete("/api/attach")
async def detach(request: Request, id: str = Query(..., max_length=64)) -> dict:
    _read_limit(request)
    return {"ok": ATTACHMENTS.delete(_listener(request), id)}


@app.get("/api/topics")
async def bank(request: Request):
    """The whole shared bank, for the mix topic picker."""
    _read_limit(request)
    return {"topics": [t.as_dict() for t in topics_mod.TOPIC_BANK]}


@app.get("/api/mixes")
async def list_mixes(request: Request):
    """This listener's DailyFAM mixes, each with its topics resolved.

    Account-gated along with the rest of /api/mixes: a mix is a thing the
    listener made and expects to find again, which is the definition this app
    uses for "needs an account". See ACCOUNT_REQUIRED.
    """
    _read_limit(request)
    user = _require_account(request)
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
        mix = MIXES.create(_require_account(request), req.name or "", req.topic_ids or [])
    except mixes_mod.MixError as exc:
        # Phrased for the listener: these are things they did, not faults.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return mix.as_dict()


@app.patch("/api/mixes/{mix_id}")
async def update_mix(mix_id: str, req: MixRequest, request: Request):
    _read_limit(request)
    account = _require_account(request)
    try:
        mix = MIXES.update(account, mix_id, req.name, req.topic_ids, req.public)
    except mixes_mod.MixError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return mix.as_dict()


@app.delete("/api/mixes/{mix_id}")
async def delete_mix(mix_id: str, request: Request):
    _read_limit(request)
    if not MIXES.delete(_require_account(request), mix_id):
        raise HTTPException(status_code=404, detail="That mix no longer exists.")
    return {"ok": True}


class PreferenceRequest(BaseModel):
    """Every field optional: the intro saves one page at a time, and the recap
    popup writes one flag from a screen that knows nothing about the rest."""

    # No `user` field, for the same reason MixRequest has none.
    interests: Optional[list[str]] = None
    language: Optional[str] = Field(None, max_length=8)
    weekly_recap: Optional[bool] = None
    intro_done: Optional[bool] = None


def _interests_for(request: Request, given: str = "") -> tuple[str, ...]:
    """Chosen facets for ranking: stored ones for an account, the query string
    for anyone else.

    Taking these off the request for an anonymous listener is not what "a
    listener id is never accepted from the client" forbids. An id is an
    identity and grants access to somebody's data; this is a ranking hint,
    validated against a fixed eight-word vocabulary, used for one response and
    never written down. An anonymous listener's intro answers live in their own
    browser and nowhere else, so this is the only route by which the ranker can
    honour them at all - and honouring them is the entire reason the intro asks.
    """
    listener = getattr(request.state, "listener", None)
    if listener is not None and listener.is_authenticated:
        return PREFS.get(listener.user_id).interests
    try:
        return prefs_mod.clean_interests(given.split(","))
    except prefs_mod.PreferenceError:
        # A malformed hint costs one less-personal feed. It must never be what
        # stops the page loading.
        return ()


@app.get("/api/preferences")
async def read_preferences(request: Request):
    """What is on offer, and what this listener chose.

    The *choices* are public - the intro is shown before anyone has an account,
    and a picker that cannot list its own options is no picker. What was chosen
    comes back only for an account, and `saved` says which of the two the
    caller is looking at, so the interface can tell the listener the truth
    about whether their answers are being kept.
    """
    _read_limit(request)
    listener = getattr(request.state, "listener", None)
    authed = bool(listener is not None and listener.is_authenticated)
    stored = (PREFS.get(listener.user_id) if authed
              else prefs_mod.Preferences(_listener(request)))
    body = {
        "interests_available": [{"id": tag, "label": label}
                                for tag, label in topics_mod.TAG_LABELS.items()],
        "languages": [dict(lang) for lang in prefs_mod.LANGUAGES],
        "max_interests": prefs_mod.MAX_INTERESTS,
        # False until per-language generation exists. Printed under the picker
        # rather than left implicit: a setting that silently changes nothing is
        # the failure mode this project has paid for most often.
        "language_active": prefs_mod.LANGUAGE_ACTIVE,
        "account": authed,
        "saved": authed,
        "account_required": ACCOUNT_REQUIRED,
    }
    body.update(stored.as_dict())
    return body


@app.post("/api/preferences")
async def write_preferences(req: PreferenceRequest, request: Request):
    """Store the intro's answers. Account only - see ACCOUNT_REQUIRED."""
    _read_limit(request)
    user = _require_account(request)
    try:
        prefs = PREFS.save(user, interests=req.interests, language=req.language,
                           weekly_recap=req.weekly_recap, intro_done=req.intro_done)
    except prefs_mod.PreferenceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return prefs.as_dict()


@app.get("/api/recap")
async def recap(request: Request):
    """This listener's week, and whether they are still owed this one.

    `due` is what decides the popup, and it is answered here rather than in the
    browser because the rule is "the first open on or after Sunday" - a
    question about a stored date, not about this session.
    """
    _read_limit(request)
    user = _require_account(request)
    body = topics_mod.weekly_recap(EVENTS, user)
    prefs = PREFS.get(user)
    body["due"] = PREFS.recap_due(user)
    body["enabled"] = prefs.weekly_recap
    return body


@app.post("/api/recap/seen")
async def recap_seen(request: Request):
    """Mark this week's recap shown, so it does not appear again until Sunday."""
    _read_limit(request)
    PREFS.mark_recap_seen(_require_account(request))
    return {"ok": True}


@app.get("/api/nextup")
async def next_up(
    request: Request,
    topic_id: str = Query("", max_length=64),
    q: str = Query("", max_length=300, description="What the finished episode asked"),
    interests: str = Query("", max_length=200),
):
    """The four tiles the post-episode popup offers.

    Costs no model call - it ranks the same fixed bank myFAM does, seeded with
    what just finished. See topics.rank_next_up for why this is the feed's
    ranker rather than a second one.
    """
    _read_limit(request)
    user = _listener(request)
    picks = topics_mod.rank_next_up(
        EVENTS, user, after_id=topic_id, after_text=q,
        interests=_interests_for(request, interests),
    )
    # Recorded on the same terms as a shelf: one tile, one listener, one
    # ranking version. Without it the popup would be the one surface whose
    # picks nobody could account for afterwards.
    if user:
        EVENTS.record_impressions(user, [("next_up", t.id) for t in picks])
    return {"topics": [t.as_dict() for t in picks], "algo": topics_mod.ALGO_VERSION}


@app.get("/api/explorenew")
async def explore_new(request: Request, interests: str = Query("", max_length=200)):
    """Explore New: episodes adjacent to a taste rather than inside it.

    This is `rank_might_like`, which has been written and tested since myFAM
    was built and shown nowhere since its shelf was removed - the only signal
    in the app that offers anything outside an established taste. Giving it a
    surface of its own is what makes it worth keeping.
    """
    _read_limit(request)
    user = _listener(request)
    body = topics_mod.build_explore_new(
        EVENTS, user, interests=_interests_for(request, interests)
    )
    if user:
        EVENTS.record_impressions(user, [("explore_new", t["id"]) for t in body["topics"]])
    body["algo"] = topics_mod.ALGO_VERSION
    return body


class EventRequest(BaseModel):
    kind: str = Field(..., max_length=16)
    topic_id: str = Field("", max_length=64)
    text: str = Field("", max_length=300)
    #: The follow-up predicted for the finished episode, so Go Deeper can offer it
    #: back later without a second lookup.
    thread: str = Field("", max_length=200)


@app.get("/api/myfam")
async def myfam(request: Request, interests: str = Query("", max_length=200)):
    """The four myFAM sections, ranked for this listener.

    Costs no model call: the topic bank is fixed and this only orders it.
    A listener with no history still gets Trending and a starter set, with
    the personal sections honestly empty rather than filled with fakes.
    """
    # A cheap read: ranking a fixed bank costs no model call, so it takes the
    # reader's limit rather than the generation one.
    _read_limit(request)
    user = _listener(request)
    feed = topics_mod.build_feed(EVENTS, user, interests=_interests_for(request, interests))
    # Logged here rather than inside build_feed, which stays a pure function of
    # the log - the whole ranking design is "computed on read, never stored",
    # and a ranker that writes cannot be tested by calling it. The impression
    # is a fact about this *request*, so it belongs at the request boundary.
    SOCIAL.seen(user)
    EVENTS.record_impressions(
        user,
        [(section["key"], topic["id"])
         for section in feed["sections"] for topic in section["topics"]],
    )
    feed["algo"] = topics_mod.ALGO_VERSION
    return feed


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
        topics_mod.Event(_listener(request), req.kind, req.topic_id, req.text, tags,
                         thread=req.thread)
    )
    SOCIAL.seen(_listener(request))
    return {"ok": True}


class PersonRequest(BaseModel):
    name: str = Field("", max_length=social_mod.MAX_NAME)
    handle: str = Field("", max_length=social_mod.MAX_HANDLE + 1)


class EchoRequest(BaseModel):
    query: str = Field(..., max_length=300)
    title: str = Field("", max_length=200)
    minutes: int = Field(3, ge=1, le=10)
    thread: str = Field("", max_length=200)


@app.post("/api/me")
async def set_me(req: PersonRequest, request: Request):
    """Name and handle for this device. Not an account - see /api/profile."""
    _read_limit(request)
    try:
        return SOCIAL.set_person(_listener(request), req.name, req.handle)
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
        echo = SOCIAL.echo(_listener(request), req.query, req.title, req.minutes, req.thread)
    except social_mod.SocialError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return echo.as_dict()


@app.delete("/api/echo")
async def delete_echo(request: Request, q: str = Query("", max_length=300),
                      minutes: int = Query(3, ge=1, le=10)):
    _read_limit(request)
    return {"ok": SOCIAL.unecho(_listener(request), q, minutes)}


@app.get("/api/profile")
async def profile(request: Request):
    """Counts and subjects from this listener's own event log. No model call."""
    _read_limit(request)
    user = _listener(request)
    body = topics_mod.summary(EVENTS, user)
    SOCIAL.seen(user)
    person = SOCIAL.person(user)
    body["name"] = person["name"]
    body["handle"] = person["handle"]
    body["joined"] = person["joined"]
    # Observed, not invented: when the server first saw this listener and when
    # it last did. `known` separates "never been here" from "here, unnamed",
    # which the page could not tell apart while a row meant "chose a name".
    body["last_seen"] = person["last_seen"]
    body["known"] = person["known"]
    body["mixes"] = [m.as_dict() for m in MIXES.public_for_user(user)]
    body["echoes"] = [e.as_dict(person["name"], person["handle"])
                      for e in SOCIAL.echoes_by(user, limit=12)]
    body["echo_count"] = len(SOCIAL.echoes_by(user, limit=200))
    return body


@app.get("/api/godeeper")
async def go_deeper(request: Request):
    """Follow-ups predicted for the episodes this listener finished.

    Costs nothing: the model named it on the episode's trailing marker line,
    which was never spoken. The episode itself does not tease it - it simply
    ends - and the suggestion is waiting here afterwards for anyone who wants
    to keep going.
    """
    _read_limit(request)
    return {"threads": EVENTS.open_threads(_listener(request))}


@app.get("/api/explore")
async def explore(request: Request, limit: int = Query(30, ge=1, le=60)):
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
    labels = SOCIAL.recent_echoes(exclude_user=_listener(request))
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
    # Same reason as /api/audio: this looks up a cache entry, and the entry it
    # looks for has to be keyed the same way the audio request keyed it.
    search: bool | None = Query(None),
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
    # `None`, not False. An omitted parameter has to stay omitted all the way
    # to plan_episode: `bool = Query(False)` turns "the listener said nothing"
    # into "the listener said no", which is a different thing and beats
    # SEARCH_MODE=auto. The browser never sends this parameter, so with a
    # False default the freshness heuristic was consulted exactly never.
    # search=1 / search=0 still win, which is what opt-in means.
    search: bool | None = Query(None, description="Force research on (1) or off (0); "
                                                  "omit to let the question decide"),
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
    user = _listener(request)
    minutes = min(minutes, entitlements.max_minutes(_tier(request),
                                                    settings.max_minutes))
    plan = _validated_plan(q, minutes, context, search, cached_only,
                           _attachments_for(user, attach))

    # After validation, so a malformed request never costs an allowance, and
    # before anything expensive starts. An Explore replay counts against a
    # different, looser allowance because it provably cannot write a script -
    # the pipeline refuses - so it costs GPU seconds and nothing else.
    reserved = _reserve(request, "explore" if cached_only else "episode")

    try:
        pipeline = _make_pipeline(voice or None)
    except TTSUnavailable as exc:
        # The server cannot speak at all. Nothing was generated and nothing was
        # billed, so the allowance goes back - this is the machine being
        # broken, not the listener spending.
        _refund(reserved, user)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Ask for a GPU now, before Claude has written a word.
    #
    # A serverless worker that has scaled to zero pays container boot plus a
    # ~10s model load on its first job. Firing that here means it happens
    # *alongside* script generation instead of in front of the first chunk -
    # which is CLAUDE.md's rule that latency is answered by starting earlier
    # rather than by filling the gap, applied to the one wait this split adds.
    # It is a hint: `wake()` never raises and never blocks, and a miss costs
    # only the cold start it was trying to hide.
    _wake_remote_voice()

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
    # Instrumentation only - nothing below changes what is served. These are
    # the marks that turn the interval between "audio exists" and "the client
    # has a byte" from an invisible cost into a measured one.
    first_pcm_at: float | None = None
    preroll_at: float | None = None
    first_byte_at: float | None = None
    chunks_primed = 0
    try:
        async for chunk in source:
            primed.append(chunk)
            chunks_primed += 1
            if first_pcm_at is None and len(chunk) > WAV_HEADER_BYTES:
                first_pcm_at = time.monotonic() - started
            if sum(len(c) for c in primed) - WAV_HEADER_BYTES >= preroll_bytes:
                preroll_at = time.monotonic() - started
                break
    except NotCached as exc:
        # Expected, not a fault: the entry expired between listing and tapping.
        # The interface drops the card and moves on.
        _refund_if_unspent(reserved, user, stats.usage)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("generation failed before any audio was produced")
        # A failure is not a refund. Research may already have been billed, and
        # a retry loop against a broken key would otherwise be the cheapest
        # thing in the ledger while being the most expensive thing on the
        # invoice.
        _record_usage(user, stats.usage,
                      surface=_surface(cached_only, topic_id, context),
                      minutes=plan.minutes, audio_seconds=stats.audio_seconds,
                      cache_hit=stats.cache == "hit")
        # Same rule as the ledger above: refunded only if nothing was billed.
        _refund_if_unspent(reserved, user, stats.usage)
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

    primed_bytes = max(0, sum(len(c) for c in primed) - WAV_HEADER_BYTES)
    primed_seconds = primed_bytes / (sample_rate * 2)

    async def body():
        nonlocal first_byte_at
        try:
            for chunk in primed:
                if first_byte_at is None:
                    first_byte_at = time.monotonic() - started
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
                "episode q=%r %s wall=%.1fs preroll=%.2fs chunks_primed=%d "
                "audio_primed=%.2fs first_pcm=%s preroll_satisfied=%s "
                "first_byte=%s marks=%s",
                plan.query, stats.as_dict(), time.monotonic() - started,
                PREROLL_SECONDS, chunks_primed, primed_seconds,
                _ms(first_pcm_at), _ms(preroll_at), _ms(first_byte_at),
                json.dumps(stats.marks.to_dict(), default=str),
            )
            # The ledger row, written last, when the numbers are final.
            #
            # Here and not at the model call because this is the only place
            # that knows *whose* episode it was - and the id comes from
            # `_listener(request)`, the session cookie, never a parameter,
            # which is the settled rule for anything per-listener.
            #
            # A disconnect mid-stream still records: the money was spent
            # whether or not it was listened to, and a ledger that only counts
            # completed plays under-reports exactly the abusive pattern of
            # starting many episodes and finishing none.
            _record_usage(
                user, stats.usage,
                surface=_surface(cached_only, topic_id, context),
                minutes=plan.minutes, audio_seconds=stats.audio_seconds,
                cache_hit=stats.cache == "hit",
            )

    # Recorded here rather than client-side: audio is being served, so the
    # play is a fact. A dropped event costs one weak signal, never the episode.
    # Both writes sit after the plan and the pipeline are ready and before the
    # response object is built, so neither is in front of the first word.
    if user:
        SOCIAL.seen(user)
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
            # Measurement headers. Additive: the player reads none of them,
            # and `tools/preroll_sweep.py` reads all of them.
            "X-Preroll-Seconds": f"{PREROLL_SECONDS:g}",
            "X-Chunks-Primed": str(chunks_primed),
            "X-Audio-Primed-Seconds": f"{primed_seconds:.3f}",
            "X-First-PCM-Seconds": f"{first_pcm_at:.4f}" if first_pcm_at is not None else "",
            "X-Preroll-Satisfied-Seconds": f"{preroll_at:.4f}" if preroll_at is not None else "",
            # The episode's own marks, so a client-side probe can read the
            # server's view of the same request rather than inferring it.
            "X-Episode-Marks": json.dumps(stats.marks.summary(), default=str),
        },
    )


#: The credential that gates the usage report. Absent by default, and absent
#: means the endpoint does not exist rather than that it is open: this data is
#: every listener's spending history, and an endpoint that is protected only
#: when somebody remembers to protect it is not protected.
ADMIN_TOKEN = os.environ.get("FAM_ADMIN_TOKEN", "").strip()


def _require_admin(request: Request) -> None:
    """Constant-time check of the admin credential, or a 404.

    404 rather than 401: an unconfigured deployment should not advertise that
    it has a billing endpoint at all, and a wrong token should not tell the
    person holding it that they got the path right.
    """
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=404, detail="Not found")
    sent = (request.headers.get("x-admin-token")
            or request.headers.get("authorization", "").removeprefix("Bearer ").strip())
    if not sent or not hmac.compare_digest(sent, ADMIN_TOKEN):
        raise HTTPException(status_code=404, detail="Not found")


@app.get("/api/usage")
async def usage(
    request: Request,
    days: float = Query(30.0, gt=0, le=3650, description="Window, ending now"),
    top: int = Query(10, ge=1, le=100, description="How many top listeners"),
    flagged: bool = Query(False, description="Also run the abuse thresholds"),
) -> dict:
    """The billing and usage report, on demand.

    Everything `tools/usage_report.py` prints, as JSON, so the same numbers are
    available to a dashboard, a finance spreadsheet and a person at a terminal
    without three implementations disagreeing about what a month is.

    Not paced by `_rate_limit`: it makes no model call, and an operator pulling
    a report should not be competing with listeners for the generation budget.
    """
    _require_admin(request)
    now = time.time()
    report = METER.report(since=now - days * 86400, until=now, top=top)
    if flagged:
        report["flagged"] = metering.suspects(METER)
    return report


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException):
    # Headers are forwarded, not dropped. A 429 from the quota carries the
    # whole verdict in `X-FAM-Quota` - what the limit was, what is left, when
    # it resets - and a handler that kept only the sentence would leave the
    # interface able to say "no" and nothing else.
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code,
                        headers=getattr(exc, "headers", None))


# Also resolved from the project root, and for the same reason as the
# databases: a relative directory follows the working directory. This one
# at least fails loudly - starting the server from anywhere else raised
# "Directory 'static' does not exist" - but it made the app impossible to
# launch from outside its own folder, which is how the quiet database
# version of this bug stayed hidden behind it.
app.mount("/", StaticFiles(directory=str(PROJECT_ROOT / "static"), html=True),
          name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host=settings.host, port=settings.port, reload=False)
