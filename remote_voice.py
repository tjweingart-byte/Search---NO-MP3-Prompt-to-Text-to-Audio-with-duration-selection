"""Chatterbox on somebody else's GPU, reached over HTTP.

The app and the voice stop being the same process. `app.py` runs on a CPU host
(Render), `ChatterboxTTS` runs on a card, and this is the seam between them:
sentences in, 16-bit PCM out, exactly as `ChatterboxEngine` does it in-process.

## Why this is one engine and not two

RunPod sells the same GPU two ways - a serverless endpoint that sleeps when
idle, and a pod that stays up - and the right answer changes with volume. So
the *transport* is configuration and the *engine* is not:

    VOICE_BACKEND=remote  REMOTE_VOICE_TRANSPORT=runpod  RUNPOD_ENDPOINT_ID=...
    VOICE_BACKEND=remote  REMOTE_VOICE_TRANSPORT=http    REMOTE_VOICE_URL=https://...

Both speak the identical JSON contract below, so one worker image serves both
and moving between them is two environment variables. Nothing in `pipeline.py`,
`script_generator.py`, the cache or the player knows which one is answering.

    request   {"text": str, "voice": str|None, "sample_rate": int,
               "format": "pcm_s16le"}
    response  {"audio": "<base64 pcm_s16le>", "sample_rate": int,
               "samples": int, "engine": "chatterbox"}

The transports differ only in the envelope: RunPod wraps the request in
`{"input": ...}` and the reply in `{"output": ...}`, and may answer a slow call
with a job id to poll. `voice_worker/` implements both entrypoints over one
`synthesise()`.

## Three things this has to get right

* **The browser is still given raw PCM.** "No MP3, no audio files" is a rule
  about what reaches the listener and what is written to disk, not about what
  two servers say to each other. Base64 over JSON is a wire encoding; it is
  decoded here, in memory, and never becomes a file. Nothing is transcoded.

* **The sample rate is known before the first call.** `app.py` writes the
  stream header from `engine.sample_rate` before any audio has been requested,
  so the engine cannot wait to be told. It is configured, it is *asked for* in
  every request, and the reply is checked against it - a worker that answered
  at a different rate would play at the wrong pitch, which is a failure you
  hear rather than one you are told about.

* **It never quietly becomes something else.** No fallback to a local engine,
  no substituted voice, no silent retry that ends in a tone. Every failure
  raises with the reason attached, so a broken endpoint arrives as an error the
  interface can show. This is the guard PROBLEMS.md §61 removed with WellSaid
  and said to re-add by hand for the next hosted engine; this is by hand.

## The cold start, and why it is answered by starting earlier

A serverless worker that has scaled to zero pays container boot plus a ~10s
model load on the first request. That is exactly the wait the one-sentence spec
refuses - and exactly the wait CLAUDE.md says to answer by *starting earlier*
rather than filling.

So `wake()` fires a throwaway job the moment a request arrives, before Claude
has written a word. The worker boots while the script is being written, which
is several seconds of cover that costs nothing and is not filler: it is the
same "do it before the listener is waiting" move as prefetching scripts on the
browse surfaces. It is a hint, not a guarantee - `wake()` never raises, never
blocks the request, and a miss costs only the cold start it was trying to hide.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import time
from dataclasses import dataclass
from typing import Any

from config import settings
from tts import TTSEngine, Voice


log = logging.getLogger(__name__)

#: What the worker is asked for and what it must answer with. Not negotiable
#: per request: the app's whole audio path is 16-bit little-endian PCM.
WIRE_FORMAT = "pcm_s16le"

#: Transports this engine knows how to speak. Adding one means adding an
#: envelope, not an engine.
TRANSPORTS = ("runpod", "http")

#: The route `voice_worker/server.py` serves the contract on, and the only part
#: of the URL this side invents. `REMOTE_VOICE_URL` is the pod's *base* URL.
SYNTH_ROUTE = "/synth"


class RemoteVoiceError(RuntimeError):
    """The remote voice could not speak, and this says why.

    Deliberately not a subclass of anything that triggers a fallback. A hosted
    voice that fails must fail, not be replaced by a different one.
    """


@dataclass(frozen=True)
class RemoteConfig:
    """The settings this engine needs, resolved and validated together.

    Built from `settings` rather than read one at a time, so "is this
    configured" is one question with one answer instead of four scattered
    truthiness checks that can each be half-right.
    """

    transport: str
    url: str
    api_key: str
    sample_rate: int
    timeout: float
    connect_timeout: float
    concurrency: int
    voice: str

    @classmethod
    def from_settings(cls) -> "RemoteConfig":
        transport = (settings.remote_voice_transport or "").strip().lower()
        if transport == "runpod":
            endpoint = (settings.runpod_endpoint_id or "").strip()
            base = (settings.runpod_base_url or "").rstrip("/")
            url = f"{base}/{endpoint}" if endpoint else ""
            key = _credential("RUNPOD_API_KEY", settings.runpod_api_key)
        else:
            url = (settings.remote_voice_url or "").strip().rstrip("/")
            key = _credential("REMOTE_VOICE_TOKEN", settings.remote_voice_token)
        return cls(
            transport=transport,
            url=url,
            api_key=key,
            sample_rate=int(settings.remote_voice_sample_rate),
            timeout=float(settings.remote_voice_timeout),
            connect_timeout=float(settings.remote_voice_connect_timeout),
            concurrency=max(1, int(settings.remote_voice_concurrency)),
            voice=(settings.remote_voice_id or "").strip(),
        )

    def synth_url(self) -> str:
        """Where the POST actually goes.

        `REMOTE_VOICE_URL` is documented as the pod's base URL and the route is
        this side's to add - but an operator who pastes the URL they were
        testing with, the one that already ends in `/synth`, has configured
        something unambiguous, and appending a second `/synth` to it produces a
        404 indistinguishable from a worker that has no route at all.
        """
        if self.url.endswith(SYNTH_ROUTE):
            return self.url
        return self.url + SYNTH_ROUTE

    def base_url(self) -> str:
        """The origin, whichever way the URL was written.

        `/health` and `/openapi.json` hang off this, so a configured
        `.../synth` must not send the probes to `.../synth/health`.
        """
        if self.url.endswith(SYNTH_ROUTE):
            return self.url[:-len(SYNTH_ROUTE)]
        return self.url

    def problem(self) -> str:
        """Why this configuration cannot be used, or "" if it can.

        Configuration only - no network. `available()` is called from
        `/api/health` and must not become a request to a third party; the
        question "does this endpoint actually speak" is answered by making it
        speak, in `warm_up()`, and reported separately.
        """
        if self.transport not in TRANSPORTS:
            return (f"REMOTE_VOICE_TRANSPORT={self.transport!r} is not a "
                    f"transport. Use one of: {', '.join(TRANSPORTS)}")
        if not self.url:
            missing = ("RUNPOD_ENDPOINT_ID" if self.transport == "runpod"
                       else "REMOTE_VOICE_URL")
            return f"{missing} is not set"
        if self.transport == "runpod" and not self.api_key:
            return "RUNPOD_API_KEY is not set"
        if self.sample_rate <= 0:
            return f"REMOTE_VOICE_SAMPLE_RATE={self.sample_rate} must be positive"
        return ""


def _credential(name: str, configured: str) -> str:
    """The value in force, preferring the credential chain to import-time state.

    `credentials.active()` reflects a rotation that happened after startup;
    `settings` is a snapshot taken at import. Falling back to the snapshot
    keeps this working when the chain has no opinion.
    """
    try:
        import credentials

        found = credentials.active(name)
        if found:
            return found
    except Exception:  # pragma: no cover - the chain is optional here
        pass
    return (configured or "").strip()


@dataclass
class Reachability:
    """The last time a real call was attempted, and what happened.

    "Verify, do not inspect" (PROBLEMS.md §52) applied to a remote voice: a
    configured endpoint is not a reachable one, and a health check that only
    reads configuration answers a cheaper question than the one being asked.
    `warm_up()` performs the real synthesis and records the answer here, so
    `/api/health` can report what was actually observed rather than what was
    set. `unknown` until something has genuinely been tried.
    """

    state: str = "unknown"  # unknown | ok | failed
    detail: str = ""
    at: float = 0.0
    latency: float = 0.0

    def as_dict(self) -> dict:
        out = {"state": self.state, "detail": self.detail}
        if self.at:
            out["age_seconds"] = round(time.time() - self.at, 1)
        if self.latency:
            out["latency_seconds"] = round(self.latency, 3)
        return out


class RemoteChatterboxEngine(TTSEngine):
    """Chatterbox over HTTP. Same voice, different machine.

    Everything about the *audio* is decided on the worker, which runs the
    generation settings `tts.CHATTERBOX_GENERATION` names and clones the same
    reference recording. This side owns the transport and nothing else, which
    is why switching a listener between an in-process card and a remote one
    changes latency and cost but not what they hear.
    """

    name = "remote"

    #: Shared across requests: `build_engine()` constructs a new instance per
    #: call, so anything that must be reused - the connection pool, the
    #: concurrency gate, what the last real call proved - lives on the class.
    _client: Any = None
    _gate: "asyncio.Semaphore | None" = None
    _gate_size: int = 0
    _reachability = Reachability()
    _woken_at: float = 0.0
    #: A route the worker named itself, kept as (base url, route) so a
    #: reconfigured endpoint is not answered with the old one's answer.
    _found_route: tuple[str, str] | None = None

    # -- configuration -----------------------------------------------------

    @classmethod
    def config(cls) -> RemoteConfig:
        return RemoteConfig.from_settings()

    @classmethod
    def diagnose(cls) -> tuple[bool, str]:
        """Why this engine can or cannot serve, in one sentence."""
        config = cls.config()
        problem = config.problem()
        if problem:
            return False, problem
        return True, f"{config.transport}, {config.sample_rate} Hz"

    @classmethod
    def available(cls) -> bool:
        """Configured well enough to try. Deliberately not memoised.

        `ChatterboxEngine` caches this because importing torch and probing a
        card is expensive and its answer cannot change while the process runs.
        Here the inputs are environment variables and a credential that
        `credentials.refresh()` can replace mid-run, so caching would pin a
        stale answer past a rotation.
        """
        return not cls.config().problem()

    @classmethod
    def voices(cls) -> list[Voice]:
        if not cls.available():
            return []
        config = cls.config()
        label = config.voice or "reference_3"
        return [Voice(id=f"remote:{label}", label="FAM", engine=cls.name,
                      detail=f"Chatterbox via {config.transport}")]

    @property
    def sample_rate(self) -> int:
        """Configured, not discovered - the header is written before the first
        call. `synth` refuses a reply that disagrees with it."""
        return int(settings.remote_voice_sample_rate)

    # -- plumbing ----------------------------------------------------------

    @classmethod
    def _http(cls, config: RemoteConfig):
        """One connection pool for the process.

        A pool per request would pay TLS setup on every chunk - roughly fifteen
        times an episode - which is exactly the kind of cost that hides inside
        an average and shows up in the first-chunk number.
        """
        if cls._client is None:
            import httpx

            cls._client = httpx.AsyncClient(
                timeout=httpx.Timeout(config.timeout,
                                      connect=config.connect_timeout),
                headers={"User-Agent": "FAM/remote-voice"},
                limits=httpx.Limits(max_connections=32),
            )
        return cls._client

    @classmethod
    async def aclose(cls) -> None:
        """Release the pool. For tests and for a clean shutdown."""
        client, cls._client = cls._client, None
        if client is not None:
            await client.aclose()

    @classmethod
    def _semaphore(cls, size: int) -> asyncio.Semaphore:
        """One generation at a time, times `size`.

        In-process Chatterbox serialises on `Semaphore(1)` because one card
        cannot run concurrent generations safely. That is a property of the
        card, not of the interface, so a remote backend that fans out across
        workers sets its own ceiling. Rebuilt when the setting changes so a
        reconfigured limit is not ignored until restart.
        """
        if cls._gate is None or cls._gate_size != size:
            cls._gate = asyncio.Semaphore(size)
            cls._gate_size = size
        return cls._gate

    @staticmethod
    def _payload(text: str, config: RemoteConfig) -> dict:
        return {
            "text": text,
            "voice": config.voice or None,
            "sample_rate": config.sample_rate,
            "format": WIRE_FORMAT,
        }

    def _headers(self, config: RemoteConfig) -> dict:
        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        return headers

    # -- the two envelopes -------------------------------------------------

    async def _call_runpod(self, payload: dict, config: RemoteConfig) -> dict:
        """RunPod's queue API: submit, and poll if the sync wait ran out.

        `/runsync` answers directly when the job finishes inside its window and
        otherwise hands back a job id with `IN_QUEUE` or `IN_PROGRESS`. A cold
        worker routinely exceeds that window, so treating the id as a failure
        would turn every cold start into a broken episode.
        """
        client = self._http(config)
        response = await client.post(f"{config.url}/runsync",
                                     json={"input": payload},
                                     headers=self._headers(config))
        body = self._decode_json(response, "runsync")
        deadline = time.monotonic() + config.timeout
        while True:
            status = str(body.get("status", "")).upper()
            if status == "COMPLETED":
                output = body.get("output")
                if not isinstance(output, dict):
                    raise RemoteVoiceError(
                        "RunPod reported COMPLETED with no output object; the "
                        "worker returned "
                        f"{type(output).__name__}. Check the endpoint's logs.")
                return output
            if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                raise RemoteVoiceError(
                    f"RunPod job {status}: {body.get('error') or 'no reason given'}")
            job_id = body.get("id")
            if not job_id:
                raise RemoteVoiceError(
                    f"RunPod answered {status or 'nothing'} with no job id: "
                    f"{str(body)[:300]}")
            if time.monotonic() >= deadline:
                raise RemoteVoiceError(
                    f"RunPod job {job_id} was still {status} after "
                    f"{config.timeout:.0f}s. A cold worker can exceed this - "
                    "raise REMOTE_VOICE_TIMEOUT, or keep a worker warm.")
            await asyncio.sleep(0.25)
            polled = await client.get(f"{config.url}/status/{job_id}",
                                      headers=self._headers(config))
            body = self._decode_json(polled, f"status/{job_id}")

    async def _call_http(self, payload: dict, config: RemoteConfig) -> dict:
        """A plain speech server: one POST, one answer, no envelope.

        The one thing that can go wrong here without going wrong on the card is
        the *address*. A 404 is not a worker failing to speak; it is nothing
        having been asked - and the two are indistinguishable in a log unless
        this says which. So a 404 is followed by one question to the worker
        itself (`/openapi.json`, which FastAPI serves for free) rather than by a
        guess: either it names the route it does serve, and this retries there,
        or nothing at that address is a FAM voice worker and the error says so
        with the two things that cause it.
        """
        client = self._http(config)
        found = type(self)._found_route
        url = found[1] if found and found[0] == config.url else config.synth_url()
        response = await client.post(url, json=payload,
                                     headers=self._headers(config))
        if response.status_code == 404:
            route = await self._route_from_worker(config, url)
            response = await client.post(route, json=payload,
                                         headers=self._headers(config))
            # Remembered only once it has answered something other than 404:
            # a second wrong route is worse than the first.
            if response.status_code != 404:
                type(self)._found_route = (config.url, route)
        return self._decode_json(response, "synth")

    async def _route_from_worker(self, config: RemoteConfig, tried: str) -> str:
        """Ask the worker which route takes the contract, or say why there is none.

        Bounded on purpose: one GET, only ever after a 404, never on the path a
        working deployment takes. It reads the worker's own schema instead of
        trying candidate paths, because a POST to a guessed route on a machine
        that is not this worker is a request to somebody else's service.
        """
        client = self._http(config)
        base = config.base_url()
        try:
            schema = await client.get(f"{base}/openapi.json",
                                      headers=self._headers(config))
        except Exception as exc:
            raise RemoteVoiceError(
                f"remote voice synth returned HTTP 404 at {tried}, and asking "
                f"the worker what it serves failed too: {type(exc).__name__}: "
                f"{exc}") from exc
        paths: dict = {}
        if schema.status_code < 400:
            try:
                body = schema.json()
                paths = body.get("paths") or {} if isinstance(body, dict) else {}
            except Exception:
                paths = {}
        posts = [path for path, methods in paths.items()
                 if isinstance(methods, dict) and "post" in methods]
        speaks = [path for path in posts if "synth" in path.lower()] or posts
        if len(speaks) == 1:
            log.warning("remote voice: %s has no %s; this worker serves POST %s "
                        "and that is what will be used", base, SYNTH_ROUTE,
                        speaks[0])
            route = speaks[0] if speaks[0].startswith("/") else "/" + speaks[0]
            return base + route
        if not paths:
            raise RemoteVoiceError(
                f"nothing at {base} answers as a FAM voice worker: POST {tried} "
                f"returned 404 and GET {base}/openapi.json returned "
                f"{schema.status_code}. The two things that cause this are a "
                "REMOTE_VOICE_URL naming a proxied port the worker is not "
                "listening on (Dockerfile.voice serves ${PORT:-8001}), and a "
                "pod started without VOICE_WORKER_MODE=http, which runs the "
                "serverless handler and opens no port at all.")
        raise RemoteVoiceError(
            f"the worker at {base} does not serve {SYNTH_ROUTE} and does not "
            f"name one route that could: it posts {sorted(posts) or 'nothing'}. "
            "Point REMOTE_VOICE_URL at a FAM voice worker, or rebuild the "
            "image from Dockerfile.voice.")

    @staticmethod
    def _decode_json(response, what: str) -> dict:
        """Turn any non-answer into a sentence naming the endpoint and the code.

        A hosted voice fails in ways a local one cannot - 401, 404, a proxy's
        HTML error page - and "expected object, got str" would send whoever
        reads the log looking in the wrong place entirely.
        """
        if response.status_code >= 400:
            raise RemoteVoiceError(
                f"remote voice {what} returned HTTP {response.status_code}: "
                f"{response.text[:300]}")
        try:
            body = response.json()
        except Exception as exc:
            raise RemoteVoiceError(
                f"remote voice {what} did not return JSON "
                f"({type(exc).__name__}): {response.text[:200]}") from exc
        if not isinstance(body, dict):
            raise RemoteVoiceError(
                f"remote voice {what} returned {type(body).__name__}, not an object")
        return body

    # -- synthesis ---------------------------------------------------------

    async def synth(self, text: str, wpm: float, voice: str | None = None) -> bytes:
        """`wpm` is accepted and ignored - Chatterbox has no rate control.

        Same as the in-process engine, and for the same reason: length is held
        by the budget and by trimming at a sentence boundary, not by speeding
        the voice up.
        """
        config = self.config()
        problem = config.problem()
        if problem:
            raise RemoteVoiceError(f"remote voice is not configured: {problem}")

        payload = self._payload(text, config)
        started = time.monotonic()
        async with self._semaphore(config.concurrency):
            try:
                if config.transport == "runpod":
                    output = await self._call_runpod(payload, config)
                else:
                    output = await self._call_http(payload, config)
            except RemoteVoiceError:
                raise
            except Exception as exc:
                # Network errors arrive as a dozen different exception types.
                # All of them mean the same thing to a listener, and none of
                # them may become a different voice.
                raise RemoteVoiceError(
                    f"remote voice ({config.transport}) failed: "
                    f"{type(exc).__name__}: {exc}") from exc

        pcm = self._pcm_from(output, config)
        elapsed = time.monotonic() - started
        seconds = len(pcm) / float(config.sample_rate * settings.sample_width or 1)
        log.debug("remote voice: %d words -> %.1fs audio in %.2fs (%.1fx realtime)",
                  len(text.split()), seconds, elapsed,
                  seconds / elapsed if elapsed else 0)
        type(self)._reachability = Reachability(
            state="ok", detail=config.transport, at=time.time(), latency=elapsed)
        return pcm

    @classmethod
    def _pcm_from(cls, output: dict, config: RemoteConfig) -> bytes:
        """Validate the reply hard, then decode it.

        Every check here is a failure that would otherwise be *audible* rather
        than reported: a wrong sample rate plays at the wrong pitch, an odd
        byte count shifts every sample by one and turns the episode into noise,
        and an empty payload is the silent-empty-episode failure this project
        has lost the most time to.
        """
        encoded = output.get("audio")
        if not encoded:
            raise RemoteVoiceError(
                "the remote voice returned no audio. Worker said: "
                f"{str(output.get('error') or output)[:300]}")
        fmt = str(output.get("format", WIRE_FORMAT)).lower()
        if fmt != WIRE_FORMAT:
            raise RemoteVoiceError(
                f"the remote voice returned {fmt!r}; this app plays "
                f"{WIRE_FORMAT} and does not transcode.")
        rate = int(output.get("sample_rate") or config.sample_rate)
        if rate != config.sample_rate:
            raise RemoteVoiceError(
                f"the remote voice answered at {rate} Hz but the stream header "
                f"already said {config.sample_rate} Hz. Set "
                f"REMOTE_VOICE_SAMPLE_RATE={rate} to match the worker; playing "
                "it would be the wrong pitch.")
        try:
            pcm = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise RemoteVoiceError(
                f"the remote voice's audio was not valid base64: {exc}") from exc
        if not pcm:
            raise RemoteVoiceError("the remote voice returned zero bytes of audio")
        if len(pcm) % settings.sample_width:
            raise RemoteVoiceError(
                f"the remote voice returned {len(pcm)} bytes, which is not a "
                f"whole number of {settings.sample_width}-byte samples")
        return pcm

    # -- the cold start ----------------------------------------------------

    @classmethod
    async def wake(cls) -> None:
        """Ask for a worker now, so one exists by the time there is audio to make.

        Fired when a request arrives, alongside script generation rather than
        in front of it. Never raises and never blocks: a failed wake costs the
        cold start it was trying to hide and nothing else, so it must not be
        able to cost an episode.
        """
        config = cls.config()
        if config.problem() or config.transport != "runpod":
            return  # nothing to wake: an always-on pod is already up
        now = time.monotonic()
        if now - cls._woken_at < settings.remote_voice_wake_interval:
            return  # already asked recently; another job would just queue
        cls._woken_at = now
        try:
            client = cls._http(config)
            await client.post(
                f"{config.url}/run",
                json={"input": {"text": "", "warm": True,
                                "sample_rate": config.sample_rate}},
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {config.api_key}"},
                timeout=5.0,
            )
            log.debug("remote voice: wake sent")
        except Exception as exc:
            log.debug("remote voice: wake failed (%s); the next request pays "
                      "the cold start", type(exc).__name__)

    # -- what is actually true --------------------------------------------

    @classmethod
    def reachability(cls) -> dict:
        return cls._reachability.as_dict()

    @classmethod
    def record_failure(cls, detail: str) -> None:
        cls._reachability = Reachability(state="failed", detail=detail,
                                         at=time.time())


def report() -> dict:
    """What `/api/health` says about the remote voice, if one is configured."""
    ok, detail = RemoteChatterboxEngine.diagnose()
    config = RemoteChatterboxEngine.config()
    return {
        "configured": ok,
        "detail": detail,
        "transport": config.transport,
        "sample_rate": config.sample_rate,
        # Where it points, never what authorises it.
        "endpoint": config.url or None,
        "reachable": RemoteChatterboxEngine.reachability(),
    }
