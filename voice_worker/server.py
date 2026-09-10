"""Always-on entrypoint: the same voice, behind a plain HTTP port.

    uvicorn voice_worker.server:app --host 0.0.0.0 --port 8001

This is the other half of the switch `remote_voice.py` describes. Serverless
sleeps and pays per second; a pod stays up and pays per hour, which becomes the
cheaper answer somewhere north of a few hours of audio a day and is always the
lower-latency one because nothing ever cold-starts. Moving between them is two
environment variables on the app, because both entrypoints wrap one
`synthesise()` and answer with the identical object.

## The port is the security boundary

A serverless endpoint is authenticated by RunPod: a job needs the account's API
key to be queued at all. A pod's exposed port is authenticated by nobody. So
`REMOTE_VOICE_TOKEN` is checked here when it is set, and its absence is logged
at every boot rather than assumed to be deliberate - an open endpoint on a
rented GPU is somebody else's free TTS service, billed to you, and it would
look exactly like your own traffic in the metering log.
"""
from __future__ import annotations

import logging
import os
import secrets
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Header, HTTPException, Request  # noqa: E402

from voice_worker.synth import WorkerError, preflight, synthesise  # noqa: E402


logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("voice_worker")

app = FastAPI(title="FAM voice worker")

#: Shared secret the app sends as `Authorization: Bearer ...`. Read at import
#: from the process environment: this is a single-purpose container, and a
#: rotation is a restart.
TOKEN = (os.environ.get("REMOTE_VOICE_TOKEN") or "").strip()


def _authorised(header: str | None) -> bool:
    """Constant-time comparison, so the check cannot be timed open."""
    if not TOKEN:
        return True
    sent = (header or "")
    prefix = "Bearer "
    if sent.startswith(prefix):
        sent = sent[len(prefix):]
    return secrets.compare_digest(sent.strip(), TOKEN)


@app.on_event("startup")
async def _startup() -> None:
    ready, detail = preflight()
    if not ready:
        # Loud, and then it keeps serving /health so an operator can see why.
        # Unlike the serverless worker there is no queue to poison here, and a
        # container that exits on boot is harder to diagnose than one that
        # answers the question.
        log.error("this worker cannot speak: %s", detail)
        return
    if not TOKEN:
        log.warning("REMOTE_VOICE_TOKEN is not set: this port will synthesise "
                    "for anyone who can reach it, billed to this pod")
    log.info("chatterbox ready (%s); loading the model", detail)
    from voice_worker.synth import _load

    log.info("model resident, emitting %s Hz", await _load())


@app.get("/health")
async def health() -> dict:
    """What this worker can actually do. Cheap enough to be a probe target."""
    ready, detail = preflight()
    return {"ready": ready, "detail": detail, "engine": "chatterbox",
            "authenticated": bool(TOKEN)}


@app.post("/synth")
async def synth(request: Request,
                authorization: str | None = Header(default=None)) -> dict:
    if not _authorised(authorization):
        raise HTTPException(status_code=401, detail="bad or missing bearer token")
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400,
                            detail=f"body is not JSON: {exc}") from exc
    try:
        return await synthesise(payload)
    except WorkerError as exc:
        # 422, not 500: everything `WorkerError` covers is something the caller
        # or the deployment can fix, and the message says which.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - depends on the card
        log.exception("synthesis failed")
        raise HTTPException(status_code=500,
                            detail=f"{type(exc).__name__}: {exc}") from exc
