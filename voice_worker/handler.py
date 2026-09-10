"""RunPod Serverless entrypoint: the voice, asleep until someone asks.

    CMD ["python", "-u", "voice_worker/handler.py"]

RunPod hands the handler `{"input": {...}}` and publishes whatever it returns
as `output`. That envelope is the *only* thing this file adds to
`voice_worker/synth.py` - which is the point, because `voice_worker/server.py`
adds a different one over the same function and the app cannot tell them apart.

## Two things that are deliberate

**The model is loaded at import, not on the first request.** RunPod starts a
worker before routing a job to it, so a load that happens here is paid during
the boot the platform already expects, rather than inside the first listener's
episode. `remote_voice.wake()` exists to trigger exactly this early.

**A failure returns `{"error": ...}` rather than raising.** RunPod marks a
raised job FAILED with a stack trace in its own console, which the app cannot
see. Returning the reason puts it in the HTTP reply, where `remote_voice.py`
turns it into a sentence the interface can show - the difference between "the
voice failed" and "no rights record beside reference_3.wav".
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_worker.synth import WorkerError, preflight, synthesise  # noqa: E402


logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("voice_worker")


async def handler(job: dict) -> dict:
    """One job. `job["input"]` is the wire contract from `remote_voice.py`."""
    try:
        return await synthesise(job.get("input") or {})
    except WorkerError as exc:
        log.warning("refused: %s", exc)
        return {"error": str(exc)}
    except Exception as exc:  # pragma: no cover - depends on the card
        log.exception("synthesis failed")
        return {"error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    ready, detail = preflight()
    if not ready:
        # Refuse to accept work this node cannot do. A worker that starts
        # anyway would take jobs off the queue and fail every one of them,
        # which reads as "the app is broken" rather than "this node has no GPU".
        log.error("this worker cannot speak: %s", detail)
        raise SystemExit(1)
    log.info("chatterbox ready (%s); loading the model before accepting work", detail)

    from voice_worker.synth import _load

    rate = asyncio.get_event_loop().run_until_complete(_load())
    log.info("model resident, emitting %s Hz", rate)

    import runpod

    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
