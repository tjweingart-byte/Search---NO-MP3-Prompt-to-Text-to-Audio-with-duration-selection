"""Reference Chatterbox endpoint, to run ON the GPU box.

This file is **not** imported by the Experiment Engineer and nothing in this
repository starts it. It exists so the endpoint contract is executable rather
than described: copy it to the pod, run it, and the `chatterbox` adapter can
talk to it.

    pip install fastapi uvicorn chatterbox-tts torch torchaudio
    python chatterbox_server_example.py          # serves on :8000

    export CHATTERBOX_ENDPOINT=https://<pod-host>/synthesise

It wraps the same `chatterbox_impl.synthesise()` the local arm uses, so
`chatterbox` and `chatterbox_local` differ in *where* they run and not in what
they do - which is the only way the comparison between them means anything.

The model is loaded at import so the first request is not paying for it, and
`gpu_seconds` reports the generate call alone, matching what the local arm
measures.
"""
from __future__ import annotations

import base64
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI                     # noqa: E402
from pydantic import BaseModel                  # noqa: E402

from experiments.adapters import chatterbox_impl  # noqa: E402

app = FastAPI(title="Chatterbox Turbo for FAM")


class Request(BaseModel):
    text: str
    #: A hint only. The response reports the model's own rate, which wins.
    sample_rate: int | None = None


@app.on_event("startup")
def warm() -> None:
    """Load and warm before the first request, so it is not paying for either."""
    device, _ = chatterbox_impl.resolve_device(None)
    model, seconds = chatterbox_impl.load_model(device)
    chatterbox_impl.warm_up(model, device)
    print(f"chatterbox turbo ready on {device} in {seconds:.1f}s", flush=True)


@app.post("/synthesise")
def synthesise(request: Request) -> dict:
    # warmup is a no-op after startup; inference_mode matches the chunked
    # benchmark. `gpu_seconds` is the fenced generate time and nothing else.
    out = chatterbox_impl.synthesise(request.text, warmup=True, inference_mode=True)
    return {
        "pcm_base64": base64.b64encode(out["pcm"]).decode(),
        "sample_rate": out["sample_rate"],
        "gpu_seconds": out["generate_seconds"],
        "device": out["device"],
        "cold": out["cold"],
    }


@app.get("/health")
def health() -> dict:
    return {"ok": True, "devices": chatterbox_impl.available_devices()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
