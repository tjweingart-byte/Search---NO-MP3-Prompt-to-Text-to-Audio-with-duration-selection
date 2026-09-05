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

app = FastAPI(title="Chatterbox for FAM")


class Request(BaseModel):
    text: str
    #: A hint only. The response reports the model's own rate, which wins.
    sample_rate: int | None = None


@app.on_event("startup")
def warm() -> None:
    """Load the model before the first request, and say what it landed on."""
    device, _ = chatterbox_impl.resolve_device(None)
    _, seconds = chatterbox_impl.load_model(device)
    print(f"chatterbox ready on {device} in {seconds:.1f}s", flush=True)


@app.post("/synthesise")
def synthesise(request: Request) -> dict:
    out = chatterbox_impl.synthesise(request.text)
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
