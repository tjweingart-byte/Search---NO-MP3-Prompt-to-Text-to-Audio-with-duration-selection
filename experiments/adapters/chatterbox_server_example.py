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

Two endpoints, and the difference between them is a *delivery* experiment, not
a model one:

* `POST /synthesise` - the original contract. One JSON object with base64 PCM
  inside it. Nothing decodes until the closing brace arrives, so the listener
  waits for the whole chunk however fast the model was.
* `POST /synthesise/stream` - the same waveform, written to the socket as raw
  16-bit PCM in small pieces, with the sample rate in a header so the client
  knows what it is receiving before any audio arrives.

**Neither is model streaming, and the second one must not be described as
such.** `ChatterboxTurboTTS.generate` returns one completed waveform; there is
no `yield` anywhere in the package. `/synthesise/stream` therefore begins
writing only *after* generation has finished. What it removes is the base64
and JSON assembly barrier on top of that - real, measurable, and strictly
smaller than the model time it sits behind.

Neither endpoint is production FAM. Nothing in the app imports this file.
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


#: Bytes written per socket write in the streaming endpoint. Small enough that
#: the client's first-byte mark is a real observation, large enough not to turn
#: the measurement into a syscall benchmark.
STREAM_CHUNK_BYTES = 8192


@app.post("/synthesise/stream")
def synthesise_stream(request: Request):
    """The same audio, delivered as raw PCM instead of base64 inside JSON.

    Honest about what it is: generation still completes first. The response
    begins at the moment the waveform exists, and the client can play the first
    piece without waiting for the last. The saving is the encode-and-assemble
    step, not any part of inference.

    `X-Generate-Seconds` carries the fenced generate time so the client can
    subtract model latency from delivery latency without a second request.
    """
    from fastapi.responses import StreamingResponse

    out = chatterbox_impl.synthesise(request.text, warmup=True, inference_mode=True)
    pcm = out["pcm"]

    def pieces():
        for index in range(0, len(pcm), STREAM_CHUNK_BYTES):
            yield pcm[index:index + STREAM_CHUNK_BYTES]

    return StreamingResponse(
        pieces(),
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": str(out["sample_rate"]),
            "X-Generate-Seconds": f"{out['generate_seconds']:.6f}",
            "X-Device": str(out["device"]),
            "X-Audio-Seconds": f"{out['audio_seconds']:.6f}",
            # Says plainly what this is, to anyone who curls it.
            "X-Streaming-Kind": "post-generation-delivery-only",
        },
    )


@app.get("/health")
def health() -> dict:
    return {"ok": True, "devices": chatterbox_impl.available_devices()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
