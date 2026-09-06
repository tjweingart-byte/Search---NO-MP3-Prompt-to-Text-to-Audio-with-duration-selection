"""The streaming-PCM endpoint, exercised without a model or a GPU.

`chatterbox_impl.synthesise` is stubbed, so what is under test is the delivery
contract - headers, framing, and the honesty of the labelling - and not the
model. That is the correct seam: the endpoint's whole claim is about delivery.
"""
from __future__ import annotations

import base64
import importlib
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient                        # noqa: E402

RATE = 24000
PCM = b"\x01\x02" * (RATE * 2)          # two seconds of 16-bit mono


@pytest.fixture
def client(monkeypatch):
    from experiments.adapters import chatterbox_impl

    def fake_synthesise(text, device=None, warmup=False, inference_mode=True):
        return {"pcm": PCM, "sample_rate": RATE, "audio_seconds": 2.0,
                "generate_seconds": 0.25, "realtime_factor": 8.0,
                "device": "test", "cold": False, "gpu_cost": 0.0}

    monkeypatch.setattr(chatterbox_impl, "synthesise", fake_synthesise)
    monkeypatch.setattr(chatterbox_impl, "load_model", lambda d: (object(), 0.0))
    monkeypatch.setattr(chatterbox_impl, "warm_up", lambda m, d: None)
    monkeypatch.setattr(chatterbox_impl, "resolve_device", lambda d: ("test", True))

    module = importlib.import_module("experiments.adapters.chatterbox_server_example")
    return TestClient(module.app)


def test_streaming_endpoint_returns_the_same_audio_as_the_json_one(client):
    """A different wrapper must not become a different waveform."""
    streamed = client.post("/synthesise/stream", json={"text": "hello"}).content
    encoded = client.post("/synthesise", json={"text": "hello"}).json()
    assert streamed == base64.b64decode(encoded["pcm_base64"]) == PCM


def test_the_client_learns_the_rate_before_any_audio_arrives(client):
    """Otherwise it cannot start playing what it is receiving."""
    reply = client.post("/synthesise/stream", json={"text": "hello"})
    assert reply.headers["x-sample-rate"] == str(RATE)
    assert reply.headers["content-type"].startswith("application/octet-stream")


def test_the_endpoint_reports_its_own_generate_time(client):
    """So a client can subtract model latency from delivery latency."""
    reply = client.post("/synthesise/stream", json={"text": "hello"})
    assert float(reply.headers["x-generate-seconds"]) == pytest.approx(0.25)


def test_the_endpoint_does_not_claim_to_be_model_streaming(client):
    """The one thing this must never imply. generate() has no yield in it."""
    reply = client.post("/synthesise/stream", json={"text": "hello"})
    assert reply.headers["x-streaming-kind"] == "post-generation-delivery-only"


def test_the_probe_reads_the_generate_header_and_splits_the_wait(client):
    """End to end: the instrument gets model and delivery apart on real HTTP."""
    from experiments import chatterbox_probe as probe

    reply = client.post("/synthesise/stream", json={"text": "hello"})
    # The TestClient is in-process, so this asserts the parsing contract the
    # probe depends on rather than a wall-clock number.
    assert probe.playable_bytes(int(reply.headers["x-sample-rate"])) == 9600
    assert "x-generate-seconds" in reply.headers
