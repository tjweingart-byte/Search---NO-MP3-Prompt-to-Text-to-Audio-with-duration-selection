"""The remote voice, checked on a machine with no GPU and no network.

Every test here answers one of the four questions that decide whether moving
the card off the app's host was safe:

* does the **switch** actually switch, and does it stay off unless asked
* does a failure **fail**, rather than becoming a different voice or silence
* does the **wire contract** survive a worker that answers wrongly
* can the transport be changed **without touching the engine**

The worker half is exercised too, with `ChatterboxEngine` stubbed: the point is
the envelope and the refusals, which are the parts that run on every request
and cannot be tested on a card this machine does not have.
"""
from __future__ import annotations

import asyncio
import base64
import dataclasses
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
import remote_voice  # noqa: E402
import tts  # noqa: E402
from remote_voice import RemoteChatterboxEngine, RemoteVoiceError  # noqa: E402


PCM = b"\x01\x02" * 240  # 480 bytes: a whole number of 16-bit samples


def run(coro):
    """The suite's convention for an async call.

    There is no pytest-asyncio here, and one file needing a plugin nothing else
    needs is a plugin that gets forgotten on the next fresh checkout.
    """
    return asyncio.run(coro)


def configure(monkeypatch, **overrides):
    """Point `settings` at a fake endpoint, without touching the environment.

    `Settings` is frozen, so this replaces it rather than mutating it - which
    also means `__post_init__` re-validates every combination a test sets up,
    and a test that configures something the app would refuse fails here rather
    than passing against a state production cannot reach.
    """
    defaults = {
        "voice_backend": "remote",
        "remote_voice_transport": "runpod",
        "runpod_endpoint_id": "endpoint-1",
        "runpod_api_key": "rp-key",
        "runpod_base_url": "https://api.runpod.ai/v2",
        "remote_voice_url": "",
        "remote_voice_token": "",
        "remote_voice_sample_rate": 24000,
        "remote_voice_timeout": 5.0,
        "remote_voice_connect_timeout": 1.0,
        "remote_voice_concurrency": 2,
        "remote_voice_id": "",
        "remote_voice_wake_interval": 60.0,
    }
    defaults.update(overrides)
    patched = dataclasses.replace(config.settings, **defaults)
    # Each of these bound `settings` at import, so each has to be told.
    for module in (config, remote_voice, tts):
        monkeypatch.setattr(module, "settings", patched)
    # `credentials.active` outranks the settings snapshot; keep it quiet so the
    # test is describing the settings it just set.
    monkeypatch.setattr(
        remote_voice, "_credential", lambda name, configured: configured)
    RemoteChatterboxEngine._client = None
    RemoteChatterboxEngine._gate = None
    RemoteChatterboxEngine._gate_size = 0
    RemoteChatterboxEngine._woken_at = 0.0
    RemoteChatterboxEngine._reachability = remote_voice.Reachability()


class FakeResponse:
    def __init__(self, payload, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text or str(payload)

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeClient:
    """Stands in for httpx.AsyncClient, and records what was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.posts = []
        self.gets = []

    async def post(self, url, json=None, headers=None, timeout=None):
        self.posts.append({"url": url, "json": json, "headers": headers or {}})
        return self._next()

    async def get(self, url, headers=None):
        self.gets.append({"url": url, "headers": headers or {}})
        return self._next()

    def _next(self):
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def aclose(self):
        pass


def install(monkeypatch, *replies):
    client = FakeClient(replies)
    monkeypatch.setattr(
        RemoteChatterboxEngine, "_http", classmethod(lambda cls, cfg: client))
    return client


def completed(audio=PCM, **overrides):
    output = {
        "audio": base64.b64encode(audio).decode("ascii"),
        "sample_rate": 24000,
        "format": "pcm_s16le",
        "engine": "chatterbox",
    }
    output.update(overrides)
    return FakeResponse({"status": "COMPLETED", "output": output})


def speak(text="Hello there."):
    return run(RemoteChatterboxEngine().synth(text, 150))


# --- the switch -------------------------------------------------------------

def test_the_remote_voice_is_not_reachable_without_being_asked_for():
    """PROBLEMS.md §61's first guard, restated as a test.

    Registering a second engine used to be enough to make it what every
    listener got. The default backend must be the local card, and the remote
    one must take an explicit decision - not an installed package, not a
    variable set somewhere else, not the absence of a GPU.
    """
    assert config.settings.voice_backend == "chatterbox"
    assert [cls.name for cls in tts.production_engines()] == ["chatterbox"]


def test_naming_the_remote_backend_switches_the_production_slot(monkeypatch):
    configure(monkeypatch)
    assert [cls.name for cls in tts.production_engines()] == ["remote"]


def test_the_switch_does_not_offer_both_voices_at_once(monkeypatch):
    """One slot, always. Two available engines is two things a listener could
    be hearing, and two answers to "why does it sound like that"."""
    configure(monkeypatch)
    assert len(tts.production_engines()) == 1


def test_an_unknown_backend_is_refused_at_construction():
    with pytest.raises(ValueError, match="VOICE_BACKEND"):
        dataclasses.replace(config.settings, voice_backend="runpod")


def test_an_unknown_transport_is_refused_at_construction():
    with pytest.raises(ValueError, match="REMOTE_VOICE_TRANSPORT"):
        dataclasses.replace(config.settings, voice_backend="remote",
                            remote_voice_transport="grpc")


def test_zero_concurrency_is_refused_rather_than_deadlocking():
    with pytest.raises(ValueError, match="REMOTE_VOICE_CONCURRENCY"):
        dataclasses.replace(config.settings, voice_backend="remote",
                            remote_voice_concurrency=0)


# --- configuration is not reachability --------------------------------------

def test_a_half_configured_endpoint_reports_why(monkeypatch):
    configure(monkeypatch, runpod_endpoint_id="")
    ok, detail = RemoteChatterboxEngine.diagnose()
    assert not ok
    assert "RUNPOD_ENDPOINT_ID" in detail


def test_a_missing_key_is_named_rather_than_tried(monkeypatch):
    configure(monkeypatch, runpod_api_key="")
    ok, detail = RemoteChatterboxEngine.diagnose()
    assert not ok and "RUNPOD_API_KEY" in detail


def test_availability_never_reaches_the_network(monkeypatch):
    """`/api/health` calls this. A health check that makes a billed
    third-party request is a health check nobody can afford to poll."""
    configure(monkeypatch)
    exploded = install(monkeypatch, AssertionError("availability made a call"))
    assert RemoteChatterboxEngine.available() is True
    assert exploded.posts == [] and exploded.gets == []


def test_configured_is_reported_separately_from_reachable(monkeypatch):
    """§52: "a credential is set" is not "the credential works". The report
    must not let one stand in for the other."""
    configure(monkeypatch)
    report = remote_voice.report()
    assert report["configured"] is True
    assert report["reachable"]["state"] == "unknown"


def test_the_report_never_carries_the_credential(monkeypatch):
    configure(monkeypatch, runpod_api_key="rp-secret-value")
    assert "rp-secret-value" not in str(remote_voice.report())


# --- the happy path ---------------------------------------------------------

def test_a_completed_job_becomes_pcm(monkeypatch):
    configure(monkeypatch)
    install(monkeypatch, completed())
    assert speak() == PCM


def test_the_request_carries_the_contract(monkeypatch):
    configure(monkeypatch)
    client = install(monkeypatch, completed())
    speak()
    sent = client.posts[0]
    assert sent["url"] == "https://api.runpod.ai/v2/endpoint-1/runsync"
    assert sent["json"]["input"]["text"] == "Hello there."
    assert sent["json"]["input"]["format"] == "pcm_s16le"
    assert sent["json"]["input"]["sample_rate"] == 24000
    assert sent["headers"]["Authorization"] == "Bearer rp-key"


def test_a_real_call_is_what_makes_it_reachable(monkeypatch):
    configure(monkeypatch)
    install(monkeypatch, completed())
    speak()
    assert remote_voice.report()["reachable"]["state"] == "ok"


def test_the_header_rate_is_known_before_any_call(monkeypatch):
    """`app.py` writes X-Sample-Rate before the body runs, so the engine
    cannot wait to be told what rate it speaks at."""
    configure(monkeypatch, remote_voice_sample_rate=24000)
    install(monkeypatch, AssertionError("sample_rate made a call"))
    assert RemoteChatterboxEngine().sample_rate == 24000


# --- a cold worker ----------------------------------------------------------

def test_a_queued_job_is_polled_rather_than_failed(monkeypatch):
    """A cold worker routinely exceeds /runsync's window. Treating the job id
    as a failure would turn every cold start into a broken episode."""
    configure(monkeypatch)
    client = install(
        monkeypatch,
        FakeResponse({"status": "IN_QUEUE", "id": "job-1"}),
        FakeResponse({"status": "IN_PROGRESS", "id": "job-1"}),
        completed(),
    )
    assert speak() == PCM
    assert [g["url"] for g in client.gets] == [
        "https://api.runpod.ai/v2/endpoint-1/status/job-1"] * 2


def test_a_job_that_never_finishes_says_so_and_says_what_to_do(monkeypatch):
    configure(monkeypatch, remote_voice_timeout=0.0)
    install(monkeypatch, FakeResponse({"status": "IN_QUEUE", "id": "job-1"}))
    with pytest.raises(RemoteVoiceError, match="REMOTE_VOICE_TIMEOUT"):
        speak()


def test_wake_does_not_synthesise_and_cannot_raise(monkeypatch):
    """The wake is worth several seconds when it lands, and must be worth
    zero when it does not."""
    configure(monkeypatch)
    client = install(monkeypatch, ConnectionError("no route to host"))
    run(RemoteChatterboxEngine.wake())
    assert client.posts[0]["url"].endswith("/run")
    assert client.posts[0]["json"]["input"]["warm"] is True


def test_wake_does_not_stampede(monkeypatch):
    configure(monkeypatch)
    client = install(monkeypatch, FakeResponse({"id": "job-1"}))
    run(RemoteChatterboxEngine.wake())
    run(RemoteChatterboxEngine.wake())
    assert len(client.posts) == 1, "a booting worker does not boot faster twice"


def test_an_always_on_pod_has_nothing_to_wake(monkeypatch):
    configure(monkeypatch, remote_voice_transport="http",
              remote_voice_url="https://pod.example")
    client = install(monkeypatch, AssertionError("woke an always-on pod"))
    run(RemoteChatterboxEngine.wake())
    assert client.posts == []


# --- failure is failure -----------------------------------------------------

def test_a_failed_job_raises_with_the_workers_reason(monkeypatch):
    configure(monkeypatch)
    install(monkeypatch,
            FakeResponse({"status": "FAILED", "error": "no rights record"}))
    with pytest.raises(RemoteVoiceError, match="no rights record"):
        speak()


def test_an_http_error_names_the_status(monkeypatch):
    configure(monkeypatch)
    install(monkeypatch, FakeResponse({}, status_code=401, text="unauthorized"))
    with pytest.raises(RemoteVoiceError, match="401"):
        speak()


def test_a_network_failure_never_becomes_another_voice(monkeypatch):
    """§61's second guard. Substituting a working engine for a broken one
    means judging one backend by another's output - and an operator debugging
    a GPU that was never being asked to speak."""
    configure(monkeypatch)
    install(monkeypatch, ConnectionError("connection reset"))
    with pytest.raises(RemoteVoiceError, match="connection reset"):
        speak()


def test_empty_audio_is_refused_rather_than_played(monkeypatch):
    """A successful response carrying nothing is the silent empty episode
    this project has lost the most time to."""
    configure(monkeypatch)
    install(monkeypatch, FakeResponse(
        {"status": "COMPLETED", "output": {"audio": "", "sample_rate": 24000}}))
    with pytest.raises(RemoteVoiceError, match="no audio"):
        speak()


def test_a_worker_at_the_wrong_rate_is_refused_not_pitched(monkeypatch):
    """The header has already claimed a rate. Playing a different one is a
    failure the listener hears and nothing anywhere explains."""
    configure(monkeypatch)
    install(monkeypatch, completed(sample_rate=22050))
    with pytest.raises(RemoteVoiceError, match="REMOTE_VOICE_SAMPLE_RATE=22050"):
        speak()


def test_a_compressed_reply_is_refused_rather_than_decoded(monkeypatch):
    """"No MP3" is a rule about what reaches the listener. A worker offering
    one is a misconfiguration, not an invitation to add a decoder."""
    configure(monkeypatch)
    install(monkeypatch, completed(format="mp3"))
    with pytest.raises(RemoteVoiceError, match="does not transcode"):
        speak()


def test_a_half_sample_is_refused(monkeypatch):
    """An odd byte count shifts every sample by one byte and turns the rest
    of the episode into noise."""
    configure(monkeypatch)
    install(monkeypatch, completed(audio=b"\x01\x02\x03"))
    with pytest.raises(RemoteVoiceError, match="whole number"):
        speak()


def test_a_proxys_html_error_page_is_a_sentence_not_a_type_error(monkeypatch):
    configure(monkeypatch)
    install(monkeypatch,
            FakeResponse(ValueError("no json"), text="<html>502</html>"))
    with pytest.raises(RemoteVoiceError, match="did not return JSON"):
        speak()


# --- the transport is the only thing that changes ---------------------------

def test_the_http_transport_posts_the_payload_unwrapped(monkeypatch):
    """One worker image, two envelopes. Switching between a serverless
    endpoint and an always-on pod must not be a code change."""
    configure(monkeypatch, remote_voice_transport="http",
              remote_voice_url="https://pod.example", remote_voice_token="shh")
    client = install(monkeypatch, FakeResponse({
        "audio": base64.b64encode(PCM).decode("ascii"),
        "sample_rate": 24000, "format": "pcm_s16le"}))
    assert speak() == PCM
    sent = client.posts[0]
    assert sent["url"] == "https://pod.example/synth"
    assert sent["json"]["text"] == "Hello there.", "no RunPod input wrapper"
    assert sent["headers"]["Authorization"] == "Bearer shh"


def test_both_transports_return_identical_audio(monkeypatch):
    """The engine is one engine. If these ever differ, something other than
    the envelope is transport-specific and should not be."""
    configure(monkeypatch)
    install(monkeypatch, completed())
    from_runpod = speak()

    configure(monkeypatch, remote_voice_transport="http",
              remote_voice_url="https://pod.example")
    install(monkeypatch, FakeResponse({
        "audio": base64.b64encode(PCM).decode("ascii"),
        "sample_rate": 24000, "format": "pcm_s16le"}))
    assert speak() == from_runpod


def test_an_http_pod_needs_no_runpod_credential(monkeypatch):
    configure(monkeypatch, remote_voice_transport="http",
              remote_voice_url="https://pod.example",
              runpod_api_key="", runpod_endpoint_id="")
    assert RemoteChatterboxEngine.available() is True


# --- the address, which is the one thing only this half can get wrong -------
#
# Production hit this: `POST https://<pod>-8002.proxy.runpod.net/synth` ->
# 404, on a deployment where the research half had just started working. A 404
# is the failure that looks like the voice and is not: nothing was asked, and
# every earlier guard here (empty audio, wrong rate, odd byte count) is about a
# worker that *answered*. These are about being sure which of the two it was.


def http_pod(monkeypatch, url="https://pod.example", **overrides):
    configure(monkeypatch, remote_voice_transport="http", remote_voice_url=url,
              remote_voice_token="shh", **overrides)
    RemoteChatterboxEngine._found_route = None


def audio_reply():
    return FakeResponse({"audio": base64.b64encode(PCM).decode("ascii"),
                         "sample_rate": 24000, "format": "pcm_s16le"})


def not_found(text='{"detail":"Not Found"}'):
    return FakeResponse({"detail": "Not Found"}, status_code=404, text=text)


def test_a_url_that_already_names_the_route_is_not_given_a_second_one(monkeypatch):
    """The 404 that reads exactly like a missing route, and is a doubled path.

    `REMOTE_VOICE_URL` is documented as a base URL, and the URL an operator
    verified the pod with by hand is the one ending in /synth. Both are
    unambiguous, so both work."""
    http_pod(monkeypatch, url="https://pod.example/synth")
    client = install(monkeypatch, audio_reply())
    assert speak() == PCM
    assert client.posts[0]["url"] == "https://pod.example/synth"
    assert not client.gets, "a working address asks the worker nothing"


def test_the_happy_path_costs_no_extra_request(monkeypatch):
    http_pod(monkeypatch)
    client = install(monkeypatch, audio_reply())
    assert speak() == PCM
    assert [p["url"] for p in client.posts] == ["https://pod.example/synth"]
    assert not client.gets


def test_a_404_asks_the_worker_which_route_it_serves_and_uses_it(monkeypatch):
    """A worker whose route is not this app's name for it is a fixable
    deployment, not a broken episode - and the retry is the worker's own
    answer, never a guessed path."""
    http_pod(monkeypatch)
    client = install(
        monkeypatch,
        not_found(),
        FakeResponse({"paths": {"/health": {"get": {}},
                                "/v1/synthesise": {"post": {}}}}),
        audio_reply())
    assert speak() == PCM
    assert client.gets[0]["url"] == "https://pod.example/openapi.json"
    assert [p["url"] for p in client.posts] == [
        "https://pod.example/synth", "https://pod.example/v1/synthesise"]


def test_the_route_the_worker_named_is_not_asked_for_twice(monkeypatch):
    """Fifteen chunks an episode. One discovery, not fifteen."""
    http_pod(monkeypatch)
    client = install(
        monkeypatch,
        not_found(),
        FakeResponse({"paths": {"/v1/synthesise": {"post": {}}}}),
        audio_reply(),
        audio_reply())
    assert speak() == PCM
    assert speak() == PCM
    assert len(client.gets) == 1
    assert client.posts[-1]["url"] == "https://pod.example/v1/synthesise"


def test_a_404_everywhere_names_the_port_and_the_mode(monkeypatch):
    """The failure production actually saw, and the sentence it needed.

    Nothing answering `/openapi.json` either means the thing being called is
    not this worker at all - which is a proxied port with nothing behind it, or
    a pod running the serverless handler, and neither is visible from a 404 on
    its own."""
    http_pod(monkeypatch)
    install(monkeypatch, not_found(), not_found())
    with pytest.raises(RemoteVoiceError) as raised:
        speak()
    said = str(raised.value)
    assert "PORT" in said and "VOICE_WORKER_MODE=http" in said
    assert "openapi.json" in said


def test_the_probes_hang_off_the_origin_not_the_route(monkeypatch):
    """`.../synth/openapi.json` would answer nothing and blame the worker."""
    http_pod(monkeypatch, url="https://pod.example/synth")
    client = install(monkeypatch, not_found(), not_found())
    with pytest.raises(RemoteVoiceError):
        speak()
    assert client.gets[0]["url"] == "https://pod.example/openapi.json"


def test_a_worker_that_serves_no_post_route_is_not_guessed_at(monkeypatch):
    http_pod(monkeypatch)
    install(monkeypatch, not_found(),
            FakeResponse({"paths": {"/health": {"get": {}}}}))
    with pytest.raises(RemoteVoiceError, match="does not serve"):
        speak()


def test_a_404_never_becomes_a_different_voice(monkeypatch):
    """§61's second guard, at the one place a retry exists at all."""
    http_pod(monkeypatch)
    install(monkeypatch, not_found(), not_found())
    with pytest.raises(RemoteVoiceError):
        speak()
    assert RemoteChatterboxEngine.name == "remote"


# --- the worker half --------------------------------------------------------

class StubEngine:
    """Stands in for ChatterboxEngine, which needs a card this machine has not."""

    spoken: list = []
    sample_rate = 24000

    async def synth(self, text, wpm=0.0, voice=None):
        StubEngine.spoken.append(text)
        return PCM


@pytest.fixture
def worker(monkeypatch):
    from voice_worker import synth as module

    StubEngine.spoken = []
    monkeypatch.setattr(module, "preflight", lambda: (True, "stub"))
    monkeypatch.setattr(module, "ChatterboxEngine", StubEngine)
    return module


def test_the_worker_refuses_a_format_it_does_not_speak(worker):
    with pytest.raises(worker.WorkerError, match="pcm_s16le"):
        run(worker.synthesise({"text": "Hello.", "format": "mp3"}))


def test_the_worker_refuses_when_it_cannot_speak(worker, monkeypatch):
    """A node with no card must refuse the job, not accept and starve it."""
    monkeypatch.setattr(
        worker, "preflight",
        lambda: (False, "no GPU: Chatterbox on CPU is slower than speech"))
    with pytest.raises(worker.WorkerError, match="no GPU"):
        run(worker.synthesise({"text": "Hello."}))


def test_the_worker_refuses_a_runaway_payload(worker):
    with pytest.raises(worker.WorkerError, match="ceiling"):
        run(worker.synthesise({"text": "x" * (worker.MAX_CHARACTERS + 1)}))


def test_the_worker_round_trips_the_contract(worker):
    """The two halves are deployed separately and can be different versions
    of this repo, so the contract is checked from both ends."""
    reply = run(worker.synthesise({"text": "Hello.", "sample_rate": 24000}))

    assert reply["format"] == remote_voice.WIRE_FORMAT
    assert base64.b64decode(reply["audio"]) == PCM
    assert reply["sample_rate"] == 24000
    # And the engine accepts, unchanged, what the worker produced.
    config_for = remote_voice.RemoteConfig(
        transport="http", url="x", api_key="", sample_rate=24000,
        timeout=1.0, connect_timeout=1.0, concurrency=1, voice="")
    assert RemoteChatterboxEngine._pcm_from(reply, config_for) == PCM


def test_a_warm_job_loads_the_model_without_synthesising(worker):
    reply = run(worker.synthesise({"warm": True}))
    assert reply["ready"] is True and "audio" not in reply
    assert StubEngine.spoken == ["Ready."], "a wake pays the load, nothing more"
