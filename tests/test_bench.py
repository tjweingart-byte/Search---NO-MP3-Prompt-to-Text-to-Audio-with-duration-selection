"""The voice bench: search and playback, and nothing that can lie to you.

Its whole job is to answer "how does *this* voice sound on FAM's own prose".
Two ways that gets quietly answered wrong, both pinned here:

* substituting a different voice when the requested one is not present, which
  makes every judgement made on the bench worthless;
* a WAV header that does not describe the audio behind it.
"""
from __future__ import annotations

import os
import struct
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bench_app  # noqa: E402


@pytest.fixture
def client():
    return TestClient(bench_app.app)


# --- the bench is only search and playback ------------------------------

def test_it_serves_only_the_voice_endpoints():
    """No myFAM, no mixes, no explore, no profile, no echoes. The point of the
    branch is a small surface, and a route added here is the surface growing
    back."""
    paths = {r.path for r in bench_app.app.routes if hasattr(r, "path")}
    assert {"/", "/api/health", "/api/voices", "/api/speak", "/api/audio"} <= paths
    for absent in ("/api/myfam", "/api/mixes", "/api/explore", "/api/profile",
                   "/api/echo", "/api/event", "/api/godeeper", "/api/attach"):
        assert absent not in paths, f"{absent} crept back onto the bench"


def test_the_main_app_is_untouched_so_the_voice_work_merges_back():
    """This branch adds files; it does not delete the product. A voice found
    here has to be a clean diff over shared files to be worth anything."""
    import app as main_app

    paths = {r.path for r in main_app.app.routes if hasattr(r, "path")}
    assert "/api/myfam" in paths and "/api/explore" in paths


# --- a voice is never quietly swapped -----------------------------------

@pytest.mark.parametrize("endpoint, params", [
    ("/api/speak", {"text": "Hello there."}),
    ("/api/audio", {"q": "what is the nasdaq", "minutes": 1}),
])
def test_an_unknown_voice_is_refused_rather_than_substituted(client, endpoint, params):
    """engine_for_voice() falls back, which is right in the product and wrong
    here: answering with a different voice than the one asked for turns the
    bench into a machine for reaching confident wrong conclusions."""
    response = client.get(endpoint, params={**params, "voice": "nosuch:voice"})
    assert response.status_code == 400
    body = response.json()["error"]
    assert "nosuch:voice" in body
    assert "rather than playing a different voice" in body


def test_a_known_voice_is_accepted(client):
    listed = client.get("/api/voices").json()["voices"]
    assert listed, "the machine reports no voices at all"
    response = client.get("/api/speak", params={"text": "Hello.", "voice": listed[0]["id"]})
    assert response.status_code == 200


# --- the audio describes itself honestly --------------------------------

def test_the_wav_header_states_the_real_length(client):
    """The streaming header deliberately lies about length, because the main
    app does not know it yet. This endpoint does know, and using the streaming
    one made a 9.6 second clip announce itself as 27 hours."""
    response = client.get("/api/speak", params={"text": "One. Two. Three."})
    assert response.status_code == 200
    body = response.content
    assert body[:4] == b"RIFF" and body[8:12] == b"WAVE"

    (riff_size,) = struct.unpack("<I", body[4:8])
    (data_size,) = struct.unpack("<I", body[40:44])
    assert data_size == len(body) - 44, "the header does not match the audio behind it"
    assert riff_size == len(body) - 8


def test_the_timings_are_reported_so_a_slow_voice_is_visible(client):
    """A voice that sounds right and cannot keep ahead of playback cannot
    ship, and that is invisible unless it is measured."""
    response = client.get("/api/speak", params={"text": "One sentence. And another."})
    for header in ("X-Sample-Rate", "X-Engine", "X-Synth-Ms",
                   "X-Audio-Ms", "X-Realtime-Factor"):
        assert header in response.headers, f"{header} is not reported"
    assert float(response.headers["X-Realtime-Factor"]) > 0


def test_the_rate_header_matches_the_wav_header(client):
    """The page reads the header to build its buffer; a disagreement plays
    everything at the wrong pitch."""
    response = client.get("/api/speak", params={"text": "Hello."})
    (in_wav,) = struct.unpack("<I", response.content[24:28])
    assert in_wav == int(response.headers["X-Sample-Rate"])


# --- refusals are refusals ----------------------------------------------

def test_empty_text_is_an_error_not_silence(client):
    assert client.get("/api/speak", params={"text": "   "}).status_code == 400


def test_an_empty_question_is_an_error(client):
    assert client.get("/api/audio", params={"q": "  "}).status_code == 400


def test_text_is_split_into_sentences_the_way_the_pipeline_does():
    """A voice judged on one long blob is not judged the way the product uses
    it - the pipeline hands the engine one sentence at a time, and the joins
    are most of what makes a voice sound stitched."""
    assert bench_app._sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]
    assert bench_app._sentences("no punctuation here") == ["no punctuation here"]
    assert bench_app._sentences("   ") == ["   "]
