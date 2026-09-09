"""An episode reaches the ledger, and only an operator can read it back.

Two separate claims, and the second is the one worth being careful about: the
usage report is every listener's history of what they asked for and what it
cost. It is the most sensitive thing this app stores after the password
hashes, and unlike those it is meant to be *read*.
"""
from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as appmod  # noqa: E402
import metering  # noqa: E402
import pipeline as pipeline_mod  # noqa: E402
from tests.test_pipeline import FakeGenerator  # noqa: E402
from tts import DebugEngine  # noqa: E402


@pytest.fixture
def metered(tmp_path, monkeypatch):
    """A client writing to a ledger of its own."""
    store = metering.MeterStore(str(tmp_path / "metering.db"))
    monkeypatch.setattr(appmod, "METER", store)
    monkeypatch.setattr(appmod, "SCRIPT_CACHE", None)
    monkeypatch.setattr(appmod, "_rate_limit", lambda request: None)
    monkeypatch.setattr(appmod, "_read_limit", lambda request: None)
    monkeypatch.setattr(appmod, "DEMO_MODE", False)
    monkeypatch.setattr(
        appmod, "_make_pipeline",
        lambda voice=None: pipeline_mod.PodcastPipeline(
            generator=FakeGenerator(), engine=DebugEngine(), cache=None, voice=voice))
    return TestClient(appmod.app), store


# --- the episode is recorded ---------------------------------------------

def test_an_episode_lands_in_the_ledger(metered):
    client, store = metered
    assert client.get("/api/audio?q=anything&minutes=1&fmt=pcm").status_code == 200
    rows = store.rows()
    assert len(rows) == 1, "an episode played and nothing was billable for it"
    assert rows[0]["audio_seconds"] > 0
    assert rows[0]["minutes"] == 1


def test_the_listener_comes_from_the_cookie_and_never_from_a_parameter(metered):
    """The settled rule (CLAUDE.md), and metering is the most tempting place to
    break it: `?user=` is right there and the ledger wants an id. Anyone who
    could set it could bill their usage to somebody else."""
    client, store = metered
    client.get("/api/audio?q=anything&minutes=1&fmt=pcm&user=someone-elses-id")
    recorded = store.rows()[0]["user_id"]
    assert recorded and recorded != "someone-elses-id"


def test_the_surface_is_recorded_so_explore_can_be_told_from_search(metered):
    """Explore replays and never writes a script, so an Explore-heavy listener
    costs a fraction of a search-heavy one. A blended number hides that."""
    client, store = metered
    client.get("/api/audio?q=one&minutes=1&fmt=pcm")
    client.get("/api/audio?q=two&minutes=1&fmt=pcm&topic_id=chip-supply")
    assert [r["surface"] for r in store.rows()] == ["search", "myfam"]


def test_a_script_request_is_metered_too(metered, monkeypatch):
    """No audio, the same money. Left out, a probe hammering this endpoint
    would be the one kind of spend the ledger could not see."""
    client, store = metered
    # This endpoint builds its own generator rather than going through
    # _make_pipeline, so it needs pointing at the fake separately.
    monkeypatch.setattr(appmod, "ScriptGenerator", FakeGenerator)
    assert client.post("/api/script", json={"query": "anything", "minutes": 1}).status_code == 200
    assert [r["surface"] for r in store.rows()] == ["script"]


def test_a_broken_ledger_does_not_break_the_episode(metered, monkeypatch):
    """Metering is bookkeeping. It must never be able to cost a listener their
    audio - the whole product is the audio."""
    client, store = metered
    monkeypatch.setattr(store, "record",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no disk")))
    assert client.get("/api/audio?q=anything&minutes=1&fmt=pcm").status_code == 200


# --- and only an operator can read it back --------------------------------

def test_the_report_does_not_exist_without_a_configured_token(metered, monkeypatch):
    """404, not 401. An unconfigured deployment should not advertise that it
    has a billing endpoint at all."""
    client, _ = metered
    monkeypatch.setattr(appmod, "ADMIN_TOKEN", "")
    assert client.get("/api/usage").status_code == 404


def test_a_listener_cannot_read_the_report(metered, monkeypatch):
    client, _ = metered
    monkeypatch.setattr(appmod, "ADMIN_TOKEN", "the-real-token")
    assert client.get("/api/usage").status_code == 404
    assert client.get("/api/usage",
                      headers={"X-Admin-Token": "guess"}).status_code == 404


def test_an_operator_can(metered, monkeypatch):
    client, store = metered
    monkeypatch.setattr(appmod, "ADMIN_TOKEN", "the-real-token")
    client.get("/api/audio?q=anything&minutes=1&fmt=pcm")
    res = client.get("/api/usage", headers={"X-Admin-Token": "the-real-token"})
    assert res.status_code == 200
    body = res.json()
    assert body["totals"]["episodes"] == 1
    assert "per_listener" in body and "by_plan" in body


def test_a_bearer_token_works_too(metered, monkeypatch):
    client, _ = metered
    monkeypatch.setattr(appmod, "ADMIN_TOKEN", "the-real-token")
    res = client.get("/api/usage",
                     headers={"Authorization": "Bearer the-real-token"})
    assert res.status_code == 200


def test_the_report_is_never_paced_by_the_generation_limiter(metered, monkeypatch):
    """It spends no model call, and an operator pulling a report should not be
    competing with listeners for the generation budget."""
    client, _ = metered
    monkeypatch.setattr(appmod, "ADMIN_TOKEN", "t")

    def explode(_request):
        raise AssertionError("the report went through the generation limiter")

    monkeypatch.setattr(appmod, "_rate_limit", explode)
    assert client.get("/api/usage", headers={"X-Admin-Token": "t"}).status_code == 200


def test_a_failed_episode_is_still_recorded(metered, monkeypatch):
    """A failure is not a refund.

    Research may already have been billed by the time writing dies, and a retry
    loop against a broken key would otherwise be the cheapest thing in the
    ledger while being the most expensive thing on the invoice.
    """
    client, store = metered

    class DiesAfterResearch:
        async def stream_sentences(self, plan, notes=None):
            if notes is not None:
                notes.usage.add_research(1, 0.005)   # Exa has billed
            raise RuntimeError("the model call failed")
            yield ""  # pragma: no cover

        async def top_up(self, plan, spoken_so_far, words_needed, notes=None):
            return
            yield ""  # pragma: no cover

    monkeypatch.setattr(
        appmod, "_make_pipeline",
        lambda voice=None: pipeline_mod.PodcastPipeline(
            generator=DiesAfterResearch(), engine=DebugEngine(), cache=None))
    assert client.get("/api/audio?q=anything&minutes=1&fmt=pcm").status_code == 502
    rows = store.rows()
    assert len(rows) == 1, "money was spent and nothing recorded it"
    assert rows[0]["exa_cost"] == pytest.approx(0.005)
