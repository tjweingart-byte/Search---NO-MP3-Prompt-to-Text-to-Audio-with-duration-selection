"""Echoes, identity, and mix visibility.

The load-bearing property of an echo is that it costs nothing: it is a row
pointing at a query whose script already exists. If echoing ever generated,
the social layer would be the most expensive thing in the product rather than
the cheapest.
"""
from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as appmod  # noqa: E402
import mixes as M  # noqa: E402
import social as S  # noqa: E402
from cache import SqliteScriptCache  # noqa: E402


@pytest.fixture
def store(tmp_path):
    return S.SocialStore(str(tmp_path / "social.db"))


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(appmod, "_rate_limit", lambda request: None)
    monkeypatch.setattr(appmod, "SOCIAL", S.SocialStore(str(tmp_path / "s.db")))
    monkeypatch.setattr(appmod, "MIXES", M.MixStore(str(tmp_path / "m.db")))
    monkeypatch.setattr(appmod, "SCRIPT_CACHE", SqliteScriptCache(str(tmp_path / "c.db")))
    return TestClient(appmod.app)


# --- identity -------------------------------------------------------------


def test_a_listener_can_name_themselves(store):
    me = store.set_person("u1", "  Ian   Solomon ", "@IanSolomon")
    assert me["name"] == "Ian Solomon"
    assert me["handle"] == "iansolomon", "handles are lowercased and stripped of @"


def test_a_handle_belongs_to_one_person(store):
    store.set_person("u1", "Ian", "ian")
    with pytest.raises(S.SocialError):
        store.set_person("u2", "Imposter", "ian")
    # ...but keeping your own handle is not a clash with yourself.
    assert store.set_person("u1", "Ian S", "ian")["name"] == "Ian S"


def test_a_bad_handle_is_refused(store):
    for bad in ("", "a", "has spaces", "way-too-long-a-handle-for-anyone-here"):
        with pytest.raises(S.SocialError):
            store.set_person("u1", "Ian", bad)


def test_a_nameless_listener_is_not_an_error(store):
    """Everyone starts anonymous; the profile shows that honestly."""
    assert store.person("nobody")["name"] == ""


# --- echoes ---------------------------------------------------------------


def test_echoing_records_the_episode_not_a_new_one(store):
    echo = store.echo("u1", "why chips are concentrated", "Who Makes the Chips", 6, "a thread")
    assert echo.query == "why chips are concentrated"
    assert echo.minutes == 6 and echo.thread == "a thread"


def test_echoing_twice_is_one_echo(store):
    """The intent is 'send this', not 'send this twice'."""
    store.echo("u1", "a question", "A Question", 3)
    store.echo("u1", "a question", "A Question", 3)
    assert len(store.echoes_by("u1")) == 1


def test_two_people_can_echo_the_same_episode(store):
    store.echo("u1", "a question", "A Question", 3)
    store.echo("u2", "a question", "A Question", 3)
    assert len(store.echoes_by("u1")) == 1 and len(store.echoes_by("u2")) == 1


def test_an_echo_can_be_taken_back(store):
    store.echo("u1", "a question", "A Question", 3)
    assert store.unecho("u1", "a question", 3) is True
    assert store.echoes_by("u1") == []
    assert store.has_echoed("u1", "a question", 3) is False


def test_the_same_question_at_a_different_length_is_a_different_episode(store):
    store.echo("u1", "a question", "A Question", 3)
    store.echo("u1", "a question", "A Question", 6)
    assert len(store.echoes_by("u1")) == 2


def test_echo_labels_carry_the_name(store):
    store.set_person("u1", "Rachel Solomon", "rachelsolomon")
    store.echo("u1", "a question", "A Question", 3)
    label = store.recent_echoes()[("a question", 3)]
    assert label["by"] == "Rachel Solomon" and label["handle"] == "rachelsolomon"


def test_your_own_echoes_are_not_labelled_back_to_you(store):
    store.set_person("u1", "Ian", "ian")
    store.echo("u1", "a question", "A Question", 3)
    assert store.recent_echoes(exclude_user="u1") == {}


def test_an_echo_from_someone_with_no_name_still_reads(store):
    store.echo("anon", "a question", "A Question", 3)
    assert store.recent_echoes()[("a question", 3)]["by"] == "Someone"


# --- mix visibility -------------------------------------------------------


def test_a_mix_is_private_until_it_is_not(tmp_path):
    mixes = M.MixStore(str(tmp_path / "m.db"))
    mix = mixes.create("u", "Morning", ["ai-agents"])
    assert mix.public is False, "a routine is personal until someone says otherwise"
    assert mixes.public_for_user("u") == []
    mixes.update("u", mix.id, public=True)
    assert [m.name for m in mixes.public_for_user("u")] == ["Morning"]


def test_visibility_survives_a_restart(tmp_path):
    path = str(tmp_path / "m.db")
    mix = M.MixStore(path).create("u", "Morning", [])
    M.MixStore(path).update("u", mix.id, public=True)
    assert M.MixStore(path).get("u", mix.id).public is True


def test_renaming_a_mix_does_not_publish_it(tmp_path):
    mixes = M.MixStore(str(tmp_path / "m.db"))
    mix = mixes.create("u", "Morning", [])
    assert mixes.update("u", mix.id, name="Early").public is False


# --- the API --------------------------------------------------------------


def test_echoing_over_http_and_seeing_it_on_the_profile(client):
    client.post("/api/me", json={"user": "u1", "name": "Ian Solomon", "handle": "iansolomon"})
    client.post("/api/echo", json={"user": "u1", "query": "why chips are concentrated",
                                   "title": "Who Makes the Chips", "minutes": 6})
    body = client.get("/api/profile?user=u1").json()
    assert body["name"] == "Ian Solomon" and body["handle"] == "iansolomon"
    assert [e["title"] for e in body["echoes"]] == ["Who Makes the Chips"]
    assert body["echo_count"] == 1


def test_only_public_mixes_reach_the_profile(client):
    made = client.post("/api/mixes", json={"user": "u1", "name": "Morning",
                                            "topic_ids": ["ai-agents"]}).json()
    assert client.get("/api/profile?user=u1").json()["mixes"] == []
    client.patch(f"/api/mixes/{made['id']}", json={"user": "u1", "public": True})
    assert [m["name"] for m in client.get("/api/profile?user=u1").json()["mixes"]] == ["Morning"]


def test_an_echo_labels_the_explore_card(client):
    appmod.SCRIPT_CACHE.put("k", ["A sentence."], 600, "why volcanoes erupt", "", 3)
    client.post("/api/me", json={"user": "u2", "name": "Rachel", "handle": "rachel"})
    client.post("/api/echo", json={"user": "u2", "query": "why volcanoes erupt",
                                   "title": "Why Volcanoes Erupt", "minutes": 3})
    card = client.get("/api/explore?user=u1").json()["episodes"][0]
    assert card["echoed_by"] == "Rachel"
    # And it is not labelled back to the person who sent it.
    assert client.get("/api/explore?user=u2").json()["episodes"][0]["echoed_by"] == ""


def test_a_taken_handle_is_a_readable_refusal(client):
    client.post("/api/me", json={"user": "u1", "name": "Ian", "handle": "ian"})
    res = client.post("/api/me", json={"user": "u2", "name": "Other", "handle": "ian"})
    assert res.status_code == 400 and "taken" in res.json()["error"]
