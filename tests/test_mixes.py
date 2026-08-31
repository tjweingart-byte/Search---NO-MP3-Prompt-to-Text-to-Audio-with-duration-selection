"""playFAM mixes: named daily subscriptions to bank topics."""
from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as appmod  # noqa: E402
import mixes as M  # noqa: E402
import topics as T  # noqa: E402


@pytest.fixture
def store(tmp_path):
    return M.MixStore(str(tmp_path / "mixes.db"))


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(appmod, "_rate_limit", lambda request: None)
    monkeypatch.setattr(appmod, "MIXES", M.MixStore(str(tmp_path / "api.db")))
    return TestClient(appmod.app)


# --- the model ------------------------------------------------------------


def test_a_mix_holds_topics_not_audio(store):
    """The design decision: a mix is a standing subscription, not a recording.

    Storing topic ids means "at the gym" is fresh every morning. Storing audio
    would make it stale the moment it was saved - and break the no-files rule.
    """
    mix = store.create("u", "At the gym", ["training-load", "the-trade"])
    assert mix.topic_ids == ["training-load", "the-trade"]
    body = mix.as_dict()
    assert [t["id"] for t in body["topics"]] == ["training-load", "the-trade"]
    assert "audio" not in body and "script" not in body


def test_membership_is_restricted_to_the_shared_bank(store):
    """Free-text members would quietly undo the shared-script cost design."""
    with pytest.raises(M.MixError):
        store.create("u", "Mine", ["something-i-made-up"])


def test_an_unknown_topic_is_an_error_not_a_silent_drop(store):
    """A mix that quietly loses a topic looks like the app forgot."""
    with pytest.raises(M.MixError):
        store.create("u", "Mine", ["ai-agents", "nope"])
    assert store.list_for_user("u") == []


def test_a_mix_needs_a_name(store):
    for bad in ("", "   "):
        with pytest.raises(M.MixError):
            store.create("u", bad, [])


def test_names_are_tidied_but_kept(store):
    assert store.create("u", "  Morning   Playlist ", []).name == "Morning Playlist"


def test_two_mixes_cannot_share_a_name(store):
    store.create("u", "Morning", [])
    with pytest.raises(M.MixError):
        store.create("u", "morning", [])


def test_different_people_can_both_have_a_morning_mix(store):
    store.create("alice", "Morning", [])
    assert store.create("bob", "Morning", []).name == "Morning"


def test_duplicate_topics_collapse_and_order_is_kept(store):
    mix = store.create("u", "M", ["ai-agents", "the-trade", "ai-agents"])
    assert mix.topic_ids == ["ai-agents", "the-trade"]


def test_a_mix_can_be_renamed_and_retopiced(store):
    mix = store.create("u", "Morning", ["ai-agents"])
    updated = store.update("u", mix.id, name="Commute", topic_ids=["sleep-science"])
    assert updated.name == "Commute" and updated.topic_ids == ["sleep-science"]
    assert store.get("u", mix.id).name == "Commute"


def test_renaming_onto_another_mixs_name_is_refused(store):
    store.create("u", "Morning", [])
    other = store.create("u", "Gym", [])
    with pytest.raises(M.MixError):
        store.update("u", other.id, name="Morning")


def test_one_listener_cannot_touch_anothers_mix(store):
    mix = store.create("alice", "Morning", [])
    assert store.get("bob", mix.id) is None
    assert store.delete("bob", mix.id) is False
    with pytest.raises(M.MixError):
        store.update("bob", mix.id, name="Hijacked")
    assert store.get("alice", mix.id).name == "Morning"


def test_mixes_survive_a_restart(tmp_path):
    path = str(tmp_path / "m.db")
    M.MixStore(path).create("u", "Morning", ["ai-agents"])
    assert [m.name for m in M.MixStore(path).list_for_user("u")] == ["Morning"]


def test_the_starter_mixes_only_use_real_topics():
    """They are offered on an empty page; a broken one is a bad first run."""
    for name, ids in M.STARTER_MIXES:
        assert name
        for topic_id in ids:
            assert topic_id in T.BANK_BY_ID, f"{name} references missing {topic_id}"


# --- the API --------------------------------------------------------------


def test_the_full_lifecycle_over_http(client):
    assert client.get("/api/mixes?user=u1").json()["mixes"] == []

    made = client.post("/api/mixes", json={
        "user": "u1", "name": "Morning", "topic_ids": ["ai-agents", "fed-next-move"]
    }).json()
    assert made["name"] == "Morning"
    assert [t["title"] for t in made["topics"]][0]

    listed = client.get("/api/mixes?user=u1").json()["mixes"]
    assert len(listed) == 1 and listed[0]["id"] == made["id"]

    client.patch(f"/api/mixes/{made['id']}", json={
        "user": "u1", "name": "Early", "topic_ids": ["sleep-science"]
    })
    again = client.get("/api/mixes?user=u1").json()["mixes"][0]
    assert again["name"] == "Early" and again["topic_ids"] == ["sleep-science"]

    assert client.delete(f"/api/mixes/{made['id']}?user=u1").status_code == 200
    assert client.get("/api/mixes?user=u1").json()["mixes"] == []


def test_a_bad_mix_returns_a_message_a_listener_can_read(client):
    res = client.post("/api/mixes", json={"user": "u", "name": "", "topic_ids": []})
    assert res.status_code == 400
    assert "name" in res.json()["error"].lower()


def test_deleting_someone_elses_mix_is_a_404(client):
    made = client.post("/api/mixes", json={"user": "alice", "name": "M"}).json()
    assert client.delete(f"/api/mixes/{made['id']}?user=bob").status_code == 404


def test_the_picker_serves_the_whole_bank(client):
    body = client.get("/api/topics").json()
    assert len(body["topics"]) == len(T.TOPIC_BANK)
    assert {"id", "title", "subtitle", "query", "tags", "icon"} <= set(body["topics"][0])


def test_starters_are_offered_for_an_empty_page(client):
    assert client.get("/api/mixes?user=new").json()["starters"]
