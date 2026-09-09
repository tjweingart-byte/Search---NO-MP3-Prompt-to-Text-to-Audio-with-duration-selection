"""What plays after an episode ends, and where Explore New gets its tiles.

The packet asked for the post-episode popup's four recommendations to "reuse
whatever content-similarity/recommendation logic already exists for feed
personalization, rather than being a separate, disconnected implementation".
That is the property most of these tests are about: `rank_next_up` is
`build_feed`'s three signals over a seeded profile, so a listener cannot be
given two different answers to "what next" on two screens.
"""
from __future__ import annotations

import os
import sys
import time

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as appmod  # noqa: E402
import topics as T  # noqa: E402


@pytest.fixture
def store(tmp_path):
    return T.EventStore(str(tmp_path / "events.db"))


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(appmod, "_rate_limit", lambda request: None)
    monkeypatch.setattr(appmod, "_read_limit", lambda request: None)
    monkeypatch.setattr(appmod, "EVENTS", T.EventStore(str(tmp_path / "e.db")))
    return TestClient(appmod.app)


def play(store, user, topic_id, kind="complete", at=None):
    topic = T.BANK_BY_ID[topic_id]
    store.record(T.Event(user, kind, topic_id, topic.title, topic.tags,
                         at or time.time()))


# --- the shape the popup needs -------------------------------------------


def test_it_always_offers_four(store):
    """The design is a 2x2 grid. Three tiles is a broken layout, not a modest
    recommendation."""
    assert len(T.rank_next_up(store, "u")) == T.NEXT_UP_SIZE


def test_four_even_when_the_listener_has_heard_almost_everything(store):
    """The last resort drops "not already played" rather than the grid: hearing
    something twice beats two empty squares."""
    for topic in T.TOPIC_BANK:
        play(store, "u", topic.id)
    assert len(T.rank_next_up(store, "u")) == T.NEXT_UP_SIZE


def test_the_episode_that_just_ended_is_never_offered_back(store):
    """The one recommendation guaranteed to be wrong."""
    picks = T.rank_next_up(store, "u", after_id="ai-agents")
    assert "ai-agents" not in [t.id for t in picks]


def test_nothing_already_played_is_offered(store):
    play(store, "u", "fed-next-move")
    picks = T.rank_next_up(store, "u", after_id="ai-agents")
    assert "fed-next-move" not in [t.id for t in picks]


# --- it is the feed's opinion, not a second one ---------------------------


def test_it_leans_on_what_just_finished(store):
    """"What next" is a question about this episode first and the history
    second, which is what JUST_HEARD_WEIGHT buys."""
    picks = T.rank_next_up(store, "u", after_id="ai-agents")   # tags: tech
    assert any("tech" in t.tags for t in picks[:2])


def test_a_search_episode_with_no_topic_id_still_gets_related_tiles(store):
    """Most of what plays on the search surface is not a bank tile at all, so
    the keyword fallback is the common path here, not an edge case."""
    picks = T.rank_next_up(store, "u", after_text="what the fed did to interest rates")
    assert any("money" in t.tags for t in picks[:2])


def test_it_agrees_with_the_shelf_it_came_from(store):
    """The anti-drift check. A listener deep in one subject should not be told
    one thing by myFAM and another by the popup."""
    for _ in range(4):
        play(store, "u", "ai-agents")
    feed = T.build_feed(store, "u")
    shelf = {t["id"] for s in feed["sections"] if s["key"] == "from_history"
             for t in s["topics"]}
    picks = {t.id for t in T.rank_next_up(store, "u", after_id="ai-agents")}
    assert picks & shelf, "the popup and the personal shelf shared nothing"


def test_a_brand_new_listener_still_gets_four_from_the_crowd(store):
    """No history, no co-listeners: trending fills it, and it is still four."""
    picks = T.rank_next_up(store, "nobody")
    assert len(picks) == 4 and len({t.id for t in picks}) == 4


def test_declared_interests_steer_it_before_there_is_any_history(store):
    picks = T.rank_next_up(store, "u", interests=["culture"])
    assert any("culture" in t.tags for t in picks)


# --- over HTTP ------------------------------------------------------------


def test_the_endpoint_needs_no_account(client):
    """Playback is never gated, and this is part of playback."""
    body = client.get("/api/nextup", params={"topic_id": "ai-agents"}).json()
    assert len(body["topics"]) == 4
    assert {"id", "title", "query"} <= set(body["topics"][0])


def test_the_popup_records_why_it_showed_what_it_showed(client, monkeypatch, tmp_path):
    """Impressions exist so "why did we show this?" has an answer. The popup is
    a shelf like any other and must not be the one surface nobody can audit."""
    events = T.EventStore(str(tmp_path / "imp.db"))
    monkeypatch.setattr(appmod, "EVENTS", events)
    user = client.get("/api/auth/me").json()["user_id"]
    client.get("/api/nextup", params={"topic_id": "ai-agents"})
    rows = events.impressions_for(user)
    assert rows and {r.section for r in rows} == {"next_up"}
    assert all(r.algo == T.ALGO_VERSION for r in rows)


def test_an_impression_from_the_popup_does_not_train_the_feed_on_itself(client, monkeypatch, tmp_path):
    """Same rule as myFAM: being shown something says nothing about liking it."""
    events = T.EventStore(str(tmp_path / "imp.db"))
    monkeypatch.setattr(appmod, "EVENTS", events)
    user = client.get("/api/auth/me").json()["user_id"]
    client.get("/api/nextup", params={"topic_id": "ai-agents"})
    assert T.taste(events.for_user(user)) == {}


# --- Explore New ----------------------------------------------------------


def test_explore_new_widens_rather_than_confirms(store):
    """It is `rank_might_like`, which suppresses the listener's strongest tag
    on purpose - it is the only signal in the app that offers a way out of a
    taste rather than more of it."""
    for _ in range(4):
        play(store, "u", "ai-agents")          # a listener who is all tech
    body = T.build_explore_new(store, "u")
    tags = {tag for t in body["topics"] for tag in t["tags"]}
    assert body["topics"]
    assert tags - {"tech"}, "every tile came back in the tag they already have"


def test_explore_new_says_why_it_is_showing_you_this(store):
    """A shelf of things you did not ask for is only a good idea if it says so."""
    assert T.build_explore_new(store, "u")["reason"]


def test_explore_new_is_not_empty_for_a_new_listener(store):
    body = T.build_explore_new(store, "nobody")
    assert body["topics"] and body["personalised"] is False


def test_explore_new_over_http(client):
    body = client.get("/api/explorenew").json()
    assert body["topics"] and body["reason"] and body["algo"] == T.ALGO_VERSION
