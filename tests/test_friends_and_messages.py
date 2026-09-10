"""The follow graph, and sending an episode to somebody inside the app."""
from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as appmod
import messages as messages_mod
import social as social_mod


@pytest.fixture
def store(tmp_path):
    return messages_mod.MessageStore(str(tmp_path / "messages.db"))


@pytest.fixture
def graph(tmp_path):
    social = social_mod.SocialStore(str(tmp_path / "social.db"))
    social.set_person("a", "Ana", "ana")
    social.set_person("b", "Ben", "ben")
    social.set_person("c", "Cy", "cy")
    return social


# --- the follow graph -----------------------------------------------------

def test_following_is_one_sided_and_friendship_is_the_mutual_case(graph):
    """Asymmetric like the copy already says, and mutuals derived rather than
    stored - so there is no accept step to get wrong and no way for the two
    directions to disagree."""
    graph.follow("a", "b")
    assert graph.is_following("a", "b") and not graph.is_following("b", "a")
    assert graph.friends("a") == []
    graph.follow("b", "a")
    assert [p["handle"] for p in graph.friends("a")] == ["ben"]


def test_following_twice_changes_nothing(graph):
    assert graph.follow("a", "b") is True
    assert graph.follow("a", "b") is False


def test_nobody_can_follow_themselves(graph):
    """Not a moral position: friends is the intersection of following and
    followers, so a self-follow puts everybody in their own circle."""
    with pytest.raises(social_mod.SocialError):
        graph.follow("a", "a")


def test_counts_are_derived_rather_than_kept(graph):
    """A denormalised counter is a number that can be wrong, and this app has
    a rule about inventing numbers on a profile page."""
    graph.follow("a", "b")
    graph.follow("c", "b")
    assert graph.follow_counts("b") == {"following": 0, "followers": 2}


def test_people_can_be_found_by_handle_or_name(graph):
    assert [p["handle"] for p in graph.find_people("be")] == ["ben"]
    assert [p["handle"] for p in graph.find_people("An")] == ["ana"]


def test_a_one_letter_search_returns_nothing(graph):
    """It would return most of the listener table, which is a directory dump
    rather than a search."""
    assert graph.find_people("a") == []


def test_somebody_with_no_handle_is_not_in_the_directory(graph):
    graph.seen("ghost")
    assert graph.find_people("gh") == []


def test_deleting_a_listener_removes_them_from_both_directions(graph):
    """Otherwise they stay in somebody else's follower count, pointing at an id
    that no longer resolves to a person."""
    graph.follow("a", "b")
    graph.follow("b", "a")
    graph.forget("b")
    assert graph.follow_counts("a") == {"following": 0, "followers": 0}


# --- messages -------------------------------------------------------------

def test_a_thread_id_is_the_same_from_both_sides(store):
    """Derived rather than allocated, so two people opening the conversation at
    once cannot create two threads and split the history."""
    assert messages_mod.thread_id("a", "b") == messages_mod.thread_id("b", "a")


def test_you_cannot_message_yourself(store):
    with pytest.raises(messages_mod.MessageError):
        store.send("a", "a", text="hello")


def test_an_episode_share_carries_the_question_and_not_the_audio(store):
    """The whole cost design: a share is a row pointing at a query whose script
    already exists, so sending an episode to ten people costs ten rows and not
    ten episodes."""
    message = store.send("a", "b", kind="episode", query="why bonds move",
                         minutes=3, title="Bonds")
    assert message.query == "why bonds move" and message.minutes == 3
    assert message.text == ""


def test_an_episode_share_needs_a_question(store):
    with pytest.raises(messages_mod.MessageError):
        store.send("a", "b", kind="episode", query="")


def test_an_empty_text_message_is_refused(store):
    with pytest.raises(messages_mod.MessageError):
        store.send("a", "b", text="   ")


def test_a_thread_reads_oldest_first(store):
    store.send("a", "b", text="first", at=100)
    store.send("b", "a", text="second", at=200)
    assert [m.text for m in store.thread("a", "b")] == ["first", "second"]


def test_the_inbox_shows_the_last_message_per_conversation(store):
    store.send("a", "b", text="old", at=100)
    store.send("a", "b", text="new", at=200)
    store.send("a", "c", text="elsewhere", at=150)
    inbox = store.inbox("a")
    assert len(inbox) == 2
    assert inbox[0]["last"]["text"] == "new"


def test_unread_counts_only_what_arrived_for_you(store):
    store.send("a", "b", text="hello")
    assert store.unread_total("b") == 1
    assert store.unread_total("a") == 0


def test_opening_a_thread_marks_it_read(store):
    store.send("a", "b", text="hello", at=100)
    store.mark_read("b", "a", at=200)
    assert store.unread_total("b") == 0


def test_a_message_arriving_after_a_read_is_unread_again(store):
    store.send("a", "b", text="hello", at=100)
    store.mark_read("b", "a", at=200)
    store.send("a", "b", text="again", at=300)
    assert store.unread_total("b") == 1


def test_mine_is_reported_rather_than_the_senders_id(store):
    """The client needs to know which side to draw the bubble on, and does not
    need somebody else's listener id to do it."""
    message = store.send("a", "b", text="hello")
    assert message.as_dict("a")["mine"] is True
    assert "sender" not in message.as_dict("b")


def test_deleting_a_listener_removes_their_side_of_a_conversation(store):
    """The other person's messages stay: those are their words, not this
    listener's data."""
    store.send("a", "b", text="mine")
    store.send("b", "a", text="theirs")
    store.forget("a")
    assert [m.text for m in store.thread("a", "b")] == ["theirs"]


# --- through the app ------------------------------------------------------

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(appmod, "_rate_limit", lambda request: None)
    with TestClient(appmod.app) as c:
        yield c


def signed_in(client, email, name, handle):
    client.post("/api/auth/signup", json={"email": email, "password": "password12"})
    client.post("/api/me", json={"name": name, "handle": handle})
    return client.get("/api/auth/me").json()["user_id"]


def test_two_listeners_can_find_follow_and_message_each_other(client):
    ana = signed_in(client, "ana@b.com", "Ana", "ana")
    client.post("/api/auth/logout")
    signed_in(client, "ben@b.com", "Ben", "ben")

    found = client.get("/api/people?q=ana").json()["people"]
    assert [p["handle"] for p in found] == ["ana"]
    assert found[0]["following"] is False

    assert client.post("/api/friends/follow",
                       json={"handle": "ana"}).json()["counts"]["following"] == 1
    sent = client.post("/api/messages", json={
        "to": ana, "query": "why bonds move", "minutes": 3, "title": "Bonds"})
    assert sent.status_code == 200, sent.text
    assert sent.json()["message"]["kind"] == "episode"

    client.post("/api/auth/logout")
    client.post("/api/auth/login", json={"email": "ana@b.com", "password": "password12"})
    inbox = client.get("/api/messages").json()
    assert inbox["unread"] == 1
    assert inbox["threads"][0]["name"] == "Ben"


def test_a_search_never_returns_yourself(client):
    signed_in(client, "ana@b.com", "Ana", "ana")
    assert client.get("/api/people?q=ana").json()["people"] == []


def test_following_somebody_who_does_not_exist_is_a_404(client):
    signed_in(client, "ana@b.com", "Ana", "ana")
    assert client.post("/api/friends/follow",
                       json={"handle": "nobody"}).status_code == 404


def test_friends_need_an_account(client):
    assert client.get("/api/friends").status_code == 401
    assert client.get("/api/messages").status_code == 401
