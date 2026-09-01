"""myFAM: the shared bank, the taste model, and the four sections.

The thing worth testing here is not that the endpoint returns JSON. It is
that the four sections run on four different signals - the failure mode of
any feed like this is four headings over one ranked list.
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
    return T.EventStore(str(tmp_path / "myfam.db"))


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(appmod, "_rate_limit", lambda request: None)
    monkeypatch.setattr(appmod, "EVENTS", T.EventStore(str(tmp_path / "api.db")))
    return TestClient(appmod.app)


def play(store, user, topic_id, kind="play", ago=0.0):
    store.record(T.Event(user, kind, topic_id, "",
                         T.BANK_BY_ID[topic_id].tags, time.time() - ago))


# --- the bank -------------------------------------------------------------


def test_the_bank_is_shared_so_scripts_can_be_shared():
    """The cost argument: one bank, one script per topic, reused by everyone."""
    queries = [t.query for t in T.TOPIC_BANK]
    assert len(set(queries)) == len(queries), "a duplicate query wastes a cache slot"
    ids = [t.id for t in T.TOPIC_BANK]
    assert len(set(ids)) == len(ids)
    for topic in T.TOPIC_BANK:
        assert topic.tags, f"{topic.id} has no tags, so it can never be ranked"
        assert set(topic.tags) <= set(T.TAG_WORDS), f"{topic.id} has an unknown tag"


def test_a_bank_query_reads_as_a_real_question():
    """Tiles generate from `query`, not `title`. A title is not a question."""
    for topic in T.TOPIC_BANK:
        assert len(topic.query.split()) >= 4, f"{topic.id}: query is too thin to brief"
        assert topic.query == topic.query.strip()


# --- taste ----------------------------------------------------------------


def test_taste_is_built_from_what_they_actually_finished():
    now = time.time()
    events = [
        T.Event("u", "complete", "", "", ("sports",), now),
        T.Event("u", "play", "", "", ("culture",), now),
    ]
    profile = T.taste(events, now)
    assert profile["sports"] > profile["culture"], "finishing beats merely starting"


def test_a_skip_counts_against_a_tag():
    """A skip is evidence, not a weak play - otherwise skipping recommends more."""
    now = time.time()
    profile = T.taste([T.Event("u", "skip", "", "", ("sports",), now)], now)
    assert profile["sports"] < 0


def test_old_interests_fade():
    now = time.time()
    events = [
        T.Event("u", "complete", "", "", ("sports",), now - 60 * T.HALF_LIFE),
        T.Event("u", "play", "", "", ("tech",), now),
    ]
    profile = T.taste(events, now)
    assert profile["tech"] > profile["sports"], "a feed should not be a museum"


def test_free_text_searches_are_tagged_so_history_counts():
    assert "money" in T.tags_for_text("what is the fed doing about inflation")
    assert "health" in T.tags_for_text("how do I fix my sleep")
    assert T.tags_for_text("zzzz nonsense") == ()


# --- the four sections ----------------------------------------------------


def test_trending_ignores_the_listener_entirely(store):
    """It is the same for everyone; that is what makes it the cheapest section."""
    for _ in range(3):
        play(store, "someone", "fed-next-move")
    play(store, "other", "golf-evolution")
    ranked = T.rank_trending(store)
    assert ranked[0].id == "fed-next-move"


def test_trending_is_not_empty_on_a_cold_start(store):
    assert len(T.rank_trending(store)) == T.SECTION_SIZE


def test_history_recommends_what_they_already_like(store):
    play(store, "u", "golf-evolution", kind="complete")
    profile = T.taste(store.for_user("u"))
    picks = T.rank_from_history(profile, exclude=set())
    assert any("sports" in p.tags for p in picks[:2])


def test_history_never_recommends_what_they_just_played(store):
    play(store, "u", "golf-evolution", kind="complete")
    events = store.for_user("u")
    picks = T.rank_from_history(T.taste(events), exclude=T._played_ids(events))
    assert all(p.id != "golf-evolution" for p in picks)


def test_might_like_is_not_the_same_list_as_history(store):
    """The whole reason for four sections: four signals, not four shuffles."""
    play(store, "u", "golf-evolution", kind="complete")
    play(store, "u", "the-trade", kind="complete")
    profile = T.taste(store.for_user("u"))
    history = [t.id for t in T.rank_from_history(profile, set())]
    might = [t.id for t in T.rank_might_like(profile, set())]
    assert history[:3] != might[:3], "might_like is just history with a new heading"


def test_might_like_suppresses_their_strongest_tag(store):
    """Ranking on raw affinity builds a filter bubble by accident."""
    for _ in range(4):
        play(store, "u", "golf-evolution", kind="complete")
    profile = T.taste(store.for_user("u"))
    picks = T.rank_might_like(profile, set())
    assert picks, "a listener with taste should still get suggestions"
    assert not all("sports" in p.tags for p in picks)


def test_followers_uses_people_who_overlap_with_you(store):
    # Two listeners share golf; the neighbour also plays the space episode.
    play(store, "me", "golf-evolution")
    play(store, "neighbour", "golf-evolution")
    play(store, "neighbour", "space-race")
    play(store, "stranger", "restaurant-scene")
    mine = T._played_ids(store.for_user("me"))
    picks = T.rank_followers(store, "me", mine, exclude={"golf-evolution"})
    ids = [p.id for p in picks]
    assert "space-race" in ids
    assert "restaurant-scene" not in ids, "a stranger's taste is not a signal"


def test_followers_is_empty_rather_than_faked_for_a_new_listener(store):
    assert T.rank_followers(store, "nobody", set(), set()) == []


# --- the whole feed -------------------------------------------------------


def test_no_topic_appears_in_two_sections(store):
    play(store, "u", "golf-evolution", kind="complete")
    play(store, "friend", "golf-evolution")
    play(store, "friend", "space-race")
    feed = T.build_feed(store, "u")
    seen = [t["id"] for s in feed["sections"] for t in s["topics"]]
    assert len(seen) == len(set(seen)), "the page repeats itself"


def test_a_new_listener_gets_an_honest_page_not_a_fake_one(store):
    feed = T.build_feed(store, "brand-new")
    by_key = {s["key"]: s for s in feed["sections"]}
    assert by_key["trending"]["topics"], "trending works with no history at all"
    assert not feed["personalised"]
    for key in ("followers", "from_history"):
        assert by_key[key]["empty_reason"], f"{key} must say why it is empty"


def test_the_four_sections_are_always_present_and_in_order(store):
    feed = T.build_feed(store, "u")
    assert [s["key"] for s in feed["sections"]] == [k for k, _ in T.SECTIONS]


def test_the_personal_section_comes_before_the_popular_one(store):
    """Someone opening myFAM should not scroll past the crowd to reach it."""
    keys = [k for k, _ in T.SECTIONS]
    assert keys.index("from_history") < keys.index("trending")
    # Fill order is the opposite on purpose: the constrained sections pick first.
    assert T.FILL_ORDER.index("from_history") < T.FILL_ORDER.index("trending")


# --- the API --------------------------------------------------------------


def test_the_feed_endpoint_works_without_a_user(client):
    body = client.get("/api/myfam").json()
    assert [s["key"] for s in body["sections"]] == [k for k, _ in T.SECTIONS]


def test_recording_an_event_changes_the_feed(client):
    before = client.get("/api/myfam?user=u1").json()
    for _ in range(3):
        client.post("/api/event", json={"user": "u1", "kind": "complete",
                                        "topic_id": "sleep-science"})
    after = client.get("/api/myfam?user=u1").json()
    assert after["personalised"] and not before["personalised"]
    history = [s for s in after["sections"] if s["key"] == "from_history"][0]
    assert history["topics"], "three completions should produce recommendations"


def test_an_unknown_event_kind_is_ignored_not_fatal(client):
    assert client.post("/api/event", json={"user": "u", "kind": "nonsense"}).status_code == 200


def test_a_broken_event_store_never_breaks_the_feed(client, monkeypatch):
    """Playback and browsing must survive the recommender falling over."""
    class Broken(T.EventStore):
        def __init__(self):
            pass
        def _conn(self):
            raise RuntimeError("disk gone")
    monkeypatch.setattr(appmod, "EVENTS", Broken())
    body = client.get("/api/myfam?user=u1").json()
    trending = [s for s in body["sections"] if s["key"] == "trending"][0]
    assert trending["topics"], "trending should still fall back to the bank"


def test_the_personal_sections_are_not_starved_by_the_generic_ones(store):
    """The bug this ordering exists to prevent.

    Filled in display order, Trending and "might like" claim the whole bank
    first - they can fall back to anything - and the two sections the listener
    actually asked for arrive empty. The personal sections must choose first.
    """
    play(store, "me", "golf-evolution", kind="complete")
    play(store, "me", "the-trade", kind="complete")
    play(store, "neighbour", "golf-evolution")
    play(store, "neighbour", "sleep-science")

    feed = T.build_feed(store, "me")
    by_key = {s["key"]: s for s in feed["sections"]}
    assert by_key["from_history"]["topics"], "history section was starved"
    assert by_key["followers"]["topics"], "co-listener section was starved"
    assert "sleep-science" in [t["id"] for t in by_key["followers"]["topics"]]
    # And the generic sections still fill, because the bank is big enough.
    assert by_key["trending"]["topics"] and by_key["might_like"]["topics"]


def test_the_bank_can_fill_every_section_without_repeating(store):
    """Four sections of six needs twenty-four topics, not twenty-two."""
    assert len(T.TOPIC_BANK) >= len(T.SECTIONS) * T.SECTION_SIZE


def test_no_section_offers_back_something_already_played(store):
    """The feed hands them the next episode, not the one they finished."""
    for topic_id in ("golf-evolution", "the-trade", "sleep-science"):
        play(store, "me", topic_id, kind="complete")
    play(store, "neighbour", "golf-evolution")
    play(store, "neighbour", "space-race")

    feed = T.build_feed(store, "me")
    shown = {t["id"] for s in feed["sections"] for t in s["topics"]}
    assert not (shown & {"golf-evolution", "the-trade", "sleep-science"})


# --- Go Deeper: the threads episodes left open ----------------------------


def test_a_finished_episode_leaves_its_thread_behind(store):
    store.record(T.Event("u", "complete", "ai-agents", "", ("tech",),
                         thread="why chip supply is so concentrated"))
    threads = store.open_threads("u")
    assert len(threads) == 1
    assert threads[0]["thread"] == "why chip supply is so concentrated"
    assert threads[0]["from_title"] == "Why Everyone Is Talking About AI Agents"


def test_a_thread_they_have_already_asked_about_is_closed(store):
    """It is only a thread while it is still open."""
    store.record(T.Event("u", "complete", "ai-agents", "", ("tech",),
                         thread="why chip supply is so concentrated"))
    store.record(T.Event("u", "search", "", "why chip supply is so concentrated", ()))
    assert store.open_threads("u") == []


def test_only_finished_episodes_leave_threads(store):
    """Starting an episode is not hearing the ending that opened the thread."""
    store.record(T.Event("u", "play", "ai-agents", "", ("tech",), thread="a thread"))
    assert store.open_threads("u") == []


def test_the_same_thread_twice_appears_once(store):
    for _ in range(2):
        store.record(T.Event("u", "complete", "ai-agents", "", ("tech",), thread="one thread"))
    assert len(store.open_threads("u")) == 1


def test_threads_survive_a_restart(tmp_path):
    path = str(tmp_path / "e.db")
    T.EventStore(path).record(
        T.Event("u", "complete", "ai-agents", "", ("tech",), thread="a lasting thread"))
    assert T.EventStore(path).open_threads("u")[0]["thread"] == "a lasting thread"


def test_the_go_deeper_endpoint_serves_them(client):
    client.post("/api/event", json={"user": "u1", "kind": "complete",
                                    "topic_id": "sleep-science",
                                    "thread": "why sleep debt cannot be repaid"})
    body = client.get("/api/godeeper?user=u1").json()["threads"]
    assert [t["thread"] for t in body] == ["why sleep debt cannot be repaid"]


def test_go_deeper_is_empty_not_broken_for_a_new_listener(client):
    assert client.get("/api/godeeper?user=nobody").json()["threads"] == []


# --- the profile summary --------------------------------------------------


def test_the_profile_counts_only_what_actually_happened(store):
    play(store, "u", "ai-agents")
    play(store, "u", "sleep-science", kind="complete")
    store.record(T.Event("u", "search", "", "how heat pumps work", ()))
    body = T.summary(store, "u")
    assert body["played"] == 1 and body["finished"] == 1 and body["searched"] == 1
    assert body["listener"] == "u"


def test_a_new_listener_has_an_empty_profile_not_a_fake_one(store):
    body = T.summary(store, "nobody")
    assert body["played"] == 0 and body["finished"] == 0
    assert body["subjects"] == []
    assert body["since"] == 0.0


def test_a_skipped_subject_is_not_listed_as_something_they_like(store):
    play(store, "u", "golf-evolution", kind="complete")
    play(store, "u", "sleep-science", kind="skip")
    subjects = T.summary(store, "u")["subjects"]
    assert "sports" in subjects
    assert "health" not in subjects, "a skip is evidence against, not for"


def test_the_profile_endpoint_serves_it(client):
    client.post("/api/event", json={"user": "p1", "kind": "complete",
                                    "topic_id": "ai-agents"})
    body = client.get("/api/profile?user=p1").json()
    assert body["finished"] == 1 and "tech" in body["subjects"]
