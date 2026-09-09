"""The two pieces of state the app was missing: who a listener is, and what
the feed showed them.

Both were added because the alternative schema on the table - a podcast-style
`episodes` table with a mutable `episode_history` row per (user, episode) -
would have replaced the append-only log the whole taste model runs on. These
tests exist to hold that line: impressions must be recorded *and* must never
reach the ranking.
"""
from __future__ import annotations

import os
import sys
import time

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as appmod  # noqa: E402
import mixes as M  # noqa: E402
import social as S  # noqa: E402
import topics as T  # noqa: E402


@pytest.fixture
def store(tmp_path):
    return T.EventStore(str(tmp_path / "myfam.db"))


@pytest.fixture
def people(tmp_path):
    return S.SocialStore(str(tmp_path / "social.db"))


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(appmod, "_rate_limit", lambda request: None)
    monkeypatch.setattr(appmod, "EVENTS", T.EventStore(str(tmp_path / "e.db")))
    monkeypatch.setattr(appmod, "SOCIAL", S.SocialStore(str(tmp_path / "s.db")))
    monkeypatch.setattr(appmod, "MIXES", M.MixStore(str(tmp_path / "m.db")))
    return TestClient(appmod.app)


# --- identity: the listener table -----------------------------------------


def test_a_listener_becomes_a_row_without_naming_themselves(people):
    """The hole this fills: before `seen`, a row meant "chose a display name",
    so the app had a taste profile for people it could not say existed."""
    assert people.person("nobody")["known"] is False
    people.seen("device-1")
    row = people.person("device-1")
    assert row["known"] is True
    assert row["joined"] > 0 and row["last_seen"] > 0


def test_first_seen_is_kept_and_last_seen_moves(people):
    people.seen("device-1", at=1000.0)
    people.seen("device-1", at=5000.0)
    row = people.person("device-1")
    assert row["joined"] == 1000.0, "the first sighting is not a moving target"
    assert row["last_seen"] == 5000.0


def test_naming_yourself_does_not_reset_when_you_arrived(people):
    """Setting a name is not new evidence about when the server first saw you."""
    people.seen("device-1", at=1000.0)
    people.set_person("device-1", "Rachel", "rachel")
    row = people.person("device-1")
    assert row["joined"] == 1000.0
    assert row["last_seen"] == 1000.0
    assert row["name"] == "Rachel"


def test_naming_yourself_first_still_leaves_a_usable_row(people):
    """The old order of events must keep working: named, then seen."""
    people.set_person("device-2", "Sam", "sam")
    people.seen("device-2", at=9000.0)
    row = people.person("device-2")
    assert row["handle"] == "sam"
    assert row["last_seen"] == 9000.0


def test_an_empty_listener_id_is_not_a_listener(people):
    people.seen("")
    assert people.active_since(0) == []


def test_active_since_is_the_recency_signal_a_daily_feed_needs(people):
    now = time.time()
    people.seen("recent", at=now - 60)
    people.seen("stale", at=now - 40 * 86400)
    assert people.active_since(now - 86400) == ["recent"]
    assert set(people.active_since(0)) == {"recent", "stale"}


def test_the_profile_endpoint_reports_observed_facts_only(client):
    body = client.get("/api/profile", params={"user": "u1"}).json()
    assert body["known"] is True, "asking for your profile is a sighting"
    assert body["last_seen"] > 0
    # And nothing invented alongside it.
    assert "email" not in body and "followers" not in body


# --- impressions: why did we show this ------------------------------------


def test_an_impression_is_recorded_with_its_section_and_algo(store):
    store.record_impressions("u", [("trending", "golf-evolution")])
    (shown,) = store.impressions_for("u")
    assert shown.topic_id == "golf-evolution"
    assert shown.section == "trending"
    assert shown.algo == T.ALGO_VERSION
    assert shown.tags == T.BANK_BY_ID["golf-evolution"].tags


def test_impressions_never_reach_the_ranking_read(store):
    """The load-bearing test. `for_user` feeds `taste`, and it is capped at
    `limit` rows - so impressions in there would train the taste model on what
    the feed showed instead of on what the listener did."""
    store.record(T.Event("u", "complete", "golf-evolution", "",
                         T.BANK_BY_ID["golf-evolution"].tags))
    for _ in range(50):
        store.record_impressions(
            "u", [(s, t.id) for s in ("trending",) for t in T.TOPIC_BANK]
        )
    events = store.for_user("u")
    assert [e.kind for e in events] == ["complete"], "impressions leaked into ranking"
    assert store.impressions_for("u", limit=10_000)


def test_a_flood_of_impressions_does_not_change_taste(store):
    """Same guarantee, stated as the outcome rather than the mechanism."""
    store.record(T.Event("u", "complete", "golf-evolution", "",
                         T.BANK_BY_ID["golf-evolution"].tags))
    before = T.taste(store.for_user("u"))
    for _ in range(30):
        store.record_impressions("u", [("trending", t.id) for t in T.TOPIC_BANK])
    assert T.taste(store.for_user("u")) == before


def test_an_impression_carries_no_taste_weight(store):
    """Belt and braces: even if one were handed straight to `taste`, being
    shown a tile must say nothing about whether you liked it."""
    assert T.IMPRESSION not in T.EVENT_WEIGHT
    assert T.IMPRESSION in T.EVENT_KINDS
    shown = [T.Event("u", T.IMPRESSION, "golf-evolution", "",
                     T.BANK_BY_ID["golf-evolution"].tags)]
    assert T.taste(shown) == {}


def test_impressions_do_not_count_as_plays_anywhere(store):
    """`plays_since` and `users_who_played` drive trending and co-listeners.
    A tile shown to everyone must not become a tile everyone played."""
    store.record_impressions("u", [("trending", "golf-evolution")])
    assert store.plays_since(0) == []
    assert store.users_who_played(["golf-evolution"]) == {}


def test_impressions_stay_out_of_the_profile_counts(store):
    store.record_impressions("u", [("trending", "golf-evolution")])
    body = T.summary(store, "u")
    assert body["played"] == 0 and body["finished"] == 0 and body["searched"] == 0


def test_old_impressions_are_pruned_and_real_events_are_not(store):
    old = time.time() - T.IMPRESSION_TTL - 86400
    store.record_impressions("u", [("trending", "golf-evolution")], at=old)
    store.record(T.Event("u", "complete", "the-trade", "",
                         T.BANK_BY_ID["the-trade"].tags, old))
    # The pruner runs at most hourly; the first write above set that clock.
    store._pruned_at = 0.0
    store.record_impressions("u", [("trending", "space-race")])
    kept = [e.topic_id for e in store.impressions_for("u")]
    assert kept == ["space-race"], "an expired impression was kept"
    assert [e.topic_id for e in store.for_user("u")] == ["the-trade"], \
        "pruning impressions must never touch behavioural events"


def test_building_the_feed_does_not_write(store):
    """`build_feed` is a pure function of the log, and the tests that call it
    directly depend on that. The write belongs at the request boundary."""
    T.build_feed(store, "u")
    assert store.impressions_for("u") == []


def who_is(client) -> str:
    """The listener id this cookie jar resolves to. There is no other way to
    ask - the id is minted by the server and kept out of the page's reach."""
    return client.get("/api/auth/me").json()["user_id"]


def test_the_endpoint_logs_one_impression_per_tile_it_returned(client):
    feed = client.get("/api/myfam").json()
    tiles = [(s["key"], t["id"]) for s in feed["sections"] for t in s["topics"]]
    assert tiles, "nothing was shown, so this proves nothing"
    assert feed["algo"] == T.ALGO_VERSION
    logged = appmod.EVENTS.impressions_for(who_is(client))
    assert sorted((e.section, e.topic_id) for e in logged) == sorted(tiles)


def test_no_impression_is_ever_attributed_to_an_empty_listener(client):
    """Every request now has a session, so nothing should land under "".

    This replaces a test that asserted an id-less request logged nothing. There
    is no such request any more: the server mints an identity rather than
    accepting one, which is the whole point of the change.
    """
    client.get("/api/myfam")
    assert appmod.EVENTS.impressions_for("") == []
    assert appmod.EVENTS.impressions_for(who_is(client)), "the feed logged nothing"


def test_the_feed_endpoint_notes_the_listener(client):
    client.get("/api/myfam")
    assert appmod.SOCIAL.person(who_is(client))["known"] is True


def test_recording_an_event_notes_the_listener(client):
    client.post("/api/event", json={"kind": "play", "topic_id": "golf-evolution"})
    assert appmod.SOCIAL.person(who_is(client))["known"] is True


def test_an_unknown_kind_is_still_refused(store):
    """Widening the accepted kinds must not turn the gate off."""
    store.record(T.Event("u", "hovered", "golf-evolution"))
    assert store.for_user("u") == []


# --- migrating a database that already has data in it ---------------------


def _legacy_events(path):
    """The events table exactly as it shipped, before section/algo existed."""
    import sqlite3
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE events (
               id       INTEGER PRIMARY KEY AUTOINCREMENT,
               user_id  TEXT NOT NULL,
               kind     TEXT NOT NULL,
               topic_id TEXT NOT NULL DEFAULT '',
               text     TEXT NOT NULL DEFAULT '',
               tags     TEXT NOT NULL DEFAULT '',
               at       REAL NOT NULL,
               thread   TEXT NOT NULL DEFAULT ''
           )"""
    )
    conn.execute("INSERT INTO events (user_id, kind, topic_id, tags, at, thread)"
                 " VALUES ('u', 'complete', 'golf-evolution', 'sports', 1000.0, 'why')")
    conn.commit()
    conn.close()


def _legacy_people(path):
    """The people table as it shipped, before last_seen existed."""
    import sqlite3
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE people (
               user_id TEXT PRIMARY KEY,
               name    TEXT NOT NULL DEFAULT '',
               handle  TEXT NOT NULL DEFAULT '',
               joined  REAL NOT NULL
           )"""
    )
    conn.execute("INSERT INTO people VALUES ('old', 'Rachel', 'rachel', 500.0)")
    conn.commit()
    conn.close()


def test_an_existing_event_log_is_widened_not_lost(tmp_path):
    """The log is not regenerable. A migration that drops it is a data loss
    bug, not an inconvenience the way a cold script cache would be."""
    path = str(tmp_path / "old.db")
    _legacy_events(path)
    store = T.EventStore(path)
    kept = store.for_user("u")
    assert [(e.kind, e.topic_id, e.thread) for e in kept] == \
        [("complete", "golf-evolution", "why")]
    # And the new columns work on the widened table.
    store.record_impressions("u", [("trending", "space-race")])
    assert store.impressions_for("u")[0].algo == T.ALGO_VERSION
    assert store.open_threads("u")[0]["thread"] == "why"


def test_an_existing_person_keeps_their_name_and_join_date(tmp_path):
    path = str(tmp_path / "old-social.db")
    _legacy_people(path)
    people = S.SocialStore(path)
    row = people.person("old")
    assert row["name"] == "Rachel" and row["joined"] == 500.0
    assert row["last_seen"] == 0.0, "a row from before the column has no sighting"
    assert row["known"] is True
    people.seen("old", at=2000.0)
    after = people.person("old")
    assert after["joined"] == 500.0 and after["last_seen"] == 2000.0


def test_opening_a_store_twice_is_safe(tmp_path):
    """Every worker on the machine runs the migration on startup."""
    path = str(tmp_path / "twice.db")
    T.EventStore(path).record_impressions("u", [("trending", "space-race")])
    assert len(T.EventStore(path).impressions_for("u")) == 1
    spath = str(tmp_path / "twice-social.db")
    S.SocialStore(spath).seen("u", at=1.0)
    assert S.SocialStore(spath).person("u")["last_seen"] == 1.0
