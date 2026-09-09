"""Interests, language, the weekly recap - and what "Skip for now" costs.

Three separate claims are under test here, and they are worth naming because
each was a product decision before it was code:

1. A declared interest **changes the feed**. A picker that stores an answer
   nobody reads is worse than no picker, and this is the check that it is read.
2. The six-interest cap is enforced **on the way in**, not only by a disabled
   button. A cap only the client applies is not a cap.
3. Anything the server keeps for you needs an account, and anything you can
   hear does not.
"""
from __future__ import annotations

import calendar
import os
import sys
import time

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as appmod  # noqa: E402
import preferences as P  # noqa: E402
import topics as T  # noqa: E402

GOOD = "a-long-enough-password"


@pytest.fixture
def store(tmp_path):
    return P.PreferenceStore(str(tmp_path / "prefs.db"))


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(appmod, "_rate_limit", lambda request: None)
    monkeypatch.setattr(appmod, "_read_limit", lambda request: None)
    monkeypatch.setattr(appmod, "EVENTS", T.EventStore(str(tmp_path / "e.db")))
    return TestClient(appmod.app)


def sign_up(client, email="one@fam.test"):
    res = client.post("/api/auth/signup", json={"email": email, "password": GOOD})
    assert res.status_code == 200, res.text
    return client


# --- the store ------------------------------------------------------------


def test_an_unknown_listener_reads_as_the_defaults(store):
    """A missing row is a listener who has not chosen yet, not an error."""
    prefs = store.get("nobody")
    assert prefs.interests == () and prefs.language == "en"
    assert prefs.weekly_recap is True and prefs.intro_done is False


def test_saving_one_page_does_not_clear_the_other(store):
    """The intro saves interests, then language, from two screens."""
    store.save("u", interests=["tech", "money"])
    store.save("u", language="es")
    prefs = store.get("u")
    assert prefs.interests == ("tech", "money") and prefs.language == "es"


def test_the_six_cap_is_enforced_on_the_way_in(store):
    """The interface disables the seventh chip. That is not the rule; this is."""
    seven = list(T.TAG_LABELS)[:7]
    assert len(seven) == 7
    with pytest.raises(P.PreferenceError):
        store.save("u", interests=seven)
    assert store.get("u").interests == ()


def test_an_invented_interest_is_refused(store):
    with pytest.raises(P.PreferenceError):
        store.save("u", interests=["astrology"])


def test_duplicates_collapse_rather_than_eating_the_cap(store):
    store.save("u", interests=["tech", "tech", "money"])
    assert store.get("u").interests == ("tech", "money")


def test_an_unknown_language_is_refused(store):
    with pytest.raises(P.PreferenceError):
        store.save("u", language="klingon")


def test_every_interest_on_offer_is_a_tag_the_ranker_scores():
    """The picker and the ranking share one vocabulary or the picker is a lie."""
    assert set(P.INTERESTS) == set(T.TAG_WORDS), (
        "an interest that is not a TAG_WORDS facet can never rank anything")


# --- the recap week -------------------------------------------------------

# 2026-09-06 is a Sunday; 2026-09-09 the Wednesday after it. timegm rather
# than mktime: week_start works in UTC, and mktime would read these as local
# time - so the test would quietly measure the machine's timezone instead.
SUNDAY = calendar.timegm(time.strptime("2026-09-06 09:00", "%Y-%m-%d %H:%M"))
WEDNESDAY = SUNDAY + 3 * 86400
NEXT_SUNDAY = SUNDAY + 7 * 86400


def test_a_week_is_named_by_the_sunday_that_started_it():
    assert P.week_start(SUNDAY) == "2026-09-06"
    assert P.week_start(WEDNESDAY) == "2026-09-06", "midweek is still that week"
    assert P.week_start(NEXT_SUNDAY) == "2026-09-13"


def test_a_recap_missed_on_sunday_is_still_owed_on_wednesday(store):
    """The trigger is 'the first open on or after Sunday', which is why this
    is a stored date and not a flag nobody would ever clear."""
    assert store.recap_due("u", SUNDAY) is True
    assert store.recap_due("u", WEDNESDAY) is True, "a missed Sunday must not be lost"


def test_seeing_it_once_settles_the_week_but_not_the_next_one(store):
    store.mark_recap_seen("u", WEDNESDAY)
    assert store.recap_due("u", WEDNESDAY + 3600) is False
    assert store.recap_due("u", NEXT_SUNDAY) is True


def test_turning_weekly_notifications_off_stops_it_for_good(store):
    """The popup's own link, and it has to outlive the popup that set it."""
    store.save("u", weekly_recap=False)
    assert store.recap_due("u", NEXT_SUNDAY) is False
    assert store.recap_due("u", NEXT_SUNDAY + 86400 * 30) is False


# --- interests actually reach the feed ------------------------------------


def test_a_declared_interest_ranks_the_feed_for_a_listener_with_no_history():
    """The reason to ask at all: "Made for you" is empty without this."""
    profile = T.taste([], interests=["science"])
    assert profile.get("science", 0) > 0
    ranked = T.rank_from_history(profile, set())
    assert ranked, "six chosen interests still produced an empty personal shelf"
    assert any("science" in t.tags for t in ranked)


def test_behaviour_outweighs_a_declaration_once_there_is_any():
    """An intro answer is a starting position, not a rule. Someone who chose
    Sport and then finished three tech episodes should get tech."""
    now = time.time()
    events = [T.Event("u", "complete", "ai-agents", "", ("tech",), now)] * 3
    profile = T.taste(events, now, interests=["sports"])
    assert profile["tech"] > profile["sports"]


# --- the API and the gate -------------------------------------------------


def test_the_choices_are_public_but_the_answers_are_not(client):
    """The intro runs before anyone has an account, so it must be able to list
    what is on offer - and must not claim an anonymous answer was saved."""
    body = client.get("/api/preferences").json()
    assert len(body["interests_available"]) == len(T.TAG_LABELS)
    assert body["languages"] and body["max_interests"] == 6
    assert body["account"] is False and body["saved"] is False


def test_a_language_that_changes_nothing_says_so(client):
    """PROBLEMS.md's oldest lesson: a setting that silently does nothing is
    the failure this project has paid for most often."""
    assert client.get("/api/preferences").json()["language_active"] is False


def test_storing_a_preference_needs_an_account(client):
    res = client.post("/api/preferences", json={"interests": ["tech"]})
    assert res.status_code == 401
    assert "account" in res.json()["error"].lower()


def test_an_account_stores_and_returns_them(client):
    sign_up(client)
    saved = client.post("/api/preferences",
                        json={"interests": ["tech", "money"], "language": "fr",
                              "intro_done": True}).json()
    assert saved["interests"] == ["tech", "money"] and saved["language"] == "fr"
    body = client.get("/api/preferences").json()
    assert body["saved"] is True and body["intro_done"] is True


def test_a_seventh_interest_is_a_message_not_a_stack_trace(client):
    sign_up(client)
    res = client.post("/api/preferences",
                      json={"interests": list(T.TAG_LABELS)[:7]})
    assert res.status_code == 400
    assert "6" in res.json()["error"] or "six" in res.json()["error"].lower()


def test_stored_interests_rank_myfam_without_being_asked_for(client):
    """The account path: the feed reads them, the client never sends them."""
    sign_up(client)
    client.post("/api/preferences", json={"interests": ["science"]})
    feed = client.get("/api/myfam").json()
    made_for_you = [s for s in feed["sections"] if s["key"] == "from_history"][0]
    assert made_for_you["topics"], "a chosen interest left the personal shelf empty"


def test_an_anonymous_listener_can_still_be_ranked_for_this_one_request(client):
    """Their intro answers live in their own browser and nowhere else, so a
    hint on the request is the only route by which the ranker can honour them.
    It is validated against a fixed vocabulary and never stored."""
    feed = client.get("/api/myfam", params={"interests": "science,health"}).json()
    made_for_you = [s for s in feed["sections"] if s["key"] == "from_history"][0]
    assert made_for_you["topics"]


def test_a_rubbish_interests_hint_costs_a_shelf_not_the_page(client):
    res = client.get("/api/myfam", params={"interests": "astrology,;;"})
    assert res.status_code == 200


# --- the recap over HTTP --------------------------------------------------


def test_the_recap_needs_an_account(client):
    assert client.get("/api/recap").status_code == 401
    assert client.post("/api/recap/seen").status_code == 401


def test_a_listener_who_heard_nothing_is_told_so_rather_than_shown_a_recap(client):
    sign_up(client)
    body = client.get("/api/recap").json()
    assert body["empty"] is True and body["reason"]
    assert body["played"] == 0 and not body["query"]


def test_a_real_week_becomes_one_episode_about_the_subjects_they_played(client):
    sign_up(client)
    for _ in range(3):
        client.post("/api/event", json={"kind": "complete", "topic_id": "ai-agents"})
    body = client.get("/api/recap").json()
    assert body["empty"] is False
    assert body["finished"] == 3 and "tech" in body["subjects"]
    assert "technology" in body["query"].lower()


def test_marking_it_seen_settles_the_week(client):
    sign_up(client)
    assert client.get("/api/recap").json()["due"] is True
    client.post("/api/recap/seen")
    assert client.get("/api/recap").json()["due"] is False


def test_disabling_weekly_notifications_sticks(client):
    sign_up(client)
    client.post("/api/preferences", json={"weekly_recap": False})
    body = client.get("/api/recap").json()
    assert body["enabled"] is False and body["due"] is False


# --- what skip mode still gets -------------------------------------------


def test_skip_mode_can_still_hear_everything(client):
    """The line the gate must not cross. Nothing that leads to audio is gated,
    because a login in front of the first word breaks the one-sentence spec."""
    for path in ("/api/myfam", "/api/topics", "/api/explore", "/api/godeeper",
                 "/api/profile", "/api/nextup", "/api/explorenew", "/api/voices"):
        assert client.get(path).status_code == 200, f"{path} was gated"


def test_skip_mode_cannot_save_a_mix(client):
    assert client.get("/api/mixes").status_code == 401
    assert client.post("/api/mixes", json={"name": "Mine"}).status_code == 401


def test_signing_up_later_keeps_what_they_already_did(client):
    """accounts.py's whole design, restated as a gate consequence: skip mode
    then signup must not read as starting over."""
    client.post("/api/event", json={"kind": "complete", "topic_id": "ai-agents"})
    before = client.get("/api/auth/me").json()["user_id"]
    sign_up(client)
    assert client.get("/api/auth/me").json()["user_id"] == before
    assert client.get("/api/recap").json()["finished"] == 1
