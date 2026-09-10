"""The server as something other than its own web page.

A version prefix, a second way to carry a session, and limits that bite.
"""
from __future__ import annotations

import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import accounts as accounts_mod
import app as appmod
import entitlements
import quotas


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(appmod, "_rate_limit", lambda request: None)
    with TestClient(appmod.app) as c:
        yield c


@pytest.fixture
def enforced(monkeypatch):
    """Limits on, and small enough to reach inside a test."""
    monkeypatch.setattr(quotas, "settings_enforcing", lambda: True)
    monkeypatch.setenv("FREE_EPISODES_PER_DAY", "2")
    monkeypatch.setenv("FREE_EXPLORE_PER_DAY", "3")
    entitlements.reload_tiers()
    yield
    for name in entitlements.LIMIT_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    entitlements.reload_tiers()


# --- the version prefix ---------------------------------------------------

def test_every_endpoint_answers_under_the_version_prefix(client):
    """An app on somebody's phone cannot be redeployed with the server. The
    prefix is what lets `/api/v2` carry a changed shape later without breaking
    every copy already installed."""
    for path in ("/health", "/entitlements", "/plans", "/auth/me", "/voices"):
        plain = client.get("/api" + path)
        versioned = client.get("/api/v1" + path)
        assert versioned.status_code == plain.status_code == 200, path


def test_the_prefix_is_a_rewrite_and_not_a_second_set_of_routes(client):
    """Two registrations of one endpoint are two places for a decorator to
    drift, and the drift would show up as a native client quietly getting
    different behaviour from the web one."""
    assert client.get("/api/v1/plans").json() == client.get("/api/plans").json()


def test_the_health_endpoint_says_what_the_prefix_is(client):
    api = client.get("/api/health").json()["api"]
    assert api["prefix"] == "/api/v1"


def test_the_rewrite_happens_before_the_session_middleware_sees_the_path(client):
    """Ordering, and it is load-bearing.

    `carry_the_session` decides from the path whether to mint a session, and
    deliberately does not for `/api/health` - so a monitoring poll does not
    accumulate a session row per request. If the version rewrite ran *after*
    it, `/api/v1/health` would not match that exclusion and every poll from a
    versioned client would mint one. The two must behave identically.
    """
    plain = TestClient(appmod.app).get("/api/health")
    versioned = TestClient(appmod.app).get("/api/v1/health")
    assert accounts_mod.COOKIE_NAME not in plain.cookies
    assert accounts_mod.COOKIE_NAME not in versioned.cookies


def test_a_prefixed_path_that_does_not_exist_is_still_a_miss(client):
    assert client.get("/api/v1/nothing-here").status_code == 404


def test_the_prefix_does_not_swallow_a_similar_looking_path(client):
    """`/api/v10` must not be read as `/api/v1` plus "0"."""
    assert client.get("/api/v10/health").status_code == 404


# --- carrying a session without a cookie ---------------------------------

def test_a_bearer_token_identifies_the_same_listener_as_the_cookie(client):
    """What the iOS client will use. iOS clears its cookie jar under conditions
    the app does not control, and a listener silently becoming a different
    listener is the worst failure available to an append-only log keyed on
    that id."""
    signed_up = client.post("/api/auth/signup", json={
        "email": "a@b.com", "password": "password12", "want_token": True}).json()
    token = signed_up["session_token"]
    assert token

    bare = TestClient(appmod.app)
    me = bare.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.json()["user_id"] == signed_up["user_id"]
    assert me.json()["email"] == "a@b.com"


def test_a_token_is_only_returned_when_it_is_asked_for(client):
    """A browser must never ask. The cookie is HttpOnly precisely so page
    script cannot read it, and a web client that requests the token has undone
    that - so it is an explicit opt-in rather than something every response
    carries."""
    body = client.post("/api/auth/signup",
                       json={"email": "a@b.com", "password": "password12"}).json()
    assert "session_token" not in body


def test_a_made_up_bearer_token_is_simply_anonymous(client):
    """Not an error: an unknown session mints a new anonymous listener, exactly
    as a missing cookie does. What it must never do is resolve to somebody."""
    bare = TestClient(appmod.app)
    me = bare.get("/api/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
    assert me.status_code == 200
    assert me.json()["authenticated"] is False


def test_a_listener_id_is_still_never_accepted_from_the_client(client):
    """The settled rule, restated for the second carrier. A bearer token is
    server-minted and unforgeable; `?user=` never was."""
    mine = client.get("/api/auth/me").json()["user_id"]
    assert client.get("/api/auth/me?user=somebody-else").json()["user_id"] == mine


def test_the_cookie_wins_when_both_are_sent(client):
    """A browser attaches its cookie automatically, so a header alongside it is
    either a mistake or somebody testing whether it overrides. It does not."""
    first = client.post("/api/auth/signup", json={
        "email": "a@b.com", "password": "password12", "want_token": True}).json()
    other = TestClient(appmod.app)
    other_id = other.get("/api/auth/me").json()["user_id"]
    assert other_id != first["user_id"]
    still_theirs = other.get(
        "/api/auth/me", headers={"Authorization": f"Bearer {first['session_token']}"})
    assert still_theirs.json()["user_id"] == other_id


# --- what the client is told about its tier ------------------------------

def test_entitlements_work_without_an_account(client):
    """An anonymous listener is on a tier too. A limit nobody can see coming is
    indistinguishable from a bug when it arrives."""
    body = client.get("/api/entitlements").json()
    assert body["tier"] == "free"
    assert body["usage"]["episode"]["limit"] == entitlements.limit_for(
        "free", "episode").count
    assert "search" in body["features"]


def test_the_plans_endpoint_lists_every_tier_and_says_which_one_is_current(client):
    body = client.get("/api/plans").json()
    assert [t["name"] for t in body["tiers"]] == list(entitlements.TIERS)
    assert body["current"] == "free"


def test_health_says_whether_limits_are_being_enforced(client):
    """A server running with them off looks identical from the outside to one
    running with them on, which is exactly the thing worth being able to ask."""
    assert "enforced" in client.get("/api/health").json()["quotas"]


# --- limits that bite -----------------------------------------------------

def test_a_free_listener_runs_out_of_scripts(client, enforced):
    for _ in range(2):
        assert client.post("/api/script",
                           json={"query": "what is a bond", "minutes": 1}
                           ).status_code == 200
    over = client.post("/api/script", json={"query": "what is a bond", "minutes": 1})
    assert over.status_code == 429
    assert "2" in over.json()["error"]


def test_the_refusal_carries_the_whole_verdict_not_just_a_sentence(client, enforced):
    """So the interface can say what is left and when it comes back, rather
    than only that something was refused."""
    for _ in range(2):
        client.post("/api/script", json={"query": "q", "minutes": 1})
    over = client.post("/api/script", json={"query": "q", "minutes": 1})
    verdict = json.loads(over.headers["X-FAM-Quota"])
    assert verdict["remaining"] == 0
    assert verdict["limit"] == 2
    assert verdict["resets_at"] > 0
    assert verdict["tier"] == "free"


def test_the_allowance_is_visible_before_it_runs_out(client, enforced):
    client.post("/api/script", json={"query": "q", "minutes": 1})
    usage = client.get("/api/entitlements").json()["usage"]["episode"]
    assert usage["used"] == 1 and usage["remaining"] == 1


def test_a_better_tier_is_not_stopped_where_a_free_one_is(client, enforced):
    client.post("/api/auth/signup", json={"email": "a@b.com", "password": "password12"})
    appmod.ACCOUNTS.set_plan(client.get("/api/auth/me").json()["user_id"], "unlimited")
    for _ in range(6):
        assert client.post("/api/script", json={"query": "q", "minutes": 1}
                           ).status_code == 200


def test_the_tier_is_read_from_the_session_and_never_from_a_parameter(client, enforced):
    """A plan is worth money, so a client-supplied one is a client-supplied
    upgrade."""
    for _ in range(2):
        client.post("/api/script", json={"query": "q", "minutes": 1})
    over = client.post("/api/script?tier=unlimited",
                       json={"query": "q", "minutes": 1, "tier": "unlimited"})
    assert over.status_code == 429


def test_explore_replays_are_counted_separately_from_generation(client, enforced):
    """A replay provably cannot write a script, so it must not eat the
    allowance that pays for one."""
    for _ in range(2):
        client.post("/api/script", json={"query": "q", "minutes": 1})
    usage = client.get("/api/entitlements").json()["usage"]
    assert usage["episode"]["remaining"] == 0
    assert usage["explore"]["remaining"] == 3


def test_a_refused_request_does_not_reach_the_generator(client, enforced, monkeypatch):
    """The point of reserving before generating. A limit checked afterwards has
    already spent the money it was there to protect."""
    for _ in range(2):
        client.post("/api/script", json={"query": "q", "minutes": 1})

    called = []
    real = appmod.DemoGenerator

    class Watched(real):
        def stream_sentences(self, *a, **k):
            called.append(1)
            return super().stream_sentences(*a, **k)

    monkeypatch.setattr(appmod, "DemoGenerator", Watched)
    assert client.post("/api/script", json={"query": "q", "minutes": 1}
                       ).status_code == 429
    assert called == []


def test_with_enforcement_off_nothing_is_refused(client):
    """conftest leaves it off, which is what the rest of the suite runs under
    and what `demo.sh` does deliberately."""
    for _ in range(8):
        assert client.post("/api/script", json={"query": "q", "minutes": 1}
                           ).status_code == 200


def test_an_episode_that_played_counts_even_though_it_was_cheap(client, enforced):
    """The regression this locks down.

    An earlier refund rule gave the allowance back whenever no Claude call had
    been made - which is true of every cache hit, and of every episode in demo
    mode. The effect was that the limit worked on a cold server and quietly
    stopped working as the cache warmed up, which is the worst possible way for
    a spending control to fail. A listener heard an episode and the GPU
    produced it; only the model call was saved.
    """
    for _ in range(2):
        assert client.get("/api/audio?q=anything&minutes=1&fmt=pcm"
                          ).status_code == 200
    assert client.get("/api/audio?q=anything&minutes=1&fmt=pcm"
                      ).status_code == 429


def test_an_episode_that_failed_before_costing_anything_is_refunded(client, enforced,
                                                                   monkeypatch):
    """A server with no voice installed is the machine being broken, not the
    listener spending."""
    def no_voice(*_a, **_k):
        raise appmod.TTSUnavailable("no engine here")

    monkeypatch.setattr(appmod, "_make_pipeline", no_voice)
    assert client.get("/api/audio?q=anything&minutes=1&fmt=pcm").status_code == 503
    assert client.get("/api/entitlements").json()["usage"]["episode"]["used"] == 0


def test_a_tier_can_shorten_an_episode_but_never_lengthen_it(client, enforced,
                                                             monkeypatch):
    """Duration is the other lever on GPU cost. A request over the tier's
    ceiling is trimmed to it rather than refused - the listener gets an
    episode, just not a ten-minute one."""
    monkeypatch.setenv("FREE_MAX_MINUTES", "2")
    entitlements.reload_tiers()
    res = client.get("/api/audio?q=anything&minutes=8&fmt=pcm")
    assert res.status_code == 200
    assert int(res.headers["X-Requested-Seconds"]) <= 2 * 60


# --- cross-origin ---------------------------------------------------------

def test_no_origins_are_allowed_by_default(client):
    """Same-origin only until somebody names an origin. A wildcard would look
    permissive, fail anyway with credentials, and hide the real fix."""
    assert client.get("/api/health").json()["api"]["cors_origins"] == []
