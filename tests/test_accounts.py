"""Accounts, sessions, and the hole they close.

The property under test is not "login works". It is that a listener id can no
longer be *asserted* by whoever is asking. Before this, `?user=alice` was
enough to read Alice's profile, rename her, and delete her mixes.
"""
from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import accounts as A  # noqa: E402
import app as appmod  # noqa: E402
import mixes as M  # noqa: E402
import social as S  # noqa: E402
import topics as T  # noqa: E402

GOOD = "a-long-enough-password"


@pytest.fixture
def store(tmp_path):
    return A.AccountStore(str(tmp_path / "accounts.db"))


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(appmod, "_rate_limit", lambda request: None)
    monkeypatch.setattr(appmod, "_read_limit", lambda request: None)
    monkeypatch.setattr(appmod, "EVENTS", T.EventStore(str(tmp_path / "e.db")))
    monkeypatch.setattr(appmod, "SOCIAL", S.SocialStore(str(tmp_path / "s.db")))
    monkeypatch.setattr(appmod, "MIXES", M.MixStore(str(tmp_path / "m.db")))
    return TestClient(appmod.app)


def other(monkeypatch=None) -> TestClient:
    """A second browser: a separate cookie jar on the same app."""
    return TestClient(appmod.app)


# --- the hole ------------------------------------------------------------


def test_a_listener_id_can_no_longer_be_asserted_by_the_caller(client):
    """The whole point. `?user=` used to be identity; it is now ignored."""
    mine = client.get("/api/auth/me").json()["user_id"]
    client.post("/api/mixes", json={"name": "Mine"})

    attacker = other()
    listed = attacker.get("/api/mixes", params={"user": mine}).json()["mixes"]
    assert listed == [], "a mix was readable by naming its owner in the query"
    assert attacker.get("/api/auth/me").json()["user_id"] != mine


def test_naming_someone_else_cannot_rename_them(client):
    client.post("/api/me", json={"name": "Ian", "handle": "ian"})
    mine = client.get("/api/auth/me").json()["user_id"]

    other().post("/api/me", params={"user": mine},
                 json={"name": "Impostor", "handle": "impostor"})
    assert appmod.SOCIAL.person(mine)["name"] == "Ian"


def test_the_id_is_minted_by_the_server_not_the_browser(client):
    """High entropy and server-chosen, so it cannot be guessed into."""
    first = client.get("/api/auth/me").json()["user_id"]
    second = other().get("/api/auth/me").json()["user_id"]
    assert first != second
    assert first.startswith("anon_") and len(first) > 20


def test_the_session_cookie_is_httponly_so_a_script_cannot_lift_it(client):
    response = client.get("/api/auth/me")
    header = response.headers.get("set-cookie", "")
    assert A.COOKIE_NAME in header
    assert "HttpOnly" in header
    assert "SameSite=lax" in header.replace("samesite", "SameSite")


def test_a_health_poll_does_not_accumulate_sessions(client):
    """Monitoring hits every minute forever; they must not each leave a row."""
    for _ in range(3):
        client.get("/api/health")
    assert "set-cookie" not in client.get("/api/health").headers


# --- signing up claims the identity you already have ----------------------


def test_signing_up_keeps_the_history_the_listener_already_had(client):
    """No migration step: signup attaches credentials to the same user_id."""
    client.post("/api/event", json={"kind": "complete", "topic_id": "golf-evolution"})
    before = client.get("/api/auth/me").json()["user_id"]

    body = client.post("/api/auth/signup",
                       json={"email": "ian@example.com", "password": GOOD}).json()
    assert body["user_id"] == before, "signing up started a second identity"
    assert body["authenticated"] is True
    assert client.get("/api/profile").json()["finished"] == 1


def test_logging_in_elsewhere_reaches_the_same_data(client):
    """The reason to have accounts at all: another device, same listener."""
    client.post("/api/event", json={"kind": "complete", "topic_id": "golf-evolution"})
    client.post("/api/auth/signup", json={"email": "ian@example.com", "password": GOOD})
    mine = client.get("/api/auth/me").json()["user_id"]

    phone = other()
    assert phone.get("/api/auth/me").json()["user_id"] != mine
    phone.post("/api/auth/login", json={"email": "ian@example.com", "password": GOOD})
    assert phone.get("/api/auth/me").json()["user_id"] == mine
    assert phone.get("/api/profile").json()["finished"] == 1


def test_logging_out_leaves_a_working_app_as_a_new_listener(client):
    client.post("/api/auth/signup", json={"email": "ian@example.com", "password": GOOD})
    mine = client.get("/api/auth/me").json()["user_id"]
    client.post("/api/auth/logout")

    after = client.get("/api/auth/me").json()
    assert after["user_id"] != mine and after["authenticated"] is False
    # And the app still works without an account, which is the constraint that
    # stopped this becoming a login screen in front of the product.
    assert client.get("/api/myfam").status_code == 200


def test_an_account_cannot_be_attached_twice(client):
    client.post("/api/auth/signup", json={"email": "ian@example.com", "password": GOOD})
    again = client.post("/api/auth/signup",
                        json={"email": "other@example.com", "password": GOOD})
    assert again.status_code == 400


def test_a_taken_email_is_refused(client):
    client.post("/api/auth/signup", json={"email": "ian@example.com", "password": GOOD})
    res = other().post("/api/auth/signup",
                       json={"email": "IAN@example.com", "password": GOOD})
    assert res.status_code == 400, "email uniqueness must be case-folded"


# --- login ----------------------------------------------------------------


def test_a_wrong_password_is_refused(client):
    client.post("/api/auth/signup", json={"email": "ian@example.com", "password": GOOD})
    res = other().post("/api/auth/login",
                       json={"email": "ian@example.com", "password": "wrong-password"})
    assert res.status_code == 401


def test_login_does_not_say_which_half_was_wrong(client):
    """Different messages would turn the form into an account-enumeration
    oracle: try an address, learn whether it is registered."""
    client.post("/api/auth/signup", json={"email": "ian@example.com", "password": GOOD})
    unknown = other().post("/api/auth/login",
                           json={"email": "nobody@example.com", "password": GOOD})
    wrong = other().post("/api/auth/login",
                         json={"email": "ian@example.com", "password": "not-the-one"})
    assert unknown.json()["error"] == wrong.json()["error"]


def test_logging_in_mints_a_fresh_token(client):
    """Session fixation: a token captured before login must not work after."""
    client.post("/api/auth/signup", json={"email": "ian@example.com", "password": GOOD})
    client.post("/api/auth/logout")

    phone = other()
    stolen = phone.get("/api/auth/me")  # an anonymous session exists now
    before = phone.cookies.get(A.COOKIE_NAME)
    phone.post("/api/auth/login", json={"email": "ian@example.com", "password": GOOD})
    after = phone.cookies.get(A.COOKIE_NAME)
    assert stolen.status_code == 200
    assert before and after and before != after


def test_changing_a_password_logs_every_device_out(client):
    client.post("/api/auth/signup", json={"email": "ian@example.com", "password": GOOD})
    mine = client.get("/api/auth/me").json()["user_id"]
    phone = other()
    phone.post("/api/auth/login", json={"email": "ian@example.com", "password": GOOD})
    assert phone.get("/api/auth/me").json()["user_id"] == mine

    client.post("/api/auth/password", json={"current": GOOD, "new": "a-brand-new-one"})
    assert phone.get("/api/auth/me").json()["user_id"] != mine, "the old device stayed in"


# --- the store's own rules ------------------------------------------------


def test_a_password_is_never_stored_in_the_clear(store, tmp_path):
    store.sign_up("u1", "ian@example.com", GOOD)
    raw = (tmp_path / "accounts.db").read_bytes()
    assert GOOD.encode() not in raw


def test_a_session_token_is_never_stored(store, tmp_path):
    """A leaked database must not hand over live sessions."""
    token, _user = store.new_session("u1")
    assert token.encode() not in (tmp_path / "accounts.db").read_bytes()
    assert store.listener_for(token).user_id == "u1"


def test_an_unused_session_expires(store):
    """No intervening use, so the original expiry stands."""
    token, _user = store.new_session("u1", at=1000.0)
    assert store.listener_for(token, at=1000.0 + A.SESSION_TTL + 10) is None


def test_using_a_session_slides_its_expiry_forward(store):
    """Someone who keeps listening keeps their session. Written at most hourly,
    so a read endpoint does not become a writer on every request."""
    token, _user = store.new_session("u1", at=1000.0)
    # A use just before expiry, which refreshes it...
    assert store.listener_for(token, at=1000.0 + A.SESSION_TTL - 10) is not None
    # ...so a moment past the *original* expiry it is still valid.
    assert store.listener_for(token, at=1000.0 + A.SESSION_TTL + 10) is not None
    # But it does not live forever without use.
    assert store.listener_for(token, at=1000.0 + 3 * A.SESSION_TTL) is None


def test_an_unknown_token_is_nobody(store):
    store.new_session("u1")
    assert store.listener_for("not-a-real-token") is None
    assert store.listener_for("") is None


def test_ending_a_session_takes_effect_immediately(store):
    token, _user = store.new_session("u1")
    assert store.end_session(token) is True
    assert store.listener_for(token) is None


@pytest.mark.parametrize("password", ["", "short", "nine-char"])
def test_a_short_password_is_refused(store, password):
    with pytest.raises(A.AuthError):
        store.sign_up("u1", "ian@example.com", password)


@pytest.mark.parametrize("email", ["", "nope", "no@domain", "@example.com", "a b@c.com"])
def test_a_malformed_email_is_refused(store, email):
    with pytest.raises(A.AuthError):
        store.sign_up("u1", email, GOOD)


def test_the_hash_carries_its_parameters(store):
    """So the cost can be raised later without locking anyone out."""
    encoded = A.hash_password(GOOD)
    assert encoded.startswith(f"scrypt${A.SCRYPT_N}${A.SCRYPT_R}${A.SCRYPT_P}$")
    assert A.verify_password(GOOD, encoded)
    assert not A.verify_password("something else", encoded)


def test_two_identical_passwords_hash_differently(store):
    """Salted, so a leaked table does not reveal who shares a password."""
    assert A.hash_password(GOOD) != A.hash_password(GOOD)


def test_a_corrupt_hash_lets_nobody_in(store):
    for broken in ("", "scrypt$bad", "notscrypt$1$2$3$aaaa$bbbb", "$$$$$"):
        assert A.verify_password(GOOD, broken) is False
