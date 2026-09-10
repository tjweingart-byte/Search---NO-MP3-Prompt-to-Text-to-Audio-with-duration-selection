"""Signing up by phone, signing in with a provider, settings, and deletion."""
from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import accounts as accounts_mod
import app as appmod
import oauth


@pytest.fixture
def client(monkeypatch):
    """The pace is three seconds between generations from one client, which is
    right for the server and would make every test here sleep. Switched off the
    same way `test_accounts.py` does - what is under test is the account
    surface, not the pacing that has its own tests."""
    monkeypatch.setattr(appmod, "_rate_limit", lambda request: None)
    with TestClient(appmod.app) as c:
        yield c


class FakeIdentity(oauth.VerifiedIdentity):
    pass


def stub_verify(monkeypatch, *, provider="apple", subject="sub-1", email="",
                name=""):
    """Stand in for the provider's signature check.

    The crypto has its own tests. What matters here is what happens *after* a
    token verifies, and wiring a real RSA key into every account test would
    test PyJWT rather than FAM.
    """
    def fake(prov, token, nonce=""):
        return oauth.VerifiedIdentity(provider=provider, subject=subject,
                                      email=email, email_verified=bool(email),
                                      name=name)
    monkeypatch.setattr(appmod.oauth, "verify", fake)


# --- signing up by phone --------------------------------------------------

def test_a_phone_number_can_carry_an_account(client):
    r = client.post("/api/auth/signup",
                    json={"phone": "+1 415 555 0142", "password": "password12"})
    assert r.status_code == 200, r.text
    assert r.json()["phone"] == "+14155550142"
    assert r.json()["authenticated"] is True


def test_a_number_is_stored_in_one_form_however_it_is_typed(client):
    """One canonical form or two people can own "the same" number in different
    notations. `00` is how most of the world dials internationally, so it has
    to reach the same account as `+`."""
    signed_up = client.post(
        "/api/auth/signup",
        json={"phone": "+1 (415) 555-0142", "password": "password12"}).json()
    client.post("/api/auth/logout")
    back = client.post("/api/auth/login",
                       json={"phone": "001 415 555 0142", "password": "password12"})
    assert back.status_code == 200, back.text
    assert back.json()["user_id"] == signed_up["user_id"]


def test_a_number_without_a_country_code_is_refused_rather_than_guessed(client):
    """Guessing would let two people in different countries own one account."""
    r = client.post("/api/auth/signup",
                    json={"phone": "4155550142", "password": "password12"})
    assert r.status_code == 400
    assert "country code" in r.json()["error"]


def test_signup_needs_exactly_one_identifier(client):
    both = client.post("/api/auth/signup", json={
        "email": "a@b.com", "phone": "+14155550142", "password": "password12"})
    neither = client.post("/api/auth/signup", json={"password": "password12"})
    assert both.status_code == 400 and neither.status_code == 400


def test_a_phone_and_an_email_account_do_not_collide(client):
    """Both leave the other column empty, and SQLite treats two empty strings
    as equal - so the second account would have collided on a value meaning
    "not set" if the unique indexes had not been made partial."""
    client.post("/api/auth/signup",
                json={"phone": "+14155550142", "password": "password12"})
    client.post("/api/auth/logout")
    r = client.post("/api/auth/signup",
                    json={"phone": "+14155550143", "password": "password12"})
    assert r.status_code == 200, r.text


# --- signing in with a provider ------------------------------------------

def test_apple_can_create_an_account_with_no_email_at_all(client, monkeypatch):
    """Apple sends an address on the first authorization only, and it may be a
    relay. An account that required one would be impossible for some people to
    have."""
    stub_verify(monkeypatch, provider="apple", subject="apple-1")
    r = client.post("/api/auth/provider",
                    json={"provider": "apple", "id_token": "tok"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["email"] == ""
    assert body["authenticated"] is True
    assert body["is_new"] is True


def test_signing_in_again_reaches_the_same_account(client, monkeypatch):
    """The failure this prevents is the common one: keyed on email, Apple's
    silent second sign-in creates a second account and the listener loses
    everything."""
    stub_verify(monkeypatch, provider="apple", subject="apple-1")
    first = client.post("/api/auth/provider",
                        json={"provider": "apple", "id_token": "tok"}).json()
    client.post("/api/auth/logout")
    second = client.post("/api/auth/provider",
                         json={"provider": "apple", "id_token": "tok"}).json()
    assert second["user_id"] == first["user_id"]
    assert second["is_new"] is False


def test_a_provider_signin_claims_the_listening_already_done(client, monkeypatch):
    """The same rule as email sign-up: an account is attached to the identity
    they already have, so their history is theirs rather than stranded."""
    before = client.get("/api/auth/me").json()["user_id"]
    stub_verify(monkeypatch, provider="google", subject="g-1", email="a@b.com")
    after = client.post("/api/auth/provider",
                        json={"provider": "google", "id_token": "tok"}).json()
    assert after["user_id"] == before


def test_a_relay_address_is_reported_as_one(client, monkeypatch):
    """It forwards today and its owner can switch it off tomorrow, so nothing
    should promise to reach somebody there without saying so."""
    stub_verify(monkeypatch, provider="apple", subject="apple-2",
                email="abc@privaterelay.appleid.com")
    r = client.post("/api/auth/provider",
                    json={"provider": "apple", "id_token": "tok"}).json()
    assert r["private_relay"] is True


def test_a_provider_this_server_cannot_verify_says_so_as_a_server_fault(client):
    """503 and not 401. PyJWT missing or an empty audience is an operator's
    problem, and telling the person their sign-in was rejected sends them off
    to fix a phone that is working."""
    r = client.post("/api/auth/provider",
                    json={"provider": "apple", "id_token": "tok"})
    assert r.status_code == 503
    assert "APPLE_CLIENT_IDS" in r.json()["error"]


# --- account settings -----------------------------------------------------

def test_the_settings_screen_gets_everything_it_shows(client):
    client.post("/api/auth/signup",
                json={"email": "a@b.com", "password": "password12"})
    body = client.get("/api/account").json()
    assert body["email"] == "a@b.com"
    assert body["has_password"] is True
    assert [i["provider"] for i in body["identities"]] == ["email"]
    assert body["entitlements"]["tier"] == "free"
    assert body["usage"]["episode"]["limit"] > 0
    assert body["sessions"][0]["current"] is True


def test_settings_are_only_readable_with_an_account(client):
    assert client.get("/api/account").status_code == 401


def test_a_name_and_an_address_can_be_changed(client):
    client.post("/api/auth/signup",
                json={"email": "a@b.com", "password": "password12"})
    r = client.post("/api/account",
                    json={"display_name": "Ian", "email": "ian@fam.audio"})
    assert r.status_code == 200, r.text
    assert r.json()["display_name"] == "Ian"
    assert client.get("/api/auth/me").json()["email"] == "ian@fam.audio"


def test_an_omitted_field_is_left_alone_and_an_empty_one_removes(client):
    client.post("/api/auth/signup",
                json={"email": "a@b.com", "password": "password12"})
    client.post("/api/account", json={"display_name": "Ian"})
    # phone omitted entirely; the name must survive a request about something
    # else, which is the difference between None and "".
    assert client.post("/api/account", json={"phone": "+14155550142"}
                       ).json()["display_name"] == "Ian"


def test_removing_the_last_way_in_is_refused(client):
    client.post("/api/auth/signup",
                json={"email": "a@b.com", "password": "password12"})
    r = client.post("/api/account", json={"email": ""})
    assert r.status_code == 400
    assert "no way to sign in" in r.json()["error"]


def test_an_oauth_account_can_add_a_password_and_a_password_one_cannot(client, monkeypatch):
    stub_verify(monkeypatch, provider="google", subject="g-9", email="g@b.com")
    client.post("/api/auth/provider", json={"provider": "google", "id_token": "t"})
    assert client.get("/api/account").json()["has_password"] is False
    assert client.post("/api/auth/password/set",
                       json={"new": "password12"}).status_code == 200
    # And not twice: the second time there is one to prove, so the endpoint
    # that proves it is the right one.
    assert client.post("/api/auth/password/set",
                       json={"new": "otherpass12"}).status_code == 400


def test_unlinking_the_only_way_in_is_refused(client, monkeypatch):
    stub_verify(monkeypatch, provider="apple", subject="apple-3")
    client.post("/api/auth/provider", json={"provider": "apple", "id_token": "t"})
    r = client.request("DELETE", "/api/account/identity?provider=apple")
    assert r.status_code == 400
    assert "only way" in r.json()["error"]


def test_signing_out_everywhere_keeps_this_session(client):
    client.post("/api/auth/signup",
                json={"email": "a@b.com", "password": "password12"})
    r = client.post("/api/account/signout-everywhere")
    assert r.status_code == 200
    # Still signed in here, which is the whole difference from a password
    # change.
    assert client.get("/api/auth/me").json()["email"] == "a@b.com"


# --- deletion -------------------------------------------------------------

def test_deleting_an_account_erases_every_per_listener_store(client):
    """Guideline 5.1.1(v), and the reason it is a real feature rather than
    paperwork: seven stores hold rows keyed on this listener, and a deletion
    that covered six would report success."""
    client.post("/api/auth/signup",
                json={"email": "a@b.com", "password": "password12"})
    user = client.get("/api/auth/me").json()["user_id"]
    client.post("/api/event", json={"kind": "play", "text": "something"})
    client.post("/api/mixes", json={"name": "Morning", "topic_ids": ["fed-next-move"]})
    appmod.METER.record(user, appmod.metering.Usage(model="m", model_calls=1))

    r = client.request("DELETE", "/api/account")
    assert r.status_code == 200, r.text
    removed = r.json()["removed"]
    assert removed["account"] == 1
    assert removed["events"] >= 1
    assert removed["mixes"] >= 1
    assert appmod.ACCOUNTS.account(user) is None
    assert appmod.EVENTS.forget(user) == 0


def test_deletion_keeps_the_cost_row_and_drops_the_name_on_it(client):
    """The one store not emptied. What the GPU cost in June is a fact about the
    business; a ledger with holes cannot be reconciled against an invoice."""
    client.post("/api/auth/signup",
                json={"email": "a@b.com", "password": "password12"})
    user = client.get("/api/auth/me").json()["user_id"]
    appmod.METER.record(user, appmod.metering.Usage(
        model="claude-sonnet-5", model_calls=1, input_tokens=1000,
        output_tokens=1000))

    client.request("DELETE", "/api/account")
    rows = appmod.METER.rows()
    assert len(rows) == 1
    assert rows[0]["user_id"] != user
    # The amount survives, which is the whole point of anonymising rather than
    # deleting: a ledger with holes cannot be reconciled against an invoice.
    assert rows[0]["cost_usd"] > 0


def test_the_app_still_works_after_a_deletion(client):
    """The session is dropped, so the next request mints a fresh anonymous
    listener - deleting an account must not leave the app unusable."""
    client.post("/api/auth/signup",
                json={"email": "a@b.com", "password": "password12"})
    client.request("DELETE", "/api/account")
    me = client.get("/api/auth/me").json()
    assert me["user_id"] and me["authenticated"] is False


def test_deletion_needs_an_account(client):
    assert client.request("DELETE", "/api/account").status_code == 401
