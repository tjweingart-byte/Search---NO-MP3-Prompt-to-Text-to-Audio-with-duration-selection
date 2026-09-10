"""Verifying a provider token, and refusing to pretend when it cannot be done."""
from __future__ import annotations

import base64
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import oauth


@pytest.fixture
def configured(monkeypatch):
    """Audiences set, as a deployed server would have them."""
    import dataclasses

    monkeypatch.setattr(config, "settings", dataclasses.replace(
        config.settings, google_client_ids="com.fam.app,web-client-id",
        apple_client_ids="com.fam.app"))


def unsigned(claims: dict) -> str:
    """A JWT with a real shape and no signature worth anything.

    Which is the point: this is exactly what an attacker sends. Anyone can
    write one saying they are anybody, because the envelope is plain base64.
    """
    def part(obj):
        raw = json.dumps(obj).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
    return f"{part({'alg': 'RS256', 'kid': 'x'})}.{part(claims)}.bm90LWEtc2ln"


# --- the rule that carries the security ----------------------------------

def test_an_unsigned_token_is_never_accepted(configured):
    """**The** test in this file.

    A JWT is a base64 envelope, so an unverified one is not a weak credential -
    it is anybody's credential. There must be no path, no setting and no
    missing library that turns this into a sign-in.
    """
    token = unsigned({"iss": "https://appleid.apple.com", "sub": "victim",
                      "aud": "com.fam.app", "email": "victim@example.com",
                      "exp": 9999999999, "iat": 1})
    with pytest.raises((oauth.OAuthError, oauth.OAuthUnavailable)):
        oauth.verify("apple", token)


def test_a_missing_library_is_unavailable_rather_than_permissive(monkeypatch, configured):
    """The degradation that must not exist. Every other optional dependency in
    this project falls back to something lesser and says so; this one has no
    lesser thing to fall back to."""
    monkeypatch.setattr(oauth, "_library",
                        lambda: (_ for _ in ()).throw(oauth.OAuthUnavailable("no PyJWT")))
    ready, reason = oauth.available("apple")
    assert ready is False and "no PyJWT" in reason
    with pytest.raises(oauth.OAuthUnavailable):
        oauth.verify("apple", unsigned({"sub": "x"}))


def test_an_unconfigured_audience_switches_the_provider_off(monkeypatch):
    """Without an audience check, a token minted for a completely different
    app - which its developers can read - would log in here. So no audience
    configured means the provider is off, not that the check is skipped."""
    import dataclasses

    monkeypatch.setattr(config, "settings", dataclasses.replace(
        config.settings, apple_client_ids=""))
    ready, reason = oauth.available("apple")
    assert ready is False
    assert "APPLE_CLIENT_IDS" in reason


def test_an_unknown_provider_is_refused(configured):
    with pytest.raises(oauth.OAuthUnavailable):
        oauth.verify("facebook", unsigned({"sub": "x"}))


def test_an_empty_token_is_refused_as_the_persons_problem(configured):
    """OAuthError, not OAuthUnavailable: the server is fine and the request is
    not, and the two get different status codes for that reason."""
    with pytest.raises(oauth.OAuthError):
        oauth.verify("apple", "")


# --- configuration reporting ---------------------------------------------

def test_the_report_answers_per_provider(configured):
    """"Sign-in works" is not a fact. A server with Google configured and Apple
    not is the normal case, and one boolean would describe neither."""
    report = oauth.report()
    assert set(report) == set(oauth.PROVIDERS)
    for entry in report.values():
        assert "ready" in entry and "reason" in entry


def test_a_reason_names_the_thing_to_change(monkeypatch):
    """Whoever reads this is trying to fix it. "Not configured" is not a fix."""
    import dataclasses

    monkeypatch.setattr(config, "settings", dataclasses.replace(
        config.settings, google_client_ids=""))
    reason = oauth.report()["google"]["reason"]
    assert "GOOGLE_CLIENT_IDS" in reason


def test_audiences_are_split_and_trimmed(configured):
    assert oauth.audiences("google") == ("com.fam.app", "web-client-id")


# --- claims ---------------------------------------------------------------

def test_a_nonce_matches_raw_or_hashed():
    """Google echoes the nonce as given; Apple is sent a SHA-256 of it and the
    token carries the hash. Accepting either means one client contract - send
    the raw nonce you used - instead of a per-provider rule."""
    import hashlib

    raw = "abc123"
    hashed = hashlib.sha256(raw.encode()).hexdigest()
    assert oauth._matches_nonce(raw, raw)
    assert oauth._matches_nonce(hashed, raw)
    assert not oauth._matches_nonce("something-else", raw)
    assert not oauth._matches_nonce("", raw)


def test_a_relay_address_is_recognised():
    """It forwards today and can be switched off tomorrow, so nothing may
    promise to reach somebody there."""
    relay = oauth.VerifiedIdentity("apple", "s", "x@privaterelay.appleid.com")
    real = oauth.VerifiedIdentity("apple", "s", "x@example.com")
    assert relay.is_private_relay and not real.is_private_relay


def test_the_subject_is_never_returned_whole_by_the_account_store(tmp_path):
    """A provider subject is a stable per-person identifier. There is no reason
    for it to leave the server; the settings screen only needs to tell two
    links apart."""
    import accounts

    store = accounts.AccountStore(str(tmp_path / "a.db"))
    _token, user = store.new_session()
    store.sign_in_with("google", "a-very-long-google-subject-1234567890",
                       email="a@b.com", current_user_id=user)
    listed = store.identities_for(user)[0]
    assert listed["subject_hint"] not in ("a-very-long-google-subject-1234567890",)
    assert len(listed["subject_hint"]) <= 8


# --- the real verifier, when the library is installed --------------------

def test_a_forged_token_is_refused_by_the_real_verifier(configured):
    """Skipped where PyJWT is absent - which is CI and this build container, so
    the check above is the one that always runs. Here to catch the day the
    dependency arrives and the wiring is wrong."""
    ready, reason = oauth.available("apple")
    if not ready:
        # `importorskip` cannot be used: a broken crypto backend raises a
        # BaseException that it does not catch, which is the whole reason
        # `_library` catches one.
        pytest.skip(reason)
    token = unsigned({"iss": "https://appleid.apple.com", "sub": "victim",
                      "aud": "com.fam.app", "exp": 9999999999, "iat": 1})
    with pytest.raises((oauth.OAuthError, oauth.OAuthUnavailable)):
        oauth.verify("apple", token)
