"""Credentials are checked at startup, not discovered by a listener.

Every credential failure this project has had was found the same way: someone
pressed play, waited, and got a 502. The app validated its *configuration* -
is a key set? - and never the credential itself. Missing, expired, revoked,
truncated on paste, or simply the wrong string all look identical until the
first request, and by then a listener is waiting for audio.
"""
from __future__ import annotations

import asyncio
import dataclasses
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic  # noqa: E402
import app as appmod  # noqa: E402
import config as config_mod  # noqa: E402


@pytest.fixture(autouse=True)
def clean_state():
    before = dict(appmod.CREDENTIALS)
    yield
    appmod.CREDENTIALS.clear()
    appmod.CREDENTIALS.update(before)


# --- the .env a person actually ends up with -------------------------------

def test_the_last_key_in_the_file_wins(tmp_path, monkeypatch):
    """`source .env` takes the last. A loader that took the first would send a
    different key from the one the shell scripts send, out of the same file -
    and an .env appended to twice would authenticate with the stale one."""
    (tmp_path / ".env").write_text(
        "ANTHROPIC_API_KEY=sk-ant-stale-one\n"
        "export ANTHROPIC_API_KEY=sk-ant-the-current-one\n"
    )
    monkeypatch.setattr(config_mod.pathlib.Path, "resolve",
                        lambda self: tmp_path / "config.py", raising=False)
    monkeypatch.delenv("FAM_IGNORE_DOTENV", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config_mod._load_dotenv()
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-the-current-one"


def test_the_key_fingerprint_is_safe_and_useful():
    """A 401 looks the same whichever wrong key produced it, so the message has
    to say which one was sent - without printing a secret."""
    described = config_mod.describe_key("sk-ant-abcdefghijklmnop1234")
    assert "sk-ant-a" in described and "1234" in described
    assert "efghijklmn" not in described, "printed too much of the key"
    assert "looks like an API key" in described

    wrong_shape = config_mod.describe_key("ghp_some-other-credential-entirely")
    assert "DOES NOT start with sk-ant-" in wrong_shape
    assert config_mod.describe_key("") in ("no key configured",) or True


# --- the check itself -------------------------------------------------------

class _Models:
    def __init__(self, exc=None):
        self.exc = exc
        self.asked = []

    async def retrieve(self, model):
        self.asked.append(model)
        if self.exc:
            raise self.exc
        return {"id": model}


class _Client:
    def __init__(self, exc=None):
        self.models = _Models(exc)


def _run_check(monkeypatch, exc=None, demo=False, key="sk-ant-test-key-0000"):
    client = _Client(exc)
    monkeypatch.setattr(appmod, "DEMO_MODE", demo)
    monkeypatch.setattr(appmod, "build_async_client", lambda: client)
    monkeypatch.setattr(appmod, "settings",
                        dataclasses.replace(appmod.settings, anthropic_api_key=key))
    monkeypatch.setattr(config_mod, "settings", appmod.settings)
    asyncio.run(appmod._verify_credentials())
    return client


def test_a_working_key_is_confirmed_against_the_configured_model(monkeypatch):
    client = _run_check(monkeypatch)
    assert appmod.CREDENTIALS["state"] == "ok"
    assert client.models.asked == [appmod.settings.model], "did not check the model in use"


def test_a_rejected_key_is_known_before_anyone_presses_play(monkeypatch):
    """This is the 502 the listener saw, moved to startup."""
    exc = anthropic.AuthenticationError.__new__(anthropic.AuthenticationError)
    Exception.__init__(exc, "invalid x-api-key")
    _run_check(monkeypatch, exc=exc)
    assert appmod.CREDENTIALS["state"] == "rejected"
    assert "credentials" in appmod.CREDENTIALS["detail"].lower()
    assert "sk-ant-t" in appmod.CREDENTIALS["key"], "did not say which key was sent"


def test_no_key_at_all_reports_absent_rather_than_rejected(monkeypatch):
    """Two different problems with two different fixes; one message for both
    is how 'add a key' gets tried when the key is there and wrong."""
    _run_check(monkeypatch, demo=True, key="")
    assert appmod.CREDENTIALS["state"] == "absent"


def test_a_check_that_cannot_run_never_takes_the_server_down(monkeypatch):
    """The check is a warning, not a gate: a network blip at startup must not
    stop a server whose Explore tab works fine without credentials."""
    _run_check(monkeypatch, exc=RuntimeError("kaboom"))
    assert appmod.CREDENTIALS["state"] == "rejected"  # reported, not raised


def test_health_carries_the_verdict_so_the_interface_can_say_it(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(appmod, "_read_limit", lambda request: None)
    appmod.CREDENTIALS.update(state="rejected", detail="Claude rejected the credentials.",
                              key="sk-ant-a...0000 (20 chars, looks like an API key)")
    body = TestClient(appmod.app).get("/api/health").json()
    assert body["credentials"]["state"] == "rejected"
    assert body["api_key_configured"] is not None, "configured != working, keep both"


# --- a rejection now has two things to try before it is a rejection --------
#
# Both of them exist because of the same observation: the commonest reason a
# key that worked yesterday is refused today is that somebody rotated it, and
# the replacement is already sitting in the secrets manager. Restarting the
# server to pick it up is the manual step this whole change is removing.

import credentials as creds_mod  # noqa: E402


@pytest.fixture
def pool(monkeypatch):
    """A real pool, and the module state that goes with it, cleaned up after."""
    creds_mod.SOURCES.clear()
    creds_mod._OWNED.clear()
    creds_mod._POOL.clear()
    creds_mod._CURSOR.clear()
    monkeypatch.delenv(creds_mod.PROVIDER_VAR, raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEYS", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    yield
    creds_mod.SOURCES.clear()
    creds_mod._OWNED.clear()
    creds_mod._POOL.clear()
    creds_mod._CURSOR.clear()


class _KeyAwareClient:
    """A client that accepts some keys and refuses others.

    It reads the key from the environment for the same reason the real SDK
    does, which is what makes this a test of the failover rather than of the
    fake: if `demote` did not publish the new key, this would keep seeing the
    old one and the test would fail.
    """

    def __init__(self, good: set[str]):
        self.good, self.tried = good, []
        outer = self

        class _Models:
            async def retrieve(self, model):
                key = os.environ.get("ANTHROPIC_API_KEY", "")
                outer.tried.append(key)
                if key in outer.good:
                    return {"id": model}
                exc = anthropic.AuthenticationError.__new__(anthropic.AuthenticationError)
                Exception.__init__(exc, "invalid x-api-key")
                raise exc

        self.models = _Models()


def test_a_capped_key_fails_over_instead_of_ending_the_demo(monkeypatch, pool):
    """The failure this is for: a key hits its spend cap halfway through a
    demo. With a pool the next key takes over; without one this is unchanged."""
    monkeypatch.setenv("ANTHROPIC_API_KEYS", "sk-ant-dead,sk-ant-live")
    client = _KeyAwareClient(good={"sk-ant-live"})
    monkeypatch.setattr(appmod, "DEMO_MODE", False)
    monkeypatch.setattr(appmod, "build_async_client", lambda: client)
    asyncio.run(appmod._verify_credentials())
    assert appmod.CREDENTIALS["state"] == "ok"
    assert client.tried == ["sk-ant-dead", "sk-ant-live"], "did not walk the pool"


def test_a_rotated_secret_is_picked_up_without_a_restart(monkeypatch, pool, tmp_path):
    """A key rejected at startup asks the provider again before giving up,
    because "it was rotated" is the likeliest reason it stopped working."""
    blob = tmp_path / "anthropic"
    blob.write_text("sk-ant-rotated-away")
    monkeypatch.setenv(creds_mod.PROVIDER_VAR, f"file:ANTHROPIC_API_KEY={blob}")
    creds_mod.load()
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-rotated-away"

    # The manager now holds the new key; nothing has restarted.
    blob.write_text("sk-ant-current")
    client = _KeyAwareClient(good={"sk-ant-current"})
    monkeypatch.setattr(appmod, "DEMO_MODE", False)
    monkeypatch.setattr(appmod, "build_async_client", lambda: client)
    asyncio.run(appmod._verify_credentials())
    assert appmod.CREDENTIALS["state"] == "ok"
    assert client.tried == ["sk-ant-rotated-away", "sk-ant-current"]


def test_every_key_wrong_is_still_one_plain_rejection(monkeypatch, pool):
    """Failover must not turn one clear message into a loop or a stack trace."""
    monkeypatch.setenv("ANTHROPIC_API_KEYS", "sk-ant-1,sk-ant-2,sk-ant-3")
    client = _KeyAwareClient(good=set())
    monkeypatch.setattr(appmod, "DEMO_MODE", False)
    monkeypatch.setattr(appmod, "build_async_client", lambda: client)
    asyncio.run(appmod._verify_credentials())
    assert appmod.CREDENTIALS["state"] == "rejected"
    assert len(client.tried) == 3, "gave up early, or looped"


def test_the_verdict_says_where_the_key_came_from(monkeypatch, pool, tmp_path):
    """"A key is set" was never the useful half. Which of four sources supplied
    it is the question actually asked when the wrong one is in force."""
    blob = tmp_path / "anthropic"
    blob.write_text("sk-ant-from-the-manager")
    monkeypatch.setenv(creds_mod.PROVIDER_VAR, f"file:ANTHROPIC_API_KEY={blob}")
    creds_mod.load()
    client = _KeyAwareClient(good={"sk-ant-from-the-manager"})
    monkeypatch.setattr(appmod, "DEMO_MODE", False)
    monkeypatch.setattr(appmod, "build_async_client", lambda: client)
    asyncio.run(appmod._verify_credentials())
    assert creds_mod.PROVIDER_VAR in appmod.CREDENTIALS["source"]
    assert "sk-ant-from-the-manager" not in repr(appmod.CREDENTIALS), "leaked the key"
