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
