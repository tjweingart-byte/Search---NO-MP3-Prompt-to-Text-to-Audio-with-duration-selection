"""Storing the Exa key once per machine, and proving it before storing it.

`RESEARCH_BACKEND` defaults to `exa`, so a researched episode needs a second
credential. The rule for the first one applies unchanged to this one:

* it lives in `~/.fam/env`, outside any project folder, so a new copy of the
  app finds it already there rather than asking again;
* it is never written into source, because a key in a commit stays in the
  history after the line is deleted;
* and nothing is stored until the key is proved to work. "A key is set" is not
  "the key works" (PROBLEMS.md 52), and a bad key stored is worse than none -
  the server starts, reports research as configured, and fails on the first
  researched episode.

Nothing here reaches Exa or the network.
"""
from __future__ import annotations

import os
import stat
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import setup_key  # noqa: E402


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    """A throwaway ~/.fam/env, so no test can touch a real one."""
    path = tmp_path / "fam" / "env"
    monkeypatch.setattr(setup_key, "shared_env_path", lambda: path)
    return path


@pytest.fixture
def exa_ok(monkeypatch):
    """Exa accepting the key, without a network call."""
    class Reply:
        results = [object()]

    class FakeExa:
        def __init__(self, key):
            self.key = key

        def search_and_contents(self, query, **kwargs):
            return Reply()

    module = types.ModuleType("exa_py")
    module.Exa = FakeExa
    monkeypatch.setitem(sys.modules, "exa_py", module)
    return module


# --------------------------------------------------------------------------
# the two keys share one file and must not evict each other
# --------------------------------------------------------------------------
def test_storing_the_exa_key_keeps_the_anthropic_one(env_file):
    """The regression this file exists for. One file, two credentials - and
    writing one must not drop the other, or setting up research would silently
    stop the app writing episodes at all."""
    setup_key.write_key("sk-ant-first", setup_key.VAR)
    setup_key.write_key("exa-second", setup_key.EXA_VAR)

    text = env_file.read_text()
    assert "ANTHROPIC_API_KEY=sk-ant-first" in text
    assert "EXA_API_KEY=exa-second" in text


def test_storing_the_anthropic_key_keeps_the_exa_one(env_file):
    setup_key.write_key("exa-first", setup_key.EXA_VAR)
    setup_key.write_key("sk-ant-second", setup_key.VAR)

    text = env_file.read_text()
    assert "EXA_API_KEY=exa-first" in text
    assert "ANTHROPIC_API_KEY=sk-ant-second" in text


def test_a_replaced_key_leaves_exactly_one_line(env_file):
    """Two lines setting the same variable means which key is sent depends on
    who reads the file - which has already cost a session of debugging a
    perfectly valid key."""
    for value in ("exa-one", "exa-two", "exa-three"):
        setup_key.write_key(value, setup_key.EXA_VAR)

    lines = [l for l in env_file.read_text().splitlines()
             if l.startswith("EXA_API_KEY=")]
    assert lines == ["EXA_API_KEY=exa-three"]


def test_the_file_is_readable_only_by_its_owner(env_file):
    setup_key.write_key("exa-secret", setup_key.EXA_VAR)
    mode = stat.S_IMODE(env_file.stat().st_mode)
    assert mode == stat.S_IRUSR | stat.S_IWUSR, oct(mode)


def test_removing_one_key_leaves_the_other(env_file):
    setup_key.write_key("sk-ant-keep", setup_key.VAR)
    setup_key.write_key("exa-drop", setup_key.EXA_VAR)
    env_file.write_text("\n".join(setup_key.without(setup_key.EXA_VAR) + [""]))

    text = env_file.read_text()
    assert "ANTHROPIC_API_KEY=sk-ant-keep" in text
    assert "EXA_API_KEY" not in text


# --------------------------------------------------------------------------
# verify, do not inspect
# --------------------------------------------------------------------------
def test_a_working_key_is_accepted_by_asking_exa(exa_ok, monkeypatch):
    import asyncio

    monkeypatch.setenv("EXA_API_KEY", "")
    ok, detail = asyncio.run(setup_key.exa_works("exa-good"))
    assert ok and "returned 1 result" in detail


def test_a_rejected_key_is_never_stored(env_file, monkeypatch):
    """The whole point of checking first."""
    import asyncio

    class Refuses:
        def __init__(self, key):
            pass

        def search_and_contents(self, query, **kwargs):
            raise RuntimeError("401 unauthorized")

    module = types.ModuleType("exa_py")
    module.Exa = Refuses
    monkeypatch.setitem(sys.modules, "exa_py", module)
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "exa-bad")

    assert setup_key.exa_main(env_file, show=False) == 1
    assert not env_file.exists(), "a rejected key reached the file"


def test_a_missing_package_is_reported_rather_than_crashing(monkeypatch):
    import asyncio

    monkeypatch.setitem(sys.modules, "exa_py", None)
    ok, detail = asyncio.run(setup_key.exa_works("exa-any"))
    assert ok is False
    assert "exa_py is not installed" in detail
    assert "requirements-exa.txt" in detail, "it must name the way out"


def test_an_accepted_key_is_stored(env_file, exa_ok, monkeypatch):
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "exa-good")
    assert setup_key.exa_main(env_file, show=False) == 0
    assert "EXA_API_KEY=exa-good" in env_file.read_text()


def test_nothing_entered_changes_nothing(env_file, exa_ok, monkeypatch):
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "   ")
    assert setup_key.exa_main(env_file, show=False) == 1
    assert not env_file.exists()


# --------------------------------------------------------------------------
# the key never reaches a screen or a log
# --------------------------------------------------------------------------
def test_the_key_is_never_printed_in_full(env_file, exa_ok, monkeypatch, capsys):
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "exa-abcdefghijklmnop")
    setup_key.exa_main(env_file, show=False)
    out = capsys.readouterr().out
    assert "exa-abcdefghijklmnop" not in out, "the key was echoed to the terminal"


def test_the_fingerprint_identifies_without_revealing():
    assert setup_key.describe_exa_key("") == "not set"
    described = setup_key.describe_exa_key("exa-abcdefghijklmnop")
    assert "mnop" in described and "abcdefgh" not in described


def test_the_prompt_hides_what_is_typed():
    """getpass, not input - so the key is not in a screenshot or in scrollback
    that gets shared later."""
    import inspect

    source = inspect.getsource(setup_key.exa_main)
    assert "getpass.getpass" in source
    assert "input(" not in source
