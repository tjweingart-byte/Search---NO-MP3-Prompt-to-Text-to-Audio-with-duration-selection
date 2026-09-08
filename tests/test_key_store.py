"""The key is set once per machine, verified before it is stored, never in source.

The pattern this closes: every new copy of the app was a fresh folder with no
`.env`, so the key had to be pasted again - into a terminal, a chat window,
whatever was to hand. `~/.fam/env` is the same answer `~/.fam/voices` already
gives for voice models: one file per machine, outside any project folder.
"""
from __future__ import annotations

import importlib
import os
import pathlib
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as config_mod  # noqa: E402
import setup_key  # noqa: E402


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / "fam" / "env"
    monkeypatch.setenv("FAM_ENV_FILE", str(path))
    monkeypatch.delenv("FAM_IGNORE_DOTENV", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return path


def test_the_key_lives_outside_the_project(store):
    """Inside the project it is lost on every new copy, which is the whole
    reason it kept being pasted somewhere it should not be."""
    assert config_mod.shared_env_path() == store
    project = pathlib.Path(config_mod.__file__).resolve().parent
    assert project not in store.parents, "the store is inside the repository"


def test_writing_then_reading_it_back(store):
    setup_key.write_key("sk-ant-example-key-0001")
    config_mod._load_dotenv()
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-example-key-0001"
    assert str(store) in config_mod.key_source()


def test_a_second_key_replaces_the_first(store):
    """Two key lines in one file means which one is sent depends on who reads
    it - already a day lost to that once."""
    setup_key.write_key("sk-ant-old-0000")
    setup_key.write_key("sk-ant-new-1111")
    assert store.read_text().count("ANTHROPIC_API_KEY=") == 1
    config_mod._load_dotenv()
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-new-1111"


def test_the_file_is_not_readable_by_anyone_else(store):
    setup_key.write_key("sk-ant-example-key-0002")
    assert oct(store.stat().st_mode & 0o777) == "0o600"


def test_other_settings_in_the_file_survive_a_key_change(store):
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text("MODEL=claude-haiku-4-5\nANTHROPIC_API_KEY=sk-ant-old\n")
    setup_key.write_key("sk-ant-new")
    assert "MODEL=claude-haiku-4-5" in store.read_text()


def test_the_project_env_still_wins(store, tmp_path, monkeypatch):
    """Machine-wide is the default, not an override: a project that pins its
    own key or model must keep doing so."""
    setup_key.write_key("sk-ant-machine-wide")
    project_env = tmp_path / "config.py"
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-this-project\n")
    monkeypatch.setattr(config_mod.pathlib.Path, "resolve",
                        lambda self: project_env, raising=False)
    config_mod._load_dotenv()
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-this-project"


def test_a_rejected_key_is_never_written(store, monkeypatch):
    """Storing a key that does not work is worse than storing none: the app
    starts, says it is live, and fails on the first episode."""
    async def reject(key):
        return False, "AuthenticationError: invalid x-api-key"

    monkeypatch.setattr(setup_key, "works", reject)
    monkeypatch.setattr(setup_key.getpass, "getpass", lambda prompt="": "sk-ant-bad-key")
    monkeypatch.setattr(sys, "argv", ["setup_key.py"])
    assert setup_key.main() == 1
    assert not store.exists(), "a rejected key was stored anyway"


def test_a_good_key_is_stored_and_confirmed(store, monkeypatch, capsys):
    async def accept(key):
        return True, "claude-sonnet-5 is reachable with this key."

    monkeypatch.setattr(setup_key, "works", accept)
    monkeypatch.setattr(setup_key.getpass, "getpass", lambda prompt="": "sk-ant-good-key-9999")
    monkeypatch.setattr(sys, "argv", ["setup_key.py"])
    assert setup_key.main() == 0
    assert "ANTHROPIC_API_KEY=sk-ant-good-key-9999" in store.read_text()
    assert "Accepted" in capsys.readouterr().out


def test_removing_it_leaves_nothing_behind(store, monkeypatch):
    setup_key.write_key("sk-ant-example-key-0003")
    monkeypatch.setattr(sys, "argv", ["setup_key.py", "--remove"])
    assert setup_key.main() == 0
    assert "ANTHROPIC_API_KEY" not in store.read_text()


def test_the_key_is_never_written_into_source():
    """A key in a commit stays in the history after the line is deleted. The
    only writer is setup_key.write_key, and it writes to the shared store."""
    root = pathlib.Path(config_mod.__file__).resolve().parent
    for path in root.glob("*.py"):
        text = path.read_text()
        assert "sk-ant-" not in text or path.name in {"setup_key.py", "config.py"}, (
            f"{path.name} contains a key-shaped literal"
        )
