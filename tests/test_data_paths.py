"""Every database is found from the project root, never the working directory.

The bug this pins was silent and pointed the wrong way: a bare filename follows
the cwd, so starting the server from somewhere else created a second, empty set
of files without raising anything. The app came up with a cold cache, an empty
feed and no echoes, and looked like a broken feature rather than a wrong path.

The first test is the whole point - it changes directory and asserts the stores
do not follow.
"""
from __future__ import annotations

import importlib
import os
import pathlib
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import attachments as A  # noqa: E402
import mixes as M  # noqa: E402
import paths  # noqa: E402
import social as S  # noqa: E402
import topics as T  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Every store, as (env var, filename, constructor). One list so a store added
#: later has to be added here, and every check below covers it automatically.
STORES = [
    ("MYFAM_DB", "myfam.db", T.EventStore),
    ("SOCIAL_DB", "social.db", S.SocialStore),
    ("MIXES_DB", "mixes.db", M.MixStore),
    ("ATTACHMENTS_PATH", "attachments.db", A.AttachmentStore),
]

ALL_VARS = [v for v, _f, _c in STORES] + ["CACHE_PATH"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ALL_VARS:
        monkeypatch.delenv(var, raising=False)


# --- the bug itself -------------------------------------------------------


def test_the_working_directory_does_not_move_the_databases(monkeypatch, tmp_path):
    """Start the app from anywhere; it opens the same files."""
    here = {var: paths.data_path(var, name) for var, name, _c in STORES}
    monkeypatch.chdir(tmp_path)
    assert {var: paths.data_path(var, name) for var, name, _c in STORES} == here


def test_a_store_built_from_elsewhere_writes_to_the_project_root(monkeypatch, tmp_path):
    """The seed_demo case: seed from one directory, serve from another. It
    wrote to the cwd, so Explore stayed empty however much you tapped it."""
    monkeypatch.chdir(tmp_path)
    store = T.EventStore()
    assert pathlib.Path(store.path).parent == ROOT
    assert not list(tmp_path.glob("*.db")), "a database was created in the cwd"


# --- resolution rules -----------------------------------------------------


@pytest.mark.parametrize("var, filename, ctor", STORES)
def test_every_store_defaults_to_an_absolute_path(var, filename, ctor, tmp_path):
    resolved = pathlib.Path(paths.data_path(var, filename))
    assert resolved.is_absolute()
    assert resolved == ROOT / filename


@pytest.mark.parametrize("var, filename, ctor", STORES)
def test_an_absolute_env_var_is_used_exactly(var, filename, ctor, monkeypatch, tmp_path):
    wanted = tmp_path / "elsewhere" / filename
    wanted.parent.mkdir()
    monkeypatch.setenv(var, str(wanted))
    assert paths.data_path(var, filename) == str(wanted)
    assert ctor().path == str(wanted)


@pytest.mark.parametrize("var, filename, ctor", STORES)
def test_an_explicit_path_still_wins(var, filename, ctor, tmp_path):
    """Tests pass paths directly; that must keep working."""
    given = str(tmp_path / filename)
    assert ctor(given).path == given


def test_a_relative_env_var_resolves_to_the_project_root_and_says_so(
    monkeypatch, tmp_path, caplog
):
    """Never the cwd, and never silently - the ambiguity is the bug."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MYFAM_DB", "somewhere.db")
    with caplog.at_level("WARNING"):
        resolved = paths.data_path("MYFAM_DB", "myfam.db")
    assert resolved == str(ROOT / "somewhere.db")
    assert "relative" in caplog.text and "MYFAM_DB" in caplog.text


def test_a_tilde_path_is_expanded(monkeypatch):
    monkeypatch.setenv("SOCIAL_DB", "~/fam-social.db")
    resolved = pathlib.Path(paths.data_path("SOCIAL_DB", "social.db"))
    assert resolved.is_absolute() and "~" not in str(resolved)
    assert resolved == pathlib.Path.home() / "fam-social.db"


def test_an_empty_env_var_is_treated_as_unset(monkeypatch):
    """An exported-but-blank variable is a deployment slip, not a request to
    open a file called ''."""
    monkeypatch.setenv("MIXES_DB", "   ")
    assert paths.data_path("MIXES_DB", "mixes.db") == str(ROOT / "mixes.db")


def test_the_cache_path_follows_the_same_rule(monkeypatch, tmp_path):
    import config

    monkeypatch.setenv("FAM_IGNORE_DOTENV", "1")
    monkeypatch.setenv("CACHE_PATH", str(tmp_path / "c.db"))
    importlib.reload(config)
    try:
        assert config.settings.cache_path == str(tmp_path / "c.db")
        monkeypatch.delenv("CACHE_PATH")
        importlib.reload(config)
        assert config.settings.cache_path == str(ROOT / "scripts.db")
    finally:
        importlib.reload(config)


# --- the deployment has to name all of them -------------------------------


def test_the_dockerfile_puts_every_database_on_the_mounted_disk():
    """Three of the five were named; social and attachments were not, so every
    redeploy discarded every listener's name, handle and echo. A store added
    later must be added to the Dockerfile too."""
    dockerfile = (ROOT / "Dockerfile").read_text()
    missing = [v for v in ALL_VARS if f"{v}=/data/" not in dockerfile]
    assert not missing, f"not pinned to the mounted disk in the Dockerfile: {missing}"


def test_the_example_ships_no_literal_database_path():
    """A default that named a directory would be wrong on every machine but
    the one it was written on, and test_env_example compares live values."""
    active = [
        line.split("=")[0].strip()
        for line in (ROOT / ".env.example").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#") and "=" in line
    ]
    assert not [v for v in ALL_VARS if v in active], (
        "a database path is set in .env.example; it should be commented out, "
        "because the working default is derived rather than written down"
    )


def test_the_example_documents_every_one_of_them():
    example = (ROOT / ".env.example").read_text()
    assert not [v for v in ALL_VARS if v not in example], "an undocumented path"


# --- the static files had the same bug, and hid it -------------------------


def test_the_interface_is_served_from_the_project_root_not_the_cwd(monkeypatch, tmp_path):
    """`StaticFiles(directory="static")` was cwd-relative too. It failed loudly
    - the app would not start from another directory at all - which is exactly
    why nobody reached the quiet database version of the same bug behind it."""
    import app as appmod

    monkeypatch.chdir(tmp_path)
    mount = next(r for r in appmod.app.routes if getattr(r, "name", "") == "static")
    served = pathlib.Path(mount.app.directory)
    assert served.is_absolute()
    assert served == ROOT / "static"


def test_the_app_module_names_no_bare_relative_directory():
    """The guard against the next one. Any `directory="..."` in app.py that is
    not built from PROJECT_ROOT will follow whoever started the process."""
    source = (ROOT / "app.py").read_text()
    bare = [
        line.strip()
        for line in source.splitlines()
        if 'directory="' in line and "PROJECT_ROOT" not in line
    ]
    assert not bare, f"cwd-relative directory in app.py: {bare}"
