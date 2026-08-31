"""Voices live once per user, not once per copy of the project.

The problem this solves: voice models are tens of megabytes and change far less
often than the code, so keeping them inside the project folder meant every new
version looked like a fresh install and re-downloaded the same files.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import voice_store  # noqa: E402


def make_voice(directory, name, sidecar=True):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.onnx").write_bytes(b"pretend-model")
    if sidecar:
        (directory / f"{name}.onnx.json").write_text('{"audio": {"sample_rate": 22050}}')
    return directory / f"{name}.onnx"


@pytest.fixture
def shared(tmp_path, monkeypatch):
    directory = tmp_path / "shared"
    monkeypatch.setenv("FAM_VOICES_DIR", str(directory))
    monkeypatch.delenv("VOICES_DIR", raising=False)
    return directory


def test_the_default_location_is_outside_any_project(monkeypatch):
    """It must not be relative, or it would follow the project folder around."""
    monkeypatch.delenv("FAM_VOICES_DIR", raising=False)
    monkeypatch.delenv("VOICES_DIR", raising=False)
    directory = voice_store.voices_dir()
    assert directory.is_absolute(), f"{directory} must be an absolute, per-user path"
    assert directory.name == "voices" and directory.parent.name == ".fam"


def test_the_location_can_be_overridden(tmp_path, monkeypatch):
    monkeypatch.setenv("FAM_VOICES_DIR", str(tmp_path / "elsewhere"))
    assert voice_store.voices_dir() == tmp_path / "elsewhere"


def test_the_older_env_var_still_works(tmp_path, monkeypatch):
    monkeypatch.delenv("FAM_VOICES_DIR", raising=False)
    monkeypatch.setenv("VOICES_DIR", str(tmp_path / "legacy-var"))
    assert voice_store.voices_dir() == tmp_path / "legacy-var"


def test_a_tilde_path_is_expanded(monkeypatch):
    monkeypatch.setenv("FAM_VOICES_DIR", "~/somewhere/voices")
    assert "~" not in str(voice_store.voices_dir())


def test_installed_lists_models(shared):
    make_voice(shared, "en_US-lessac-medium")
    make_voice(shared, "en_GB-alba-medium")
    assert {p.stem for p in voice_store.installed()} == {
        "en_US-lessac-medium",
        "en_GB-alba-medium",
    }


def test_voices_from_an_older_project_are_reused_not_redownloaded(
    shared, tmp_path, monkeypatch
):
    """The point of the whole change."""
    old_project = tmp_path / "fam-podcast 6"
    make_voice(old_project / "voices", "en_US-lessac-medium")
    monkeypatch.setattr(voice_store, "__file__", str(old_project / "voice_store.py"))

    result = voice_store.ensure_ready()
    assert result["adopted"] == ["en_US-lessac-medium"]
    assert [p.stem for p in voice_store.installed()] == ["en_US-lessac-medium"]


def test_adopting_leaves_the_old_folder_intact(shared, tmp_path, monkeypatch):
    """Copy, not move: the old version keeps working if anything goes wrong."""
    old_project = tmp_path / "fam-podcast 6"
    original = make_voice(old_project / "voices", "en_US-lessac-medium")
    monkeypatch.setattr(voice_store, "__file__", str(old_project / "voice_store.py"))

    voice_store.ensure_ready()
    assert original.exists(), "the original must not be deleted"


def test_adopting_is_idempotent(shared, tmp_path, monkeypatch):
    old_project = tmp_path / "fam-podcast 6"
    make_voice(old_project / "voices", "en_US-lessac-medium")
    monkeypatch.setattr(voice_store, "__file__", str(old_project / "voice_store.py"))

    assert voice_store.ensure_ready()["adopted"] == ["en_US-lessac-medium"]
    assert voice_store.ensure_ready()["adopted"] == [], "must not re-adopt on every start"


def test_an_already_present_voice_is_not_overwritten(shared, tmp_path, monkeypatch):
    make_voice(shared, "en_US-lessac-medium")
    (shared / "en_US-lessac-medium.onnx").write_bytes(b"the-good-one")
    old_project = tmp_path / "fam-podcast 6"
    make_voice(old_project / "voices", "en_US-lessac-medium")
    monkeypatch.setattr(voice_store, "__file__", str(old_project / "voice_store.py"))

    voice_store.ensure_ready()
    assert (shared / "en_US-lessac-medium.onnx").read_bytes() == b"the-good-one"


def test_a_model_without_its_sidecar_is_skipped(shared, tmp_path, monkeypatch):
    """Piper needs the .onnx.json; half a voice is worse than none."""
    old_project = tmp_path / "fam-podcast 6"
    make_voice(old_project / "voices", "broken-voice", sidecar=False)
    monkeypatch.setattr(voice_store, "__file__", str(old_project / "voice_store.py"))

    assert voice_store.ensure_ready()["adopted"] == []


def test_ensure_ready_creates_the_store_and_survives_being_empty(shared):
    result = voice_store.ensure_ready()
    assert result["voices"] == []
    assert result["adopted"] == []
    assert shared.is_dir(), "the store directory should be created"


def test_the_engine_reads_from_the_shared_store(shared, monkeypatch):
    """The production path must resolve to the same place as setup does."""
    make_voice(shared, "en_US-lessac-medium")
    import dataclasses

    import tts

    monkeypatch.setattr(
        tts, "settings", dataclasses.replace(tts.settings, voices_dir=str(shared))
    )
    assert [p.stem for p in tts.PiperEngine.installed_models()] == ["en_US-lessac-medium"]
    assert [v.id for v in tts.PiperEngine.voices()] == ["piper:en_US-lessac-medium"]


def test_describe_is_useful_when_empty_and_when_not(shared):
    assert "no voices" in voice_store.describe()
    make_voice(shared, "en_US-lessac-medium")
    assert "en_US-lessac-medium" in voice_store.describe()
