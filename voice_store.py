"""Where voice models live: one shared folder, not one per project copy.

Voice models are tens of megabytes and change far less often than the code. If
they live inside the project folder, every new version of the app looks like a
fresh install with no voices, and the same files get downloaded again.

They are therefore kept in a stable per-user location outside any project
folder, so unzipping a new version and running it just works.

Resolution order:

1. ``FAM_VOICES_DIR`` - explicit override, for deployments and tests.
2. ``VOICES_DIR`` - the older name, still honoured.
3. ``~/.fam/voices`` - the default, shared by every version of the app.

This module deliberately does not import `config`, so `config` can use it to
compute its default without a circular import.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

#: Per-user default. Stable across app versions, and easy to find or delete.
DEFAULT_DIR = Path.home() / ".fam" / "voices" if Path.home() else Path("voices")

#: Project-local folders older versions used. Checked once, to adopt rather
#: than re-download what has already been fetched.
LEGACY_DIR_NAMES = ("voices",)


def voices_dir(create: bool = False) -> Path:
    """The directory voice models are read from and written to."""
    override = os.environ.get("FAM_VOICES_DIR") or os.environ.get("VOICES_DIR")
    directory = Path(override).expanduser() if override else DEFAULT_DIR
    if create:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # read-only home, unusual container, etc.
            log.warning("could not create %s: %s", directory, exc)
    return directory


def installed(directory: Path | None = None) -> list[Path]:
    """Voice models present, as .onnx paths."""
    directory = directory or voices_dir()
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.glob("*.onnx") if p.is_file())


def sidecar_for(model: Path) -> Path | None:
    """A model's config JSON, which Piper needs alongside the .onnx."""
    for candidate in (Path(str(model) + ".json"), model.with_suffix(".json")):
        if candidate.exists():
            return candidate
    return None


def legacy_dirs(start: Path | None = None) -> list[Path]:
    """Project-local `voices/` folders from older versions of the app."""
    root = start or Path(__file__).resolve().parent
    found = []
    for name in LEGACY_DIR_NAMES:
        candidate = root / name
        if candidate.is_dir() and candidate.resolve() != voices_dir().resolve():
            if installed(candidate):
                found.append(candidate)
    return found


def adopt_legacy(target: Path | None = None, move: bool = False) -> list[str]:
    """Copy models from an older project-local folder into the shared one.

    Copying rather than moving by default: the old project folder keeps working
    if anything goes wrong, and the user can delete it when they are ready. A
    model already present in the shared folder is never overwritten.

    Returns the names adopted.
    """
    target = target or voices_dir(create=True)
    adopted: list[str] = []
    have = {p.stem for p in installed(target)}

    for source_dir in legacy_dirs():
        for model in installed(source_dir):
            if model.stem in have:
                continue
            sidecar = sidecar_for(model)
            if sidecar is None:
                log.warning("skipping %s: its .onnx.json is missing", model.name)
                continue
            try:
                # Copy to a temporary name first, so an interrupted copy never
                # leaves a half-written model that looks installed.
                for src in (model, sidecar):
                    partial = target / (src.name + ".partial")
                    (shutil.move if move else shutil.copy2)(str(src), partial)
                    partial.replace(target / src.name)
                adopted.append(model.stem)
                have.add(model.stem)
                log.info("adopted voice %s from %s", model.stem, source_dir)
            except OSError as exc:
                log.warning("could not adopt %s: %s", model.name, exc)
                for leftover in target.glob(f"{model.stem}*.partial"):
                    leftover.unlink(missing_ok=True)
    return adopted


def ensure_ready() -> dict:
    """Prepare the shared store. Safe to call on every startup.

    Adopts voices from an older project folder only when the shared store has
    none, so this is a one-time cost on the first run of a new version and a
    no-op every time after.
    """
    directory = voices_dir(create=True)
    present = installed(directory)
    adopted: list[str] = []

    if not present:
        adopted = adopt_legacy(directory)
        present = installed(directory)
        if adopted:
            log.info(
                "reused %d voice(s) from a previous version: %s",
                len(adopted), ", ".join(adopted),
            )

    return {
        "dir": str(directory),
        "voices": [p.stem for p in present],
        "adopted": adopted,
    }


def describe() -> str:
    directory = voices_dir()
    names = [p.stem for p in installed(directory)]
    if not names:
        return f"no voices in {directory}"
    return f"{len(names)} voice(s) in {directory}: " + ", ".join(names)
