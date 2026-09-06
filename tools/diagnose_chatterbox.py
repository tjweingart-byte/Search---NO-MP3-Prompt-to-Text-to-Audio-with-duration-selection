#!/usr/bin/env python3
"""Why Chatterbox Turbo failed to load, said precisely rather than guessed.

The failure that prompted this:

    self.watermarker = perth.PerthImplicitWatermarker()
    TypeError: 'NoneType' object is not callable

`perth/__init__.py` does this:

    try:
        from .perth_net.perth_net_implicit.perth_watermarker import \\
            PerthImplicitWatermarker
    except ImportError:
        PerthImplicitWatermarker = None

So a *real* ImportError inside that module is swallowed and replaced with
None, and the error you finally see is a TypeError thousands of lines later
with no mention of what actually failed. This tool re-runs that import without
the try/except, so the original error surfaces with its own message.

Nothing here downloads, installs, deletes or moves anything. It reports the
weight cache location and size so you can confirm the download survived.

    python tools/diagnose_chatterbox.py
"""
from __future__ import annotations

import importlib
import os
import pathlib
import sys

#: The module perth imports behind its try/except.
WATERMARKER_MODULE = "perth.perth_net.perth_net_implicit.perth_watermarker"

#: `pkg_resources` shipped with setuptools through 81.0.0 and was removed in
#: 82.0.0. `perth` still imports it (`from pkg_resources import
#: resource_filename`), so a current setuptools breaks perth silently.
LAST_SETUPTOOLS_WITH_PKG_RESOURCES = "81.0.0"


def _version(name: str) -> str:
    try:
        import importlib.metadata as md

        return md.version(name)
    except Exception:
        return "(not installed)"


def hf_cache() -> pathlib.Path:
    """Where snapshot_download put the weights."""
    for env in ("HF_HUB_CACHE", "HF_HOME"):
        value = os.environ.get(env)
        if value:
            base = pathlib.Path(value)
            return base / "hub" if env == "HF_HOME" else base
    return pathlib.Path.home() / ".cache" / "huggingface" / "hub"


def _size_gb(path: pathlib.Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except OSError:
            continue
    return total / 1024 ** 3


def main() -> int:
    print(f"\npython      {sys.version.split()[0]}")
    for package in ("setuptools", "resemble-perth", "chatterbox-tts", "torch",
                    "torchaudio", "librosa", "numpy", "scipy", "transformers"):
        print(f"  {package:<16}{_version(package)}")

    print("\nmps")
    try:
        import torch
        built = torch.backends.mps.is_built()
        available = torch.backends.mps.is_available()
        print(f"  built into torch  {built}")
        print(f"  available here    {available}")
        if built and available:
            print("  Chatterbox will run on mps. Nothing in the fix below "
                  "changes the device.")
    except ImportError:
        print("  torch is not installed, so the device cannot be checked")

    print("\nweights")
    cache = hf_cache()
    turbo = cache / "models--ResembleAI--chatterbox-turbo"
    print(f"  cache      {cache}")
    if turbo.exists():
        print(f"  turbo      present, {_size_gb(turbo):.2f} GB - nothing here "
              "will re-download it")
    else:
        print("  turbo      NOT FOUND at that path; a run would download it")

    print("\npkg_resources")
    try:
        from pkg_resources import resource_filename          # noqa: F401
        print("  present")
    except Exception as exc:
        print(f"  MISSING - {type(exc).__name__}: {exc}")
        print(f"  setuptools {_version('setuptools')} does not ship it; it was "
              f"removed after {LAST_SETUPTOOLS_WITH_PKG_RESOURCES}.")

    print("\nperth")
    try:
        import perth
    except Exception as exc:
        print(f"  perth itself will not import: {type(exc).__name__}: {exc}")
        return 1

    watermarker = getattr(perth, "PerthImplicitWatermarker", None)
    if watermarker is not None:
        print("  PerthImplicitWatermarker is available - this is not the fault")
        return 0

    print("  PerthImplicitWatermarker is None. The real error, with perth's "
          "try/except taken off:\n")
    try:
        importlib.import_module(WATERMARKER_MODULE)
        print("    ...it imports cleanly now. The None was cached from an "
              "earlier import in this process, or the environment changed.")
        return 0
    except Exception as exc:
        print(f"    {type(exc).__name__}: {exc}\n")
        return _prescribe(exc)


def _prescribe(exc: Exception) -> int:
    """The smallest fix for what was actually found. No guessing past the data."""
    text = f"{type(exc).__name__}: {exc}"
    print("  smallest safe fix")
    if "pkg_resources" in text:
        print(f'    pip install "setuptools<82"')
        print("    perth imports pkg_resources, which setuptools removed in "
              "82.0.0. Reinstalling a setuptools that still ships it restores "
              "the watermarker with nothing disabled and nothing re-downloaded.")
    elif "torchaudio" in text or "torch" in text:
        print("    A torch/torchaudio mismatch. chatterbox-tts pins "
              "torch==2.6.0 and torchaudio==2.6.0; install both at those "
              "versions together, not one at a time.")
    elif "librosa" in text or "numba" in text or "numpy" in text:
        print("    A numpy/librosa mismatch. chatterbox-tts pins "
              "librosa==0.11.0 and numpy<2 below Python 3.13.")
    else:
        print("    Not a failure this tool does not have a prescription for "
              "- the error above is the real one; fix that dependency and "
              "rerun.")
    print("\n    Do not stub the watermarker out. It runs inside every "
          "generate() call, so removing it would change both the audio and "
          "the number this benchmark exists to measure.\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
