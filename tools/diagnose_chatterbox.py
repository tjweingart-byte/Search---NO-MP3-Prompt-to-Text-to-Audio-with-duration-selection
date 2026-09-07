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

#: transformers hides import failures the same way, behind a lazy module:
#: `raise ModuleNotFoundError(f"Could not import module '{name}'. Are this
#: object's requirements defined correctly?") from e` in
#: utils/import_utils.py. The `from e` is the real error, and it is not
#: printed. Importing the implementing module directly surfaces it.
LAZY_MODULES = {
    "LlamaModel": "transformers.models.llama.modeling_llama",
    "LlamaConfig": "transformers.models.llama.configuration_llama",
    "LlamaPreTrainedModel": "transformers.models.llama.modeling_llama",
    "GPT2Model": "transformers.models.gpt2.modeling_gpt2",
    "GPT2Config": "transformers.models.gpt2.configuration_gpt2",
    "GenerationMixin": "transformers.generation.utils",
    "AutoTokenizer": "transformers.models.auto.tokenization_auto",
}

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

    print("\nchatterbox")
    try:
        importlib.import_module("chatterbox.tts_turbo")
        print("  chatterbox.tts_turbo imports cleanly")
    except Exception as exc:
        print(f"  {type(exc).__name__}: {exc}")
        code = _unmask_transformers(exc)
        if code is not None:
            return code

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


def _unmask_transformers(exc: Exception) -> int | None:
    """Re-raise what transformers' lazy module swallowed, if that is the fault.

    Same shape as the perth failure: a real ImportError is caught and replaced
    with a message that names the symbol and not the cause. The symbol tells
    us which module to import directly, and that import raises the truth.
    """
    message = str(exc)
    if "Could not import module" not in message:
        return None

    symbol = message.split("'")[1] if "'" in message else ""
    target = LAZY_MODULES.get(symbol)
    print(f"\n  transformers hid the real error behind the symbol {symbol!r}.")
    if target is None:
        print("    This tool does not know which module implements it. Import "
              "it yourself to see the cause:\n"
              "      python -c \"import transformers; transformers." + symbol + "\"")
        return 1

    print(f"    Importing {target} directly:\n")
    try:
        importlib.import_module(target)
        print("    ...it imports cleanly on its own. The failure is in how "
              "chatterbox reaches it, not in the module.")
        return 1
    except Exception as real:
        import traceback

        # stdout, not stderr: this output gets piped and pasted, and a
        # traceback on stderr is the half that goes missing when it is.
        traceback.print_exc(file=sys.stdout)
        print()
        return _prescribe_transformers(real)


def _prescribe_transformers(exc: Exception) -> int:
    """Name the version conflict, from the error and the installed versions."""
    text = f"{type(exc).__name__}: {exc}"
    versions = {name: _version(name) for name in
                ("transformers", "tokenizers", "huggingface-hub", "torch",
                 "numpy", "diffusers", "safetensors")}
    print("  installed")
    for name, value in versions.items():
        print(f"    {name:<18}{value}")

    print("\n  smallest safe fix")
    lowered = text.lower()
    if "huggingface_hub" in lowered or "huggingface-hub" in lowered:
        print("    transformers 5.2.0 needs huggingface-hub>=1.3.0,<2.0. "
              "Something installed an older one.\n"
              '      pip install "huggingface-hub>=1.3,<2"')
    elif "tokenizers" in lowered:
        print("    transformers 5.2.0 needs tokenizers>=0.22.0,<=0.23.0.\n"
              '      pip install "tokenizers>=0.22,<=0.23"')
    elif "torch" in lowered:
        print("    A torch API transformers 5.2.0 expects is missing. It "
              "declares torch>=2.4 and chatterbox pins torch==2.6.0, so if "
              "this is the cause the two pins disagree in practice.\n"
              "    Report the traceback above before changing the torch "
              "version - it is what makes the CUDA build work.")
    else:
        print("    The traceback above is the real error. Send it before "
              "changing any pin; a speculative pin risks breaking the working "
              "CUDA environment.")
    print("\n    Do not downgrade torch to chase this without checking: "
          "torch 2.6.0+cu124 is what makes the 4090 work here.\n")
    return 1


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
