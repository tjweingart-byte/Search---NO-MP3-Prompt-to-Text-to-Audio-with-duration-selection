#!/usr/bin/env python3
"""Why Chatterbox will not load here, said precisely rather than guessed.

Two packages on this dependency chain hide their real import errors behind a
try/except and then fail much later with a message naming neither the module
nor the version that broke. Both cost a session on a billing GPU. This tool
re-runs those imports without the mask so the original error surfaces.

    perth/__init__.py

        try:
            from .perth_net.perth_net_implicit.perth_watermarker import \\
                PerthImplicitWatermarker
        except ImportError:
            PerthImplicitWatermarker = None

    ...so a real ImportError becomes None, and what you finally see is
    `TypeError: 'NoneType' object is not callable` from from_pretrained,
    thousands of lines away and after a 4 GB download.

    transformers/utils/import_utils.py

        raise ModuleNotFoundError(f"Could not import module '{name}'. Are
        this object's requirements defined correctly?") from e

    ...and the `from e` - the real error - is never printed.

Nothing here downloads, installs, deletes or moves anything. It reports the
weight cache location and size so you can confirm a download survived.

    python tools/diagnose_chatterbox.py

This is the tool named by every pin in `requirements-chatterbox.txt`. Run it
before changing one.
"""
from __future__ import annotations

import importlib
import os
import pathlib
import sys

#: The module perth imports behind its try/except.
WATERMARKER_MODULE = "perth.perth_net.perth_net_implicit.perth_watermarker"

#: What production imports: `from chatterbox.tts import ChatterboxTTS`, the
#: base model. Turbo is a different module and class and is not what FAM's
#: voice was chosen on - see tts.py.
PRODUCTION_MODULE = "chatterbox.tts"

#: transformers' lazy module names the symbol and discards the cause. The
#: symbol tells us which module implements it; importing that directly raises
#: the truth.
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
#: 82.0.0. `perth` still imports it, so a current setuptools breaks perth
#: silently.
LAST_SETUPTOOLS_WITH_PKG_RESOURCES = "81.0.0"

#: torchvision pins an exact torch, and transformers imports torchvision on the
#: way to LlamaModel. A mismatched pair fails at
#: `@torch.library.register_fake("torchvision::nms")` with "operator
#: torchvision::nms does not exist" - which names neither package's version.
#: Verified from the wheels' own Requires-Dist, not from memory.
TORCHVISION_FOR_TORCH = {
    "2.6.0": "0.21.0",
    "2.7.0": "0.22.0",
    "2.8.0": "0.23.0",
}


def _version(name: str) -> str:
    try:
        import importlib.metadata as md

        return md.version(name)
    except Exception:
        return "(not installed)"


def torchvision_pairing() -> dict:
    """What torch and torchvision are installed, and whether they match.

    Returns `expected: None` for a torch this table does not cover, rather than
    extrapolating - a wrong "expected" would send someone to reinstall a
    working package.
    """
    out = {"torch": None, "torchvision": None, "expected": None, "ok": None}
    try:
        import torch
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    # "2.6.0+cu124" -> "2.6.0"; the CUDA suffix is not part of the pairing.
    out["torch"] = str(torch.__version__)
    out["expected"] = TORCHVISION_FOR_TORCH.get(out["torch"].split("+")[0])
    try:
        import torchvision

        out["torchvision"] = str(torchvision.__version__)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    if out["expected"]:
        out["ok"] = out["torchvision"].split("+")[0] == out["expected"]
    return out


def hf_cache() -> pathlib.Path:
    """Where the weights were downloaded to."""
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
                    "torchaudio", "torchvision", "librosa", "numpy", "scipy",
                    "transformers", "huggingface-hub", "tokenizers"):
        print(f"  {package:<16}{_version(package)}")

    print("\ndevice")
    try:
        import torch

        print(f"  cuda available    {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  card              {torch.cuda.get_device_name(0)}")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None:
            print(f"  mps built/avail   {mps.is_built()} / {mps.is_available()}")
    except ImportError:
        print("  torch is not installed, so the device cannot be checked")
    try:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
        from tts import ChatterboxEngine

        ok, detail = ChatterboxEngine.diagnose()
        print(f"  FAM would use     {ChatterboxEngine.device()}")
        print(f"  FAM verdict       {'available' if ok else 'UNAVAILABLE'}: {detail}")
    except Exception as exc:
        print(f"  could not ask FAM: {type(exc).__name__}: {exc}")

    print("\ntorch / torchvision pairing")
    pair = torchvision_pairing()
    print(f"  torch        {pair['torch'] or pair.get('error')}")
    print(f"  torchvision  {pair['torchvision'] or pair.get('error')}")
    if pair["expected"] and pair["ok"] is False:
        print(f"  MISMATCH     torch {pair['torch']} needs torchvision "
              f"{pair['expected']}")
        print("               transformers imports torchvision on the way to "
              "LlamaModel, and a mismatched")
        print('               pair dies at register_fake("torchvision::nms") '
              "naming neither version.")
    elif pair["ok"]:
        print("  matched")

    print("\nweights")
    cache = hf_cache()
    print(f"  cache      {cache}")
    found = sorted(cache.glob("models--*hatterbox*")) if cache.exists() else []
    if found:
        for model in found:
            print(f"  {model.name}  {_size_gb(model):.2f} GB - nothing here "
                  "re-downloads it")
    else:
        print("  no chatterbox weights at that path; a run would download ~4 GB")

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
        importlib.import_module(PRODUCTION_MODULE)
        print(f"  {PRODUCTION_MODULE} imports cleanly")
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
    """Re-raise what transformers' lazy module swallowed, if that is the fault."""
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
                 "torchvision", "numpy", "safetensors")}
    print("  installed")
    for name, value in versions.items():
        print(f"    {name:<18}{value}")

    print("\n  smallest safe fix")
    lowered = text.lower()
    if "torchvision" in lowered or "nms" in lowered:
        print("    torchvision is built for a different torch. transformers "
              "imports it on the way to LlamaModel.\n"
              "    Repair torchvision alone, with --no-deps so pip cannot "
              "touch torch:\n"
              "      pip install --no-deps torchvision==0.21.0 \\\n"
              "        --index-url https://download.pytorch.org/whl/cu124\n"
              "    (0.21.0 is the version whose metadata requires torch==2.6.0.)")
    elif "huggingface_hub" in lowered or "huggingface-hub" in lowered:
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
        print('    pip install "setuptools<82"')
        print("    perth imports pkg_resources, which setuptools removed in "
              "82.0.0. Reinstalling a setuptools that still ships it restores "
              "the watermarker with nothing disabled and nothing re-downloaded.")
    elif "torchvision::nms" in text or "torchvision" in text:
        print("    torchvision does not match torch. Repair torchvision alone, "
              "never torch - the CUDA build is what works:\n"
              "      pip install --no-deps torchvision==<paired version> \\\n"
              "        --index-url https://download.pytorch.org/whl/cu124")
    elif "torchaudio" in text or "torch" in text:
        print("    A torch/torchaudio mismatch. chatterbox-tts pins "
              "torch==2.6.0 and torchaudio==2.6.0; install both at those "
              "versions together, not one at a time.")
    elif "librosa" in text or "numba" in text or "numpy" in text:
        print("    A numpy/librosa mismatch. chatterbox-tts pins "
              "librosa==0.11.0 and numpy<2 below Python 3.13.")
    else:
        print("    Not a failure this tool has a prescription for - the error "
              "above is the real one; fix that dependency and rerun.")
    print("\n    Do not stub the watermarker out. It runs inside every "
          "generate() call, so removing it would change the audio, the "
          "latency, and whether FAM's speech is watermarked at all.\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
