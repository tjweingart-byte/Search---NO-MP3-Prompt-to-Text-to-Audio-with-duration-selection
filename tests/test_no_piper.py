"""A fresh install with no Piper anywhere is a supported state, not a gap.

Piper was FAM's interim voice. It is gone - engine class, configuration,
package and dependency - and this file is what stops it coming back by
accident.

Why removed rather than switched off: a second engine that can speak is a
second engine that can be *selected*. Piper reached listeners three ways that
had nothing to do with anyone choosing it - `build_engine` fell through to it,
`engine_for_voice` fell back to it, and `list_voices` offered it when the
production slot was empty - and an app that quietly sounds worse than intended
is the failure this project has lost the most time to (PROBLEMS.md). A knob left
behind is an invitation to turn it back on, and this one would have turned
itself.

What replaces it is not a worse voice but *no* voice: a placeholder tone that
nobody can mistake for FAM, with `/api/health` saying `interim: true` and
`build_engine` logging why Chatterbox was unavailable. Two honest states, and
the failure is audible as a failure.

Nothing in this file needs a GPU, a model, a credential or an optional package.
"""
from __future__ import annotations

import importlib
import os
import pathlib
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tts  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Everything the server imports on the request path. If any of these needed
#: `piper`, a fresh install could not serve a request.
RUNTIME_MODULES = (
    "app", "pipeline", "tts", "config", "script_generator", "cache",
    "speech_assembly", "script_buffer", "episode_marks", "voice_store",
    "audio_utils", "attachments", "topics", "mixes", "social", "demo_script",
    "anthropic_client",
)


# --------------------------------------------------------------------------
# the dependency surface
# --------------------------------------------------------------------------
def test_piper_is_not_a_production_dependency():
    """A fresh `pip install -r requirements.txt` must not pull it in."""
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        requirement = line.split("#", 1)[0].strip()
        assert "piper" not in requirement.lower(), line


def test_no_runtime_module_imports_piper_at_any_depth():
    """Not at module scope, not inside a function, not lazily."""
    offenders = []
    for name in RUNTIME_MODULES:
        source = (ROOT / f"{name}.py").read_text()
        for number, line in enumerate(source.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("import piper", "from piper")):
                offenders.append(f"{name}.py:{number}: {stripped}")
    assert not offenders, "piper is imported by the runtime:\n" + "\n".join(offenders)


def test_the_whole_runtime_imports_with_piper_unimportable():
    """The real check, not a search: import every runtime module with `piper`
    poisoned in sys.modules, which is what a machine without it looks like."""
    saved = sys.modules.get("piper")
    sys.modules["piper"] = None  # makes `import piper` raise
    try:
        for name in RUNTIME_MODULES:
            importlib.import_module(name)
    finally:
        if saved is None:
            sys.modules.pop("piper", None)
        else:  # pragma: no cover - only when the package is installed
            sys.modules["piper"] = saved


def test_the_engine_class_is_gone_not_merely_unused():
    assert not hasattr(tts, "PiperEngine")
    assert not hasattr(tts, "default_piper_model")
    assert not hasattr(tts, "_prettify_piper_name")


def test_there_is_no_piper_configuration_left_to_set():
    """A setting that configures a removed engine is a promise it still exists."""
    import config

    fields = set(config.Settings.__dataclass_fields__)
    assert not [f for f in fields if "piper" in f.lower()], sorted(fields)
    assert "PIPER" not in (ROOT / ".env.example").read_text().upper().replace(
        "PIPER USED TO", "")


# --------------------------------------------------------------------------
# it cannot come back by accident
# --------------------------------------------------------------------------
def test_no_request_can_be_served_by_anything_but_chatterbox_or_the_tone():
    """The three routes a voice reaches a listener by, all of them."""
    assert [cls.name for cls in tts.PRODUCTION_ENGINES] == ["chatterbox"]
    assert tts.PLACEHOLDER_ENGINE is tts.DebugEngine
    assert tts.build_engine().name in {"chatterbox", "debug"}
    assert tts.engine_for_voice("piper:en_US-lessac-medium").name in {
        "chatterbox", "debug"}, "a stale piper voice id resurrected the engine"
    assert {v.engine for v in tts.list_voices()} <= {"chatterbox", "debug"}


def test_the_development_override_cannot_name_piper():
    """`TTS_ENGINE=piper` used to select it. It must now be refused by name,
    not silently ignored - an ignored setting reads as a working one."""
    with pytest.raises(tts.TTSUnavailable) as exc:
        tts.build_engine("piper")
    assert "not a known engine" in str(exc.value)


def test_the_development_engines_are_all_non_production():
    """espeak, `say` and the tone stay reachable for local work. None of them
    is in the production slot, and none can enter it through this door."""
    production = {cls.name for cls in tts.PRODUCTION_ENGINES}
    assert not (set(tts.DEV_ENGINES) & production)


def test_debug_survived_the_removal():
    """It is deterministic test infrastructure, not a Piper leftover. The whole
    suite runs on it, and it is what a machine with no GPU serves."""
    assert tts.DebugEngine.available() is True
    assert "debug" in tts.DEV_ENGINES
    assert tts.DebugEngine.voices(), "the picker must never be empty"


# --------------------------------------------------------------------------
# what a machine with no voice tells its listener
# --------------------------------------------------------------------------
def test_a_machine_with_no_voice_says_so_rather_than_serving_quietly():
    report = tts.engine_report()
    if report["selected"] == "chatterbox":  # pragma: no cover - not on CI
        pytest.skip("this machine can run the production engine")
    assert report["interim"] is True, "a tone was reported as the product"
    assert report["selected"] == "debug"
    assert report["production_engines"] == ["chatterbox"]


def test_the_fallback_names_the_reason_out_loud(caplog):
    """`interim: true` in a JSON response is not enough on its own - the reason
    Chatterbox could not run has to reach the log too, every time an engine is
    built, or the operator is left guessing."""
    import logging

    with caplog.at_level(logging.WARNING, logger="tts"):
        engine = tts.build_engine()
    if engine.name == "chatterbox":  # pragma: no cover - not on CI
        pytest.skip("this machine can run the production engine")
    # getMessage() applies the args; record.message is the unformatted one.
    messages = [record.getMessage() for record in caplog.records]
    assert any("placeholder tone" in message for message in messages), messages
    assert any("chatterbox" in message for message in messages), (
        "the warning must name which engine was unavailable")


def test_the_voice_store_survived_because_chatterbox_uses_it():
    """`voice_store` is engine-agnostic and was not removed with the engine
    that happened to be using it: the reference recording lives there."""
    import voice_store

    assert tts.ChatterboxEngine.reference_path().parent == voice_store.voices_dir()


# --------------------------------------------------------------------------
# the interface, deliberately untouched
# --------------------------------------------------------------------------
def test_the_voice_picker_still_names_piper_and_that_is_inert():
    """`static/index.html` was not edited by the removal, on purpose.

    The picker groups voices by engine through a label map and an ordering
    array that both still contain "piper". Both are keyed off what
    `/api/voices` actually returns and skip any group with no members
    (`if(!group.length) return;`), so with no piper voice ever served the entry
    is dead data rather than a code path.

    This is asserted rather than tidied because the interface is out of scope
    for this change and "the frontend is unchanged" is a claim worth being able
    to make exactly. One cosmetic consequence is recorded here so it is not
    discovered as a surprise: Chatterbox is not in the ordering array either,
    so its voice appears under the picker's "Other" heading until the interface
    is next touched.
    """
    page = (ROOT / "static" / "index.html").read_text()
    assert 'var order = ["piper", "say", "espeak", "debug"];' in page
    assert "if(!group.length) return;" in page, (
        "the empty-group skip is what makes the stale entry inert")
    assert "chatterbox" not in page, (
        "if the interface has learned about chatterbox, update this test and "
        "the ordering array together")
    # The thing that actually keeps it inert.
    assert not [v for v in tts.list_voices() if v.engine == "piper"]
