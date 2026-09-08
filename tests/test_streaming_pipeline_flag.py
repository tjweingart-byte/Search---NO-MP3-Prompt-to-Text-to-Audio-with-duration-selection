"""The switch between the shipped pipeline and the Phase 6 one.

The flag lands before the path it selects, so that when the Phase 6 path does
arrive, turning it off is one environment variable and a restart rather than a
revert. Until then `phase6` is a name that selects nothing, and these tests say
so out loud - a flag that silently did something would be worse than no flag.
"""
from __future__ import annotations

import importlib
import os
import pathlib
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

ROOT = pathlib.Path(__file__).resolve().parent.parent


def reloaded(monkeypatch, value=None):
    """config as it would be built in a fresh process with this environment."""
    monkeypatch.setenv("FAM_IGNORE_DOTENV", "1")
    if value is None:
        monkeypatch.delenv("STREAMING_PIPELINE", raising=False)
    else:
        monkeypatch.setenv("STREAMING_PIPELINE", value)
    return importlib.reload(config)


@pytest.fixture(autouse=True)
def _restore():
    yield
    for name in ("STREAMING_PIPELINE", "FAM_IGNORE_DOTENV"):
        os.environ.pop(name, None)
    importlib.reload(config)


# --------------------------------------------------------------------------
# the default
# --------------------------------------------------------------------------
def test_the_default_is_legacy(monkeypatch):
    """Rollback is trivial because the shipped path is what you get by
    default, on every machine that has never heard of this setting."""
    assert reloaded(monkeypatch).settings.streaming_pipeline == "legacy"


def test_an_empty_value_is_refused_rather_than_treated_as_unset(monkeypatch):
    """`STREAMING_PIPELINE=` in a .env is a mistake, not a request for the
    default - and guessing which would be the silent choice."""
    with pytest.raises(ValueError, match="not a pipeline"):
        reloaded(monkeypatch, "")


# --------------------------------------------------------------------------
# both valid values parse
# --------------------------------------------------------------------------
@pytest.mark.parametrize("value", ["legacy", "phase6"])
def test_both_valid_values_are_accepted(monkeypatch, value):
    assert reloaded(monkeypatch, value).settings.streaming_pipeline == value


@pytest.mark.parametrize("value, expected", [
    ("LEGACY", "legacy"), ("Phase6", "phase6"), ("  phase6  ", "phase6"),
])
def test_case_and_surrounding_space_do_not_change_the_meaning(monkeypatch, value,
                                                              expected):
    assert reloaded(monkeypatch, value).settings.streaming_pipeline == expected


def test_the_accepted_set_is_the_one_source_of_truth():
    assert config.STREAMING_PIPELINES == ("legacy", "phase6")
    assert config.settings.streaming_pipeline in config.STREAMING_PIPELINES


# --------------------------------------------------------------------------
# an unsupported value must fail loudly, not fall back
# --------------------------------------------------------------------------
@pytest.mark.parametrize("value", ["phase-6", "phase_6", "6", "new", "off", "1"])
def test_an_unsupported_value_refuses_to_start(monkeypatch, value):
    """Silently choosing a path on a typo is the failure mode this project has
    paid for repeatedly. Refusing to import is the correct outcome: a server
    that will not start is better than one serving the wrong pipeline."""
    with pytest.raises(ValueError, match="not a pipeline"):
        reloaded(monkeypatch, value)


def test_the_refusal_names_the_value_and_the_alternatives(monkeypatch):
    with pytest.raises(ValueError) as caught:
        reloaded(monkeypatch, "phase-6")
    message = str(caught.value)
    assert "STREAMING_PIPELINE" in message and "'phase-6'" in message
    assert "legacy" in message and "phase6" in message


def test_validation_also_catches_a_replaced_settings_object():
    """`pipeline._answer_first` builds its plans with dataclasses.replace, so
    validation that only ran on the environment would have a hole in it."""
    import dataclasses

    with pytest.raises(ValueError, match="not a pipeline"):
        dataclasses.replace(config.settings, streaming_pipeline="phase7")
    assert dataclasses.replace(
        config.settings, streaming_pipeline="phase6").streaming_pipeline == "phase6"


# --------------------------------------------------------------------------
# it selects nothing yet, and that is the point of this step
# --------------------------------------------------------------------------
def test_only_the_pipeline_reads_the_flag():
    """Step 6B wired it, and only where a pump is made or spoken. Nothing
    else in the app gets to branch on which architecture is running."""
    readers = sorted(path.name for path in ROOT.glob("*.py")
                     if path.name != "config.py"
                     and "streaming_pipeline" in path.read_text())
    assert readers == ["pipeline.py"], f"unexpected readers: {readers}"


def test_the_legacy_queue_is_untouched():
    """The shipped path must still be the bounded queue it has always been.

    Step 6B made Phase 6 selectable, so this no longer forbids the flag or
    the call. What it holds is that the legacy pump is still built the way it
    always was, and that each Phase 6 half is named exactly once - inside the
    selector - so the two can never be mixed. The Step 5 change to `_start`
    itself is pinned against trunk in `tests/test_phase6_equivalence.py`.
    """
    import re

    source = (ROOT / "pipeline.py").read_text()
    assert "asyncio.Queue(maxsize=QUEUE_DEPTH)" in source
    for name in ("_start_phase6", "_speak_phase6"):
        assert re.search(rf"def {name}\b", source), f"{name} is missing"
    # Phase 6 is now reachable, but only through the selector - never named
    # directly at a call site, which is how the two halves stay in step.
    assert source.count("self._start_phase6(") == 1
    assert source.count("self._speak_phase6(") == 1
    assert "self._start(sentences)" in source, "the legacy pump is still built"
