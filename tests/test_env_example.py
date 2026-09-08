"""The shipped .env.example must agree with the settled defaults.

It did not, and it turned on the exact three things that produce "a few seconds
of audio, then thirty seconds of nothing": the cold open (filler by
construction), web search on every request (10-25s before the first word), and
the slower model. Copying the example - the documented thing to do - configured
the product against its own one-sentence spec.

A comment saying "off by default" in config.py is not a defence when the file
people actually copy says 1.
"""
from __future__ import annotations

import os
import pathlib
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = pathlib.Path(__file__).resolve().parent.parent


def example_values() -> dict[str, str]:
    values = {}
    for line in (ROOT / ".env.example").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        values[name.strip()] = value.split("#")[0].strip()
    return values


@pytest.fixture
def defaults(monkeypatch):
    """config's own defaults, with the environment out of the way."""
    import importlib

    for name in list(os.environ):
        if name.startswith(("ANTHROPIC_", "MODEL", "ENABLE_", "EFFORT", "MAX_WEB",
                            "CACHE_", "TARGET_", "ALLOW_")):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FAM_IGNORE_DOTENV", "1")
    import config

    importlib.reload(config)
    yield config.settings
    importlib.reload(config)


@pytest.mark.parametrize("key, attr", [
    ("MODEL", "model"),
    ("EFFORT", "effort"),
    ("SEARCH_MODE", "search_mode"),
])
def test_text_settings_match(defaults, key, attr):
    assert example_values()[key] == str(getattr(defaults, attr))


def test_search_is_not_on_for_every_episode(defaults):
    """The one-sentence spec expressed as a setting. `always` pays 10-25s on
    every episode including the ones that never needed it."""
    assert defaults.search_mode == "auto", "the default search mode drifted"
    assert example_values()["SEARCH_MODE"] == "auto", (
        "SEARCH_MODE in .env.example is not auto: copying it puts seconds in "
        "front of the first word, which is the one thing the product refuses"
    )


def test_the_filler_setting_is_gone_from_both(defaults):
    """The cold open was removed, not switched off. A setting left behind is
    an invitation to turn it back on."""
    assert not hasattr(defaults, "enable_cold_open")
    assert "ENABLE_COLD_OPEN" not in example_values()
    assert "COLD_OPEN" not in (pathlib.Path(__file__).resolve().parent.parent
                               / ".env.example").read_text()


def test_no_setting_in_the_example_disagrees_with_the_code(defaults):
    """The general form. A per-setting test only catches the ones anyone
    thought to list; this catches the next one to drift."""
    import config

    booleans = {"1": True, "0": False, "true": True, "false": False}
    # The key line is a placeholder by design - the example cannot ship a
    # credential, and setup_key.py is where a real one goes.
    placeholders = {"ANTHROPIC_API_KEY"}
    mismatches = []
    for key, raw in example_values().items():
        attr = key.lower()
        if key in placeholders or not hasattr(defaults, attr):
            continue
        actual = getattr(defaults, attr)
        if isinstance(actual, bool):
            wanted = booleans.get(raw.lower())
            if wanted is not None and wanted is not actual:
                mismatches.append(f"{key}={raw} but default is {actual}")
        elif isinstance(actual, str) and raw and raw != actual:
            mismatches.append(f"{key}={raw} but default is {actual}")
    assert not mismatches, "the example disagrees with the code: " + "; ".join(mismatches)
