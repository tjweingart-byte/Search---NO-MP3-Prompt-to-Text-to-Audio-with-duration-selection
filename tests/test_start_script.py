"""The launcher must actually stop and ask.

It did not. `read -r answer` returns immediately when stdin is not an
interactive terminal, so the question printed and was declined in the same
breath - which from the outside is indistinguishable from never being asked,
and is exactly how it was reported: "FAM starts without prompting me".

These read the script rather than running it, because running it installs a
virtual environment and downloads a voice. What they pin is the shape of the
decision, which is where the bug was.
"""
from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
START = ROOT / "start.sh"


@pytest.fixture(scope="module")
def script() -> str:
    return START.read_text()


def test_the_launcher_exists_and_is_executable():
    assert START.exists()
    assert START.stat().st_mode & 0o111, "start.sh is not executable"


def test_a_missing_key_reaches_the_setup_script_without_a_gate(script):
    """No "would you like to?" in front of it. The gate added a way to fail
    and nothing else: setup_wellsaid.py already treats an empty line as
    "nothing changed", so it can simply be run."""
    assert "setup_wellsaid.py" in script
    assert "[y/N]" not in script, "a yes/no gate is back in front of the prompt"
    assert "read -r answer" not in script, "the self-answering read is back"


def test_a_non_interactive_run_says_so_instead_of_pretending_to_ask(script):
    """The honest version of the bug: if the key cannot be typed here, say
    that and give the command, rather than printing a question and answering
    it."""
    assert "[ -t 0 ]" in script, "stdin is not checked for being a terminal"
    assert "interactive terminal" in script


def test_it_says_where_it_looked_for_the_key(script):
    """"The key is not set" is not much help when you cannot tell which file
    was read - the same lesson as key_source() in config.py."""
    assert "looked for it in" in script


def test_piper_is_untouched_by_any_of_this(script):
    """The brief was explicit: do not change existing Piper behaviour."""
    assert "setup_voices.py" in script
    assert "PiperEngine.available()" in script
