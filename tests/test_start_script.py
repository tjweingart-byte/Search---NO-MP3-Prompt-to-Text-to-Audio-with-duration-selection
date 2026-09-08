"""The launcher must actually stop and ask for the key.

It once did not. `read -r answer` returns immediately when stdin is not an
interactive terminal, so a question printed and was declined in the same
breath - which from the outside is indistinguishable from never being asked,
and is exactly how it was reported: "FAM starts without prompting me".

The lesson outlived the feature it was learned on (a hosted voice, since
removed): **never gate a key prompt behind a yes/no `read`**, and if the key
cannot be typed here, say so rather than staging a conversation that answers
itself.

These read the script rather than running it, because running it builds a
virtual environment and downloads a voice model.
"""
from __future__ import annotations

import pathlib

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
    """setup_key.py already asks, and already treats an empty line as
    "nothing changed". A y/N gate in front of it adds a way to fail and
    nothing else."""
    assert "setup_key.py" in script
    assert "[y/N]" not in script, "a yes/no gate is back in front of a key prompt"
    assert "read -r answer" not in script, "the self-answering read is back"


def test_piper_is_what_the_launcher_sets_up(script):
    """The voice runs on the listener's machine. No key, no quota, no
    per-word cost - which is the property the hosted experiment failed on."""
    assert "setup_voices.py" in script
    assert "PiperEngine.available()" in script


def test_nothing_hosted_survived_the_removal(script):
    """A knob left behind is an invitation to turn it back on."""
    assert "wellsaid" not in script.lower()


def test_it_reports_what_you_will_hear_before_serving(script):
    """Starting quietly broken is the failure this project has paid for most."""
    assert "default_voice" in script and "list_voices" in script
