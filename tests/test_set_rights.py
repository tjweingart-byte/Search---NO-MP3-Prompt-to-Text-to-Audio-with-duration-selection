"""Rights records, matched to the recording rather than to a neutral id."""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

numpy = pytest.importorskip("numpy")

from experiments import voice_bakeoff as bake
from tools import check_reference_audio as check
from tools import equalise_references as eq
from tools import set_rights

RATE = 24000
FILLED = dict(recorded_for="recorded today for the experiment",
              consent=True, commercial_use=True,
              synthetic_voice_cleared=True, notes="release on file")


def _folder(tmp_path, durations=(16.7, 16.6, 14.4)):
    for name, seconds in zip(("Ian.wav", "Kennedy.wav", "TJ.wav"), durations):
        n = int(RATE * seconds)
        t = numpy.linspace(0, seconds, n)
        bake.write_wav(tmp_path / name,
                       (numpy.sin(2 * numpy.pi * 130 * t) * 0.4).astype("float32"),
                       RATE)
    check.adopt(tmp_path)
    return tmp_path


def _write(folder, source, speaker, **overrides):
    values = {**FILLED, **overrides}
    key = set_rights.find(folder, source)
    (folder / f"{key}.rights.json").write_text(json.dumps({
        "source": values["recorded_for"], "speaker": speaker,
        "consent": values["consent"], "commercial_use": values["commercial_use"],
        "synthetic_voice_cleared": values["synthetic_voice_cleared"],
        "notes": values["notes"]}, indent=2), encoding="utf-8")
    return key


def test_a_recording_is_found_by_stem_or_full_name(tmp_path):
    folder = _folder(tmp_path)
    assert set_rights.find(folder, "Ian") == set_rights.find(folder, "Ian.wav")
    assert set_rights.find(folder, "ian") == set_rights.find(folder, "Ian")


def test_matching_survives_equalising(tmp_path, monkeypatch):
    """After equalising, sources.json points at working/reference_N.wav.

    Matching by neutral id would then be matching on the wrong thing, and a
    person's consent could be attached to somebody else's voice.
    """
    folder = _folder(tmp_path)
    before = {name: set_rights.find(folder, name)
              for name in ("Ian", "Kennedy", "TJ")}

    monkeypatch.setattr(sys, "argv", ["equalise", str(folder)])
    eq.main()

    after = {name: set_rights.find(folder, name)
             for name in ("Ian", "Kennedy", "TJ")}
    assert before == after
    assert len(set(after.values())) == 3


def test_each_person_lands_on_their_own_reference(tmp_path, monkeypatch):
    folder = _folder(tmp_path)
    monkeypatch.setattr(sys, "argv", ["equalise", str(folder)])
    eq.main()

    keys = {name: _write(folder, name, f"{name} Surname")
            for name in ("Ian", "Kennedy", "TJ")}
    manifest = json.loads(
        (folder / eq.WORKING / "MANIFEST.json").read_text())["references"]
    for name, key in keys.items():
        assert manifest[key]["source"].startswith(name)
        record = json.loads((folder / f"{key}.rights.json").read_text())
        assert record["speaker"] == f"{name} Surname"


def test_an_unknown_recording_is_refused_with_the_known_ones(tmp_path):
    folder = _folder(tmp_path)
    with pytest.raises(SystemExit) as caught:
        set_rights.find(folder, "Someone")
    message = str(caught.value)
    assert "no recording called 'Someone'" in message
    assert "Ian.wav" in message


def test_yes_no_refuses_anything_ambiguous():
    assert set_rights.yes_no("Yes") is True
    assert set_rights.yes_no("no") is False
    for value in ("not yet", "maybe", "pending", ""):
        with pytest.raises(Exception):
            set_rights.yes_no(value)


def test_the_clearances_have_no_defaults():
    """A clearance that was never stated must not become a true.

    argparse is configured with required=True for all three; this asserts it,
    because a default here would silently manufacture permission.
    """
    actions = {option: action
               for action in set_rights.build_parser()._actions
               for option in action.option_strings}
    for flag in ("--consent", "--commercial-use", "--synthetic-voice-cleared"):
        action = actions[flag]
        assert action.required is True, flag
        assert action.default is None, flag
        assert action.type is set_rights.yes_no, flag


def test_a_refused_clearance_writes_false_and_still_blocks(tmp_path):
    folder = _folder(tmp_path)
    key = _write(folder, "Ian", "Ian Surname", synthetic_voice_cleared=False)
    record = json.loads((folder / f"{key}.rights.json").read_text())
    assert record["synthetic_voice_cleared"] is False
    assert any("synthetic_voice_cleared is false" in problem
               for problem in check.check_rights(folder, key))


def test_a_completed_record_passes_the_gate(tmp_path):
    folder = _folder(tmp_path)
    key = _write(folder, "Kennedy", "Kennedy Surname")
    assert check.check_rights(folder, key) == []
