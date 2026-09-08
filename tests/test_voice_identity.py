"""The identity bake-off: seeding, blinding, the rights gate, the sheet."""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

numpy = pytest.importorskip("numpy")

from experiments import voice_bakeoff as bake
from experiments import voice_identity as identity
from tools import check_reference_audio as check
from tools import voice_identity_bakeoff as runner

PASSAGES = [
    {"id": "intimate", "label": "intimate", "text": "One."},
    {"id": "energetic", "label": "energetic", "text": "Two."},
    {"id": "authoritative", "label": "news", "text": "Three."},
]
KEYS = [k for k, _, _ in identity.IDENTITIES]


def test_the_candidates_are_three_neutrally_named_references():
    """Named after nothing.

    A filename like magnetic.wav would assert which speaker is the magnetic
    one - a mapping nobody has verified - and prime whoever listens.
    """
    assert KEYS == ["reference_1", "reference_2", "reference_3"]
    for key, label, _note in identity.IDENTITIES:
        for direction in ("magnetic", "storyteller", "authority", "human"):
            assert direction not in key.lower()
            assert direction not in label.lower()


def test_the_qualities_sought_are_recorded_but_unattached():
    """The four directions still exist as what we are listening for; they are
    just not claims about any particular file."""
    qualities = dict(identity.QUALITIES_SOUGHT)
    assert set(qualities) == {"magnetic", "human", "storyteller",
                              "modern authority"}
    notes = " ".join(note for _k, _l, note in identity.IDENTITIES).lower()
    for direction in qualities:
        assert direction not in notes


def test_every_identity_shares_one_seed_per_passage():
    """The point of the seed: no voice may win on a luckier sample.

    All four identities generate the same passage under the same random
    stream, so the difference between them is the reference and nothing else.
    """
    for passage in PASSAGES:
        seeds = {identity.seed_for(passage["id"], 42) for _ in KEYS}
        assert len(seeds) == 1


def test_seeds_differ_between_passages():
    """Same seed everywhere would correlate the sampling across passages."""
    seeds = {identity.seed_for(p["id"], 42) for p in PASSAGES}
    assert len(seeds) == len(PASSAGES)


def test_generation_settings_are_pinned_at_the_defaults():
    """A voice must not win because it got different settings.

    These are Chatterbox Base's own defaults; delivery tuning is the next
    experiment, deliberately not this one.
    """
    assert identity.GENERATION == {
        "exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8,
        "repetition_penalty": 1.2, "min_p": 0.05, "top_p": 1.0}


def test_letters_are_randomised_per_passage():
    maps = [bake.assign_letters(KEYS, p["id"], 20260908) for p in PASSAGES]
    assert not all(m == maps[0] for m in maps[1:])
    for mapping in maps:
        assert sorted(mapping.values()) == ["A", "B", "C"]


def test_the_choice_sheet_asks_only_what_was_asked_for():
    letters = {p["id"]: bake.assign_letters(KEYS, p["id"], 1) for p in PASSAGES}
    sheet = identity.choice_sheet(PASSAGES, letters)
    assert "Best voice for this passage" in sheet
    assert "Overall favourite" in sheet
    assert "Would you ship this as FAM's voice?" in sheet
    assert "Notes" in sheet
    # The eight-axis scorecard is explicitly not wanted here: no scoring rows.
    for axis in ("naturalness", "warmth", "intrigue", "expressiveness"):
        assert f"| {axis} |" not in sheet
        assert axis not in sheet.split("## Overall")[0]


def test_neither_sheet_nor_page_names_an_identity():
    letters = {p["id"]: bake.assign_letters(KEYS, p["id"], 1) for p in PASSAGES}
    sheet = identity.choice_sheet(PASSAGES, letters).lower()
    page = identity.identity_player_html(PASSAGES, letters).lower()
    for name in ("magnetic", "storyteller", "modern authority",
                 "reference_1", "reference_2", "reference_3"):
        assert name not in sheet, name
        assert name not in page, name


def test_the_page_loads_nothing_from_the_network():
    letters = {p["id"]: bake.assign_letters(KEYS, p["id"], 1) for p in PASSAGES}
    page = identity.identity_player_html(PASSAGES, letters)
    for marker in ("http://", "https://", "<script"):
        assert marker not in page


def test_a_missing_rights_record_blocks_the_voice(tmp_path):
    assert any("no rights record" in p
               for p in check.check_rights(tmp_path, "reference_1"))


def test_an_unanswered_rights_field_blocks_the_voice(tmp_path):
    (tmp_path / "reference_1.rights.json").write_text(json.dumps({
        "source": "TODO", "speaker": "A person", "consent": True,
        "commercial_use": True, "synthetic_voice_cleared": True}))
    problems = check.check_rights(tmp_path, "reference_1")
    assert any("'source' is not answered" in p for p in problems)


def test_a_refused_clearance_blocks_the_voice(tmp_path):
    """False is not 'unanswered'; it is a no, and it must stop the run."""
    (tmp_path / "reference_1.rights.json").write_text(json.dumps({
        "source": "a recording", "speaker": "A person", "consent": True,
        "commercial_use": True, "synthetic_voice_cleared": False,
        "notes": "the agreement does not cover synthesis"}))
    problems = check.check_rights(tmp_path, "reference_1")
    assert any("synthetic_voice_cleared is false" in p for p in problems)


def test_a_complete_rights_record_passes(tmp_path):
    (tmp_path / "reference_1.rights.json").write_text(json.dumps({
        "source": "recorded in-house 2026-09-08", "speaker": "A person",
        "consent": True, "commercial_use": True,
        "synthetic_voice_cleared": True, "notes": "signed release on file"}))
    assert check.check_rights(tmp_path, "reference_1") == []


def test_references_that_differ_in_length_are_rejected():
    """The speaker embedding reads the whole file while the other two
    conditionings are truncated, so unequal lengths feed the voices unequally."""
    assert check.DURATION_TOLERANCE == 2.0
    assert check.MIN_SECONDS >= check.DEC_COND_SECONDS


def test_a_short_reference_is_rejected(tmp_path):
    rate = 24000
    tone = numpy.sin(numpy.linspace(0, 800 * numpy.pi, rate * 5)).astype("float32")
    path = tmp_path / "reference_1.wav"
    bake.write_wav(path, tone * 0.5, rate)
    _info, problems, _warnings = check.check_one(path)
    assert any("shorter than" in p for p in problems)


def test_a_clipped_reference_is_rejected(tmp_path):
    rate = 24000
    tone = numpy.sin(numpy.linspace(0, 900 * numpy.pi, rate * 13)).astype("float32")
    path = tmp_path / "reference_2.wav"
    bake.write_wav(path, tone, rate)          # peaks at full scale
    _info, problems, _warnings = check.check_one(path)
    assert any("clipped" in p for p in problems)


def test_a_good_reference_passes(tmp_path):
    rate = 24000
    n = rate * 13
    speech = (numpy.sin(numpy.linspace(0, 1400 * numpy.pi, n))
              * (0.25 + 0.25 * numpy.sin(numpy.linspace(0, 30 * numpy.pi, n))))
    path = tmp_path / "reference_3.wav"
    bake.write_wav(path, speech.astype("float32"), rate)
    info, problems, _warnings = check.check_one(path)
    assert problems == []
    assert 12.5 < info["seconds"] < 13.5


def test_missing_reference_audio_stops_before_any_model_loads(tmp_path):
    with pytest.raises(SystemExit) as caught:
        runner.resolve_references(tmp_path)
    message = str(caught.value)
    assert "missing reference audio" in message
    for key in KEYS:
        assert key in message
