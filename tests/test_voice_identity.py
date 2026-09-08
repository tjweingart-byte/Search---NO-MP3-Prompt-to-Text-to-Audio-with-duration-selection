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


def _stub_audio(path, seconds=13.0, rate=44100, amplitude=0.4):
    n = int(rate * seconds)
    t = numpy.linspace(0, seconds, n)
    wave = (numpy.sin(2 * numpy.pi * 140 * t) * amplitude).astype("float32")
    bake.write_wav(path, wave, rate)


def test_adoption_keeps_the_source_filenames(tmp_path):
    """The recordings arrived with names; renaming them is not our business."""
    for name in ("Ian.wav", "Kennedy.wav", "TJ.wav"):
        _stub_audio(tmp_path / name)
    mapping = check.adopt(tmp_path)

    assert sorted(mapping) == ["reference_1", "reference_2", "reference_3"]
    assert sorted(mapping.values()) == ["Ian.wav", "Kennedy.wav", "TJ.wav"]
    for name in ("Ian.wav", "Kennedy.wav", "TJ.wav"):
        assert (tmp_path / name).exists(), "a source recording was moved"


def test_adoption_is_not_alphabetical(tmp_path):
    """Belt and braces: a mapping anyone could infer from the filenames is
    not worth keeping, even though judging only ever shows letters."""
    for name in ("Ian.wav", "Kennedy.wav", "TJ.wav"):
        _stub_audio(tmp_path / name)
    mapping = check.adopt(tmp_path)
    alphabetical = {"reference_1": "Ian.wav", "reference_2": "Kennedy.wav",
                    "reference_3": "TJ.wav"}
    assert mapping != alphabetical


def test_adoption_writes_rights_stubs_that_do_not_pass(tmp_path):
    """A stub must be filled in, not merely present."""
    for name in ("Ian.wav", "Kennedy.wav", "TJ.wav"):
        _stub_audio(tmp_path / name)
    check.adopt(tmp_path)
    for key in ("reference_1", "reference_2", "reference_3"):
        assert (tmp_path / f"{key}.rights.json").exists()
        assert check.check_rights(tmp_path, key), f"{key} stub passed the gate"


def test_adoption_does_not_overwrite_a_completed_rights_record(tmp_path):
    for name in ("Ian.wav", "Kennedy.wav", "TJ.wav"):
        _stub_audio(tmp_path / name)
    filled = {"source": "in-house", "speaker": "someone", "consent": True,
              "commercial_use": True, "synthetic_voice_cleared": True,
              "notes": "release on file"}
    (tmp_path / "reference_1.rights.json").write_text(json.dumps(filled))
    check.adopt(tmp_path)
    assert json.loads(
        (tmp_path / "reference_1.rights.json").read_text())["speaker"] == "someone"


def test_the_wrong_number_of_recordings_is_refused(tmp_path):
    for name in ("Ian.wav", "Kennedy.wav"):
        _stub_audio(tmp_path / name)
    with pytest.raises(SystemExit) as caught:
        check.adopt(tmp_path)
    assert "found 2 recordings" in str(caught.value)


def test_m4a_is_recognised_as_a_candidate_format():
    """Chatterbox calls librosa.load, which reads m4a. The checker's format
    list has to agree, or a valid recording is refused for no reason."""
    assert ".m4a" in check.AUDIO_SUFFIXES
    for suffix in (".wav", ".mp3", ".flac"):
        assert suffix in check.AUDIO_SUFFIXES


def test_the_runner_resolves_through_the_mapping(tmp_path):
    for name in ("Ian.m4a", "Kennedy.m4a", "TJ.m4a"):
        (tmp_path / name).write_bytes(b"not really audio")
    check.adopt(tmp_path)
    resolved = runner.resolve_references(tmp_path)
    assert sorted(resolved) == ["reference_1", "reference_2", "reference_3"]
    assert sorted(p.name for p in resolved.values()) == [
        "Ian.m4a", "Kennedy.m4a", "TJ.m4a"]


def test_the_roster_log_line_carries_no_speaker_name(tmp_path):
    """progress.log persists on disk while the blind judging happens.

    A speaker's name in it gives the answer away before KEY.json is opened.
    """
    for name in ("Ian.m4a", "Kennedy.m4a", "TJ.m4a"):
        (tmp_path / name).write_bytes(b"x" * 2048)
    check.adopt(tmp_path)
    resolved = runner.resolve_references(tmp_path)

    lines = []
    for key, label, _note in identity.IDENTITIES:
        path = resolved[key]
        lines.append(f"  {key:<14}{path.suffix.lstrip('.'):<6}"
                     f"{path.stat().st_size / 1024:>8.0f} KB  {label}")
    blob = "\n".join(lines)
    for speaker in ("Ian", "Kennedy", "TJ"):
        assert speaker not in blob, f"{speaker} would be written to progress.log"


def test_sources_json_is_git_ignored():
    """It maps neutral ids to real recordings, so it is as identifying as the
    audio itself."""
    ignore = (pathlib.Path(__file__).resolve().parent.parent
              / ".gitignore").read_text()
    assert "experiments/references/*" in ignore
