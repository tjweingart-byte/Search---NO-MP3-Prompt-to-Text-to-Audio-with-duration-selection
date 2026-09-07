"""The blind listening test's machinery: blinding, loudness, and the sheets."""
from __future__ import annotations

import json
import pathlib
import sys
import wave

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

numpy = pytest.importorskip("numpy")

from experiments import voice_bakeoff as bake

PASSAGES = [
    {"id": "intimate", "label": "intimate", "stresses": "warmth",
     "watch_for": "leaning in?", "text": "One."},
    {"id": "energetic", "label": "energetic", "stresses": "momentum",
     "watch_for": "stress?", "text": "Two."},
    {"id": "authoritative", "label": "news", "stresses": "numbers",
     "watch_for": "decimals?", "text": "Three."},
]
KEYS = ["chatterbox_base", "chatterbox_turbo", "kokoro", "piper"]


def test_letters_do_not_carry_across_passages():
    """Otherwise a listener learns 'C is the fast one' on passage two."""
    maps = [bake.assign_letters(KEYS, p["id"], seed=7) for p in PASSAGES]
    assert not all(m == maps[0] for m in maps[1:]), (
        "every passage got the same arrangement; the blinding is not blind")


def test_letters_are_a_bijection():
    mapping = bake.assign_letters(KEYS, "intimate", seed=7)
    assert sorted(mapping) == sorted(KEYS)
    assert sorted(mapping.values()) == ["A", "B", "C", "D"]


def test_the_arrangement_is_reproducible_from_the_seed():
    """A second opinion later must hear the same arrangement."""
    assert (bake.assign_letters(KEYS, "intimate", 7)
            == bake.assign_letters(KEYS, "intimate", 7))
    assert (bake.assign_letters(KEYS, "intimate", 7)
            != bake.assign_letters(KEYS, "intimate", 8))


def test_the_arrangement_does_not_follow_the_input_order():
    """If it did, the roster order in the source would leak the answer."""
    forward = bake.assign_letters(KEYS, "intimate", 7)
    backward = bake.assign_letters(list(reversed(KEYS)), "intimate", 7)
    assert forward != backward or len(set(forward.values())) == len(KEYS)
    # The first-listed candidate must not reliably be Voice A.
    firsts = {bake.assign_letters(KEYS, f"p{i}", 7)[KEYS[0]] for i in range(12)}
    assert firsts != {"A"}


def test_loudness_normalisation_removes_a_gain_difference():
    """Louder wins blind tests for the wrong reason."""
    rate = 24000
    tone = numpy.sin(numpy.linspace(0, 400 * numpy.pi, rate)).astype("float32")
    quiet, _, _ = bake.normalise(tone * 0.05, rate)
    loud, _, _ = bake.normalise(tone * 0.9, rate)

    def rms(x):
        return float(numpy.sqrt(numpy.mean(x ** 2)))

    assert rms(loud) / rms(quiet) == pytest.approx(1.0, abs=0.15)


def test_normalisation_never_clips_past_full_scale():
    rate = 24000
    tone = numpy.sin(numpy.linspace(0, 400 * numpy.pi, rate)).astype("float32")
    out, _, _ = bake.normalise(tone * 0.999, rate)
    assert float(numpy.max(numpy.abs(out))) <= 1.0


def test_normalising_silence_does_not_explode():
    out, _, _ = bake.normalise(numpy.zeros(1000, dtype="float32"), 24000)
    assert numpy.all(numpy.isfinite(out))


def test_wav_round_trips_at_the_right_rate(tmp_path):
    rate = 24000
    tone = numpy.sin(numpy.linspace(0, 200 * numpy.pi, rate)).astype("float32")
    path = tmp_path / "a.wav"
    bake.write_wav(path, tone, rate)
    with wave.open(str(path)) as handle:
        assert handle.getframerate() == rate
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getnframes() == rate


def test_the_scorecard_names_no_engine():
    """A sheet that mentions Chatterbox is not a blind sheet."""
    letters = {p["id"]: bake.assign_letters(KEYS, p["id"], 7) for p in PASSAGES}
    sheet = bake.scorecard_markdown(PASSAGES, letters)
    lowered = sheet.lower()
    for name in ("chatterbox", "kokoro", "piper", "turbo", "xtts"):
        assert name not in lowered or name == "piper", name
    # Piper may be named once, as the thing to beat, without saying which voice.
    assert "Voice A" in sheet
    for _, question in bake.SCORES:
        assert question in sheet


def test_the_scorecard_covers_every_axis_and_every_passage():
    letters = {p["id"]: bake.assign_letters(KEYS, p["id"], 7) for p in PASSAGES}
    sheet = bake.scorecard_markdown(PASSAGES, letters)
    for name, _ in bake.SCORES:
        assert f"| {name} |" in sheet
    for passage in PASSAGES:
        assert passage["text"] in sheet
        assert passage["watch_for"] in sheet


def test_the_player_points_at_letters_not_engine_names():
    letters = {p["id"]: bake.assign_letters(KEYS, p["id"], 7) for p in PASSAGES}
    page = bake.player_html(PASSAGES, letters)
    assert "chatterbox" not in page.lower()
    assert "kokoro" not in page.lower()
    for passage in PASSAGES:
        for letter in letters[passage["id"]].values():
            assert f'clips/{passage["id"]}/{letter}.wav' in page
    assert "KEY.json" in page          # the warning must be on the page


def test_the_player_loads_nothing_from_the_network():
    """It has to work on a laptop with the wifi off."""
    page = bake.player_html(PASSAGES, {p["id"]: bake.assign_letters(KEYS, p["id"], 7)
                                       for p in PASSAGES})
    for marker in ("http://", "https://", "//cdn", "<script"):
        assert marker not in page
