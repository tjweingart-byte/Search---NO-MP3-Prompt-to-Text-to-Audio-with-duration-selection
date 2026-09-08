"""Equal-length working references, derived without touching the originals."""
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

RATE = 24000


def _write(path, seconds, offset=0.0, freq=140.0):
    n = int(RATE * seconds)
    t = numpy.linspace(0, seconds, n)
    wave = (numpy.sin(2 * numpy.pi * freq * t) * 0.4 + offset).astype("float32")
    bake.write_wav(path, wave, RATE)
    return wave


def _folder(tmp_path, durations=(16.7, 14.4, 16.6)):
    names = ("Ian.wav", "TJ.wav", "Kennedy.wav")
    for name, seconds in zip(names, durations):
        _write(tmp_path / name, seconds)
    check.adopt(tmp_path)
    return tmp_path


def test_the_crop_is_head_anchored(tmp_path):
    """The s3gen reference reads the first 10 s and the T3 prompt the first 6 s.

    A head-anchored crop leaves both bit-identical to the original; a centre or
    tail crop would shift them and quietly change all three conditionings.
    """
    original = _write(tmp_path / "a.wav", 16.0)
    cropped, _given = eq.crop_at_zero_crossing(original, RATE, 14.0)
    head = int(RATE * 10)
    assert numpy.array_equal(cropped[:head], original[:head])


def test_nothing_is_padded_scaled_or_faded(tmp_path):
    """Only samples are dropped. Padding would dilute the very embedding this
    is equalising, and a fade would alter amplitude."""
    original = _write(tmp_path / "a.wav", 16.0)
    cropped, _given = eq.crop_at_zero_crossing(original, RATE, 14.0)
    assert len(cropped) < len(original)
    assert numpy.array_equal(cropped, original[:len(cropped)])


def test_the_cut_lands_on_a_zero_crossing(tmp_path):
    """So the working file ends without a step discontinuity."""
    # An offset sine crosses zero away from the exact target sample.
    original = _write(tmp_path / "a.wav", 16.0, offset=0.05, freq=97.0)
    cropped, given = eq.crop_at_zero_crossing(original, RATE, 14.0)
    assert given >= 0
    tail = float(abs(cropped[-1]))
    typical = float(numpy.median(numpy.abs(original)))
    assert tail < typical, "the file ends on a large sample, i.e. a step"


def test_the_search_window_is_short_enough_to_be_irrelevant(tmp_path):
    original = _write(tmp_path / "a.wav", 16.0, offset=0.05, freq=97.0)
    _cropped, given = eq.crop_at_zero_crossing(original, RATE, 14.0)
    assert given <= eq.ZERO_CROSSING_WINDOW_MS / 1000.0 * RATE


def test_a_target_longer_than_the_shortest_is_refused(tmp_path, monkeypatch):
    """Reaching it would mean padding, which is the thing being avoided."""
    folder = _folder(tmp_path)
    monkeypatch.setattr(sys, "argv",
                        ["equalise", str(folder), "--seconds", "20"])
    with pytest.raises(SystemExit) as caught:
        eq.main()
    assert "padding" in str(caught.value)


def test_it_equalises_and_leaves_the_originals_alone(tmp_path, monkeypatch):
    folder = _folder(tmp_path)
    before = {p.name: p.read_bytes()
              for p in folder.iterdir() if p.suffix == ".wav"}

    monkeypatch.setattr(sys, "argv", ["equalise", str(folder)])
    assert eq.main() == 0

    for name, blob in before.items():
        assert (folder / name).read_bytes() == blob, f"{name} was modified"

    working = folder / eq.WORKING
    durations = []
    for key in check.neutral_ids():
        path = working / f"{key}.wav"
        assert path.exists()
        samples, rate = eq.decode(path)
        durations.append(len(samples) / rate)
    assert max(durations) - min(durations) < 0.01
    assert min(durations) == pytest.approx(14.4, abs=0.01)


def test_sources_is_repointed_at_the_working_copies(tmp_path, monkeypatch):
    folder = _folder(tmp_path)
    monkeypatch.setattr(sys, "argv", ["equalise", str(folder)])
    eq.main()

    sources = check.load_sources(folder)
    for key in check.neutral_ids():
        assert sources[key].parent.name == eq.WORKING
        assert sources[key].exists()


def test_the_manifest_records_where_each_working_copy_came_from(tmp_path,
                                                                monkeypatch):
    """Equalising must not lose the provenance of the originals."""
    folder = _folder(tmp_path)
    monkeypatch.setattr(sys, "argv", ["equalise", str(folder)])
    eq.main()

    manifest = json.loads(
        (folder / eq.WORKING / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["target_seconds"] == pytest.approx(14.4, abs=0.01)
    sources = {entry["source"] for entry in manifest["references"].values()}
    assert sources == {"Ian.wav", "TJ.wav", "Kennedy.wav"}
    for entry in manifest["references"].values():
        assert entry["removed_seconds"] >= 0
        assert entry["working_seconds"] == pytest.approx(14.4, abs=0.01)


def test_the_checker_passes_on_duration_after_equalising(tmp_path, monkeypatch,
                                                          capsys):
    folder = _folder(tmp_path)
    monkeypatch.setattr(sys, "argv", ["equalise", str(folder)])
    eq.main()
    capsys.readouterr()

    sources = check.load_sources(folder)
    durations = []
    for key in check.neutral_ids():
        info, problems, _warnings = check.check_one(sources[key])
        assert not any("shorter than" in p for p in problems)
        durations.append(info["seconds"])
    assert max(durations) - min(durations) <= check.DURATION_TOLERANCE
