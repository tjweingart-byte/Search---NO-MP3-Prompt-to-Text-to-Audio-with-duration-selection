"""The combined 4090 runner, checked on this machine before any card is rented.

Every fault this file catches is one that would otherwise have been found on a
billing meter: a corpus missing a bucket, a run silently overwriting the last
one, a preflight that passes without a reference voice.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tools import combined_4090_experiment as combined


def _corpus(path: pathlib.Path, buckets=("short", "medium", "long")) -> pathlib.Path:
    chunks = [{"bucket": bucket, "words": 30, "text": f"A {bucket} chunk."}
              for bucket in buckets for _ in range(3)]
    path.write_text(json.dumps({"chunks": chunks}), encoding="utf-8")
    return path


def test_the_corpus_gives_every_bucket_the_same_number_of_chunks(tmp_path):
    picked = combined.load_corpus(_corpus(tmp_path / "c.json"), per_bucket=2)
    counts = {b: sum(1 for c in picked if c["bucket"] == b) for b in combined.BUCKETS}
    assert counts == {"short": 2, "medium": 2, "long": 2}


def test_a_corpus_missing_a_bucket_stops_the_run(tmp_path):
    """Rather than quietly benchmarking two thirds of the length range."""
    path = _corpus(tmp_path / "c.json", buckets=("short", "medium"))
    with pytest.raises(SystemExit, match="long"):
        combined.load_corpus(path, per_bucket=2)


def test_a_second_run_never_overwrites_the_first(tmp_path):
    out = tmp_path / "combined"
    out.mkdir()
    (out / "pipeline.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="already exists"):
        combined.main(["--experiment", "b", "--out", str(out),
                       "--reference", str(_corpus(tmp_path / "r.wav"))])


def test_running_without_a_reference_voice_is_refused(tmp_path):
    """--reference names the voice the identity bake-off chose. There is no
    default, because a default would silently benchmark the wrong voice."""
    with pytest.raises(SystemExit, match="reference"):
        combined.main(["--experiment", "a", "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit, match="reference"):
        combined.main(["--preflight"])


def test_analyse_writes_a_report_from_whichever_half_has_finished(tmp_path):
    """Experiment A and B are separate processes; if B fails, A's numbers must
    still be readable rather than lost with it."""
    out = tmp_path / "combined"
    out.mkdir()
    (out / "bench_chatterbox_base.json").write_text(json.dumps({
        "reference": "reference_1.wav", "generation_settings": {},
        "watermarking": "applied by chatterbox; not bypassed",
        "cold_load_seconds": 11.2, "first_generation_seconds": 3.4,
        "gpu_peak": {"available": False},
        "by_bucket": {"short": {"n": 3, "words_mean": 28.0,
                                "generate_p50": 2.18, "audio_p50": 9.9,
                                "realtime_p50": 4.5}},
    }), encoding="utf-8")
    assert combined.analyse(out) == 0
    report = (out / "ANALYSIS.md").read_text(encoding="utf-8")
    assert "cold model load: 11.20s" in report
    assert "DEVELOPMENT BENCHMARK" in report
    assert json.loads((out / "results.json").read_text())["pipeline"] is None


def test_analyse_refuses_an_empty_directory(tmp_path):
    with pytest.raises(SystemExit, match="nothing to analyse"):
        combined.analyse(tmp_path)


def test_exa_is_verified_by_calling_it_not_by_reading_the_environment(monkeypatch):
    """"A key is set" is not "the key works" - the rule that cost this project
    four sessions."""
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    ok, detail = combined.verify_exa()
    assert ok is False and "EXA_API_KEY" in detail


# --------------------------------------------------------------------------
# what leaves this machine
# --------------------------------------------------------------------------
from tools import pack_for_pod                                   # noqa: E402


def _voice(tmp_path: pathlib.Path, **rights) -> pathlib.Path:
    root = tmp_path / "references"
    (root / "working").mkdir(parents=True)
    wav = root / "working" / "reference_2.wav"
    wav.write_bytes(b"RIFF....WAVE")
    record = {"source": "a recording", "speaker": "Someone Real",
              "consent": "yes", "commercial_use": "yes",
              "synthetic_voice_cleared": "yes", "notes": "-"}
    record.update(rights)
    (root / "reference_2.rights.json").write_text(json.dumps(record),
                                                  encoding="utf-8")
    return wav


def test_a_cleared_recording_packs_and_reports_its_digest(tmp_path):
    info = pack_for_pod.check_reference(_voice(tmp_path))
    assert info["name"] == "reference_2.wav" and len(info["sha256"]) == 16
    assert "path" in info and "rights" not in info


def test_an_uncleared_recording_is_refused_before_it_reaches_a_pod(tmp_path):
    """The rights gate has to hold in the packer too, or it holds nowhere."""
    wav = _voice(tmp_path, commercial_use=None)
    with pytest.raises(SystemExit, match="commercial_use"):
        pack_for_pod.check_reference(wav)


def test_a_recording_with_no_rights_record_is_refused(tmp_path):
    root = tmp_path / "references" / "working"
    root.mkdir(parents=True)
    wav = root / "reference_3.wav"
    wav.write_bytes(b"RIFF")
    with pytest.raises(SystemExit, match="no rights record"):
        pack_for_pod.check_reference(wav)
