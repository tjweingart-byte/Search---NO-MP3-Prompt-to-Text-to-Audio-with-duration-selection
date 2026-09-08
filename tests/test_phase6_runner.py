"""The Phase 6 runner and the policy-fitting tool, exercised end to end here.

Everything in this file runs with no GPU, no key and no network, and every
assertion is one that would otherwise have been discovered on a billing meter.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments import decoupled_pipeline as dp
from experiments.speech_assembler import AssemblyPolicy
from tools import fit_chunk_policy, phase6_experiment as runner


def _run(tmp_path, *extra) -> pathlib.Path:
    out = tmp_path / "phase6_STUB"
    code = runner.main(["--dry-run", "--runs", "1", "--out", str(out), *extra])
    return out, code


# --------------------------------------------------------------------------
# the whole path, on stubs
# --------------------------------------------------------------------------
def test_the_decoupled_run_passes_every_assertion(tmp_path):
    out, code = _run(tmp_path)
    assert code == 0
    results = json.loads((out / "results.json").read_text())
    record = results["runs"][0]
    assert record["passed"] is True and record["problems"] == []
    assert record["timing"]["claude_reader_blocked_seconds"] == 0.0
    assert record["decoupling"]["exercised"] is True
    assert record["decoupling"]["tts_queue_saturated"] is True


def test_the_stub_actually_saturates_the_queue(tmp_path):
    """A stub whose voice is instant proves the decoupling in the one case
    where nothing tests it. This one follows the measured 4090 curve."""
    out, _ = _run(tmp_path)
    record = json.loads((out / "results.json").read_text())["runs"][0]
    assert record["timing"]["peak_tts_queue_depth"] == dp.QUEUE_DEPTH
    assert record["timing"]["assembler_blocked_seconds"] > 0


def test_the_coupled_fixture_fails_the_decoupling_assertion(tmp_path):
    """Phase 5's shape, rebuilt on purpose. If this passes, the assertion is
    decoration and every Phase 6 result is worthless."""
    out = tmp_path / "phase6_STUB_coupled"
    assert runner.main(["--dry-run", "--coupled", "--runs", "1",
                        "--out", str(out)]) == 0     # the fixture failing is a pass
    record = json.loads((out / "results.json").read_text())["runs"][0]
    assert record["passed"] is False
    assert any("reader was blocked" in p for p in record["problems"])
    assert "coupled fixture" in (out / "ANALYSIS.md").read_text().lower()


def test_the_coupled_fixture_makes_the_phase_5_mistake_visible(tmp_path):
    """One sentence per synthesis, and a Claude number inflated by our queue."""
    decoupled, _ = _run(tmp_path / "a")
    coupled = tmp_path / "b" / "phase6_STUB_coupled"
    runner.main(["--dry-run", "--coupled", "--runs", "1", "--out", str(coupled)])

    good = json.loads((decoupled / "results.json").read_text())["runs"][0]
    bad = json.loads((coupled / "results.json").read_text())["runs"][0]
    assert bad["chunking"]["tts_invocations"] == bad["chunking"]["raw_sentences"]
    assert good["chunking"]["tts_invocations"] < good["chunking"]["raw_sentences"]
    assert bad["chunking"]["chunks_under_10_words"] > 0
    assert good["chunking"]["chunks_under_10_words"] == 0
    assert (bad["timing"]["claude_stream_seconds"]
            > good["timing"]["claude_stream_seconds"])


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------
def test_the_analysis_opens_with_the_six_plain_english_answers(tmp_path):
    out, _ = _run(tmp_path)
    report = (out / "ANALYSIS.md").read_text()
    head = report[:report.index("## Architecture")]
    assert "## Executive result" in head
    for number in range(1, 7):
        assert f"{number}. **" in head, f"answer {number} missing"
    assert "Search to first listen" in head
    assert "Claude decoupled from TTS backpressure" in head
    assert "Playback stalls" in head


def test_the_report_never_reconstructs_phase_5_per_chunk_data(tmp_path):
    """Phase 5's chunk table is not in this repository. Absent beats invented."""
    out, _ = _run(tmp_path)
    report = (out / "ANALYSIS.md").read_text()
    assert "not reported" in report
    assert "not in this repository" in report


def test_a_stub_is_stamped_and_fenced(tmp_path):
    out, _ = _run(tmp_path)
    assert "NOT A RESULT" in json.loads((out / "results.json").read_text())["label"]
    assert "NOT A RESULT" in (out / "ANALYSIS.md").read_text()
    assert "RTX 4090" not in (out / "ANALYSIS.md").read_text()
    with pytest.raises(SystemExit, match="STUB"):
        runner.main(["--dry-run", "--out", str(tmp_path / "phase6_4090_real")])
    with pytest.raises(SystemExit, match="stub directory"):
        runner.main(["--out", str(tmp_path / "phase6_STUB2"),
                     "--reference", "x.wav"])


def test_the_artefacts_a_run_must_leave_behind(tmp_path):
    out, _ = _run(tmp_path)
    for name in ("ANALYSIS.md", "results.json", "events.jsonl", "chunks.json"):
        assert (out / name).exists(), name
    assert (out / "audio" / "cold_episode.wav").exists()
    assert (out / "audio" / "cold_script.txt").exists()
    assert list((out / "audio" / "cold").glob("chunk_*.wav"))
    chunks = json.loads((out / "chunks.json").read_text())["cold"]
    assert len(chunks["raw_sentences"]) > len(chunks["chunks"])


def test_a_real_run_needs_the_selected_reference_voice():
    with pytest.raises(SystemExit, match="reference"):
        runner.main(["--runs", "1"])


def test_a_second_run_never_overwrites_the_first(tmp_path):
    out = tmp_path / "phase6_STUB"
    out.mkdir()
    (out / "results.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="already exists"):
        runner.main(["--dry-run", "--out", str(out)])


# --------------------------------------------------------------------------
# the policy-fitting tool
# --------------------------------------------------------------------------
def test_the_fitter_replays_the_assembler_over_a_real_run(tmp_path):
    coupled = tmp_path / "phase6_STUB_coupled"
    runner.main(["--dry-run", "--coupled", "--runs", "1", "--out", str(coupled)])
    found = fit_chunk_policy.report(coupled / "results.json", AssemblyPolicy())
    body = found["runs"]["cold"]
    assert body["as_run"]["under_10_words"] > 0
    assert body["assembler_would_produce"]["under_10_words"] == 0
    assert body["assembler_would_produce"]["chunks"] < body["as_run"]["chunks"]
    assert body["assembler_would_produce"]["text_integrity"].startswith("exact")


def test_the_fitter_reads_phase_5_shaped_files_too(tmp_path):
    """Phase 5 wrote one entry per sentence and no `raw_sentences` key."""
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"label": "phase 5", "runs": [{"run": "warm",
        "chunks": [{"text": "One two three four five.", "words": 5,
                    "generate_seconds": 0.6},
                   {"text": "Six seven eight.", "words": 3,
                    "generate_seconds": 0.4},
                   {"text": " ".join(f"w{i}" for i in range(30)) + ".",
                    "words": 31, "generate_seconds": 2.6}]}]}),
        encoding="utf-8")
    body = fit_chunk_policy.report(path, AssemblyPolicy())["runs"]["warm"]
    assert body["sentences"]["chunks"] == 3
    assert body["assembler_would_produce"]["chunks"] < 3
    assert body["generate_fit"]["points"] == 3
    assert body["generate_fit"]["slope_seconds_per_word"] > 0


def test_the_fitter_says_so_rather_than_fitting_nothing(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"runs": [{"run": "warm", "chunks": [
        {"text": "Only one.", "words": 2, "generate_seconds": 0.3}]}]}),
        encoding="utf-8")
    body = fit_chunk_policy.report(path, AssemblyPolicy())["runs"]["warm"]
    assert "too few" in body["generate_fit"]["note"]


def test_the_fitter_refuses_a_missing_file():
    with pytest.raises(SystemExit, match="no results file"):
        fit_chunk_policy.main(["/nowhere/results.json"])
