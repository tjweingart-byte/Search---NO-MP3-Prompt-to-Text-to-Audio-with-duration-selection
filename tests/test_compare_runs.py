"""Verifying and comparing two first-audio runs."""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tools import compare_runs


def _run(device, seconds, chunks=6, trials=3, truncate=None, **top):
    rows = []
    names = ("short", "medium", "long")
    sources = [(f"o.md:A:t{i}:1", names[i % 3], 30 + i) for i in range(chunks)]
    for trial in range(1, trials + 1):
        for source, bucket, words in sources:
            rows.append({
                "trial": trial, "bucket": bucket, "ok": True, "source": source,
                "words": words, "device": device, "model_seconds": seconds,
                "delivery_seconds": 0.0, "first_playable_seconds": seconds,
                "audio_seconds": words / 150 * 60,
            })
    if truncate:
        rows = rows[:truncate]
    payload = {"label": f"LOCAL {device.upper()}", "smoke_run": False,
               "simulated": False, "is_production_latency": False,
               "transport": "local",
               "summary": {"collapsed_marks": ["first_audio_bytes", "stream_begin"],
                           "cold_start": {"load_seconds": 1.0, "warmup_seconds": 2.0,
                                          "total_cold_seconds": 3.0,
                                          "excluded_from_trials": True}},
               "rows": rows}
    payload.update(top)
    return payload


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_a_complete_run_verifies_clean(tmp_path, capsys):
    path = _write(tmp_path, "a.json", _run("cuda", 0.5))
    assert compare_runs.verify("4090", compare_runs.load(path), path) == []
    out = capsys.readouterr().out
    assert "6 x 3 = 18 expected" in out
    assert "verdict         OK" in out


def test_a_truncated_run_is_caught(tmp_path):
    """The whole point of verifying first: a short file still looks finished."""
    path = _write(tmp_path, "a.json", _run("cuda", 0.5, truncate=11))
    problems = compare_runs.verify("4090", compare_runs.load(path), path)
    assert any("11 rows but" in p and "expected" in p for p in problems)


def test_a_smoke_run_is_refused_as_a_result(tmp_path):
    path = _write(tmp_path, "a.json", _run("cuda", 0.5, smoke_run=True))
    problems = compare_runs.verify("4090", compare_runs.load(path), path)
    assert any("smoke run" in p for p in problems)


def test_a_simulated_run_is_refused_as_a_measurement(tmp_path):
    path = _write(tmp_path, "a.json", _run("cuda", 0.5, simulated=True))
    problems = compare_runs.verify("4090", compare_runs.load(path), path)
    assert any("SIMULATED" in p for p in problems)


def test_rows_spanning_two_devices_are_flagged(tmp_path):
    payload = _run("cuda", 0.5)
    payload["rows"][0]["device"] = "cpu"
    path = _write(tmp_path, "a.json", payload)
    problems = compare_runs.verify("4090", compare_runs.load(path), path)
    assert any("more than one device" in p for p in problems)


def test_a_missing_bucket_is_flagged(tmp_path):
    payload = _run("cuda", 0.5)
    payload["rows"] = [r for r in payload["rows"] if r["bucket"] != "long"]
    path = _write(tmp_path, "a.json", payload)
    problems = compare_runs.verify("4090", compare_runs.load(path), path)
    assert any("'long' has no rows" in p for p in problems)


def test_a_file_that_is_not_a_run_is_refused(tmp_path):
    path = tmp_path / "x.json"
    path.write_text('{"hello": 1}', encoding="utf-8")
    with pytest.raises(SystemExit) as caught:
        compare_runs.load(path)
    assert "not a first-audio run" in str(caught.value)


def test_invalid_json_is_refused(tmp_path):
    path = tmp_path / "x.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit) as caught:
        compare_runs.load(path)
    assert "not valid JSON" in str(caught.value)


def test_percentiles_are_nearest_rank_not_a_library_guess():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert compare_runs.percentile(values, 0.50) == 5
    assert compare_runs.percentile(values, 0.90) == 9
    assert compare_runs.percentile(values, 0.95) == 10
    assert compare_runs.percentile([], 0.5) is None


def test_the_words_to_latency_slope_is_measured():
    rows = [{"words": w, "first_playable_seconds": 0.1 * w + 2.0}
            for w in range(25, 60)]
    beta, intercept, r = compare_runs.slope(rows)
    assert beta == pytest.approx(0.1, abs=1e-9)
    assert intercept == pytest.approx(2.0, abs=1e-9)
    assert r == pytest.approx(1.0, abs=1e-9)


def test_slope_declines_on_too_few_points():
    assert compare_runs.slope([{"words": 30, "first_playable_seconds": 1.0}]) is None


def test_impossible_timings_are_reported_as_anomalies(tmp_path, capsys):
    """Playable before the model finished cannot happen; say so, don't average it."""
    payload = _run("cuda", 0.5)
    payload["rows"][0]["first_playable_seconds"] = 0.1
    a = _write(tmp_path, "a.json", _run("mps", 10.0))
    b = _write(tmp_path, "b.json", payload)
    compare_runs.compare("MPS", compare_runs.load(a), "4090", compare_runs.load(b))
    assert "playable before the model finished" in capsys.readouterr().out


def test_nonzero_delivery_in_process_is_an_anomaly(tmp_path, capsys):
    payload = _run("cuda", 0.5)
    payload["rows"][0]["delivery_seconds"] = 0.4
    a = _write(tmp_path, "a.json", _run("mps", 10.0))
    b = _write(tmp_path, "b.json", payload)
    compare_runs.compare("MPS", compare_runs.load(a), "4090", compare_runs.load(b))
    assert "delivery > 1 ms" in capsys.readouterr().out


def test_a_clean_pair_reports_no_anomalies(tmp_path, capsys):
    a = _write(tmp_path, "a.json", _run("mps", 10.0))
    b = _write(tmp_path, "b.json", _run("cuda", 0.5))
    compare_runs.compare("MPS", compare_runs.load(a), "4090", compare_runs.load(b))
    out = capsys.readouterr().out
    assert "anomalies" in out and "  none" in out
    assert "20.0x" in out          # 10.0 / 0.5, computed from the files
