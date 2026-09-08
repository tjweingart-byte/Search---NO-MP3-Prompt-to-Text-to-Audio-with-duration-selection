"""One clock across the request, and honest gaps where a stage cannot be seen."""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments import pipeline_probe as probe


def _log():
    log = probe.EventLog()
    log.events = [
        {"name": "request_start", "at": 0.0, "detail": {}},
        {"name": "exa_request_start", "at": 0.01, "detail": {}},
        {"name": "exa_complete", "at": 0.51, "detail": {}},
        {"name": "claude_request_start", "at": 0.53, "detail": {}},
        {"name": "claude_first_token", "at": 1.43, "detail": {}},
        {"name": "claude_complete", "at": 3.03, "detail": {}},
        {"name": "tts_start", "at": 3.05, "detail": {}},
        {"name": "first_playable_audio", "at": 5.85, "detail": {}},
        {"name": "audio_complete", "at": 5.86, "detail": {}},
    ]
    return log


def test_spans_come_from_marks_not_assumptions():
    log = _log()
    assert log.span("exa_request_start", "exa_complete") == pytest.approx(0.50)
    assert log.span("claude_request_start", "claude_first_token") == pytest.approx(0.90)


def test_a_missing_mark_gives_none_rather_than_a_guess():
    log = _log()
    assert log.span("request_start", "never_happened") is None
    assert log.at("never_happened") is None


def test_the_waterfall_excludes_unmeasured_rows_from_the_share():
    """A row that could not be measured must not silently absorb the remainder."""
    log = _log()
    rows = probe.waterfall(log, probe.PIPELINE_SEGMENTS
                           + [("imaginary", "nope", "also_nope")])
    by_label = {row["label"]: row for row in rows}
    assert by_label["imaginary"]["seconds"] is None
    assert by_label["imaginary"]["share"] is None
    shares = [r["share"] for r in rows if r["share"] is not None]
    assert sum(shares) == pytest.approx(1.0)


def test_the_headline_numbers_are_the_two_the_product_is_judged_on():
    values = probe.headlines(_log())
    assert values["search_to_first_listen"] == pytest.approx(5.85)
    assert values["search_to_complete_audio"] == pytest.approx(5.86)
    assert values["claude_ttft"] == pytest.approx(0.90)
    assert values["exa_latency"] == pytest.approx(0.50)


def test_unmeasurable_stages_are_declared_with_a_reason():
    """A plausible number in a waterfall is worse than a gap: the gap gets
    investigated and the number gets believed."""
    log = probe.EventLog()
    log.cannot("first_exa_result", "one blocking call, no intermediate point")
    assert "first_exa_result" in log.as_dict()["unavailable"]
    assert "blocking" in log.as_dict()["unavailable"]["first_exa_result"]


def test_gpu_memory_reports_absence_rather_than_zero(monkeypatch):
    import builtins

    real = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    out = probe.gpu_memory()
    assert "error" in out or out.get("available") is False
    assert "allocated_mb" not in out


def test_marks_are_monotonic_on_one_clock():
    log = probe.EventLog()
    first = log.mark("a")
    second = log.mark("b")
    assert 0 <= first <= second
