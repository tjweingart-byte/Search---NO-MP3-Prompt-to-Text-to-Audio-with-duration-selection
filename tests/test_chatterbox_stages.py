"""Phase 1 stage-split logic, tested without a GPU or the chatterbox package."""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

numpy = pytest.importorskip("numpy")

from experiments import chatterbox_stages as stages
from tools import chatterbox_stage_split as runner


def _timing(**kwargs):
    base = dict(text="a chunk", words=30, device="cuda", text_prep=0.01,
                t3=1.0, flow=0.5, hift=0.2, watermark=0.05, tokens=200,
                mel_frames=400, samples=48000, sample_rate=24000)
    base.update(kwargs)
    return stages.StageTiming(**base)


def test_stage_shares_sum_to_one():
    shares = _timing().shares()
    assert sum(shares.values()) == pytest.approx(1.0)
    assert shares["t3"] > shares["flow"] > shares["hift"] > shares["watermark"]


def test_per_token_cost_and_projection():
    timing = _timing(t3=2.0, tokens=200)
    assert timing.seconds_per_token == pytest.approx(0.01)
    projection = stages.first_chunk_projection(timing, chunk_tokens=25)
    assert projection["projected_t3_to_first_chunk"] == pytest.approx(0.25)
    assert projection["projected_chunk_tokens"] == 25


def test_a_projection_never_exceeds_the_tokens_that_exist():
    """Asking for 25 tokens from a 10-token utterance must not invent 15."""
    projection = stages.first_chunk_projection(_timing(tokens=10), chunk_tokens=25)
    assert projection["projected_chunk_tokens"] == 10


def test_no_projection_without_tokens():
    assert stages.first_chunk_projection(_timing(tokens=0), 25) == {}


def test_seam_ratio_finds_a_click_and_ignores_a_clean_join():
    clean = numpy.sin(numpy.linspace(0, 40 * numpy.pi, 4000)).astype("float32")
    assert max(stages.seam_ratios(clean, [2000])) < 5

    clicked = clean.copy()
    clicked[2000:] += 0.9          # a step discontinuity at the join
    assert max(stages.seam_ratios(clicked, [2000])) > 20


def test_seam_ratios_ignore_out_of_range_boundaries():
    wav = numpy.zeros(100, dtype="float32")
    assert stages.seam_ratios(wav, [0, 500, -3]) == []


def test_a_nonlinear_per_token_cost_invalidates_the_projection(capsys):
    """The projection assumes a flat per-token cost. If the data disagrees,
    the report has to say so rather than let the number stand."""
    rows = [{"bucket": "short", "text_words": 28, "total_seconds": 1.0,
             "stage_t3": 0.5, "stage_flow": 0.3, "stage_hift": 0.1,
             "stage_watermark": 0.1, "tokens": 100, "seconds_per_token": 0.005,
             "projected_t3_to_first_chunk": 0.125},
            {"bucket": "long", "text_words": 50, "total_seconds": 4.0,
             "stage_t3": 3.0, "stage_flow": 0.6, "stage_hift": 0.3,
             "stage_watermark": 0.1, "tokens": 200, "seconds_per_token": 0.015,
             "projected_t3_to_first_chunk": 0.375}]
    summary = runner.summarise(rows, [])
    assert summary["seconds_per_token_spread"]["max_over_min"] == pytest.approx(3.0)
    runner.report(summary)
    assert "must be discarded" in capsys.readouterr().out


def test_a_chunked_failure_is_reported_not_averaged(capsys):
    chunked = [{"mode": "delta_only", "ok": False, "error": "RuntimeError: shape",
                "worst_seam_ratio": None, "first_chunk_seconds": None,
                "total_seconds": None}]
    summary = runner.summarise([], chunked)
    assert summary["delta_only"]["failed"] == 1
    runner.report(summary)
    assert "FAILED" in capsys.readouterr().out


def test_picking_takes_from_every_bucket():
    corpus = [{"bucket": b, "text": f"{b} {i}"} for b in ("short", "medium", "long")
              for i in range(5)]
    picked = runner.pick(corpus, 2)
    assert len(picked) == 6
    assert {c["bucket"] for c in picked} == {"short", "medium", "long"}


def test_the_probe_keeps_the_generate_constants():
    """A stage split using different settings would time a different
    computation from the benchmark it is meant to explain.

    Both values are read from `tts_turbo.generate`: n_cfm_timesteps=2 and the
    6561 out-of-vocabulary threshold.
    """
    assert stages.N_CFM_TIMESTEPS == 2
    assert stages.OOV_THRESHOLD == 6561
