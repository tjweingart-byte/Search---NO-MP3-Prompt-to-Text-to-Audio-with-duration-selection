"""Claude and Chatterbox overlapping, proved on a laptop before a card is rented.

The previous combined run detected its first speakable chunk at ~2.4s and then
waited for `claude_complete` before synthesising anything. Every test here
exists so that shape cannot pass for a concurrent one again.
"""
from __future__ import annotations

import asyncio
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments import concurrent_pipeline as cp
from experiments.pipeline_probe import EventLog

SENTENCES = [
    "The oldest working clock in Europe has no face.",
    "It was built to ring, not to be read.",
    "Salisbury Cathedral has kept it turning since about thirteen eighty-six.",
    "Nobody there needed to know the minute.",
    "They needed to know when to pray.",
]


async def _stream(sentences=SENTENCES, delay=0.02):
    """A model writing at a steady pace, one sentence at a time."""
    for sentence in sentences:
        await asyncio.sleep(delay)
        yield sentence


def _synth(seconds=0.05, rate=24000, audio_per_word=0.4):
    """A voice that takes time, and whose audio is longer than its wall-clock."""
    async def synth(text: str):
        await asyncio.sleep(seconds)
        return [0] * int(rate * audio_per_word * len(text.split())), rate
    return synth


def _run(stream=None, synth=None, **kwargs) -> cp.PipelineRun:
    log = EventLog()
    log.mark("request_start")
    log.mark("claude_start")
    return asyncio.run(cp.run_concurrent(stream or _stream(),
                                         synth or _synth(), log, **kwargs))


# --------------------------------------------------------------------------
# the thing the experiment exists to prove
# --------------------------------------------------------------------------
def test_tts_starts_while_the_model_is_still_writing():
    run = _run()
    first_start = run.log.at("first_tts_start")
    claude_done = run.log.at("claude_complete")
    assert first_start < claude_done, (
        f"first_tts_start {first_start:.3f}s must precede claude_complete "
        f"{claude_done:.3f}s")
    assert cp.concurrency_problems(run) == []


def test_a_sequential_pipeline_fails_the_assertion():
    """The guard has teeth. Collect everything, then speak - the shape the
    previous combined run had - and this must be reported as a failure."""
    async def sequential() -> cp.PipelineRun:
        log = EventLog()
        log.mark("request_start")
        log.mark("claude_start")
        collected = [s async for s in _stream()]
        log.mark("claude_complete")

        async def replay():
            for sentence in collected:
                yield sentence

        run = await cp.run_concurrent(replay(), _synth(), log)
        # The replay marks its own claude_complete after the fact; the real one
        # is the earlier mark, which `at()` returns.
        return run

    run = asyncio.run(sequential())
    problems = cp.concurrency_problems(run)
    assert any("sequential pipeline" in p for p in problems), problems
    with pytest.raises(AssertionError, match="sequential"):
        cp.assert_concurrent(run)


def test_the_first_synthesis_is_the_first_emitted_chunk():
    run = _run()
    assert run.chunks[0].text == SENTENCES[0]
    assert run.first_emitted == SENTENCES[0]
    assert run.chunks[0].text != run.full_script


def test_a_run_that_speaks_the_whole_script_first_is_rejected():
    """Concatenating and then synthesising is the failure mode this rules out."""
    run = _run()
    run.chunks[0].text = run.full_script
    run.first_emitted = run.full_script
    assert any("whole final script" in p for p in cp.concurrency_problems(run))


def test_chunks_are_synthesised_in_order():
    run = _run()
    assert [c.index for c in run.chunks] == list(range(len(SENTENCES)))
    assert [c.text for c in run.chunks] == SENTENCES
    starts = [c.tts_start for c in run.chunks]
    assert starts == sorted(starts)


def test_every_chunk_carries_its_own_timings():
    run = _run()
    for chunk in run.chunks:
        assert chunk.tts_complete > chunk.tts_start
        assert chunk.audio_seconds > 0
        assert chunk.realtime_factor > 1  # this stub voice is faster than real time
        assert chunk.waited_in_queue >= 0


def test_the_marks_are_ordered_the_way_the_pipeline_runs():
    log = _run().log
    order = ["request_start", "claude_start", "chunk_ready:00",
             "first_speakable_chunk_ready", "first_tts_start",
             "first_tts_complete", "first_playable_audio", "claude_complete",
             "final_audio_complete"]
    times = [log.at(name) for name in order]
    assert None not in times, dict(zip(order, times))
    assert times == sorted(times), dict(zip(order, times))


# --------------------------------------------------------------------------
# queue and playback
# --------------------------------------------------------------------------
def test_queue_depth_is_sampled_on_both_sides_and_never_negative():
    run = _run()
    assert [row["event"] for row in run.queue_depth].count("enqueue") == len(SENTENCES)
    assert all(row["depth"] >= 0 for row in run.queue_depth)
    assert run.peak_queue_depth >= 1


def test_a_full_queue_throttles_the_model_and_the_delay_is_measured():
    """Production bounds the queue at 4. When TTS is the slow half the model
    gets backpressured, and that has to be visible rather than folded into
    claude_complete."""
    run = _run(stream=_stream(SENTENCES * 3, delay=0.001),
               synth=_synth(seconds=0.05), queue_depth=2)
    assert run.peak_queue_depth <= 2
    assert run.backpressure_seconds > 0


def test_a_voice_slower_than_playback_is_reported_as_an_underrun():
    """Dead air is the number that decides whether this architecture works."""
    # A third of a second of work for four hundredths of a second of speech.
    slow = _run(synth=_synth(seconds=0.30, audio_per_word=0.005))
    report = cp.playback_analysis(slow)
    assert report["kept_ahead"] is False
    assert report["underruns"] and report["stall_seconds"] > 0
    assert report["max_stall_seconds"] > 0


def test_a_voice_faster_than_playback_keeps_ahead_with_headroom():
    fast = _run(synth=_synth(seconds=0.01, audio_per_word=0.4))
    report = cp.playback_analysis(fast)
    assert report["kept_ahead"] is True
    assert report["underruns"] == []
    assert report["headroom_seconds"] > 0


def test_the_summary_reports_only_stages_that_were_marked():
    run = _run()
    summary = cp.summarise(run)
    assert summary["chunks"] == len(SENTENCES)
    assert summary["overlap_seconds"] > 0
    assert summary["search_to_first_listen"] < summary["search_to_complete_audio"]
    # Exa was never run in this stub, so it stays absent rather than becoming 0.
    assert summary["exa_latency"] is None


def test_audio_is_kept_in_order_for_concatenation():
    run = _run()
    assert len(run.samples) == len(SENTENCES)
    assert [len(s) for s in run.samples] == [
        int(24000 * 0.4 * len(s.split())) for s in SENTENCES]


# --------------------------------------------------------------------------
# the runner's own wiring
# --------------------------------------------------------------------------
from tools import streaming_4090_experiment as runner            # noqa: E402


def test_the_whole_runner_path_runs_on_stubs_and_overlaps(tmp_path):
    """The end-to-end proof that does not need a card: real code path, real
    chunk files, real analysis, stubbed stages."""
    out = tmp_path / "streaming_STUB"
    assert runner.main(["--dry-run", "--runs", "1", "--out", str(out)]) == 0

    results = __import__("json").loads((out / "results.json").read_text())
    assert results["stub"] is True
    assert "NOT A RESULT" in results["label"]
    record = results["runs"][0]
    assert record["concurrent"] is True and record["concurrency_problems"] == []
    assert record["summary"]["overlap_seconds"] > 0

    report = (out / "ANALYSIS.md").read_text()
    assert "NOT A RESULT" in report
    assert "RTX 4090" not in report        # a stub must not wear the card's name
    assert (out / "audio" / "cold_episode.wav").exists()
    assert sorted(p.name for p in (out / "audio" / "cold").iterdir()) == [
        f"chunk_{i:02d}.wav" for i in range(5)]


def test_a_stub_may_not_write_where_a_result_would_live(tmp_path):
    """Announcing a demo mode is not enough if what it writes outlives the run."""
    with pytest.raises(SystemExit, match="STUB"):
        runner.main(["--dry-run", "--out", str(tmp_path / "streaming_4090_real")])
    with pytest.raises(SystemExit, match="stub directory"):
        runner.main(["--out", str(tmp_path / "streaming_STUB"),
                     "--reference", "x.wav"])


def test_a_real_run_needs_the_selected_reference_voice():
    with pytest.raises(SystemExit, match="reference"):
        runner.main(["--runs", "1"])


def test_a_second_run_never_overwrites_the_first(tmp_path):
    out = tmp_path / "streaming_STUB"
    out.mkdir()
    (out / "results.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="already exists"):
        runner.main(["--dry-run", "--out", str(out)])


def test_the_concatenated_episode_carries_the_production_gap(tmp_path):
    import wave

    out = tmp_path / "streaming_STUB"
    runner.main(["--dry-run", "--runs", "1", "--out", str(out)])
    with wave.open(str(out / "audio" / "cold_episode.wav")) as handle:
        total = handle.getnframes() / handle.getframerate()
    chunks = 0.0
    for index in range(5):
        with wave.open(str(out / "audio" / "cold" / f"chunk_{index:02d}.wav")) as h:
            chunks += h.getnframes() / h.getframerate()
    assert total == pytest.approx(chunks + 4 * cp.SENTENCE_GAP, abs=0.01)


def test_the_first_text_delta_is_marked_once(tmp_path):
    """claude_ttft comes from wrapping the client, so production's chunker is
    measured rather than a copy of it."""
    log = EventLog()

    class _Inner:
        @property
        def text_stream(self):
            async def gen():
                for delta in ("Hello", " there", "."):
                    yield delta
            return gen()

    async def drain():
        stream = runner._TimedStream(_Inner(), log)
        return [d async for d in stream.text_stream]

    assert asyncio.run(drain()) == ["Hello", " there", "."]
    marks = [e["name"] for e in log.events]
    assert marks.count("claude_ttft") == 1
