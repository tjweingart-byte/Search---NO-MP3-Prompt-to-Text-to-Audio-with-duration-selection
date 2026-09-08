"""Three stages that cannot stall each other, proved before a card is rented.

Phase 5's `claude_total` of 66.7s carried 64.3s of our own backpressure. Every
test here exists so a number like that cannot be reported as Claude's again.
"""
from __future__ import annotations

import asyncio
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments import decoupled_pipeline as dp
from experiments.pipeline_probe import EventLog
from experiments.speech_assembler import AssemblyPolicy

#: A script shaped like Phase 5's complaint: a few full sentences and several
#: fragments of two to twelve words.
SCRIPT = [
    "The oldest working clock in Europe has no face, and for six hundred years "
    "nobody thought that was strange.",
    "It was built to ring.",
    "Not to be read.",
    "Salisbury Cathedral has kept it turning since about thirteen eighty-six, "
    "through a civil war and two restorations.",
    "Nobody there needed the minute.",
    "They needed to know when to pray.",
    "So what changed?",
    "Clocks grew faces when somebody else started owning your hours, which is a "
    "sentence about factories more than it is about horology.",
    "The face is the invoice.",
    "It still is.",
]


async def _stream(script=SCRIPT, delay=0.01):
    for sentence in script:
        await asyncio.sleep(delay)
        yield sentence


def _synth(seconds=0.08, rate=24000, audio_per_word=0.4):
    async def synth(text: str):
        await asyncio.sleep(seconds)
        return [0] * int(rate * audio_per_word * len(text.split())), rate
    return synth


def _run(stream=None, synth=None, **kwargs) -> dp.DecoupledRun:
    log = EventLog()
    log.mark("request_start")
    log.mark("claude_start")
    return asyncio.run(dp.run_decoupled(stream or _stream(),
                                        synth or _synth(), log, **kwargs))


# --------------------------------------------------------------------------
# 1 - genuine overlap, still
# --------------------------------------------------------------------------
def test_tts_starts_while_the_model_is_still_writing():
    run = _run()
    assert run.log.at("first_tts_start") < run.log.at("claude_complete")
    assert dp.phase6_problems(run) == []


# --------------------------------------------------------------------------
# 4 - the central Phase 6 assertion, and proof it has teeth
# --------------------------------------------------------------------------
def test_a_slow_voice_does_not_stop_the_reader():
    """A voice far slower than the model, and a script long enough to saturate
    the TTS queue. Claude must still finish early and unblocked."""
    run = _run(stream=_stream(SCRIPT * 3, delay=0.005), synth=_synth(seconds=0.25))
    assert run.reader_blocked_seconds == 0.0
    assert dp.phase6_problems(run) == []
    evidence = dp.decoupling_evidence(run)
    assert evidence["exercised"] is True
    assert evidence["tts_outlived_claude_by"] > 0
    assert evidence["tts_queue_saturated"] is True
    assert run.assembler_blocked_seconds > 0      # downstream blocking is fine


def test_the_coupled_build_is_caught_by_the_assertion():
    """Phase 5's shape, built deliberately: the reader writes straight onto the
    TTS queue. The decoupling assertion must reject it, or it is decoration."""
    run = _run(stream=_stream(SCRIPT * 3, delay=0.005), synth=_synth(seconds=0.25),
               coupled=True)
    assert run.reader_blocked_seconds > dp.BLOCKED_EPSILON
    problems = dp.phase6_problems(run)
    assert any("reader was blocked" in p for p in problems), problems
    with pytest.raises(AssertionError, match="reader was blocked"):
        dp.assert_phase6(run)


def test_the_coupled_build_inflates_claude_the_way_phase_5_did():
    """The point of the fixture: the same script, the same voice, and a
    `claude_stream_seconds` that is mostly our own queue."""
    fast = _run(stream=_stream(SCRIPT * 3, delay=0.005), synth=_synth(seconds=0.25))
    slow = _run(stream=_stream(SCRIPT * 3, delay=0.005), synth=_synth(seconds=0.25),
                coupled=True)
    assert (dp.timing_report(slow)["claude_stream_seconds"]
            > 2 * dp.timing_report(fast)["claude_stream_seconds"])
    assert dp.timing_report(fast)["claude_reader_blocked_seconds"] == 0.0


def test_decoupling_is_reported_as_untested_when_tts_never_falls_behind():
    """A fast voice does not prove the architecture; saying so would overclaim."""
    run = _run(stream=_stream(delay=0.3), synth=_synth(seconds=0.001))
    evidence = dp.decoupling_evidence(run)
    assert evidence["exercised"] is False
    assert "not exercised" in evidence["verdict"]


# --------------------------------------------------------------------------
# 2, 3, 5, 6 - the payload and the text
# --------------------------------------------------------------------------
def test_the_first_payload_is_the_first_assembled_chunk():
    run = _run()
    assert run.chunks[0].chunk.index == 0
    assert run.chunks[0].chunk.text.startswith("The oldest working clock")
    assert run.chunks[0].chunk.text != " ".join(SCRIPT)


def test_no_text_is_lost_duplicated_or_reordered():
    run = _run()
    assert run.assembler_witness is None
    spoken = " ".join(s.chunk.text for s in run.chunks).split()
    assert spoken == " ".join(SCRIPT).split()


def test_every_chunk_is_synthesised_exactly_once_and_in_order():
    run = _run()
    indexes = [s.chunk.index for s in run.chunks]
    assert indexes == sorted(indexes) == list(range(len(indexes)))


# --------------------------------------------------------------------------
# the efficiency question
# --------------------------------------------------------------------------
def test_assembly_removes_the_pathological_tiny_calls():
    """The script has fragments of three and four words. None may be spoken
    alone once the opening is out."""
    run = _run()
    report = dp.chunk_report(run)
    assert report["raw_sentences"] == len(SCRIPT)
    assert report["tts_invocations"] < report["raw_sentences"]
    assert report["chunks_under_5_words"] == 0
    later = [s.chunk.words for s in run.chunks[1:-1]]
    assert all(w >= AssemblyPolicy().min_words for w in later), later


def test_the_last_chunk_may_be_short_and_that_is_correct():
    run = _run()
    assert run.chunks[-1].chunk.reason == "end of script"


# --------------------------------------------------------------------------
# the listener's timeline
# --------------------------------------------------------------------------
def test_a_voice_faster_than_playback_never_stalls_and_gains_headroom():
    run = _run(synth=_synth(seconds=0.05, audio_per_word=0.4))
    report = dp.playback_report(run)
    assert report["playback_stalls"] == 0
    assert report["total_stall_seconds"] == 0.0
    assert report["minimum_playback_headroom"] > 0
    assert report["maximum_playback_headroom"] >= report["median_playback_headroom"]
    assert report["headroom_at_final_tts"] > 0


def test_a_voice_slower_than_playback_reports_every_stall():
    run = _run(synth=_synth(seconds=0.5, audio_per_word=0.005))
    report = dp.playback_report(run)
    assert report["playback_stalls"] > 0
    assert report["total_stall_seconds"] > 0
    assert all(s["ready_at"] > s["needed_at"] for s in report["stalls"])
    assert report["minimum_playback_headroom"] <= 0


def test_headroom_is_recorded_at_claude_complete():
    run = _run()
    assert dp.playback_report(run)["headroom_at_claude_complete"] is not None


# --------------------------------------------------------------------------
# the five timing concepts must stay distinguishable
# --------------------------------------------------------------------------
def test_the_report_separates_claude_from_our_own_blocking():
    run = _run(stream=_stream(SCRIPT * 3, delay=0.005), synth=_synth(seconds=0.25))
    report = dp.timing_report(run)
    assert report["claude_reader_blocked_seconds"] == 0.0
    assert report["assembler_blocked_seconds"] > 0
    assert report["claude_local_processing_seconds"] >= 0
    # Claude finished long before the audio did, which is the whole point.
    assert report["claude_stream_seconds"] < report["search_to_complete_audio"]
    assert report["overlap_seconds"] > 0


def test_the_buffers_are_sampled_on_both_sides_and_stay_bounded():
    run = _run(stream=_stream(SCRIPT * 3, delay=0.005), synth=_synth(seconds=0.25))
    assert run.peak_tts_queue <= dp.QUEUE_DEPTH
    assert 0 < run.peak_script_buffer < dp.SCRIPT_BUFFER_CHARS
    assert {r["event"] for r in run.script_buffer_series} == {"append", "consume"}
    assert {r["event"] for r in run.tts_queue_series} == {"enqueue", "dequeue"}


def test_the_marks_are_ordered_the_way_the_pipeline_runs():
    log = _run().log
    order = ["request_start", "claude_start", "raw_sentence:00",
             "first_sentence_ready", "first_speech_chunk_ready",
             "first_tts_start", "first_tts_complete", "first_playable_audio",
             "playback_start", "claude_complete", "final_audio_complete"]
    times = [log.at(name) for name in order]
    assert None not in times, dict(zip(order, times))
    assert times == sorted(times), dict(zip(order, times))


def test_a_tiny_script_still_completes_and_flushes():
    """One sentence means one chunk, released at flush - so TTS cannot start
    before the stream ends, and demanding overlap there would be a false
    failure. The evidence block says overlap was not applicable."""
    run = _run(stream=_stream(["Just the one sentence, and it is short."]))
    assert len(run.chunks) == 1
    assert dp.phase6_problems(run) == []
    assert dp.decoupling_evidence(run)["overlap_applicable"] is False
    assert dp.playback_report(run)["playback_stalls"] == 0


def test_overlap_is_still_demanded_whenever_there_is_more_than_one_chunk():
    run = _run()
    assert dp.decoupling_evidence(run)["overlap_applicable"] is True
    run.log.events = [e for e in run.log.events if e["name"] != "claude_complete"]
    run.log.events.append({"name": "claude_complete", "at": 0.0, "detail": {}})
    assert any("sequential" in p for p in dp.phase6_problems(run))
