"""The Chatterbox first-audio instrument, tested without a GPU or an endpoint."""
from __future__ import annotations

import base64
import json
import pathlib
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments import chatterbox_probe as probe
from tools import extract_chunks

RATE = 24000


def _pcm(seconds: float) -> bytes:
    return b"\x00\x01" * int(RATE * seconds)


class _Handler(BaseHTTPRequestHandler):
    pcm = _pcm(2.0)
    mode = "json"

    def log_message(self, *args):        # keep the test output clean
        pass

    def do_POST(self):
        json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.mode == "stream":
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("X-Sample-Rate", str(RATE))
            self.end_headers()
            step = len(self.pcm) // 4
            for index in range(0, len(self.pcm), step):
                self.wfile.write(self.pcm[index:index + step])
                self.wfile.flush()
            return
        payload = json.dumps({
            "pcm_base64": base64.b64encode(self.pcm).decode(),
            "sample_rate": RATE, "device": "test"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/synthesise"
    httpd.shutdown()


def test_json_base64_collapses_first_audio_onto_completion(server):
    """The finding the benchmark exists to produce, measured not asserted.

    Base64 inside JSON cannot be decoded until the closing brace lands, so a
    listener cannot hear anything before the whole episode chunk has arrived.
    """
    _Handler.mode = "json"
    result = probe.measure_http(server, "a sentence worth speaking")
    assert result.first_audio_bytes == result.complete
    assert result.playable == result.complete
    assert set(result.collapsed) == {"first_audio_bytes", "playable"}
    assert result.detail["contract"] == "json_base64"


def test_a_streaming_contract_separates_all_four_marks(server):
    """And the instrument has to be able to see the difference, or it is useless."""
    _Handler.mode = "stream"
    result = probe.measure_http(server, "a sentence worth speaking")
    assert result.collapsed == []
    assert result.playable is not None
    assert result.playable <= result.complete
    assert result.detail["contract"] == "streaming_pcm"
    _Handler.mode = "json"


def test_playable_needs_more_than_one_byte():
    """'Playable' is a buffer, not a first byte; a click is not playback."""
    assert probe.playable_bytes(24000, 0.2) == 9600
    assert probe.playable_bytes(24000, 0.0) == 0


def test_audio_duration_is_derived_from_the_response_rate(server):
    _Handler.mode = "json"
    result = probe.measure_http(server, "x")
    assert result.sample_rate == RATE
    assert result.audio_seconds == pytest.approx(2.0, abs=0.01)


def _openings_file(tmp_path, rows, name="warm"):
    """A real openings file, written by the generator the parser must match."""
    from experiments.report import openings_by_arm_markdown
    from experiments.spec import Arm, ExperimentSpec

    arms = sorted({r["arm"] for r in rows})
    spec = ExperimentSpec(name=name, trials=1, minutes=3,
                          queries=sorted({r["query"] for r in rows}),
                          arms=[Arm(a, search="none", tts="none") for a in arms])
    trials = [{"arm": r["arm"], "query": r["query"], "index": r.get("index", 1),
               "ok": True, "first_chunk_text": r["text"],
               "metrics": {"truncated": r.get("truncated", False)}} for r in rows]
    (tmp_path / "openings_by_arm.md").write_text(
        openings_by_arm_markdown(spec, trials), encoding="utf-8")
    return tmp_path


def test_the_parser_matches_the_generator_that_writes_the_file(tmp_path):
    """The bug: the parser only read fenced blocks, and the file has none.

    It found the markdown, parsed nothing, and said "no chunks extracted",
    which reads like the run had no openings. Pinned by round-tripping through
    the real generator rather than through a format written from memory.
    """
    text = " ".join(f"w{i}" for i in range(30)) + " and it ends here."
    _openings_file(tmp_path, [{"arm": "A-control", "query": "why tides", "text": text}])

    chunks, _ = extract_chunks.extract(tmp_path, words=25)
    assert len(chunks) == 1
    assert chunks[0]["text"] == text
    assert chunks[0]["source"].startswith("openings_by_arm.md:A-control:why tides")


def test_a_chunk_that_spans_lines_is_reconstructed_whole(tmp_path):
    """The reported failure: "says 40 words, the parsed text has 23".

    `first_chunk_ready` strips only the ends, so a newline the model wrote
    inside its opening survives into the recorded chunk and the report writes
    it inline across several markdown lines. Taking the first line alone gave
    a short fragment that still looked like prose.
    """
    text = ("Monza gave us one for the history books this weekend, and if you "
            "only saw the final classification you missed the whole story of "
            "it.\n\nThe pass that decided it came on lap forty six.")
    assert "\n" in text and len(text.split()) == 35
    _openings_file(tmp_path, [{"arm": "A-control", "query": "f1", "text": text}])

    chunks, _ = extract_chunks.extract(tmp_path, words=25)
    assert len(chunks) == 1
    assert chunks[0]["words"] == 35
    # Rejoined as the model wrote it: this is the string a voice would receive.
    assert chunks[0]["text"] == text
    assert chunks[0]["lines"] > 1


def test_a_multi_line_chunk_does_not_swallow_the_next_row(tmp_path):
    """Reconstruction must stop at the next structural marker, not run on."""
    spanning = ("First part of it here across a break.\n\nSecond part of it "
                "continues after the break and ends properly.")
    plain = " ".join(f"w{i}" for i in range(30)) + " end."
    _openings_file(tmp_path, [
        {"arm": "A", "query": "q", "text": spanning, "index": 1},
        {"arm": "A", "query": "q", "text": plain, "index": 2}])

    chunks, _ = extract_chunks.extract(tmp_path, words=10)
    assert len(chunks) == 2
    assert chunks[0]["text"] == spanning
    assert chunks[1]["text"] == plain


def test_a_multi_line_chunk_at_the_end_of_an_arm_stops_at_the_heading(tmp_path):
    spanning = "Across a break here.\n\nAnd the rest of it lands here."
    plain = " ".join(f"w{i}" for i in range(30)) + " end."
    _openings_file(tmp_path, [
        {"arm": "A", "query": "q", "text": spanning, "index": 1},
        {"arm": "B", "query": "q", "text": plain, "index": 1}])

    chunks, _ = extract_chunks.extract(tmp_path, words=5)
    by_arm = {c["source"].split(":")[1]: c for c in chunks}
    assert by_arm["A"]["text"] == spanning
    assert "## B" not in by_arm["A"]["text"]


def test_truncated_marks_the_response_not_the_chunk(tmp_path, capsys):
    """The question behind 'truncated 100 of 109'.

    `truncated` is `stop_reason == "max_tokens"` on the final message - read
    after the stream ended, about the whole response. The first chunk was
    emitted long before. The audit has to say so from the text, not assert it.
    """
    text = " ".join(f"w{i}" for i in range(30)) + " and it ends here."
    _openings_file(tmp_path, [
        {"arm": "A-control", "query": "q", "text": text, "truncated": True}])

    assert extract_chunks.audit(tmp_path, words=25) == 0
    out = capsys.readouterr().out
    assert "truncated response, complete chunk 1" in out
    assert "truncated response, CUT chunk      0" in out
    assert "chunks are intact and are valid benchmark input" in out


def test_the_audit_names_a_cut_chunk_and_its_cause(tmp_path, capsys):
    """A cut chunk with a truncated response is explained by the fallback."""
    cut = " ".join(f"w{i}" for i in range(30)) + " and then it stops mid"
    _openings_file(tmp_path, [
        {"arm": "E", "query": "q", "text": cut, "truncated": True}])

    assert extract_chunks.audit(tmp_path, words=25) == 0
    out = capsys.readouterr().out
    assert "really were cut mid-sentence" in out
    assert "first_chunk = buffer.strip()" in out
    assert "excluded from the corpus automatically" in out


def test_the_audit_fails_on_a_cut_chunk_with_no_truncation_to_explain_it(
        tmp_path, capsys):
    """Otherwise the reassurance is worthless: that would be a parse fault."""
    cut = " ".join(f"w{i}" for i in range(30)) + " and then it stops mid"
    _openings_file(tmp_path, [{"arm": "A", "query": "q", "text": cut}])

    assert extract_chunks.audit(tmp_path, words=25) == 1
    assert "parse fault, not a data case" in capsys.readouterr().out


def test_extraction_refuses_a_chunk_that_is_not_sentence_ended(tmp_path):
    """The chunk rule cuts only at . ! ? - anything else is not its output."""
    cut = " ".join(f"w{i}" for i in range(30)) + " and then it stops mid"
    _openings_file(tmp_path, [{"arm": "E", "query": "q", "text": cut}])

    with pytest.raises(SystemExit) as caught:
        extract_chunks.extract(tmp_path, words=25)
    assert "does not end at a sentence boundary" in str(caught.value)


def test_a_cut_chunk_is_excluded_not_benchmarked(tmp_path, capsys):
    """harness.py records the raw buffer when the rule is never satisfied:

        if first_chunk is None:
            first_chunk = buffer.strip()

    So a truncated response can leave a mid-sentence fragment in the openings
    file. It is real pipeline behaviour and it is not what FAM would speak, so
    it must not reach the voice - and it must not silently vanish either.
    """
    good = " ".join(f"w{i}" for i in range(30)) + " and it ends here."
    cut = " ".join(f"w{i}" for i in range(30)) + " and then it just"
    _openings_file(tmp_path, [
        {"arm": "A-control", "query": "q", "text": good, "truncated": True},
        {"arm": "E-max-tokens-96", "query": "q", "text": cut, "truncated": True}])

    chunks, excluded = extract_chunks.extract(tmp_path, words=25)
    assert [c["text"] for c in chunks] == [good]
    assert all(c["ends_complete"] for c in chunks)
    assert [(e["reason"], e["arm"]) for e in excluded] == [("cut", "E-max-tokens-96")]
    assert "cut mid-sentence" in capsys.readouterr().out


def test_a_complete_chunk_from_a_capped_response_is_kept(tmp_path):
    """The 106: the response hit max_tokens long after the chunk was emitted."""
    good = " ".join(f"w{i}" for i in range(30)) + " and it ends here."
    _openings_file(tmp_path, [
        {"arm": "A-control", "query": "q", "text": good, "truncated": True}])

    chunks, excluded = extract_chunks.extract(tmp_path, words=25)
    assert len(chunks) == 1
    assert chunks[0]["truncated_response"] is True
    assert excluded == []


def test_a_cut_chunk_without_a_truncated_response_is_still_fatal(tmp_path):
    """The fallback only fires on a stream that ended early.

    A cut chunk from an untruncated response has no explanation, so it means
    the parse is wrong - and excluding it would hide that.
    """
    cut = " ".join(f"w{i}" for i in range(30)) + " and then it just"
    _openings_file(tmp_path, [{"arm": "A", "query": "q", "text": cut}])

    with pytest.raises(SystemExit) as caught:
        extract_chunks.extract(tmp_path, words=25)
    assert "does not end at a sentence boundary" in str(caught.value)
    assert "was not truncated" in str(caught.value)


def test_every_exclusion_is_recorded_in_the_corpus_file(tmp_path):
    """So the count is auditable without re-running the extractor."""
    good = " ".join(f"w{i}" for i in range(30)) + " and it ends here."
    cut = " ".join(f"w{i}" for i in range(30)) + " and then it just"
    short = "Only a few words here."
    _openings_file(tmp_path, [
        {"arm": "A", "query": "q", "text": good, "index": 1},
        {"arm": "E", "query": "q", "text": cut, "index": 2, "truncated": True},
        {"arm": "E", "query": "q", "text": short, "index": 3, "truncated": True}])

    _, excluded = extract_chunks.extract(tmp_path, words=25)
    reasons = sorted(e["reason"] for e in excluded)
    assert reasons == ["cut", "short"]
    assert all(e["truncated"] for e in excluded)


def test_sentence_end_detection_allows_closing_punctuation():
    assert extract_chunks.ends_complete("It ended here.")
    assert extract_chunks.ends_complete('He said "it ended here."')
    assert extract_chunks.ends_complete("Did it end here?")
    assert not extract_chunks.ends_complete("It did not end here")
    assert not extract_chunks.ends_complete("It ended with a comma,")


def test_short_chunks_are_named_with_their_provenance(tmp_path, capsys):
    """They must never be counted away without an explanation."""
    long_enough = " ".join(f"w{i}" for i in range(30)) + " end."
    _openings_file(tmp_path, [
        {"arm": "A-control", "query": "q", "text": long_enough, "index": 1},
        {"arm": "E-max-tokens-96", "query": "q", "index": 1,
         "text": "Cut off early here.", "truncated": True},
        {"arm": "B-thinking-off", "query": "q", "index": 1,
         "text": "Short for no reason."}])

    extract_chunks.extract(tmp_path, words=25)
    out = capsys.readouterr().out
    assert "2 recorded chunk(s) excluded from the corpus" in out
    assert "E-max-tokens-96/q/1  - below the 25-word rule" in out
    assert "B-thinking-off/q/1  - below the 25-word rule - UNEXPLAINED" in out
    assert "1 exclusion(s) are UNEXPLAINED" in out


def test_the_recorded_chunk_is_taken_verbatim_not_re_chunked(tmp_path):
    """It is already the output of the chunk rule; applying it again truncates.

    This text has a sentence ending after 26 words and continues. Re-running
    the 25-word rule would cut it there and silently shorten the thing being
    timed.
    """
    text = (" ".join(f"w{i}" for i in range(25)) + " done. "
            + " ".join(f"x{i}" for i in range(20)) + " finished.")
    _openings_file(tmp_path, [{"arm": "A", "query": "q", "text": text}])

    chunks, _ = extract_chunks.extract(tmp_path, words=25)
    assert chunks[0]["text"] == text
    assert chunks[0]["text"].endswith("finished.")


def test_no_chunk_markers_never_reach_the_corpus(tmp_path):
    """'(no chunk)' is a marker for a trial that produced nothing."""
    good = " ".join(f"w{i}" for i in range(30)) + " end."
    _openings_file(tmp_path, [
        {"arm": "A", "query": "q", "text": good, "index": 1},
        {"arm": "A", "query": "q", "text": "(no chunk)", "index": 2}])
    chunks, _ = extract_chunks.extract(tmp_path, words=25)
    assert len(chunks) == 1
    assert "(no chunk)" not in chunks[0]["text"]


def test_a_word_count_mismatch_stops_rather_than_writing_bad_chunks(tmp_path):
    """The report prints the count; a disagreement means the line parsed wrong."""
    text = " ".join(f"w{i}" for i in range(30)) + " end."
    _openings_file(tmp_path, [{"arm": "A", "query": "q", "text": text}])
    path = tmp_path / "openings_by_arm.md"
    path.write_text(path.read_text().replace("(31w)", "(99w)"), encoding="utf-8")

    with pytest.raises(SystemExit) as caught:
        extract_chunks.extract(tmp_path, words=25)
    assert "parse mismatch" in str(caught.value)


def test_a_missing_openings_file_names_what_it_did_find(tmp_path):
    """report.md is markdown too; 'no markdown' was the wrong complaint."""
    (tmp_path / "report.md").write_text("# a report\n", encoding="utf-8")
    with pytest.raises(SystemExit) as caught:
        extract_chunks.extract(tmp_path, words=25)
    message = str(caught.value)
    assert "no openings file" in message and "report.md" in message


def test_identical_openings_are_not_timed_twice(tmp_path):
    same = " ".join(f"w{i}" for i in range(30)) + " end."
    _openings_file(tmp_path, [{"arm": "A", "query": "q", "text": same},
                              {"arm": "B", "query": "q", "text": same}])
    chunks, _ = extract_chunks.extract(tmp_path, words=25)
    assert len(chunks) == 1
    assert chunks[0]["also_from"] == ["B/1"]


def test_buckets_split_a_real_corpus_three_ways(tmp_path):
    """Fixed boundaries put every real chunk in one bucket.

    The chunk rule guarantees >= 25 words, so 'short = under 15' can never be
    populated and the length question would be answered over two buckets.
    """
    rows = [{"arm": "A", "query": f"q{n}", "index": n,
             "text": " ".join(f"w{i}" for i in range(n)) + " end."}
            for n in range(25, 55)]
    _openings_file(tmp_path, rows)
    chunks, _ = extract_chunks.extract(tmp_path, words=25)
    ranges = extract_chunks.assign_buckets(chunks)

    assert set(ranges) == {"short", "medium", "long"}
    for name in extract_chunks.BUCKET_NAMES:
        assert [c for c in chunks if c["bucket"] == name]
    assert ranges["short"][1] < ranges["long"][0]


def test_a_corpus_too_small_to_split_says_medium_rather_than_guessing():
    chunks = [{"words": 30}, {"words": 31}]
    assert extract_chunks.assign_buckets(chunks) == {"medium": (30, 31)}


def test_preflight_does_not_take_a_named_device_on_trust(monkeypatch, capsys):
    """It passed `--device mps` on a Linux box with no Metal.

    `resolve_device` honours an explicit name by design - that is the escape
    hatch for timing CPU deliberately. The preflight has to ask the machine.
    """
    import pathlib as _pathlib

    from experiments.adapters import chatterbox_impl
    from tools import chatterbox_first_audio as runner

    monkeypatch.setattr(chatterbox_impl, "available_devices",
                        lambda: {"cpu": True, "cuda": False, "mps": False})
    assert runner.preflight(_pathlib.Path("nope.json"), "mps") == 1
    assert "device 'mps' exists on this machine" in capsys.readouterr().out


def test_preflight_reports_a_missing_corpus_with_the_commands_to_fix_it(
        monkeypatch, capsys):
    import pathlib as _pathlib

    from tools import chatterbox_first_audio as runner

    runner.preflight(_pathlib.Path("definitely-absent.json"), None)
    out = capsys.readouterr().out
    assert "chunk corpus present" in out
    assert "tools/extract_chunks.py" in out


def test_preflight_catches_a_none_watermarker_before_the_download(monkeypatch, capsys):
    """The 4 GB failure.

    `perth/__init__.py` swallows an ImportError and sets
    PerthImplicitWatermarker to None. Chatterbox calls it inside
    from_pretrained, so the TypeError only appears after the weights have been
    fetched. Importing chatterbox.tts_turbo succeeds either way, so the
    preflight has to check the attribute.
    """
    import pathlib as _pathlib
    import sys as _sys
    import types as _types

    from tools import chatterbox_first_audio as runner

    stub = _types.ModuleType("perth")
    stub.PerthImplicitWatermarker = None
    monkeypatch.setitem(_sys.modules, "perth", stub)

    runner.preflight(_pathlib.Path("absent.json"), None)
    out = capsys.readouterr().out
    assert "FAIL  perth watermarker is loadable" in out
    assert "diagnose_chatterbox" in out

    stub.PerthImplicitWatermarker = object
    runner.preflight(_pathlib.Path("absent.json"), None)
    assert "ok    perth watermarker is loadable" in capsys.readouterr().out


def test_the_diagnosis_prescribes_by_the_error_it_found(capsys):
    """It must not prescribe a setuptools pin for a torch problem."""
    from tools.diagnose_chatterbox import _prescribe

    _prescribe(ImportError("No module named 'pkg_resources'"))
    assert 'setuptools<82' in capsys.readouterr().out

    _prescribe(ImportError("No module named 'torchaudio'"))
    out = capsys.readouterr().out
    assert "torch==2.6.0" in out and "setuptools" not in out

    _prescribe(ImportError("something else entirely"))
    out = capsys.readouterr().out
    assert "does not have a prescription for" in out
    assert "setuptools" not in out and "torch==" not in out


def test_the_pod_bundle_refuses_to_ship_a_corpus_with_a_cut_chunk(tmp_path, monkeypatch):
    """The pod must run the corpus the Mac ran, or the comparison is void."""
    from tools import pack_for_pod

    corpus = tmp_path / "first_chunks.json"
    corpus.write_text(json.dumps({
        "min_words": 25,
        "chunks": [{"text": "Cut off here", "words": 30, "bucket": "short",
                    "ends_complete": False}],
    }), encoding="utf-8")
    monkeypatch.setattr(pack_for_pod, "CORPUS", corpus)

    with pytest.raises(SystemExit) as caught:
        pack_for_pod.check_corpus()
    assert "do not end at a sentence boundary" in str(caught.value)


def test_the_pod_bundle_reports_a_digest_for_the_corpus(tmp_path, monkeypatch, capsys):
    """So the pod can prove it received the same file, not a similar one."""
    from tools import pack_for_pod

    corpus = tmp_path / "first_chunks.json"
    corpus.write_text(json.dumps({
        "min_words": 25, "excluded": [],
        "chunks": [{"text": "It ends here.", "words": 30, "bucket": "short",
                    "ends_complete": True}],
    }), encoding="utf-8")
    monkeypatch.setattr(pack_for_pod, "CORPUS", corpus)

    summary = pack_for_pod.check_corpus()
    assert summary["chunks"] == 1
    assert len(summary["sha256"]) == 16
    assert "CUT included 0" in capsys.readouterr().out


def test_a_missing_corpus_names_the_command_that_builds_it(tmp_path, monkeypatch):
    from tools import pack_for_pod

    monkeypatch.setattr(pack_for_pod, "CORPUS", tmp_path / "absent.json")
    with pytest.raises(SystemExit) as caught:
        pack_for_pod.check_corpus()
    assert "tools/extract_chunks.py" in str(caught.value)
