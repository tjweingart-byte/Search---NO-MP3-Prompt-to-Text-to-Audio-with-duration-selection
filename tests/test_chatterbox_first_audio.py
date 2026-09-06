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


def test_chunks_come_from_real_openings_through_the_production_rule(tmp_path):
    """Nothing in the corpus may be written for the benchmark."""
    opening = ("Boards used to remove founders in an afternoon. That is no "
               "longer true, and the reason is a share class.")
    (tmp_path / "openings_by_arm.md").write_text(
        f"## A-control\n\n```\n{opening}\n```\n", encoding="utf-8")

    chunks = extract_chunks.extract(tmp_path, words=15)
    assert len(chunks) == 1
    # The production rule keeps whole sentences, so a 15-word threshold runs
    # past the short first sentence to the next ending.
    assert chunks[0]["text"].endswith("share class.")
    assert chunks[0]["words"] >= 15
    assert chunks[0]["source"].endswith(":A-control")


def test_identical_openings_are_not_timed_twice(tmp_path):
    same = "One sentence that is quite long indeed and ends here properly now."
    (tmp_path / "a.md").write_text(f"## A\n\n```\n{same}\n```\n", encoding="utf-8")
    (tmp_path / "b.md").write_text(f"## B\n\n```\n{same}\n```\n", encoding="utf-8")
    assert len(extract_chunks.extract(tmp_path, words=5)) == 1


def test_an_empty_results_folder_says_so_rather_than_inventing_text(tmp_path):
    with pytest.raises(SystemExit) as caught:
        extract_chunks.extract(tmp_path, words=25)
    assert "preserve a run first" in str(caught.value)


def test_buckets_cover_every_length():
    assert extract_chunks.bucket_for(1) == "short"
    assert extract_chunks.bucket_for(20) == "medium"
    assert extract_chunks.bucket_for(45) == "long"
    assert extract_chunks.bucket_for(5000) == "very long"
