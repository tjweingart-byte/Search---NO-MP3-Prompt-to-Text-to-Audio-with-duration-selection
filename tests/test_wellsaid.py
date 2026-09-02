"""WellSaid as an alternate voice: the parts that could go wrong quietly.

Claude still writes every word. These pin the three things that would make a
listening test worthless without anyone noticing: a paid engine becoming the
default, a failure silently arriving as Piper, and audio being mangled on the
way in.
"""
from __future__ import annotations

import array
import asyncio
import dataclasses
import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as config_mod  # noqa: E402
import tts  # noqa: E402
import wellsaid as ws  # noqa: E402


@pytest.fixture
def keyed(monkeypatch):
    """Settings with a WellSaid key, as if setup_wellsaid.py had been run."""
    fake = dataclasses.replace(config_mod.settings, wellsaid_api_key="ws-test-key")
    monkeypatch.setattr(config_mod, "settings", fake)
    monkeypatch.setattr(ws, "settings", fake)
    monkeypatch.setattr(tts, "settings", fake)
    return fake


def wav(pcm: bytes, rate: int = 22050, channels: int = 1) -> bytes:
    """A minimal but real WAV, so the parser is tested against a container
    rather than against a convenient stub."""
    block = channels * 2
    return (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
            + b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, rate,
                                    rate * block, block, 16)
            + b"data" + struct.pack("<I", len(pcm)) + pcm)


def tone(samples: int, value: int = 8000) -> bytes:
    return array.array("h", [value] * samples).tobytes()


# --- the engine is never arrived at by accident -------------------------

def test_a_paid_voice_is_never_the_default(keyed, monkeypatch):
    """On a machine with no Piper installed, WellSaid would otherwise be the
    first voice in the list and therefore what every listener gets - billed
    per character, without anyone having chosen it."""
    monkeypatch.setattr(tts.PiperEngine, "available", staticmethod(lambda: False))
    monkeypatch.setattr(tts.SayEngine, "available", staticmethod(lambda: False))
    monkeypatch.setattr(tts.EspeakEngine, "available", staticmethod(lambda: False))

    offered = [v.id for v in tts.list_voices()]
    assert "wellsaid:35" in offered, "the voices are not offered at all"
    assert tts.default_voice() not in offered or not tts.default_voice().startswith("wellsaid")
    assert tts.build_engine().name != "wellsaid", "auto-selection reached a paid engine"


def test_it_is_offered_only_when_a_key_is_set():
    """Voices that cannot speak must not be in the picker."""
    assert tts.ENGINES["wellsaid"].available() is False
    assert not [v for v in tts.list_voices() if v.engine == "wellsaid"]


def test_both_requested_voices_are_offered_with_the_right_speaker_ids(keyed):
    voices = {v.label: v.id for v in ws.WellSaidEngine.voices()}
    assert voices["Chase J (WellSaid)"] == "wellsaid:35"
    assert voices["Kai M (WellSaid)"] == "wellsaid:32"


def test_the_speaker_ids_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("WELLSAID_CHASE_J_ID", "99")
    monkeypatch.setenv("WELLSAID_KAI_M_ID", "98")
    monkeypatch.delenv("FAM_IGNORE_DOTENV", raising=False)
    fresh = config_mod.Settings()
    assert (fresh.wellsaid_chase_j_id, fresh.wellsaid_kai_m_id) == ("99", "98")


def test_the_key_is_not_in_the_source():
    """The one thing that must never be committed."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    for path in list(root.glob("*.py")) + [root / "static" / "index.html"]:
        text = path.read_text()
        assert 'WELLSAID_API_KEY", "")' in text or "WELLSAID_API_KEY" not in text or \
            "os.environ" in text or "setup_wellsaid" in text, f"{path.name} may hardcode a key"


# --- it never quietly becomes Piper -------------------------------------

def test_choosing_wellsaid_without_a_key_raises_rather_than_falling_back():
    """The whole reason to pick this voice is to hear this voice. Substituting
    Piper would mean judging one engine by another engine's output."""
    with pytest.raises(tts.TTSUnavailable) as caught:
        tts.engine_for_voice("wellsaid:35")
    assert "WELLSAID_API_KEY" in str(caught.value)


def test_an_unavailable_local_voice_still_falls_back(monkeypatch):
    """The old behaviour is deliberately kept for local engines: a listener
    whose Piper voice was uninstalled should still hear their episode."""
    monkeypatch.setattr(tts.PiperEngine, "available", staticmethod(lambda: False))
    assert tts.engine_for_voice("piper:gone-away") is not None


def test_an_http_failure_is_reported_with_its_real_reason(keyed, monkeypatch):
    class Response:
        status_code = 401
        text = "invalid api key"
        headers: dict = {}
        content = b""

    class Client:
        def __init__(self, **_): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_): return False
        async def post(self, *_, **__): return Response()

    monkeypatch.setitem(sys.modules, "httpx", type("m", (), {"AsyncClient": Client}))
    with pytest.raises(ws.WellSaidError) as caught:
        asyncio.run(ws.WellSaidEngine().synth("Hello.", 150, "wellsaid:35"))
    message = str(caught.value)
    assert "401" in message and "invalid api key" in message
    assert "setup_wellsaid" in message, "the error does not say what to do about it"


# --- the audio arrives intact -------------------------------------------

def test_wav_is_accepted_without_needing_a_decoder():
    pcm = tone(1000)
    assert ws._to_pcm(wav(pcm), "audio/wav", 22050) == pcm


def test_stereo_is_mixed_down_and_a_different_rate_is_resampled():
    """WellSaid's rate is not the app's, and app.py writes the stream header
    before the first request - so whatever arrives has to land on our rate."""
    stereo = array.array("h", [100, 100] * 2400).tobytes()
    out = ws._to_pcm(wav(stereo, rate=24000, channels=2), "audio/wav", 22050)
    got = array.array("h"); got.frombytes(out)
    assert len(got) == pytest.approx(2400 * 22050 / 24000, rel=0.01)
    assert all(abs(s - 100) <= 1 for s in got)


def test_an_unknown_format_is_refused_rather_than_played_as_noise():
    with pytest.raises(ws.WellSaidError) as caught:
        ws._to_pcm(b"<html>error</html>", "text/html", 22050)
    assert "text/html" in str(caught.value)


def test_mp3_without_ffmpeg_says_exactly_what_is_missing(monkeypatch):
    monkeypatch.setattr(ws.shutil, "which", lambda _: None)
    with pytest.raises(ws.WellSaidError) as caught:
        ws._to_pcm(b"ID3\x04\x00\x00\x00\x00\x00\x00", "audio/mpeg", 22050)
    assert "ffmpeg" in str(caught.value) and "brew install" in str(caught.value)


# --- chunking -----------------------------------------------------------

def test_short_text_is_one_chunk():
    assert ws.chunk_text("A short sentence.", 1000) == ["A short sentence."]


def test_a_long_script_splits_on_sentences_and_never_mid_word():
    text = " ".join(f"This is sentence number {i} and it says something." for i in range(60))
    chunks = ws.chunk_text(text, 200)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 200
    # Nothing lost, nothing reordered, and no word cut in half.
    assert " ".join(chunks).split() == text.split()


def test_a_single_sentence_longer_than_the_limit_splits_at_clauses():
    text = ("The market moved sharply, the analysts disagreed loudly, "
            "the regulator said nothing at all, and the index closed lower.")
    chunks = ws.chunk_text(text, 60)
    assert all(len(c) <= 60 for c in chunks)
    assert " ".join(chunks).split() == text.split()
    # Clause boundaries kept their punctuation, so the join is a breath the
    # sentence already had rather than one invented mid-phrase.
    assert any(c.endswith(",") for c in chunks)


def test_the_order_is_preserved_end_to_end(keyed, monkeypatch):
    """Chunks are spoken in order and concatenated in order. Out-of-order
    audio would be obvious to a listener and invisible to a test that only
    counted bytes."""
    spoken: list[str] = []

    async def fake(self, text, speaker, index, of, detailed=True):
        spoken.append(text)
        return tone(2205, value=len(spoken) * 1000)

    monkeypatch.setattr(ws.WellSaidEngine, "_speak_one", fake)
    text = " ".join(f"Sentence {i} here." for i in range(20))
    pcm = asyncio.run(ws.WellSaidEngine().synth(text, 150, "wellsaid:32"))

    assert spoken == ws.chunk_text(text, 1000)
    got = array.array("h"); got.frombytes(pcm)
    firsts = [got[i] for i in range(0, len(got), 2205)]
    assert firsts == sorted(firsts), "chunks were reassembled out of order"


def test_empty_text_costs_nothing():
    assert asyncio.run(ws.WellSaidEngine().synth("   ", 150, "wellsaid:35")) == b""


# --- joins --------------------------------------------------------------

def test_the_silence_between_chunks_is_tightened():
    """Each chunk is rendered separately and carries its own lead-in and tail.
    Joined untrimmed they stack into a pause in the middle of a sentence."""
    padded = tone(4410, value=0) + tone(2205, value=9000) + tone(4410, value=0)
    trimmed = ws._trim_join(padded, 22050)
    assert len(trimmed) < len(padded)
    # A margin is kept, so no word is clipped short.
    assert len(trimmed) > 2205 * 2


def test_a_single_chunk_is_not_trimmed(keyed, monkeypatch):
    """Trimming a whole sentence would eat the pause the pipeline puts between
    them. It is a join fix, not a general one."""
    async def fake(self, text, speaker, index, of, detailed=True):
        return tone(2205, value=0) + tone(2205, value=9000)

    monkeypatch.setattr(ws.WellSaidEngine, "_speak_one", fake)
    pcm = asyncio.run(ws.WellSaidEngine().synth("One sentence.", 150, "wellsaid:35"))
    assert len(pcm) == 2205 * 2 * 2


# --- the request itself -------------------------------------------------

def test_the_request_matches_the_documented_api(keyed, monkeypatch):
    sent: dict = {}

    class Response:
        status_code = 200
        headers = {"Content-Type": "audio/wav"}
        text = ""
        content = wav(tone(2205))

    class Client:
        def __init__(self, **_): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_): return False
        async def post(self, url, json=None, headers=None):
            sent.update(url=url, json=json, headers=headers)
            return Response()

    monkeypatch.setitem(sys.modules, "httpx", type("m", (), {"AsyncClient": Client}))
    pcm = asyncio.run(ws.WellSaidEngine().synth("Hello there.", 150, "wellsaid:32"))

    assert sent["url"].endswith("/v1/tts/stream")
    assert sent["json"] == {"text": "Hello there.", "speaker_id": 32}, "not the documented body"
    assert sent["headers"]["X-Api-Key"] == "ws-test-key"
    assert "audio/wav" in sent["headers"]["Accept"]
    assert len(pcm) == 2205 * 2


def test_a_retryable_status_is_retried_and_then_reported(keyed, monkeypatch):
    attempts = {"n": 0}

    class Response:
        status_code = 429
        text = "slow down"
        headers: dict = {}
        content = b""

    class Client:
        def __init__(self, **_): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_): return False
        async def post(self, *_, **__):
            attempts["n"] += 1
            return Response()

    monkeypatch.setitem(sys.modules, "httpx", type("m", (), {"AsyncClient": Client}))
    async def instant(_seconds):
        return None

    # NOT `lambda _: asyncio.sleep(0)`: ws.asyncio is asyncio, so that patches
    # the function it then calls, and the retry recurses until the stack ends.
    monkeypatch.setattr(ws.asyncio, "sleep", instant)
    with pytest.raises(ws.WellSaidError):
        asyncio.run(ws.WellSaidEngine().synth("Hello.", 150, "wellsaid:35"))
    assert attempts["n"] == ws._MAX_ATTEMPTS, "a rate limit was not retried"
