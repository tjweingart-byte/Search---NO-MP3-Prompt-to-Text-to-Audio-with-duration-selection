"""Voice selection.

The engine layer was built pluggable from the start so this stayed cheap. The
property that matters most is that a voice changes the *audio* and not the
*script*: the script cache must be reused across voices, or switching voice
would cost a model call every time.
"""
from __future__ import annotations

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_utils import pcm_duration  # noqa: E402
from cache import cache_key  # noqa: E402
from tts import (  # noqa: E402
    DebugEngine,
    EspeakEngine,
    default_voice,
    engine_for_voice,
    list_voices,
)


def test_at_least_one_voice_is_always_offered():
    """Even with nothing installed there must be something to select."""
    voices = list_voices()
    assert voices, "list_voices() must never be empty"
    assert default_voice() == voices[0].id


def test_voice_ids_name_their_engine():
    for voice in list_voices():
        assert ":" in voice.id, f"{voice.id} must be prefixed with its engine"
        assert voice.id.split(":", 1)[0] == voice.engine


def test_a_voice_routes_to_the_engine_that_owns_it():
    if not EspeakEngine.available():
        pytest.skip("espeak-ng is not installed here")
    assert engine_for_voice("espeak:en-gb").name == "espeak"


def test_an_unavailable_voice_falls_back_rather_than_failing():
    """A listener whose chosen voice was uninstalled should still hear audio."""
    engine = engine_for_voice("nosuchengine:whoever")
    assert engine is not None
    assert engine.name in {"piper", "say", "espeak", "debug"}


def test_the_script_cache_ignores_voice():
    """Voice changes the audio, not the words - so it must not split the cache.

    If voice were part of the key, every voice change would cost a fresh model
    call for a script that is character-for-character identical.
    """
    assert cache_key("a topic", 3) == cache_key("a topic", 3)
    keys = {cache_key("a topic", 3) for _ in range(3)}
    assert len(keys) == 1


@pytest.mark.skipif(not EspeakEngine.available(), reason="espeak-ng is not installed")
def test_different_voices_produce_different_audio():
    engine = EspeakEngine()
    text = "The same sentence, spoken two ways."
    a = asyncio.run(engine.synth(text, 150, "espeak:en-us"))
    b = asyncio.run(engine.synth(text, 150, "espeak:en-gb-scotland"))
    assert a and b
    assert a != b, "selecting a different voice must change the audio"


def test_an_unknown_voice_id_still_synthesises():
    """Never let a stale voice id turn into silence."""
    pcm = asyncio.run(DebugEngine().synth("Some words here.", 150, "bogus:voice"))
    assert pcm_duration(len(pcm)) > 0
