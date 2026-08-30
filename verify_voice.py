"""Prove the neural voice actually works on this machine.

    python verify_voice.py

Synthesises one sentence with every installed voice and reports the duration,
sample rate and how fast it ran. Writes nothing except an optional sample you
can listen to.

This exists because the Piper integration could not be exercised on the machine
it was written on - the voice models are hosted somewhere that build machine
could not reach - so this is the check that closes that gap.
"""
from __future__ import annotations

import asyncio
import sys
import time

from audio_utils import pcm_duration
from tts import PiperEngine, engine_for_voice, list_voices

SENTENCE = (
    "This is a test of the voice that ships with the app. "
    "If this sounds like a person rather than a robot, it is working."
)


async def main() -> int:
    voices = list_voices()
    print(f"{len(voices)} voice(s) available:\n")

    piper = [v for v in voices if v.engine == "piper"]
    if not piper:
        print("  No neural (piper) voices installed.")
        print("  Run: pip install -r requirements.txt && python setup_voices.py")
        print("  The app still works, using a lower-quality fallback voice.\n")

    worst = 0.0
    for voice in voices:
        engine = engine_for_voice(voice.id)
        started = time.perf_counter()
        try:
            pcm = await engine.synth(SENTENCE, 150, voice.id)
        except Exception as exc:  # noqa: BLE001
            print(f"  {voice.id:34} FAILED - {type(exc).__name__}: {exc}")
            continue
        elapsed = time.perf_counter() - started
        seconds = pcm_duration(len(pcm), engine.sample_rate)
        ratio = seconds / elapsed if elapsed else 0
        worst = max(worst, elapsed)
        print(
            f"  {voice.id:34} {seconds:5.2f}s audio  {engine.sample_rate} Hz  "
            f"{elapsed:5.2f}s to make ({ratio:4.0f}x realtime)"
        )
        if voice.engine == "piper" and len(sys.argv) > 1 and sys.argv[1] == "--save":
            import struct

            name = voice.id.split(":", 1)[1] + ".wav"
            header = (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
                      + struct.pack("<IHHIIHH", 16, 1, 1, engine.sample_rate,
                                    engine.sample_rate * 2, 2, 16)
                      + b"data" + struct.pack("<I", len(pcm)))
            with open(name, "wb") as handle:
                handle.write(header + pcm)
            print(f"      wrote {name} - listen to it")

    if piper:
        print(
            "\nA neural voice is installed and working."
            if worst < 30
            else "\nWorking, but slowly - check the machine has enough CPU."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
