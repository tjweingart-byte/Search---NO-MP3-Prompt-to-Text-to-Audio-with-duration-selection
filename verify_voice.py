"""Prove the voice actually works on this machine.

    python verify_voice.py
    python verify_voice.py --save     # also write a .wav to listen to

Synthesises one sentence with every voice the server would offer and reports
the duration, sample rate and how fast it ran. Writes nothing except the
optional sample.

This exists because the production engine cannot be exercised on every machine
that builds FAM - Chatterbox needs a GPU and a reference recording, and the
build container has neither - so this is the check that closes that gap on a
machine that does. It reports what it finds and does not install anything.

It is engine-agnostic on purpose: it asks `list_voices()` what the server would
serve rather than naming an engine, so it keeps working when the engine
changes. It said "piper" everywhere until Piper was removed, which is exactly
the coupling that made it need rewriting rather than reading.
"""
from __future__ import annotations

import asyncio
import struct
import sys
import time

import voice_store
from audio_utils import pcm_duration
from tts import PRODUCTION_ENGINES, engine_for_voice, engine_report, list_voices

SENTENCE = (
    "This is a test of the voice this app speaks with. "
    "If this sounds like a person rather than a robot, it is working."
)

#: Below this and an episode starves: the listener hears gaps, because
#: synthesis is not keeping ahead of playback.
REALTIME_FLOOR = 1.0


def write_wav(name: str, pcm: bytes, rate: int) -> None:
    header = (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
              + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
              + b"data" + struct.pack("<I", len(pcm)))
    with open(name, "wb") as handle:
        handle.write(header + pcm)


async def main() -> int:
    store = voice_store.ensure_ready()
    print(f"Voice store: {store['dir']}")

    report = engine_report()
    print(f"Production engine(s): {', '.join(report['production_engines'])}")
    print(f"Speaking with       : {report['selected']}")

    if report["interim"]:
        # Not a lower-quality voice. There is no second engine any more, so
        # this is a tone - and saying "the app still works" here is exactly the
        # silent success this project has lost the most time to.
        print("\n  NO VOICE ON THIS MACHINE. What plays is a placeholder tone,")
        print("  not FAM. Nothing below is a judgement of how FAM sounds.")
        for cls in PRODUCTION_ENGINES:
            print(f"    {cls.name}: {cls.diagnose()[1]}")
        print("    Fix: pip install -r requirements-chatterbox.txt, on a GPU")
        print("         machine, with a reference recording whose rights record")
        print("         clears consent, commercial use and synthetic voice.")
        print("         See RUNPOD_PRODUCTION.md.\n")

    voices = list_voices()
    print(f"\n{len(voices)} voice(s) offered:\n")

    slowest = None
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
        ratio = seconds / elapsed if elapsed else 0.0
        slowest = ratio if slowest is None else min(slowest, ratio)
        print(
            f"  {voice.id:34} {seconds:5.2f}s audio  {engine.sample_rate} Hz  "
            f"{elapsed:5.2f}s to make ({ratio:6.1f}x realtime)"
        )
        if "--save" in sys.argv[1:]:
            name = voice.id.replace(":", "_") + ".wav"
            write_wav(name, pcm, engine.sample_rate)
            print(f"      wrote {name} - listen to it")

    if report["interim"]:
        print("\nThis machine cannot speak. The timings above are a tone "
              "generator's.")
        return 1
    if slowest is not None and slowest < REALTIME_FLOOR:
        print(f"\nWorking, but at {slowest:.1f}x realtime an episode would "
              "starve mid-sentence.")
        print("Chatterbox refuses CPU for this reason; check the device it "
              "actually loaded on.")
        return 1
    print("\nThe production voice is installed and working on this machine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
