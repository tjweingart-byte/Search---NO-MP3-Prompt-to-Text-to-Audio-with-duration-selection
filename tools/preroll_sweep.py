#!/usr/bin/env python3
"""What the preroll gate actually costs, measured across values and openings.

`PREROLL_SECONDS` is a **quantity** gate, not a delay: `app.py` counts bytes of
audio, not elapsed time. At `TARGET_WPM = 150` the shipped 1.5s is 3.75 words,
so an ordinary opening sentence satisfies it on the first chunk and it costs
nothing. This measures that claim rather than repeating it, and measures the
case where it is not true - an unusually short opening, which the Phase 6
first-chunk rule deliberately allows.

It drives the real `/api/audio` endpoint, so the preroll gate, the
empty-episode guard and the response headers are all the production ones.

    python3 tools/preroll_sweep.py                          # whatever is present
    python3 tools/preroll_sweep.py --engine chatterbox      # the one that matters
    python3 tools/preroll_sweep.py --runs 3 --minutes 3

**Chatterbox is the engine to run this on**, which means a GPU machine. The
debug engine synthesises arithmetic and is effectively infinitely fast, so it
answers the quantity question (how many chunks) exactly and the timing question
not at all. The tool prints which engine it used and refuses to pretend
otherwise.

This measured Piper until Piper was removed. Numbers from those runs describe a
CPU voice at ~1x realtime and do not carry over to a GPU one at ~4.6x - the
quantity results do, since the gate counts bytes.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

#: The values to sweep. Zero is deliberately absent: `config` refuses it,
#: because a streamed WAV's 44-byte header alone would satisfy a zero gate and
#: every episode would be refused as empty.
DEFAULT_VALUES = (1.5, 1.0, 0.75, 0.5, 0.25, 0.1)

#: Two opening shapes. The first is what FAM writes; the second is the short
#: opening the Phase 6 first-chunk rule permits, and the only case where the
#: gate can force extra synthesis.
OPENINGS = {
    "normal": "The oldest working clock in Europe has no face, and for six "
              "hundred years nobody thought that was strange.",
    "short": "It rang.",
}

BODY = ("Salisbury Cathedral has kept it turning since about thirteen "
        "eighty-six, through a civil war and two restorations.")


class Opening:
    """A deterministic script with a chosen first sentence."""

    def __init__(self, first: str, sentences: int = 60):
        self.first, self.sentences = first, sentences

    async def stream_sentences(self, plan, notes=None):
        import asyncio

        yield self.first
        for _ in range(self.sentences):
            await asyncio.sleep(0)
            yield BODY

    async def top_up(self, plan, spoken_so_far, words_needed):
        async for sentence in self.stream_sentences(plan):
            yield sentence


def build_engine(name: str):
    from tts import ChatterboxEngine, DebugEngine

    if name == "chatterbox":
        if not ChatterboxEngine.available():
            raise SystemExit(
                "Chatterbox is not available here, so this run would measure "
                "the debug engine\n  while claiming to measure the production "
                f"voice.\n  Reason: {ChatterboxEngine.diagnose()[1]}\n"
                "  See RUNPOD_PRODUCTION.md.")
        return ChatterboxEngine()
    if name == "debug":
        return DebugEngine()
    for cls in (ChatterboxEngine, DebugEngine):
        if cls.available():
            return cls()
    raise SystemExit("no speech engine available")


def one_run(client, appmod, pipeline_mod, engine, opening: str, minutes: int) -> dict:
    """One real request. Returns the marks the endpoint reported."""
    pipeline_mod_ref = pipeline_mod

    def make(voice=None):
        return pipeline_mod_ref.PodcastPipeline(
            generator=Opening(OPENINGS[opening]), engine=engine, cache=None,
            voice=voice)

    appmod._make_pipeline = make
    started = time.perf_counter()
    with client.stream("GET", f"/api/audio?q=preroll+sweep&minutes={minutes}"
                              "&fmt=pcm") as response:
        response.raise_for_status()
        first_byte = None
        received = 0
        for block in response.iter_bytes():
            if first_byte is None and block:
                first_byte = time.perf_counter() - started
            received += len(block)
        headers = dict(response.headers)

    rate = int(headers.get("x-sample-rate", 22050))

    def mark(name):
        raw = headers.get(name, "")
        return float(raw) if raw else None

    return {
        "opening": opening,
        "client_first_audio": first_byte,
        "audio_seconds": received / (rate * 2),
        "chunks_primed": int(headers.get("x-chunks-primed", 0)),
        "audio_primed": float(headers.get("x-audio-primed-seconds", 0.0)),
        "first_pcm": mark("x-first-pcm-seconds"),
        "preroll_satisfied": mark("x-preroll-satisfied-seconds"),
        "total": time.perf_counter() - started,
    }


def summarise(rows: list) -> dict:
    def median(key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return statistics.median(values) if values else None

    first_pcm, preroll = median("first_pcm"), median("preroll_satisfied")
    return {
        "runs": len(rows),
        "chunks_primed": statistics.median(r["chunks_primed"] for r in rows),
        "audio_primed": median("audio_primed"),
        "first_pcm": first_pcm,
        "preroll_satisfied": preroll,
        # The number the whole question is about: how long the gate held the
        # response after audio already existed.
        "gate_cost": (None if first_pcm is None or preroll is None
                      else preroll - first_pcm),
        "client_first_audio": median("client_first_audio"),
        "audio_seconds": median("audio_seconds"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine", default="auto",
                        choices=("auto", "chatterbox", "debug"))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--minutes", type=int, default=3)
    parser.add_argument("--values", default=",".join(str(v) for v in DEFAULT_VALUES))
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    import importlib
    import os

    os.environ["FAM_IGNORE_DOTENV"] = "1"
    engine = build_engine(args.engine)
    values = [float(v) for v in args.values.split(",")]

    print(f"\nengine: {engine.name}  ({engine.sample_rate} Hz)")
    if engine.name == "debug":
        print("  NOTE: the debug engine is arithmetic, not speech. Chunk counts "
              "are exact;\n        every timing below is a floor, not the "
              "production voice's.")
    print(f"runs per cell: {args.runs}   episode: {args.minutes} min\n")

    results = {"engine": engine.name, "minutes": args.minutes,
               "runs": args.runs, "cells": []}

    for value in values:
        os.environ["PREROLL_SECONDS"] = str(value)
        import config

        importlib.reload(config)
        import app as appmod
        import pipeline as pipeline_mod

        importlib.reload(pipeline_mod)
        importlib.reload(appmod)
        appmod.SCRIPT_CACHE = None
        appmod.DEMO_MODE = False
        appmod._rate_limit = lambda request: None
        from fastapi.testclient import TestClient

        client = TestClient(appmod.app)
        assert appmod.PREROLL_SECONDS == value, appmod.PREROLL_SECONDS

        for opening in OPENINGS:
            rows = [one_run(client, appmod, pipeline_mod, engine, opening,
                            args.minutes) for _ in range(args.runs)]
            cell = {"preroll": value, **summarise(rows)}
            results["cells"].append(cell)
            print(f"  preroll {value:<5} {opening:<7} "
                  f"chunks={cell['chunks_primed']:>3.0f}  "
                  f"primed={cell['audio_primed']:>6.2f}s  "
                  f"first_pcm={_s(cell['first_pcm'])}  "
                  f"gate_cost={_s(cell['gate_cost'])}  "
                  f"client_first_audio={_s(cell['client_first_audio'])}")

    os.environ.pop("PREROLL_SECONDS", None)
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(results, indent=2),
                                          encoding="utf-8")
        print(f"\nwrote {args.out}")
    print()
    return 0


def _s(value) -> str:
    return "     -" if value is None else f"{value * 1000:5.0f}ms"


if __name__ == "__main__":
    raise SystemExit(main())
