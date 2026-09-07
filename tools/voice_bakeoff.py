#!/usr/bin/env python3
"""Generate the blind FAM voice bake-off. Local, free, no GPU rental.

Roster, from the Phase 0 source audit:

  chatterbox_base    ChatterboxTTS        MIT, expressive controls available
  chatterbox_turbo   ChatterboxTurboTTS   MIT, faster, exaggeration ignored
  kokoro             KPipeline            Apache-2.0 code; weights unverified
  piper              production tts.py    the incumbent, entered unnamed

Excluded, with reasons, so the roster is a decision rather than a default:
  XTTS-v2   weights CPML non-commercial (P18)
  F5-TTS    code MIT but pre-trained models CC-BY-NC, per its own README
  Parler    no readable licence in the distribution, no streaming interface

    python tools/voice_bakeoff.py --device mps --out experiments/results/bakeoff

Writes clips/, listen.html, scorecard.md and KEY.json. Open listen.html, score
in scorecard.md, and open KEY.json only afterwards.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments import voice_bakeoff as bake                # noqa: E402

PASSAGES = "experiments/passages/fam_voice_passages.json"


def build_candidates(device: str | None, piper_voice: str | None) -> list:
    """Every candidate, each reporting honestly whether it can run here."""
    out = []

    def chatterbox(key, label, module, cls, notes):
        def make():
            import importlib

            import torch

            from experiments.adapters.chatterbox_impl import resolve_device

            resolved, _ = resolve_device(device)
            model = getattr(importlib.import_module(module), cls).from_pretrained(
                device=resolved)
            actual = str(getattr(model, "device", resolved))
            if actual.split(":")[0] != resolved:
                raise RuntimeError(
                    f"asked for {resolved!r}, model loaded on {actual!r}")

            def synth(text):
                with torch.inference_mode():
                    wav = model.generate(text)
                return wav.squeeze(0).detach().cpu().numpy(), int(model.sr)
            return synth

        try:
            import importlib

            importlib.import_module(module)
            out.append(bake.Candidate(key=key, label=label, synth=make,
                                      notes=notes))
        except Exception as exc:
            out.append(bake.Candidate(key=key, label=label, synth=None,
                                      available=False,
                                      reason=f"{type(exc).__name__}: {exc}",
                                      notes=notes))

    chatterbox("chatterbox_base", "Chatterbox (base)", "chatterbox.tts",
               "ChatterboxTTS", "MIT; exposes exaggeration / cfg_weight")
    chatterbox("chatterbox_turbo", "Chatterbox Turbo", "chatterbox.tts_turbo",
               "ChatterboxTurboTTS", "MIT; ignores exaggeration (P17)")

    def kokoro_make():
        import numpy as np
        from kokoro import KPipeline

        pipeline = KPipeline(lang_code="a")

        def synth(text):
            pieces = [r.audio for r in pipeline(text, voice="af_heart")
                      if r.audio is not None]
            if not pieces:
                raise RuntimeError("kokoro produced no audio")
            joined = np.concatenate([np.asarray(p).squeeze() for p in pieces])
            return joined, 24000
        return synth

    try:
        import kokoro                                        # noqa: F401
        out.append(bake.Candidate("kokoro", "Kokoro-82M", kokoro_make,
                                  "Apache-2.0 code; weights unverified (P19)"))
    except Exception as exc:
        out.append(bake.Candidate("kokoro", "Kokoro-82M", None, available=False,
                                  reason=f"{type(exc).__name__}: {exc}",
                                  notes="pip install kokoro"))

    def piper_make():
        import asyncio

        import numpy as np

        from audio_utils import pcm_duration                 # noqa: F401
        from config import settings
        from tts import engine_for_voice

        engine = engine_for_voice(piper_voice)

        def synth(text):
            pcm = asyncio.run(engine.synth(text, float(settings.target_wpm),
                                           piper_voice))
            samples = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
            return samples, int(engine.sample_rate)
        return synth

    try:
        from tts import DebugEngine, engine_for_voice

        engine = engine_for_voice(piper_voice)
        if isinstance(engine, DebugEngine):
            raise RuntimeError("only the debug tone is installed; "
                               "run python setup_voices.py")
        out.append(bake.Candidate("piper", "Piper (the incumbent)", piper_make,
                                  "GPL-3.0-or-later; the voice to beat"))
    except Exception as exc:
        out.append(bake.Candidate("piper", "Piper (the incumbent)", None,
                                  available=False,
                                  reason=f"{type(exc).__name__}: {exc}",
                                  notes="the baseline; the test is weaker without it"))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", help="mps / cuda for the Chatterbox models")
    parser.add_argument("--piper-voice", dest="piper_voice")
    parser.add_argument("--passages", default=PASSAGES)
    parser.add_argument("--out", default="experiments/results/bakeoff")
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--only", help="comma-separated candidate keys")
    args = parser.parse_args()

    passages = json.loads(
        pathlib.Path(args.passages).read_text(encoding="utf-8"))["passages"]
    candidates = build_candidates(args.device, args.piper_voice)
    if args.only:
        wanted = {k.strip() for k in args.only.split(",")}
        candidates = [c for c in candidates if c.key in wanted]

    print("\nroster")
    for candidate in candidates:
        print(f"  {'ok  ' if candidate.available else 'SKIP'}  "
              f"{candidate.key:<18}{candidate.label}"
              + (f"   ({candidate.reason})" if not candidate.available else ""))
    usable = [c for c in candidates if c.available]
    if len(usable) < 2:
        raise SystemExit("\nfewer than two candidates can run; nothing to compare")
    if len(usable) < len(candidates):
        print("\n  A missing candidate is a weaker test, not a failed one - but "
              "note which, because\n  the comparison cannot speak for it.")

    out = pathlib.Path(args.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)

    letters_by_passage = {
        p["id"]: bake.assign_letters([c.key for c in usable], p["id"], args.seed)
        for p in passages}

    print("\ngenerating")
    built, clips = {}, []
    for candidate in usable:
        try:
            built[candidate.key] = candidate.synth()
        except Exception as exc:
            print(f"  {candidate.key}: FAILED to load - {type(exc).__name__}: {exc}")
            continue

        for passage in passages:
            letter = letters_by_passage[passage["id"]][candidate.key]
            started = time.perf_counter()
            try:
                samples, rate = built[candidate.key](passage["text"])
            except Exception as exc:
                print(f"  {candidate.key}/{passage['id']}: FAILED - {exc}")
                continue
            elapsed = time.perf_counter() - started
            samples, before, after = bake.normalise(samples, rate)
            folder = out / "clips" / passage["id"]
            folder.mkdir(parents=True, exist_ok=True)
            bake.write_wav(folder / f"{letter}.wav", samples, rate)
            seconds = len(samples) / rate if rate else 0.0
            clips.append(bake.Clip(passage["id"], candidate.key, letter, seconds,
                                   rate, elapsed, before, after))
            print(f"  {candidate.key:<18}{passage['id']:<14}-> Voice {letter}  "
                  f"{seconds:5.1f}s audio in {elapsed:5.1f}s")

    if not clips:
        raise SystemExit("no clips were generated")

    (out / "listen.html").write_text(
        bake.player_html(passages, letters_by_passage), encoding="utf-8")
    (out / "scorecard.md").write_text(
        bake.scorecard_markdown(passages, letters_by_passage), encoding="utf-8")
    (out / "KEY.json").write_text(json.dumps({
        "DO_NOT_OPEN_UNTIL_SCORED": True,
        "seed": args.seed,
        "candidates": {c.key: {"label": c.label, "notes": c.notes}
                       for c in usable},
        "letters": letters_by_passage,
        "clips": [c.__dict__ for c in clips],
    }, indent=2), encoding="utf-8")

    print(f"\nwrote {out}/")
    print(f"  listen.html    open this")
    print(f"  scorecard.md   fill this in while listening")
    print(f"  KEY.json       open only afterwards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
