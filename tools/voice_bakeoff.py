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


def log_line(handle, message: str) -> None:
    """Say it on stdout and in the file, and flush both.

    The failure this exists for is `zsh: killed` - the process does not get to
    finish, so anything buffered is gone. A flushed file is what tells us which
    candidate and which passage were in flight when the kernel stepped in.
    """
    print(message, flush=True)
    if handle is not None:
        handle.write(message + "\n")
        handle.flush()


def rss_mb() -> str:
    """Resident memory, if psutil is here. Absent is fine; guessing is not."""
    try:
        import os

        import psutil

        return f"{psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.0f}MB"
    except Exception:
        return "?"


def generate_all(candidates, passages, out, device, log, force: bool) -> dict:
    """One engine at a time: load, speak, persist, release. Then the next.

    Every clip is written before the next is attempted, so a kill costs the
    engine in flight and nothing that came before it.
    """
    done, failed = {}, {}
    for candidate in candidates:
        wanted = [p for p in passages
                  if force or not bake.raw_path(out, candidate.key, p["id"]).exists()]
        already = len(passages) - len(wanted)
        if already:
            log_line(log, f"[{candidate.key}] {already}/{len(passages)} already on "
                          f"disk - not regenerating (first-take rule)")
        if not wanted:
            done[candidate.key] = len(passages)
            continue

        log_line(log, f"[{candidate.key}] LOADING   rss={rss_mb()}")
        started = time.perf_counter()
        try:
            synth = candidate.synth()
        except Exception as exc:
            failed[candidate.key] = f"load: {type(exc).__name__}: {exc}"
            log_line(log, f"[{candidate.key}] LOAD FAILED {failed[candidate.key]}")
            bake.release_memory(device)
            continue
        log_line(log, f"[{candidate.key}] loaded in {time.perf_counter() - started:.1f}s"
                      f"  rss={rss_mb()}")

        count = already
        for passage in wanted:
            log_line(log, f"[{candidate.key}] GENERATING {passage['id']}  "
                          f"rss={rss_mb()}")
            began = time.perf_counter()
            try:
                samples, rate = synth(passage["text"])
            except Exception as exc:
                failed.setdefault(candidate.key,
                                  f"{passage['id']}: {type(exc).__name__}: {exc}")
                log_line(log, f"[{candidate.key}] {passage['id']} FAILED "
                              f"{type(exc).__name__}: {exc}")
                continue
            elapsed = time.perf_counter() - began
            path = bake.raw_path(out, candidate.key, passage["id"])
            path.parent.mkdir(parents=True, exist_ok=True)
            bake.write_wav(path, samples, rate)
            seconds = len(samples) / rate if rate else 0.0
            count += 1
            log_line(log, f"[{candidate.key}] wrote {passage['id']}  "
                          f"{seconds:.1f}s audio in {elapsed:.1f}s  rss={rss_mb()}")
            del samples

        # Release before the next engine loads. This is the whole fix.
        del synth
        bake.release_memory(device)
        log_line(log, f"[{candidate.key}] RELEASED  rss={rss_mb()}")
        done[candidate.key] = count

    return {"done": done, "failed": failed}


def label(candidates, passages, out, seed: int, log) -> tuple:
    """Second pass: read the raw clips back, normalise, assign blind letters.

    Only candidates with a complete set are labelled. A half-generated engine
    would otherwise appear on some passages and not others, which tells the
    listener something the test is meant to hide.
    """
    complete = [c for c in candidates
                if all(bake.raw_path(out, c.key, p["id"]).exists() for p in passages)]
    incomplete = [c.key for c in candidates if c not in complete]
    if incomplete:
        log_line(log, f"not labelled (incomplete): {', '.join(incomplete)}")
    if len(complete) < 2:
        return complete, {}, []

    letters_by_passage = {
        p["id"]: bake.assign_letters([c.key for c in complete], p["id"], seed)
        for p in passages}

    clips = []
    for passage in passages:
        folder = out / "clips" / passage["id"]
        folder.mkdir(parents=True, exist_ok=True)
        for candidate in complete:
            samples, rate = bake.read_wav(bake.raw_path(out, candidate.key,
                                                        passage["id"]))
            samples, before, after = bake.normalise(samples, rate)
            letter = letters_by_passage[passage["id"]][candidate.key]
            bake.write_wav(folder / f"{letter}.wav", samples, rate)
            clips.append(bake.Clip(passage["id"], candidate.key, letter,
                                   len(samples) / rate if rate else 0.0, rate,
                                   0.0, before, after))
    return complete, letters_by_passage, clips


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", help="mps / cuda for the Chatterbox models")
    parser.add_argument("--piper-voice", dest="piper_voice")
    parser.add_argument("--passages", default=PASSAGES)
    parser.add_argument("--out", default="experiments/results/bakeoff")
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--only", help="comma-separated candidate keys")
    parser.add_argument("--force", action="store_true",
                        help="regenerate clips that are already on disk")
    args = parser.parse_args()

    passages = json.loads(
        pathlib.Path(args.passages).read_text(encoding="utf-8"))["passages"]
    candidates = build_candidates(args.device, args.piper_voice)
    if args.only:
        wanted = {k.strip() for k in args.only.split(",")}
        candidates = [c for c in candidates if c.key in wanted]

    out = pathlib.Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "raw" / "DO_NOT_BROWSE.txt").write_text(
        "Raw clips, named by engine. Opening this folder tells you which voice "
        "is which and spoils the blind test. Use listen.html.\n", encoding="utf-8")

    with (out / "progress.log").open("a", encoding="utf-8") as log:
        log_line(log, f"\n=== run {time.strftime('%Y-%m-%d %H:%M:%S')} "
                      f"device={args.device} ===")
        log_line(log, "roster")
        for candidate in candidates:
            log_line(log, f"  {'ok  ' if candidate.available else 'SKIP'}  "
                          f"{candidate.key:<18}{candidate.label}"
                          + (f"   ({candidate.reason})"
                             if not candidate.available else ""))
        usable = [c for c in candidates if c.available]
        if len(usable) < 2:
            raise SystemExit("\nfewer than two candidates can run")

        log_line(log, "\ngenerating - one engine loaded at a time")
        outcome = generate_all(usable, passages, out, args.device, log, args.force)

        log_line(log, "\nlabelling")
        complete, letters_by_passage, clips = label(
            usable, passages, out, args.seed, log)

        if outcome["failed"]:
            log_line(log, "\nfailures")
            for key, reason in outcome["failed"].items():
                log_line(log, f"  {key}: {reason}")
            log_line(log, "  Rerun the same command; finished clips are kept and "
                          "only the missing ones are generated.")

        if not clips:
            log_line(log, "\nfewer than two complete candidates - nothing to "
                          "compare yet. Rerun to continue.")
            return 1

        (out / "listen.html").write_text(
            bake.player_html(passages, letters_by_passage), encoding="utf-8")
        (out / "scorecard.md").write_text(
            bake.scorecard_markdown(passages, letters_by_passage), encoding="utf-8")
        (out / "KEY.json").write_text(json.dumps({
            "DO_NOT_OPEN_UNTIL_SCORED": True,
            "seed": args.seed,
            "candidates": {c.key: {"label": c.label, "notes": c.notes}
                           for c in complete},
            "letters": letters_by_passage,
            "clips": [c.__dict__ for c in clips],
            "failed": outcome["failed"],
        }, indent=2), encoding="utf-8")

        log_line(log, f"\nlabelled {len(complete)} candidate(s) across "
                      f"{len(passages)} passages")
        log_line(log, f"  {out}/listen.html    open this")
        log_line(log, f"  {out}/scorecard.md   fill this in while listening")
        log_line(log, f"  {out}/KEY.json       open only afterwards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
