#!/usr/bin/env python3
"""Chatterbox Base, three reference voices, blind. Local, free, no GPU rental.

Engine selection is finished; this chooses the voice. Chatterbox Base is the
only engine here, loaded once, and the only thing that varies between the
candidates is which reference recording conditions it. The references are named
neutrally - a filename asserting a direction would prime the listener and claim
a mapping nobody has verified.

    python tools/check_reference_audio.py experiments/references     # gate
    python tools/voice_identity_bakeoff.py --device mps \\
        --out experiments/results/identity

Same discipline as the engine bake-off: blind letters randomised per passage,
loudness matched, first takes kept, every clip checkpointed, and one model
resident at a time. Plus a fixed random seed per passage shared by every
identity, so no voice wins on a luckier sample.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments import voice_bakeoff as bake                # noqa: E402
from experiments import voice_identity as identity           # noqa: E402
from tools.voice_bakeoff import log_line, rss_mb             # noqa: E402

PASSAGES = "experiments/passages/fam_voice_passages.json"
REFERENCES = "experiments/references"


def resolve_references(folder: pathlib.Path) -> dict:
    """One audio file per identity, or a refusal naming what is missing.

    Resolution goes through `check_reference_audio.load_sources`, so the
    recordings keep whatever names they arrived with and the neutral ids come
    from `sources.json`. Nothing downstream ever sees a source filename.
    """
    from tools.check_reference_audio import load_sources

    sources = load_sources(folder)
    found, missing = {}, []
    for key, _label, _note in identity.IDENTITIES:
        path = sources.get(key)
        if path is not None and path.exists():
            found[key] = path
        else:
            missing.append(key)
    if missing:
        raise SystemExit(
            f"missing reference audio for: {', '.join(missing)}\n"
            f"  Put the recordings in {folder} under any names, then:\n"
            f"    python tools/check_reference_audio.py {folder} --adopt\n"
            f"    python tools/check_reference_audio.py {folder}")
    return found


def rights_cleared(folder: pathlib.Path) -> list:
    """Refuse to synthesise a voice with no rights record. No exceptions."""
    from tools.check_reference_audio import check_rights

    problems = []
    for key, _label, _note in identity.IDENTITIES:
        problems += check_rights(folder, key)
    return problems


def generate_all(model, references: dict, passages: list, out: pathlib.Path,
                 base_seed: int, log, force: bool) -> dict:
    """Every identity x passage, checkpointed, with the seed pinned per passage."""
    import torch

    done, failed = {}, {}
    for key, _label, _note in identity.IDENTITIES:
        count = 0
        for passage in passages:
            path = bake.raw_path(out, key, passage["id"])
            if path.exists() and not force:
                log_line(log, f"[{key}] {passage['id']} already on disk - "
                              "not regenerating (first-take rule)")
                count += 1
                continue

            seed = identity.seed_for(passage["id"], base_seed)
            identity.apply_seed(seed)
            log_line(log, f"[{key}] GENERATING {passage['id']}  seed={seed}  "
                          f"rss={rss_mb()}")
            began = time.perf_counter()
            try:
                with torch.inference_mode():
                    wav = model.generate(
                        passage["text"],
                        audio_prompt_path=str(references[key]),
                        **identity.GENERATION)
                samples = wav.squeeze(0).detach().cpu().numpy()
            except Exception as exc:
                failed.setdefault(key, f"{passage['id']}: "
                                       f"{type(exc).__name__}: {exc}")
                log_line(log, f"[{key}] {passage['id']} FAILED "
                              f"{type(exc).__name__}: {exc}")
                continue
            elapsed = time.perf_counter() - began
            rate = int(getattr(model, "sr", 24000))
            path.parent.mkdir(parents=True, exist_ok=True)
            bake.write_wav(path, samples, rate)
            count += 1
            log_line(log, f"[{key}] wrote {passage['id']}  "
                          f"{len(samples) / rate:.1f}s audio in {elapsed:.1f}s  "
                          f"rss={rss_mb()}")
            del samples, wav
        done[key] = count
    return {"done": done, "failed": failed}


def label(passages: list, out: pathlib.Path, seed: int, log) -> tuple:
    """Read the raw clips back, normalise, assign blind letters."""
    keys = [k for k, _, _ in identity.IDENTITIES
            if all(bake.raw_path(out, k, p["id"]).exists() for p in passages)]
    incomplete = [k for k, _, _ in identity.IDENTITIES if k not in keys]
    if incomplete:
        log_line(log, f"not labelled (incomplete): {', '.join(incomplete)}")
    if len(keys) < 2:
        return keys, {}, []

    letters_by_passage = {p["id"]: bake.assign_letters(keys, p["id"], seed)
                          for p in passages}
    clips = []
    for passage in passages:
        folder = out / "clips" / passage["id"]
        folder.mkdir(parents=True, exist_ok=True)
        for key in keys:
            samples, rate = bake.read_wav(bake.raw_path(out, key, passage["id"]))
            samples, before, after = bake.normalise(samples, rate)
            letter = letters_by_passage[passage["id"]][key]
            bake.write_wav(folder / f"{letter}.wav", samples, rate)
            clips.append(identity.IdentityClip(
                passage["id"], key, letter,
                len(samples) / rate if rate else 0.0, rate, 0.0,
                identity.seed_for(passage["id"], seed), before, after))
    return keys, letters_by_passage, clips


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", help="mps / cuda")
    parser.add_argument("--references", default=REFERENCES)
    parser.add_argument("--passages", default=PASSAGES)
    parser.add_argument("--out", default="experiments/results/identity")
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-rights-check", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    folder = pathlib.Path(args.references)
    references = resolve_references(folder)
    if not args.skip_rights_check:
        problems = rights_cleared(folder)
        if problems:
            print("\nrights records are not in order:\n")
            for problem in problems:
                print(f"  - {problem}")
            raise SystemExit(
                "\nRefusing to synthesise a voice without a rights record.\n"
                "  See experiments/references/README.md.\n")

    passages = json.loads(
        pathlib.Path(args.passages).read_text(encoding="utf-8"))["passages"]
    out = pathlib.Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "raw" / "DO_NOT_BROWSE.txt").write_text(
        "Raw clips, named by identity. Opening this folder spoils the blind "
        "test. Use listen.html.\n", encoding="utf-8")

    with (out / "progress.log").open("a", encoding="utf-8") as log:
        log_line(log, f"\n=== identity run {time.strftime('%Y-%m-%d %H:%M:%S')} "
                      f"device={args.device} ===")
        # Never the source filename: progress.log persists on disk while the
        # blind judging happens, and a speaker's name in it would give the
        # answer away before KEY.json is opened. Size and format are enough to
        # debug with.
        for key, label_text, _note in identity.IDENTITIES:
            path = references[key]
            log_line(log, f"  {key:<14}{path.suffix.lstrip('.'):<6}"
                          f"{path.stat().st_size / 1024:>8.0f} KB  {label_text}")
        log_line(log, f"  settings held constant: {identity.GENERATION}")

        needed = [(k, p) for k, _, _ in identity.IDENTITIES for p in passages
                  if args.force or not bake.raw_path(out, k, p["id"]).exists()]
        model = None
        if needed:
            from experiments.adapters.chatterbox_impl import resolve_device

            resolved, _ = resolve_device(args.device)
            log_line(log, f"\nloading Chatterbox Base on {resolved} "
                          f"({len(needed)} clip(s) to generate)")
            from chatterbox.tts import ChatterboxTTS

            started = time.perf_counter()
            model = ChatterboxTTS.from_pretrained(device=resolved)
            actual = str(getattr(model, "device", resolved))
            if actual.split(":")[0] != resolved:
                raise SystemExit(f"asked for {resolved!r}, model loaded on "
                                 f"{actual!r}")
            log_line(log, f"loaded in {time.perf_counter() - started:.1f}s  "
                          f"rss={rss_mb()}")
            outcome = generate_all(model, references, passages, out, args.seed,
                                   log, args.force)
            del model
            bake.release_memory(args.device)
            log_line(log, f"released  rss={rss_mb()}")
        else:
            log_line(log, "\nevery clip is already on disk - labelling only")
            outcome = {"done": {}, "failed": {}}

        keys, letters_by_passage, clips = label(passages, out, args.seed, log)
        if outcome["failed"]:
            log_line(log, "\nfailures")
            for key, reason in outcome["failed"].items():
                log_line(log, f"  {key}: {reason}")
            log_line(log, "  Rerun the same command; finished clips are kept.")
        if not clips:
            log_line(log, "\nfewer than two complete identities - rerun.")
            return 1

        (out / "listen.html").write_text(
            identity.identity_player_html(passages, letters_by_passage),
            encoding="utf-8")
        (out / "choices.md").write_text(
            identity.choice_sheet(passages, letters_by_passage), encoding="utf-8")
        (out / "KEY.json").write_text(json.dumps({
            "DO_NOT_OPEN_UNTIL_CHOSEN": True,
            "engine": "Chatterbox Base (chatterbox.tts.ChatterboxTTS)",
            "seed": args.seed,
            "generation_settings": identity.GENERATION,
            # No "direction" field: nothing here asserts which speaker is the
            # magnetic one. The qualities being listened for are recorded
            # separately, unattached to any reference.
            "identities": {k: {"label": lab, "note": note,
                               "reference": references[k].name}
                           for k, lab, note in identity.IDENTITIES if k in keys},
            "qualities_sought": dict(identity.QUALITIES_SOUGHT),
            "letters": letters_by_passage,
            "clips": [c.__dict__ for c in clips],
            "failed": outcome["failed"],
        }, indent=2), encoding="utf-8")

        log_line(log, f"\nlabelled {len(keys)} identities across {len(passages)} "
                      "passages")
        log_line(log, f"  {out}/listen.html   open this")
        log_line(log, f"  {out}/choices.md    record your choices")
        log_line(log, f"  {out}/KEY.json      open only afterwards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
