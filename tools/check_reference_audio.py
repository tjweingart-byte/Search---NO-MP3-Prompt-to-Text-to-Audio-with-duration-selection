#!/usr/bin/env python3
"""Check the three reference recordings before a single clip is generated.

Two jobs, and the second is the one that matters more.

**Technical.** Chatterbox Base conditions on a reference recording in three
places, and they do not consume the same amount of it (`tts.py:182-206`):

    s3gen reference    first 10s   DEC_COND_LEN = 10 * 24000
    T3 prompt tokens   first 6s    ENC_COND_LEN = 6 * 16000
    speaker embedding  the WHOLE file, untruncated

So reference length is not a free variable: a 30s clip and a 12s clip feed the
speaker embedding different amounts while feeding the other two the same. This
checks that all four are the same length and long enough, along with rate,
channels, clipping, silence and noise floor.

**Rights.** Every reference must have a rights record beside it. The tool
refuses to pass a voice without one. That is deliberate: cloning a voice
without permission is the one failure in this project that cannot be fixed by
re-running something.

    python tools/check_reference_audio.py experiments/references

Nothing here is legal advice. It checks that a record exists and is filled in;
whether the rights are actually sufficient is for the project owner and their
counsel.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

#: From tts.py. The reference is consumed at three different lengths.
DEC_COND_SECONDS = 10.0
ENC_COND_SECONDS = 6.0

#: Long enough to fill the 10s s3gen window with room to spare, short enough to
#: record cleanly in one take.
MIN_SECONDS = 12.0
MAX_SECONDS = 30.0

#: All references must match within this, or the speaker embedding sees
#: unequal amounts of each voice.
DURATION_TOLERANCE = 2.0

#: A reference that clips has distortion baked into the cloned voice.
PEAK_CEILING = 0.99

#: Maps neutral ids to the real files. Git-ignored, and never read by anything
#: that produces judging output.
SOURCES = "sources.json"

#: Anything librosa can open. Chatterbox itself calls `librosa.load`, so if a
#: format is readable here it is readable there - and if it is not, that is a
#: real finding rather than a checker limitation.
AUDIO_SUFFIXES = (".wav", ".m4a", ".mp3", ".flac", ".aac", ".ogg", ".aiff",
                  ".aif", ".mp4", ".opus", ".wma")


def neutral_ids(count: int = 3) -> list:
    return [f"reference_{index + 1}" for index in range(count)]


def load_sources(folder: pathlib.Path) -> dict:
    """Neutral id -> real path, from sources.json or from reference_N files.

    The mapping lives in a file rather than in a filename so the recordings can
    keep whatever names they arrived with. Nothing that judging touches ever
    reads it.
    """
    mapping_path = folder / SOURCES
    if mapping_path.exists():
        raw = json.loads(mapping_path.read_text(encoding="utf-8"))
        entries = raw.get("map", raw)
        return {key: folder / name for key, name in entries.items()
                if not key.startswith("_")}
    found = {}
    for key in neutral_ids():
        matches = [m for m in sorted(folder.glob(f"{key}.*"))
                   if m.suffix.lower() in AUDIO_SUFFIXES]
        if matches:
            found[key] = matches[0]
    return found


def adopt(folder: pathlib.Path, seed: int = 20260908) -> dict:
    """Assign neutral ids to whatever recordings are in the folder.

    The assignment is shuffled rather than alphabetical. The listener only ever
    sees the per-passage letters, so this is belt and braces - but a mapping
    anyone could infer from the filenames is not worth keeping.

    Source recordings are never renamed, moved or converted.
    """
    import hashlib
    import random

    audio = sorted(path for path in folder.iterdir()
                   if path.suffix.lower() in AUDIO_SUFFIXES)
    if not audio:
        raise SystemExit(f"no audio files in {folder}")
    # The expected count comes from the experiment, not from what happens to
    # be in the folder - deriving it from the folder made the check below
    # tautological, so two recordings would have been adopted silently.
    ids = neutral_ids()
    if len(audio) != len(ids):
        raise SystemExit(
            f"found {len(audio)} recordings but the experiment expects "
            f"{len(ids)}: {', '.join(p.name for p in audio)}")

    digest = hashlib.sha256(str(seed).encode()).hexdigest()
    order = list(audio)
    random.Random(int(digest[:16], 16)).shuffle(order)
    mapping = {key: path.name for key, path in zip(ids, order)}

    (folder / SOURCES).write_text(json.dumps({
        "_comment": ("Neutral id -> source recording. Git-ignored. Nothing "
                     "that produces judging output reads this file."),
        "map": mapping,
    }, indent=2) + "\n", encoding="utf-8")

    for key in ids:
        rights = folder / f"{key}.rights.json"
        if rights.exists():
            continue
        rights.write_text(json.dumps({
            "_comment": ("Answer every field. The runner refuses to synthesise "
                         "a voice whose record is incomplete."),
            "source": "TODO",
            "speaker": "TODO",
            "consent": None,
            "commercial_use": None,
            "synthetic_voice_cleared": None,
            "notes": "TODO",
        }, indent=2) + "\n", encoding="utf-8")
    return mapping


#: Fields a rights record must actually answer.
RIGHTS_FIELDS = ("source", "speaker", "consent", "commercial_use",
                 "synthetic_voice_cleared", "notes")


def probe(path: pathlib.Path) -> dict:
    """Duration, rate, channels, peak and a rough noise floor."""
    import numpy as np

    try:
        import soundfile as sf

        data, rate = sf.read(str(path), always_2d=True)
        channels = data.shape[1]
        mono = data.mean(axis=1)
    except Exception:
        try:
            import librosa

            mono, rate = librosa.load(str(path), sr=None, mono=True)
            channels = 1
        except Exception:
            # Last resort: the standard library reads plain WAV. This matters
            # because preparing references is exactly when Chatterbox and its
            # audio stack may not be installed yet, and the check should still
            # work then.
            try:
                import wave as _wave

                with _wave.open(str(path), "rb") as handle:
                    rate = handle.getframerate()
                    channels = handle.getnchannels()
                    width = handle.getsampwidth()
                    frames = handle.readframes(handle.getnframes())
                if width != 2:
                    raise ValueError(f"{width * 8}-bit WAV; expected 16-bit")
                data = np.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0
                mono = data.reshape(-1, channels).mean(axis=1) if channels > 1 else data
            except Exception as exc:
                return {"ok": False,
                        "error": f"{type(exc).__name__}: {exc}  "
                                 "(install soundfile or librosa for non-WAV)"}

    mono = np.asarray(mono, dtype="float32")
    if mono.size == 0:
        return {"ok": False, "error": "no samples"}

    seconds = len(mono) / rate
    peak = float(np.max(np.abs(mono)))
    # Noise floor: the quietest tenth of 50 ms frames, which is background
    # rather than speech in any normal recording.
    frame = max(1, int(rate * 0.05))
    frames = mono[: len(mono) // frame * frame].reshape(-1, frame)
    rms = np.sqrt(np.mean(frames ** 2, axis=1)) if frames.size else np.array([0.0])
    quiet = float(np.percentile(rms, 10))
    loud = float(np.percentile(rms, 90)) or 1e-9
    return {
        "ok": True, "seconds": seconds, "rate": int(rate), "channels": channels,
        "peak": peak,
        "noise_floor_db": 20.0 * float(np.log10(max(quiet, 1e-9))),
        "speech_db": 20.0 * float(np.log10(loud)),
        "dynamic_range_db": 20.0 * float(np.log10(loud / max(quiet, 1e-9))),
    }


def check_one(path: pathlib.Path) -> tuple:
    """Returns (info, problems, warnings) for one reference."""
    info = probe(path)
    problems, warnings = [], []
    if not info.get("ok"):
        return info, [f"unreadable: {info.get('error')}"], []

    if info["seconds"] < MIN_SECONDS:
        problems.append(
            f"{info['seconds']:.1f}s is shorter than {MIN_SECONDS:.0f}s; the "
            f"s3gen window alone takes {DEC_COND_SECONDS:.0f}s")
    if info["seconds"] > MAX_SECONDS:
        warnings.append(f"{info['seconds']:.1f}s is longer than needed; only the "
                        "speaker embedding sees past 10s")
    if info["peak"] >= PEAK_CEILING:
        problems.append(f"peaks at {info['peak']:.3f} - likely clipped, and "
                        "distortion is cloned along with the voice")
    if info["rate"] < 16000:
        problems.append(f"{info['rate']} Hz is below the 16 kHz the tokenizer "
                        "needs; upsampling will not restore what is missing")
    if info["dynamic_range_db"] < 20:
        warnings.append(f"only {info['dynamic_range_db']:.0f} dB between "
                        "background and speech - a noisy or heavily compressed "
                        "recording clones its room")
    if info["channels"] > 1:
        warnings.append(f"{info['channels']} channels; librosa will downmix to "
                        "mono, which is fine but do it yourself to be sure")
    return info, problems, warnings


def check_rights(folder: pathlib.Path, name: str) -> list:
    """A reference without a filled-in rights record does not pass."""
    path = folder / f"{name}.rights.json"
    if not path.exists():
        return [f"no rights record at {path.name} - refusing to use this voice"]
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{path.name} is not valid JSON: {exc}"]

    problems = []
    for field in RIGHTS_FIELDS:
        value = record.get(field)
        if field == "notes":
            continue
        if value in (None, "", "TODO", "unknown"):
            problems.append(f"{path.name}: {field!r} is not answered")
    for field in ("consent", "commercial_use", "synthetic_voice_cleared"):
        if record.get(field) is False:
            problems.append(f"{path.name}: {field} is false - this voice may "
                            "not be used")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", nargs="?", default="experiments/references")
    parser.add_argument("--adopt", action="store_true",
                        help="assign neutral ids to the recordings that are "
                             "already in the folder, without renaming them")
    args = parser.parse_args()
    folder = pathlib.Path(args.folder)

    if args.adopt:
        mapping = adopt(folder)
        print(f"\nwrote {folder / SOURCES}")
        for key in sorted(mapping):
            print(f"  {key}  <-  {mapping[key]}")
        print("\n  Source recordings were not renamed, moved or converted.")
        print("  Rights templates created for any that had none.\n")

    if not folder.exists():
        raise SystemExit(f"no folder at {folder}\n"
                         "  Put the three recordings in it under any names, "
                         "then run this with --adopt to assign neutral ids.")

    # Neutral names on purpose: a filename that asserts a direction primes the
    # listener and claims a mapping nobody has verified.
    expected = neutral_ids()
    sources = load_sources(folder)
    ok, results = True, {}
    print(f"\nreference audio in {folder}\n")
    if (folder / SOURCES).exists():
        print(f"  mapping from {SOURCES}; source recordings untouched\n")
    for name in expected:
        path = sources.get(name)
        if path is None or not path.exists():
            print(f"  MISSING  {name}: no audio file"
                  + (f" at {path.name}" if path is not None else ""))
            ok = False
            continue
        info, problems, warnings = check_one(path)
        problems += check_rights(folder, name)
        results[name] = info
        status = "ok  " if not problems else "FAIL"
        ok &= not problems
        if info.get("ok"):
            print(f"  {status}  {name:<12}{path.name:<22}"
                  f"{info['seconds']:>6.1f}s  {info['rate']:>6} Hz  "
                  f"peak {info['peak']:.2f}  "
                  f"floor {info['noise_floor_db']:.0f} dB")
        else:
            print(f"  {status}  {name:<12}{path.name}")
        for problem in problems:
            print(f"          - {problem}")
        for warning in warnings:
            print(f"          ~ {warning}")

    durations = [i["seconds"] for i in results.values() if i.get("ok")]
    if len(durations) > 1:
        spread = max(durations) - min(durations)
        print(f"\n  duration spread {spread:.1f}s "
              f"({min(durations):.1f}-{max(durations):.1f}s)")
        if spread > DURATION_TOLERANCE:
            print(f"  FAIL  more than {DURATION_TOLERANCE:.0f}s apart. The "
                  "speaker embedding reads the whole file while the other two\n"
                  "        conditionings are truncated, so unequal lengths feed "
                  "the voices unequally.")
            ok = False

    print()
    if ok and len(results) == len(expected):
        print(f"  all {len(expected)} references pass. Generation can "
              "proceed.\n")
        return 0
    print("  not ready. Fix the above before generating anything.\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
