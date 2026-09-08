#!/usr/bin/env python3
"""Derive equal-length working references. Originals are never touched.

The reference is read three ways, and only one of them is affected by total
length (`tts.py:182-206`):

    s3gen reference    first 10 s      truncated
    T3 prompt tokens   first 6 s       truncated
    speaker embedding  the whole file  NOT truncated

So unequal lengths feed the speaker embedding unequal amounts of each voice,
while the other two conditionings are already identical. Equalising is
therefore about the embedding and nothing else.

**Crop, do not pad.** Padding with silence would put non-speech into the very
conditioning being equalised - the embedding averages over the whole file, so
appended silence dilutes the speaker's frames and dilutes the shortest
recording most. That is the opposite of the intended effect.

**Crop from the head.** The two truncated conditionings read from the start, so
a head-anchored crop leaves them bit-identical to what the original would have
produced. Only the embedding changes, which is exactly what needs to change. A
centre or tail crop would shift the first 6 and 10 seconds and quietly alter
all three.

**Cut at a zero crossing.** The last sample before the cut is walked back to the
nearest zero crossing within a few milliseconds, so the file ends without a
step discontinuity. No fade, no gain change, no padding - nothing is added or
scaled, a few samples are simply not included.

    python tools/equalise_references.py experiments/references

Writes `working/reference_N.wav` and points `sources.json` at them. The
originals stay where they are, unmodified and still mapped in the manifest.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tools.check_reference_audio import (SOURCES, load_sources,  # noqa: E402
                                         neutral_ids)

WORKING = "working"

#: How far back to look for a zero crossing. Long enough to find one in any
#: voiced speech, short enough that the trim is inaudible and irrelevant to the
#: embedding.
ZERO_CROSSING_WINDOW_MS = 10.0


def decode(path: pathlib.Path):
    """Mono float32 samples and the rate, by the same route Chatterbox uses.

    Chatterbox calls `librosa.load`, so anything readable here is readable
    there. m4a needs librosa (and ffmpeg behind it); WAV works without.
    """
    import numpy as np

    try:
        import soundfile as sf

        data, rate = sf.read(str(path), always_2d=True)
        return np.asarray(data.mean(axis=1), dtype="float32"), int(rate)
    except Exception:
        pass
    try:
        import librosa

        samples, rate = librosa.load(str(path), sr=None, mono=True)
        return np.asarray(samples, dtype="float32"), int(rate)
    except Exception:
        pass
    import wave as _wave

    with _wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise SystemExit(f"{path.name}: {width * 8}-bit WAV is not supported; "
                         "install librosa or soundfile")
    data = np.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0
    mono = data.reshape(-1, channels).mean(axis=1) if channels > 1 else data
    return mono, int(rate)


def crop_at_zero_crossing(samples, rate: int, seconds: float):
    """Head-anchored crop to `seconds`, ending at a zero crossing.

    Returns the cropped samples and how many samples the zero-crossing search
    gave back, so the manifest can record that it was negligible.
    """
    import numpy as np

    target = int(round(seconds * rate))
    if target >= len(samples):
        return samples, 0
    window = max(1, int(ZERO_CROSSING_WINDOW_MS / 1000.0 * rate))
    start = max(1, target - window)
    segment = samples[start:target + 1]
    if segment.size > 1:
        signs = np.signbit(segment)
        crossings = np.flatnonzero(signs[1:] != signs[:-1])
        if crossings.size:
            # The crossing nearest the target, not the earliest in the window.
            index = start + int(crossings[-1]) + 1
            return samples[:index], target - index
    return samples[:target], 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", nargs="?", default="experiments/references")
    parser.add_argument("--seconds", type=float,
                        help="target duration; default is the shortest "
                             "recording, which is the only choice that needs "
                             "no padding")
    args = parser.parse_args()

    from experiments.voice_bakeoff import write_wav

    folder = pathlib.Path(args.folder)
    sources = load_sources(folder)
    missing = [key for key in neutral_ids() if key not in sources]
    if missing:
        raise SystemExit(f"no mapping for {', '.join(missing)}; run "
                         f"check_reference_audio.py {folder} --adopt first")

    decoded = {}
    print(f"\nreading originals in {folder}\n")
    for key in neutral_ids():
        path = sources[key]
        samples, rate = decode(path)
        seconds = len(samples) / rate
        decoded[key] = (samples, rate, path, seconds)
        print(f"  {key:<14}{seconds:>7.2f}s  {rate:>6} Hz  {path.suffix.lstrip('.')}")

    shortest = min(entry[3] for entry in decoded.values())
    target = args.seconds or shortest
    if target > shortest:
        raise SystemExit(
            f"{target:.2f}s is longer than the shortest recording "
            f"({shortest:.2f}s). Reaching it would mean padding, and padding "
            "with silence dilutes the speaker embedding it is meant to "
            "equalise. Choose {shortest:.2f}s or less.")

    print(f"\n  target {target:.2f}s - the shortest recording, so every file is "
          "cropped and none is padded")

    working = folder / WORKING
    working.mkdir(parents=True, exist_ok=True)
    manifest = {
        "_comment": ("Working references derived from the originals. The "
                     "originals are unmodified. Head-anchored crop to a common "
                     "duration, ending at a zero crossing; no padding, no fade, "
                     "no gain change."),
        "target_seconds": target,
        "method": "head-anchored crop to the shortest recording, cut at the "
                  "nearest zero crossing within "
                  f"{ZERO_CROSSING_WINDOW_MS:.0f} ms",
        "why_head_anchored": ("the s3gen reference reads the first 10 s and the "
                              "T3 prompt the first 6 s, so a head crop leaves "
                              "both identical to the original and changes only "
                              "the untruncated speaker embedding"),
        "references": {},
    }

    print()
    for key in neutral_ids():
        samples, rate, path, seconds = decoded[key]
        cropped, given_back = crop_at_zero_crossing(samples, rate, target)
        out_path = working / f"{key}.wav"
        write_wav(out_path, cropped, rate)
        final = len(cropped) / rate
        manifest["references"][key] = {
            "source": path.name,
            "source_seconds": round(seconds, 3),
            "working_seconds": round(final, 3),
            "removed_seconds": round(seconds - final, 3),
            "zero_crossing_samples_given_back": given_back,
            "sample_rate": rate,
        }
        print(f"  {key:<14}{seconds:>7.2f}s -> {final:>6.2f}s   "
              f"(-{seconds - final:.2f}s, zero-crossing trim "
              f"{given_back} samples)")

    (working / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # Point the experiment at the working copies. The originals stay mapped in
    # the manifest, so nothing about where they came from is lost.
    (folder / SOURCES).write_text(json.dumps({
        "_comment": ("Neutral id -> the file the experiment reads. These are "
                     f"the equal-length working copies in {WORKING}/; the "
                     "originals are unmodified and recorded in "
                     f"{WORKING}/MANIFEST.json."),
        "map": {key: f"{WORKING}/{key}.wav" for key in neutral_ids()},
    }, indent=2) + "\n", encoding="utf-8")

    spread = (max(entry["working_seconds"] for entry in manifest["references"].values())
              - min(entry["working_seconds"] for entry in manifest["references"].values()))
    print(f"\n  duration spread now {spread * 1000:.0f} ms")
    print(f"  wrote {working}/ and repointed {SOURCES}")
    print("  originals untouched\n")
    print("  Re-run the checker to confirm:")
    print(f"    python tools/check_reference_audio.py {folder}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
