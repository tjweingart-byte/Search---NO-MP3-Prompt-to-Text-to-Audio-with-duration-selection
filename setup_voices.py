"""Install the neural voices the app speaks with.

    python setup_voices.py            # install anything missing
    python setup_voices.py --list     # show what is already installed
    python setup_voices.py en_GB-alba-medium

Voices are stored **once per user**, in `~/.fam/voices` by default, not inside
the project folder. That is the whole point: a new version of the app finds the
voices already there instead of downloading tens of megabytes again. Override
the location with `FAM_VOICES_DIR` if you need to.

This is safe to run repeatedly - it only fetches what is actually absent, and
it first adopts anything an older project folder already downloaded.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import voice_store

# A small, deliberate set: two accents, two speakers, so the picker offers a
# real choice without a wall of near-identical options. "medium" sounds good
# without being slow on a CPU-only machine.
DEFAULT_VOICES = [
    "en_US-lessac-medium",
    "en_US-amy-medium",
    "en_GB-alba-medium",
    "en_GB-northern_english_male-medium",
]


def show(directory: Path) -> int:
    found = voice_store.installed(directory)
    if not found:
        print(f"No voices in {directory}\nRun: python setup_voices.py")
        return 1
    print(f"{len(found)} voice(s) in {directory}:")
    for path in found:
        print(f"  {path.stem:38} {path.stat().st_size / 1e6:5.1f} MB")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("voices", nargs="*", help="voice names; default is a curated set")
    parser.add_argument("--dir", default=None, help="override the voice directory")
    parser.add_argument("--list", action="store_true", help="show what is installed and exit")
    parser.add_argument(
        "--move-legacy",
        action="store_true",
        help="move rather than copy voices found in an old project folder",
    )
    args = parser.parse_args()

    directory = Path(args.dir).expanduser() if args.dir else voice_store.voices_dir(create=True)
    print(f"Voice store: {directory}\n")

    if args.list:
        return show(directory)

    # Reuse before downloading: an older project folder may already have these.
    adopted = voice_store.adopt_legacy(directory, move=args.move_legacy)
    if adopted:
        verb = "Moved" if args.move_legacy else "Copied"
        print(f"{verb} {len(adopted)} voice(s) from a previous version: {', '.join(adopted)}\n")

    have = {p.stem for p in voice_store.installed(directory)}
    wanted = args.voices or DEFAULT_VOICES
    missing = [name for name in wanted if name not in have]

    for name in wanted:
        if name in have:
            print(f"  {name}: already installed")

    if missing:
        try:
            from piper.download_voices import download_voice
        except ImportError:
            print("\npiper-tts is not installed. Run: pip install -r requirements.txt")
            return 1

        failures = []
        for name in missing:
            print(f"  {name}: downloading…", flush=True)
            try:
                download_voice(name, directory)
                print(f"  {name}: installed")
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"  {name}: FAILED - {type(exc).__name__}: {exc}")
                failures.append(name)
        if failures:
            print(
                "\nSome downloads failed. They come from huggingface.co; if that is\n"
                "blocked, download the .onnx and .onnx.json files by hand into\n"
                f"{directory} and they will be picked up."
            )

    found = voice_store.installed(directory)
    print(f"\n{len(found)} voice(s) available in {directory}")
    if not found:
        print("No voices installed - the app falls back to a lower-quality voice.")
        return 1
    print("Every version of the app will now find these automatically.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
