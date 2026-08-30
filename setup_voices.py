"""Install the neural voices the app ships with.

    python setup_voices.py            # the default set
    python setup_voices.py --list     # what is installed now
    python setup_voices.py en_GB-alba-medium

Voices live in `voices/` inside the project, not in a system directory, so the
app sounds the same wherever it runs - a laptop, a container, a server. That is
the whole point: espeak only exists if someone installed it, and macOS `say`
does not exist on Linux at all, so an app relying on either sounds different
once deployed.

Models are a few tens of MB each and are not committed to git; run this once
after cloning, and as a build step when deploying.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# A small, deliberate set: two accents, two speakers, so the picker offers a
# real choice without a wall of near-identical options. "medium" is the quality
# tier that sounds good without being slow on a CPU-only box.
DEFAULT_VOICES = [
    "en_US-lessac-medium",
    "en_US-amy-medium",
    "en_GB-alba-medium",
    "en_GB-northern_english_male-medium",
]


def installed(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob("*.onnx")) if directory.is_dir() else []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("voices", nargs="*", help="voice names; default is a curated set")
    parser.add_argument("--dir", default="voices", help="where to install (default: voices/)")
    parser.add_argument("--list", action="store_true", help="show what is installed and exit")
    args = parser.parse_args()

    directory = Path(args.dir)

    if args.list:
        found = installed(directory)
        if not found:
            print(f"No voices in {directory}/. Run: python setup_voices.py")
            return 1
        print(f"{len(found)} voice(s) in {directory}/:")
        for path in found:
            size = path.stat().st_size / 1e6
            print(f"  {path.stem:38} {size:5.1f} MB")
        return 0

    try:
        from piper.download_voices import download_voice
    except ImportError:
        print("piper-tts is not installed. Run: pip install -r requirements.txt")
        return 1

    directory.mkdir(parents=True, exist_ok=True)
    wanted = args.voices or DEFAULT_VOICES
    already = {p.stem for p in installed(directory)}

    failures = []
    for name in wanted:
        if name in already:
            print(f"  {name}: already installed")
            continue
        print(f"  {name}: downloading…", flush=True)
        try:
            download_voice(name, directory)
            print(f"  {name}: installed")
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  {name}: FAILED - {type(exc).__name__}: {exc}")
            failures.append(name)

    found = installed(directory)
    print(f"\n{len(found)} voice(s) available in {directory}/")
    if failures:
        print(
            "\nSome downloads failed. They come from huggingface.co; if you are behind\n"
            "a proxy or firewall that blocks it, download the .onnx and .onnx.json\n"
            f"files by hand into {directory}/ and they will be picked up."
        )
    if not found:
        print("\nNo voices installed - the app will fall back to a lower-quality voice.")
        return 1
    print("Restart the server and they will appear in the voice picker.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
