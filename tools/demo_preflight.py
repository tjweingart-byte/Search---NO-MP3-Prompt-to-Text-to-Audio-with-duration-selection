"""What this machine will actually do if you start the demo now.

Every line here answers a question that otherwise gets answered by pressing
play and hearing the wrong thing - or worse, hearing something plausible. The
demo has three ways to look like it is working when it is not:

    no API key      the server serves a canned script that reads well and is
                    the same every time, so "testing the model" tests nothing
    no voice model  speech falls back to whatever the host OS has, or nothing
    empty cache     Explore is blank, and it can never fill itself: it replays
                    other people's episodes and refuses to generate

So this prints the state before the server starts, and says which of the five
tabs will be real. Exit code 0 means every tab will do what it claims; 1 means
something will be missing but the demo still runs; 2 means it is not worth
starting.

    python tools/demo_preflight.py            report and exit
    python tools/demo_preflight.py --quiet    exit code only
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache import build_cache
from config import describe_key, key_source, settings

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"


def cached_episodes() -> int:
    store = build_cache()
    if store is None:
        return -1
    try:
        return len(store.recent(200))
    except Exception:
        return -1


def voice_report() -> tuple[int, str]:
    """Real voices, and the engine that will speak.

    "debug" is not a voice. It is a placeholder tone the TTS layer falls back
    to when nothing is installed, and counting it would report a working demo
    that plays a beep - the exact silent success this project keeps paying for.
    """
    try:
        import tts

        report = tts.engine_report()
        engine = report.get("selected", "unknown")
        real = [v for v in tts.list_voices() if getattr(v, "engine", "") != "debug"]
        return len(real), engine
    except Exception as exc:
        return 0, f"unavailable ({exc})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    live = bool(settings.anthropic_api_key)
    voices, engine = voice_report()
    episodes = cached_episodes()
    worst = 0
    say = (lambda *a, **k: None) if args.quiet else print

    say(f"\n{BOLD}Demo preflight{RESET}")

    if live:
        say(f"  writing    {BOLD}live{RESET}  ·  {settings.model}  ·  "
            f"search {'on' if settings.enable_web_search else 'off by default'}")
        # Which file the key came from, because "a key is set" has been the
        # wrong answer to "is the right key set" more than once here.
        say(f"{DIM}             key {describe_key()} from {key_source()}{RESET}")
    else:
        say(f"  writing    {BOLD}CANNED{RESET} - no ANTHROPIC_API_KEY.")
        say(f"{DIM}             Every episode will be the same built-in sample script.")
        say(f"             You cannot judge the model from this. Set the key in .env.{RESET}")
        worst = max(worst, 2)

    if voices:
        say(f"  speech     {voices} voice(s)  ·  engine {engine}")
    else:
        say(f"  speech     {BOLD}NO VOICE MODEL{RESET} - engine is \"{engine}\", which "
            f"plays a placeholder tone,")
        say(f"{DIM}             not speech. Run: python setup_voices.py{RESET}")
        worst = max(worst, 2)

    if episodes < 0:
        say(f"  cache      {BOLD}off{RESET} (CACHE_ENABLED=0) - Explore reads from it "
            f"and will stay empty")
        worst = max(worst, 1)
    elif episodes == 0:
        say(f"  cache      {BOLD}empty{RESET} - explore has nothing to show, and by "
            f"design cannot fill itself")
        say(f"{DIM}             Fix: python tools/seed_demo.py{RESET}")
        worst = max(worst, 1)
    else:
        say(f"  cache      {episodes} episode(s) ready - explore has cards")

    say(f"\n{BOLD}What each tab will do{RESET}")
    fresh = "writes a real episode" if live else "plays the canned sample"
    say(f"  search     type anything, pick a length  ->  {fresh}")
    say(f"  myFAM      tap a tile  ->  {fresh}; rails rank the shared bank")
    say(f"  DailyFAM   starter mixes work cold; tap a mix to play it through")
    say(f"  explore    replays {'cached episodes' if episodes > 0 else 'nothing yet'} "
        f"- never generates, by design")
    say(f"  profile    shows what the event log holds, and nothing it does not")
    say("")
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
