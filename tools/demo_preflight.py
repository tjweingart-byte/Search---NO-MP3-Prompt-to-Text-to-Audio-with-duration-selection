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

import credentials
import research
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


def where_a_key_could_come_from() -> list[str]:
    """The lines printed instead of "set the key in .env".

    Every machine that reaches this point is a machine somebody is about to
    paste a key into, and the .env they paste it into is gone the next time the
    app is unpacked. So the second line is the one that ends the loop rather
    than repeating it.
    """
    report = credentials.report()
    if report["state"] == "failed":
        return [f"{credentials.PROVIDER_VAR} is set and failed: {report['detail']}",
                "Fix the provider - the app is doing what it was told."]
    if report["configured"]:
        return [f"{credentials.PROVIDER_VAR} is set ({', '.join(report['provider'])}) "
                f"but returned no ANTHROPIC_API_KEY."]
    return ["Once on this machine:   python setup_key.py",
            "Once for every machine: set FAM_SECRETS. See CREDENTIALS.md."]


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
            f"search {settings.search_mode}")
        # The two settings that put seconds in front of the first word. Both
        # are off by default and both have been turned on by accident, by an
        # .env copied from an example that disagreed with the code.
        if settings.search_mode == "always":
            say(f"  {BOLD}SEARCH_MODE=always{RESET} - every episode waits 10-25s "
                f"before the first word,")
            say(f"{DIM}             including the ones that did not need it. "
                f"SEARCH_MODE=auto reads the question.{RESET}")
            worst = max(worst, 1)
        # Which file the key came from, because "a key is set" has been the
        # wrong answer to "is the right key set" more than once here.
        say(f"{DIM}             key {describe_key()} from {key_source()}{RESET}")
    else:
        say(f"  writing    {BOLD}CANNED{RESET} - no ANTHROPIC_API_KEY.")
        say(f"{DIM}             Every episode will be the same built-in sample script.")
        say(f"             You cannot judge the model from this.{RESET}")
        for line in where_a_key_could_come_from():
            say(f"{DIM}             {line}{RESET}")
        worst = max(worst, 2)

    # A provider that is set and broken is worth saying even when a key was
    # found some other way: it means the next machine will not find one.
    secrets = credentials.report()
    if secrets["state"] == "failed":
        say(f"  secrets    {BOLD}{credentials.PROVIDER_VAR} FAILED{RESET} - "
            f"{secrets['detail']}")
        say(f"{DIM}             This machine got its key elsewhere or not at all. "
            f"A machine with only{RESET}")
        say(f"{DIM}             the provider set would be starting with nothing.{RESET}")
        worst = max(worst, 1)
    elif secrets["configured"]:
        say(f"  secrets    {', '.join(secrets['provider'])} · "
            f"supplied {', '.join(secrets['supplied']) or 'nothing'}")

    if voices:
        say(f"  speech     {voices} voice(s)  ·  engine {engine}")
    else:
        say(f"  speech     {BOLD}NO VOICE{RESET} - engine is \"{engine}\", which "
            f"plays a placeholder tone,")
        say(f"{DIM}             not speech. There is no second engine to fall back "
            f"to: Chatterbox needs")
        say(f"             a GPU and requirements-chatterbox.txt. "
            f"See RUNPOD_PRODUCTION.md.{RESET}")
        worst = max(worst, 2)

    # Research is the fourth way this can look like it is working. The default
    # backend is Exa, which needs an optional package and a second key; without
    # them a question the heuristic marks time-sensitive fails outright rather
    # than quietly answering from memory. That is deliberate, but it is not
    # something to discover by asking about today's news.
    research_ok, research_detail = research.diagnose()
    backend = settings.research_backend
    if backend == "claude":
        say(f"  research   {BOLD}claude{RESET} - the model searches during the call "
            f"(costs 10-25s before the first word)")
    elif research_ok:
        say(f"  research   {BOLD}exa{RESET} - retrieves first, then Claude writes "
            f"from the packet")
    else:
        say(f"  research   {BOLD}UNAVAILABLE{RESET} - backend is \"exa\" but "
            f"{research_detail}.")
        say(f"{DIM}             A time-sensitive question ('latest', 'today', a score) "
            f"will FAIL,")
        say(f"             not fall back. Everything else is unaffected.")
        say(f"             Fix: pip install -r requirements-exa.txt and set "
            f"EXA_API_KEY,")
        say(f"             or set RESEARCH_BACKEND=claude.{RESET}")
        worst = max(worst, 1)

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
    researched = ("  ·  a time-sensitive question will FAIL (research unavailable)"
                  if settings.research_backend == "exa" and not research_ok else "")
    say(f"  search     type anything, pick a length  ->  {fresh}{researched}")
    say(f"  myFAM      tap a tile  ->  {fresh}; rails rank the shared bank")
    say(f"  DailyFAM   starter mixes work cold; tap a mix to play it through")
    say(f"  explore    replays {'cached episodes' if episodes > 0 else 'nothing yet'} "
        f"- never generates, by design")
    say(f"  profile    shows what the event log holds, and nothing it does not")
    say("")
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
