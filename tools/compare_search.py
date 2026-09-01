"""What does research actually buy, and how much of it is worth paying for?

Two questions, one experiment. Runs the same query at several search depths and
reports, for each: how long until the first sentence, how long in total, how
many words, and the script itself - because the only real measure of "better"
here is reading them side by side.

    python tools/compare_search.py "what is the NASDAQ"
    python tools/compare_search.py "latest news on the fed" --depths 0,1,3,5
    python tools/compare_search.py "why founders took back control" --scripts

Depth 0 means no search at all: the model answers from what it knows. Depths
1-5 cap `max_uses` on the web-search tool.

**A cap is not a target.** The model uses as many searches as it judges it
needs, up to the cap, so raising the cap does not buy more research - it
raises the ceiling on how much the model *may* do, and each one it takes costs
seconds before the first word and money on top of the episode. That is the
thing this tool measures: how many the model actually takes at each cap, and
whether the extra ones changed the script.

Needs an API key. Every depth is a real episode: at 3 minutes each run is
roughly a cent plus the searches, so the default sweep costs a few cents.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
import script_generator
from script_generator import ScriptGenerator, ScriptNotes, count_words, plan_episode


async def one_run(query: str, minutes: int, depth: int) -> dict:
    """One episode at one search depth."""
    patched = dataclasses.replace(
        settings,
        search_mode="never" if depth == 0 else "always",
        max_web_searches=max(1, depth),
    )
    script_generator.settings = patched

    plan = plan_episode(query, minutes, search=depth > 0)
    generator = ScriptGenerator()
    notes = ScriptNotes()
    sentences: list[str] = []
    started = time.perf_counter()
    first_at = None
    try:
        async for sentence in generator.stream_sentences(plan, notes):
            if first_at is None:
                first_at = time.perf_counter() - started
            sentences.append(sentence)
    except Exception as exc:  # noqa: BLE001 - one depth failing must not stop the sweep
        return {"depth": depth, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        script_generator.settings = settings

    text = " ".join(sentences)
    return {
        "depth": depth,
        "first_at": first_at or 0.0,
        "total": time.perf_counter() - started,
        "words": count_words(text),
        "script": text,
        "thread": notes.thread,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("query")
    ap.add_argument("--minutes", type=int, default=3)
    ap.add_argument("--depths", default="0,1,3,5",
                    help="search caps to compare; 0 means no search")
    ap.add_argument("--scripts", action="store_true",
                    help="print each script in full, for reading side by side")
    args = ap.parse_args()

    if not settings.anthropic_api_key:
        print("No ANTHROPIC_API_KEY. Run: python setup_key.py", file=sys.stderr)
        return 2

    try:
        depths = [int(d) for d in args.depths.split(",") if d.strip() != ""]
    except ValueError:
        print("--depths takes a comma-separated list of integers", file=sys.stderr)
        return 2

    print(f'"{args.query}"  ·  {args.minutes} min  ·  {settings.model}\n')
    results = []
    for depth in depths:
        label = "no search" if depth == 0 else f"up to {depth} search{'es' if depth > 1 else ''}"
        print(f"  {label:<18} …", end="", flush=True)
        result = asyncio.run(one_run(args.query, args.minutes, depth))
        results.append(result)
        if "error" in result:
            print(f" FAILED  {result['error']}")
        else:
            print(f"\r  {label:<18} first word {result['first_at']:>6.1f}s  ·  "
                  f"total {result['total']:>6.1f}s  ·  {result['words']:>4} words")

    good = [r for r in results if "error" not in r]
    if len(good) > 1:
        base = good[0]
        print(f"\n  Cost of research, against {'no search' if base['depth'] == 0 else 'depth ' + str(base['depth'])}:")
        for r in good[1:]:
            delta = r["first_at"] - base["first_at"]
            print(f"    depth {r['depth']}: {delta:+.1f}s before the first word")
        print("\n  Whether that bought anything is a reading judgement, not a number.")
        print("  Run with --scripts and read the openings against each other.")

    if args.scripts:
        for r in good:
            head = "NO SEARCH" if r["depth"] == 0 else f"UP TO {r['depth']} SEARCHES"
            print(f"\n{'=' * 72}\n{head}  ({r['words']} words, "
                  f"first word after {r['first_at']:.1f}s)\n{'=' * 72}\n")
            print(r["script"])
            if r["thread"]:
                print(f'\n  predicted follow-up (never spoken): "{r["thread"]}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
