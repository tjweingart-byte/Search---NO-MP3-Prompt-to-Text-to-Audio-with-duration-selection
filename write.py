"""Read a briefing instead of listening to it.

    python write.py "recap of the tour championship" --minutes 3
    python write.py "why is the sky blue" --minutes 1 --no-search
    python write.py "the fed decision" --minutes 5 --model claude-sonnet-5

The script is the product. Audio is just how it gets delivered, and listening to
a five minute episode to judge one prompt change is a slow way to work. This
prints the script, how long it will actually run, and how much it cost, in a few
seconds.

Judge it on: does the first sentence tell you something true and specific? Would
any of it survive being written about a different topic? Does it end because it
is finished, or because it ran out of room?

Read the last two sentences hardest. They are the part with no test behind it -
the rules say land it and stop, never tease, no rhetorical question, no recap -
and a prompt rule is only a request until you have seen the model obey it. The
predicted follow-up is printed under the script: it is never spoken, and the
script is barred from gesturing at it, so it is checked separately.
"""
from __future__ import annotations

import argparse
import asyncio
import time

from anthropic_client import build_async_client
from config import settings
from script_generator import ScriptGenerator, ScriptNotes, count_words, plan_episode

PRICES = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--minutes", type=int, default=3)
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-search", action="store_true")
    ap.add_argument("--prompt", action="store_true", help="print the brief that was sent")
    args = ap.parse_args()

    import dataclasses

    import script_generator

    overrides = {}
    if args.model:
        overrides["model"] = args.model
    if args.no_search:
        overrides["enable_web_search"] = False
    if overrides:
        script_generator.settings = dataclasses.replace(settings, **overrides)

    plan = plan_episode(args.query, args.minutes)
    active = script_generator.settings

    if args.prompt:
        print(script_generator.build_prompt(plan))
        print("\n" + "=" * 72 + "\n")

    print(f'"{args.query}"  ·  {plan.minutes} min  ·  {active.model}'
          f'  ·  search {"off" if not active.enable_web_search else "on"}\n')

    generator = ScriptGenerator()
    generator.client = build_async_client()

    started = time.perf_counter()
    first_at = None
    sentences = []
    notes = ScriptNotes()
    async for sentence in generator.stream_sentences(plan, notes):
        if first_at is None:
            first_at = time.perf_counter() - started
        sentences.append(sentence)

    elapsed = time.perf_counter() - started
    text = " ".join(sentences)
    words = count_words(text)
    spoken_minutes = words / active.target_wpm

    print(text)
    print("\n" + "-" * 72)
    print(f"  {words} words  ->  {spoken_minutes:.1f} min spoken "
          f"(you asked for {plan.minutes})")
    print(f"  first sentence after {first_at or 0:.1f}s, finished in {elapsed:.1f}s")
    # Never spoken. Printed because half of the ending rewrite lives here: the
    # suggestion moved out of the script and into this line, so judging the
    # script alone would only see half of the change.
    if notes.thread:
        print(f'  predicted follow-up (never spoken): "{notes.thread}"')
    else:
        print("  NOTE: no predicted follow-up. Go Deeper falls back without one.")
    if words < plan.word_budget * 0.8:
        print("  NOTE: came in short. That is allowed now - it should mean it ran "
              "out of things worth saying, not that it gave up.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
