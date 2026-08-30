"""Settle the model question on your own queries instead of guessing.

Generates the same briefing with several models and reports, for each: time to
first word, total time, how close it landed to the word budget, real token cost
from the API's own usage numbers, and the script itself so you can read them
side by side.

    python compare_models.py "recap of week 5 of the NFL season" --minutes 3
    python compare_models.py "why is the sky blue" --minutes 1 --no-search

Quality is the one thing that cannot be measured from the outside, so this
prints the scripts and lets you judge. Everything else it measures exactly.
"""
from __future__ import annotations

import argparse
import asyncio
import time

import anthropic

from config import settings
from script_generator import SYSTEM_PROMPT, build_prompt, count_words, plan_episode

# $ per million tokens (input, output), from the published price list.
PRICES = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


async def run_one(client, model: str, plan, use_search: bool) -> dict:
    kwargs = {
        "model": model,
        "max_tokens": settings.max_output_tokens,
        "system": SYSTEM_PROMPT,
        "output_config": {"effort": settings.effort},
        "messages": [{"role": "user", "content": build_prompt(plan)}],
    }
    if use_search:
        kwargs["tools"] = [
            {"type": "web_search_20260209", "name": "web_search",
             "max_uses": settings.max_web_searches}
        ]

    started = time.perf_counter()
    first_token_at = None
    text = ""
    async with client.messages.stream(**kwargs) as stream:
        async for delta in stream.text_stream:
            if first_token_at is None:
                first_token_at = time.perf_counter() - started
            text += delta
        final = await stream.get_final_message()

    usage = final.usage
    price_in, price_out = PRICES.get(model, (0.0, 0.0))
    # Cached reads are billed at a tenth; creation at 1.25x. Close enough for
    # a comparison, and exact for the common uncached case.
    # These fields are None rather than 0 when unused, so coerce each one.
    billed_in = (usage.input_tokens or 0) + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
    cost = billed_in / 1e6 * price_in + usage.output_tokens / 1e6 * price_out

    return {
        "model": model,
        "first_word": first_token_at or 0.0,
        "total": time.perf_counter() - started,
        "words": count_words(text),
        "input_tokens": billed_in,
        "output_tokens": usage.output_tokens,
        "cost": cost,
        "text": text.strip(),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--minutes", type=int, default=3)
    ap.add_argument("--models", default="claude-opus-5,claude-sonnet-5,claude-haiku-4-5")
    ap.add_argument("--no-search", action="store_true", help="skip live web search")
    ap.add_argument("--full", action="store_true", help="print whole scripts, not excerpts")
    args = ap.parse_args()

    plan = plan_episode(args.query, args.minutes)
    use_search = settings.enable_web_search and not args.no_search
    client = anthropic.AsyncAnthropic()

    print(f'\nQuery   : "{args.query}"')
    print(f"Length  : {plan.minutes} min  (budget {plan.min_words}-{plan.max_words} words)")
    print(f"Search  : {'on' if use_search else 'off'}\n")

    results = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        try:
            results.append(await run_one(client, model, plan, use_search))
        except Exception as exc:
            print(f"  {model}: FAILED - {type(exc).__name__}: {exc}")

    if not results:
        return

    header = f"{'model':18} {'1st word':>9} {'total':>8} {'words':>7} {'on budget':>11} {'cost':>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        inside = plan.min_words <= r["words"] <= plan.max_words
        print(f"{r['model']:18} {r['first_word']:>8.1f}s {r['total']:>7.1f}s "
              f"{r['words']:>7} {('yes' if inside else 'NO'):>11} {'$%.4f' % r['cost']:>9}")

    cheapest = min(results, key=lambda r: r["cost"])
    dearest = max(results, key=lambda r: r["cost"])
    if cheapest is not dearest and cheapest["cost"]:
        print(f"\n{dearest['model']} costs {dearest['cost'] / cheapest['cost']:.1f}x "
              f"more than {cheapest['model']} for this episode.")

    print("\n" + "=" * 72)
    print("Read these and decide. Cost and speed are measured; quality is yours to judge.")
    for r in results:
        print("\n" + "=" * 72)
        print(f"{r['model']}  ({r['words']} words)")
        print("=" * 72)
        print(r["text"] if args.full else r["text"][:900] + ("…" if len(r["text"]) > 900 else ""))


if __name__ == "__main__":
    asyncio.run(main())
