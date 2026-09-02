"""What does research actually cost, and what does it buy?

Answers the questions in the latency brief with measurements rather than
estimates, across repeated trials so the numbers have a spread rather than a
single lucky run.

    python tools/compare_search.py --dry-run          # the plan, spends nothing
    python tools/compare_search.py                    # the standard sweep
    python tools/compare_search.py --trials 5 --minutes 1,3,5
    python tools/compare_search.py --queries my_questions.txt

Every run is a real episode. The sweep prints its size and estimated cost and
asks before spending anything.

WHAT IS MEASURED, and how each maps to the brief
------------------------------------------------
research latency      TTFT(searched) - TTFT(same query, unsearched). The
                      searched call cannot write a word until its searches have
                      returned and been read, so the difference between the two
                      first-token times *is* the research cost, isolated from
                      model and prompt.

Claude-ready time     Time to the first complete sentence. That is the moment
                      the pipeline has something it can hand to the voice, so
                      it is exactly "enough information to start writing".

time to first audio   Claude-ready time plus the real synthesis time of that
                      first sentence, measured on whatever engine this machine
                      actually has. The number the listener feels.

source quality        Not scoreable by a machine, so this does not pretend to.
                      It records how many searches the model actually ran
                      (against the cap), how many distinct sources came back,
                      and every domain, so a person can judge them. The search
                      count also answers "is a cap of five doing anything" -
                      if the model never runs more than two, the cap is theatre.

The request is built with the app's own `_request_kwargs`, so what is timed is
the production request shape, not an approximation of it. Nothing in the
pipeline is imported for its side effects and nothing here writes to the cache.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import dataclasses
import os
import statistics
import sys
import time
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anthropic_client import build_async_client
from config import settings
import script_generator
from script_generator import ScriptGenerator, count_words, plan_episode

#: Questions whose answers move. The brief's own examples, trimmed to ones that
#: stay meaningful whenever this is run.
NEEDS_SEARCH = [
    "what happened with Nvidia today",
    "what is the latest on interest rates",
    "what are the newest AI model releases",
    "what major technology news happened today",
    "what is happening with the market this morning",
]

#: Questions a good answer does not need the web for. Included deliberately:
#: the cost of researching one of these is the cost of the heuristic guessing
#: wrong, and that number should be known.
NO_SEARCH = [
    "what is the NASDAQ",
    "how does a heat pump work",
    "why the Roman republic fell",
    "what habit research actually shows about lasting change",
    "why is the sky blue",
]

PRICES = {  # $ per million tokens (input, output)
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

_SENTENCE_END = script_generator._SENTENCE_END


@dataclasses.dataclass
class Run:
    query: str
    kind: str          # "needs-search" | "no-search"
    minutes: int
    depth: int         # 0 = no search; otherwise the max_uses cap
    trial: int
    first_token: float = 0.0
    first_sentence: float = 0.0
    first_audio: float = 0.0
    total: float = 0.0
    words: int = 0
    searches: int = 0
    sources: tuple = ()
    input_tokens: int = 0
    output_tokens: int = 0
    error: str = ""

    @property
    def cost(self) -> float:
        pin, pout = PRICES.get(settings.model, (0.0, 0.0))
        return self.input_tokens / 1e6 * pin + self.output_tokens / 1e6 * pout


def _domains(message) -> tuple:
    """Every distinct source the search actually returned, in order."""
    found: list[str] = []
    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", "") != "web_search_tool_result":
            continue
        content = getattr(block, "content", None)
        if not isinstance(content, list):  # an error object, not results
            continue
        for result in content:
            url = getattr(result, "url", "") or ""
            host = urlparse(url).netloc.replace("www.", "")
            if host and host not in found:
                found.append(host)
    return tuple(found)


def _search_count(message) -> int:
    """How many searches the model chose to run, against whatever the cap was."""
    return sum(
        1 for block in (getattr(message, "content", []) or [])
        if getattr(block, "type", "") == "server_tool_use"
        and getattr(block, "name", "") == "web_search"
    )


async def one_run(run: Run, engine) -> Run:
    """One episode, timed. Uses the app's own request shape."""
    patched = dataclasses.replace(
        settings,
        search_mode="never" if run.depth == 0 else "always",
        max_web_searches=max(1, run.depth),
    )
    script_generator.settings = patched
    try:
        plan = plan_episode(run.query, run.minutes, search=run.depth > 0)
        generator = ScriptGenerator()
        generator.client = build_async_client()
        kwargs = generator._request_kwargs(plan)

        buffer, sentences = "", []
        started = time.perf_counter()
        async with generator.client.messages.stream(**kwargs) as stream:
            async for delta in stream.text_stream:
                if not run.first_token:
                    run.first_token = time.perf_counter() - started
                buffer += delta
                match = _SENTENCE_END.search(buffer)
                while match:
                    sentence = buffer[: match.end()].strip()
                    buffer = buffer[match.end():]
                    if sentence and not run.first_sentence:
                        run.first_sentence = time.perf_counter() - started
                        # Real synthesis of the real first sentence: this is the
                        # moment sound would reach the listener.
                        synth_started = time.perf_counter()
                        try:
                            await engine.synth(sentence, settings.target_wpm)
                        except Exception:
                            pass  # a dead engine must not lose the model timings
                        run.first_audio = (
                            run.first_sentence + time.perf_counter() - synth_started
                        )
                    if sentence:
                        sentences.append(sentence)
                    match = _SENTENCE_END.search(buffer)
            final = await stream.get_final_message()

        run.total = time.perf_counter() - started
        if buffer.strip():
            sentences.append(buffer.strip())
        run.words = count_words(" ".join(sentences))
        run.searches = _search_count(final)
        run.sources = _domains(final)
        usage = getattr(final, "usage", None)
        run.input_tokens = getattr(usage, "input_tokens", 0) or 0
        run.output_tokens = getattr(usage, "output_tokens", 0) or 0
    except Exception as exc:  # one cell failing must not lose the sweep
        run.error = f"{type(exc).__name__}: {str(exc)[:120]}"
    finally:
        script_generator.settings = settings
    return run


def summarise(runs: list[Run], label: str) -> None:
    good = [r for r in runs if not r.error]
    if not good:
        print(f"  {label:<34} all trials failed: {runs[0].error if runs else ''}")
        return

    def spread(values):
        values = sorted(values)
        if len(values) == 1:
            return f"{values[0]:5.1f}"
        return f"{statistics.median(values):5.1f} ({min(values):.1f}-{max(values):.1f})"

    print(f"  {label:<34} "
          f"first token {spread([r.first_token for r in good]):>16}  "
          f"ready {spread([r.first_sentence for r in good]):>16}  "
          f"audio {spread([r.first_audio for r in good]):>16}  "
          f"{statistics.median([r.words for r in good]):>4.0f}w  "
          f"${statistics.median([r.cost for r in good]):.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--minutes", default="1,3,5")
    ap.add_argument("--depths", default="0,3",
                    help="search caps to compare; 0 means no search at all")
    ap.add_argument("--queries", help="file of questions, one per line")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, spend nothing")
    ap.add_argument("--yes", action="store_true", help="do not ask before spending")
    ap.add_argument("--csv", default="search_benchmark.csv")
    args = ap.parse_args()

    minutes = [int(m) for m in args.minutes.split(",") if m.strip()]
    depths = [int(d) for d in args.depths.split(",") if d.strip()]
    if args.queries:
        lines = [q.strip() for q in open(args.queries) if q.strip()]
        pairs = [(q, "custom") for q in lines]
    else:
        pairs = ([(q, "needs-search") for q in NEEDS_SEARCH]
                 + [(q, "no-search") for q in NO_SEARCH])

    plan = [
        Run(query=q, kind=kind, minutes=m, depth=d, trial=t)
        for q, kind in pairs for m in minutes for d in depths
        for t in range(1, args.trials + 1)
    ]
    est = len(plan) * 0.015
    # Runs are sequential on purpose: two episodes in flight would contend for
    # the same rate limit and each would time the other's queueing. Sequential
    # is slow and honest. Unsearched episodes take roughly 10-20s to write in
    # full, searched ones 30-50s.
    slow = sum(1 for r in plan if r.depth > 0)
    minutes_est = ((len(plan) - slow) * 15 + slow * 40) / 60

    print(f"\n{len(plan)} episodes  ·  {len(pairs)} questions × {len(minutes)} lengths "
          f"× {len(depths)} depths × {args.trials} trials")
    print(f"model {settings.model}  ·  rough cost ${est:.2f}  "
          f"(searched runs cost more; searches bill separately)")
    print(f"roughly {minutes_est:.0f} minutes, run one at a time so they do not "
          f"time each other's queueing")
    if minutes_est > 30:
        print(f"  smaller first pass:  --trials 2 --minutes 1,3   "
              f"(~{(len(pairs) * 2 * 2 * 2 * 27.5) / 60:.0f} min)")

    if args.dry_run:
        for kind in sorted({k for _, k in pairs}):
            print(f"\n  {kind}:")
            for q, k in pairs:
                if k == kind:
                    print(f"    {q}")
        print(f"\n  lengths {minutes}  depths {depths}  trials {args.trials}")
        print("\nNothing was run and nothing was spent. Drop --dry-run to do it.")
        return 0

    if not settings.anthropic_api_key:
        print("\nNo ANTHROPIC_API_KEY. Run: python setup_key.py", file=sys.stderr)
        return 2

    import tts

    engine = tts.build_engine()
    print(f"speech engine {engine.name} (time-to-first-audio is measured on it)\n")

    if not args.yes and sys.stdin.isatty():
        if input("Run it? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Nothing was run.")
            return 1

    done: list[Run] = []
    for index, run in enumerate(plan, 1):
        tag = "no search" if run.depth == 0 else f"depth {run.depth}"
        print(f"  [{index}/{len(plan)}] {run.minutes}min {tag} t{run.trial}  "
              f"{run.query[:44]:<44}", end="", flush=True)
        result = asyncio.run(one_run(run, engine))
        done.append(result)
        print(f" ready {result.first_sentence:5.1f}s  audio {result.first_audio:5.1f}s"
              f"{'  ' + result.error if result.error else ''}")

    with open(args.csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "kind", "minutes", "depth", "trial", "first_token_s",
                         "claude_ready_s", "first_audio_s", "total_s", "words",
                         "searches_run", "distinct_sources", "sources",
                         "input_tokens", "output_tokens", "cost_usd", "error"])
        for r in done:
            writer.writerow([r.query, r.kind, r.minutes, r.depth, r.trial,
                             f"{r.first_token:.2f}", f"{r.first_sentence:.2f}",
                             f"{r.first_audio:.2f}", f"{r.total:.2f}", r.words,
                             r.searches, len(r.sources), " ".join(r.sources),
                             r.input_tokens, r.output_tokens, f"{r.cost:.4f}", r.error])

    print(f"\n{'=' * 78}\nMedian seconds, with the range across trials\n{'=' * 78}")
    for kind in sorted({r.kind for r in done}):
        for m in minutes:
            for d in depths:
                cell = [r for r in done if r.kind == kind and r.minutes == m and r.depth == d]
                if cell:
                    tag = "no search" if d == 0 else f"depth {d}"
                    summarise(cell, f"{kind} {m}min {tag}")
        print()

    # The decomposition the brief asks for: what did research actually cost?
    print(f"{'=' * 78}\nResearch latency = searched first-token minus unsearched, same query\n{'=' * 78}")
    for kind in sorted({r.kind for r in done}):
        for m in minutes:
            base = [r.first_token for r in done
                    if r.kind == kind and r.minutes == m and r.depth == 0 and not r.error]
            for d in depths:
                if d == 0:
                    continue
                got = [r.first_token for r in done
                       if r.kind == kind and r.minutes == m and r.depth == d and not r.error]
                if base and got:
                    print(f"  {kind} {m}min depth {d}: "
                          f"+{statistics.median(got) - statistics.median(base):.1f}s")

    print(f"\n{'=' * 78}\nSource quality - to be judged, not scored\n{'=' * 78}")
    searched = [r for r in done if r.depth > 0 and not r.error]
    if searched:
        counts = [r.searches for r in searched]
        print(f"  searches actually run: median {statistics.median(counts):.0f}, "
              f"max {max(counts)} (cap was {max(depths)})")
        if max(counts) < max(depths):
            print(f"  the cap was never reached - raising it above {max(counts)} "
                  f"would change nothing")
        seen: dict[str, int] = {}
        for r in searched:
            for host in r.sources:
                seen[host] = seen.get(host, 0) + 1
        print(f"  {len(seen)} distinct sources across {len(searched)} researched episodes:")
        for host, n in sorted(seen.items(), key=lambda kv: -kv[1])[:25]:
            print(f"    {n:>3}x  {host}")
        print("\n  Score these yourself: primary sources and established outlets are "
              "what\n  makes an episode trustworthy, and no script can tell you that.")

    print(f"\nPer-run detail written to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
