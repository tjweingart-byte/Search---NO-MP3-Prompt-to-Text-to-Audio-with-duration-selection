#!/usr/bin/env python3
"""Does matching *near* neighbours actually raise the cache hit rate, and what
does it break on the way?

The script cache keys on an exact normalised token set, so two people asking
the same question in different words each pay for their own episode. `cache.
canonical_key` fixes that with a model call in front of every request;
`CACHE_VECTOR` fixes it with a vector computed at write time. This measures the
second one, because the reason to prefer it is a claim about numbers and claims
about numbers in this repo have a habit of being wrong.

Two things are being measured at once, and they pull against each other:

* **recall** - of the differently-worded questions that deserve an episode
  already in the cache, how many find it?
* **precision** - of the questions that deserve a *different* episode, how many
  are wrongly handed one anyway?

A false hit is far worse than a miss. A miss costs about a cent and a few
seconds; a false hit plays a confident, fluent answer to a question the
listener did not ask, and they will not forgive it. So the table below is read
by finding the highest recall at **zero** false hits, not the best trade.

    python tools/bench_vector_cache.py            # the sweep and the verdict
    python tools/bench_vector_cache.py --verbose  # every pair and its score
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cache as cache_mod  # noqa: E402
import embeddings  # noqa: E402
from config import settings  # noqa: E402

#: Questions that deserve the same episode. The first is what got generated
#: and cached; the rest are how other people would ask for it.
SAME = [
    ["Give me a recap of week 5 of the NFL season", "NFL week 5 recap",
     "recap: week 5, NFL", "what happened in NFL week 5"],
    ["why is the sky blue", "what makes the sky blue",
     "explain why the sky is blue"],
    ["how do vaccines work", "how does a vaccine work",
     "explain how vaccines work"],
    ["what is quantum computing", "explain quantum computing",
     "tell me about quantum computing"],
    ["the fall of the roman empire", "why did the roman empire fall",
     "how the roman empire fell"],
    ["how does inflation work", "explain inflation", "what is inflation"],
    ["climate change and the oceans", "how climate change affects the oceans",
     "effects of climate change on oceans"],
    ["who was Ada Lovelace", "tell me about Ada Lovelace",
     "Ada Lovelace biography"],
    ["how do black holes form", "the formation of black holes",
     "explain how a black hole forms"],
    ["what caused the 2008 financial crisis",
     "causes of the 2008 financial crisis",
     "explain the 2008 financial crash"],
    ["how does a nuclear reactor work", "explain nuclear reactors",
     "the way nuclear reactors work"],
    ["the history of the internet", "how the internet was invented",
     "internet history"],
    ["what is machine learning", "explain machine learning",
     "machine learning explained"],
    ["how do solar panels work", "explain how solar panels work",
     "the way solar panels generate electricity"],
    ["why do cats purr", "what makes cats purr", "the reason cats purr"],
    ["the rules of cricket explained", "how does cricket work",
     "explain cricket rules"],
    ["what is the Fermi paradox", "explain the Fermi paradox",
     "Fermi paradox explained"],
    ["how does GPS work", "explain GPS", "the way GPS works"],
    ["the causes of world war one", "what started world war one",
     "why did world war one begin"],
    ["how does sleep affect memory", "sleep and memory",
     "the effect of sleep on memory"],
]

#: Pairs that must never collapse. Each is close enough in wording to be a
#: plausible mistake and different enough that making it would be indefensible.
DIFFERENT = [
    ("NFL week 5 recap", "NFL week 6 recap"),
    ("why is the sky blue", "why is the ocean blue"),
    ("how tall is the eiffel tower", "how old is the eiffel tower"),
    ("what is quantum computing", "what is quantum entanglement"),
    ("how do vaccines work", "how do vaccine mandates work"),
    ("best pizza in new york", "best pizza in chicago"),
    ("the causes of world war one", "the causes of world war two"),
    ("how does a nuclear reactor work", "how does a nuclear bomb work"),
    ("what caused the 2008 financial crisis",
     "what caused the 1929 financial crisis"),
    ("how do solar panels work", "how do solar eclipses work"),
    ("the history of the internet", "the history of the telephone"),
    ("how does GPS work", "how does radar work"),
    ("what is machine learning", "what is deep learning"),
    ("who was Ada Lovelace", "who was Alan Turing"),
    ("how do black holes form", "how do stars form"),
    ("why do cats purr", "why do dogs bark"),
    ("apple stock price", "tesla stock price"),
    ("the rules of cricket explained", "the rules of baseball explained"),
    ("how does sleep affect memory", "how does caffeine affect memory"),
    ("explain inflation", "explain deflation"),
]

MINUTES = 3


def rows_for(queries):
    """`(key, query, vector)` triples, the shape `cache.best_match` reads."""
    return [
        (cache_mod.cache_key(q, MINUTES), q,
         embeddings.pack(embeddings.embed(cache_mod.normalize_query(q))))
        for q in queries
    ]


def guard_only() -> tuple[int, int]:
    """What the cheap guards find with the cosine switched off entirely.

    The control this bench exists to run. `comparable` is free - a token
    overlap, a digit comparison and a freshness check, no vector involved - so
    any recall it reaches on its own is recall the embedding did not earn.
    Printing the two side by side is the difference between "vectors work" and
    "something worked".
    """
    found = false = 0
    for group in SAME:
        for asked in group[1:]:
            found += not cache_mod.comparable(asked, group[0])
    for stored, asked in DIFFERENT:
        false += not cache_mod.comparable(asked, stored)
    return found, false


def exact_hits() -> tuple[int, int]:
    """The baseline: how far identical-key matching already gets."""
    hits = total = 0
    for group in SAME:
        stored = cache_mod.cache_key(group[0], MINUTES)
        for asked in group[1:]:
            total += 1
            hits += cache_mod.cache_key(asked, MINUTES) == stored
    return hits, total


def measure(threshold: float, overlap: float, verbose: bool = False) -> dict:
    object.__setattr__(settings, "cache_vector_overlap", overlap)

    found = missed = 0
    detail = []
    for group in SAME:
        rows = rows_for([group[0]])
        want = rows[0][0]
        for asked in group[1:]:
            match = cache_mod.best_match(asked, rows, threshold=threshold)
            if match and match[0] == want:
                found += 1
                detail.append(("hit ", match[1], asked, group[0], ""))
            else:
                missed += 1
                score = embeddings.cosine(
                    embeddings.embed(cache_mod.normalize_query(asked)),
                    embeddings.unpack(rows[0][2]),
                )
                why = cache_mod.comparable(asked, group[0]) or "below threshold"
                detail.append(("MISS", score, asked, group[0], why))

    # How close the nearest *wrong* answer got. Zero false hits is not the same
    # as being safe: a pair that scores 0.676 against a threshold of 0.68 is a
    # false hit waiting for a question phrased slightly differently. What makes
    # a setting genuinely safe is that every wrong pair is refused by a guard -
    # numbers, overlap, freshness - rather than by four thousandths of cosine.
    false_hits = 0
    nearest_wrong = 0.0
    for stored, asked in DIFFERENT:
        rows = rows_for([stored])
        match = cache_mod.best_match(asked, rows, threshold=threshold)
        if match:
            false_hits += 1
            detail.append(("WRONG", match[1], asked, stored, "served anyway"))
        elif not cache_mod.comparable(asked, stored):
            # Nothing but the threshold stood between this and being served.
            nearest_wrong = max(nearest_wrong, embeddings.cosine(
                embeddings.embed(cache_mod.normalize_query(asked)),
                embeddings.unpack(rows[0][2])))

    if verbose:
        for tag, score, asked, other, why in detail:
            print("  %-5s %.3f  %-46s <- %-40s %s"
                  % (tag, score, asked[:46], other[:40], why))

    return {"threshold": threshold, "overlap": overlap, "found": found,
            "missed": missed, "false": false_hits,
            #: None when every wrong pair was refused by a guard - the safe
            #: case, where the threshold has nothing riding on it.
            "margin": None if not nearest_wrong else threshold - nearest_wrong}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true",
                    help="print every pair, its score, and why it missed")
    ap.add_argument("--threshold", type=float,
                    help="measure one operating point instead of sweeping")
    ap.add_argument("--overlap", type=float, default=settings.cache_vector_overlap)
    args = ap.parse_args()

    described = embeddings.describe()
    print("Embedding backend: %s (%d dims)" % (described["backend"], described["dims"]))
    if not described["semantic"]:
        # The single most important line of output. Everything below is a
        # measurement of *lexical* similarity dressed as a vector, and reading
        # it as evidence about meaning is the mistake this header exists to
        # prevent.
        print("  NOT SEMANTIC. No sentence model is installed, so these are")
        print("  lexical vectors: word and character-ngram overlap. They cannot")
        print("  match 'car' to 'automobile' and never will. Numbers below are")
        print("  a floor on what a real model would do, not a estimate of it.")
        if described["error"]:
            print("  (%s)" % described["error"])
    print()

    hits, total = exact_hits()
    print("Baseline, exact keys only: %d/%d re-phrasings found their episode (%.0f%%)"
          % (hits, total, 100.0 * hits / total))
    print("%d pairs that must NOT collapse" % len(DIFFERENT))
    print()

    if args.threshold is not None:
        print("threshold %.2f, overlap floor %.2f" % (args.threshold, args.overlap))
        result = measure(args.threshold, args.overlap, verbose=True)
        print("\n  found %d/%d  false hits %d"
              % (result["found"], total, result["false"]))
        return 0

    print("  thresh  overlap   found/%d   recall   false   margin" % total)
    best = None
    for overlap in (0.0, 0.4, 0.5, 0.6):
        for threshold in [x / 100.0 for x in range(60, 96, 2)]:
            r = measure(threshold, overlap)
            safe = r["false"] == 0 and (r["margin"] is None or r["margin"] >= 0.05)
            flag = ""
            if safe and (best is None or r["found"] > best["found"]):
                best, flag = r, "  <- best safe"
            print("  %.2f    %.1f       %2d        %3.0f%%     %d     %s%s"
                  % (threshold, overlap, r["found"], 100.0 * r["found"] / total,
                     r["false"],
                     "guards" if r["margin"] is None else "%+.3f" % r["margin"],
                     flag))
        print()
    print("  margin: how far the threshold sat above the closest wrong answer.")
    print("  \"guards\" means every wrong pair was refused by a guard instead,")
    print("  which is the only column worth trusting on 20 negative pairs.")
    print()

    guarded, guard_false = guard_only()
    shipped = measure(settings.cache_vector_threshold, settings.cache_vector_overlap)

    print("Verdict")
    if best and best["found"]:
        print("  Highest recall with no wrong answer and nothing near one:")
        print("  threshold %.2f, overlap %.1f"
              % (best["threshold"], best["overlap"]))
        print("  %d of %d re-phrasings found an existing episode, up from %d."
              % (best["found"], total, hits))
        print("  That is %d episodes not written, at ~$0.0096 each."
              % (best["found"] - hits))
    else:
        print("  No setting found anything without also serving a wrong episode.")
        print("  Leave CACHE_VECTOR=0.")
    print()
    print("  Shipped default (threshold %.2f, overlap %.1f): %d/%d found, %d false."
          % (settings.cache_vector_threshold, settings.cache_vector_overlap,
             shipped["found"], total, shipped["false"]))
    print("  Control - the guards alone, cosine ignored: %d/%d found, %d false."
          % (guarded, total, guard_false))
    if guarded >= shipped["found"]:
        # The result that matters more than the recall number above it.
        print("  So at the safe operating point the vector adds NOTHING: the")
        print("  token-overlap guard finds everything the cosine does. That is")
        print("  what a lexical embedding is worth here. The mechanism is not")
        print("  the problem - the embedding is - and this line is how you will")
        print("  know a real sentence model has changed the answer.")

    started = time.perf_counter()
    rows = rows_for([g[0] for g in SAME] * 20)
    for _ in range(20):
        cache_mod.best_match("why is the sky blue", rows)
    per = (time.perf_counter() - started) / 20.0
    print()
    print("  Scan cost: %.2f ms over %d vectors (the whole miss-path overhead)."
          % (per * 1000.0, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
