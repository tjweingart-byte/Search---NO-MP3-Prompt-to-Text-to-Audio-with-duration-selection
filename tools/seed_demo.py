"""Fill a fresh install with real episodes, so every page has something to play.

Four of the five tabs work off state that a new install does not have:

    explore   reads finished scripts out of the shared cache and, by design,
              can never generate one. On a fresh database it is empty, and no
              amount of tapping will fill it.
    myFAM     ranks the shared topic bank by trending and by co-listener
              overlap. Both are global signals over an event log, so with no
              other listeners on record two of its three rails have nothing
              to rank and fall back to the bank.
    DailyFAM  works cold - the starter mixes are built in - but its tiles play
              instantly rather than waiting when their scripts are cached.
    profile   shows what the event log holds, which on a fresh install is
              nothing.

So a demo of the product needs a product that has been used. This writes that
history: it generates a handful of real episodes, puts them in the shared cache
where Explore reads from, and records other listeners having played them.

It costs one model call per episode and no speech synthesis at all - the script
is the expensive half, and seeding audio would be pointless when it is
regenerated from the script in milliseconds anyway.

    python tools/seed_demo.py --dry-run     what it would do, spends nothing
    python tools/seed_demo.py               eight episodes, three minutes each
    python tools/seed_demo.py --episodes 4 --minutes 2

Everything it writes is ordinary app state (scripts.db, myfam.db, social.db),
so a second run tops it up and deleting those files resets it.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache import build_cache, cache_key, is_shareable, ttl_for
from config import settings
from script_generator import ScriptGenerator, ScriptNotes, count_words, plan_episode
import social as social_mod
import topics as topics_mod

#: Other people, so co-listener overlap has someone to overlap with and an
#: Explore card can say who sent it. Invented listeners are fine; invented
#: *episodes* would not be, which is why every script below is really written.
LISTENERS = [
    ("demo-rachel", "Rachel Kim", "rachel"),
    ("demo-tom", "Tom Alvarez", "tomal"),
    ("demo-priya", "Priya Nair", "priya"),
]


def pick_topics(count: int) -> list:
    """Spread the seed across tags, so the myFAM rails differ from each other.

    Taking the first N of the bank would seed one corner of it, and then
    co-listener overlap and trending would rank the same handful.
    """
    by_tag: dict[str, list] = {}
    for topic in topics_mod.TOPIC_BANK:
        by_tag.setdefault(topic.tags[0] if topic.tags else "", []).append(topic)
    chosen: list = []
    while len(chosen) < count and any(by_tag.values()):
        for tag in sorted(by_tag):
            if by_tag[tag] and len(chosen) < count:
                chosen.append(by_tag[tag].pop(0))
    return chosen


def record_history(events, topic, minutes: int, thread: str, rng: random.Random) -> int:
    """Log other listeners playing this, at plausible times in the recent past.

    Timestamps matter: trending only counts the last few days, and every event
    decays, so stamping them all "now" would produce a feed that is technically
    populated and behaves nothing like a used app.
    """
    written = 0
    # Somebody always hears it. An episode in the cache that nobody played is a
    # real state, but it is invisible to trending, so seeding one is just a
    # model call spent on a card that cannot be ranked.
    certain = rng.choice(LISTENERS)[0]
    for user_id, _, _ in LISTENERS:
        if user_id != certain and rng.random() < 0.35:
            continue  # not everybody hears everything, or overlap is meaningless
        at = time.time() - rng.uniform(600, topics_mod.TRENDING_WINDOW * 0.9)
        events.record(topics_mod.Event(
            user_id=user_id, kind="play", topic_id=topic.id,
            text=topic.query, tags=topic.tags, at=at,
        ))
        written += 1
        if rng.random() < 0.7:
            events.record(topics_mod.Event(
                user_id=user_id, kind="complete", topic_id=topic.id,
                text=topic.query, tags=topic.tags, at=at + minutes * 60, thread=thread,
            ))
            written += 1
    return written


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--minutes", type=int, default=3)
    ap.add_argument("--search", action="store_true",
                    help="ground each episode in live sources (much slower)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be generated; spends nothing")
    ap.add_argument("--seed", type=int, default=7, help="for repeatable history")
    args = ap.parse_args()

    chosen = pick_topics(max(1, args.episodes))
    rng = random.Random(args.seed)

    print(f"Seeding {len(chosen)} episode(s) at {args.minutes} min, "
          f"model {settings.model}, search {'on' if args.search else 'off'}")

    if args.dry_run:
        for topic in chosen:
            print(f"  would write  {topic.id:<22} {topic.query}")
        print("\nNothing was generated and nothing was written. Drop --dry-run to do it.")
        return 0

    # Demo mode is a real trap here: with no key the app happily serves a canned
    # script, and a cache seeded with it looks exactly like a cache seeded with
    # real episodes until you press play. Refuse rather than fake it.
    if not settings.anthropic_api_key:
        print("\nNo ANTHROPIC_API_KEY, so there is no model to write anything.\n"
              "  Set it in .env or the environment and run this again.\n"
              "  (The server itself still runs without one - it falls back to a\n"
              "   canned demo script - but seeding the cache with that would put\n"
              "   fake episodes behind Explore, which is worse than an empty one.)",
              file=sys.stderr)
        return 2

    store = build_cache()
    if store is None:
        print("\nThe script cache is switched off (CACHE_ENABLED=0), and it is what\n"
              "Explore reads from. Turn it on and run this again.", file=sys.stderr)
        return 2

    events = topics_mod.EventStore()
    social = social_mod.SocialStore()
    for user_id, name, handle in LISTENERS:
        try:
            social.set_person(user_id, name, handle)
        except social_mod.SocialError:
            pass  # already there under a different id; not worth failing over

    generator = ScriptGenerator()
    written = 0
    for index, topic in enumerate(chosen, 1):
        if not is_shareable(topic.query):
            print(f"  [{index}/{len(chosen)}] skipped {topic.id} (not shareable)")
            continue
        plan = plan_episode(topic.query, args.minutes, search=args.search)
        notes = ScriptNotes()
        started = time.perf_counter()
        try:
            sentences = [s async for s in generator.stream_sentences(plan, notes)]
        except Exception as exc:  # a seed that half-fails must say which half
            print(f"  [{index}/{len(chosen)}] FAILED {topic.id}: {exc}", file=sys.stderr)
            continue
        elapsed = time.perf_counter() - started
        words = count_words(" ".join(sentences))
        key = cache_key(topic.query, args.minutes, searched=plan.search)
        store.put(key, sentences, ttl_for(topic.query), topic.query,
                  notes.thread, args.minutes)
        logged = record_history(events, topic, args.minutes, notes.thread, rng)
        written += 1
        print(f"  [{index}/{len(chosen)}] {topic.id:<22} {words:>4} words  "
              f"{elapsed:>5.1f}s  {logged} event(s)")

    if not written:
        print("\nNothing was written. Explore will still be empty.", file=sys.stderr)
        return 1

    # An echo changes what an Explore card says - "Rachel sent you this" rather
    # than "someone asked this" - and costs nothing, because the script it
    # points at already exists.
    for topic in chosen[:2]:
        user_id, name, _ = LISTENERS[0] if topic is chosen[0] else LISTENERS[1]
        try:
            social.echo(user_id, topic.query, topic.title, args.minutes)
            print(f"  echoed by {name}: {topic.title}")
        except social_mod.SocialError as exc:
            print(f"  could not echo {topic.id}: {exc}", file=sys.stderr)

    print(f"\n{written} episode(s) in the shared cache.")
    print("  explore   has cards now, and plays them without spending anything")
    print("  myFAM     Trending ranks these immediately. Your circle stays empty")
    print("            until you play one - it ranks overlap with you, and you")
    print("            have not overlapped with anyone yet. That is honest, not broken.")
    print("  DailyFAM  starter-mix tiles that hit a seeded topic start instantly")
    print("  search    still writes fresh episodes for anything not seeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
