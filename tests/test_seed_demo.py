"""Seeding a demo: what it writes, and the one thing it must refuse to write.

The seeder exists so that Explore, the myFAM rails and the profile have real
state to show. The failure worth testing for is not a crash - it is a seed that
*looks* successful and has put canned demo output behind Explore, where it is
indistinguishable from a real episode until someone presses play.
"""
from __future__ import annotations

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import seed_demo  # noqa: E402
import topics as T  # noqa: E402


def test_the_seed_is_spread_across_the_bank():
    """Eight episodes from one corner would leave the rails ranking one tag."""
    chosen = seed_demo.pick_topics(8)
    assert len(chosen) == 8
    assert len({t.id for t in chosen}) == 8, "no topic seeded twice"
    first_tags = {t.tags[0] for t in chosen if t.tags}
    assert len(first_tags) >= 4, f"only {first_tags} - too narrow to rank"


def test_asking_for_more_than_the_bank_holds_stops_at_the_bank():
    chosen = seed_demo.pick_topics(500)
    assert len(chosen) == len(T.TOPIC_BANK)


def test_history_is_spread_through_the_recent_past(tmp_path):
    """All-at-once timestamps produce a feed that is populated and behaves
    nothing like a used app: trending counts a window, and events decay."""
    store = T.EventStore(str(tmp_path / "myfam.db"))
    topic = T.TOPIC_BANK[0]
    rng = random.Random(1)
    written = sum(
        seed_demo.record_history(store, t, 3, "", rng)
        for t in T.TOPIC_BANK[:6]
    )
    assert written > 0
    stamps = []
    for user_id, _, _ in seed_demo.LISTENERS:
        stamps += [e.at for e in store.for_user(user_id)]
    assert len(stamps) == written
    assert len(set(stamps)) > 1, "every event landed at the same moment"
    span = max(stamps) - min(stamps)
    assert span > 3600, "history covers under an hour; trending cannot rank it"


def test_history_gives_co_listeners_something_to_overlap_on(tmp_path):
    store = T.EventStore(str(tmp_path / "myfam.db"))
    rng = random.Random(4)
    for topic in seed_demo.pick_topics(8):
        seed_demo.record_history(store, topic, 3, "", rng)
    heard = {
        user_id: {e.topic_id for e in store.for_user(user_id)}
        for user_id, _, _ in seed_demo.LISTENERS
    }
    assert all(heard.values()), "a listener with no history is not a co-listener"
    pairs = [(a, b) for a in heard for b in heard if a < b]
    assert any(heard[a] & heard[b] for a, b in pairs), "nobody overlaps with anybody"


def test_it_refuses_to_seed_from_the_canned_demo_script(monkeypatch, capsys):
    """No key means DemoGenerator, and a cache full of that is worse than an
    empty one: Explore would show episodes that are not episodes."""
    import dataclasses

    monkeypatch.setattr(seed_demo, "settings",
                        dataclasses.replace(seed_demo.settings, anthropic_api_key=""))
    monkeypatch.setattr(sys, "argv", ["seed_demo.py", "--episodes", "2"])
    import asyncio

    code = asyncio.run(seed_demo.main())
    assert code == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_a_dry_run_spends_nothing_and_writes_nothing(monkeypatch, capsys):
    """The one command that is safe to run without reading the source first."""
    import asyncio
    import dataclasses

    monkeypatch.setattr(seed_demo, "settings",
                        dataclasses.replace(seed_demo.settings, anthropic_api_key=""))

    def explode(*a, **k):
        raise AssertionError("a dry run built a cache")

    monkeypatch.setattr(seed_demo, "build_cache", explode)
    monkeypatch.setattr(sys, "argv", ["seed_demo.py", "--dry-run", "--episodes", "3"])
    assert asyncio.run(seed_demo.main()) == 0
    out = capsys.readouterr().out
    assert "would write" in out
    assert "Nothing was generated" in out
