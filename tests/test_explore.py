"""Explore: a vertical feed that replays other listeners' episodes.

The load-bearing property is negative: Explore must never cause a script to be
generated. That guarantee is tested against the pipeline, not the interface -
a rule that lives only in the frontend is one refactor from being broken
silently and expensively.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as appmod  # noqa: E402
import pipeline as pipeline_mod  # noqa: E402
from cache import MemoryScriptCache, SqliteScriptCache, cache_key  # noqa: E402
from pipeline import GenerationStats, NotCached, PodcastPipeline  # noqa: E402
from script_generator import plan_episode  # noqa: E402
from tts import DebugEngine  # noqa: E402


class CountingGenerator:
    """Fails the test loudly if anything asks it to write."""

    client = None

    def __init__(self):
        self.calls = 0

    async def stream_sentences(self, plan, notes=None):
        self.calls += 1
        yield "This should never have been generated."

    async def top_up(self, plan, spoken_so_far, words_needed):
        return
        yield ""  # pragma: no cover


def drain(pipeline, plan, stats=None):
    async def run():
        async for _ in pipeline.stream_pcm(plan, stats or GenerationStats()):
            pass

    asyncio.run(run())


# --- the guarantee --------------------------------------------------------


def test_a_replay_only_miss_refuses_instead_of_generating():
    gen = CountingGenerator()
    pipeline = PodcastPipeline(generator=gen, engine=DebugEngine(), cache=MemoryScriptCache())
    plan = plan_episode("nothing anyone has asked", 1, cached_only=True)

    with pytest.raises(NotCached):
        drain(pipeline, plan)
    assert gen.calls == 0, "Explore spent a model call - the whole point is that it cannot"


def test_a_replay_only_hit_plays_without_generating():
    cache = MemoryScriptCache()
    gen = CountingGenerator()
    pipeline = PodcastPipeline(generator=gen, engine=DebugEngine(), cache=cache)

    # One listener generates it the normal way...
    normal = plan_episode("why the sky is blue", 1)
    stats = GenerationStats()
    drain(pipeline, normal, stats)
    assert gen.calls == 1

    # ...and Explore replays it for everyone else, for free.
    drain(pipeline, plan_episode("why the sky is blue", 1, cached_only=True))
    assert gen.calls == 1, "a cached episode was regenerated for Explore"


def test_replay_only_with_no_cache_at_all_still_refuses():
    """Caching switched off must not quietly become 'generate everything'."""
    gen = CountingGenerator()
    pipeline = PodcastPipeline(generator=gen, engine=DebugEngine(), cache=None)
    with pytest.raises(NotCached):
        drain(pipeline, plan_episode("anything", 1, cached_only=True))
    assert gen.calls == 0


def test_the_flag_is_off_unless_asked_for():
    assert plan_episode("q", 1).cached_only is False


# --- what the feed is made of --------------------------------------------


def test_recent_returns_live_entries_newest_first(tmp_path):
    cache = SqliteScriptCache(str(tmp_path / "c.db"))
    cache.put("old", ["One."], 600, "an older question", "", 2)
    time.sleep(0.01)
    cache.put("new", ["Two."], 600, "a newer question", "", 3)
    queries = [e["query"] for e in cache.recent()]
    assert queries == ["a newer question", "an older question"]


def test_expired_entries_never_appear(tmp_path):
    cache = SqliteScriptCache(str(tmp_path / "c.db"))
    cache.put("gone", ["One."], -1, "stale question", "", 2)
    assert cache.recent() == []


def test_an_entry_with_no_duration_is_skipped_not_guessed(tmp_path):
    """Replaying a one-minute script as five minutes pads it with silence."""
    cache = SqliteScriptCache(str(tmp_path / "c.db"))
    cache.put("nomin", ["One."], 600, "a question", "")   # minutes defaults to 0
    assert cache.recent() == []


def test_generating_an_episode_records_its_duration_for_replay(tmp_path):
    cache = SqliteScriptCache(str(tmp_path / "c.db"))
    pipeline = PodcastPipeline(
        generator=CountingGenerator(), engine=DebugEngine(), cache=cache
    )
    drain(pipeline, plan_episode("a fresh question", 2))
    entry = cache.recent()[0]
    assert entry["minutes"] == 2 and entry["query"] == "a fresh question"


def test_personal_queries_never_reach_the_feed(tmp_path):
    """Explore shows strangers' episodes, so the privacy filter is load-bearing."""
    cache = SqliteScriptCache(str(tmp_path / "c.db"))
    pipeline = PodcastPipeline(
        generator=CountingGenerator(), engine=DebugEngine(), cache=cache
    )
    drain(pipeline, plan_episode("what do my lab results mean", 1))
    assert cache.recent() == [], "a personal query was published to Explore"


# --- the API --------------------------------------------------------------


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(appmod, "_rate_limit", lambda request: None)
    monkeypatch.setattr(appmod, "SCRIPT_CACHE", SqliteScriptCache(str(tmp_path / "e.db")))
    return TestClient(appmod.app)


def test_the_feed_is_empty_before_anyone_has_searched(client):
    assert client.get("/api/explore").json()["episodes"] == []


def test_the_feed_lists_what_has_been_generated(client):
    appmod.SCRIPT_CACHE.put("k", ["A sentence."], 600, "why volcanoes erupt", "how magma moves", 3)
    body = client.get("/api/explore").json()["episodes"]
    assert len(body) == 1
    assert body[0]["query"] == "why volcanoes erupt"
    assert body[0]["title"] == "Why volcanoes erupt"
    assert body[0]["minutes"] == 3
    assert body[0]["thread"] == "how magma moves"
    assert body[0]["age_seconds"] >= 0


def test_tapping_an_expired_card_is_a_409_not_a_generation(client, monkeypatch):
    gen = CountingGenerator()
    monkeypatch.setattr(appmod, "DEMO_MODE", False)
    monkeypatch.setattr(
        appmod, "_make_pipeline",
        lambda voice=None: PodcastPipeline(
            generator=gen, engine=DebugEngine(), cache=appmod.SCRIPT_CACHE, voice=voice
        ),
    )
    res = client.get("/api/audio?q=long+gone&minutes=1&fmt=pcm&cached_only=true")
    assert res.status_code == 409
    assert gen.calls == 0
    assert "Explore" in res.json()["error"]
