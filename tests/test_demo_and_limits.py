"""Three faults found by running the app on a real machine, pinned so they stay fixed.

All three came from one session: a DailyFAM tile that played a script about how
episode generation works instead of the episode, and then 429s across ordinary
navigation.

1. `python app.py` never read .env, so a perfectly good key was invisible and
   the app fell back to the canned sample script.
2. Demo mode wrote that canned script into the *shared* cache, under whatever
   the listener actually asked, where Explore and every other listener would be
   served it later as a real episode.
3. One 3-second-per-client pace was applied to all eighteen endpoints, and the
   interface fires several cheap reads whenever a tab opens - so correct use
   answered itself with "Slow down a moment, then try again."
"""
from __future__ import annotations

import asyncio
import dataclasses
import importlib
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as appmod  # noqa: E402
import config as config_mod  # noqa: E402
import pipeline as pipeline_module  # noqa: E402
from cache import MemoryScriptCache, cache_key  # noqa: E402
from demo_script import DemoGenerator  # noqa: E402
from pipeline import GenerationStats, PodcastPipeline  # noqa: E402
from script_generator import plan_episode  # noqa: E402
from tests.test_pipeline import DebugEngine  # noqa: E402


# --- 1. the key must be found however the server is started ----------------

def test_dotenv_is_read_so_python_app_py_finds_the_key(tmp_path, monkeypatch):
    """run.sh and demo.sh source .env; app.py's own __main__ block did not."""
    env = tmp_path / ".env"
    env.write_text(
        '# a comment\n'
        'ANTHROPIC_API_KEY="sk-quoted-value"\n'
        'export MODEL=claude-haiku-4-5\n'
        'EMPTY=\n'
    )
    monkeypatch.setattr(config_mod.pathlib.Path, "resolve",
                        lambda self: tmp_path / "config.py", raising=False)
    monkeypatch.delenv("FAM_IGNORE_DOTENV", raising=False)
    for name in ("ANTHROPIC_API_KEY", "MODEL"):
        monkeypatch.delenv(name, raising=False)

    config_mod._load_dotenv()
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-quoted-value", "quotes must be stripped"
    assert os.environ["MODEL"] == "claude-haiku-4-5", "`export FOO=bar` must work"


def test_a_real_environment_variable_beats_the_file(tmp_path, monkeypatch):
    """`MODEL=x python app.py` must still override .env."""
    (tmp_path / ".env").write_text("MODEL=claude-haiku-4-5\n")
    monkeypatch.setattr(config_mod.pathlib.Path, "resolve",
                        lambda self: tmp_path / "config.py", raising=False)
    monkeypatch.delenv("FAM_IGNORE_DOTENV", raising=False)
    monkeypatch.setenv("MODEL", "claude-sonnet-5")
    config_mod._load_dotenv()
    assert os.environ["MODEL"] == "claude-sonnet-5"


def test_the_suite_itself_ignores_dotenv():
    """conftest sets this. Without it, a developer's key changes the tests."""
    assert os.environ.get("FAM_IGNORE_DOTENV") == "1"


# --- 2. demo mode may read the cache, never write it ------------------------

def _run(pipeline, plan, stats):
    async def go():
        async for _ in pipeline.stream_pcm(plan, stats):
            pass
    asyncio.run(go())


def test_demo_mode_never_puts_the_canned_script_in_the_shared_cache():
    """The sample script describes the audio pipeline. It does not answer the
    question, so caching it hands a briefing about streaming to the next person
    who asks about habits - for the whole TTL, and after a key is added."""
    cache = MemoryScriptCache()
    plan = plan_episode("what habit research actually shows about lasting change", 1)
    pipe = PodcastPipeline(generator=DemoGenerator(), engine=DebugEngine(),
                           cache=cache, cache_writes=False)
    _run(pipe, plan, GenerationStats())

    assert cache.get(cache_key(plan.query, plan.minutes)) is None
    assert cache.recent(10) == [], "demo mode put an episode on Explore"


def test_demo_mode_still_reads_the_cache():
    """Explore replays, and replaying needs no credentials at all - so reads
    must survive the fix that stopped the writes."""
    cache = MemoryScriptCache()
    plan = plan_episode("how reusable rockets changed spaceflight", 1)
    real = ["A first sentence.", "A second sentence."]
    cache.put(cache_key(plan.query, plan.minutes), real, 3600, plan.query, "", 1)

    stats = GenerationStats()
    pipe = PodcastPipeline(generator=DemoGenerator(), engine=DebugEngine(),
                           cache=cache, cache_writes=False)
    _run(pipe, plan, stats)
    assert stats.cache == "hit"
    assert stats.script == real, "a cached real episode must play, not the sample"


def test_the_app_builds_its_demo_pipeline_with_writes_off():
    """The guarantee has to hold where the app actually constructs it."""
    if not appmod.DEMO_MODE:
        pytest.skip("this build has credentials, so demo mode is not in play")
    assert appmod._make_pipeline().cache_writes is False


# --- 3. the pace belongs on generation, not on opening a tab ----------------

@pytest.fixture
def client(monkeypatch, tmp_path):
    import topics as T

    monkeypatch.setattr(appmod, "EVENTS", T.EventStore(str(tmp_path / "e.db")))
    appmod._read_hits.clear()
    appmod._last_request.clear()
    return TestClient(appmod.app)


def test_opening_a_tab_does_not_rate_limit_itself(client):
    """These are the calls the interface makes when a tab opens. Under the old
    limiter the second one 429'd, which is what the listener saw."""
    for path in ("/api/topics", "/api/mixes?user=me", "/api/myfam?user=me",
                 "/api/explore?user=me", "/api/profile?user=me", "/api/voices"):
        assert client.get(path).status_code == 200, f"{path} was throttled"


def test_a_burst_of_cheap_reads_is_allowed_then_bounded(client, monkeypatch):
    """A ceiling, not a pace: bursts pass, a hammering script does not."""
    monkeypatch.setattr(appmod, "settings",
                        dataclasses.replace(appmod.settings, read_limit_per_window=8))
    codes = [client.get("/api/topics").status_code for _ in range(12)]
    assert codes[:8] == [200] * 8, "a normal burst was throttled"
    assert 429 in codes[8:], "the ceiling never applied"


def test_generation_is_still_paced(client, monkeypatch):
    """The expensive path keeps its limit - that is what it was for."""
    monkeypatch.setattr(appmod, "_read_limit", lambda request: None)
    first = client.post("/api/script", json={"query": "why the sky is blue", "minutes": 1})
    second = client.post("/api/script", json={"query": "why the sea is blue", "minutes": 1})
    assert first.status_code == 200
    assert second.status_code == 429, "two generations back to back went through"


def test_explore_replays_are_not_paced(client):
    """A replay-only request cannot spend a model call, so pacing it only stops
    someone swiping the feed at a normal speed."""
    codes = [
        client.get("/api/audio?q=nothing+cached+here&minutes=1&cached_only=1").status_code
        for _ in range(3)
    ]
    assert 429 not in codes, "swiping Explore hit the generation pace"
    assert set(codes) == {409}, "replay-only must fail as a cache miss, not a throttle"
