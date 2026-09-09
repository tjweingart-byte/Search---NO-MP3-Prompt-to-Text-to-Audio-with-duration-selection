"""Near matching: what it finds, and everything it must refuse to find.

The cache key is an exact normalised token set, so two people asking the same
question in different words each pay for their own episode. `CACHE_VECTOR`
lets a lookup consider neighbours as well.

The tests below are lopsided on purpose. One of them checks that a re-phrasing
finds its episode; the rest check that things which are *not* the same question
are refused. That is the right shape for this feature: a miss costs a cent and
a few seconds, while a false hit plays a fluent, confident answer to a question
the listener did not ask - which fails the first duty of an episode before a
word of it is wrong.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cache as cache_mod  # noqa: E402
import embeddings  # noqa: E402
from cache import MemoryScriptCache, SqliteScriptCache, cache_key, key_bucket  # noqa: E402
from config import settings  # noqa: E402


@pytest.fixture
def near_matching():
    """Turn near matching on for one test, at the shipped thresholds.

    `object.__setattr__` because Settings is frozen - which is the right thing
    for it to be, and means a test that flips a setting has to put it back.
    """
    object.__setattr__(settings, "cache_vector", True)
    try:
        yield settings
    finally:
        object.__setattr__(settings, "cache_vector", False)


def store(store_, query, minutes=3, sentences=("one.", "two.")):
    key = cache_key(query, minutes)
    bucket = key_bucket(minutes)
    store_.put(key, list(sentences), 600, query, "", minutes, bucket)
    return key, bucket


@pytest.fixture(params=["memory", "sqlite"])
def backend(request, tmp_path):
    """Both stores, because they are two implementations of one promise.

    The memory cache is what most tests and single-worker development run on;
    sqlite is what a deployment runs on. A near match found by one and not the
    other would be a difference nobody would notice until it mattered.
    """
    if request.param == "memory":
        return MemoryScriptCache()
    return SqliteScriptCache(str(tmp_path / "near" / "scripts.db"))


# --- what it should find ---------------------------------------------------

def test_a_rephrasing_finds_the_episode_an_exact_key_would_miss(backend, near_matching):
    key, bucket = store(backend, "why do cats purr")
    assert backend.get(cache_key("what makes cats purr", 3)) is None, (
        "the exact key already matched, so this test is not testing anything"
    )
    near = backend.nearest(bucket, "what makes cats purr")
    assert near and near[0] == key


def test_a_spelled_number_reaches_the_episode_cached_with_a_digit(backend, near_matching):
    """"week five" and "week 5" are the same week. The lexical key cannot see
    that; folding numerals to digits before embedding can."""
    key, bucket = store(backend, "NFL week 5 recap")
    near = backend.nearest(bucket, "recap of week five in the NFL")
    assert near and near[0] == key


# --- what it must refuse ---------------------------------------------------

@pytest.mark.parametrize("stored, asked", [
    ("NFL week 5 recap", "NFL week 6 recap"),
    ("the causes of world war one", "the causes of world war two"),
    ("what caused the 2008 financial crisis", "what caused the 1929 financial crisis"),
])
def test_a_different_number_is_a_different_episode(backend, near_matching, stored, asked):
    """The failure a vector is worst at. "week 5" and "week 6" score high and
    share every other word; spelled-out numbers hide from a digit test unless
    they are folded first."""
    _, bucket = store(backend, stored)
    assert backend.nearest(bucket, asked) is None


@pytest.mark.parametrize("stored, asked", [
    ("why is the sky blue", "why is the ocean blue"),
    ("what is machine learning", "what is deep learning"),
    ("how do vaccines work", "how do vaccine mandates work"),
    ("how does GPS work", "how does radar work"),
    ("the history of the internet", "the history of the telephone"),
])
def test_a_similar_shape_is_not_the_same_question(backend, near_matching, stored, asked):
    _, bucket = store(backend, stored)
    assert backend.nearest(bucket, asked) is None


def test_a_durable_question_never_reuses_a_current_one(backend, near_matching):
    """Freshness is not a property of the subject, and an episode written to
    describe how something is *now* is the wrong answer to a question about how
    it works in general - even when every word matches."""
    _, bucket = store(backend, "the latest iphone")
    assert backend.nearest(bucket, "the iphone") is None


def test_a_different_duration_is_a_different_episode(backend, near_matching):
    """A 3-minute script is written differently from a 10-minute one, not cut
    down from it, so the two are not interchangeable however alike the
    questions are."""
    store(backend, "why do cats purr", minutes=3)
    other = key_bucket(10)
    assert backend.nearest(other, "what makes cats purr") is None


def test_off_by_default(backend):
    """Nothing near-matches unless someone turned it on."""
    assert settings.cache_vector is False
    _, bucket = store(backend, "why do cats purr")
    assert backend.nearest(bucket, "what makes cats purr") is None


# --- the pieces the decision rests on --------------------------------------

def test_the_guards_say_why_they_refused():
    """A heuristic nobody can see the workings of is a heuristic nobody can
    tune - the same reason `research_reason` returns a string."""
    assert "numbers" in cache_mod.comparable("NFL week 5", "NFL week 6")
    assert cache_mod.comparable("why do cats purr", "what makes cats purr") == ""


def test_vectors_are_the_same_everywhere():
    """Written by one worker, compared by another. A vector that depended on a
    seed or on insertion order would produce matches that made sense to
    nobody."""
    a = embeddings.embed("why is the sky blue")
    b = embeddings.embed("why is the sky blue")
    assert a == b
    assert embeddings.unpack(embeddings.pack(a)) == pytest.approx(a, abs=1e-6)


def test_an_unrelated_question_scores_near_zero():
    assert embeddings.cosine(
        embeddings.embed("blue sky"), embeddings.embed("vaccines work")
    ) < 0.1


def test_a_vector_from_another_space_is_a_miss_not_a_crash():
    """Backend and width are part of the bucket, so this should not arise -
    but a cache file is not worth a migration framework, and a stale row must
    cost a regeneration rather than an exception."""
    assert embeddings.cosine([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0


def test_the_backend_in_force_is_reported_honestly():
    """Lexical vectors dressed as semantic ones would make every measurement
    taken with them wrong in the same direction."""
    described = embeddings.describe()
    assert described["backend"] in ("hashing", "onnx")
    assert described["semantic"] is (described["backend"] == "onnx")


def test_the_bucket_changes_with_the_vector_space(monkeypatch):
    """Switching embedding backend must retire the old vectors rather than
    compare coordinates that no longer mean the same thing."""
    before = key_bucket(3)
    monkeypatch.setattr(embeddings, "space", lambda: "something-else:512")
    assert key_bucket(3) != before


def test_a_personal_episode_is_not_findable_by_being_near_one(near_matching):
    """An attachment makes an episode the listener's own. `_bucket` returns ""
    for one, and an empty bucket can never be scanned - the same guarantee the
    empty cache key gives, extended to the new lookup."""
    from pipeline import PodcastPipeline
    from script_generator import plan_episode

    pipe = PodcastPipeline(cache=MemoryScriptCache())
    plan = plan_episode("my lab results", 3)
    plan.attachments = [{"id": "x"}]
    assert pipe._bucket(plan) == ""


# --- the whole way through -------------------------------------------------

def _episode(pipeline, plan, stats):
    import asyncio

    async def run():
        total = 0
        async for chunk in pipeline.stream_pcm(plan, stats):
            total += len(chunk)
        return total

    return asyncio.run(run())


def test_a_second_listener_phrased_it_differently_and_still_paid_nothing(near_matching):
    """The claim the whole feature rests on, measured end to end rather than
    at the store: two people, two phrasings that share no cache key, one model
    call between them."""
    from pipeline import GenerationStats, PodcastPipeline
    from script_generator import plan_episode
    from tts import DebugEngine

    from test_pipeline import CountingGenerator

    store_ = MemoryScriptCache()
    gen = CountingGenerator()

    first = GenerationStats()
    audio_first = _episode(
        PodcastPipeline(generator=gen, engine=DebugEngine(), cache=store_),
        plan_episode("why do cats purr", 3), first,
    )
    spent = gen.calls

    second = GenerationStats()
    audio_second = _episode(
        PodcastPipeline(generator=gen, engine=DebugEngine(), cache=store_),
        plan_episode("what makes cats purr", 3), second,
    )

    assert first.cache == "miss" and first.match == ""
    assert second.cache == "hit" and second.match == "near"
    assert second.match_score >= settings.cache_vector_threshold
    assert gen.calls == spent, "the second listener spent a model call"
    assert second.script == first.script
    assert audio_second == audio_first


def test_explore_never_gets_handed_a_neighbour(near_matching):
    """Explore offers an episode whose title the listener has already read.
    Playing a near neighbour instead would answer a question they did not tap,
    so `cached_only` refuses rather than approximates."""
    from pipeline import GenerationStats, NotCached, PodcastPipeline
    from script_generator import plan_episode
    from tts import DebugEngine

    from test_pipeline import CountingGenerator

    store_ = MemoryScriptCache()
    _, bucket = store(store_, "why do cats purr")
    assert store_.nearest(bucket, "what makes cats purr"), (
        "near matching cannot find it either, so this test proves nothing"
    )

    plan = plan_episode("what makes cats purr", 3)
    plan.cached_only = True
    pipe = PodcastPipeline(generator=CountingGenerator(), engine=DebugEngine(), cache=store_)
    with pytest.raises(NotCached):
        _episode(pipe, plan, GenerationStats())
