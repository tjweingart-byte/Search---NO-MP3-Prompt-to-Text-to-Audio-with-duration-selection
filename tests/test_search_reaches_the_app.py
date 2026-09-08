"""The freshness heuristic has to survive the HTTP layer.

It did not, and the symptom was "the model doesn't seem to be searching the
web no matter what the prompt is". The heuristic itself was fine - and well
tested, which is the interesting part.

`plan_episode` consults it only when `search is None`, because an explicit
True/False is the listener's own choice and must win. But the endpoint
declared `search: bool = Query(False)`, so FastAPI turned an *omitted*
parameter into an explicit `False` before the planner ever saw it. The
browser never sends `search=` at all. So every episode the app ever produced
said "the listener asked for no research", and SEARCH_MODE=auto was dead code
in production while passing every test.

The tests missed it because they called `plan_episode(...)` directly with the
argument left off - which is not what the app does. **A default that is
correct in the function and wrong at the boundary is invisible to any test
that starts inside the boundary.** These start outside it.
"""
from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_mod  # noqa: E402


@pytest.fixture
def seen(monkeypatch):
    """Capture the plan the endpoint actually built."""
    captured = {}
    real = app_mod.plan_episode

    def spy(query, minutes, context="", search=None, cached_only=False, attachments=()):
        plan = real(query, minutes, context, search, cached_only, attachments)
        captured["search_arg"] = search
        captured["plan"] = plan
        return plan

    monkeypatch.setattr(app_mod, "plan_episode", spy)
    return captured


@pytest.fixture
def client(monkeypatch):
    # The pace exists to bound model spend, and these fire faster than a
    # person can. Without this the endpoint 429s before it builds a plan, and
    # every assertion below fails for a reason that has nothing to do with
    # search.
    monkeypatch.setattr(app_mod, "_rate_limit", lambda request: None)
    monkeypatch.setattr(app_mod, "_read_limit", lambda request: None)
    return TestClient(app_mod.app)


def _ask(client, **params):
    """Fire a real request and stop as soon as the plan has been built."""
    with client.stream("GET", "/api/audio", params={"minutes": 1, **params}) as r:
        r.read()


# --- an omitted parameter must stay omitted -----------------------------

def test_a_request_that_says_nothing_about_search_leaves_it_undecided(client, seen):
    """The whole bug in one line: `None` reaches the planner, not `False`."""
    _ask(client, q="what is the nasdaq")
    assert seen["search_arg"] is None, (
        "an omitted search parameter arrived as an explicit choice, which "
        "overrides SEARCH_MODE=auto and disables the heuristic entirely")


def test_a_question_about_a_moving_target_is_researched_through_the_api(client, seen):
    """End to end, the way the browser actually calls it."""
    _ask(client, q="latest news on the fed")
    assert seen["plan"].search is True


def test_an_evergreen_question_is_still_answered_from_memory(client, seen):
    """Over-triggering is cheap, not free."""
    _ask(client, q="how does a heat pump work")
    assert seen["plan"].search is False


@pytest.mark.parametrize("query", [
    "who runs OpenAI",
    "what is the price of bitcoin",
    "who won the super bowl",
    "is the merger still going ahead",
])
def test_the_widened_heuristic_reaches_the_api_too(client, seen, query):
    """None of these say "latest", and all of them go stale. These are the
    cases the wider heuristic was added for, and they were the ones most
    obviously not working."""
    _ask(client, q=query)
    assert seen["plan"].search is True, f"{query!r} was not researched"


# --- an explicit choice still wins --------------------------------------

@pytest.mark.parametrize("value, expected", [("1", True), ("0", False),
                                             ("true", True), ("false", False)])
def test_an_explicit_request_still_decides(client, seen, value, expected):
    """"Opt in" has to mean the listener can opt in - and out."""
    _ask(client, q="how does a heat pump work", search=value)
    assert seen["search_arg"] is expected
    assert seen["plan"].search is expected


# --- the other two entry points -----------------------------------------

def test_the_thread_lookup_keys_the_same_way_the_audio_did(client, seen):
    """/api/next reads the cache entry /api/audio wrote. If the two disagree
    about whether the episode was researched they disagree about the key, and
    Go Deeper silently never finds a thread."""
    client.get("/api/next", params={"q": "latest news on the fed", "minutes": 1})
    assert seen["search_arg"] is None


def test_the_script_endpoint_passes_the_flag_on_at_all(client, seen):
    """It accepted a `search` field and then dropped it on the floor, so
    `{"search": true}` did nothing at all."""
    client.post("/api/script", json={"query": "how does a heat pump work",
                                     "minutes": 1, "search": True})
    assert seen["search_arg"] is True
    assert seen["plan"].search is True
