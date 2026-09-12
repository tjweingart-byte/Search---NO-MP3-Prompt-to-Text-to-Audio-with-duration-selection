"""Every episode is researched. The question does not get a vote.

This is a reversal, and the log line that caused it is worth keeping in the
file it is tested in::

    SEARCH no '49ers game last night' - nothing in it reads as time-sensitive;
    answering from what the model knows

That was production, on Render, on a question about a game played the previous
evening. The heuristic was not broken so much as mispriced: it was written when
research meant Anthropic's server-side `web_search`, which front-loads 10-25
seconds, and at that price guessing was worth it. With `RESEARCH_BACKEND=exa`
retrieval costs about half a second, so every question the guess gets wrong is
answered from memory that may be a year stale and every question it gets right
saves nothing a listener can hear. A keyword list can always be widened by one
more word, and the next question it misses is already written somewhere.

So `SEARCH_MODE=always` is the production default, and these tests exist to
stop it drifting back. `auto` and `never` are kept - `write.py` offline,
`tools/compare_search.py`, a deployment with no Exa key - and are proved here
to be reachable *only* by asking for them explicitly.

Nothing here needs a key, `exa_py`, or the network: the retrieval is stubbed at
`research_mod.retrieve`, which is the seam production actually calls.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_mod  # noqa: E402
import config as config_mod  # noqa: E402
import research as research_mod  # noqa: E402
import script_generator as sg  # noqa: E402
from script_generator import ScriptGenerator, build_prompt, plan_episode  # noqa: E402

#: The question from the Render log, and four more that no keyword list would
#: ever flag. These are the ones the old default got wrong.
TIMELESS = [
    "what is the NASDAQ",
    "how does a heat pump work",
    "why the Roman republic fell",
    "explain compound interest",
    "what makes sourdough rise",
]

#: Questions the old heuristic did flag. They must still be researched - a fix
#: that only moved which half was wrong would pass a one-sided test.
CURRENT = [
    "49ers game last night",
    "latest news on the fed",
    "what happened today in golf",
    "breaking news about the election",
]


# --- the default, in the two places it can drift ----------------------------

def test_the_shipped_default_is_always():
    """Read from a fresh Settings rather than the import-time singleton, so an
    environment variable set by another test cannot make this pass."""
    assert config_mod.Settings().search_mode == "always"


@pytest.mark.parametrize("query", TIMELESS + CURRENT)
def test_every_question_plans_research(query):
    assert plan_episode(query, 3).search is True, (
        f"{query!r} was planned without research")


@pytest.mark.parametrize("query", TIMELESS + CURRENT)
def test_every_question_plans_research_through_the_api(client, seen, query):
    """The planner is not the boundary. A `bool = Query(False)` default once
    turned every omitted parameter into an explicit "no research" and the
    search mode became dead code in production while passing every test -
    so this asks the way the browser asks."""
    _ask(client, q=query)
    assert seen["search_arg"] is None, "the endpoint decided on the listener's behalf"
    assert seen["plan"].search is True, f"{query!r} was not researched through /api/audio"


def test_the_log_line_that_caused_this_cannot_be_emitted(caplog):
    """A regression test on the exact symptom, not only on the cause."""
    with caplog.at_level(logging.INFO, logger=sg.log.name):
        for query in TIMELESS + CURRENT:
            plan_episode(query, 3)
    text = caplog.text
    assert "answering from what the model knows" not in text
    assert "SEARCH no" not in text
    assert "SEARCH yes" in text


# --- research is actually invoked, not merely flagged -----------------------

@pytest.mark.parametrize("query", ["what is the NASDAQ", "49ers game last night"])
def test_the_retrieval_really_runs_for_both_kinds_of_question(monkeypatch, query):
    """`plan.search is True` is a flag. This follows it to the call.

    The stub sits on `research_mod.retrieve` - the function `ScriptGenerator.
    research` calls - so the production path is exercised rather than a mock of
    it."""
    calls = []

    class Packet:
        context = "SOURCE 1 / Title: x / Key evidence: y"
        searches = 1
        cost = 0.0

        def __bool__(self):
            return True

        def as_dict(self):
            return {"context": self.context}

    async def fake_retrieve(q):
        calls.append(q)
        return Packet()

    monkeypatch.setattr(research_mod, "retrieve", fake_retrieve)
    plan = plan_episode(query, 3)
    researched = asyncio.run(ScriptGenerator(api_key="").research(plan))

    assert calls == [query], "the episode was written without retrieving anything"
    assert researched.evidence == Packet.context


def test_a_research_failure_surfaces_instead_of_silently_answering(monkeypatch):
    """The one thing worse than waiting for research is being told you got it.

    A backend that cannot run must reach the listener as a sentence they can
    act on. Falling back to model knowledge here would restore exactly the
    behaviour this change removed, and would do it invisibly."""
    async def broken(q):
        raise research_mod.ResearchUnavailable("EXA_API_KEY is not set")

    monkeypatch.setattr(research_mod, "retrieve", broken)
    plan = plan_episode("what is the NASDAQ", 3)

    with pytest.raises(research_mod.ResearchUnavailable):
        asyncio.run(ScriptGenerator(api_key="").research(plan))

    said = app_mod.friendly_error(
        research_mod.ResearchUnavailable("EXA_API_KEY is not set"))
    assert "EXA_API_KEY is not set" in said, "the remedy was thrown away"


# --- the non-production modes are still reachable, and only on request ------

@pytest.mark.parametrize("query", TIMELESS)
def test_auto_still_reads_the_question_when_asked_for(monkeypatch, query):
    """The heuristic is kept, not deleted - `write.py` offline and
    `tools/compare_search.py` both need it. It just is not the default."""
    monkeypatch.setattr(
        sg, "settings", dataclasses.replace(sg.settings, search_mode="auto"))
    assert plan_episode(query, 3).search is False


@pytest.mark.parametrize("query", TIMELESS[:2] + CURRENT[:2])
def test_never_still_turns_it_off_entirely(monkeypatch, query):
    monkeypatch.setattr(
        sg, "settings", dataclasses.replace(sg.settings, search_mode="never"))
    assert plan_episode(query, 3).search is False


@pytest.mark.parametrize("value, expected", [("0", False), ("1", True)])
def test_an_explicit_request_still_wins_over_the_default(client, seen, value, expected):
    """Opt-out has to mean something, or `search=0` is a lie in the API docs."""
    _ask(client, q="49ers game last night", search=value)
    assert seen["plan"].search is expected


# --- fixtures ---------------------------------------------------------------

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
    # These fire faster than a person can, and the pace would 429 the request
    # before it ever built a plan - a failure with nothing to do with search.
    monkeypatch.setattr(app_mod, "_rate_limit", lambda request: None)
    monkeypatch.setattr(app_mod, "_read_limit", lambda request: None)
    return TestClient(app_mod.app)


def _ask(client, **params):
    """Fire a real request and stop as soon as the plan has been built."""
    with client.stream("GET", "/api/audio", params={"minutes": 1, **params}) as r:
        r.read()


# --- a tool is not an instruction -------------------------------------------
#
# The always-on default made an older, quieter bug reachable on every episode
# instead of the few the keyword list flagged. `_request_kwargs` attaches the
# web_search tool whenever an episode is researched and no evidence packet came
# back - the `claude` backend, or Exa returning nothing usable - and
# `build_prompt` said nothing about it. So the model was handed a capability it
# was never asked to use, wrote from memory, and reported honestly that it had
# nothing: "I don't have any information on the 49ers game last night. I can't
# confirm the score, the opponent, or the plays."
#
# That is the silent bypass in its last hiding place. These pin the fix.

def _no_packet_plan(query="49ers game last night"):
    """A researched episode that got no evidence - the reachable failure."""
    plan = plan_episode(query, 3)
    assert plan.search is True
    assert not plan.evidence, "this fixture is about the empty-packet path"
    return plan


def test_an_episode_with_no_packet_still_gets_the_search_tool():
    kwargs = ScriptGenerator.__new__(ScriptGenerator)._request_kwargs(_no_packet_plan())
    assert [t["name"] for t in kwargs.get("tools", [])] == ["web_search"]


def test_and_is_actually_told_to_use_it():
    """The half that was missing. Attaching a tool is not asking for research."""
    prompt = build_prompt(_no_packet_plan())
    assert "web search tool" in prompt, (
        "the model was handed a search tool and never asked to search")
    assert "Search first" in prompt


def test_the_exact_answer_production_gave_is_named_as_unacceptable():
    """Pinned to the symptom. A model that will not search should at least have
    been told that refusing to look is the one answer that is not allowed."""
    prompt = build_prompt(_no_packet_plan())
    assert "I can't confirm" in prompt
    assert "don't have that information" in prompt


def test_an_episode_that_has_evidence_is_not_told_to_search():
    """The tool and the packet are alternatives - never both, or the model
    searches on top of what it was handed and the episode is unattributable."""
    plan = dataclasses.replace(_no_packet_plan(), evidence="SOURCE 1\nTitle: x")
    prompt = build_prompt(plan)
    assert "web search tool" not in prompt
    kwargs = ScriptGenerator.__new__(ScriptGenerator)._request_kwargs(plan)
    assert "tools" not in kwargs


def test_an_unresearched_episode_is_told_nothing_about_searching():
    prompt = build_prompt(plan_episode("how does a heat pump work", 3, search=False))
    assert "web search tool" not in prompt


def test_the_claude_backend_reaches_the_same_instruction(monkeypatch):
    """`research()` returns the plan untouched on `claude` - the model does its
    own looking - so that backend lands on this path for *every* episode."""
    monkeypatch.setattr(
        sg, "settings", dataclasses.replace(sg.settings, research_backend="claude"))
    plan = plan_episode("what is the NASDAQ", 3)
    returned = asyncio.run(ScriptGenerator(api_key="").research(plan))
    assert returned.evidence == "", "the claude backend must retrieve nothing"
    assert "web search tool" in build_prompt(returned)


# --- the server can say which code it is running ----------------------------

def test_health_reports_the_commit_and_where_search_mode_came_from(client):
    """"Is the fix deployed?" was unanswerable from outside the server, so it
    got answered by reasoning about what should have happened instead."""
    body = client.get("/api/health").json()
    assert body["build"]["commit"], "no way to tell which code is serving"
    assert body["search_mode"] == "always"
    assert body["search_mode_source"] == "config.py default"


def test_health_says_when_an_env_var_is_overriding_the_default(client, monkeypatch):
    """An env var beats the code default silently and outlives any number of
    pushes. That is a different claim from "the default was changed"."""
    monkeypatch.setenv("SEARCH_MODE", "auto")
    body = client.get("/api/health").json()
    assert body["search_mode_source"] == "SEARCH_MODE env var"
