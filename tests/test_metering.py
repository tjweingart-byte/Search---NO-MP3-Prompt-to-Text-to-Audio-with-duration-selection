"""What a listener cost, recorded once and answerable later.

The provider bills this organisation, not this listener. So every question
about pricing, per-user limits and abuse reduces to "which listener produced
which request", and if that is not answered at the moment of spend it cannot be
answered at all. These pin the answer: that it is recorded, that it is costed
from real usage rather than estimated from text, that the report separates what
was billed from what is assumed, and that the shape of the distribution
survives - because the mean is the number most likely to be quoted and least
likely to be true.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import metering  # noqa: E402


@pytest.fixture
def store(tmp_path):
    return metering.MeterStore(str(tmp_path / "metering.db"))


def usage(model="claude-sonnet-5", inp=1000, out=1000, cache_read=0,
          cache_write=0, **kw):
    """One episode's usage. `kw` sets the non-token fields (audio_seconds, ...)."""
    u = metering.Usage(model=model)
    if inp or out or cache_read or cache_write:
        u.add_model_call(model, {"input_tokens": inp, "output_tokens": out,
                                 "cache_read_input_tokens": cache_read,
                                 "cache_creation_input_tokens": cache_write})
    for name, value in kw.items():
        setattr(u, name, value)
    return u


# --- the cost model -------------------------------------------------------

def test_tokens_are_costed_at_the_published_rate():
    """Sonnet 5 is $2/$10 per million. One million of each is twelve dollars."""
    cost = metering.price_of(usage(inp=1_000_000, out=1_000_000))
    assert cost.claude_input == pytest.approx(2.00)
    assert cost.claude_output == pytest.approx(10.00)


def test_cached_input_is_cheaper_and_writing_the_cache_is_dearer():
    """Both are billed against the input rate and neither at it. Costing a
    cache read as full input would overstate a re-asked question by 10x."""
    cost = metering.price_of(usage(inp=0, out=0, cache_read=1_000_000,
                                   cache_write=1_000_000, model="claude-sonnet-5"))
    assert cost.cache_read == pytest.approx(2.00 * 0.1)
    assert cost.cache_write == pytest.approx(2.00 * 1.25)


def test_an_unpriced_model_is_flagged_rather_than_counted_as_free():
    """The failure this project keeps paying for is a number that looks like an
    answer. A model with no rate card must not average in at zero."""
    cost = metering.price_of(usage(model="claude-from-the-future"))
    assert cost.priced is False
    assert cost.claude_input == 0.0


def test_a_cache_hit_costs_only_what_it_actually_spent():
    """No model call at all - the whole point of the shared script cache. It
    still costs GPU seconds, and those are still real."""
    u = metering.Usage(model="", audio_seconds=180, cache_hit=True)
    cost = metering.price_of(u)
    assert cost.priced is True, "no model was called, so nothing was mispriced"
    assert cost.claude_input == cost.claude_output == 0.0
    assert cost.gpu_marginal > 0


def test_synthesis_is_nearly_free_and_that_is_the_point():
    """~330x realtime means a three-minute episode is under a second of card.
    If this ever costs real money the prefetch plan's premise is gone."""
    assert metering.gpu_cost(180) < 0.001


def test_usage_from_a_missing_field_is_zero_not_a_crash():
    """`usage` fields are None when a feature was not used, not absent."""
    u = metering.Usage()
    u.add_model_call("claude-sonnet-5", {"input_tokens": 5, "output_tokens": None})
    assert (u.input_tokens, u.output_tokens) == (5, 0)


def test_two_calls_on_one_episode_add_up():
    """A researched episode runs the cover and the research at once. Counting
    one of them would have reported the expensive episodes at half price."""
    u = metering.Usage()
    u.add_model_call("claude-sonnet-5", {"input_tokens": 100, "output_tokens": 200})
    u.add_model_call("claude-sonnet-5", {"input_tokens": 300, "output_tokens": 400})
    assert (u.model_calls, u.input_tokens, u.output_tokens) == (2, 400, 600)


# --- the ledger -----------------------------------------------------------

def test_the_cost_is_stored_not_recomputed(store, monkeypatch):
    """A row is evidence for a bill. Repricing history when the rate card
    changes would make "what did March cost" depend on when you ask."""
    store.record("u1", usage(inp=1_000_000, out=0))
    monkeypatch.setitem(metering.PRICES, "claude-sonnet-5", (99.0, 99.0))
    assert store.report()["totals"]["cost_usd"] == pytest.approx(2.00)


def test_a_broken_ledger_never_breaks_an_episode(store, monkeypatch):
    """The listener already has their audio. A metering failure that failed the
    request would trade a recoverable billing gap for a broken product."""
    def boom(*_a, **_k):
        raise metering.sqlite3.OperationalError("disk full")
    monkeypatch.setattr(store, "_conn", boom)
    assert store.record("u1", usage()) == 0


def test_the_plan_is_stamped_at_write_time(store):
    """Someone who upgrades on the 20th did not cost paid-plan money on the
    5th. Joining to today's plan would rewrite what free users cost."""
    store.record("u1", usage(), plan="free", at=100)
    store.record("u1", usage(), plan="plus", at=200)
    rows = store.rows()
    assert [r["plan"] for r in rows] == ["free", "plus"]


def test_a_legacy_paid_row_folds_into_the_paid_tier_not_into_free(store):
    """The column shipped with two values and now names three tiers.

    "paid" is what an account created before `entitlements.py` is stamped with,
    and the one thing it must not become is "free": that is the direction that
    silently reports a paying listener as costing nothing, in the exact split
    the column was added to make possible.
    """
    store.record("legacy", usage(), plan="paid")
    assert store.rows()[0]["plan"] == "plus"
    assert store.report()["by_plan"]["plus"]["listeners"] == 1


def test_an_unknown_plan_falls_back_to_free_rather_than_inventing_a_bucket(store):
    store.record("u1", usage(), plan="enterprise-gold")
    assert store.rows()[0]["plan"] == "free"


# --- the report -----------------------------------------------------------

def test_the_report_separates_billed_from_assumed(store):
    """Rolling the GPU into a per-listener average says more about how many
    listeners there are than about what a listener costs."""
    store.record("u1", usage(audio_seconds=180))
    report = store.report()
    assert "billed" in report["totals"]["basis"]
    assert "assumed" in report["fixed"]["basis"]
    assert report["fixed"]["gpu_usd"] >= 0
    # The fixed floor is reported, never folded in.
    assert report["totals"]["cost_usd"] < report["fixed"]["gpu_usd"] + 1


def test_the_tail_survives_the_mean(store):
    """One listener costing 100x the rest is the case that decides whether a
    flat price needs a cap, and it is exactly what a mean hides."""
    for i in range(99):
        store.record(f"quiet-{i}", usage(inp=1000, out=1000))
    store.record("whale", usage(inp=1_000_000, out=1_000_000))
    spread = store.report()["per_listener"]["cost_usd"]
    assert spread["max"] > spread["median"] * 100
    assert spread["p99"] > spread["median"]
    assert spread["max"] >= spread["p99"] >= spread["p90"] >= spread["median"]


def test_a_single_listener_does_not_break_the_percentiles(store):
    """statistics.quantiles raises on one data point, and the first day of a
    deployment is one data point."""
    store.record("only", usage())
    spread = store.report()["per_listener"]["cost_usd"]
    assert spread["median"] == spread["p99"] == spread["max"] > 0


def test_paid_and_unpaid_are_reported_separately(store):
    store.record("free-1", usage(inp=1000, out=1000), plan="free")
    store.record("free-2", usage(inp=1000, out=1000), plan="free")
    store.record("plus-1", usage(inp=4000, out=4000), plan="plus")
    by_plan = store.report()["by_plan"]
    assert by_plan["free"]["listeners"] == 2
    assert by_plan["plus"]["listeners"] == 1
    assert by_plan["plus"]["cost_per_listener"] > by_plan["free"]["cost_per_listener"]


def test_every_tier_gets_a_bucket_even_with_nobody_in_it(store):
    """A pricing decision is made by comparing tiers, so a tier that is
    missing from the report because nobody is on it yet is the one you cannot
    reason about."""
    import entitlements

    store.record("free-1", usage(), plan="free")
    assert set(store.report()["by_plan"]) == set(entitlements.TIERS)


def test_the_breakdown_adds_up_to_the_total(store):
    """A total nobody can decompose is a total nobody can argue with, which is
    the wrong property for a number a price is set from."""
    u = usage(inp=5000, out=5000, audio_seconds=200)
    u.add_research(2, 0.01)
    store.record("u1", u)
    report = store.report()
    parts = sum(report["breakdown"].values())
    assert parts == pytest.approx(report["totals"]["cost_usd"], abs=1e-6)


def test_the_cache_saving_is_estimated_and_says_so(store):
    """It is the one cost line that improves as listeners are added, so a
    forecast that ignores it overstates the bill at scale."""
    for _ in range(3):
        store.record("u1", usage(inp=10_000, out=10_000))
    hit = metering.Usage(audio_seconds=180, cache_hit=True)
    store.record("u2", hit)
    cache = store.report()["cache"]
    assert cache["hits"] == 1 and cache["misses"] == 3
    assert cache["estimated_saving_usd"] > 0
    assert "assumed" in cache["basis"]


def test_a_window_excludes_what_is_outside_it(store):
    import time
    now = time.time()
    store.record("old", usage(), at=now - 40 * 86400)
    store.record("new", usage(), at=now - 1)
    report = store.report(since=now - 7 * 86400, until=now + 1)
    assert report["totals"]["listeners"] == 1


def test_an_empty_window_reports_zero_rather_than_dividing_by_it(store):
    report = store.report()
    assert report["totals"]["episodes"] == 0
    assert report["totals"]["cost_per_episode"] == 0.0
    assert report["per_listener"]["cost_usd"]["mean"] == 0.0


# --- abuse ----------------------------------------------------------------

def test_volume_and_spend_are_caught_separately(store):
    """They look different: a script hammering the endpoint is many cheap
    requests; an expensive afternoon is few requests and a lot of money.
    Either threshold alone misses the other."""
    import time
    now = time.time()
    for _ in range(40):
        store.record("chatty", metering.Usage(), at=now - 10)
    store.record("spendy", usage(inp=1_000_000, out=1_000_000), at=now - 10)
    flagged = {u["user_id"]: u for u in metering.suspects(store, now=now)}
    assert "chatty" in flagged and "episodes" in flagged["chatty"]["reasons"][0]
    assert "spendy" in flagged and "$" in flagged["spendy"]["reasons"][0]


def test_an_ordinary_listener_is_not_flagged(store):
    import time
    now = time.time()
    for _ in range(3):
        store.record("normal", usage(), at=now - 10)
    assert metering.suspects(store, now=now) == []


def test_flagging_reports_and_never_blocks():
    """An automatic block on a metering heuristic eventually locks out a real
    listener who has no way to tell anyone. This module must not grow a verb."""
    for name in dir(metering):
        assert not name.startswith(("ban", "block", "suspend"))


# --- the numbers come from the provider, not from counting words ----------

class _Stream:
    """The shape `messages.stream` returns: an async context manager with a
    text stream and a final message carrying `usage`."""

    def __init__(self, text: str, usage: dict):
        self._text, self._usage = text, usage

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    @property
    def text_stream(self):
        async def gen():
            for word in self._text.split(" "):
                yield word + " "
        return gen()

    async def get_final_message(self):
        return type("F", (), {"stop_reason": "end_turn",
                              "usage": type("U", (), self._usage)()})()


class _Messages:
    def __init__(self, stream):
        self._stream = stream

    def stream(self, **_kwargs):
        return self._stream


def test_the_generator_records_what_the_provider_billed(monkeypatch):
    """Not estimated from the script.

    Output tokens could be guessed from the words; input tokens could not - the
    prompt, the examples and any evidence packet are invisible from the text -
    and cache reads are invisible from both. A word-count estimate would have
    been wrong in the direction that flatters the bill.
    """
    import asyncio

    import script_generator as sg

    gen = sg.ScriptGenerator.__new__(sg.ScriptGenerator)
    gen.client = type("C", (), {"messages": _Messages(_Stream(
        "One sentence here. And a second one.",
        {"input_tokens": 4321, "output_tokens": 99,
         "cache_read_input_tokens": 800, "cache_creation_input_tokens": 0},
    ))})()

    # search=False - the default researches every episode (PROBLEMS.md 76)
    # and what is being counted here is model tokens, not retrieval.
    plan = sg.plan_episode("anything", 1, search=False)
    notes = sg.ScriptNotes()

    async def drain():
        return [s async for s in gen.stream_sentences(plan, notes)]

    spoken = asyncio.run(drain())
    assert spoken, "the fake produced no sentences, so this proved nothing"
    assert notes.usage.input_tokens == 4321, "input tokens were not read from usage"
    assert notes.usage.output_tokens == 99
    assert notes.usage.cache_read_tokens == 800
    assert notes.usage.output_tokens != len(" ".join(spoken).split()), (
        "this looks like a word count, which is the thing being ruled out")
