"""Counting an allowance: the windows, the race, and the refund."""
from __future__ import annotations

import calendar
import datetime as dt
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import entitlements
import quotas

#: The real function, captured at import - before conftest's `isolated_quotas`
#: fixture replaces it with one that always says no. A test that asked
#: `quotas.settings_enforcing` at call time would be measuring the stub.
REAL_SETTINGS_ENFORCING = quotas.settings_enforcing


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A real store, with enforcement on - conftest turns it off for the rest
    of the suite, and a quota test that inherited that would pass by measuring
    nothing."""
    monkeypatch.setattr(quotas, "settings_enforcing", lambda: True)
    return quotas.QuotaStore(str(tmp_path / "quotas.db"))


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setenv("FREE_EPISODES_PER_DAY", "2")
    monkeypatch.setenv("FREE_EXPLORE_PER_DAY", "3")
    entitlements.reload_tiers()
    yield
    for name in entitlements.LIMIT_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    entitlements.reload_tiers()


def at(y, m, d, h=12):
    return calendar.timegm(dt.datetime(y, m, d, h).utctimetuple())


# --- windows --------------------------------------------------------------

def test_a_window_key_is_the_identity_of_the_window():
    """Stored rather than recomputed from a timestamp on read, so a later
    change to the maths cannot silently reinterpret counters already written."""
    assert quotas.window_key("day", at(2026, 9, 10)) == "day:2026-09-10"
    assert quotas.window_key("week", at(2026, 9, 10)) == "week:2026-W37"


def test_every_day_of_one_week_shares_a_weekly_window():
    keys = {quotas.window_key("week", at(2026, 9, d)) for d in range(7, 14)}
    assert len(keys) == 1


def test_a_week_resets_on_monday_and_a_day_at_midnight():
    """The number the listener is told. A ceiling nobody can see the end of
    reads as a fault rather than a limit."""
    thursday = at(2026, 9, 10)
    assert quotas.window_end("day", thursday) == at(2026, 9, 11, 0)
    assert quotas.window_end("week", thursday) == at(2026, 9, 14, 0)


def test_the_counter_rolls_over_at_the_window_boundary(store, small):
    for _ in range(2):
        store.reserve("u1", "free", "episode", at=at(2026, 9, 10))
    with pytest.raises(quotas.QuotaExceeded):
        store.reserve("u1", "free", "episode", at=at(2026, 9, 10, 23))
    # Tomorrow is a different window, so the allowance is back.
    assert store.reserve("u1", "free", "episode", at=at(2026, 9, 11)).allowed


# --- spending -------------------------------------------------------------

def test_reserving_up_to_the_limit_is_allowed_and_the_next_one_is_not(store, small):
    assert store.reserve("u1", "free", "episode").used == 1
    assert store.reserve("u1", "free", "episode").used == 2
    with pytest.raises(quotas.QuotaExceeded) as exc:
        store.reserve("u1", "free", "episode")
    assert exc.value.verdict.remaining == 0


def test_a_refusal_does_not_consume_the_allowance_it_refused(store, small):
    """The reservation is taken first so the check is atomic, which means a
    refusal has to give it straight back - otherwise every rejected attempt
    would push the reset further away and the counter would run away from the
    listener."""
    for _ in range(2):
        store.reserve("u1", "free", "episode")
    for _ in range(5):
        with pytest.raises(quotas.QuotaExceeded):
            store.reserve("u1", "free", "episode")
    assert store.used("u1", "episode", "day") == 2


def test_the_two_resources_are_counted_separately(store, small):
    for _ in range(2):
        store.reserve("u1", "free", "episode")
    # Explore has its own, looser allowance and is untouched by the above.
    assert store.reserve("u1", "free", "explore").allowed


def test_one_listener_running_out_does_not_affect_another(store, small):
    for _ in range(2):
        store.reserve("u1", "free", "episode")
    assert store.reserve("u2", "free", "episode").allowed


def test_an_unlimited_tier_is_still_counted(store, small):
    """Unlimited is not unmeasured. The report that says what a listener costs
    is only as good as the rows underneath it."""
    for _ in range(20):
        assert store.reserve("u1", "unlimited", "episode").allowed
    assert store.used("u1", "episode", "day") == 20


def test_a_legacy_paid_listener_gets_the_paid_allowance(store, small):
    """The tier arrives from a column that may still say "paid". Resolving that
    to "free" would apply a five-a-day cap to somebody who is paying."""
    verdict = store.reserve("u1", "paid", "episode")
    assert verdict.tier == "plus"
    assert verdict.limit == entitlements.limit_for("plus", "episode").count


# --- refunds --------------------------------------------------------------

def test_a_refund_gives_the_allowance_back(store, small):
    verdict = store.reserve("u1", "free", "episode")
    store.refund("u1", verdict.resource, verdict.window)
    assert store.used("u1", "episode", "day") == 0


def test_a_refund_can_never_drive_a_counter_negative(store, small):
    """Two refunds for one reservation would otherwise hand out free episodes
    to whoever could cause them."""
    store.reserve("u1", "free", "episode")
    for _ in range(5):
        store.refund("u1", "episode", "day")
    assert store.used("u1", "episode", "day") == 0
    assert store.reserve("u1", "free", "episode").used == 1


# --- switched off ---------------------------------------------------------

def test_with_enforcement_off_nothing_is_counted_or_refused(tmp_path, monkeypatch):
    """`demo.sh` runs this way and says so. Judging the writing must not stop
    after five episodes."""
    monkeypatch.setattr(quotas, "settings_enforcing", lambda: False)
    store = quotas.QuotaStore(str(tmp_path / "q.db"))
    for _ in range(50):
        assert store.reserve("u1", "free", "episode").allowed
    assert store.used("u1", "episode", "day") == 0


def test_a_broken_settings_read_enforces_rather_than_opening_the_gate(monkeypatch):
    """Fail closed. The thing behind this is a GPU and a metered key."""
    import config

    # Settings without the field at all - what a rollback to an older config
    # looks like from in here.
    monkeypatch.setattr(config, "settings", object())
    assert REAL_SETTINGS_ENFORCING() is True


# --- what the listener is told -------------------------------------------

def test_the_refusal_says_the_number_and_when_it_comes_back(store, small):
    for _ in range(2):
        store.reserve("u1", "free", "episode")
    with pytest.raises(quotas.QuotaExceeded) as exc:
        store.reserve("u1", "free", "episode")
    message = str(exc.value)
    assert "2" in message and "UTC" in message
    assert "plan" in message.lower()


def test_the_top_tier_is_not_offered_an_upgrade(store, monkeypatch):
    """There is nothing to sell somebody on the unlimited tier, and a refusal
    that tries to reads as a paywall rather than an error."""
    monkeypatch.setenv("UNLIMITED_MAX_MINUTES", "10")
    limit = entitlements.Limit("episode", "day", 1)
    line = quotas._refusal("episode", "unlimited", limit, 1, 0.0)
    assert "plan" not in line.lower()


def test_status_never_refuses_even_when_the_allowance_is_gone(store, small):
    for _ in range(2):
        store.reserve("u1", "free", "episode")
    verdict = store.status("u1", "free", "episode")
    assert verdict.allowed is False
    assert verdict.remaining == 0
    assert verdict.resets_at > 0


def test_status_does_not_spend(store, small):
    for _ in range(10):
        store.status("u1", "free", "episode")
    assert store.used("u1", "episode", "day") == 0


# --- deletion -------------------------------------------------------------

def test_forget_erases_a_listeners_counters(store, small):
    store.reserve("u1", "free", "episode")
    store.reserve("u2", "free", "episode")
    assert store.forget("u1") == 1
    assert store.used("u1", "episode", "day") == 0
    assert store.used("u2", "episode", "day") == 1
