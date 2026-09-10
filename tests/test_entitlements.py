"""What a tier allows, and the guard that makes taking something away deliberate."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import entitlements


@pytest.fixture
def limits(monkeypatch):
    """Set tier limits from the environment and put them back afterwards.

    `reload_tiers` mutates a module-level table, so a test that changed it and
    did not restore it would hand the next test a different free tier - the
    exact class of order-dependent failure conftest exists to prevent.
    """
    def apply(**values):
        for name, value in values.items():
            monkeypatch.setenv(name, str(value))
        entitlements.reload_tiers()
    yield apply
    for name in entitlements.LIMIT_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    entitlements.reload_tiers()


# --- the tiers themselves -------------------------------------------------

def test_the_tiers_are_ordered_cheapest_first():
    """`at_least` compares by position, so the order is not cosmetic: reversing
    it would silently grant every paid feature to the free tier."""
    assert entitlements.TIERS == ("free", "plus", "unlimited")
    assert entitlements.rank("free") < entitlements.rank("plus") < entitlements.rank("unlimited")


def test_a_legacy_paid_account_becomes_the_paid_tier_not_the_free_one():
    """The column shipped holding "paid". Falling back to "free" would
    downgrade a paying account on the day tiers arrived, and it would look
    like nothing happened."""
    assert entitlements.normalise("paid") == "plus"


def test_an_unknown_tier_resolves_to_free_rather_than_raising():
    """Total on purpose. A plan column written by a newer version, read after a
    rollback, must not raise in the middle of somebody's episode - and "free"
    is the direction that under-serves rather than handing out an allowance
    nobody bought."""
    assert entitlements.normalise("enterprise-gold") == "free"
    assert entitlements.normalise("") == "free"
    assert entitlements.normalise(None) == "free"


def test_free_is_capped_per_day_and_plus_per_week():
    """The shape of the product decision, not the numbers. A daily cap tells
    the listener enjoying it most to come back tomorrow; that is acceptable for
    a free tier and is the wrong way to treat somebody who is paying."""
    assert entitlements.limit_for("free", "episode").window == "day"
    assert entitlements.limit_for("plus", "episode").window == "week"


def test_unlimited_really_has_no_ceiling():
    for resource in entitlements.RESOURCES:
        assert entitlements.limit_for("unlimited", resource).unlimited


def test_explore_is_allowed_more_than_generation_on_every_tier():
    """Explore provably cannot write a script - the pipeline refuses - so it
    costs GPU seconds and no model call. Pricing the two the same would make
    the cheapest surface feel like the most expensive one."""
    for name in entitlements.TIERS:
        episode = entitlements.limit_for(name, "episode")
        explore = entitlements.limit_for(name, "explore")
        assert explore.unlimited or explore.count > episode.count, name


def test_limits_come_from_the_environment(limits):
    limits(FREE_EPISODES_PER_DAY=2)
    assert entitlements.limit_for("free", "episode").count == 2


def test_a_nonsense_limit_falls_back_rather_than_crashing_the_server(limits):
    """A typo in a deployment variable must not take the process down. It is
    the one setting most likely to be edited in a hurry, by whoever is awake."""
    limits(FREE_EPISODES_PER_DAY="lots")
    assert entitlements.limit_for("free", "episode").count == 5


# --- features -------------------------------------------------------------

def test_every_feature_is_available_on_every_tier_today():
    """**Read this before changing it.**

    The feature registry is a mechanism, and its policy is deliberately empty:
    every listener who has ever used FAM has had all of it, and shipping a tier
    system that silently takes things away is the "quietly worse than intended"
    failure this project has lost most time to - with the twist that here the
    listener notices and is right.

    So this test fails the day a feature moves behind a tier. That is not a
    reason to avoid doing it. It is a reason to do it on purpose: change the
    `min_tier` on the feature, change this test, and say in the commit which
    capability free listeners no longer have and why.
    """
    for name in entitlements.TIERS:
        assert entitlements.features_for(name) == set(entitlements.FEATURE_KEYS), (
            f"the {name} tier no longer has every feature - if that is "
            f"intended, update this test and say what was taken away"
        )


def test_a_gate_on_a_feature_that_does_not_exist_fails_closed():
    """A typo in a gate must refuse and be noticed, never allow and be relied
    on. The opposite default turns a misspelling into a permanent hole."""
    assert entitlements.allows("unlimited", "featur_that_is_mispelled") is False


def test_every_feature_names_a_real_tier():
    for feature in entitlements.FEATURES:
        assert feature.min_tier in entitlements.TIERS, feature.key


def test_a_tier_can_only_narrow_the_servers_own_length_ceiling(limits):
    """Raising a tier's number must never hand out a length the engine was not
    configured to produce, so the two are combined with min() and never with
    the tier winning."""
    limits(UNLIMITED_MAX_MINUTES=60)
    assert entitlements.max_minutes("unlimited", ceiling=10) == 10
    limits(FREE_MAX_MINUTES=3)
    assert entitlements.max_minutes("free", ceiling=10) == 3


# --- what the client is told ---------------------------------------------

def test_describe_says_the_tier_the_limits_and_the_features():
    described = entitlements.describe("free")
    assert described["tier"] == "free"
    assert set(described["limits"]) == set(entitlements.RESOURCES)
    assert "search" in described["features"]


def test_the_catalogue_lists_every_tier_in_order():
    names = [t["name"] for t in entitlements.catalogue()["tiers"]]
    assert names == list(entitlements.TIERS)


def test_the_catalogue_carries_no_prices():
    """A price here and a price in App Store Connect are two places for one
    fact, and the wrong one is always the one the listener is reading. It also
    cannot be right per country from in here."""
    blob = repr(entitlements.catalogue())
    assert "price" not in blob.lower()
    assert "$" not in blob
