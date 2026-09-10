"""Tiers: what a plan costs, what it allows, and what it unlocks.

Until now `plan` was a single column on an account with two values, written by
nobody, existing only so that the day payment arrived the cost report could
split on it (`metering.py`). This module is that day arriving: it is the one
place that says which tiers exist, what each one may do, and how much of it.

**Three tiers**, and the shape of them is a product decision recorded here so
it is not re-derived from whichever endpoint happens to be under discussion:

* `free` - a daily ceiling. Small enough that the fixed GPU floor is not being
  spent on somebody who never comes back, large enough to hear what FAM is.
* `plus` - money, and a much larger weekly allowance. Weekly rather than daily
  on purpose: someone paying should be able to spend a Sunday listening
  without being told to come back tomorrow, which is the failure a daily cap
  produces for exactly the listener who is enjoying it most.
* `unlimited` - no ceiling at all.

## Two different things live here, and confusing them is the trap

**Limits** are quantities: how many episodes, over what window. They are
decided, they are enforced today, and `quotas.py` counts against them.

**Features** are capabilities: attachments, research, saved mixes. The registry
below exists so that a feature can be moved behind a tier by editing one line -
and **today every feature is available on every tier**, deliberately.

That is not an oversight, it is the only safe starting state. Every listener
who has ever used FAM has had all of it; shipping a tier system that silently
takes things away is the "quietly worse than intended" failure this project has
lost the most time to (CLAUDE.md), with the added twist that the listener would
be able to tell and would be right. So the mechanism ships switched on and the
policy ships empty, and `tests/test_entitlements.py` fails the day someone
changes that - not to prevent it, but to make it a decision somebody made on
purpose rather than a line that slipped in.

## What this module deliberately does not do

No payment. No subscription lifecycle, no receipts, no App Store server
notifications, no proration. `set_plan` moves an account between tiers and that
is all; the thing that decides *when* to call it does not exist yet. Keeping
billing out of here means the enforcement path can be finished, tested and
trusted before any money moves, and that the day a payment provider is chosen
it plugs into one function rather than into every endpoint.

Nor does it meter cost - `metering.py` already records what was actually spent,
and the two must not be merged: a quota is a promise made in advance and a
ledger is a fact recorded afterwards. When they disagree the ledger is right.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

#: Ascending. Order is meaningful: `at_least` compares by position, so a
#: feature gated at "plus" is available to "plus" and "unlimited" without
#: anybody having to list them.
TIERS: tuple[str, ...] = ("free", "plus", "unlimited")

#: What the column used to hold. An account created before tiers existed is
#: stamped "paid", and there is exactly one paid tier it can mean. Mapped
#: rather than dropped, because falling back to "free" would silently downgrade
#: a paying account and would misreport their cost in the split that the column
#: was added for in the first place.
LEGACY_TIERS: dict[str, str] = {"paid": "plus"}

#: Windows a limit can be counted over. `quotas.py` owns the arithmetic; this
#: is only the vocabulary, kept here so a tier cannot name a window that
#: nothing knows how to count.
WINDOWS: tuple[str, ...] = ("day", "week")

#: Things that can be counted. Two, and they are separate because they cost
#: differently: an `episode` may write a script (a Claude call, ~$0.03) and
#: always synthesises audio; an `explore` replay provably cannot write one -
#: the pipeline refuses - so it costs GPU seconds and nothing else. A single
#: combined allowance would either price Explore as if it wrote scripts, which
#: makes the cheapest surface feel the most expensive, or price episodes as if
#: they did not.
RESOURCES: tuple[str, ...] = ("episode", "explore")

#: A limit of this value is "no ceiling". Not `0`, which is a real and useful
#: value meaning "none at all" - a distinction worth having before the first
#: feature needs to be switched off for a tier entirely.
UNLIMITED = -1


#: Every environment variable this module reads. Listed rather than derived
#: because `tests/conftest.py` has to clear them and its staleness guard only
#: scans `config.py` - the same hand-listing the voice directories get, for
#: the same reason: a developer's widened cap must not quietly widen the suite.
LIMIT_ENVIRONMENT: tuple[str, ...] = (
    "FREE_EPISODES_PER_DAY", "FREE_EXPLORE_PER_DAY", "FREE_MAX_MINUTES",
    "FREE_MAX_DOWNLOADS",
    "PLUS_EPISODES_PER_WEEK", "PLUS_EXPLORE_PER_WEEK", "PLUS_MAX_MINUTES",
    "PLUS_MAX_DOWNLOADS",
    "UNLIMITED_MAX_MINUTES", "UNLIMITED_MAX_DOWNLOADS",
)


def _env_int(name: str, default: int) -> int:
    """Limits are tunable from the environment, and that is on purpose.

    The right number for a free tier is a business decision informed by
    `tools/usage_report.py`, and it will be wrong on the first guess. A beta
    that has to be redeployed to widen a cap will be widened by whoever is
    awake rather than by whoever decided.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Limit:
    """How much of one resource, over one window."""

    resource: str
    window: str
    count: int

    @property
    def unlimited(self) -> bool:
        return self.count == UNLIMITED

    def as_dict(self) -> dict:
        return {"resource": self.resource, "window": self.window,
                "count": self.count, "unlimited": self.unlimited}


@dataclass(frozen=True)
class Tier:
    name: str
    label: str
    #: Shown to a listener. No price here: the price lives with the payment
    #: provider that does not exist yet, and a number written in two places
    #: disagrees in one of them.
    blurb: str
    limits: tuple[Limit, ...]
    #: The longest episode this tier may ask for, capped again by
    #: `settings.max_minutes`. Duration is the other lever on GPU cost, and
    #: having it here means it can be used without a second mechanism.
    max_minutes: int
    #: How many episodes may be held offline at once. A **standing capacity**,
    #: not a rate - which is why it is here and not in `quotas.py`. A windowed
    #: counter would hand out a fresh download allowance every morning and
    #: never require anybody to delete anything, which is the opposite of what
    #: a shelf limit is for.
    max_downloads: int = 3

    def limit_for(self, resource: str) -> Optional[Limit]:
        for limit in self.limits:
            if limit.resource == resource:
                return limit
        return None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "blurb": self.blurb,
            "max_minutes": self.max_minutes,
            "max_downloads": self.max_downloads,
            "limits": [limit.as_dict() for limit in self.limits],
            "features": sorted(features_for(self.name)),
        }


#: The defaults are placeholders in the honest sense: they are the numbers to
#: start a free beta with, not numbers anybody has measured. `usage_report.py`
#: prints the median and the p99 per listener, and on the one measured run the
#: worst listener cost 68x the median - so the first real setting of these
#: should come from that report rather than from taste.
def _build_tiers() -> dict[str, "Tier"]:
    """The table, read from the environment each time it is called.

    A function rather than a literal so `reload_tiers()` exists: the limits are
    env-tunable, and a setting that can only be changed by restarting the
    process cannot be tested without reaching into module state. This is the
    same reason `paths.data_path` reads its variable at call time.

    The defaults are placeholders in the honest sense: they are numbers to
    start a free beta with, not numbers anybody has measured.
    `tools/usage_report.py` prints the median and the p99 per listener, and on
    the one measured run the worst listener cost 68x the median - so the first
    real setting of these should come from that report rather than from taste.
    """
    return {
        "free": Tier(
            name="free",
            label="FAM",
            blurb="A few episodes a day, and the whole of Explore to listen through.",
            limits=(
                Limit("episode", "day", _env_int("FREE_EPISODES_PER_DAY", 5)),
                Limit("explore", "day", _env_int("FREE_EXPLORE_PER_DAY", 25)),
            ),
            max_minutes=_env_int("FREE_MAX_MINUTES", 10),
            max_downloads=_env_int("FREE_MAX_DOWNLOADS", 3),
        ),
        "plus": Tier(
            name="plus",
            label="FAM Plus",
            blurb="Enough episodes a week to listen the way you actually listen.",
            limits=(
                Limit("episode", "week", _env_int("PLUS_EPISODES_PER_WEEK", 150)),
                Limit("explore", "week", _env_int("PLUS_EXPLORE_PER_WEEK", UNLIMITED)),
            ),
            max_minutes=_env_int("PLUS_MAX_MINUTES", 10),
            max_downloads=_env_int("PLUS_MAX_DOWNLOADS", 25),
        ),
        "unlimited": Tier(
            name="unlimited",
            label="FAM Unlimited",
            blurb="No ceiling.",
            limits=(
                Limit("episode", "day", UNLIMITED),
                Limit("explore", "day", UNLIMITED),
            ),
            max_minutes=_env_int("UNLIMITED_MAX_MINUTES", 10),
            # No server-side cap. The honest caveat, so nobody reads this as a
            # promise the app cannot keep: the real limit on an unlimited tier
            # is the device's storage, not this number.
            max_downloads=_env_int("UNLIMITED_MAX_DOWNLOADS", UNLIMITED),
        ),
    }


TIER_TABLE: dict[str, "Tier"] = _build_tiers()


def reload_tiers() -> dict[str, "Tier"]:
    """Re-read the limit variables. For tests, and for anything that changes
    the environment after import."""
    TIER_TABLE.clear()
    TIER_TABLE.update(_build_tiers())
    return TIER_TABLE


@dataclass(frozen=True)
class Feature:
    """One capability, and the lowest tier that has it.

    `min_tier` is a name from TIERS. Everything ships at "free" - see the
    module docstring for why that is the deliberate starting state rather than
    an empty policy nobody got round to filling in.
    """

    key: str
    label: str
    min_tier: str
    #: Why this could be gated, written now while the reasoning is fresh. It is
    #: the argument the person flipping it will want and will not otherwise
    #: have; several of these are "should not be gated", which is just as
    #: useful to have written down.
    note: str


#: The registry. Adding a capability to FAM means adding a line here, so that
#: "what does a plan get you" has one answer rather than being distributed
#: across the endpoints that happen to implement each thing.
FEATURES: tuple[Feature, ...] = (
    Feature("search", "Ask anything", "free",
            "The one-sentence spec. Gating this is gating the product; the "
            "daily limit is the lever, not the feature."),
    Feature("explore", "Explore", "free",
            "Replays only, and provably cannot write a script. The cheapest "
            "surface there is - gate the count, never the access."),
    Feature("go_deeper", "Go Deeper", "free",
            "Costs one cached lookup and offers the predicted follow-up. "
            "Gating it would make the free tier feel like a demo."),
    Feature("research", "Live research", "free",
            "Exa searches are billed per call and are the one per-episode "
            "cost that is not the model. The strongest candidate for a gate."),
    Feature("attachments", "Documents and photos", "free",
            "Extraction is done at attach time, so the marginal cost is "
            "storage and a longer prompt. A plausible Plus feature."),
    Feature("voice_choice", "Choose a voice", "free",
            "Voice is not part of the cache key, so switching reuses the "
            "script at ~90ms and zero API cost. Nearly free to give away."),
    Feature("mixes", "Daily mixes", "free",
            "Already gated on having an account, which is a different axis. "
            "A mix holds topic ids and no audio, so it costs storage."),
    Feature("weekly_recap", "Weekly recap", "free",
            "Reads the event log. Account-gated already."),
    Feature("explore_new", "Explore New", "free",
            "The only surface offering anything outside an established taste. "
            "Gating discovery makes the free tier a smaller world, not a "
            "cheaper one."),
    Feature("long_episodes", "Longer episodes", "free",
            "Handled by `max_minutes` per tier rather than as a flag, because "
            "it is a quantity. Listed so the registry is a complete answer."),
    Feature("downloads", "Offline downloads", "free",
            "Handled by `max_downloads` per tier rather than as a flag, "
            "because it is a quantity. The bytes sit on the listener's own "
            "device, so the marginal cost to FAM is one stream that would "
            "have happened anyway - it is a retention feature, not a cost."),
    Feature("share_external", "Share outside FAM", "free",
            "A share link costs a row and reaches somebody who does not have "
            "the app. Gating the cheapest route to a new listener would be a "
            "strange way to grow."),
    Feature("api_access", "API access", "free",
            "There is no public API token scheme yet. When there is, this is "
            "where it is gated, and it is the clearest paid-tier feature "
            "here - it is the one that lets somebody else's software spend "
            "the GPU."),
)

FEATURE_KEYS: tuple[str, ...] = tuple(f.key for f in FEATURES)
_FEATURE_TABLE: dict[str, Feature] = {f.key: f for f in FEATURES}


class UnknownTier(ValueError):
    """A tier name nothing knows about, phrased so it can be shown."""


def normalise(tier: str) -> str:
    """Any stored plan value to a tier this module knows.

    Total by design: a plan column read from a database written by an older
    version, or by a version newer than this one after a rollback, must resolve
    to *something* rather than raising in the middle of a request. Unknown
    means "free", which is the safe direction - it under-serves rather than
    handing out an allowance nobody bought.
    """
    tier = (tier or "").strip().lower()
    tier = LEGACY_TIERS.get(tier, tier)
    return tier if tier in TIER_TABLE else "free"


def tier(name: str) -> Tier:
    """The Tier record. Normalises, so this never raises on stored data."""
    return TIER_TABLE[normalise(name)]


def rank(name: str) -> int:
    return TIERS.index(normalise(name))


def at_least(have: str, need: str) -> bool:
    """Is `have` this tier or better? The only ordering comparison anywhere -
    every other module asks this rather than comparing names itself."""
    return rank(have) >= rank(need)


def feature(key: str) -> Optional[Feature]:
    return _FEATURE_TABLE.get(key)


def allows(tier_name: str, feature_key: str) -> bool:
    """May this tier use this feature?

    An unknown feature key is False, not True. A typo in a gate must fail
    closed and be noticed, rather than open and be relied on.
    """
    known = _FEATURE_TABLE.get(feature_key)
    if known is None:
        return False
    return at_least(tier_name, known.min_tier)


def features_for(tier_name: str) -> set[str]:
    return {f.key for f in FEATURES if at_least(tier_name, f.min_tier)}


def limit_for(tier_name: str, resource: str) -> Optional[Limit]:
    return tier(tier_name).limit_for(resource)


def max_minutes(tier_name: str, ceiling: int) -> int:
    """The longest episode this tier may ask for.

    Takes the server's own ceiling and returns the smaller. A tier can only
    ever narrow `settings.max_minutes`, never widen it - so raising a tier's
    number can never accidentally hand out a length the engine was not
    configured to produce.
    """
    return min(tier(tier_name).max_minutes, int(ceiling))


def max_downloads(tier_name: str) -> int:
    """How many episodes this tier may hold offline at once.

    `UNLIMITED` means no server-side cap - which is not the same as no cap, and
    anything showing this to a listener should not imply otherwise: the device
    runs out of storage long before the server runs out of rows.
    """
    return tier(tier_name).max_downloads


def describe(tier_name: str) -> dict:
    """What the client is told about the tier it is on. One shape, used by
    `/api/entitlements` and by anything that needs to explain a refusal."""
    resolved = tier(tier_name)
    return {
        "tier": resolved.name,
        "label": resolved.label,
        "blurb": resolved.blurb,
        "max_minutes": resolved.max_minutes,
        "max_downloads": resolved.max_downloads,
        "limits": {limit.resource: limit.as_dict() for limit in resolved.limits},
        "features": sorted(features_for(resolved.name)),
    }


def catalogue() -> dict:
    """Every tier, for a pricing screen. Ascending, so the client does not
    have to know the order."""
    return {
        "tiers": [TIER_TABLE[name].as_dict() for name in TIERS],
        "features": [
            {"key": f.key, "label": f.label, "min_tier": f.min_tier}
            for f in FEATURES
        ],
    }
