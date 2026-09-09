"""What each listener costs, recorded at the moment it is spent.

CLAUDE.md's §70 note ends "per-listener metering still does not exist", and the
reason it has to is structural: **the provider only ever sees one account.**
Anthropic bills this organisation, Exa bills this key, the GPU bills by the
hour. None of them can say which listener produced which request, so if that
question is not answered here it cannot be answered anywhere - and every
question about pricing, per-user limits and abuse is that question wearing a
different hat.

## What is recorded

One row per episode, written where the listener id is known (`app.py`, from
`_listener(request)` - never from a parameter, per the settled constraint), and
appended, never updated. A row carries what was actually consumed:

    tokens in / out / cached      what Claude billed
    exa searches and their cost   what retrieval billed
    audio seconds                 what the GPU did
    cache hit                     whether a script was written at all

and the **cost computed at write time**. Denormalised deliberately: prices
change, and a row that recomputed itself against today's price list would
quietly restate what last quarter cost.

## The three costs, which behave differently

This is the part that matters for pricing, and the part a single "cost per
user" number hides:

* **Claude tokens and Exa searches are marginal.** Nobody listens, nobody is
  billed. These scale with use, per listener, and are what a price per listener
  has to cover.
* **The GPU is fixed.** Chatterbox runs in-process on a card that costs the
  same whether it is synthesising or idle. Its *marginal* cost per episode is
  near zero - synthesis runs at ~330x realtime, so a three-minute episode is
  under a second of GPU - and its *real* cost is a monthly floor that exists
  before the first listener arrives.
* **The script cache is a discount that grows with listeners.** Two people who
  ask the same thing pay for one script. Reporting only what was spent would
  make the cache invisible, so `report` estimates what the hits avoided.

So `report` gives marginal cost, the fixed floor, and the two combined at an
assumed listener count - and never rolls the fixed floor silently into a
per-user average, because doing so makes the cheapest possible product look
expensive at ten users and free at ten thousand.

## Honest about what is estimated

Every number here is one of three things, and the report says which:

* **billed** - from the provider's own usage figures (Claude's `usage`, Exa's
  reported `cost_dollars`).
* **priced** - billed quantities times a published rate in `PRICES`.
* **assumed** - the GPU allocation and the cache saving, which rest on
  configuration (`GPU_USD_PER_HOUR`, `SYNTHESIS_REALTIME_FACTOR`) rather than
  on an invoice.

A model with no entry in `PRICES` is counted as **unpriced** and named in the
report rather than costed at zero. A silent $0 is the same failure shape this
project keeps paying for: a number that looks like an answer.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

from paths import data_path

log = logging.getLogger("metering")

#: USD per million tokens, input and output, as published. Checked against the
#: rate card on 2026-09-09. A model missing from here is not costed at zero -
#: `record` marks the row unpriced and `report` says how many there were.
PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.00, 50.00),
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

#: Cached input is billed at a fraction of the input rate; writing to the cache
#: costs a premium over it. Published multipliers, not measured here.
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25

#: What a synthesis GPU costs per hour. Default is the middle of the L4
#: on-demand range; a reserved card or a neocloud is cheaper. Only ever used
#: for an *allocation* - see the module docstring on why this is not a price.
GPU_USD_PER_HOUR = float(os.environ.get("GPU_USD_PER_HOUR", "0.60"))

#: How much faster than realtime Chatterbox synthesises. CLAUDE.md's figure.
#: This is what makes the marginal audio cost negligible and the fixed cost
#: everything.
SYNTHESIS_REALTIME_FACTOR = float(os.environ.get("SYNTHESIS_REALTIME_FACTOR", "330"))

#: Hours per day the card is actually paid for. 24 is the honest default for a
#: service that answers at any hour; scale-to-zero would lower it and add cold
#: starts, which is a product decision rather than a metering one.
GPU_HOURS_PER_DAY = float(os.environ.get("GPU_HOURS_PER_DAY", "24"))

#: Plans a row can be stamped with. "free" is everyone today: nothing in the
#: app sets "paid", because nothing takes payment yet. The column exists so the
#: split is available the day something does, rather than being backfilled from
#: a log that never recorded it.
PLANS = ("free", "paid")


@dataclass
class Usage:
    """What one episode consumed. Accumulated during generation, written once.

    Mutable and shared on purpose: a researched episode runs two model calls at
    once (`_answer_first`), and both must land in the same total. A per-call
    copy would report the cover as free.
    """

    model: str = ""
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    exa_searches: int = 0
    exa_cost: float = 0.0
    audio_seconds: float = 0.0
    cache_hit: bool = False

    def add_model_call(self, model: str, usage: object) -> None:
        """Fold in one Claude response's `usage` block.

        Takes the SDK object rather than numbers so every call site records the
        same four fields; a call site that pulled out only input and output
        would under-report cache traffic without looking wrong.
        """
        self.model = model or self.model
        self.model_calls += 1
        self.input_tokens += _int_attr(usage, "input_tokens")
        self.output_tokens += _int_attr(usage, "output_tokens")
        self.cache_read_tokens += _int_attr(usage, "cache_read_input_tokens")
        self.cache_write_tokens += _int_attr(usage, "cache_creation_input_tokens")

    def add_research(self, searches: int, cost: float) -> None:
        self.exa_searches += int(searches or 0)
        self.exa_cost += float(cost or 0.0)

    def as_dict(self) -> dict:
        return asdict(self)


def _int_attr(obj: object, name: str) -> int:
    """One usage field, from an SDK object or a plain dict, defaulting to 0.

    `usage` fields are None rather than absent when a feature was not used, so
    `getattr(..., 0)` alone is not enough.
    """
    if isinstance(obj, dict):
        value = obj.get(name)
    else:
        value = getattr(obj, name, None)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


@dataclass
class Cost:
    """One episode's cost, split so a total can be explained rather than
    asserted. Dollars, already multiplied out."""

    claude_input: float = 0.0
    claude_output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    exa: float = 0.0
    gpu_marginal: float = 0.0
    priced: bool = True

    @property
    def total(self) -> float:
        return round(self.claude_input + self.claude_output + self.cache_read
                     + self.cache_write + self.exa + self.gpu_marginal, 6)

    def as_dict(self) -> dict:
        out = {k: round(v, 6) for k, v in asdict(self).items() if k != "priced"}
        out["priced"] = self.priced
        out["total"] = self.total
        return out


def price_of(usage: Usage) -> Cost:
    """What that usage cost, at today's published rates.

    A model with no price yields a Cost with `priced=False` and zeroes for the
    Claude half - the Exa and GPU parts are still real and still counted. The
    caller stores the flag so the report can say "N episodes on an unpriced
    model" rather than quietly averaging them in at nothing.
    """
    cost = Cost()
    rate = PRICES.get(usage.model)
    if rate is None:
        cost.priced = not (usage.model_calls or usage.input_tokens or usage.output_tokens)
    else:
        per_in, per_out = rate
        cost.claude_input = usage.input_tokens / 1_000_000 * per_in
        cost.claude_output = usage.output_tokens / 1_000_000 * per_out
        cost.cache_read = (usage.cache_read_tokens / 1_000_000
                           * per_in * CACHE_READ_MULTIPLIER)
        cost.cache_write = (usage.cache_write_tokens / 1_000_000
                            * per_in * CACHE_WRITE_MULTIPLIER)
    cost.exa = float(usage.exa_cost or 0.0)
    cost.gpu_marginal = gpu_cost(usage.audio_seconds)
    return cost


def gpu_cost(audio_seconds: float) -> float:
    """The card time one episode's audio actually occupied.

    An allocation, not a bill. It answers "if the GPU were rented by the
    second, what would this episode owe" - useful for comparing against a
    hosted per-second voice, and useless as a forecast of the invoice, which is
    `fixed_costs()`.
    """
    if not audio_seconds or SYNTHESIS_REALTIME_FACTOR <= 0:
        return 0.0
    gpu_seconds = float(audio_seconds) / SYNTHESIS_REALTIME_FACTOR
    return gpu_seconds / 3600.0 * GPU_USD_PER_HOUR


def fixed_costs(days: float) -> dict:
    """What the machine costs over a window whether or not anybody listens.

    Reported beside the marginal total and never folded into it. A per-listener
    average that includes this says more about how many listeners there are
    than about what a listener costs.
    """
    hours = max(0.0, days) * GPU_HOURS_PER_DAY
    return {
        "gpu_usd": round(hours * GPU_USD_PER_HOUR, 4),
        "gpu_hours": round(hours, 2),
        "usd_per_hour": GPU_USD_PER_HOUR,
        "hours_per_day": GPU_HOURS_PER_DAY,
        "basis": "assumed - configuration, not an invoice "
                 "(GPU_USD_PER_HOUR, GPU_HOURS_PER_DAY)",
    }


class MeterStore:
    """The ledger. Append-only, one row per episode.

    Append-only is not a style preference. A usage row is evidence for a bill
    and for a report about somebody's behaviour, and a table that can be
    updated in place is a table where the answer to "what did this cost in
    March" depends on when you ask.
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = data_path("METERING_DB", "metering.db", path)
        self._local = threading.local()
        with self._conn() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS usage (
                       id                INTEGER PRIMARY KEY AUTOINCREMENT,
                       at                REAL NOT NULL,
                       user_id           TEXT NOT NULL,
                       plan              TEXT NOT NULL DEFAULT 'free',
                       surface           TEXT NOT NULL DEFAULT '',
                       model             TEXT NOT NULL DEFAULT '',
                       minutes           INTEGER NOT NULL DEFAULT 0,
                       model_calls       INTEGER NOT NULL DEFAULT 0,
                       input_tokens      INTEGER NOT NULL DEFAULT 0,
                       output_tokens     INTEGER NOT NULL DEFAULT 0,
                       cache_read_tokens INTEGER NOT NULL DEFAULT 0,
                       cache_write_tokens INTEGER NOT NULL DEFAULT 0,
                       exa_searches      INTEGER NOT NULL DEFAULT 0,
                       exa_cost          REAL NOT NULL DEFAULT 0,
                       audio_seconds     REAL NOT NULL DEFAULT 0,
                       cache_hit         INTEGER NOT NULL DEFAULT 0,
                       cost_usd          REAL NOT NULL DEFAULT 0,
                       claude_usd        REAL NOT NULL DEFAULT 0,
                       exa_usd           REAL NOT NULL DEFAULT 0,
                       gpu_usd           REAL NOT NULL DEFAULT 0,
                       priced            INTEGER NOT NULL DEFAULT 1
                   )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS usage_at ON usage(at)")
            conn.execute("CREATE INDEX IF NOT EXISTS usage_user ON usage(user_id, at)")

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
            # WAL so a reader running a report does not block the writer
            # recording an episode. The report is the long query here, and it
            # is exactly the one that must not stall a listener.
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    def record(self, user_id: str, usage: Usage, *, plan: str = "free",
               surface: str = "", minutes: int = 0, at: float = 0.0) -> int:
        """Append one episode. Returns the row id.

        Never raises into the request path: the caller records after the
        listener already has their audio, and a metering failure must not
        become a failed episode. It logs and returns 0 instead.
        """
        cost = price_of(usage)
        claude_usd = (cost.claude_input + cost.claude_output
                      + cost.cache_read + cost.cache_write)
        row = (
            at or time.time(), user_id or "", plan if plan in PLANS else "free",
            surface, usage.model, int(minutes or 0), usage.model_calls,
            usage.input_tokens, usage.output_tokens, usage.cache_read_tokens,
            usage.cache_write_tokens, usage.exa_searches, round(usage.exa_cost, 6),
            round(usage.audio_seconds, 3), 1 if usage.cache_hit else 0,
            cost.total, round(claude_usd, 6), round(cost.exa, 6),
            round(cost.gpu_marginal, 8), 1 if cost.priced else 0,
        )
        try:
            with self._conn() as conn:
                cur = conn.execute(
                    """INSERT INTO usage (at, user_id, plan, surface, model, minutes,
                           model_calls, input_tokens, output_tokens, cache_read_tokens,
                           cache_write_tokens, exa_searches, exa_cost, audio_seconds,
                           cache_hit, cost_usd, claude_usd, exa_usd, gpu_usd, priced)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", row)
        except sqlite3.Error:
            # Loud, because an unrecorded episode is an unbillable one and a
            # gap in the ledger is invisible by construction - but not fatal,
            # because the listener already has their audio and failing the
            # request now would trade a billing gap for a broken episode.
            log.exception("could not record usage for %r; this episode is unbilled",
                          user_id)
            return 0
        return int(cur.lastrowid or 0)

    def rows(self, since: float = 0.0, until: float = 0.0) -> list[sqlite3.Row]:
        sql = "SELECT * FROM usage WHERE at >= ?"
        args: list = [since]
        if until:
            sql += " AND at < ?"
            args.append(until)
        return list(self._conn().execute(sql + " ORDER BY at", args))

    def count(self) -> int:
        return int(self._conn().execute("SELECT COUNT(*) FROM usage").fetchone()[0])

    # --- reporting ---------------------------------------------------------
    #
    # One method rather than a dozen endpoints, because every question asked of
    # this data ("what is the average user costing us", "who are the worst
    # ten", "what would we charge") is a different slice of the same window,
    # and a report assembled from separately-filtered queries has been wrong
    # here before in ways nobody could see.

    def report(self, since: float = 0.0, until: float = 0.0,
               top: int = 10) -> dict:
        """Everything the ledger can honestly say about a window.

        `since`/`until` are epoch seconds; both zero means all of it. Costs are
        the ones stored at write time, so a price change does not restate the
        past.
        """
        rows = self.rows(since, until)
        first = min((r["at"] for r in rows), default=since or time.time())
        last = max((r["at"] for r in rows), default=until or time.time())
        days = max((last - first) / 86400.0, 0.0)

        per_user: dict[str, dict] = {}
        for r in rows:
            u = per_user.setdefault(r["user_id"], {
                "user_id": r["user_id"], "plan": r["plan"], "episodes": 0,
                "cost_usd": 0.0, "audio_seconds": 0.0, "minutes_requested": 0,
                "cache_hits": 0, "input_tokens": 0, "output_tokens": 0,
                "first_seen": r["at"], "last_seen": r["at"],
            })
            u["episodes"] += 1
            u["cost_usd"] += r["cost_usd"]
            u["audio_seconds"] += r["audio_seconds"]
            u["minutes_requested"] += r["minutes"]
            u["cache_hits"] += r["cache_hit"]
            u["input_tokens"] += r["input_tokens"]
            u["output_tokens"] += r["output_tokens"]
            u["last_seen"] = max(u["last_seen"], r["at"])
            # A listener who upgrades mid-window has rows under both plans.
            # The later stamp is the one that describes them now.
            if r["at"] >= u["last_seen"]:
                u["plan"] = r["plan"]

        listeners = list(per_user.values())
        for u in listeners:
            u["cost_usd"] = round(u["cost_usd"], 6)
            u["audio_seconds"] = round(u["audio_seconds"], 1)

        costs = sorted(u["cost_usd"] for u in listeners)
        episodes = sorted(u["episodes"] for u in listeners)
        audio = sorted(u["audio_seconds"] for u in listeners)
        spend = sum(costs)
        hits = sum(r["cache_hit"] for r in rows)

        by_plan = {}
        for plan in PLANS:
            group = [u for u in listeners if u["plan"] == plan]
            group_cost = round(sum(u["cost_usd"] for u in group), 6)
            group_eps = sum(u["episodes"] for u in group)
            by_plan[plan] = {
                "listeners": len(group),
                "episodes": group_eps,
                "cost_usd": group_cost,
                "cost_per_listener": round(group_cost / len(group), 6) if group else 0.0,
                "episodes_per_listener": round(group_eps / len(group), 2) if group else 0.0,
            }

        breakdown = {
            "claude_usd": round(sum(r["claude_usd"] for r in rows), 6),
            "exa_usd": round(sum(r["exa_usd"] for r in rows), 6),
            "gpu_marginal_usd": round(sum(r["gpu_usd"] for r in rows), 8),
        }
        unpriced = sum(1 for r in rows if not r["priced"])

        return {
            "window": {
                "since": since, "until": until or last,
                "first_event": first if rows else 0.0,
                "last_event": last if rows else 0.0,
                "days": round(days, 3),
            },
            "totals": {
                "episodes": len(rows),
                "listeners": len(listeners),
                "cost_usd": round(spend, 6),
                "audio_hours": round(sum(r["audio_seconds"] for r in rows) / 3600.0, 3),
                "input_tokens": sum(r["input_tokens"] for r in rows),
                "output_tokens": sum(r["output_tokens"] for r in rows),
                "exa_searches": sum(r["exa_searches"] for r in rows),
                "cost_per_episode": round(spend / len(rows), 6) if rows else 0.0,
                "basis": "billed - provider usage figures at published rates",
            },
            # The shape of the distribution, not just its middle. Mean alone
            # hides the case that decides the price: a handful of listeners who
            # cost an order of magnitude more than the median.
            "per_listener": {
                "cost_usd": _spread(costs),
                "episodes": _spread(episodes),
                "audio_seconds": _spread(audio),
            },
            "top_listeners": sorted(
                listeners, key=lambda u: u["cost_usd"], reverse=True)[:top],
            "by_plan": by_plan,
            "breakdown": breakdown,
            "unpriced_episodes": unpriced,
            "unpriced_models": sorted({r["model"] for r in rows if not r["priced"]}),
            "cache": _cache_effect(rows, hits),
            "fixed": fixed_costs(days),
        }


def _spread(values: list[float]) -> dict:
    """Mean, median, the tail, and the worst one.

    p90 and p99 are here because the question "what does a user cost" has two
    answers that a mean cannot tell apart: a population where everyone costs
    the same, and one where 1% cost a hundred times the rest. Only the second
    needs a usage cap, and only the second breaks a flat price.
    """
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0,
                "p99": 0.0, "max": 0.0, "min": 0.0, "total": 0.0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": round(sum(ordered) / len(ordered), 6),
        "median": round(_percentile(ordered, 50), 6),
        "p90": round(_percentile(ordered, 90), 6),
        "p99": round(_percentile(ordered, 99), 6),
        "max": round(ordered[-1], 6),
        "min": round(ordered[0], 6),
        "total": round(sum(ordered), 6),
    }


def _percentile(ordered: list[float], pct: float) -> float:
    """Linear-interpolated percentile over an already-sorted list.

    Written out rather than reached for in `statistics` because
    `quantiles(n=100)` needs at least two data points and raises on one, and
    the first day of a new deployment is exactly one data point.
    """
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return float(ordered[low] + (ordered[high] - ordered[low]) * (rank - low))


def _cache_effect(rows: Iterable[sqlite3.Row], hits: int) -> dict:
    """What the shared script cache saved, estimated from what a miss cost.

    Estimated, and labelled as such: a hit records what it actually cost (the
    audio, and nothing else), so the saving is the difference between that and
    what writing the script would have cost. The best available stand-in for
    "would have cost" is the mean Claude spend of a miss in the same window.

    It matters for pricing rather than for accounting: the cache is the one
    cost line that gets *better* as listeners are added, so a forecast built
    from today's per-episode cost overstates tomorrow's bill.
    """
    misses = [r for r in rows if not r["cache_hit"]]
    if not misses or not hits:
        return {"hits": hits, "misses": len(misses), "hit_rate": 0.0,
                "estimated_saving_usd": 0.0,
                "basis": "assumed - no hits, or no misses to price them against"}
    mean_miss = sum(r["claude_usd"] + r["exa_usd"] for r in misses) / len(misses)
    total = hits + len(misses)
    return {
        "hits": hits,
        "misses": len(misses),
        "hit_rate": round(hits / total, 4) if total else 0.0,
        "estimated_saving_usd": round(hits * mean_miss, 6),
        "basis": "assumed - hits times the mean writing cost of a miss",
    }


def suspects(store: "MeterStore", window_seconds: float = 3600.0,
             episodes: int = 30, spend_usd: float = 1.00,
             now: float = 0.0) -> list[dict]:
    """Listeners worth looking at, and why - never an automatic ban.

    Two thresholds because the two abuses look different. Volume catches a
    script hammering the endpoint; spend catches someone asking for 10-minute
    researched episodes all afternoon, which is few requests and a lot of
    money. Either alone misses the other.

    This reports. It does not act: an automatic block on a metering signal
    would eventually lock out a real listener on a bad heuristic, and there is
    no way for them to tell anyone. `_rate_limit` remains the thing that
    actually paces requests.
    """
    now = now or time.time()
    rows = store.rows(since=now - window_seconds, until=now + 1)
    by_user: dict[str, dict] = {}
    for r in rows:
        u = by_user.setdefault(r["user_id"], {
            "user_id": r["user_id"], "plan": r["plan"], "episodes": 0,
            "cost_usd": 0.0, "reasons": []})
        u["episodes"] += 1
        u["cost_usd"] += r["cost_usd"]
    flagged = []
    for u in by_user.values():
        u["cost_usd"] = round(u["cost_usd"], 6)
        if u["episodes"] >= episodes:
            u["reasons"].append(
                f"{u['episodes']} episodes in {window_seconds / 3600:.1f}h "
                f"(threshold {episodes})")
        if u["cost_usd"] >= spend_usd:
            u["reasons"].append(
                f"${u['cost_usd']:.2f} in {window_seconds / 3600:.1f}h "
                f"(threshold ${spend_usd:.2f})")
        if u["reasons"]:
            flagged.append(u)
    return sorted(flagged, key=lambda u: u["cost_usd"], reverse=True)
