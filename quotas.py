"""Quotas: the ceiling a tier promises, counted and enforced.

`metering.py` records what a listener actually cost, after the fact, and
deliberately does not enforce anything - it says so out loud, because a ledger
that also blocks is a ledger you cannot trust to be complete. This module is
the other half: a promise made in advance, counted before the money is spent.

They must stay separate. When they disagree the ledger is right, because it
recorded a thing that happened and this recorded a thing that was allowed.

## What is counted

Two resources, from `entitlements.RESOURCES`:

* **episode** - anything that may write a script and always synthesises audio.
  Search, myFAM, DailyFAM, Go Deeper. A cache hit still counts: the listener
  heard an episode and the GPU produced it, and only the Claude call was saved.
* **explore** - a replay the pipeline provably cannot satisfy by generating.
  Counted separately and far more loosely, because it costs GPU seconds and
  nothing else.

## Windows are calendar windows, in UTC

A day is a UTC calendar day and a week is a UTC ISO week, not a rolling 24
hours. Rolling is fairer - it cannot be gamed by waiting until 00:01 - but it
needs a row per event rather than a counter, and it cannot answer "when do I
get more?" with a time the listener can plan around. A ceiling nobody can see
the end of feels like a fault rather than a limit.

The cost of that choice, stated so it is a known trade: a listener in
California gets their reset in the late afternoon. `resets_at` is returned with
every verdict so the interface can say when, in their own clock.

## Reserve, then refund

The obvious shape - check, generate, then count - double-spends under
concurrency: two requests both check against the same count and both pass. So
a spend is **reserved first**, atomically, and refunded if the episode never
happened. `/api/audio` primes its stream before returning a status code, which
is exactly the window in which a refund is still honest.

An over-refund is impossible (a refund never takes a counter below zero) and a
lost refund costs the listener one episode of allowance, which is the direction
to fail in.

## What this cannot do, and it matters

An anonymous listener is a session cookie, and a session cookie can be thrown
away. Someone who wants more than the free tier allows can clear it and have a
fresh allowance, and nothing here stops them - the id is server-minted, so it
cannot be forged, but it can be *abandoned*.

That is a deliberate consequence of the settled constraint that an account
gates what is kept and never what is heard: the alternative is a login in front
of the first word. So the free quota is a **budget shaped like a limit**, not a
security control. If the beta shows it being routinely walked around, the
answer is device attestation or an account requirement for generation - both of
which are product decisions, and neither of which belongs in a counter.
"""
from __future__ import annotations

import calendar
import datetime as _dt
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Optional

import entitlements
from paths import data_path

log = logging.getLogger(__name__)

#: Counters older than this are pruned. Long enough that a weekly window is
#: never swept out from under a listener who is mid-week, short enough that the
#: table does not grow forever.
KEEP_SECONDS = 60 * 86400


class QuotaExceeded(Exception):
    """The listener is over their allowance.

    Carries the whole verdict rather than only a message, because the useful
    response is not "no" - it is "no, you have had five today, and you get five
    more at 00:00 UTC", which the interface can turn into an upgrade prompt
    that is honest about what upgrading buys.
    """

    def __init__(self, verdict: "Verdict") -> None:
        super().__init__(verdict.message)
        self.verdict = verdict


def window_key(window: str, at: float) -> str:
    """`day:2026-09-10` or `week:2026-W37`, in UTC.

    The string is the identity of the window, so it is stored rather than
    recomputed from a timestamp on read - a stored key cannot be reinterpreted
    by a later change to this function, which is the kind of silent
    reinterpretation that turns a quota into a mystery.
    """
    when = _dt.datetime.fromtimestamp(at, _dt.timezone.utc)
    if window == "week":
        year, week, _day = when.isocalendar()
        return f"week:{year}-W{week:02d}"
    return f"day:{when:%Y-%m-%d}"


def window_end(window: str, at: float) -> float:
    """When the current window ends, in epoch seconds. The moment the listener
    gets their allowance back, and the only number this module returns that
    they can act on."""
    when = _dt.datetime.fromtimestamp(at, _dt.timezone.utc)
    if window == "week":
        # Monday of next week, 00:00 UTC.
        start = when - _dt.timedelta(days=when.weekday())
        nxt = (start + _dt.timedelta(days=7)).replace(
            hour=0, minute=0, second=0, microsecond=0)
    else:
        nxt = (when + _dt.timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0)
    return calendar.timegm(nxt.utctimetuple())


@dataclass(frozen=True)
class Verdict:
    """What the listener may do, and what to tell them if they may not."""

    allowed: bool
    resource: str
    tier: str
    window: str
    #: How much of the allowance is gone, *including* the reservation this
    #: verdict came from when it was granted.
    used: int
    #: `entitlements.UNLIMITED` when there is no ceiling.
    limit: int
    resets_at: float
    message: str = ""

    @property
    def unlimited(self) -> bool:
        return self.limit == entitlements.UNLIMITED

    @property
    def remaining(self) -> int:
        if self.unlimited:
            return entitlements.UNLIMITED
        return max(0, self.limit - self.used)

    def as_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "resource": self.resource,
            "tier": self.tier,
            "window": self.window,
            "used": self.used,
            "limit": self.limit,
            "unlimited": self.unlimited,
            "remaining": self.remaining,
            "resets_at": self.resets_at,
            "message": self.message,
        }


def _refusal(resource: str, tier_name: str, limit: "entitlements.Limit",
             used: int, at: float) -> str:
    """The sentence a listener reads when they run out.

    Says the number, says when it comes back, and says what would change it -
    in that order, because the first two are facts they can act on and the
    third is a sales pitch. A refusal that leads with the pitch reads as a
    paywall dressed up as an error.
    """
    when = _dt.datetime.fromtimestamp(window_end(limit.window, at), _dt.timezone.utc)
    thing = "episodes" if resource == "episode" else "Explore episodes"
    per = "today" if limit.window == "day" else "this week"
    back = when.strftime("%H:%M UTC on %-d %B") if limit.window == "week" \
        else when.strftime("%H:%M UTC")
    line = (f"That is all {limit.count} of your {thing} for {per}. "
            f"You get more at {back}.")
    if tier_name != entitlements.TIERS[-1]:
        line += " A bigger plan lifts the limit."
    return line


class QuotaStore:
    """Counters, one row per (listener, resource, window).

    Its own database rather than a table beside the accounts: this is written
    on the hot path of every episode, and the store holding password hashes is
    the last one that should be taking a write lock several times a minute.
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = data_path("QUOTAS_DB", "quotas.db", path)
        self._local = threading.local()
        self._pruned_at = 0.0
        with self._conn() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS counters (
                       user_id    TEXT NOT NULL,
                       resource   TEXT NOT NULL,
                       window_key TEXT NOT NULL,
                       count      INTEGER NOT NULL DEFAULT 0,
                       updated    REAL NOT NULL,
                       PRIMARY KEY (user_id, resource, window_key)
                   )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS counters_updated"
                         " ON counters(updated)")

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    # --- reading ----------------------------------------------------------

    def used(self, user_id: str, resource: str, window: str,
             at: float = 0.0) -> int:
        now = at or time.time()
        try:
            row = self._conn().execute(
                "SELECT count FROM counters"
                " WHERE user_id = ? AND resource = ? AND window_key = ?",
                (user_id, resource, window_key(window, now)),
            ).fetchone()
        except Exception:
            log.exception("could not read a quota counter")
            return 0
        return int(row[0]) if row else 0

    def status(self, user_id: str, tier_name: str, resource: str,
               at: float = 0.0) -> Verdict:
        """Where this listener stands, without spending anything.

        Never raises and never refuses: a status read that could fail would
        make the interface unable to say what the allowance is, which is worse
        than the allowance being wrong.
        """
        now = at or time.time()
        tier_name = entitlements.normalise(tier_name)
        limit = entitlements.limit_for(tier_name, resource)
        if limit is None or limit.unlimited:
            window = limit.window if limit else "day"
            return Verdict(True, resource, tier_name, window,
                           self.used(user_id, resource, window, now),
                           entitlements.UNLIMITED, window_end(window, now))
        used = self.used(user_id, resource, limit.window, now)
        allowed = used < limit.count
        return Verdict(
            allowed, resource, tier_name, limit.window, used, limit.count,
            window_end(limit.window, now),
            "" if allowed else _refusal(resource, tier_name, limit, used, now),
        )

    # --- spending ---------------------------------------------------------

    def reserve(self, user_id: str, tier_name: str, resource: str,
                at: float = 0.0) -> Verdict:
        """Take one from the allowance, or raise `QuotaExceeded`.

        The increment and the test happen inside one `BEGIN IMMEDIATE`, so two
        requests arriving together cannot both see the same count and both
        pass. Under the previous check-then-generate shape that race spent a
        GPU second and a Claude call that the tier had not bought.
        """
        now = at or time.time()
        tier_name = entitlements.normalise(tier_name)
        limit = entitlements.limit_for(tier_name, resource)

        if not settings_enforcing():
            window = limit.window if limit else "day"
            return Verdict(True, resource, tier_name, window, 0,
                           entitlements.UNLIMITED, window_end(window, now))

        if limit is None or limit.unlimited:
            # Still counted. An unlimited tier is not an unmeasured one, and
            # the report that says what a listener costs is only as good as
            # the rows underneath it.
            window = limit.window if limit else "day"
            count = self._bump(user_id, resource, window, 1, now)
            return Verdict(True, resource, tier_name, window, count,
                           entitlements.UNLIMITED, window_end(window, now))

        count = self._bump(user_id, resource, limit.window, 1, now)
        if count > limit.count:
            # Put it back: they did not get an episode, so they should not
            # have been charged for one. Refunding rather than never taking it
            # is what makes the check atomic.
            self._bump(user_id, resource, limit.window, -1, now)
            verdict = Verdict(
                False, resource, tier_name, limit.window, limit.count,
                limit.count, window_end(limit.window, now),
                _refusal(resource, tier_name, limit, limit.count, now),
            )
            raise QuotaExceeded(verdict)
        return Verdict(True, resource, tier_name, limit.window, count,
                       limit.count, window_end(limit.window, now))

    def refund(self, user_id: str, resource: str, window: str,
               at: float = 0.0) -> None:
        """Give one back, for an episode that was reserved and never produced.

        Never fails the request it is unwinding: a failed refund costs one
        episode of allowance, and an exception raised here would replace a
        listener's failed episode with a second, different failure.
        """
        try:
            self._bump(user_id, resource, window, -1, at or time.time())
        except Exception:
            log.exception("could not refund a quota reservation for %r", user_id)

    def _bump(self, user_id: str, resource: str, window: str, delta: int,
              now: float) -> int:
        key = window_key(window, now)
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                "INSERT INTO counters (user_id, resource, window_key, count, updated)"
                " VALUES (?, ?, ?, 0, ?)"
                " ON CONFLICT (user_id, resource, window_key) DO NOTHING",
                (user_id, resource, key, now),
            )
            # `MAX(0, ...)` so a refund can never drive a counter negative -
            # which would hand out free episodes to whoever managed to cause
            # two refunds for one reservation.
            conn.execute(
                "UPDATE counters SET count = MAX(0, count + ?), updated = ?"
                " WHERE user_id = ? AND resource = ? AND window_key = ?",
                (delta, now, user_id, resource, key),
            )
            row = conn.execute(
                "SELECT count FROM counters"
                " WHERE user_id = ? AND resource = ? AND window_key = ?",
                (user_id, resource, key),
            ).fetchone()
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:  # noqa: BLE001 - nothing left to do about it
                pass
            raise
        self._maybe_prune(now)
        return int(row[0]) if row else 0

    # --- housekeeping -----------------------------------------------------

    def forget(self, user_id: str) -> int:
        """Erase one listener's counters. Called by account deletion.

        Deleting these is safe in a way that deleting their *cost* rows is not:
        a counter is a promise about the future, not a record of money spent.
        """
        cur = self._conn().execute(
            "DELETE FROM counters WHERE user_id = ?", (user_id,))
        return cur.rowcount or 0

    def _maybe_prune(self, now: float) -> None:
        if now - self._pruned_at < 3600:
            return
        self._pruned_at = now
        try:
            self._conn().execute("DELETE FROM counters WHERE updated < ?",
                                 (now - KEEP_SECONDS,))
        except Exception:
            log.exception("could not prune quota counters; continuing")


def settings_enforcing() -> bool:
    """Whether quotas bite on this server.

    Read through `config` at call time rather than captured at import, so a
    test - and `demo.sh`, which turns them off and says so - gets what it set.
    Imported inside the function because config imports nothing from here and
    this module is imported by app.py before settings are final.
    """
    try:
        from config import settings
        return bool(settings.enforce_quotas)
    except Exception:  # noqa: BLE001 - a missing setting must not open the gate
        log.exception("could not read enforce_quotas; enforcing")
        return True
