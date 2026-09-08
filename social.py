"""Echoes, and the little identity a listener needs to make one.

An **echo** is a deliberate act: someone finishes an episode and pushes it to
the people who follow them. The episode may well have reached those people
anyway - scripts are shared, so a popular question is already sitting in the
cache Explore reads from - but an echo changes what the card *says*. It stops
being "someone asked this" and becomes "Rachel sent you this", which is a
different reason to press play.

That is the whole design: an echo costs no generation at all. It is a row
pointing at a query that already exists, so the social layer is free in exactly
the way the browse surfaces are.

Identity is the minimum an echo needs to make sense: a name to put on it and a
handle to find it by. There are still no accounts - this is keyed on the same
anonymous per-device id as everything else, and is documented as such on the
profile page rather than dressed up as a login.

`people` is nonetheless the app's **listener table**, and `seen()` is what makes
it one. Until it existed a row appeared only when someone chose a name, so the
app could not answer "who is out there" or "when were they last here" for the
overwhelming majority of its listeners - they had a taste profile and no
record. Two things need that record. Recency: a daily feed has to know a
listener is still listening, and a decayed event log tells you what they liked,
not whether they came back. And identity itself: when accounts arrive, the work
is attaching credentials to rows that already exist, rather than inventing a
user table underneath live data.

What it deliberately does not store is anything nobody has supplied - no email,
no device fingerprint, no invented counters. `first_seen` and `last_seen` are
facts the server observed; `name` and `handle` are facts the listener typed.
"""
from __future__ import annotations

import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass

from paths import data_path

log = logging.getLogger(__name__)

MAX_NAME = 40
MAX_HANDLE = 24
_HANDLE_OK = re.compile(r"^[a-z0-9_.]{2,24}$")


class SocialError(ValueError):
    """Something the listener did wrong, phrased so it can be shown to them."""


@dataclass
class Echo:
    id: int
    user_id: str
    query: str
    title: str
    minutes: int
    thread: str
    at: float

    def as_dict(self, name: str = "", handle: str = "") -> dict:
        return {
            "id": self.id, "query": self.query, "title": self.title,
            "minutes": self.minutes, "thread": self.thread, "at": self.at,
            "by": name, "handle": handle,
        }


def clean_handle(handle: str) -> str:
    handle = str(handle).strip().lstrip("@").lower()[:MAX_HANDLE]
    if not _HANDLE_OK.match(handle):
        raise SocialError("A handle is 2-24 letters, numbers, dots or underscores.")
    return handle


class SocialStore:
    def __init__(self, path: str | None = None) -> None:
        self.path = path or data_path("SOCIAL_DB", "social.db")
        self._local = threading.local()
        with self._conn() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS people (
                       user_id TEXT PRIMARY KEY,
                       name    TEXT NOT NULL DEFAULT '',
                       handle  TEXT NOT NULL DEFAULT '',
                       joined  REAL NOT NULL
                   )"""
            )
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS people_handle"
                         " ON people(handle) WHERE handle != ''")
            # Added after the table shipped. `joined` already records the
            # first sighting, so it doubles as first_seen and is not renamed:
            # a rename would mean rewriting every reader for no new fact.
            try:
                conn.execute("ALTER TABLE people ADD COLUMN"
                             " last_seen REAL NOT NULL DEFAULT 0")
            except sqlite3.OperationalError:
                pass  # already there
            conn.execute("CREATE INDEX IF NOT EXISTS people_last_seen"
                         " ON people(last_seen)")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS echoes (
                       id      INTEGER PRIMARY KEY AUTOINCREMENT,
                       user_id TEXT NOT NULL,
                       query   TEXT NOT NULL,
                       title   TEXT NOT NULL DEFAULT '',
                       minutes INTEGER NOT NULL DEFAULT 0,
                       thread  TEXT NOT NULL DEFAULT '',
                       at      REAL NOT NULL
                   )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS echoes_user ON echoes(user_id, at)")
            # One echo per person per episode: echoing twice is the same
            # statement made twice, not two statements.
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS echoes_once"
                         " ON echoes(user_id, query, minutes)")

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    # --- who someone is ---------------------------------------------------

    def person(self, user_id: str) -> dict:
        row = None
        try:
            row = self._conn().execute(
                "SELECT name, handle, joined, last_seen FROM people WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        except Exception:
            log.exception("could not read person")
        return {
            "user_id": user_id,
            "name": row[0] if row else "",
            "handle": row[1] if row else "",
            "joined": row[2] if row else 0.0,
            "last_seen": row[3] if row else 0.0,
            # Whether the app has ever seen this id before, as opposed to
            # whether they got around to naming themselves. The profile page
            # needs to tell those apart; before `seen()` it could not.
            "known": bool(row),
        }

    def seen(self, user_id: str, at: float = 0.0) -> None:
        """Note that this listener exists and is here now.

        Called on the surfaces that already touch a database, never on the
        generation path: `/api/audio` has one job, and a write in front of the
        first word is the single cost this product refuses. A play still lands
        here, just after the stream has been handed over.

        Failure is swallowed for the same reason every other write in this
        module swallows it - knowing when someone last visited is worth less
        than the visit.
        """
        if not user_id:
            return
        now = at or time.time()
        try:
            self._conn().execute(
                "INSERT INTO people (user_id, name, handle, joined, last_seen)"
                " VALUES (?, '', '', ?, ?)"
                " ON CONFLICT(user_id) DO UPDATE SET last_seen = excluded.last_seen",
                (user_id[:64], now, now),
            )
        except Exception:
            log.exception("could not record listener; continuing")

    def active_since(self, since: float, limit: int = 500) -> list[str]:
        """Listeners seen since `since`, most recent first.

        The recency signal a daily feed needs: who to build tomorrow morning
        for. Deliberately returns ids and nothing else - what to build is the
        feed's question, not this table's.
        """
        try:
            rows = self._conn().execute(
                "SELECT user_id FROM people WHERE last_seen >= ?"
                " ORDER BY last_seen DESC LIMIT ?",
                (float(since), int(limit)),
            ).fetchall()
        except Exception:
            log.exception("could not read active listeners")
            return []
        return [r[0] for r in rows]

    def set_person(self, user_id: str, name: str, handle: str) -> dict:
        if not user_id:
            raise SocialError("No listener id.")
        name = " ".join(str(name).split())[:MAX_NAME]
        if not name:
            raise SocialError("Give yourself a name.")
        handle = clean_handle(handle)
        taken = self._conn().execute(
            "SELECT user_id FROM people WHERE handle = ? AND user_id != ?",
            (handle, user_id),
        ).fetchone()
        if taken:
            raise SocialError(f"@{handle} is taken.")
        existing = self.person(user_id)
        now = time.time()
        # Only name and handle are overwritten. `joined` and `last_seen` are
        # observations the server made and naming yourself is not new evidence
        # about either, so an update must not quietly reset them.
        self._conn().execute(
            "INSERT INTO people (user_id, name, handle, joined, last_seen)"
            " VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT(user_id) DO UPDATE SET name = excluded.name,"
            " handle = excluded.handle",
            (user_id, name, handle, existing["joined"] or now,
             existing["last_seen"] or now),
        )
        return self.person(user_id)

    # --- echoes -----------------------------------------------------------

    def echo(self, user_id: str, query: str, title: str, minutes: int,
             thread: str = "") -> Echo:
        if not user_id:
            raise SocialError("No listener id.")
        query = " ".join(str(query).split())[:300]
        if not query:
            raise SocialError("Nothing to echo.")
        now = time.time()
        conn = self._conn()
        # An echo of something already echoed just moves it to the top: the
        # listener's intent is "send this", not "send this twice".
        conn.execute(
            "INSERT INTO echoes (user_id, query, title, minutes, thread, at)"
            " VALUES (?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(user_id, query, minutes) DO UPDATE SET at = excluded.at,"
            " title = excluded.title, thread = excluded.thread",
            (user_id, query, str(title)[:200], int(minutes), str(thread)[:200], now),
        )
        row = conn.execute(
            "SELECT id, user_id, query, title, minutes, thread, at FROM echoes"
            " WHERE user_id = ? AND query = ? AND minutes = ?",
            (user_id, query, int(minutes)),
        ).fetchone()
        return Echo(*row)

    def unecho(self, user_id: str, query: str, minutes: int) -> bool:
        cur = self._conn().execute(
            "DELETE FROM echoes WHERE user_id = ? AND query = ? AND minutes = ?",
            (user_id, " ".join(str(query).split())[:300], int(minutes)),
        )
        return bool(cur.rowcount)

    def echoes_by(self, user_id: str, limit: int = 40) -> list[Echo]:
        try:
            rows = self._conn().execute(
                "SELECT id, user_id, query, title, minutes, thread, at FROM echoes"
                " WHERE user_id = ? ORDER BY at DESC LIMIT ?",
                (user_id, int(limit)),
            ).fetchall()
        except Exception:
            log.exception("could not read echoes")
            return []
        return [Echo(*r) for r in rows]

    def has_echoed(self, user_id: str, query: str, minutes: int) -> bool:
        try:
            return bool(self._conn().execute(
                "SELECT 1 FROM echoes WHERE user_id = ? AND query = ? AND minutes = ?",
                (user_id, " ".join(str(query).split())[:300], int(minutes)),
            ).fetchone())
        except Exception:
            return False

    def recent_echoes(self, limit: int = 200, exclude_user: str = "") -> dict:
        """(query, minutes) -> who echoed it, newest first.

        Used to label Explore cards. There is no follow graph yet, so every
        echo is visible to everyone; when follows exist this is where the
        filter goes, and nothing else has to change.
        """
        try:
            rows = self._conn().execute(
                "SELECT e.query, e.minutes, p.name, p.handle, e.at, e.user_id"
                " FROM echoes e LEFT JOIN people p ON p.user_id = e.user_id"
                " ORDER BY e.at DESC LIMIT ?", (int(limit),),
            ).fetchall()
        except Exception:
            log.exception("could not read echo labels")
            return {}
        out: dict = {}
        for query, minutes, name, handle, at, user_id in rows:
            if user_id == exclude_user:
                continue
            key = (query, minutes)
            if key not in out:
                out[key] = {"by": name or "Someone", "handle": handle or "", "at": at}
        return out
