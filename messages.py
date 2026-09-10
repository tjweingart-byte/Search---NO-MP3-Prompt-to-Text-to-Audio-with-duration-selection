"""Sending an episode to somebody, inside the app.

An **echo** (social.py) is a broadcast: you push an episode at everybody who
follows you and you do not choose who. A **share** is directed - one person,
one episode, deliberately - and it is the thing people actually do with
something they liked.

Both rest on the same property, which is what makes the whole social layer
affordable: **a share generates nothing.** It is a row pointing at a query
whose script already exists, so sending an episode to ten people costs ten rows
and not ten episodes. The recipient's tap is what synthesises audio, against
their own allowance, from the same cached script. That is the same design the
browse surfaces use and the reason the cost model survives a social feature at
all.

## Threads are between two listeners, and their id is derived

A thread id is the two user ids sorted and joined. Derived rather than
allocated, so:

* opening a conversation needs no write - the id is knowable from both sides
  before anything has been said;
* two people opening the same conversation simultaneously cannot create two
  threads, which is the classic bug in this shape and produces a split history
  that nobody can merge afterwards.

Group threads are not here. Not because they are hard, but because a group
changes what a share *means* - a share to a group is closer to an echo, and the
right time to decide that is when somebody wants one, not now.

## What a message may be

Three kinds, and the distinction is what the recipient's client renders:

* `episode` - an episode, carried as its query and length. Never as audio, and
  never as a script: the script lives in the shared cache and is fetched at
  play time, so a share stays one small row however long the episode is.
* `text` - what somebody typed alongside it.
* `system` - "Ana followed you", and anything else the app says on its own
  behalf. Kept in the same table so a thread is one ordered list rather than
  two that have to be interleaved on read.

## Reading, and what is deliberately not counted

`read_at` is stamped when a thread is opened, and the unread count is derived
from it. There are no delivery receipts and no typing indicators: both are
promises about somebody else's attention, and this app has a rule against
inventing state it does not have.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Optional

from paths import data_path

log = logging.getLogger(__name__)

MAX_TEXT = 1000
MAX_TITLE = 200
MAX_QUERY = 500

KINDS = ("episode", "text", "system")


class MessageError(ValueError):
    """Something the sender can fix, phrased so it can be shown to them."""


def thread_id(a: str, b: str) -> str:
    """The conversation between two listeners, derived from who they are.

    Sorted so that both sides compute the same string. This is the whole reason
    there is no threads table: an id nobody has to allocate cannot be allocated
    twice.
    """
    if not a or not b:
        raise MessageError("A conversation needs two people.")
    if a == b:
        raise MessageError("You cannot start a conversation with yourself.")
    return "|".join(sorted((a, b)))


@dataclass
class Message:
    id: int
    thread: str
    sender: str
    recipient: str
    kind: str
    text: str
    query: str
    minutes: int
    title: str
    at: float

    def as_dict(self, me: str = "") -> dict:
        return {
            "id": self.id,
            "thread": self.thread,
            # "mine" rather than the sender's id: the client needs to know
            # which side to draw the bubble on, and does not need somebody
            # else's listener id to do it.
            "mine": bool(me) and self.sender == me,
            "kind": self.kind,
            "text": self.text,
            "query": self.query,
            "minutes": self.minutes,
            "title": self.title,
            "at": self.at,
        }


class MessageStore:
    def __init__(self, path: str | None = None) -> None:
        self.path = data_path("MESSAGES_DB", "messages.db", path)
        self._local = threading.local()
        with self._conn() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS messages (
                       id        INTEGER PRIMARY KEY AUTOINCREMENT,
                       thread    TEXT NOT NULL,
                       sender    TEXT NOT NULL,
                       recipient TEXT NOT NULL,
                       kind      TEXT NOT NULL DEFAULT 'text',
                       text      TEXT NOT NULL DEFAULT '',
                       query     TEXT NOT NULL DEFAULT '',
                       minutes   INTEGER NOT NULL DEFAULT 0,
                       title     TEXT NOT NULL DEFAULT '',
                       at        REAL NOT NULL
                   )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS messages_thread"
                         " ON messages(thread, at)")
            conn.execute("CREATE INDEX IF NOT EXISTS messages_recipient"
                         " ON messages(recipient, at)")
            # When each person last opened each thread. One row per person per
            # thread rather than a flag on each message: marking a hundred
            # messages read is one write here and a hundred there.
            conn.execute(
                """CREATE TABLE IF NOT EXISTS reads (
                       thread  TEXT NOT NULL,
                       user_id TEXT NOT NULL,
                       read_at REAL NOT NULL,
                       PRIMARY KEY (thread, user_id)
                   )"""
            )

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    # --- sending ----------------------------------------------------------

    def send(self, sender: str, recipient: str, *, kind: str = "text",
             text: str = "", query: str = "", minutes: int = 0,
             title: str = "", at: float = 0.0) -> Message:
        """Put one message in a thread.

        An episode share carries the *query and the length* and nothing else.
        Not the script - that is in the shared cache, keyed on exactly those
        two things, so the recipient's play is a cache hit and the share cost
        one row. Not the audio, obviously: the whole product is that audio is
        made at play time.
        """
        if kind not in KINDS:
            raise MessageError(f"Unknown message kind {kind!r}.")
        thread = thread_id(sender, recipient)
        text = " ".join(str(text or "").split())[:MAX_TEXT]
        query = " ".join(str(query or "").split())[:MAX_QUERY]
        title = " ".join(str(title or "").split())[:MAX_TITLE]
        if kind == "episode" and not query:
            raise MessageError("That episode has no question attached to it.")
        if kind == "text" and not text:
            raise MessageError("Nothing to send.")
        now = at or time.time()
        cur = self._conn().execute(
            "INSERT INTO messages (thread, sender, recipient, kind, text,"
            " query, minutes, title, at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (thread, sender, recipient, kind, text, query,
             max(0, int(minutes or 0)), title, now),
        )
        return Message(cur.lastrowid, thread, sender, recipient, kind, text,
                       query, max(0, int(minutes or 0)), title, now)

    # --- reading ----------------------------------------------------------

    def thread(self, user_id: str, other_id: str, limit: int = 200) -> list[Message]:
        """One conversation, oldest first - which is the order it is read in."""
        tid = thread_id(user_id, other_id)
        try:
            rows = self._conn().execute(
                "SELECT id, thread, sender, recipient, kind, text, query,"
                " minutes, title, at FROM messages WHERE thread = ?"
                " ORDER BY at DESC, id DESC LIMIT ?", (tid, int(limit)),
            ).fetchall()
        except Exception:
            log.exception("could not read a thread")
            return []
        return [Message(*r) for r in reversed(rows)]

    def inbox(self, user_id: str, limit: int = 50) -> list[dict]:
        """Every conversation this listener is in, most recent first.

        One row per thread with its last message and unread count - what a
        message list shows. Assembled here rather than by the endpoint so that
        "what does a conversation look like in a list" has one answer.
        """
        try:
            rows = self._conn().execute(
                "SELECT m.thread, m.id, m.sender, m.recipient, m.kind, m.text,"
                "       m.query, m.minutes, m.title, m.at"
                "  FROM messages m"
                "  JOIN (SELECT thread, MAX(at) AS top FROM messages"
                "         WHERE sender = ? OR recipient = ?"
                "         GROUP BY thread) t"
                "    ON t.thread = m.thread AND t.top = m.at"
                " ORDER BY m.at DESC LIMIT ?",
                (user_id, user_id, int(limit)),
            ).fetchall()
        except Exception:
            log.exception("could not read the inbox")
            return []

        out = []
        seen = set()
        for r in rows:
            last = Message(r[1], r[0], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9])
            if last.thread in seen:
                continue  # two messages sharing a timestamp; one is enough
            seen.add(last.thread)
            other = last.recipient if last.sender == user_id else last.sender
            out.append({
                "thread": last.thread,
                "with": other,
                "last": last.as_dict(user_id),
                "unread": self.unread_in(user_id, last.thread),
            })
        return out

    def unread_in(self, user_id: str, thread: str) -> int:
        try:
            row = self._conn().execute(
                "SELECT read_at FROM reads WHERE thread = ? AND user_id = ?",
                (thread, user_id)).fetchone()
            since = row[0] if row else 0.0
            return int(self._conn().execute(
                "SELECT COUNT(*) FROM messages WHERE thread = ?"
                " AND recipient = ? AND at > ?",
                (thread, user_id, since)).fetchone()[0])
        except Exception:
            log.exception("could not count unread messages")
            return 0

    def unread_total(self, user_id: str) -> int:
        """What the badge on the Messages button shows."""
        try:
            rows = self._conn().execute(
                "SELECT m.thread, COUNT(*) FROM messages m"
                " LEFT JOIN reads r ON r.thread = m.thread AND r.user_id = ?"
                " WHERE m.recipient = ? AND m.at > COALESCE(r.read_at, 0)"
                " GROUP BY m.thread", (user_id, user_id)).fetchall()
        except Exception:
            log.exception("could not count unread messages")
            return 0
        return sum(int(r[1]) for r in rows)

    def mark_read(self, user_id: str, other_id: str, at: float = 0.0) -> None:
        tid = thread_id(user_id, other_id)
        now = at or time.time()
        try:
            self._conn().execute(
                "INSERT INTO reads (thread, user_id, read_at) VALUES (?, ?, ?)"
                " ON CONFLICT (thread, user_id) DO UPDATE SET read_at = ?"
                " WHERE read_at < ?", (tid, user_id, now, now, now))
        except Exception:
            log.exception("could not mark a thread read")

    # --- housekeeping -----------------------------------------------------

    def forget(self, user_id: str) -> int:
        """Erase this listener from every conversation they were in.

        Their side of a thread goes; the other person's messages stay, because
        those are that person's words and not this one's data. What the
        remaining side sees is a conversation with somebody who is no longer
        there - which is what actually happened.
        """
        removed = 0
        try:
            cur = self._conn().execute(
                "DELETE FROM messages WHERE sender = ?", (user_id,))
            removed += cur.rowcount or 0
            cur = self._conn().execute(
                "DELETE FROM reads WHERE user_id = ?", (user_id,))
            removed += cur.rowcount or 0
        except Exception:
            log.exception("could not erase messages for %r", user_id)
        return removed
