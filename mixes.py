"""playFAM: named daily mixes of topics.

A mix is a standing subscription, not a saved recording: it holds *topic ids*,
and each day the episodes for those topics are whatever today's briefing on
them is. That distinction is the whole design. Saving audio would mean a mix
goes stale the moment it is made, and would break the no-files rule the rest
of the product is built on; saving topic ids means "at the gym" is fresh every
morning and costs nothing to keep.

Mixes draw from the same shared bank as myFAM (`topics.py`), for the same
reason: two people with "Morning" mixes that both include the Fed episode
share one script through `cache.py`. A mix that could contain arbitrary
free-text queries would quietly undo that, so membership is validated against
the bank and unknown ids are rejected rather than silently stored.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Optional, Sequence

from topics import BANK_BY_ID

log = logging.getLogger(__name__)

MAX_NAME = 60
MAX_TOPICS = 20
MAX_MIXES_PER_USER = 30


class MixError(ValueError):
    """Something the listener did wrong, phrased so it can be shown to them."""


@dataclass
class Mix:
    id: str
    user_id: str
    name: str
    topic_ids: list[str]
    created_at: float
    updated_at: float

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "topic_ids": list(self.topic_ids),
            "topics": [
                BANK_BY_ID[t].as_dict() for t in self.topic_ids if t in BANK_BY_ID
            ],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def clean_name(name: str) -> str:
    name = " ".join(str(name).split())[:MAX_NAME]
    if not name:
        raise MixError("Give the mix a name.")
    return name


def clean_topics(topic_ids: Sequence[str]) -> list[str]:
    """Validated against the shared bank, de-duplicated, order preserved.

    An unknown id is an error rather than something to drop quietly: a mix
    that silently loses a topic looks like the app forgot, which is exactly
    the kind of invisible failure this project keeps paying for.
    """
    seen: list[str] = []
    for topic_id in topic_ids:
        if topic_id not in BANK_BY_ID:
            raise MixError(f"There is no topic called {topic_id!r}.")
        if topic_id not in seen:
            seen.append(topic_id)
    if len(seen) > MAX_TOPICS:
        raise MixError(f"A mix holds up to {MAX_TOPICS} topics.")
    return seen


class MixStore:
    def __init__(self, path: str = "mixes.db") -> None:
        self.path = path
        self._local = threading.local()
        with self._conn() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS mixes (
                       id         TEXT PRIMARY KEY,
                       user_id    TEXT NOT NULL,
                       name       TEXT NOT NULL,
                       topic_ids  TEXT NOT NULL DEFAULT '',
                       created_at REAL NOT NULL,
                       updated_at REAL NOT NULL
                   )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS mixes_user ON mixes(user_id, created_at)")

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    def _row_to_mix(self, row) -> Mix:
        return Mix(row[0], row[1], row[2],
                   [t for t in row[3].split(",") if t], row[4], row[5])

    def list_for_user(self, user_id: str) -> list[Mix]:
        if not user_id:
            return []
        try:
            rows = self._conn().execute(
                "SELECT id, user_id, name, topic_ids, created_at, updated_at"
                " FROM mixes WHERE user_id = ? ORDER BY created_at",
                (user_id,),
            ).fetchall()
        except Exception:
            log.exception("could not list mixes")
            return []
        return [self._row_to_mix(r) for r in rows]

    def get(self, user_id: str, mix_id: str) -> Optional[Mix]:
        try:
            row = self._conn().execute(
                "SELECT id, user_id, name, topic_ids, created_at, updated_at"
                " FROM mixes WHERE id = ? AND user_id = ?",
                (mix_id, user_id),
            ).fetchone()
        except Exception:
            log.exception("could not read mix")
            return None
        return self._row_to_mix(row) if row else None

    def create(self, user_id: str, name: str, topic_ids: Sequence[str] = ()) -> Mix:
        if not user_id:
            raise MixError("No listener id; mixes are saved per person.")
        name = clean_name(name)
        topics = clean_topics(topic_ids)
        existing = self.list_for_user(user_id)
        if len(existing) >= MAX_MIXES_PER_USER:
            raise MixError(f"You already have {MAX_MIXES_PER_USER} mixes.")
        if any(m.name.lower() == name.lower() for m in existing):
            raise MixError(f"You already have a mix called {name}.")
        now = time.time()
        mix = Mix(uuid.uuid4().hex[:12], user_id, name, topics, now, now)
        self._conn().execute(
            "INSERT INTO mixes (id, user_id, name, topic_ids, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (mix.id, user_id, name, ",".join(topics), now, now),
        )
        return mix

    def update(
        self,
        user_id: str,
        mix_id: str,
        name: Optional[str] = None,
        topic_ids: Optional[Sequence[str]] = None,
    ) -> Mix:
        mix = self.get(user_id, mix_id)
        if not mix:
            raise MixError("That mix no longer exists.")
        if name is not None:
            mix.name = clean_name(name)
            clash = [
                m for m in self.list_for_user(user_id)
                if m.id != mix_id and m.name.lower() == mix.name.lower()
            ]
            if clash:
                raise MixError(f"You already have a mix called {mix.name}.")
        if topic_ids is not None:
            mix.topic_ids = clean_topics(topic_ids)
        mix.updated_at = time.time()
        self._conn().execute(
            "UPDATE mixes SET name = ?, topic_ids = ?, updated_at = ?"
            " WHERE id = ? AND user_id = ?",
            (mix.name, ",".join(mix.topic_ids), mix.updated_at, mix_id, user_id),
        )
        return mix

    def delete(self, user_id: str, mix_id: str) -> bool:
        cur = self._conn().execute(
            "DELETE FROM mixes WHERE id = ? AND user_id = ?", (mix_id, user_id)
        )
        return bool(cur.rowcount)


#: Offered on an empty playFAM page. Starting from a named example is easier
#: than starting from a blank field, and these are only suggestions - the
#: listener names their own.
STARTER_MIXES = (
    ("Morning", ("fed-next-move", "ai-agents", "morning-mindset")),
    ("At the gym", ("training-load", "the-trade", "habits-research")),
    ("Wind down", ("sleep-science", "anxiety-loop", "hollywood-comebacks")),
)
