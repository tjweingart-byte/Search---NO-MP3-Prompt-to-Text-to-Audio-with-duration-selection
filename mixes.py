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

import hashlib
import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Optional, Sequence

from topics import BANK_BY_ID, tags_for_text

log = logging.getLogger(__name__)

MAX_NAME = 60
MAX_QUERY = 200
MAX_TOPICS = 20
MAX_MIXES_PER_USER = 30


class MixError(ValueError):
    """Something the listener did wrong, phrased so it can be shown to them."""


@dataclass(frozen=True)
class MixItem:
    """One line in a mix: a bank topic, or something the listener typed.

    Both play the same way - a query goes to the pipeline and the shared
    script cache - but they cost differently, and the interface says so. A
    bank topic is shared by everyone who has it in a mix, so the second
    listener's copy is free. A typed one is only shared with people who happen
    to phrase the same question the same way, which for a niche question is
    nobody: it is a script a day, for one person.
    """

    id: str
    title: str
    query: str
    custom: bool
    subtitle: str = ""
    icon: str = "leaf"

    def as_dict(self) -> dict:
        return {
            "id": self.id, "title": self.title, "query": self.query,
            "custom": self.custom, "subtitle": self.subtitle, "icon": self.icon,
        }


def _bank_item(topic_id: str) -> MixItem:
    topic = BANK_BY_ID[topic_id]
    return MixItem(topic.id, topic.title, topic.query, False, topic.subtitle, topic.icon)


def custom_item(query: str, title: str = "") -> MixItem:
    """A topic the listener typed. Its id is derived from the query, so the
    same question added twice is one entry rather than two."""
    query = " ".join(str(query).split())[:MAX_QUERY]
    if not query:
        raise MixError("Type what you want to hear about.")
    ident = "q:" + hashlib.sha1(query.lower().encode("utf-8")).hexdigest()[:10]
    title = " ".join(str(title).split())[:MAX_NAME] or query[:1].upper() + query[1:]
    tags = tags_for_text(query)
    return MixItem(ident, title, query, True, "Added by you", _ICON_FOR_TAG.get(
        tags[0] if tags else "", "leaf"))


#: A typed topic still deserves a picture. Reuses the bank's icon vocabulary.
_ICON_FOR_TAG = {
    "sports": "sports", "business": "business", "money": "business",
    "tech": "tech", "science": "rocket", "health": "leaf",
    "culture": "music", "world": "business",
}


@dataclass
class Mix:
    id: str
    user_id: str
    name: str
    items: list[MixItem]
    created_at: float
    updated_at: float
    #: Public mixes appear on the listener's profile. Private is the default:
    #: a mix is a routine, and a routine is personal until someone decides
    #: otherwise.
    public: bool = False

    @property
    def topic_ids(self) -> list[str]:
        """Bank topics only - what the shared-cost design is measured on."""
        return [i.id for i in self.items if not i.custom]

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "topic_ids": list(self.topic_ids),
            "items": [i.as_dict() for i in self.items],
            # Kept for anything still reading `topics`; bank entries only.
            "topics": [i.as_dict() for i in self.items if not i.custom],
            "custom_count": sum(1 for i in self.items if i.custom),
            "public": self.public,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def clean_name(name: str) -> str:
    name = " ".join(str(name).split())[:MAX_NAME]
    if not name:
        raise MixError("Give the mix a name.")
    return name


def clean_items(raw: Sequence) -> list[MixItem]:
    """Normalise whatever the interface sent into an ordered list of items.

    Accepts bank ids as bare strings, and typed topics as
    `{"query": "...", "title": "..."}`. De-duplicated, order preserved.

    An unknown bank id is an error rather than something to drop quietly: a
    mix that silently loses a topic looks like the app forgot, which is the
    kind of invisible failure this project keeps paying for. A *typed* topic
    cannot be unknown - it is whatever they wrote.
    """
    items: list[MixItem] = []
    seen: set[str] = set()
    for entry in raw:
        if isinstance(entry, str):
            if entry not in BANK_BY_ID:
                raise MixError(f"There is no topic called {entry!r}.")
            item = _bank_item(entry)
        elif isinstance(entry, dict) and entry.get("query"):
            item = custom_item(entry["query"], entry.get("title", ""))
        elif isinstance(entry, dict) and entry.get("id") in BANK_BY_ID:
            item = _bank_item(entry["id"])
        else:
            raise MixError("A mix entry needs either a topic id or a question.")
        if item.id not in seen:
            seen.add(item.id)
            items.append(item)
    if len(items) > MAX_TOPICS:
        raise MixError(f"A mix holds up to {MAX_TOPICS} topics.")
    return items


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
            # Mixes shipped holding bank ids only. Typed topics need more than
            # an id, so the full ordered list moved to JSON; `topic_ids` stays
            # as the bank-only view an older row would have written.
            try:
                conn.execute("ALTER TABLE mixes ADD COLUMN items TEXT NOT NULL DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE mixes ADD COLUMN public INTEGER NOT NULL DEFAULT 0")
            except sqlite3.OperationalError:
                pass

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    def _row_to_mix(self, row) -> Mix:
        items: list[MixItem] = []
        raw = row[6] if len(row) > 6 else ""
        if raw:
            try:
                items = [
                    MixItem(d["id"], d["title"], d["query"], d["custom"],
                            d.get("subtitle", ""), d.get("icon", "leaf"))
                    for d in json.loads(raw)
                ]
            except Exception:
                log.exception("unreadable mix items; falling back to bank ids")
        if not items:
            items = [_bank_item(t) for t in row[3].split(",") if t in BANK_BY_ID]
        return Mix(row[0], row[1], row[2], items, row[4], row[5],
                   bool(row[7]) if len(row) > 7 else False)

    def public_for_user(self, user_id: str) -> list[Mix]:
        """What this listener has chosen to show on their profile."""
        return [m for m in self.list_for_user(user_id) if m.public]

    def list_for_user(self, user_id: str) -> list[Mix]:
        if not user_id:
            return []
        try:
            rows = self._conn().execute(
                "SELECT id, user_id, name, topic_ids, created_at, updated_at, items, public"
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
                "SELECT id, user_id, name, topic_ids, created_at, updated_at, items, public"
                " FROM mixes WHERE id = ? AND user_id = ?",
                (mix_id, user_id),
            ).fetchone()
        except Exception:
            log.exception("could not read mix")
            return None
        return self._row_to_mix(row) if row else None

    def create(self, user_id: str, name: str, topic_ids: Sequence = ()) -> Mix:
        if not user_id:
            raise MixError("No listener id; mixes are saved per person.")
        name = clean_name(name)
        items = clean_items(topic_ids)
        existing = self.list_for_user(user_id)
        if len(existing) >= MAX_MIXES_PER_USER:
            raise MixError(f"You already have {MAX_MIXES_PER_USER} mixes.")
        if any(m.name.lower() == name.lower() for m in existing):
            raise MixError(f"You already have a mix called {name}.")
        now = time.time()
        mix = Mix(uuid.uuid4().hex[:12], user_id, name, items, now, now)
        self._conn().execute(
            "INSERT INTO mixes (id, user_id, name, topic_ids, created_at, updated_at, items)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (mix.id, user_id, name, ",".join(mix.topic_ids), now, now,
             json.dumps([i.as_dict() for i in items])),
        )
        return mix

    def update(
        self,
        user_id: str,
        mix_id: str,
        name: Optional[str] = None,
        topic_ids: Optional[Sequence] = None,
        public: Optional[bool] = None,
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
            mix.items = clean_items(topic_ids)
        if public is not None:
            mix.public = bool(public)
        mix.updated_at = time.time()
        self._conn().execute(
            "UPDATE mixes SET name = ?, topic_ids = ?, updated_at = ?, items = ?,"
            " public = ? WHERE id = ? AND user_id = ?",
            (mix.name, ",".join(mix.topic_ids), mix.updated_at,
             json.dumps([i.as_dict() for i in mix.items]), int(mix.public),
             mix_id, user_id),
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
