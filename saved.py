"""Save for later, and download - which are deliberately not the same thing.

The distinction is the whole design, and it is easy to get wrong because both
look like a bookmark from the outside:

* **Save for later** is a *pointer*: the question, the length, a title, a
  folder. It costs one row, there is no sensible limit on it, and playing one
  **needs the internet** - the episode is synthesised (or replayed from the
  shared cache) when it is tapped, exactly like every other episode in FAM.
* **Download** is *the audio on the device*: the listener's phone keeps the
  samples it already received, so it plays with the network off. It has a hard
  limit, and running out means clearing something.

A download is therefore an *upgrade to* a saved item rather than a separate
list - which is why the interface offers it as a question the moment something
is saved, and why one row carries both states.

## Where the audio lives, and why that keeps the settled constraint

CLAUDE.md: **no MP3, no audio files.** Raw PCM streams from the engine to the
client and is played as it arrives; writing a *file* is not compatible with
that.

A download does not break it, because **the server still writes nothing.** The
episode streams exactly as it always does, and the client retains the bytes it
was already sent - IndexedDB in a browser, the app's own container on iOS. No
file is created on the server, nothing is cached as audio, and there is no URL
anywhere that serves a stored episode. What changes is only that the listener's
own device stops throwing the samples away.

Two consequences that follow from that, and both matter:

1. **This module holds a registry, not audio.** It records that a listener
   claims to be holding an episode, so the limit can be enforced and the list
   can be shown. The bytes are somewhere it cannot see.
2. **The registry can drift.** A phone that is wiped, or a browser whose
   storage is evicted, still has rows here. So `release` exists, a client
   re-syncs by releasing what it no longer holds, and the count is treated as
   *what the listener has claimed* rather than as ground truth. Drift costs a
   slot, which is why the fix is one tap and not a support ticket.

Uncompressed PCM is 2.65 MB per minute, so a three-minute episode is about
8 MB and ten of them is 80 MB. That is fine on a phone and heavy in a browser,
and it is one more argument for Opus over the stream - which IOS_APP.md already
has as a prerequisite of the app rather than a scale question.

## Limits

Per tier, from `entitlements.max_downloads`, and a **standing capacity** rather
than a rate: unlike episodes per day, a download is not consumed by time. You
hold three, or you hold none, until you change it. That is why it is not in
`quotas.py` - a windowed counter would let somebody accumulate a new download
allowance every morning and never delete anything.

When the shelf is full the refusal names what to clear, because "you have
reached your limit" without a list is a dead end on a phone.
"""
from __future__ import annotations

import logging
import re
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Optional

import entitlements
from paths import data_path

log = logging.getLogger(__name__)

MAX_NAME = 40
MAX_TITLE = 200
MAX_QUERY = 500
#: How many folders one listener may have. Not a tier feature - it is a guard
#: against a script making a million of them, and nobody has forty folders.
MAX_FOLDERS = 40

#: 16-bit mono. The one number that turns a length into a size, so the popup
#: can say "about 8 MB" before the listener agrees to it rather than after.
BYTES_PER_SECOND = 22050 * 2


class SavedError(ValueError):
    """Something the listener can fix, phrased so it can be shown to them."""


class DownloadLimit(SavedError):
    """The shelf is full. Carries what to clear, because a limit without a
    remedy is a dead end - especially on a phone, where the listener cannot
    go and look somewhere else."""

    def __init__(self, message: str, candidates: list) -> None:
        super().__init__(message)
        self.candidates = candidates


def clean_name(name: str) -> str:
    name = " ".join(str(name or "").split())[:MAX_NAME]
    if not name:
        raise SavedError("Give the folder a name.")
    return name


def estimated_bytes(minutes: int) -> int:
    """What a download of this length will take on the device.

    Deliberately an estimate and named as one: an episode ends when it runs out
    of substance (duration is a ceiling, not a quota), so the real size is
    usually smaller. Over-stating is the right direction - a listener told
    8 MB and charged 6 is pleased, and the reverse is a bug report.
    """
    return int(max(1, int(minutes or 0)) * 60 * BYTES_PER_SECOND)


@dataclass
class SavedItem:
    id: str
    user_id: str
    folder_id: str
    query: str
    minutes: int
    title: str
    source: str
    created: float
    downloaded: bool
    bytes: int
    downloaded_at: float
    last_played: float

    def as_dict(self) -> dict:
        return {
            "id": self.id, "folder_id": self.folder_id, "query": self.query,
            "minutes": self.minutes, "title": self.title, "source": self.source,
            "created": self.created, "downloaded": self.downloaded,
            "bytes": self.bytes, "downloaded_at": self.downloaded_at,
            "last_played": self.last_played,
            "estimated_bytes": estimated_bytes(self.minutes),
        }


class SavedStore:
    def __init__(self, path: str | None = None) -> None:
        self.path = data_path("SAVED_DB", "saved.db", path)
        self._local = threading.local()
        with self._conn() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS folders (
                       id      TEXT PRIMARY KEY,
                       user_id TEXT NOT NULL,
                       name    TEXT NOT NULL,
                       created REAL NOT NULL
                   )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS folders_user"
                         " ON folders(user_id, created)")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS items (
                       id            TEXT PRIMARY KEY,
                       user_id       TEXT NOT NULL,
                       folder_id     TEXT NOT NULL DEFAULT '',
                       query         TEXT NOT NULL,
                       minutes       INTEGER NOT NULL DEFAULT 0,
                       title         TEXT NOT NULL DEFAULT '',
                       source        TEXT NOT NULL DEFAULT '',
                       created       REAL NOT NULL,
                       downloaded    INTEGER NOT NULL DEFAULT 0,
                       bytes         INTEGER NOT NULL DEFAULT 0,
                       downloaded_at REAL NOT NULL DEFAULT 0,
                       last_played   REAL NOT NULL DEFAULT 0
                   )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS items_user"
                         " ON items(user_id, created)")
            conn.execute("CREATE INDEX IF NOT EXISTS items_downloaded"
                         " ON items(user_id, downloaded)")
            # Saving the same episode twice is the same statement twice. The
            # unique key is (question, length) because that is also the script
            # cache's key - two saves that differ only in something the cache
            # ignores are one episode.
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS items_once"
                         " ON items(user_id, query, minutes)")

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    # --- folders ----------------------------------------------------------

    def folders(self, user_id: str) -> list[dict]:
        """The listener's folders, with a count each.

        There is no "All" row and no implicit default folder in the table: an
        item with an empty `folder_id` is simply unfiled, and the interface
        shows those first. A real default folder would need creating on first
        use, which is a write on a read path and a row for people who never
        make a folder at all.
        """
        try:
            rows = self._conn().execute(
                "SELECT f.id, f.name, f.created, COUNT(i.id)"
                " FROM folders f LEFT JOIN items i ON i.folder_id = f.id"
                " WHERE f.user_id = ? GROUP BY f.id ORDER BY f.created",
                (user_id,)).fetchall()
        except Exception:
            log.exception("could not read folders")
            return []
        return [{"id": r[0], "name": r[1], "created": r[2], "items": int(r[3])}
                for r in rows]

    def create_folder(self, user_id: str, name: str, at: float = 0.0) -> dict:
        name = clean_name(name)
        existing = self.folders(user_id)
        if len(existing) >= MAX_FOLDERS:
            raise SavedError(f"That is the most folders one listener can have "
                             f"({MAX_FOLDERS}). Rename or remove one first.")
        if any(f["name"].lower() == name.lower() for f in existing):
            raise SavedError(f"You already have a folder called {name}.")
        folder_id = "fld_" + secrets.token_urlsafe(8)
        now = at or time.time()
        self._conn().execute(
            "INSERT INTO folders (id, user_id, name, created) VALUES (?, ?, ?, ?)",
            (folder_id, user_id, name, now))
        return {"id": folder_id, "name": name, "created": now, "items": 0}

    def rename_folder(self, user_id: str, folder_id: str, name: str) -> dict:
        name = clean_name(name)
        cur = self._conn().execute(
            "UPDATE folders SET name = ? WHERE id = ? AND user_id = ?",
            (name, folder_id, user_id))
        if not cur.rowcount:
            raise SavedError("No such folder.")
        return {"id": folder_id, "name": name}

    def delete_folder(self, user_id: str, folder_id: str) -> int:
        """Remove a folder. **Its episodes are unfiled, not deleted.**

        Deleting somebody's saved episodes because they tidied up their folders
        is the kind of surprise that stops people using a feature at all - and
        a download inside it is bytes on their phone that would then be
        orphaned, held against their limit with nothing pointing at them.
        """
        moved = self._conn().execute(
            "UPDATE items SET folder_id = '' WHERE folder_id = ? AND user_id = ?",
            (folder_id, user_id)).rowcount or 0
        self._conn().execute("DELETE FROM folders WHERE id = ? AND user_id = ?",
                             (folder_id, user_id))
        return moved

    # --- saving -----------------------------------------------------------

    def save(self, user_id: str, query: str, minutes: int, *, title: str = "",
             source: str = "", folder_id: str = "", at: float = 0.0) -> SavedItem:
        """Save an episode for later. Idempotent on (question, length).

        Saving something already saved moves it into the folder given rather
        than failing: from the listener's side they pressed save and it is
        saved, which is true either way.
        """
        query = " ".join(str(query or "").split())[:MAX_QUERY]
        if not query:
            raise SavedError("There is no episode to save.")
        minutes = max(0, int(minutes or 0))
        title = " ".join(str(title or "").split())[:MAX_TITLE]
        folder_id = self._checked_folder(user_id, folder_id)
        now = at or time.time()

        existing = self.find(user_id, query, minutes)
        if existing:
            if folder_id and folder_id != existing.folder_id:
                self._conn().execute(
                    "UPDATE items SET folder_id = ? WHERE id = ?",
                    (folder_id, existing.id))
                return self.item(user_id, existing.id)
            return existing

        item_id = "sav_" + secrets.token_urlsafe(8)
        self._conn().execute(
            "INSERT INTO items (id, user_id, folder_id, query, minutes, title,"
            " source, created) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (item_id, user_id, folder_id, query, minutes, title, source[:40], now))
        return self.item(user_id, item_id)

    def _checked_folder(self, user_id: str, folder_id: str) -> str:
        if not folder_id:
            return ""
        row = self._conn().execute(
            "SELECT 1 FROM folders WHERE id = ? AND user_id = ?",
            (folder_id, user_id)).fetchone()
        if not row:
            raise SavedError("No such folder.")
        return folder_id

    def find(self, user_id: str, query: str, minutes: int) -> Optional[SavedItem]:
        row = self._conn().execute(
            "SELECT id FROM items WHERE user_id = ? AND query = ? AND minutes = ?",
            (user_id, query, int(minutes))).fetchone()
        return self.item(user_id, row[0]) if row else None

    def item(self, user_id: str, item_id: str) -> Optional[SavedItem]:
        try:
            row = self._conn().execute(
                "SELECT id, user_id, folder_id, query, minutes, title, source,"
                " created, downloaded, bytes, downloaded_at, last_played"
                " FROM items WHERE id = ? AND user_id = ?",
                (item_id, user_id)).fetchone()
        except Exception:
            log.exception("could not read a saved item")
            return None
        if not row:
            return None
        return SavedItem(row[0], row[1], row[2], row[3], row[4], row[5], row[6],
                         row[7], bool(row[8]), row[9], row[10], row[11])

    def items(self, user_id: str, folder_id: Optional[str] = None,
              downloaded_only: bool = False, limit: int = 500) -> list[SavedItem]:
        sql = ("SELECT id, user_id, folder_id, query, minutes, title, source,"
               " created, downloaded, bytes, downloaded_at, last_played"
               " FROM items WHERE user_id = ?")
        args: list = [user_id]
        if folder_id is not None:
            sql += " AND folder_id = ?"
            args.append(folder_id)
        if downloaded_only:
            sql += " AND downloaded = 1"
        sql += " ORDER BY created DESC LIMIT ?"
        args.append(int(limit))
        try:
            rows = self._conn().execute(sql, tuple(args)).fetchall()
        except Exception:
            log.exception("could not read saved items")
            return []
        return [SavedItem(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7],
                          bool(r[8]), r[9], r[10], r[11]) for r in rows]

    def remove(self, user_id: str, item_id: str) -> bool:
        """Unsave. Also releases the download slot, if it held one - an item
        that is gone cannot still be occupying space."""
        cur = self._conn().execute("DELETE FROM items WHERE id = ? AND user_id = ?",
                                   (item_id, user_id))
        return bool(cur.rowcount)

    def move(self, user_id: str, item_id: str, folder_id: str) -> Optional[SavedItem]:
        folder_id = self._checked_folder(user_id, folder_id)
        self._conn().execute(
            "UPDATE items SET folder_id = ? WHERE id = ? AND user_id = ?",
            (folder_id, item_id, user_id))
        return self.item(user_id, item_id)

    def played(self, user_id: str, item_id: str, at: float = 0.0) -> None:
        """Note that a saved episode was played, so the "what to clear" list
        can offer the ones nobody has been back to."""
        self._conn().execute(
            "UPDATE items SET last_played = ? WHERE id = ? AND user_id = ?",
            (at or time.time(), item_id, user_id))

    # --- downloads --------------------------------------------------------

    def download_status(self, user_id: str, tier_name: str) -> dict:
        """How much of the shelf is used. What the popup shows before asking."""
        held = self.items(user_id, downloaded_only=True)
        limit = entitlements.max_downloads(tier_name)
        return {
            "used": len(held),
            "limit": limit,
            "unlimited": limit == entitlements.UNLIMITED,
            "remaining": (entitlements.UNLIMITED if limit == entitlements.UNLIMITED
                          else max(0, limit - len(held))),
            "bytes": sum(i.bytes or estimated_bytes(i.minutes) for i in held),
            "tier": entitlements.normalise(tier_name),
        }

    def reserve_download(self, user_id: str, item_id: str, tier_name: str,
                         at: float = 0.0) -> SavedItem:
        """Take a slot on the shelf, or raise `DownloadLimit` saying what to
        clear.

        The server records the claim; the device holds the bytes. So this
        cannot verify that a download happened - only that the listener is
        entitled to one more and has said they are taking it.
        """
        item = self.item(user_id, item_id)
        if not item:
            raise SavedError("No such saved episode.")
        if item.downloaded:
            return item

        status = self.download_status(user_id, tier_name)
        if not status["unlimited"] and status["remaining"] <= 0:
            held = self.items(user_id, downloaded_only=True)
            # Least recently useful first: never played, then longest since.
            # Offering the *oldest* would suggest clearing the one they saved
            # first, which is often the one they keep on purpose.
            candidates = sorted(held, key=lambda i: (i.last_played or 0, i.downloaded_at))
            raise DownloadLimit(
                f"You are holding {status['used']} downloaded episodes, which "
                f"is all your plan keeps offline. Remove one to make room.",
                [c.as_dict() for c in candidates[:5]])

        now = at or time.time()
        self._conn().execute(
            "UPDATE items SET downloaded = 1, downloaded_at = ?, bytes = ?"
            " WHERE id = ? AND user_id = ?",
            (now, estimated_bytes(item.minutes), item_id, user_id))
        return self.item(user_id, item_id)

    def confirm_download(self, user_id: str, item_id: str, size: int) -> Optional[SavedItem]:
        """The client says how much it actually stored.

        Worth a round trip because the estimate is deliberately generous and
        the difference is what the listener sees on a storage screen. A
        confirmation that never arrives leaves the estimate standing, which is
        the safe direction.
        """
        self._conn().execute(
            "UPDATE items SET bytes = ? WHERE id = ? AND user_id = ? AND downloaded = 1",
            (max(0, int(size or 0)), item_id, user_id))
        return self.item(user_id, item_id)

    def release_download(self, user_id: str, item_id: str) -> bool:
        """Give the slot back. The episode stays saved.

        Also how a client re-syncs after its storage was evicted: release what
        it no longer holds. Deliberately separate from `remove` - "I need the
        space" and "I am not interested any more" are different requests, and
        merging them loses somebody's list when they were tidying their phone.
        """
        cur = self._conn().execute(
            "UPDATE items SET downloaded = 0, bytes = 0, downloaded_at = 0"
            " WHERE id = ? AND user_id = ? AND downloaded = 1",
            (item_id, user_id))
        return bool(cur.rowcount)

    # --- housekeeping -----------------------------------------------------

    def forget(self, user_id: str) -> int:
        removed = 0
        for table in ("items", "folders"):
            try:
                cur = self._conn().execute(
                    f"DELETE FROM {table} WHERE user_id = ?", (user_id,))
                removed += cur.rowcount or 0
            except Exception:
                log.exception("could not erase %s for %r", table, user_id)
        return removed
