"""Shared script cache.

The expensive part of an episode is the Claude call; synthesis runs at ~330x
realtime and costs essentially nothing. So the cache stores the **script**, not
the audio:

* a 10-minute script is ~9 KB; the same episode as PCM is ~26 MB (~2900x bigger)
* one cached script re-synthesises in milliseconds
* audio can't be shared across durations anyway, but the text is what cost money

A cache hit therefore costs zero API tokens and drops time-to-first-audio from
seconds to milliseconds.

Storage is SQLite because it needs no new dependency, survives restarts, and -
unlike a dict in the process - is shared by every worker on the machine, which
is the whole point when the users are different people. Swap in Redis by
implementing the same two methods if you run more than one machine.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
from typing import Optional, Protocol

from config import settings

log = logging.getLogger(__name__)

_PUNCT = re.compile(r"[^\w\s]")
_SPACE = re.compile(r"\s+")
# Words that carry no topic meaning, so "give me a recap of week 5" and
# "week 5 recap" land on the same entry.
_FILLER = {
    "a", "an", "the", "of", "for", "on", "in", "to", "and", "is", "are", "was",
    "were", "please", "give", "me", "tell", "about", "what", "whats", "who",
    "how", "why", "can", "you", "i", "want", "would", "like", "do", "does",
    "explain", "describe", "summarize", "summarise", "recap", "briefing",
    "podcast", "episode", "some", "any", "there", "this", "that", "with",
}
# A query mentioning any of these is asking about a moving target. Caching one
# for hours is how you serve yesterday's score as though it were live.
_VOLATILE = {
    "latest", "today", "todays", "tonight", "now", "current", "currently",
    "live", "breaking", "update", "updates", "score", "scores", "happening",
    "right", "moment", "just", "recent", "recently", "this", "upcoming",
    "tomorrow", "yesterday",
}
# Signals the query is about the person asking, which must never be shared.
# Deliberately limited to possessives and contact details: "me", "I" and "we"
# appear in perfectly ordinary phrasing ("give me a recap of week 5") and
# treating those as personal silently disables the cache for most real traffic.
_PERSONAL = re.compile(
    r"\b(my|mine|our|ours)\b"          # possessives - "my lab results"
    r"|[\w.+-]+@[\w-]+\.[\w.]+"        # email addresses
    r"|\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",  # phone numbers
    re.I,
)


def normalize_query(query: str) -> str:
    """Reduce a query to a canonical topic key.

    Lowercase, strip punctuation, drop filler words, sort the remainder. So all
    of these collapse to the same key:

        "Give me a recap of week 5 of the NFL season"
        "NFL week 5 recap"
        "recap: week 5, NFL"
    """
    text = _PUNCT.sub(" ", query.lower())
    tokens = [t for t in _SPACE.split(text) if t and t not in _FILLER]
    return " ".join(sorted(set(tokens)))


def is_shareable(query: str) -> bool:
    """Should this query's output ever be served to a different person?

    Anything that reads as personal is generated fresh and never stored. This is
    a blunt heuristic, deliberately biased towards not sharing.
    """
    return not _PERSONAL.search(query)


def research_words() -> set:
    """The words that make a question a researched one.

    Served to the interface so it can say "checking recent sources" *before*
    the request goes out, from the same list the server decides with. A second
    copy in JavaScript would drift, and the two would disagree about what the
    listener was told.
    """
    return set(_VOLATILE)


def needs_fresh_information(query: str) -> bool:
    """Does answering this honestly require something that happened recently?

    The same signal the cache already uses to decide that "latest news on X"
    goes stale in minutes while "why is the sky blue" is good for a month. It
    is exactly the question "should this episode be researched", so search now
    reads it too rather than being on or off for everything.

    Deliberately a keyword test, not a model call: classifying the query with a
    model would put a round trip in front of the first word, which is the one
    cost this product refuses - and being wrong costs one episode answered from
    memory that could have been fresher, not a broken episode.
    """
    tokens = set(_SPACE.split(_PUNCT.sub(" ", query.lower())))
    return bool(tokens & _VOLATILE)


def ttl_for(query: str) -> int:
    """How long a script for this query stays usable, in seconds.

    Freshness is the hard part of a shared cache. "Why is the sky blue" is good
    for a month; "latest news on X" is stale in minutes, and serving it from
    cache is worse than being slow. The heuristic below is intentionally
    conservative - see the note in README about upgrading it to a classifier.
    """
    tokens = set(_SPACE.split(_PUNCT.sub(" ", query.lower())))
    if tokens & _VOLATILE:
        return settings.cache_ttl_volatile
    return settings.cache_ttl_seconds


def cache_key(
    query: str,
    minutes: int,
    canonical: Optional[str] = None,
    canonical_context: str = "",
    searched: bool = False,
) -> str:
    """Key on the canonical topic, the duration, and the voice settings.

    Duration is part of the key because a 3-minute episode is structured
    differently from a trimmed 10-minute one - it is not the same script cut
    short. Model is included so changing MODEL doesn't serve stale output from
    the previous one.
    """
    payload = json.dumps(
        {
            "q": canonical or normalize_query(query),
            "m": int(minutes),
            # A follow-up is a different episode from the same words asked cold.
            "ctx": canonical_context or "",
            # A researched episode is a different thing from an instant one.
            "search": bool(searched),
            "model": settings.model,
            "wpm": settings.target_wpm,
            "v": 2,  # bump to invalidate everything after a prompt change
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ScriptCache(Protocol):
    def get(self, key: str) -> Optional[list[str]]: ...
    def put(
        self, key: str, sentences: list[str], ttl: int, query: str, thread: str = ""
    ) -> None: ...
    #: The go-deeper thread stored with the script, or "" if there was none.
    #: Kept beside the sentences rather than inside them so a replayed episode
    #: can never speak it by accident.
    def thread(self, key: str) -> str: ...
    #: Live entries, newest first. Explore replays these and never generates.
    def recent(self, limit: int = 40) -> list[dict]: ...


class MemoryScriptCache:
    """Process-local cache. Fine for tests and single-worker development."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[float, list[str], str, str, int]] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[list[str]]:
        entry = self._data.get(key)
        if not entry or entry[0] < time.time():
            self.misses += 1
            return None
        self.hits += 1
        return list(entry[1])

    def put(
        self, key: str, sentences: list[str], ttl: int, query: str = "",
        thread: str = "", minutes: int = 0
    ) -> None:
        self._data[key] = (time.time() + ttl, list(sentences), thread, query, int(minutes))

    def recent(self, limit: int = 40) -> list[dict]:
        live = [
            {"key": k, "query": v[3], "minutes": v[4], "created": v[0],
             "plays": 0, "thread": v[2]}
            for k, v in self._data.items()
            if v[0] >= time.time() and v[3] and v[4] > 0
        ]
        live.sort(key=lambda e: -e["created"])
        return live[:limit]

    def thread(self, key: str) -> str:
        entry = self._data.get(key)
        if not entry or entry[0] < time.time():
            return ""
        return entry[2]

    def stats(self) -> dict:
        return {"backend": "memory", "entries": len(self._data), "hits": self.hits, "misses": self.misses}


class SqliteScriptCache:
    """Cross-process cache with no new dependencies.

    Every uvicorn worker on the box shares one file, so a hit generated by one
    listener is immediately available to the next - which is the entire point of
    sharing output between different users.
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = path or settings.cache_path
        self._local = threading.local()
        with self._conn() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS scripts (
                       key       TEXT PRIMARY KEY,
                       expires   REAL NOT NULL,
                       created   REAL NOT NULL,
                       hits      INTEGER NOT NULL DEFAULT 0,
                       query     TEXT,
                       sentences TEXT NOT NULL
                   )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS scripts_expires ON scripts(expires)")
            # Added after the table shipped, so existing caches need widening
            # rather than recreating - a cache file is not worth a migration
            # framework, and losing it would only cost one regeneration.
            for column, ddl in (
                ("thread", "ALTER TABLE scripts ADD COLUMN thread TEXT NOT NULL DEFAULT ''"),
                # Explore replays cached scripts, and a script cannot be
                # replayed without knowing how long it was written to run.
                ("minutes", "ALTER TABLE scripts ADD COLUMN minutes INTEGER NOT NULL DEFAULT 0"),
            ):
                try:
                    conn.execute(ddl)
                except sqlite3.OperationalError:
                    pass  # already there

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
            # WAL lets readers proceed while another worker is writing, which
            # matters when several episodes are being generated at once.
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    def get(self, key: str) -> Optional[list[str]]:
        try:
            conn = self._conn()
            row = conn.execute(
                "SELECT sentences, expires FROM scripts WHERE key = ?", (key,)
            ).fetchone()
            if not row or row[1] < time.time():
                return None
            conn.execute("UPDATE scripts SET hits = hits + 1 WHERE key = ?", (key,))
            return json.loads(row[0])
        except Exception:
            # A cache is an optimisation. If it breaks, the episode is still
            # generated the slow way rather than failing.
            log.exception("script cache read failed; regenerating")
            return None

    def put(
        self, key: str, sentences: list[str], ttl: int, query: str = "",
        thread: str = "", minutes: int = 0
    ) -> None:
        if not sentences:
            return
        try:
            now = time.time()
            self._conn().execute(
                "INSERT OR REPLACE INTO scripts"
                " (key, expires, created, hits, query, sentences, thread, minutes)"
                " VALUES (?, ?, ?, 0, ?, ?, ?, ?)",
                (key, now + ttl, now, query[:500], json.dumps(sentences),
                 thread[:200], int(minutes)),
            )
        except Exception:
            log.exception("script cache write failed; continuing")

    def thread(self, key: str) -> str:
        try:
            row = self._conn().execute(
                "SELECT thread, expires FROM scripts WHERE key = ?", (key,)
            ).fetchone()
            if not row or row[1] < time.time():
                return ""
            return row[0] or ""
        except Exception:
            log.exception("script cache thread read failed")
            return ""

    def recent(self, limit: int = 40) -> list[dict]:
        """Live cache entries, newest first - the raw material for Explore.

        Only *shareable* queries are ever written here (see `is_shareable`),
        so everything in this table is already safe to show another listener.
        That property is what makes an Explore feed possible at all, and it is
        a reason to be careful about ever loosening the personal-query filter.

        Entries with no recorded duration are skipped rather than guessed at: a
        script written for one minute replayed as a five-minute episode would
        be padded with silence.
        """
        try:
            rows = self._conn().execute(
                "SELECT key, query, minutes, created, hits, thread FROM scripts"
                " WHERE expires >= ? AND query != '' AND minutes > 0"
                " ORDER BY created DESC LIMIT ?",
                (time.time(), int(limit)),
            ).fetchall()
        except Exception:
            log.exception("could not read recent scripts")
            return []
        return [
            {"key": r[0], "query": r[1], "minutes": r[2], "created": r[3],
             "plays": r[4], "thread": r[5] or ""}
            for r in rows
        ]

    def purge_expired(self) -> int:
        try:
            cur = self._conn().execute("DELETE FROM scripts WHERE expires < ?", (time.time(),))
            return cur.rowcount or 0
        except Exception:
            return 0

    def stats(self) -> dict:
        try:
            row = self._conn().execute(
                "SELECT COUNT(*), COALESCE(SUM(hits), 0) FROM scripts WHERE expires >= ?",
                (time.time(),),
            ).fetchone()
            return {"backend": "sqlite", "path": self.path, "entries": row[0], "hits_served": row[1]}
        except Exception:
            return {"backend": "sqlite", "path": self.path, "error": "unavailable"}


CANONICAL_PROMPT = """Reduce this listener request to a canonical topic label so \
that differently-worded requests for the SAME briefing collapse together.

Rules:
- Lowercase. Three to eight words. No punctuation.
- Keep every distinguishing detail: subject, number, season, year, place.
- Drop phrasing, politeness and format words ("give me", "a recap of", "podcast").
- Two requests that deserve the SAME briefing must produce the identical label.
  Two requests that deserve DIFFERENT briefings must not.

Examples:
  "Give me a recap of week 5 of the NFL season" -> nfl season week 5 recap
  "NFL week 5 recap" -> nfl season week 5 recap
  "what happened in week five of the nfl" -> nfl season week 5 recap
  "Why is the sky blue?" -> why the sky is blue

Output only the label."""


async def canonical_key(query: str, client) -> str:
    """Collapse equivalent phrasings that lexical normalisation cannot.

    `normalize_query` handles punctuation, word order and filler, but it is
    lexical: "week 5 of the NFL season" and "NFL week 5" differ by one real
    word ("season") and so miss each other. A small model closes that gap.

    The tradeoff is honest: this adds one fast call (~300-500 ms, ~$0.0002) to
    the front of every request, which is pure overhead on a miss and a large win
    on a hit. Worth enabling when traffic concentrates on popular topics;
    leave it off for a long tail of unique queries. Off by default.
    """
    try:
        message = await client.messages.create(
            model=settings.canonical_key_model,
            max_tokens=40,
            system=CANONICAL_PROMPT,
            messages=[{"role": "user", "content": query}],
        )
        if message.stop_reason == "refusal":
            return normalize_query(query)
        text = " ".join(b.text for b in message.content if b.type == "text")
        label = _SPACE.sub(" ", _PUNCT.sub(" ", text.lower())).strip()
        # Never let a chatty answer become the key.
        return label if 0 < len(label) <= 80 else normalize_query(query)
    except Exception:
        log.warning("canonical key lookup failed; using lexical key", exc_info=True)
        return normalize_query(query)


def build_cache() -> Optional[ScriptCache]:
    if not settings.cache_enabled:
        return None
    if settings.cache_backend == "memory":
        return MemoryScriptCache()
    return SqliteScriptCache()
