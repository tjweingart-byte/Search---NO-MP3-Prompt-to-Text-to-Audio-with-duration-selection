"""What a listener *chose*, as opposed to what they did.

`topics.py` infers taste from an append-only log of behaviour, and is
deliberately good at it: a profile there is a query over the log rather than a
stored object, so it cannot drift out of step with what actually happened.

This module is the other half, and it exists because three of the things the
interface now needs cannot be inferred at all:

* **Interests**, chosen in the intro before there is any behaviour to read.
  A brand-new listener's "Made for you" shelf was honestly empty; six chosen
  facets give the ranker something on the very first open. `topics.taste`
  folds them in at roughly the weight of one play, so real listening overtakes
  a declared interest within an evening rather than being fought by it.
* **Language**, which is stored and not yet acted on - see LANGUAGES.
* **Whether they want the weekly recap.** "Do you want this popup" is a
  question with an answer; deriving it from behaviour would be a guess.

Being *stored* is the whole reason this is gated on having an account. Nothing
here works for an anonymous listener, by decision: a preference the server
keeps for you is precisely the class of thing an account is for. An anonymous
listener still sees the intro - their answers stay in their own browser, are
passed to the ranker for that request only, and the interface says so plainly
rather than implying they were saved.

**Why the recap week is a date and not a flag.** "Show it the first time they
open on or after Sunday" cannot be a boolean, because nothing clears it: a
listener who does not open the app until Wednesday must still get Sunday's
recap, and must not then get it again on Thursday. Storing the Sunday that
started the week they last saw it answers both - it is due whenever the
current week's Sunday differs from the stored one - and it needs no scheduled
job, which this app has no way to run anyway.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import topics
from paths import data_path

log = logging.getLogger(__name__)

#: How many interests the intro accepts. The interface disables further
#: selection at this number rather than validating after the fact, and this is
#: the same rule enforced on the way in - a cap only the client applies is not
#: a cap.
MAX_INTERESTS = 6

#: The facets an interest can be. Deliberately the *tag* vocabulary from
#: topics.py and not the 28-topic bank: a tag is what `taste` scores and what
#: `tags_for_text` maps a free search onto, so choosing tags feeds the ranker
#: directly. Choosing six of the twenty-eight topics would seed six tiles and
#: teach the feed nothing.
INTERESTS = tuple(topics.TAG_LABELS)

#: Offered in the intro and stored. **Not wired to generation**: every episode
#: is still written and spoken in English, whatever is chosen here. That is the
#: scope this pass was given, and saying so is the point - a language picker
#: that silently changes nothing is the "silent success" failure this project
#: has lost the most time to, so `/api/preferences` returns `language_active`
#: false and the interface prints it under the picker.
LANGUAGES = (
    {"code": "en", "label": "English", "endonym": "English"},
    {"code": "es", "label": "Spanish", "endonym": "Español"},
    {"code": "fr", "label": "French", "endonym": "Français"},
    {"code": "de", "label": "German", "endonym": "Deutsch"},
    {"code": "pt", "label": "Portuguese", "endonym": "Português"},
    {"code": "it", "label": "Italian", "endonym": "Italiano"},
    {"code": "hi", "label": "Hindi", "endonym": "हिन्दी"},
    {"code": "ar", "label": "Arabic", "endonym": "العربية"},
    {"code": "zh", "label": "Chinese", "endonym": "中文"},
    {"code": "ja", "label": "Japanese", "endonym": "日本語"},
)

LANGUAGE_CODES = frozenset(lang["code"] for lang in LANGUAGES)
DEFAULT_LANGUAGE = "en"

#: True once per-language generation actually exists. Read by /api/preferences
#: and printed in the interface, so the day it flips the claim flips with it.
LANGUAGE_ACTIVE = False


class PreferenceError(ValueError):
    """A choice that cannot be stored, phrased so a listener can act on it."""


def week_start(now: Optional[float] = None) -> str:
    """The Sunday that began the week `now` falls in, as YYYY-MM-DD (UTC).

    UTC rather than local time, because the server has no idea where the
    listener is and a recap that arrives a few hours early is a smaller wrong
    than one that arrives twice.
    """
    now = time.time() if now is None else now
    stamp = time.gmtime(now)
    # tm_wday is Monday=0 .. Sunday=6; a Sunday-started week needs Sunday=0.
    since_sunday = (stamp.tm_wday + 1) % 7
    return time.strftime("%Y-%m-%d", time.gmtime(now - since_sunday * 86400))


def clean_interests(values: Iterable[str]) -> tuple[str, ...]:
    """Known facets, de-duplicated, order preserved, capped."""
    seen: list[str] = []
    for raw in values or ():
        tag = str(raw).strip().lower()
        if not tag:
            continue
        if tag not in topics.TAG_LABELS:
            raise PreferenceError(f"{tag!r} is not one of the interests on offer.")
        if tag not in seen:
            seen.append(tag)
    if len(seen) > MAX_INTERESTS:
        raise PreferenceError(f"Choose at most {MAX_INTERESTS} interests.")
    return tuple(seen)


def clean_language(code: str) -> str:
    lang = (code or "").strip().lower()
    if not lang:
        return DEFAULT_LANGUAGE
    if lang not in LANGUAGE_CODES:
        raise PreferenceError(f"{code!r} is not a language this app offers.")
    return lang


@dataclass(frozen=True)
class Preferences:
    """One listener's declared settings. Absent rows read as the defaults."""

    user_id: str
    interests: tuple[str, ...] = ()
    language: str = DEFAULT_LANGUAGE
    weekly_recap: bool = True
    #: The Sunday of the week whose recap they have already been shown.
    recap_week: str = ""
    intro_done: bool = False

    def as_dict(self) -> dict:
        return {
            "interests": list(self.interests),
            "language": self.language,
            "weekly_recap": self.weekly_recap,
            "recap_week": self.recap_week,
            "intro_done": self.intro_done,
        }


class PreferenceStore:
    """One row per listener. Small, boring, and read on nearly every screen."""

    def __init__(self, path: str | None = None) -> None:
        self.path = data_path("PREFS_DB", "preferences.db", path)
        self._local = threading.local()
        with self._conn() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS preferences (
                       user_id      TEXT PRIMARY KEY,
                       interests    TEXT NOT NULL DEFAULT '',
                       language     TEXT NOT NULL DEFAULT 'en',
                       weekly_recap INTEGER NOT NULL DEFAULT 1,
                       recap_week   TEXT NOT NULL DEFAULT '',
                       intro_done   INTEGER NOT NULL DEFAULT 0,
                       updated      REAL NOT NULL
                   )"""
            )

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    def get(self, user_id: str) -> Preferences:
        """This listener's settings, or the defaults. Never raises."""
        if not user_id:
            return Preferences("")
        try:
            row = self._conn().execute(
                "SELECT interests, language, weekly_recap, recap_week, intro_done"
                " FROM preferences WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        except Exception:
            # A settings read must never be what stops an episode playing.
            log.exception("could not read preferences")
            return Preferences(user_id)
        if row is None:
            return Preferences(user_id)
        return Preferences(
            user_id=user_id,
            interests=tuple(t for t in row[0].split(",") if t),
            language=row[1] or DEFAULT_LANGUAGE,
            weekly_recap=bool(row[2]),
            recap_week=row[3] or "",
            intro_done=bool(row[4]),
        )

    def save(
        self,
        user_id: str,
        interests: Optional[Iterable[str]] = None,
        language: Optional[str] = None,
        weekly_recap: Optional[bool] = None,
        intro_done: Optional[bool] = None,
        recap_week: Optional[str] = None,
        at: float = 0.0,
    ) -> Preferences:
        """Write only the fields given. Raises PreferenceError on a bad value.

        Partial by design: the intro saves interests on one page and the
        language on the next, and the recap popup writes one flag from a screen
        that knows nothing about either.
        """
        if not user_id:
            raise PreferenceError("There is no listener to save this for.")
        current = self.get(user_id)
        merged = Preferences(
            user_id=user_id,
            interests=(clean_interests(interests) if interests is not None
                       else current.interests),
            language=(clean_language(language) if language is not None
                      else current.language),
            weekly_recap=(bool(weekly_recap) if weekly_recap is not None
                          else current.weekly_recap),
            recap_week=(recap_week if recap_week is not None else current.recap_week),
            intro_done=(bool(intro_done) if intro_done is not None
                        else current.intro_done),
        )
        self._conn().execute(
            """INSERT INTO preferences
                   (user_id, interests, language, weekly_recap, recap_week,
                    intro_done, updated)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET
                   interests    = excluded.interests,
                   language     = excluded.language,
                   weekly_recap = excluded.weekly_recap,
                   recap_week   = excluded.recap_week,
                   intro_done   = excluded.intro_done,
                   updated      = excluded.updated""",
            (user_id, ",".join(merged.interests), merged.language,
             int(merged.weekly_recap), merged.recap_week, int(merged.intro_done),
             at or time.time()),
        )
        return merged

    def recap_due(self, user_id: str, now: Optional[float] = None) -> bool:
        """Is this week's recap still owed to this listener?

        True on the first open of a new week and false thereafter, whichever
        day of the week that open happens on.
        """
        prefs = self.get(user_id)
        if not prefs.weekly_recap:
            return False
        return prefs.recap_week != week_start(now)

    def mark_recap_seen(self, user_id: str, now: Optional[float] = None) -> Preferences:
        return self.save(user_id, recap_week=week_start(now))
