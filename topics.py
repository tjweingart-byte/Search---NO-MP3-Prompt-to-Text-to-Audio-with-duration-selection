"""myFAM: a shared topic bank, plus per-user ranking over it.

The cost argument decides the shape of this module. Generating an episode
costs a model call; ranking a list costs nothing. So **every user sees the
same bank of topics and a different ordering of it**. Two people who tap the
same tile share one script through `cache.py`, and the second tap is free and
instant. A per-user *bank* would mean a per-user script for every tile, which
is the same product at many times the price.

Four sections, and the point is that each runs on a *different* signal - four
shuffles of one score would be one section wearing four hats:

    trending          what everyone is playing now      (global, not personal)
    might_like        adjacent to your taste             (exploration)
    followers         what co-listeners played           (social proxy - see below)
    from_history      closest to what you played         (exploitation)

Everything is derived from an append-only event log, so there is no profile to
keep in sync - a taste profile is a query, not a stored object.

**"What your followers are listening to" is a label over data this app does
not have.** There are no accounts and no follow graph. What it actually ranks
is co-listener overlap: people who played what you played also played this.
That is a real signal and a standard one, but it is not your followers, and
the honest thing is to say so here rather than let the heading imply a social
network that does not exist.
"""
from __future__ import annotations

import logging
import math
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

log = logging.getLogger(__name__)

#: How long an event keeps influencing ranking, in seconds. A taste that never
#: fades makes the feed a museum of what someone cared about last month.
HALF_LIFE = 14 * 86400
#: What "now" means for trending.
TRENDING_WINDOW = 3 * 86400
#: Never show the same tile in two sections; the feed should look wider than
#: the bank actually is.
SECTION_SIZE = 6


@dataclass(frozen=True)
class Topic:
    """One tile. `query` is what gets generated; `title` is only a label."""

    id: str
    title: str
    subtitle: str
    query: str
    tags: tuple[str, ...]
    icon: str

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "subtitle": self.subtitle,
            "query": self.query,
            "tags": list(self.tags),
            "icon": self.icon,
        }


# Keywords that map a free-text search onto the same facets the bank uses.
# Deliberately dumb: a wrong tag costs one mediocre recommendation, and a
# model call to classify every search would cost more than the episode it is
# recommending. See PROBLEMS for why this is the right trade at this size.
TAG_WORDS: dict[str, tuple[str, ...]] = {
    "sports": ("nfl", "nba", "football", "basketball", "golf", "soccer", "tennis",
               "olympics", "coach", "playoff", "draft", "league", "match"),
    "business": ("startup", "founder", "company", "ceo", "ipo", "merger", "layoff",
                 "strategy", "brand", "hiring", "venture"),
    "money": ("fed", "inflation", "rates", "market", "stocks", "economy", "tariff",
              "recession", "housing", "oil", "currency", "bond"),
    "tech": ("ai", "software", "chip", "robot", "app", "model", "data", "code",
             "computing", "algorithm", "internet"),
    "science": ("physics", "space", "rocket", "climate", "biology", "brain",
                "research", "study", "energy", "quantum", "genome"),
    "health": ("sleep", "diet", "exercise", "habit", "stress", "anxiety", "mindset",
               "motivation", "longevity", "cortisol", "mental", "therapy"),
    "culture": ("film", "movie", "music", "album", "book", "art", "hollywood",
                "song", "show", "artist", "fashion"),
    "world": ("election", "war", "treaty", "border", "sanctions", "summit",
              "government", "protest", "strait", "diplomacy", "policy"),
}

_WORD = re.compile(r"[a-z0-9]+")


def tags_for_text(text: str) -> tuple[str, ...]:
    """Best-effort facets for a free-text search, so history can be ranked."""
    words = set(_WORD.findall(text.lower()))
    found = tuple(tag for tag, keys in TAG_WORDS.items() if words & set(keys))
    return found


TOPIC_BANK: tuple[Topic, ...] = (
    Topic("nil-arms-race", "The New College Football Arms Race",
          "NIL money, facilities, and the new power brokers.",
          "how NIL money changed college football recruiting", ("sports", "money"), "sports"),
    Topic("operator-ceos", "Why Founders Are Taking Back Control",
          "Leadership, product, and the rise of operator CEOs.",
          "why boards are keeping founders as CEO", ("business",), "business"),
    Topic("ai-agents", "Why Everyone Is Talking About AI Agents",
          "What they are, how they work, why now.",
          "what AI agents are and why they matter now", ("tech",), "tech"),
    Topic("hollywood-comebacks", "Inside the Best Hollywood Comebacks",
          "The stories, the risks, the second acts.",
          "how Hollywood comeback stories actually happen", ("culture",), "camera"),
    Topic("golf-evolution", "The Quiet Evolution of Golf",
          "New players. New formats. Same obsession.",
          "how professional golf formats are changing", ("sports",), "golf"),
    Topic("habits-research", "The Habits That Actually Change Your Life",
          "What the research says, and what people ignore.",
          "what habit research actually shows about lasting change", ("health",), "leaf"),
    Topic("fed-next-move", "The Fed's Next Move, Explained",
          "Rates, inflation data, and what markets expect.",
          "what the Federal Reserve is likely to do about interest rates",
          ("money",), "business"),
    Topic("space-race", "Inside the New Space Race",
          "Reusable rockets, private missions, who's winning.",
          "how reusable rockets changed the economics of spaceflight",
          ("science", "business"), "rocket"),
    Topic("song-breaks-internet", "How One Song Breaks the Internet",
          "Playlists, algorithms, and the new path to a hit.",
          "how a song becomes a hit through playlists and short video",
          ("culture", "tech"), "music"),
    Topic("restaurant-scene", "The Restaurants Everyone's Talking About",
          "Openings, closings, and where the lines form.",
          "why some restaurants become impossible to book", ("culture",), "food"),
    Topic("the-trade", "The Trade That Changed Everything",
          "Front offices, cap space, and deals nobody saw.",
          "how a single trade reshapes a sports franchise", ("sports",), "sports"),
    Topic("sleep-science", "What We Actually Know About Sleep",
          "The research, the myths, what isn't settled.",
          "what sleep research actually establishes", ("health", "science"), "leaf"),
    Topic("chip-supply", "Who Actually Makes the World's Chips",
          "Fabs, bottlenecks, and why it is so concentrated.",
          "why semiconductor manufacturing is concentrated in so few places",
          ("tech", "world"), "tech"),
    Topic("hormuz", "The Two-Mile Lane That Moves the Oil Price",
          "Chokepoints, insurance, and why geography decides.",
          "why the Strait of Hormuz moves the oil price", ("world", "money"), "business"),
    Topic("morning-mindset", "What to Do With the First Ten Minutes",
          "Why waking up feels the way it does.",
          "a good mindset for when I wake up in the morning", ("health",), "leaf"),
    Topic("founder-motivation", "Where Motivation Actually Comes From",
          "Progress, evidence, and the founder's problem.",
          "finding the motivation for my startup", ("health", "business"), "leaf"),
    Topic("housing-market", "Why Houses Cost What They Cost",
          "Supply, rates, and the arguments that repeat.",
          "what actually drives house prices", ("money",), "business"),
    Topic("longevity-claims", "Sorting the Longevity Claims",
          "What holds up, what is marketing.",
          "which longevity interventions have real evidence", ("health", "science"), "leaf"),
    Topic("streaming-economics", "Why Streaming Keeps Getting Worse",
          "Licensing, churn, and the maths underneath.",
          "why streaming services keep raising prices and losing shows",
          ("culture", "business"), "camera"),
    Topic("election-mechanics", "How a Close Election Is Actually Called",
          "Counting, models, and why it takes days.",
          "how news organisations decide to call an election", ("world",), "business"),
    Topic("energy-grid", "What the Grid Does When the Wind Drops",
          "Storage, baseload, and the balancing act.",
          "how electricity grids handle intermittent renewable power",
          ("science", "money"), "rocket"),
    Topic("attention-economy", "The Fight for Fifteen Seconds",
          "How short video rewired everything downstream.",
          "how short-form video changed the media business", ("tech", "culture"), "music"),
    Topic("transfer-window", "How a Transfer Window Actually Works",
          "Agents, deadlines, and the money underneath.",
          "how football transfer deals actually get done", ("sports", "money"), "sports"),
    Topic("stadium-money", "Who Really Pays for a Stadium",
          "Public money, private returns, and the argument.",
          "who actually pays for new sports stadiums", ("sports", "money"), "business"),
    Topic("anxiety-loop", "Why Worry Feels Productive",
          "The loop, and what actually interrupts it.",
          "why worrying feels useful when it is not", ("health",), "leaf"),
    Topic("pricing-psychology", "Why Everything Ends in Ninety-Nine",
          "What the pricing research does and does not show.",
          "what the evidence says about psychological pricing", ("business", "money"), "business"),
    Topic("food-supply", "How Food Gets to a City",
          "Logistics, margins, and the fragile bits.",
          "how a city's food supply chain actually works", ("world", "business"), "food"),
    Topic("training-load", "How Athletes Are Actually Trained Now",
          "Load, recovery, and the data behind it.",
          "how modern athletic training load is managed", ("sports", "health"), "sports"),
)

BANK_BY_ID = {t.id: t for t in TOPIC_BANK}

#: Sections are FILLED in this order and DISPLAYED in SECTIONS order. The two
#: personal sections have the fewest eligible topics, so they choose first;
#: trending can fall back to the whole bank and therefore chooses last.
FILL_ORDER = ("from_history", "followers", "might_like", "trending")

#: Display order: personal first, global last. Someone opening myFAM is more
#: likely to want what was chosen for them than what is popular, and the page
#: should not make them scroll past the crowd to reach it. (Fill order is
#: separate - see FILL_ORDER - because the constrained sections must still
#: choose their topics first.)
SECTIONS = (
    ("from_history", "Made for you"),
    ("might_like", "A little sideways from that"),
    ("followers", "Your circle is on this"),
    ("trending", "What FAM can't stop playing"),
)

#: How much each kind of interaction says about taste. Finishing an episode is
#: the strongest signal there is; a skip is real evidence in the other
#: direction and must not be treated as a weak play.
EVENT_WEIGHT = {"search": 1.0, "play": 1.0, "complete": 2.5, "skip": -1.5}


def _decay(age_seconds: float) -> float:
    return 0.5 ** (age_seconds / HALF_LIFE)


@dataclass
class Event:
    user_id: str
    kind: str
    topic_id: str = ""
    text: str = ""
    tags: tuple[str, ...] = ()
    at: float = field(default_factory=time.time)
    #: The thread this episode left open, carried on the event that finished
    #: it. Recorded here rather than joined back to the cache at read time,
    #: because the cache key depends on settings that may have moved on and a
    #: thread the listener was actually offered should not disappear.
    thread: str = ""


class EventStore:
    """Append-only interaction log. SQLite for the same reasons as the cache:
    no new dependency, survives restarts, shared by every worker."""

    def __init__(self, path: str = "myfam.db") -> None:
        self.path = path
        self._local = threading.local()
        with self._conn() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS events (
                       id       INTEGER PRIMARY KEY AUTOINCREMENT,
                       user_id  TEXT NOT NULL,
                       kind     TEXT NOT NULL,
                       topic_id TEXT NOT NULL DEFAULT '',
                       text     TEXT NOT NULL DEFAULT '',
                       tags     TEXT NOT NULL DEFAULT '',
                       at       REAL NOT NULL
                   )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS events_user ON events(user_id, at)")
            conn.execute("CREATE INDEX IF NOT EXISTS events_topic ON events(topic_id, at)")
            try:
                conn.execute("ALTER TABLE events ADD COLUMN thread TEXT NOT NULL DEFAULT ''")
            except sqlite3.OperationalError:
                pass  # already there

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    def record(self, event: Event) -> None:
        if event.kind not in EVENT_WEIGHT:
            log.warning("ignoring unknown event kind %r", event.kind)
            return
        try:
            self._conn().execute(
                "INSERT INTO events (user_id, kind, topic_id, text, tags, at, thread)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (event.user_id[:64], event.kind, event.topic_id[:64], event.text[:300],
                 ",".join(event.tags), event.at, event.thread[:200]),
            )
        except Exception:
            # A feed is a nicety. Losing an event must never break playback.
            log.exception("could not record interaction; continuing")

    def for_user(self, user_id: str, limit: int = 400) -> list[Event]:
        try:
            rows = self._conn().execute(
                "SELECT kind, topic_id, text, tags, at, thread FROM events"
                " WHERE user_id = ? ORDER BY at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        except Exception:
            log.exception("could not read interactions")
            return []
        return [
            Event(user_id, r[0], r[1], r[2], tuple(t for t in r[3].split(",") if t),
                  r[4], r[5] or "")
            for r in rows
        ]

    def open_threads(self, user_id: str, limit: int = 8) -> list[dict]:
        """Threads from episodes this listener finished, newest first.

        The point of the widening ending was to leave one specific thing
        unresolved. This is where those land: an episode they have already
        heard, and the question it opened, one tap from being answered.

        A thread they have since asked about is dropped - it is no longer
        open - which is why this reads the log rather than a stored list.
        """
        events = self.for_user(user_id, limit=200)
        asked = " ".join(e.text.lower() for e in events if e.kind == "search")
        seen: set[str] = set()
        out: list[dict] = []
        for event in events:
            thread = (event.thread or "").strip()
            if not thread or event.kind != "complete":
                continue
            key = thread.lower()
            if key in seen or key in asked:
                continue
            seen.add(key)
            out.append({
                "thread": thread,
                "title": thread[:1].upper() + thread[1:],
                "from_title": (BANK_BY_ID[event.topic_id].title
                               if event.topic_id in BANK_BY_ID else event.text),
                "at": event.at,
            })
            if len(out) >= limit:
                break
        return out

    def plays_since(self, since: float) -> list[tuple[str, str]]:
        """(user_id, topic_id) for every bank topic played in the window."""
        try:
            rows = self._conn().execute(
                "SELECT user_id, topic_id FROM events"
                " WHERE at >= ? AND topic_id != '' AND kind IN ('play', 'complete')",
                (since,),
            ).fetchall()
        except Exception:
            log.exception("could not read plays")
            return []
        return [(r[0], r[1]) for r in rows]

    def users_who_played(self, topic_ids: Iterable[str]) -> dict[str, set[str]]:
        """topic_id -> the users who played it. The co-listener join."""
        ids = list(topic_ids)
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        try:
            rows = self._conn().execute(
                f"SELECT topic_id, user_id FROM events WHERE topic_id IN ({marks})"
                " AND kind IN ('play', 'complete')",
                ids,
            ).fetchall()
        except Exception:
            log.exception("could not read co-listeners")
            return {}
        out: dict[str, set[str]] = {}
        for topic_id, user_id in rows:
            out.setdefault(topic_id, set()).add(user_id)
        return out


def taste(events: Iterable[Event], now: Optional[float] = None) -> dict[str, float]:
    """Tag affinity for one listener: recency-weighted, signed, normalised.

    Computed on read rather than stored. A stored profile is a cache that can
    disagree with the log it came from; this cannot.
    """
    now = time.time() if now is None else now
    scores: dict[str, float] = {}
    for event in events:
        weight = EVENT_WEIGHT.get(event.kind, 0.0) * _decay(max(0.0, now - event.at))
        tags = event.tags or (tags_for_text(event.text) if event.text else ())
        for tag in tags:
            scores[tag] = scores.get(tag, 0.0) + weight
    peak = max((abs(v) for v in scores.values()), default=0.0)
    return {k: v / peak for k, v in scores.items()} if peak else {}


def _affinity(topic: Topic, profile: dict[str, float]) -> float:
    if not topic.tags:
        return 0.0
    return sum(profile.get(tag, 0.0) for tag in topic.tags) / math.sqrt(len(topic.tags))


def _played_ids(events: Iterable[Event]) -> set[str]:
    return {e.topic_id for e in events if e.topic_id and e.kind in ("play", "complete")}


def rank_trending(
    store: EventStore, now: Optional[float] = None, exclude: Optional[set[str]] = None
) -> list[Topic]:
    """Global play counts. Deliberately identical for everyone, which is what
    makes it the cheapest section to serve: one script, every listener."""
    now = time.time() if now is None else now
    exclude = exclude or set()
    counts: dict[str, int] = {}
    for _user, topic_id in store.plays_since(now - TRENDING_WINDOW):
        counts[topic_id] = counts.get(topic_id, 0) + 1
    ranked = sorted(
        (t for t in TOPIC_BANK if t.id in counts and t.id not in exclude),
        key=lambda t: (-counts[t.id], t.id),
    )
    # A cold bank has no plays yet. A stable slice beats an empty section, and
    # beats a random one - random means the tile a listener saw this morning is
    # gone this afternoon, and it defeats the shared script cache.
    filler = [t for t in TOPIC_BANK if t.id not in counts and t.id not in exclude]
    return (ranked + filler)[:SECTION_SIZE]


def rank_from_history(profile: dict[str, float], exclude: set[str]) -> list[Topic]:
    """Closest match to what they already play. Exploitation."""
    scored = [
        (_affinity(t, profile), t) for t in TOPIC_BANK if t.id not in exclude
    ]
    scored = [(s, t) for s, t in scored if s > 0]
    scored.sort(key=lambda pair: (-pair[0], pair[1].id))
    return [t for _s, t in scored[:SECTION_SIZE]]


def rank_might_like(profile: dict[str, float], exclude: set[str]) -> list[Topic]:
    """Adjacent, not identical. Exploration.

    Their strongest tag is deliberately suppressed. Ranking purely on affinity
    gives four sections of the same thing and a listener who only ever hears
    back what they already told you - the filter bubble, arrived at by
    accident. This section is the one that widens the bank.

    Built in tiers that top each other up rather than replace each other. A
    listener whose entire history is one tag has an empty profile the moment
    it is muted, and they are exactly who this section exists for; the earlier
    version returned nothing for them, which the tests caught.
    """
    if not profile:
        return [t for t in TOPIC_BANK if t.id not in exclude][:SECTION_SIZE]

    top_tag = max(profile, key=lambda k: profile[k])
    muted = {k: v for k, v in profile.items() if k != top_tag}
    picks: list[Topic] = []
    taken = set(exclude)

    def add(candidates: list[tuple[float, Topic]]) -> None:
        candidates.sort(key=lambda pair: (-pair[0], pair[1].id))
        for _score, topic in candidates:
            if len(picks) >= SECTION_SIZE:
                return
            if topic.id not in taken:
                picks.append(topic)
                taken.add(topic.id)

    # 1. Their other interests, with a nudge toward anything that also brings
    #    a tag they have never touched.
    tier = []
    for topic in TOPIC_BANK:
        if topic.id in taken:
            continue
        score = _affinity(topic, muted)
        if score <= 0:
            continue
        if any(tag not in profile for tag in topic.tags):
            score *= 1.4
        tier.append((score, topic))
    add(tier)

    # 2. Bridges out of the tag they already have: keep the familiar tag, but
    #    only where it is paired with something new, so it leads somewhere.
    if len(picks) < SECTION_SIZE:
        add([
            (float(sum(1 for tag in t.tags if tag not in profile)), t)
            for t in TOPIC_BANK
            if t.id not in taken and top_tag in t.tags
            and any(tag not in profile for tag in t.tags)
        ])

    # 3. Anything genuinely unseen. An empty shelf helps nobody, and a narrow
    #    listener is the one who most needs a way out of the bubble.
    if len(picks) < SECTION_SIZE:
        add([
            (float(sum(1 for tag in t.tags if tag not in profile)), t)
            for t in TOPIC_BANK if t.id not in taken
        ])

    return picks


def rank_followers(
    store: EventStore, user_id: str, mine: set[str], exclude: set[str]
) -> list[Topic]:
    """Co-listener overlap: people who played what you played also played this.

    Named "followers" in the interface. It is not a follow graph - the app has
    no accounts and no follows - and this is the closest real signal to it.
    """
    if not mine:
        return []
    by_topic = store.users_who_played(BANK_BY_ID.keys())
    neighbours: set[str] = set()
    for topic_id in mine:
        neighbours |= by_topic.get(topic_id, set())
    neighbours.discard(user_id)
    if not neighbours:
        return []
    scored = []
    for topic in TOPIC_BANK:
        if topic.id in exclude:
            continue
        overlap = len(by_topic.get(topic.id, set()) & neighbours)
        if overlap:
            scored.append((overlap, topic))
    scored.sort(key=lambda pair: (-pair[0], pair[1].id))
    return [t for _s, t in scored[:SECTION_SIZE]]


def build_feed(store: EventStore, user_id: str, now: Optional[float] = None) -> dict:
    """The whole myFAM page for one listener.

    Sections are filled in order and never repeat a topic, so the page looks
    as wide as possible from a deliberately small bank.
    """
    events = store.for_user(user_id) if user_id else []
    profile = taste(events, now)
    mine = _played_ids(events)
    used: set[str] = set()
    picked: dict[str, list[Topic]] = {}

    # Filled most-constrained first, displayed in the order the product asks
    # for. Filling in display order starves the two personal sections: the
    # generic ones can fall back to the whole bank, so they claim the very
    # topics the personal ones needed and those arrive empty - which is
    # exactly backwards, since the personal sections are the point.
    for key in FILL_ORDER:
        # Nothing they have already played, in any section. The feed's job is
        # to hand them the next episode; trending stays globally *ranked*, it
        # just stops offering back the one they finished this morning.
        seen = used | mine
        if key == "from_history":
            picks = rank_from_history(profile, seen)
        elif key == "followers":
            picks = rank_followers(store, user_id, mine, seen)
        elif key == "might_like":
            picks = rank_might_like(profile, seen)
        else:
            picks = rank_trending(store, now, seen)
        picked[key] = picks
        used |= {t.id for t in picks}

    # An empty section is honest, not broken: a new listener genuinely has no
    # history and no co-listeners. The interface says so rather than padding
    # it with picks that pretend to be personal.
    sections = [
        {
            "key": key,
            "title": title,
            "topics": [t.as_dict() for t in picked[key]],
            "empty_reason": _empty_reason(key) if not picked[key] else "",
        }
        for key, title in SECTIONS
    ]
    return {"sections": sections, "personalised": bool(profile)}


def summary(store: EventStore, user_id: str, now: Optional[float] = None) -> dict:
    """What this app actually knows about a listener.

    Deliberately only what the event log really holds. A profile page is the
    easiest place in an app to invent numbers - followers, streaks, hours
    saved - and every invented one is a promise the product has to keep later.
    """
    events = store.for_user(user_id, limit=1000)
    profile = taste(events, now)
    top = sorted(profile.items(), key=lambda kv: -kv[1])
    return {
        "listener": user_id,
        "played": sum(1 for e in events if e.kind == "play"),
        "finished": sum(1 for e in events if e.kind == "complete"),
        "searched": sum(1 for e in events if e.kind == "search"),
        "open_threads": len(store.open_threads(user_id)),
        # Only tags they are actually positive about; a skip pushes a tag
        # negative and it has no business on a list of what someone likes.
        "subjects": [tag for tag, weight in top if weight > 0][:5],
        "since": min((e.at for e in events), default=0.0),
    }


def _empty_reason(key: str) -> str:
    return {
        "trending": "Nothing has been played yet today.",
        "might_like": "Listen to a few episodes and this fills in.",
        "followers": "Nobody you overlap with has listened yet.",
        "from_history": "Your first episode starts this one off.",
    }.get(key, "")
