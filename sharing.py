"""Sharing an episode outside FAM: a link, the words to send with it, and a card.

Three things a share needs, and this module makes all three:

1. **A link** that opens the episode for somebody who may not have the app.
2. **The words**, written per destination, because a LinkedIn post and a text
   message to one friend are not the same message.
3. **A card**, for the destinations that cannot take a link at all.

## What this deliberately does not do: post anything

FAM holds no Facebook token, no LinkedIn token, no Snapchat token, and asks for
none. It does not post on anybody's behalf, and it cannot.

That is not a gap to be filled later - it is the correct shape. Every platform
here is reached from the phone: iOS hands the app a share sheet, and the story
formats have their own SDK hand-off where the app passes an image and a link
and the *platform's* app does the posting, with the person looking at it. So
the server's job is to produce the payload, and a share is something the
listener completes.

Which means: no OAuth to maintain, no tokens to leak, no scope reviews with
four companies, and nothing that can post while somebody is asleep.

## Stories cannot take a link, and that is why there is a card

Instagram and Snapchat stories are **images**. You cannot post a sentence with
a URL in it; you attach a sticker to a picture. So a story share needs a
picture, and if the app does not make one the listener shares a screenshot of
whatever was on screen - which is a player UI, not an invitation.

`story_card` renders one as SVG: text on FAM's own colours, at 1080x1920. SVG
rather than a rasteriser because it needs no dependency, is a few kilobytes,
and both clients can turn it into a bitmap - a browser through canvas, iOS
through its own renderer. The card is generated per episode and never stored:
it is a function of the title and the question, both of which we already have.

## The link, and what it is allowed to promise

`PUBLIC_BASE_URL` is where the app is reachable from the internet. Unset, a
share still works - the link comes back relative and the interface can still
copy it - but it names no host, and anything showing it must not pretend
otherwise. A share link posted to LinkedIn that resolves to `localhost` is the
kind of quiet failure this project keeps a rule about.

A share row is the unit, not the query string, so an episode can be shared once
and the link stays the same however many places it goes - and so opens can be
counted per share rather than per platform, which is the number that tells you
whether sharing works at all.
"""
from __future__ import annotations

import html
import logging
import re
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Optional

from paths import data_path

log = logging.getLogger(__name__)

MAX_TITLE = 200
MAX_QUERY = 500


class ShareError(ValueError):
    """Something the listener can fix, phrased so it can be shown to them."""


@dataclass(frozen=True)
class Target:
    """One destination, and what it can carry.

    `kind` is the thing that actually differs:

    * `link` - a URL with text around it. Facebook, LinkedIn, X.
    * `message` - a private message to somebody. SMS, email, WhatsApp.
    * `story` - an image with a link attached. Instagram, Snapchat.
    * `copy` - the link on its own, for everywhere not listed.
    """

    key: str
    label: str
    kind: str
    #: What the platform will actually show before truncating. 0 means no
    #: practical limit. Enforced here rather than hoped for, because a caption
    #: cut off mid-question reads as a broken app rather than a long post.
    max_chars: int
    #: Whether a picture is required for this to be postable at all.
    needs_image: bool
    #: `{title}`, `{question}`, `{minutes}`, `{url}` are substituted.
    template: str
    #: Only used where the platform has a subject line of its own.
    subject: str = ""


#: The house voice for a share, per destination. These are **defaults the
#: listener edits**, not copy that gets posted unseen - which is why they are
#: written to be finished by somebody rather than to be complete.
#:
#: The shape is the same everywhere and is the argument for pressing play:
#: what it is about, that it is short, and where to hear it. What is
#: deliberately absent is any claim about the episode being good, because FAM
#: did not write that opinion and the person sharing it has not typed one yet.
TARGETS: tuple[Target, ...] = (
    Target("copy", "Copy link", "copy", 0, False,
           "{title} - a {minutes}-minute FAM episode: {url}"),
    Target("sms", "Message", "message", 0, False,
           "Listen to this - {title}. About {minutes} minutes: {url}"),
    Target("email", "Email", "message", 0, False,
           "I asked FAM \"{question}\" and it made a {minutes}-minute episode "
           "answering it.\n\nHave a listen: {url}",
           subject="{title}"),
    Target("whatsapp", "WhatsApp", "message", 0, False,
           "Listen to this - {title}. About {minutes} minutes: {url}"),
    # 280 including the URL, which platforms shorten to a fixed length; the
    # template is kept well under so an edited version still fits.
    Target("x", "X", "link", 240, False,
           "{question}\n\nFAM made me a {minutes}-minute episode on it. {url}"),
    Target("facebook", "Facebook", "link", 0, False,
           "I asked FAM \"{question}\" - here is the {minutes}-minute answer. {url}"),
    # Long-form by convention, and the one place a share reads as a post rather
    # than a message, so the template leaves an obvious place to add a thought.
    Target("linkedin", "LinkedIn", "link", 2800, False,
           "\"{question}\"\n\nFAM turned that into a {minutes}-minute briefing. "
           "Worth a listen if you have been wondering the same thing.\n\n{url}"),
    Target("instagram_story", "Instagram story", "story", 0, True,
           "{title}"),
    Target("snapchat_story", "Snapchat story", "story", 0, True,
           "{title}"),
)

TARGET_KEYS: tuple[str, ...] = tuple(t.key for t in TARGETS)
_BY_KEY = {t.key: t for t in TARGETS}


def target(key: str) -> Optional[Target]:
    return _BY_KEY.get(key)


def render(target_key: str, *, title: str, question: str, minutes: int,
           url: str) -> dict:
    """The text to hand the platform, trimmed to what it will show.

    Trimming happens on a word boundary and adds an ellipsis, because a caption
    that stops mid-word looks like the app broke rather than like the platform
    has a limit.
    """
    chosen = target(target_key)
    if chosen is None:
        raise ShareError(f"Unknown share destination {target_key!r}.")
    values = {
        "title": (title or "A FAM episode").strip()[:MAX_TITLE],
        "question": (question or "").strip()[:MAX_QUERY],
        "minutes": max(1, int(minutes or 0)),
        "url": url or "",
    }
    text = chosen.template.format(**values)
    if chosen.max_chars and len(text) > chosen.max_chars:
        keep = text[:chosen.max_chars - 1]
        # Never cut the URL off: a share whose link is truncated is worse than
        # a share with a shorter sentence, so the trim is taken out of the text
        # and the link is re-attached.
        if values["url"] and values["url"] not in keep:
            room = chosen.max_chars - len(values["url"]) - 4
            keep = text.split(values["url"])[0][:max(0, room)]
            keep = keep.rsplit(" ", 1)[0] + "… " + values["url"]
        else:
            keep = keep.rsplit(" ", 1)[0] + "…"
        text = keep
    return {
        "target": chosen.key,
        "label": chosen.label,
        "kind": chosen.kind,
        "needs_image": chosen.needs_image,
        "text": text,
        "subject": chosen.subject.format(**values) if chosen.subject else "",
        "url": values["url"],
    }


# --- the story card -------------------------------------------------------

def _wrap(text: str, per_line: int, max_lines: int) -> list[str]:
    """Greedy word wrap. Good enough for a card and it needs no font metrics -
    which would mean a rendering dependency for a picture made of five lines
    of text."""
    words = str(text or "").split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = (current + " " + word).strip()
        if len(candidate) <= per_line:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if len(lines) == max_lines and len(" ".join(words)) > sum(len(l) + 1 for l in lines):
        lines[-1] = lines[-1].rstrip(" ,.") + "…"
    return lines


def story_card(title: str, question: str, minutes: int, handle: str = "") -> str:
    """A 1080x1920 story image, as SVG.

    Portrait and full-bleed because that is the only shape a story is. The
    colours are FAM's own from `static/index.html`, restated here rather than
    imported: a share card that drifts from the app's palette looks like
    somebody else's product, and a stylesheet is not reachable from a server
    that renders this without a browser.

    Everything is escaped - the question is text a listener typed, and this is
    markup.
    """
    lines = _wrap(title or "A FAM episode", per_line=18, max_lines=4)
    ask = _wrap(question or "", per_line=34, max_lines=2)
    minutes = max(1, int(minutes or 0))
    esc = html.escape

    title_svg = "".join(
        f'<tspan x="90" dy="{0 if i == 0 else 104}">{esc(line)}</tspan>'
        for i, line in enumerate(lines))
    ask_svg = "".join(
        f'<tspan x="90" dy="{0 if i == 0 else 46}">{esc(line)}</tspan>'
        for i, line in enumerate(ask))
    who = esc(("@" + handle.lstrip("@")) if handle else "on FAM")

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="1920" viewBox="0 0 1080 1920">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="0.6" y2="1">
      <stop offset="0%" stop-color="#2A2733"/>
      <stop offset="55%" stop-color="#1E1B27"/>
      <stop offset="100%" stop-color="#38334A"/>
    </linearGradient>
    <radialGradient id="glow" cx="0.2" cy="0.12" r="0.7">
      <stop offset="0%" stop-color="#E0B563" stop-opacity="0.20"/>
      <stop offset="100%" stop-color="#E0B563" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect width="1080" height="1920" fill="url(#bg)"/>
  <rect width="1080" height="1920" fill="url(#glow)"/>

  <text x="90" y="250" fill="#E0B563" font-family="'Space Grotesk',Helvetica,Arial,sans-serif"
        font-size="34" font-weight="700" letter-spacing="6">FAM</text>

  <text x="90" y="700" fill="#F4EFE4" font-family="Georgia,'Times New Roman',serif"
        font-size="92" font-weight="600">{title_svg}</text>

  <text x="90" y="1180" fill="#ABA3C4" font-family="'Space Grotesk',Helvetica,Arial,sans-serif"
        font-size="38">{ask_svg}</text>

  <rect x="90" y="1320" width="{110 + len(str(minutes)) * 26}" height="64" rx="32"
        fill="none" stroke="#E0B563" stroke-width="2"/>
  <text x="{120 + len(str(minutes)) * 4}" y="1362" fill="#E0B563"
        font-family="'Space Grotesk',Helvetica,Arial,sans-serif" font-size="30"
        font-weight="600">{minutes} min</text>

  <text x="90" y="1700" fill="#8FAE9A" font-family="'Space Grotesk',Helvetica,Arial,sans-serif"
        font-size="32" font-weight="500">{who}</text>
  <text x="90" y="1760" fill="#8A83A0" font-family="'Space Grotesk',Helvetica,Arial,sans-serif"
        font-size="28">Ask anything. Hear the answer.</text>
</svg>'''


# --- the store ------------------------------------------------------------

class ShareStore:
    def __init__(self, path: str | None = None) -> None:
        self.path = data_path("SHARES_DB", "shares.db", path)
        self._local = threading.local()
        with self._conn() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS shares (
                       id      TEXT PRIMARY KEY,
                       user_id TEXT NOT NULL,
                       query   TEXT NOT NULL,
                       minutes INTEGER NOT NULL DEFAULT 0,
                       title   TEXT NOT NULL DEFAULT '',
                       created REAL NOT NULL,
                       opens   INTEGER NOT NULL DEFAULT 0
                   )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS shares_user"
                         " ON shares(user_id, created)")
            # One share per episode per listener, so sharing the same thing to
            # four platforms produces one link with four destinations rather
            # than four links whose open counts have to be added up.
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS shares_once"
                         " ON shares(user_id, query, minutes)")

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    def create(self, user_id: str, query: str, minutes: int, title: str = "",
               at: float = 0.0) -> dict:
        query = " ".join(str(query or "").split())[:MAX_QUERY]
        if not query:
            raise ShareError("There is no episode to share.")
        minutes = max(0, int(minutes or 0))
        title = " ".join(str(title or "").split())[:MAX_TITLE]
        now = at or time.time()

        existing = self._conn().execute(
            "SELECT id FROM shares WHERE user_id = ? AND query = ? AND minutes = ?",
            (user_id, query, minutes)).fetchone()
        if existing:
            return self.get(existing[0]) or {}

        share_id = secrets.token_urlsafe(9)
        self._conn().execute(
            "INSERT INTO shares (id, user_id, query, minutes, title, created)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (share_id, user_id, query, minutes, title, now))
        return self.get(share_id) or {}

    def get(self, share_id: str) -> Optional[dict]:
        try:
            row = self._conn().execute(
                "SELECT id, user_id, query, minutes, title, created, opens"
                " FROM shares WHERE id = ?", (share_id,)).fetchone()
        except Exception:
            log.exception("could not read a share")
            return None
        if not row:
            return None
        return {"id": row[0], "user_id": row[1], "query": row[2],
                "minutes": row[3], "title": row[4], "created": row[5],
                "opens": row[6]}

    def opened(self, share_id: str) -> None:
        """Somebody followed the link. The only number here, and it is the one
        that says whether sharing does anything at all."""
        try:
            self._conn().execute(
                "UPDATE shares SET opens = opens + 1 WHERE id = ?", (share_id,))
        except Exception:
            log.exception("could not count a share open")

    def forget(self, user_id: str) -> int:
        try:
            cur = self._conn().execute("DELETE FROM shares WHERE user_id = ?",
                                       (user_id,))
            return cur.rowcount or 0
        except Exception:
            log.exception("could not erase shares for %r", user_id)
            return 0
