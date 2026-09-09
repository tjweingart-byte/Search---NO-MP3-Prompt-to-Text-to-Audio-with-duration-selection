"""The line that draws itself while the episode plays.

The picture is **one continuous stroke**: a single SVG `<path>`, revealed from
its start to its end over the length of the episode. Blank canvas at 0:00, the
line moving by 0:18, the subject recognisable around two thirds through, the
completed drawing at the end - and that completed drawing is the episode
thumbnail, so the same asset serves both and nothing is generated twice.

Three decisions carry this, and each is here rather than somewhere more
convenient for a reason:

**It is a path, not an image.** A stroke revealed by `stroke-dashoffset` is
what makes "drawn, not faded in" possible at all: the browser knows the
path's length, so the fraction of it that is visible can be set from the
audio's own `currentTime`. That also makes it text - which means Claude can
write it, no image model is involved, and it caches beside the script as a
few hundred bytes rather than as a file. FAM does not write audio files; it
does not want to start writing picture files either.

**It is a separate call, off the generation path.** Nothing in this module is
imported by `pipeline.py` and nothing here runs before the first word. The
one-sentence spec - type a question, hear the answer within about a second -
is the constraint the whole product is bent around, and a few hundred
coordinates in front of the first sentence would break it for a picture
nobody is looking at yet. The design has slack by construction: the canvas is
*meant* to be blank when the audio starts, so the drawing is allowed to
arrive seconds later and still be early. This is the same reasoning as
`<<NEXT: ...>>` - do the work beside the episode, not in front of it - taken
one step further, because a path is far longer than a follow-up question.

**Continuity is verified, not requested.** "A single, continuous line. No
cuts. No jumps." is a promise about the drawing, and a model asked politely
will sometimes lift the pen. In SVG a lifted pen has exactly one spelling: a
second moveto. So `validate` counts them, and a path with two is rejected
rather than shown - because a drawing that jumps is off-brand in a way a
listener notices instantly, and §52 says a thing that reports readiness must
check the real property rather than trust that it was asked for.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field

from anthropic_client import build_async_client
from config import settings

log = logging.getLogger(__name__)

#: The coordinate space every path is drawn in. Square, because the canvas in
#: the player is square and the thumbnail is square. 1000 rather than 100 so
#: the model has room to place a curve precisely; the client scales it.
VIEWBOX = "0 0 1000 1000"
VIEWBOX_SIZE = 1000.0

#: How far outside the box a coordinate may stray before the path is refused.
#: Not zero: a curve's control point legitimately sits outside the visible
#: area, and clipping a few pixels of overshoot is better than rejecting an
#: otherwise good drawing. Well beyond this and the model has lost the frame.
COORD_MARGIN = VIEWBOX_SIZE * 0.25

#: Commands that continue the stroke from wherever it already is. Every SVG
#: path command except moveto does; that is what makes moveto the only way to
#: break the line, and the only thing continuity has to count.
_CONTINUING = "LlHhVvCcSsQqTtAaZz"
_ALL_COMMANDS = "Mm" + _CONTINUING
_COMMAND_RE = re.compile(rf"[{_ALL_COMMANDS}]")
_NUMBER_RE = re.compile(r"-?\d*\.?\d+(?:[eE][-+]?\d+)?")
#: A path with fewer commands than this is a scribble, not a drawing. Chosen
#: low: some subjects really are a dozen strokes, and refusing those would be
#: worse than showing them.
MIN_COMMANDS = 8
#: Guard on what is stored and sent. Comfortably above a detailed line drawing
#: and far below anything that would bloat a cache row.
MAX_PATH_CHARS = 20000


class IllustrationError(RuntimeError):
    """A path that cannot be drawn, phrased so the log says what to fix."""


@dataclass(frozen=True)
class Illustration:
    """One episode's drawing, and what a client needs to animate it."""

    #: The `d` attribute. One moveto, then an unbroken stroke.
    path: str = ""
    #: The coordinate space `path` is drawn in.
    view_box: str = VIEWBOX
    #: What the drawing is of, in a few words. Alt text, and a way to see at a
    #: glance whether the model drew the right subject without rendering it.
    subject: str = ""
    #: Where this came from: "model" | "cache" | "" (nothing available).
    source: str = ""
    #: Set when there is no drawing, saying why in a sentence a person can act
    #: on. Never silently empty - a blank canvas that stays blank has to
    #: explain itself, like every other fallback in this app.
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "view_box": self.view_box,
            "subject": self.subject,
            "source": self.source,
            "detail": self.detail,
            "commands": count_commands(self.path) if self.path else 0,
        }


def count_commands(path: str) -> int:
    return len(_COMMAND_RE.findall(path or ""))


def validate(path: str) -> str:
    """Return a cleaned path, or raise `IllustrationError` saying what is wrong.

    The checks are ordered by how badly each breaks the effect, so the message
    names the real problem rather than the first symptom of it.
    """
    cleaned = re.sub(r"\s+", " ", (path or "").strip())
    if not cleaned:
        raise IllustrationError("the model returned no path")
    if len(cleaned) > MAX_PATH_CHARS:
        raise IllustrationError(
            f"path is {len(cleaned)} characters, over the {MAX_PATH_CHARS} limit")

    stray = set(re.findall(rf"[A-Za-z]", cleaned)) - set(_ALL_COMMANDS) - set("eE")
    if stray:
        raise IllustrationError(
            f"path contains commands that are not SVG path commands: "
            f"{''.join(sorted(stray))}")

    if not cleaned[0] in "Mm":
        raise IllustrationError("path does not begin with a moveto")

    # The continuity check, and the reason this function exists. Every command
    # other than moveto continues from the current point, so the count of
    # movetos is exactly the number of times the pen was lifted.
    lifts = len(re.findall(r"[Mm]", cleaned))
    if lifts != 1:
        raise IllustrationError(
            f"the line is cut into {lifts} pieces ({lifts - 1} pen "
            f"lift{'s' if lifts > 2 else ''}); it must be one continuous stroke")

    # `Z` closes back to the start, which is still one stroke - but only if
    # nothing follows it, since continuing after a close needs a new moveto
    # and would have been caught above anyway.
    close = re.search(r"[Zz]", cleaned)
    if close and close.end() != len(cleaned):
        raise IllustrationError("the path continues after closing")

    commands = count_commands(cleaned)
    if commands < MIN_COMMANDS:
        raise IllustrationError(
            f"only {commands} path commands; too little to be a drawing")

    for raw in _NUMBER_RE.findall(cleaned):
        try:
            value = float(raw)
        except ValueError as exc:
            raise IllustrationError(f"unparseable coordinate {raw!r}") from exc
        if not math.isfinite(value):
            raise IllustrationError(f"non-finite coordinate {raw!r}")
        if not -COORD_MARGIN <= value <= VIEWBOX_SIZE + COORD_MARGIN:
            raise IllustrationError(
                f"coordinate {value:g} is outside the {VIEWBOX} frame")
    return cleaned


_SUBJECT_RE = re.compile(r"<<\s*SUBJECT\s*:\s*([^<>]{1,80}?)\s*>>", re.I)
_PATH_RE = re.compile(r"<<\s*PATH\s*:\s*([^<>]+?)\s*>>", re.I | re.S)


def parse(text: str) -> tuple[str, str]:
    """Pull the subject and the path out of a model reply.

    Markers rather than JSON: the path is full of commas, decimals and
    minus signs, and every one of them is a chance for a JSON parser to be
    handed something that needs escaping. A marker pair cannot be broken by
    the contents of what it wraps.
    """
    subject = _SUBJECT_RE.search(text or "")
    path = _PATH_RE.search(text or "")
    if not path:
        raise IllustrationError("the reply contained no <<PATH: ...>> line")
    return (
        re.sub(r"\s+", " ", subject.group(1)).strip(" .\"'") if subject else "",
        path.group(1),
    )


def build_prompt(query: str, minutes: int, script_hint: str = "") -> str:
    """What to draw, and the one rule that cannot be broken."""
    hint = ""
    if script_hint:
        hint = (
            "\nThe episode itself opens like this, so draw what it is actually "
            f"about rather than what the question sounds like:\n\"{script_hint}\"\n"
        )
    return f"""Draw a single-line illustration for a {minutes}-minute audio episode
answering this question:

"{query}"
{hint}
This is line art in the style of a continuous-line portrait: one unbroken
stroke that wanders the frame and resolves into a recognisable subject. It is
revealed gradually, from the start of the path to its end, while the episode
plays - so the order of the stroke is part of the effect. Begin somewhere
that reads as exploratory and let the recognisable form arrive later.

Rules, in the order they matter:

1. ONE continuous stroke. The path must contain exactly one moveto - the `M`
   at the very beginning - and nothing after that may lift the pen. This is
   the whole idea; a path with a second `M` is rejected outright.
2. Draw a concrete subject, not a diagram, not a chart, not lettering, and
   not an abstract squiggle. A person, an object, a place, an animal - the
   one thing the episode is most about.
3. Use the coordinate space {VIEWBOX}. Keep the drawing inside it, roughly
   centred, with a little breathing room at the edges.
4. Prefer curves (`C`, `S`, `Q`, `T`) over straight lines. A continuous-line
   drawing is made of flowing curves; polylines look like a graph.
5. Between {MIN_COMMANDS} and 120 commands. Enough to be recognisable, few
   enough to stay elegant.
6. No `fill` - this is a stroke. Do not close the path unless the subject
   genuinely ends where it started.

Reply with exactly two lines and nothing else:

<<SUBJECT: three to six words naming what you drew>>
<<PATH: M ... the complete d attribute ...>>"""


class IllustrationGenerator:
    """Asks Claude for one continuous line, and refuses anything that is not.

    Deliberately not a streaming generator. There is nobody waiting on the
    first character of a path - the canvas is blank on purpose until the whole
    stroke has arrived, because a partial path cannot be measured and a stroke
    that is being revealed by length has to know its length before it starts.
    """

    def __init__(self, api_key: str | None = None):
        key = api_key if api_key is not None else settings.anthropic_api_key
        self.client = build_async_client(key)

    async def draw(self, query: str, minutes: int, script_hint: str = "") -> Illustration:
        kwargs = {
            "model": settings.illustration_model,
            "max_tokens": settings.illustration_max_tokens,
            "messages": [{
                "role": "user",
                "content": build_prompt(query, minutes, script_hint),
            }],
        }
        reply = await self.client.messages.create(**kwargs)
        text = "".join(
            block.text for block in reply.content if getattr(block, "type", "") == "text"
        )
        subject, raw = parse(text)
        path = validate(raw)
        log.info("illustration: %d commands, subject %r", count_commands(path), subject)
        return Illustration(path=path, subject=subject, source="model")
