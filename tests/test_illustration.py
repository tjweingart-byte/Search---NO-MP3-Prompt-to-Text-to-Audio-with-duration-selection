"""The line that draws itself.

The promise on the design is "a single, continuous line. No cuts. No jumps."
Most of what is asserted here is that promise, because it is the one property
a listener notices being broken and the one a model will quietly break.
"""
from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as appmod  # noqa: E402
import cache as cache_mod  # noqa: E402
import illustration as illo  # noqa: E402
from script_generator import plan_episode  # noqa: E402

# A plausible continuous-line figure: one moveto, then curves all the way
# back round. Ten commands, comfortably over MIN_COMMANDS.
GOOD = ("M 500 120 C 620 180 700 300 660 430 S 520 600 500 720 "
        "Q 480 840 560 900 L 600 920 C 700 880 760 760 720 640 "
        "S 600 500 560 380 Q 540 260 500 120")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(appmod, "_read_limit", lambda request: None)
    return TestClient(appmod.app)


# --- continuity: the whole idea -------------------------------------------


def test_a_single_stroke_is_accepted():
    assert illo.validate(GOOD) == GOOD


def test_a_second_moveto_is_refused_because_it_is_a_pen_lift():
    """This is the check the feature exists around.

    In SVG every command except moveto continues from the current point, so a
    second `M` is the only way to break the line - and a broken line is not
    the product. A model asked politely for one stroke will sometimes return
    two, which is why this is verified rather than requested.
    """
    with pytest.raises(illo.IllustrationError) as exc:
        illo.validate(GOOD + " M 100 100 L 200 200")
    assert "continuous" in str(exc.value)
    assert "2 pieces" in str(exc.value)


def test_the_message_counts_the_pieces():
    with pytest.raises(illo.IllustrationError) as exc:
        illo.validate(GOOD + " M 1 1 L 2 2 M 3 3 L 4 4")
    assert "3 pieces" in str(exc.value)


def test_a_path_that_does_not_start_with_a_moveto_is_refused():
    with pytest.raises(illo.IllustrationError, match="moveto"):
        illo.validate("L 10 10 C 1 2 3 4 5 6 L 7 8 L 9 10 L 11 12")


def test_closing_the_loop_is_still_one_stroke():
    """`Z` returns to the start without lifting the pen, so it is allowed -
    but only at the end, since drawing on after a close needs a new moveto."""
    assert illo.validate(GOOD + " Z")
    with pytest.raises(illo.IllustrationError, match="continues after closing"):
        illo.validate(GOOD + " Z L 10 10")


# --- the other ways a path arrives unusable -------------------------------


def test_an_empty_path_is_refused():
    with pytest.raises(illo.IllustrationError, match="no path"):
        illo.validate("   ")


def test_a_stub_is_refused_as_too_small_to_be_a_drawing():
    with pytest.raises(illo.IllustrationError, match="too little"):
        illo.validate("M 10 10 L 20 20")


def test_coordinates_outside_the_frame_are_refused():
    with pytest.raises(illo.IllustrationError, match="outside"):
        illo.validate(GOOD + " L 9999 600")


def test_non_path_commands_are_refused():
    with pytest.raises(illo.IllustrationError, match="not SVG path commands"):
        illo.validate(GOOD + " GARBAGE 7 8")


def test_scientific_notation_is_not_mistaken_for_a_command():
    """`e` is part of a number, not a stray letter. Reading it as one would
    reject a perfectly good path for containing the number it was given."""
    assert illo.validate(GOOD.replace("C 620 180", "C 1e2 180"))


def test_whitespace_is_normalised_rather_than_rejected():
    messy = "  " + GOOD.replace(" ", "\n  ").replace("M", "M\t") + "   "
    assert illo.validate(messy) == GOOD


# --- pulling the two pieces out of a reply --------------------------------


def test_parse_reads_the_subject_and_the_path():
    subject, path = illo.parse(
        f"<<SUBJECT: a seated figure>>\n<<PATH: {GOOD}>>")
    assert subject == "a seated figure"
    assert path == GOOD


def test_parse_says_so_when_there_is_no_path():
    with pytest.raises(illo.IllustrationError, match="no <<PATH"):
        illo.parse("I drew you a lovely picture.")


def test_a_path_containing_commas_and_minus_signs_survives_parsing():
    """The reason these are markers and not JSON: path data is full of exactly
    the characters a parser would need escaped."""
    tricky = "M -10,20 C 1,-2 3,4 5,6 S 7,-8 9,10 Q 11,12 13,14"
    _, path = illo.parse(f"<<PATH: {tricky}>>")
    assert path == tricky


# --- the endpoint ---------------------------------------------------------


def test_the_endpoint_never_generates_without_a_key(client, monkeypatch):
    """Demo mode draws nothing and says why. A canned script does not answer
    the question, so a picture of it would be a picture of the wrong thing -
    and §51 is about demo output reaching the shared cache and outliving it."""
    monkeypatch.setattr(appmod, "DEMO_MODE", True)
    body = client.get("/api/illustration?q=anything&minutes=3").json()
    assert body["path"] == ""
    assert "No API key" in body["detail"]


def test_the_endpoint_is_off_when_switched_off(client, monkeypatch):
    import dataclasses
    monkeypatch.setattr(appmod, "settings",
                        dataclasses.replace(appmod.settings, illustration_enabled=False))
    body = client.get("/api/illustration?q=anything&minutes=3").json()
    assert body["path"] == ""
    assert "switched off" in body["detail"]


def test_a_rejected_drawing_is_a_blank_canvas_with_a_reason(client, monkeypatch):
    """A cut line must never reach the player. It becomes no picture, plus a
    sentence - not a broken one, and not a silent empty response."""
    monkeypatch.setattr(appmod, "DEMO_MODE", False)

    class Cut:
        client = None
        async def draw(self, *a, **k):
            raise illo.IllustrationError("the line is cut into 2 pieces")

    monkeypatch.setattr(illo, "IllustrationGenerator", lambda *a, **k: Cut())
    body = client.get("/api/illustration?q=anything&minutes=3").json()
    assert body["path"] == ""
    assert "cut into 2 pieces" in body["detail"]


def test_a_drawing_is_cached_and_the_second_ask_is_free(client, monkeypatch):
    monkeypatch.setattr(appmod, "DEMO_MODE", False)
    store = cache_mod.MemoryScriptCache()
    monkeypatch.setattr(appmod, "SCRIPT_CACHE", store)
    calls = []

    class Once:
        client = None
        async def draw(self, query, minutes, hint=""):
            calls.append(query)
            return illo.Illustration(path=GOOD, subject="a seated figure",
                                     source="model")

    monkeypatch.setattr(illo, "IllustrationGenerator", lambda *a, **k: Once())
    first = client.get("/api/illustration?q=how+sleep+works&minutes=3").json()
    assert first["source"] == "model" and first["path"] == GOOD

    second = client.get("/api/illustration?q=how+sleep+works&minutes=3").json()
    assert second["source"] == "cache"
    assert second["path"] == GOOD
    assert second["subject"] == "a seated figure"
    assert len(calls) == 1, "the second ask regenerated instead of replaying"


def test_the_drawing_is_filed_under_the_key_the_episode_uses(monkeypatch):
    """The guard against silent drift.

    `episode_key` and `PodcastPipeline._cache_key` must agree, or every
    drawing is written to a slot the episode never looks in - which would look
    exactly like "caching does not work" and cost a model call every play.
    """
    import asyncio
    import pipeline as pipeline_mod
    from tts import DebugEngine

    store = cache_mod.MemoryScriptCache()
    monkeypatch.setattr(appmod, "SCRIPT_CACHE", store)
    plan = plan_episode("how sleep works", 3)
    pipe = pipeline_mod.PodcastPipeline(generator=None, engine=DebugEngine(),
                                        cache=store)

    mine = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        appmod.episode_key(plan, None))
    theirs = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        pipe._cache_key(plan))
    assert mine == theirs, "illustration and episode disagree about the cache key"


def test_an_attachment_episode_is_never_drawn_into_the_shared_cache(monkeypatch):
    """An episode built on somebody's own document is theirs. No key means no
    write, so its picture cannot reach another listener either."""
    import asyncio
    from attachments import Attachment

    store = cache_mod.MemoryScriptCache()
    monkeypatch.setattr(appmod, "SCRIPT_CACHE", store)
    plan = plan_episode("summarise this", 3, attachments=(
        Attachment(id="a1", kind="document", name="mine.txt", text="private"),))
    key = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        appmod.episode_key(plan, None))
    assert key == "", "an attachment episode got a shared cache key"


def test_the_demo_pages_own_drawing_would_pass_the_server():
    """The demo page is the integration spec handed to whoever draws.

    A spec that shows a path the server would reject is worse than no spec, so
    its stroke goes through the same validator as a model's.
    """
    import pathlib
    import re

    html = (pathlib.Path(__file__).resolve().parent.parent
            / "preview" / "illustration-demo.html").read_text()
    match = re.search(r'id="line" d="(.*?)"', html, re.S)
    assert match, "the demo page has no path to check"
    assert illo.count_commands(illo.validate(match.group(1))) >= illo.MIN_COMMANDS
