"""The standalone loading-screen page is extracted, not written twice.

The screen exists for as long as an episode takes to start, which is 450ms on
a cache hit and less against fixtures - correct, and useless for judging how
it looks. `tools/build_loading_demo.py` holds it still.

The risk that creates is the one this project keeps paying for: a second copy
that drifts, so the thing being looked at is not the thing that ships. These
pin the extraction rather than the appearance.
"""
from __future__ import annotations

import os
import pathlib
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import build_loading_demo  # noqa: E402


@pytest.fixture(scope="module")
def page():
    return build_loading_demo.build()


def test_the_markup_is_the_shipped_markup(page):
    """Not "looks similar" - the same element, lifted out."""
    index = (ROOT / "static" / "index.html").read_text()
    for fragment in ('<div class="fam-loading" id="famLoading"',
                     'class="fam-loading-word">FAMiliarizing',
                     'id="famLoadingStatus"',
                     '<path class="chev c3"'):
        assert fragment in index, f"{fragment!r} is no longer in the app"
        assert fragment in page, f"{fragment!r} did not reach the demo page"


def test_the_animation_is_the_shipped_animation(page):
    for rule in ("@keyframes famRise", "@keyframes famDot",
                 "prefers-reduced-motion", "animation-delay:.96s"):
        assert rule in page, f"{rule} was not extracted"


def test_the_palette_comes_from_the_app(page):
    assert "--copper:#E0B563" in page
    assert "var(--copper)" in page


def test_the_honest_wait_is_shown_not_replaced(page):
    """The brand animation frames the status line; it does not stand in for
    it. A demo that dropped the line would be showing a nicer version of the
    thing PROBLEMS.md 55 deleted."""
    assert "Writing your episode" in page
    assert "checking sources underneath" in page
    assert ".gen-text{" in page


def test_the_cache_hit_floor_is_demonstrated_at_its_real_value(page):
    """450 in the demo has to be the 450 the app actually holds for."""
    index = (ROOT / "static" / "index.html").read_text()
    assert "GEN_MIN_VISIBLE_MS = 450" in index
    assert "450" in page


def test_a_renamed_screen_fails_the_build_rather_than_shipping_an_empty_page():
    """The whole point of extracting is that it cannot silently succeed."""
    with pytest.raises(SystemExit):
        build_loading_demo.slice_between("nothing here", "<start>", "<end>", "a test")
