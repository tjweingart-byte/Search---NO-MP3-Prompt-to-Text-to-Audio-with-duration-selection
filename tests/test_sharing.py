"""Sharing an episode outside FAM: the link, the words, and the card."""
from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sharing


@pytest.fixture
def store(tmp_path):
    return sharing.ShareStore(str(tmp_path / "shares.db"))


EPISODE = {"title": "Who Makes the Chips",
           "question": "why is semiconductor manufacturing concentrated",
           "minutes": 3, "url": "https://fam.audio/s/abc123"}


# --- the templates --------------------------------------------------------

def test_every_target_renders_without_a_leftover_placeholder():
    """A share whose text says `{title}` is worse than no template at all, and
    it is the failure a typo in one of nine strings produces."""
    for target in sharing.TARGETS:
        rendered = sharing.render(target.key, **EPISODE)
        assert "{" not in rendered["text"], target.key
        assert "}" not in rendered["text"], target.key


def test_a_link_destination_carries_the_link():
    for key in ("copy", "sms", "email", "whatsapp", "x", "facebook", "linkedin"):
        assert EPISODE["url"] in sharing.render(key, **EPISODE)["text"], key


def test_email_has_a_subject_and_the_others_do_not():
    assert sharing.render("email", **EPISODE)["subject"]
    assert not sharing.render("sms", **EPISODE)["subject"]


def test_a_long_question_is_trimmed_to_what_the_platform_shows():
    """A caption cut off mid-word by the platform reads as a broken app rather
    than as a long post."""
    long = dict(EPISODE, question="why " + "semiconductor manufacturing " * 20)
    text = sharing.render("x", **long)["text"]
    assert len(text) <= sharing.target("x").max_chars


def test_trimming_never_cuts_the_link_off():
    """A share whose link is truncated is worse than one with a shorter
    sentence, so the trim comes out of the text and the link is re-attached."""
    long = dict(EPISODE, question="why " + "semiconductor manufacturing " * 20)
    text = sharing.render("x", **long)["text"]
    assert text.endswith(EPISODE["url"])


def test_an_unknown_destination_is_refused():
    with pytest.raises(sharing.ShareError):
        sharing.render("myspace", **EPISODE)


def test_stories_are_the_ones_that_need_a_picture():
    """Instagram and Snapchat stories cannot carry a link as text - they are
    pictures with a sticker on them. Getting this wrong means the listener
    shares a screenshot of a player UI."""
    needs = {t.key for t in sharing.TARGETS if t.needs_image}
    assert needs == {"instagram_story", "snapchat_story"}


def test_no_template_claims_the_episode_is_good():
    """FAM did not write that opinion and the person sharing has not typed one
    yet. The template is a default they finish, not a review."""
    for target in sharing.TARGETS:
        text = sharing.render(target.key, **EPISODE)["text"].lower()
        for word in ("amazing", "incredible", "best", "must-listen", "brilliant"):
            assert word not in text, target.key


# --- the card -------------------------------------------------------------

def test_the_card_is_a_portrait_story_sized_image():
    card = sharing.story_card(**{k: EPISODE[k] for k in ("title", "question", "minutes")})
    root = ET.fromstring(card)
    assert root.get("width") == "1080" and root.get("height") == "1920"


def test_the_card_is_well_formed_with_a_question_full_of_markup():
    """The question is text a listener typed and this is markup. An unescaped
    apostrophe or angle bracket produces an image that does not render, on the
    one surface where the failure is public."""
    card = sharing.story_card("A <b>bold</b> title & more",
                              'why do people say "it\'s <fine>"?', 3)
    ET.fromstring(card)  # raises if it is not well-formed
    assert "<b>" not in card


def test_a_long_title_is_wrapped_rather_than_running_off_the_card():
    card = sharing.story_card("A very long title " * 10, "a question", 3)
    root = ET.fromstring(card)
    spans = [e for e in root.iter() if e.tag.endswith("tspan")]
    assert 1 < len(spans) <= 8


def test_the_card_names_the_person_when_they_have_a_handle():
    assert "@ana" in sharing.story_card("T", "q", 3, handle="ana")
    assert "on FAM" in sharing.story_card("T", "q", 3)


# --- the store ------------------------------------------------------------

def test_sharing_the_same_episode_twice_returns_one_link(store):
    """One link with four destinations, not four links whose open counts have
    to be added up."""
    first = store.create("u", "why bonds move", 3, "Bonds")
    second = store.create("u", "why bonds move", 3, "Bonds")
    assert first["id"] == second["id"]


def test_two_listeners_sharing_the_same_episode_get_their_own_links(store):
    """Whose share was opened is the interesting question."""
    assert store.create("a", "q", 3)["id"] != store.create("b", "q", 3)["id"]


def test_opens_are_counted(store):
    share = store.create("u", "why bonds move", 3)
    store.opened(share["id"])
    store.opened(share["id"])
    assert store.get(share["id"])["opens"] == 2


def test_an_episode_with_no_question_cannot_be_shared(store):
    with pytest.raises(sharing.ShareError):
        store.create("u", "  ", 3)


def test_an_unknown_share_is_none_rather_than_an_error(store):
    assert store.get("not-a-share") is None


def test_forget_erases_a_listeners_shares(store):
    store.create("u", "q", 3)
    store.create("them", "q", 3)
    assert store.forget("u") == 1
    assert store.get(store.create("them", "q", 3)["id"]) is not None
