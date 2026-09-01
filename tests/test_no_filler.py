"""The filler is gone, and search is decided per question.

Two decisions, taken deliberately after hearing the product run:

**The cold open was removed, not switched off.** It existed to cover the
research wait. At 18 words it covered five seconds of a 30-45 second wait,
which is not covering anything - and it was prompted to state no facts, so
the five seconds it did cover were worthless. A setting left behind is an
invitation to turn it back on, so there is no setting.

**Search is opt-in, and the question opts in.** Paying 10-25 seconds on every
episode bought nothing for "what is the NASDAQ". The same keyword signal the
cache already uses to decide freshness now decides research.
"""
from __future__ import annotations

import asyncio
import dataclasses
import os
import pathlib
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cache as cache_mod  # noqa: E402
import config as config_mod  # noqa: E402
import demo_script  # noqa: E402
import pipeline as pipeline_mod  # noqa: E402
import script_generator as sg  # noqa: E402
from pipeline import GenerationStats, PodcastPipeline  # noqa: E402
from script_generator import plan_episode  # noqa: E402
from tts import DebugEngine  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


# --- the filler is gone -----------------------------------------------------

def test_no_module_can_still_produce_an_opener():
    """Not a flag: the code path does not exist."""
    assert not hasattr(sg.ScriptGenerator, "cold_open")
    assert not hasattr(demo_script.DemoGenerator, "cold_open")
    assert not hasattr(pipeline_mod.PodcastPipeline, "_run_cold_open")
    assert not hasattr(config_mod.settings, "enable_cold_open")


def test_the_settings_are_gone_too():
    """A knob left behind is an invitation to turn it back on."""
    for name in ("enable_cold_open", "cold_open_model", "cold_open_words",
                 "cold_open_max_seconds", "cold_open_grace"):
        assert not hasattr(config_mod.settings, name), f"{name} survived"


def test_no_source_file_still_speaks_before_the_script():
    """The whole point: nothing is spoken until the real episode is."""
    for path in ROOT.glob("*.py"):
        text = path.read_text()
        assert "cold_open" not in text, f"{path.name} still references the opener"


def test_a_generator_that_still_offers_one_is_ignored():
    """An old or third-party generator with a cold_open method must not
    resurrect the behaviour through duck typing."""

    class Nostalgic:
        async def stream_sentences(self, plan, notes=None):
            for _ in range(6):
                yield "A real sentence of the actual episode."

        async def cold_open(self, plan, spoken=""):  # pragma: no cover - must not run
            raise AssertionError("the pipeline called an opener")

    stats = GenerationStats()
    pipe = PodcastPipeline(generator=Nostalgic(), engine=DebugEngine(), cache=None)

    async def run():
        async for _ in pipe.stream_pcm(plan_episode("a question", 1), stats):
            pass

    asyncio.run(run())
    assert stats.sentences > 0


# --- search is decided by the question --------------------------------------

@pytest.mark.parametrize("query", [
    "latest news on the fed",
    "what happened today in golf",
    "the current score",
    "breaking news about the election",
    "what is happening right now with rates",
])
def test_a_question_about_a_moving_target_is_researched(query):
    assert cache_mod.needs_fresh_information(query) is True
    assert plan_episode(query, 3).search is True


@pytest.mark.parametrize("query", [
    "what is the NASDAQ",
    "why founders are taking back control",
    "how does a heat pump work",
    "why the Roman republic fell",
    "what habit research actually shows about lasting change",
])
def test_an_evergreen_question_answers_from_memory(query):
    """These are the questions that were paying 30-45 seconds for nothing."""
    assert cache_mod.needs_fresh_information(query) is False
    assert plan_episode(query, 3).search is False


def test_an_explicit_request_still_wins_both_ways():
    """"Opt in" has to mean the listener can actually opt in - and out."""
    assert plan_episode("what is the NASDAQ", 3, search=True).search is True
    assert plan_episode("latest news on the fed", 3, search=False).search is False


@pytest.mark.parametrize("mode, expected", [("always", True), ("never", False)])
def test_the_mode_overrides_the_question(monkeypatch, mode, expected):
    monkeypatch.setattr(sg, "settings", dataclasses.replace(sg.settings, search_mode=mode))
    assert plan_episode("what is the NASDAQ", 3).search is expected
    assert plan_episode("latest news on the fed", 3).search is expected


def test_the_old_switch_still_forces_search_on(monkeypatch):
    """ENABLE_WEB_SEARCH=1 in someone's .env must not silently become auto."""
    monkeypatch.setenv("ENABLE_WEB_SEARCH", "1")
    monkeypatch.delenv("FAM_IGNORE_DOTENV", raising=False)
    fresh = config_mod.Settings()
    assert fresh.search_mode == "always"


# --- the interface can tell the truth about the wait ------------------------

def test_the_interface_is_served_the_same_word_list_the_server_decides_with():
    """A second copy in JavaScript would drift, and then the listener is told
    one thing while the server does another."""
    words = cache_mod.research_words()
    assert "latest" in words and "today" in words
    index = (ROOT / "static" / "index.html").read_text()
    assert "h.research_words" in index, "the interface does not read the list"
    assert "RESEARCH_WORDS" in index and "willResearch" in index


def test_the_wait_says_what_it_is_waiting_for():
    index = (ROOT / "static" / "index.html").read_text()
    assert "checking sources underneath" in index, "no honest state for a researched episode"
    assert "Writing your episode" in index, "no honest state for an instant one"
    assert "startHonestWait" in index
