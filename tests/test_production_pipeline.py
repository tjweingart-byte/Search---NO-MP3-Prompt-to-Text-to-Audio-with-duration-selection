"""Phase 6 is what production runs, and nothing quietly makes it legacy.

Making an architecture the default is not one line - it is a claim about every
path a deployment can take to a running server. This file is that claim,
checked at each layer rather than at the one that is easiest to assert:

    the constant        config.DEFAULT_PIPELINE
    the setting         Settings.streaming_pipeline, computed in a fresh process
    the request path    PodcastPipeline._phase6, and the pump a request is served
    the health report   what a running server says it is doing
    the repository      no file carrying a legacy default the others do not know

The second half is about the failure this project has paid for most. `_phase6`
used to be `settings.streaming_pipeline == "phase6"`, which makes every value
that is not exactly that string mean *legacy* - silently, at request time, on a
listener's episode. Harmless while legacy was the default and the intended
answer. A downgrade nobody chose now that it is not.
"""
from __future__ import annotations

import dataclasses
import importlib
import os
import pathlib
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
import pipeline as pipeline_module  # noqa: E402
from config import DEFAULT_PIPELINE, STREAMING_PIPELINES, settings  # noqa: E402
from pipeline import GenerationStats, PodcastPipeline  # noqa: E402
from speech_assembly import AssembledChunk  # noqa: E402
from tts import DebugEngine  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENGINE = DebugEngine()


def fresh_config(monkeypatch, value=None):
    """config as a brand-new process computes it, not as this one holds it."""
    monkeypatch.delenv("STREAMING_PIPELINE", raising=False)
    if value is not None:
        monkeypatch.setenv("STREAMING_PIPELINE", value)
    monkeypatch.setenv("FAM_IGNORE_DOTENV", "1")
    return importlib.reload(config)


# --------------------------------------------------------------------------
# a fresh deployment, with nothing set
# --------------------------------------------------------------------------
def test_a_fresh_deployment_selects_phase6(monkeypatch):
    """No STREAMING_PIPELINE anywhere: a new container, a new server, a new
    machine. It must arrive at Phase 6 without being told."""
    fresh = fresh_config(monkeypatch)
    try:
        assert "STREAMING_PIPELINE" not in os.environ
        assert fresh.DEFAULT_PIPELINE == "phase6"
        assert fresh.settings.streaming_pipeline == "phase6"
    finally:
        importlib.reload(config)


def test_the_default_is_one_fact_in_one_place():
    """The constant and the setting cannot disagree, because the setting is
    built from the constant rather than repeating the literal."""
    source = (ROOT / "config.py").read_text()
    assert 'os.environ.get(\n            "STREAMING_PIPELINE", DEFAULT_PIPELINE)' in source
    assert settings.streaming_pipeline == DEFAULT_PIPELINE


def test_a_real_request_is_served_by_the_phase6_pump():
    """The setting is not the behaviour. This asserts the pump a request
    actually gets, which is the thing a listener hears."""
    pipe = PodcastPipeline(generator=None, engine=ENGINE, cache=None)
    assert pipe._phase6() is True


def test_the_running_server_says_which_architecture_it_is():
    """A deployment rolled back by hand is indistinguishable from one that was
    not, unless it says so."""
    import asyncio

    import app

    report = asyncio.run(app.health())
    assert report["streaming_pipeline"] == "phase6"
    assert report["streaming_pipeline_default"] is True


def test_the_env_example_ships_the_default_it_documents():
    """Following the documented setup must not configure the product against
    itself - `.env.example` shipped the cold open and web search on while
    config had them off, and that cost a session (PROBLEMS.md 54)."""
    lines = [line.strip() for line in (ROOT / ".env.example").read_text().splitlines()]
    assert f"STREAMING_PIPELINE={DEFAULT_PIPELINE}" in lines


def test_no_file_in_the_repository_carries_a_stale_legacy_default():
    """The search the requirement asks for, as a test rather than a one-off.

    Any file that names STREAMING_PIPELINE and the word legacy together is
    either explaining the choice, taking a deliberate baseline, or is a stale
    default nobody has noticed. The first two are listed; anything else fails.
    """
    allowed = {
        "config.py",                     # declares both, defaults to phase6
        ".env.example",                  # documents both, ships phase6
        "pipeline.py",                   # branches on it
        "tools/pod_production_test.sh",  # runs both: the comparison
    }
    offenders = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix not in {".py", ".sh", ".example", ".md", ".yaml", ".txt"}:
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith("tests/") or relative in allowed:
            continue
        if relative in {"PROBLEMS.md", "CLAUDE.md"}:
            continue                      # the engineering log; history, not config
        text = path.read_text(errors="replace")
        if "STREAMING_PIPELINE" in text and "legacy" in text:
            offenders.append(relative)
    assert not offenders, f"a legacy default may be hiding in: {offenders}"


# --------------------------------------------------------------------------
# production cannot silently fall back to legacy
# --------------------------------------------------------------------------
@pytest.mark.parametrize("value", ["", "phase-6", "leagcy", "6", "phase6x",
                                   "none", "off", "true"])
def test_an_unrecognised_value_is_refused_at_import(value, monkeypatch):
    """The first gate. A typo must not start a server at all."""
    try:
        with pytest.raises(ValueError, match="not a pipeline"):
            fresh_config(monkeypatch, value)
    finally:
        # Clear it before reloading, or this reload raises the very error the
        # test just asserted - monkeypatch does not tear down until after the
        # test body, so the bad value is still in the environment here.
        monkeypatch.delenv("STREAMING_PIPELINE", raising=False)
        importlib.reload(config)


@pytest.mark.parametrize("value", ["PHASE6", " phase6 ", "Phase6", "LEGACY"])
def test_case_and_whitespace_are_normalised_rather_than_refused(value,
                                                                monkeypatch):
    """The tolerance is deliberate and worth pinning: a deployment writing
    `STREAMING_PIPELINE=PHASE6` means phase6, and refusing that would be
    pedantry rather than safety. It is the *unrecognisable* that is refused."""
    try:
        fresh = fresh_config(monkeypatch, value)
        assert fresh.settings.streaming_pipeline == value.strip().lower()
    finally:
        monkeypatch.delenv("STREAMING_PIPELINE", raising=False)
        importlib.reload(config)


@pytest.mark.parametrize("value", ["", "phase-6", "leagcy", "legacy_", "PHASE6"])
def test_an_unrecognised_value_is_refused_again_at_request_time(value):
    """The second gate, and the one that matters.

    `Settings.__post_init__` is bypassable - a test substituting a settings
    object, or any code that builds one without validation, reaches the request
    path with a value nobody checked. If `_phase6` answered by equality, every
    one of these would mean legacy: a downgrade, silent, mid-episode.
    """
    pipe = PodcastPipeline(generator=None, engine=ENGINE, cache=None)
    original = pipeline_module.settings
    pipeline_module.settings = type("Unvalidated", (),
                                    {"streaming_pipeline": value})()
    try:
        with pytest.raises(ValueError) as exc:
            pipe._phase6()
        assert "not a pipeline" in str(exc.value)
        assert "falling back" in str(exc.value), (
            "the refusal must say why it is not choosing for you")
    finally:
        pipeline_module.settings = original


def test_the_selector_is_not_an_equality_test_against_phase6():
    """Stated against the source, because the failure is a shape rather than a
    value: `== "phase6"` passes every test that only ever supplies valid
    values, and turns every invalid one into legacy in production."""
    import inspect

    source = inspect.getsource(PodcastPipeline._phase6)
    assert "in STREAMING_PIPELINES" in source, (
        "the selector must validate before it chooses")
    assert "raise" in source, "an unknown value must be refused, not defaulted"


def test_legacy_is_reachable_only_by_naming_it():
    """It is kept for the comparison baseline and the equivalence suite, and
    must never be arrived at by accident."""
    assert "legacy" in STREAMING_PIPELINES, "rollback must stay possible"
    pipe = PodcastPipeline(generator=None, engine=ENGINE, cache=None)
    original = pipeline_module.settings
    pipeline_module.settings = dataclasses.replace(
        settings, streaming_pipeline="legacy")
    try:
        assert pipe._phase6() is False
    finally:
        pipeline_module.settings = original


def test_an_episode_end_to_end_uses_the_phase6_pump_by_default():
    """Nothing stubbed but the model and the voice: the default carries all the
    way to the chunks a listener is served."""
    import asyncio

    from script_generator import plan_episode
    from tests.test_pipeline import FakeGenerator

    seen: list = []

    async def run():
        pipe = PodcastPipeline(generator=FakeGenerator(1.0), engine=ENGINE,
                               cache=None)
        real_speak = pipe._speak_chunk

        def spy(chunk, pace, stats):
            seen.append(chunk)
            return real_speak(chunk, pace, stats)

        pipe._speak_chunk = spy
        stats = GenerationStats()
        async for _ in pipe.stream_pcm(plan_episode("what is the nasdaq", 1),
                                       stats):
            pass

    asyncio.run(run())
    assert seen, "no chunk reached synthesis; the phase6 pump was not used"
    assert all(isinstance(chunk, AssembledChunk) for chunk in seen)


# --------------------------------------------------------------------------
# the voice, unchanged by any of this
# --------------------------------------------------------------------------
def test_chatterbox_is_still_the_only_production_voice():
    import tts

    assert [cls.name for cls in tts.PRODUCTION_ENGINES] == ["chatterbox"]
    assert tts.PLACEHOLDER_ENGINE is tts.DebugEngine


def test_piper_is_still_gone():
    import tts

    assert not hasattr(tts, "PiperEngine")
    assert "piper" not in {name.lower() for name in tts.DEV_ENGINES}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        assert "piper" not in line.split("#", 1)[0].lower()
