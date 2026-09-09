"""Exa retrieval, and Claude reading the packet instead of searching.

Two ways to research an episode, and the difference is who does the looking.
On `claude` the model gets Anthropic's `web_search` tool and searches inside
its own turn. On `exa` this codebase retrieves first, builds an evidence packet
and puts it in the prompt - so the model reads rather than searches.

The packet is byte-for-byte the one the manual benchmark measured on
2026-09-05 and the experiment layer then repeated. That is the point of it:
the numbers already taken by hand stay comparable, so a change of shape has to
be deliberate. Several tests below pin that shape for exactly that reason.

Nothing here needs an EXA_API_KEY, `exa_py`, or the network. The client is
stubbed at its import, so the real call signature is still exercised - a test
that stubbed `retrieve` itself would prove only that the stub works.
"""
from __future__ import annotations

import asyncio
import dataclasses
import os
import sys
import threading
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
import research  # noqa: E402
import script_generator as sg  # noqa: E402
from config import RESEARCH_BACKENDS, settings  # noqa: E402
from script_generator import ScriptNotes, build_prompt, plan_episode  # noqa: E402


class FakeResult:
    def __init__(self, title, url, highlights):
        self.title, self.url, self.highlights = title, url, highlights


class FakeReply:
    def __init__(self, results, cost=None):
        self.results = results
        self.cost_dollars = types.SimpleNamespace(total=cost) if cost else None


RESULTS = [
    FakeResult("Fed holds rates", "https://reuters.com/a",
               ["Rates held at 4.25%.", "Third hold running.", "Ignored third."]),
    FakeResult("Markets react", "https://ft.com/b", ["Yields fell 6bp."]),
    FakeResult("What it means", "https://reuters.com/c", ["Cuts priced for June."]),
    FakeResult("Fourth source", "https://bbc.co.uk/d", ["Should not appear."]),
]


@pytest.fixture
def exa(monkeypatch):
    """A stubbed Exa client, recording exactly how it was called."""
    calls: list = []

    class FakeExa:
        def __init__(self, key):
            calls.append({"key": key})

        def search_and_contents(self, query, **kwargs):
            calls.append({"query": query, "thread": threading.current_thread().name,
                          **kwargs})
            return FakeReply(RESULTS, cost=0.0031)

    module = types.ModuleType("exa_py")
    module.Exa = FakeExa
    monkeypatch.setitem(sys.modules, "exa_py", module)
    monkeypatch.setenv("EXA_API_KEY", "exa-test-key-not-real")
    monkeypatch.setattr(research, "settings",
                        dataclasses.replace(settings, research_backend="exa"))
    return calls


def use_backend(monkeypatch, value: str):
    patched = dataclasses.replace(settings, research_backend=value)
    monkeypatch.setattr(research, "settings", patched)
    monkeypatch.setattr(sg, "settings", patched)
    return patched


# --------------------------------------------------------------------------
# the packet, pinned to the benchmark
# --------------------------------------------------------------------------
def test_the_packet_is_shaped_as_the_benchmark_built_it():
    """`SOURCE n / Title: / Key evidence:`, top 3, 2 highlights each."""
    packet = research.build_packet(RESULTS, packet_sources=3,
                                   highlights_per_source=2)
    assert packet.startswith("SOURCE 1\nTitle: Fed holds rates\nKey evidence:\n")
    assert "SOURCE 3" in packet and "SOURCE 4" not in packet
    assert "Ignored third." not in packet, "more than 2 highlights reached the packet"
    assert "Should not appear." not in packet, "a 4th source reached the packet"


def test_the_packet_carries_no_urls_or_numbers_for_the_voice_to_read():
    """It is spoken aloud downstream. A URL in the packet is a URL a model can
    read out, and a listener is not looking at a citation list."""
    packet = research.build_packet(RESULTS, 3, 2)
    assert "https://" not in packet
    assert "reuters.com" not in packet


def test_an_empty_result_set_is_an_empty_packet():
    assert research.build_packet([], 3, 2) == ""
    assert not research.Packet(context="")
    assert not research.Packet(context="   \n ")


def test_domains_are_distinct_and_ordered_for_a_person_to_judge():
    assert research.domains(RESULTS) == ["reuters.com", "ft.com", "bbc.co.uk"]


# --------------------------------------------------------------------------
# the call
# --------------------------------------------------------------------------
def test_retrieval_makes_the_call_the_benchmark_made(exa):
    packet = asyncio.run(research.retrieve("what did the fed do"))
    call = [c for c in exa if "query" in c][0]
    assert call["query"] == "what did the fed do"
    assert call["type"] == research.DEFAULT_SEARCH_TYPE == "fast"
    assert call["num_results"] == 8
    assert call["highlights"] is True
    assert packet.backend == "exa" and packet.searches == 1
    assert packet.results_returned == 4


def test_retrieval_does_not_run_on_the_event_loop(exa):
    """The cover is speaking and the assembler is batching while this runs. A
    synchronous HTTP call on the loop would stop both - the same guarantee the
    speech engines are held to."""
    asyncio.run(research.retrieve("anything"))
    call = [c for c in exa if "query" in c][0]
    assert call["thread"] != "MainThread", (
        f"retrieval ran on the event loop, in {call['thread']}")


def test_the_reported_cost_is_the_response_s_own_when_it_gives_one(exa):
    assert asyncio.run(research.retrieve("q")).cost == pytest.approx(0.0031)


def test_a_response_without_a_cost_falls_back_to_the_published_rate(monkeypatch):
    class Silent:
        def __init__(self, key):
            pass

        def search_and_contents(self, query, **kwargs):
            return FakeReply(RESULTS, cost=None)

    module = types.ModuleType("exa_py")
    module.Exa = Silent
    monkeypatch.setitem(sys.modules, "exa_py", module)
    monkeypatch.setenv("EXA_API_KEY", "k")
    monkeypatch.setattr(research, "settings",
                        dataclasses.replace(settings, research_backend="exa"))
    assert asyncio.run(research.retrieve("q")).cost == research.COST_PER_SEARCH


def test_the_credential_never_reaches_the_result(exa):
    packet = asyncio.run(research.retrieve("q"))
    assert "exa-test-key-not-real" not in str(packet.as_dict())
    assert "exa-test-key-not-real" not in packet.context


# --------------------------------------------------------------------------
# it refuses rather than guessing
# --------------------------------------------------------------------------
def test_the_claude_backend_retrieves_nothing_and_that_is_not_an_error(monkeypatch):
    use_backend(monkeypatch, "claude")
    packet = asyncio.run(research.retrieve("what did the fed do"))
    assert packet.backend == "claude"
    assert not packet, "the claude backend must not produce a packet"


@pytest.mark.parametrize("value", ["exaa", "web", "google", "none", "exa-py"])
def test_an_unrecognised_backend_is_refused_at_retrieval(value, monkeypatch):
    """The second gate. `Settings.__post_init__` is bypassable; this is not.

    `""` is deliberately not here: an empty override means "not specified", so
    it falls through to the configured backend rather than being a bad value.
    Nor is `"Exa "` - case and surrounding whitespace normalise, exactly as
    they do for STREAMING_PIPELINE, because a deployment writing `EXA` means
    exa and refusing that is pedantry rather than safety.
    """
    monkeypatch.setattr(research, "settings",
                        dataclasses.replace(settings, research_backend="claude"))
    with pytest.raises(research.ResearchUnavailable) as exc:
        asyncio.run(research.retrieve("q", backend=value))
    assert "is not a backend" in str(exc.value)
    assert "falling back" in str(exc.value)


@pytest.mark.parametrize("value", ["EXA", " exa ", "Exa", "CLAUDE"])
def test_case_and_whitespace_normalise_rather_than_being_refused(value,
                                                                 monkeypatch,
                                                                 exa):
    """The tolerance is deliberate, and the same as STREAMING_PIPELINE's: a
    deployment writing `RESEARCH_BACKEND=EXA` means exa. It is the
    unrecognisable that is refused, not the differently-cased."""
    packet = asyncio.run(research.retrieve("q", backend=value))
    assert packet.backend == value.strip().lower()


def test_a_missing_package_says_so_rather_than_failing_obscurely(monkeypatch):
    monkeypatch.setitem(sys.modules, "exa_py", None)
    monkeypatch.setenv("EXA_API_KEY", "k")
    monkeypatch.setattr(research, "settings",
                        dataclasses.replace(settings, research_backend="exa"))
    with pytest.raises(research.ResearchUnavailable, match="exa_py is not installed"):
        asyncio.run(research.retrieve("q"))


def test_a_missing_key_names_the_variable_and_the_way_out(monkeypatch):
    module = types.ModuleType("exa_py")
    module.Exa = lambda key: None
    monkeypatch.setitem(sys.modules, "exa_py", module)
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.setattr(research, "settings",
                        dataclasses.replace(settings, research_backend="exa"))
    with pytest.raises(research.ResearchUnavailable) as exc:
        asyncio.run(research.retrieve("q"))
    assert "EXA_API_KEY" in str(exc.value)
    assert "RESEARCH_BACKEND=claude" in str(exc.value)


def test_exa_failure_does_not_become_a_claude_search():
    """The failure this refuses. An episode that asked for Exa and quietly got
    the model's own search is unattributable - and would report Exa's cost of
    zero while paying Claude's.

    Parsed rather than grepped: a substring search for "except" also matches
    the word "exception" in a comment, which is how this test first passed
    while proving nothing.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(research.retrieve)))
    handlers = [node for node in ast.walk(tree)
                if isinstance(node, ast.ExceptHandler)]
    assert not handlers, (
        "retrieve catches something; a failed backend must reach the caller "
        "rather than being turned into a different kind of research")


def test_a_backend_that_cannot_run_raises_instead_of_switching(monkeypatch):
    """The behaviour that check is about, exercised rather than read."""
    monkeypatch.setitem(sys.modules, "exa_py", None)
    monkeypatch.setattr(research, "settings",
                        dataclasses.replace(settings, research_backend="exa"))
    with pytest.raises(research.ResearchUnavailable):
        asyncio.run(research.retrieve("q"))


# --------------------------------------------------------------------------
# Claude reads the packet
# --------------------------------------------------------------------------
def test_evidence_reaches_the_prompt_and_the_tool_does_not(exa, monkeypatch):
    """The whole point: on `exa`, Claude reads rather than searches."""
    use_backend(monkeypatch, "exa")
    monkeypatch.setattr(research, "settings",
                        dataclasses.replace(settings, research_backend="exa"))
    generator = sg.ScriptGenerator.__new__(sg.ScriptGenerator)
    plan = plan_episode("what did the fed do today", 3, search=True)
    notes = ScriptNotes()

    researched = asyncio.run(generator.research(plan, notes))
    assert researched.evidence, "no evidence was attached to the plan"

    prompt = build_prompt(researched)
    assert "<evidence>" in prompt
    assert "Rates held at 4.25%." in prompt
    kwargs = generator._request_kwargs(researched)
    assert "tools" not in kwargs, (
        "the search tool was attached on top of an evidence packet")


def test_the_claude_backend_keeps_the_tool_and_adds_no_evidence(monkeypatch):
    use_backend(monkeypatch, "claude")
    generator = sg.ScriptGenerator.__new__(sg.ScriptGenerator)
    plan = plan_episode("what did the fed do today", 3, search=True)
    researched = asyncio.run(generator.research(plan, ScriptNotes()))
    assert researched.evidence == ""
    assert "<evidence>" not in build_prompt(researched)
    assert generator._request_kwargs(researched)["tools"][0]["name"] == "web_search"


def test_an_unresearched_episode_never_retrieves(exa, monkeypatch):
    """The default question costs nothing either way, and must not reach Exa."""
    use_backend(monkeypatch, "exa")
    generator = sg.ScriptGenerator.__new__(sg.ScriptGenerator)
    plan = plan_episode("what is the nasdaq", 3, search=False)
    assert asyncio.run(generator.research(plan, ScriptNotes())) is plan
    assert not [c for c in exa if "query" in c], "an unresearched episode searched"


def test_an_empty_packet_leaves_the_tool_attached(monkeypatch):
    """Retrieval succeeded and found nothing. The episode is still answerable,
    and the model searches after all rather than being handed an empty packet
    and told it is research."""
    class Empty:
        def __init__(self, key):
            pass

        def search_and_contents(self, query, **kwargs):
            return FakeReply([])

    module = types.ModuleType("exa_py")
    module.Exa = Empty
    monkeypatch.setitem(sys.modules, "exa_py", module)
    monkeypatch.setenv("EXA_API_KEY", "k")
    use_backend(monkeypatch, "exa")

    generator = sg.ScriptGenerator.__new__(sg.ScriptGenerator)
    plan = plan_episode("todays news", 3, search=True)
    researched = asyncio.run(generator.research(plan, ScriptNotes()))
    assert researched.evidence == ""
    assert "tools" in generator._request_kwargs(researched)


def test_the_prompt_tells_the_model_not_to_read_sources_aloud(exa, monkeypatch):
    use_backend(monkeypatch, "exa")
    generator = sg.ScriptGenerator.__new__(sg.ScriptGenerator)
    plan = asyncio.run(generator.research(
        plan_episode("what did the fed do today", 3, search=True), ScriptNotes()))
    prompt = build_prompt(plan)
    assert "Never read a source's title, number or URL aloud" in prompt
    assert "they win and you say so plainly" in prompt, (
        "the model must be told the evidence outranks what it recalls")


def test_what_retrieval_cost_is_recorded_for_a_person_to_read(exa, monkeypatch):
    use_backend(monkeypatch, "exa")
    generator = sg.ScriptGenerator.__new__(sg.ScriptGenerator)
    notes = ScriptNotes()
    asyncio.run(generator.research(
        plan_episode("what did the fed do today", 3, search=True), notes))
    assert notes.research["backend"] == "exa"
    assert notes.research["sources"] == ["reuters.com", "ft.com", "bbc.co.uk"]
    assert notes.research["cost"] == pytest.approx(0.0031)
    assert notes.research["packet_chars"] > 0


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
def test_a_fresh_deployment_researches_with_exa():
    """The production default. Reversed from `claude`, deliberately, and with
    a real cost attached: it needs a second credential."""
    assert config.DEFAULT_RESEARCH_BACKEND == "exa"
    assert settings.research_backend == "exa"
    assert "claude" in RESEARCH_BACKENDS and "exa" in RESEARCH_BACKENDS


def test_the_default_is_one_fact_in_one_place():
    """The constant and the setting cannot disagree, because the setting is
    built from the constant rather than repeating the literal."""
    from pathlib import Path

    source = (Path(__file__).resolve().parent.parent / "config.py").read_text()
    assert '"RESEARCH_BACKEND", DEFAULT_RESEARCH_BACKEND' in source
    assert settings.research_backend == config.DEFAULT_RESEARCH_BACKEND


def test_rollback_to_claude_is_still_one_variable(monkeypatch):
    """A deployment with no Exa key must have a working configuration to move
    to, not a broken one to endure."""
    patched = dataclasses.replace(settings, research_backend="claude")
    monkeypatch.setattr(research, "settings", patched)
    assert patched.research_backend == "claude"
    assert not asyncio.run(research.retrieve("q")), (
        "the fallback configuration must need no key and no package")


def test_the_env_example_ships_the_default_it_documents():
    """Following the documented setup must not configure the product against
    itself - PROBLEMS.md 54."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    lines = [line.strip() for line in (root / ".env.example").read_text().splitlines()]
    assert f"RESEARCH_BACKEND={config.DEFAULT_RESEARCH_BACKEND}" in lines


@pytest.mark.parametrize("value", ["exaa", "web", "", "google"])
def test_an_unrecognised_backend_is_refused_at_import(value):
    with pytest.raises(ValueError, match="is not a research backend"):
        dataclasses.replace(settings, research_backend=value)


@pytest.mark.parametrize("field, value", [
    ("exa_num_results", 0), ("exa_packet_sources", 0),
    ("exa_highlights_per_source", 0), ("exa_num_results", -1),
])
def test_a_zero_knob_is_refused_rather_than_sending_an_empty_packet(field, value):
    with pytest.raises(ValueError, match="must be at least 1"):
        dataclasses.replace(settings, **{field: value})


def test_the_packet_cannot_ask_for_more_sources_than_were_fetched():
    with pytest.raises(ValueError, match="exceeds"):
        dataclasses.replace(settings, exa_num_results=3, exa_packet_sources=5)


def test_health_says_which_backend_and_whether_it_can_run():
    report = research.report()
    assert report["backend"] == "exa"
    assert "exa_detail" in report
    # In this container there is no EXA_API_KEY, and the default is now exa -
    # so `unavailable` is true, and that is the honest answer rather than a
    # test failure. What must never happen is it reading false while research
    # cannot run.
    ok, _ = research.diagnose()
    assert report["unavailable"] is (not ok)


def test_a_deployment_that_cannot_research_says_so_at_startup(caplog):
    """Not on a listener's first researched question. `exa` is the default and
    needs a credential; a missing one discovered mid-episode is the shape of
    failure this project has paid for most."""
    import logging

    import app

    with caplog.at_level(logging.WARNING, logger="app"):
        app._announce_research()

    if research.available():  # pragma: no cover - not in this container
        pytest.skip("this machine can research; nothing to announce")
    messages = [record.getMessage() for record in caplog.records]
    assert any("RESEARCH UNAVAILABLE" in m for m in messages), messages
    assert any("will FAIL rather than search another way" in m for m in messages)
    assert any("RESEARCH_BACKEND=claude" in m for m in messages), (
        "the warning must name the working configuration to move to")


def test_the_startup_warning_is_not_fatal():
    """Most questions are not researched. An app that refuses to start because
    one path is unconfigured is worse than one that starts and says which."""
    import app

    assert app._announce_research() is None


def test_health_flags_a_deployment_that_asked_for_exa_and_cannot_run_it(monkeypatch):
    monkeypatch.setattr(research, "settings",
                        dataclasses.replace(settings, research_backend="exa"))
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    report = research.report()
    assert report["unavailable"] is True
    assert "EXA_API_KEY" in report["exa_detail"]


def test_exa_is_not_a_fresh_install_requirement():
    """`requirements.txt` must not pull it in - the default backend needs it
    never, and the suite runs without it."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for line in (root / "requirements.txt").read_text().splitlines():
        assert "exa" not in line.split("#", 1)[0].lower()
    assert (root / "requirements-exa.txt").exists()


# --------------------------------------------------------------------------
# through the real pipeline, end to end
# --------------------------------------------------------------------------
def test_a_researched_episode_plays_with_claude_reading_the_exa_packet(exa,
                                                                       monkeypatch):
    """The whole path: Exa retrieves, the packet reaches the prompt, the model
    writes from it, the assembler chunks it and the engine speaks it.

    Nothing is stubbed but Exa itself and the two things that always are in
    this suite - the model and the voice. In particular the pipeline, the
    Phase 6 assembler and the duration contract are the real ones.
    """
    import pipeline as pipeline_mod
    from pipeline import GenerationStats, PodcastPipeline
    from tts import DebugEngine

    use_backend(monkeypatch, "exa")
    monkeypatch.setattr(research, "settings",
                        dataclasses.replace(settings, research_backend="exa"))
    # This test is about the split, so it asks for the cover. Unset,
    # ANSWER_FIRST now follows the backend and Exa does not get one.
    monkeypatch.setattr(pipeline_mod, "settings", dataclasses.replace(
        pipeline_mod.settings, answer_first=True))

    seen_prompts: list = []

    class ReadsThePacket(sg.ScriptGenerator):
        """The real generator's research step, with only the model faked."""

        def __init__(self):
            pass

        async def stream_sentences(self, plan, notes=None):
            plan = await self.research(plan, notes)
            seen_prompts.append(build_prompt(plan))
            for i in range(30):
                yield f"Rates were held at four and a quarter percent, point {i}."

        async def top_up(self, plan, spoken_so_far, words_needed):
            return
            yield ""  # pragma: no cover

    async def run():
        stats = GenerationStats()
        pipe = PodcastPipeline(generator=ReadsThePacket(), engine=DebugEngine(),
                               cache=None)
        total = 0
        async for chunk in pipe.stream_pcm(
                plan_episode("what did the fed do today", 1, search=True), stats):
            total += len(chunk)
        return stats, total

    stats, total = asyncio.run(run())

    assert total > 0, "no audio was produced"
    assert stats.sentences > 0
    assert seen_prompts, "the generator never ran"

    # This is a researched question, so ANSWER_FIRST split it in two and both
    # halves came through here. Which half got the evidence is the thing worth
    # asserting, and it is not the first one:
    #
    #   the cover      search=False, answers from knowledge, no retrieval
    #   the research   search=True, reads the packet
    #
    # A single assertion on `seen_prompts[0]` would have been checking the
    # cover and calling it the researched half.
    covers = [p for p in seen_prompts if "<evidence>" not in p]
    researched = [p for p in seen_prompts if "<evidence>" in p]
    assert covers, "no from-knowledge half ran; the listener waited on Exa"
    assert researched, "the packet never reached the model"
    assert "Rates held at 4.25%." in researched[0]
    assert "Yields fell 6bp." in researched[0], "only one source reached the prompt"

    # One retrieval for the episode - not one per sentence, and not one per half.
    assert len([c for c in exa if "query" in c]) == 1
    assert stats.answered_first is True
    assert stats.audio_seconds <= 60 + 5, "the duration ceiling did not hold"


def test_research_runs_once_per_episode_not_once_per_sentence(exa, monkeypatch):
    """`stream_sentences` calls `research` on entry. If that were inside the
    streaming loop it would pay for a retrieval per sentence, silently."""
    use_backend(monkeypatch, "exa")
    monkeypatch.setattr(research, "settings",
                        dataclasses.replace(settings, research_backend="exa"))
    generator = sg.ScriptGenerator.__new__(sg.ScriptGenerator)
    plan = plan_episode("what did the fed do today", 3, search=True)

    researched = asyncio.run(generator.research(plan, ScriptNotes()))
    # Researching an already-researched plan must be a no-op, or the answer
    # -first path (two calls on one plan) would retrieve twice.
    again = asyncio.run(generator.research(researched, ScriptNotes()))
    assert again is researched
    assert len([c for c in exa if "query" in c]) == 1


def test_the_evidence_survives_the_answer_first_split(exa, monkeypatch):
    """`_answer_first` replaces the plan twice - once for each half. The
    researched half must keep its evidence through that."""
    use_backend(monkeypatch, "exa")
    monkeypatch.setattr(research, "settings",
                        dataclasses.replace(settings, research_backend="exa"))
    generator = sg.ScriptGenerator.__new__(sg.ScriptGenerator)
    plan = plan_episode("what did the fed do today", 3, search=True)

    # The cover half: search off, so it must never retrieve.
    instant = dataclasses.replace(plan, search=False, role="opening")
    assert asyncio.run(generator.research(instant, ScriptNotes())) is instant
    assert not [c for c in exa if "query" in c], "the cover half searched"

    # The researched half retrieves and keeps the packet through `replace`.
    continuation = dataclasses.replace(plan, search=True, role="continuation")
    done = asyncio.run(generator.research(continuation, ScriptNotes()))
    assert done.evidence and done.role == "continuation"
