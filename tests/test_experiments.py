"""The experiment engine, checked without a key, a GPU or a speech engine.

The tests that matter most here are not the arithmetic ones - they are the
refusals. An engine that quietly measures the wrong thing, spends money nobody
approved, or writes a key into a report is worse than no engine, so each of
those is pinned by a test that fails loudly.
"""
from __future__ import annotations

import asyncio
import json
import os
import pathlib

import pytest

from experiments import cost as cost_mod
from experiments import nl, redact, registry, report, stats, store
from experiments.adapters.base import Availability, InfrastructureRequired
from experiments.adapters.search import SEARCH_ADAPTERS, AnthropicWebSearch, ExaSearch
from experiments.adapters.tts import CHATTERBOX_ENDPOINT_ENV, ChatterboxTTS, LocalTTS
from experiments.fakes import FakeGenerator, FakeSearch, FakeTTS
from experiments.harness import Harness, first_chunk_ready
from experiments.spec import Arm, ExperimentSpec
from experiments.timeline import Timeline


# --------------------------------------------------------------------------
# Secrets never reach disk
# --------------------------------------------------------------------------
SAMPLE_KEYS = [
    "sk-ant-api03-AAAABBBBCCCCDDDDEEEEFFFF",
    "sk-proj-ABCDEFGHIJKLMNOPQRSTUVWX",
    "1f0e3dad-9999-4b35-8a2c-99992a4dfa11",
    "hf_ABCDEFGHIJKLMNOPQRSTUV",
    "rpa_ABCDEFGHIJKLMNOPQRSTUVWX",
]


@pytest.mark.parametrize("key", SAMPLE_KEYS)
def test_every_key_shape_is_scrubbed(key):
    assert key not in redact.scrub_text(f"before {key} after")
    assert redact.looks_like_secret(key)


def test_scrub_reaches_into_nested_structures():
    payload = {"api_key": SAMPLE_KEYS[0], "nested": [{"token": SAMPLE_KEYS[1]}], "n": 5}
    clean = redact.scrub(payload)
    assert clean["api_key"] == redact.MASK
    assert SAMPLE_KEYS[1] not in json.dumps(clean)
    assert clean["n"] == 5


def test_env_secret_is_scrubbed_even_with_an_unknown_shape(monkeypatch):
    monkeypatch.setenv("SOMETHING_API_KEY", "totally-unusual-shape-9999")
    assert "totally-unusual-shape-9999" not in redact.scrub_text("k totally-unusual-shape-9999")


def test_short_config_values_are_not_mistaken_for_secrets(monkeypatch):
    monkeypatch.setenv("CACHE_KEY_MODE", "0")
    assert redact.scrub_text("mode is 0") == "mode is 0"


def test_store_writes_nothing_credential_shaped(tmp_path):
    spec = ExperimentSpec(name="leak check", arms=[Arm("a")], queries=["q"], trials=3,
                          notes=f"key {SAMPLE_KEYS[0]}")
    run = store.create(spec, root=tmp_path)
    run.append_trial({"index": 1, "ok": True, "api_key": SAMPLE_KEYS[0]})
    run.write_report(f"the key was {SAMPLE_KEYS[0]}")
    run.write_summary({"token": SAMPLE_KEYS[1]})
    assert run.verify_clean() == []
    for path in run.path.rglob("*"):
        if path.is_file():
            assert SAMPLE_KEYS[0] not in path.read_text(errors="ignore")


# --------------------------------------------------------------------------
# Paid infrastructure is never started, and never silently skipped
# --------------------------------------------------------------------------
def test_chatterbox_without_an_endpoint_needs_approval(monkeypatch):
    monkeypatch.delenv(CHATTERBOX_ENDPOINT_ENV, raising=False)
    state = ChatterboxTTS().available()
    assert state.ok is False
    assert state.needs_approval is True
    assert "GPU" in state.reason


def test_chatterbox_raises_rather_than_provisioning(monkeypatch):
    monkeypatch.delenv(CHATTERBOX_ENDPOINT_ENV, raising=False)
    with pytest.raises(InfrastructureRequired):
        asyncio.run(ChatterboxTTS().synth("hello", Timeline()))


def test_no_module_in_the_engine_can_control_a_pod():
    """No provisioning SDK, and no lifecycle verb, anywhere in the package."""
    root = pathlib.Path(__file__).resolve().parent.parent / "experiments"
    banned = ("import runpod", "from runpod", "pods.create", "create_pod",
              "start_pod", "stop_pod", "resume_pod", "terminate_pod")
    for path in root.rglob("*.py"):
        text = path.read_text()
        for needle in banned:
            assert needle not in text, f"{path.name} contains {needle!r}"


def test_preflight_separates_approval_from_misconfiguration(monkeypatch):
    monkeypatch.delenv(CHATTERBOX_ENDPOINT_ENV, raising=False)
    spec = ExperimentSpec(
        name="p", trials=3, queries=["q"],
        arms=[Arm("gpu", search="none", tts="chatterbox"),
              Arm("exa", search="exa", tts="none")])
    pre = registry.preflight(spec)
    assert pre.ok is False
    assert any("chatterbox" in a for a in pre.approvals)
    assert any("exa" in b for b in pre.blockers)


# --------------------------------------------------------------------------
# Nothing is measured that cannot honestly be measured
# --------------------------------------------------------------------------
def test_exa_adapter_refuses_rather_than_inventing_results():
    state = ExaSearch().available()
    assert state.ok is False
    assert "exa_claude_benchmark" in state.reason or "implementation" in state.reason
    with pytest.raises(RuntimeError):
        asyncio.run(ExaSearch().search("q", Timeline()))


def test_local_tts_will_not_pass_the_debug_tone_off_as_piper():
    """The silent-success failure this project has paid for twice."""
    from tts import DebugEngine, engine_for_voice

    if not isinstance(engine_for_voice(None), DebugEngine):
        pytest.skip("this machine has a real speech engine")
    assert LocalTTS().available().ok is False
    assert LocalTTS(voice="debug:tone").available().ok is True


def test_server_side_search_is_declared_not_separable():
    assert AnthropicWebSearch().separable is False
    result = asyncio.run(AnthropicWebSearch().search("q", Timeline()))
    assert result.context == ""


def test_a_non_separable_fake_records_no_search_stage():
    fake = FakeSearch(0.0, adapter_id="anthropic_web_search")
    fake.separable = False
    timeline = Timeline()
    asyncio.run(fake.search("q", timeline))
    assert [s.name for s in timeline.stages] == []


# --------------------------------------------------------------------------
# Timeline, including the cross-machine part
# --------------------------------------------------------------------------
def test_timeline_records_a_failed_stage_rather_than_losing_it():
    timeline = Timeline()
    with pytest.raises(ValueError):
        with timeline.span("boom"):
            raise ValueError("no")
    assert timeline.stages[0].error.startswith("ValueError")
    assert timeline.stages[0].duration >= 0


def test_remote_time_is_recorded_beside_wall_time_never_instead_of_it():
    timeline = Timeline()
    with timeline.span("synthesis", host="runpod-gpu") as stage:
        stage.remote_seconds = 0.0
    stage = timeline.stages[0]
    assert stage.host == "runpod-gpu"
    assert stage.remote_seconds == 0.0
    # Overhead is wall minus remote: the price of the stage being elsewhere.
    assert stage.overhead == pytest.approx(stage.duration, abs=1e-6)


def test_stages_from_several_hosts_share_one_clock():
    timeline = Timeline()
    for name, host in (("search", "exa-api"), ("generate", "anthropic-api"),
                       ("synthesis", "runpod-gpu")):
        with timeline.span(name, host=host):
            pass
    starts = [s.start for s in timeline.stages]
    assert starts == sorted(starts)
    assert {s.host for s in timeline.stages} == {"exa-api", "anthropic-api", "runpod-gpu"}


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
def test_first_chunk_by_sentence():
    assert first_chunk_ready("One two. Three", 0) == "One two."
    assert first_chunk_ready("One two", 0) is None


def test_first_chunk_by_word_budget():
    assert first_chunk_ready("a b c d e f", 3) == "a b c"
    assert first_chunk_ready("a b", 3) is None


def test_a_word_budget_is_a_minimum_not_a_maximum():
    """`first_chunk_words=N` means "at least N words, then the next sentence end".

    A sentence that ends before the budget is *not* enough: a chunk-size
    experiment that silently returned two words when asked for ten would be
    measuring something other than the size it claims.
    """
    assert first_chunk_ready("Short one. then more words here", 10) is None
    assert first_chunk_ready("Short one. then more words here now ok", 6) == "Short one."


# --------------------------------------------------------------------------
# Statistics refuse to overclaim
# --------------------------------------------------------------------------
def test_a_real_difference_is_detected():
    fast = [1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98, 1.08, 0.92, 1.0]
    slow = [2.0, 2.2, 1.9, 2.1, 2.05, 1.95, 2.15, 2.0, 2.1, 1.98]
    cmp = stats.bootstrap_diff(fast, slow, a_label="fast", b_label="slow")
    assert cmp.significant is True
    assert "fast is faster" in cmp.verdict


def test_noise_is_reported_as_noise_not_as_a_winner():
    a = [1.0, 1.4, 0.7, 1.2, 0.8, 1.3, 0.9, 1.1, 1.05, 0.95]
    b = [1.05, 1.35, 0.75, 1.15, 0.85, 1.25, 0.95, 1.05, 1.0, 1.0]
    cmp = stats.bootstrap_diff(a, b, a_label="a", b_label="b")
    assert cmp.significant is False
    assert "No detectable difference" in cmp.verdict
    assert cmp.required_n is None or cmp.required_n >= 3


def test_comparison_is_reproducible_from_the_same_data():
    a = [1.0, 1.4, 0.7, 1.2, 0.8]
    b = [2.0, 2.4, 1.7, 2.2, 1.8]
    first = stats.bootstrap_diff(a, b, seed=7)
    second = stats.bootstrap_diff(a, b, seed=7)
    assert first.ci_low == second.ci_low and first.ci_high == second.ci_high


def test_too_few_samples_gives_no_comparison_rather_than_a_bad_one():
    assert stats.bootstrap_diff([1.0], [2.0]) is None
    assert stats.summarise([]) is None


# --------------------------------------------------------------------------
# Spec validation
# --------------------------------------------------------------------------
def test_spec_rejects_concurrency_until_it_is_implemented():
    spec = ExperimentSpec(name="c", arms=[Arm("a")], queries=["q"], concurrency=4)
    assert any("Concurrency" in p for p in spec.validate())


def test_spec_rejects_a_comparison_too_small_to_support_one():
    spec = ExperimentSpec(name="c", arms=[Arm("a"), Arm("b")], queries=["q"], trials=2)
    assert any("cannot support a comparison" in p for p in spec.validate())


def test_spec_round_trips_through_json():
    spec = ExperimentSpec(name="rt", arms=[Arm("a", search="exa", params={"w": 8})],
                          queries=["q"], trials=4)
    assert ExperimentSpec.from_json(spec.to_json()).to_dict() == spec.to_dict()


def test_arms_report_how_many_dimensions_they_differ_in():
    a = Arm("a", search="exa", tts="chatterbox")
    b = Arm("b", search="anthropic_web_search", tts="piper")
    assert set(a.dimensions_vs(b)) == {"search", "tts"}


# --------------------------------------------------------------------------
# Plain English
# --------------------------------------------------------------------------
def test_the_headline_request_compiles_to_the_right_experiment():
    parsed = nl.compile_request(
        "Run the full Exa -> Claude -> Chatterbox pipeline 10 times and tell me "
        "where the latency bottleneck is")
    spec = parsed.spec
    assert spec.trials == 10
    assert spec.kind == "pipeline"
    names = {(a.search, a.tts) for a in spec.arms}
    assert ("exa", "chatterbox") in names
    # A pipeline timing with no control cannot say whether it is any good.
    assert ("anthropic_web_search", "piper") in names
    assert spec.validate() == []


def test_a_comparison_varies_one_dimension_and_holds_the_rest():
    spec = nl.compile_request("compare piper vs chatterbox, 10 trials").spec
    assert [a.tts for a in spec.arms] == ["piper", "chatterbox"]
    assert {a.search for a in spec.arms} == {"none"}


def test_every_assumption_is_declared():
    parsed = nl.compile_request("compare exa vs current search")
    assert any("trial" in a.lower() or "queries" in a.lower() for a in parsed.assumptions)


# --------------------------------------------------------------------------
# The harness end to end, with no key and no GPU
# --------------------------------------------------------------------------
def _fake_harness(spec, run=None, **kw):
    searches = {name: FakeSearch(0.0, adapter_id=name) for name in
                ("exa", "anthropic_web_search", "none")}
    searches["anthropic_web_search"].separable = False
    ttss = {name: FakeTTS(0.0, adapter_id=name) for name in ("piper", "chatterbox", "none")}
    return Harness(spec, run=run,
                   generator_factory=lambda: FakeGenerator(0.0, 0.0),
                   search_factory=lambda n: searches[n],
                   tts_factory=lambda n, v=None: ttss[n], **kw)


def test_a_full_pipeline_run_produces_every_checkpoint(tmp_path):
    spec = ExperimentSpec(name="e2e", trials=3, minutes=1, queries=["why do tides happen"],
                          arms=[Arm("a", search="exa", tts="chatterbox")])
    run = store.create(spec, root=tmp_path)
    results = asyncio.run(_fake_harness(spec, run).run_all())
    assert len(results) == 3
    assert all(r.ok for r in results)
    for key in ("first_token", "first_chunk", "first_audio"):
        assert all(r.metrics[key] is not None for r in results)
    assert all(r.metrics["first_token"] <= r.metrics["first_audio"] for r in results)


def test_trials_are_written_as_they_happen_not_at_the_end(tmp_path):
    """The run that dies at trial 17 of 20 must still leave 16 on disk."""
    spec = ExperimentSpec(name="crash", trials=4, minutes=1, queries=["q"],
                          arms=[Arm("a", search="none", tts="none")])
    run = store.create(spec, root=tmp_path)
    harness = _fake_harness(spec, run)
    seen = []

    def progress(result):
        seen.append(result)
        # On disk already, before the sweep is anywhere near finished.
        assert len(run.trials()) == len(seen)

    asyncio.run(harness.run_all(progress=progress))
    assert len(run.trials()) == 4


def test_a_failing_trial_is_recorded_and_excluded_not_silently_dropped(tmp_path):
    class Exploding:
        id = "boom"
        host = "local"

        def available(self):
            return Availability(ok=True)

        async def synth(self, text, timeline, **params):
            raise RuntimeError("the voice died")

    spec = ExperimentSpec(name="fail", trials=2, minutes=1, queries=["q"],
                          arms=[Arm("a", search="none", tts="piper")])
    run = store.create(spec, root=tmp_path)
    harness = _fake_harness(spec, run)
    harness.tts_factory = lambda n, v=None: Exploding()
    results = asyncio.run(harness.run_all())

    assert all(not r.ok for r in results)
    assert all("the voice died" in (r.error or "") for r in results)
    assert len(run.trials()) == 2
    analysis = report.analyse(spec, results)
    assert analysis["arms"][0]["failures"] == 2
    assert analysis["arms"][0]["headline"] is None
    assert "every trial failed" in report.render(spec, analysis).lower()


def test_infrastructure_required_stops_the_sweep_rather_than_being_swallowed(tmp_path):
    class NeedsGpu:
        id = "chatterbox"
        host = "runpod-gpu"

        def available(self):
            return Availability(ok=False, needs_approval=True, reason="no endpoint")

        async def synth(self, text, timeline, **params):
            raise InfrastructureRequired("chatterbox", "a GPU endpoint is required")

    spec = ExperimentSpec(name="gpu", trials=3, minutes=1, queries=["q"],
                          arms=[Arm("a", search="none", tts="chatterbox")])
    harness = _fake_harness(spec, store.create(spec, root=tmp_path))
    harness.tts_factory = lambda n, v=None: NeedsGpu()
    with pytest.raises(InfrastructureRequired):
        asyncio.run(harness.run_all())


def test_the_report_names_the_bottleneck_stage(tmp_path):
    spec = ExperimentSpec(name="bn", trials=3, minutes=1, queries=["q"],
                          arms=[Arm("slow_tts", search="none", tts="chatterbox")])
    searches = {"none": FakeSearch(0.0, adapter_id="none")}
    harness = Harness(spec, generator_factory=lambda: FakeGenerator(0.0, 0.0),
                      search_factory=lambda n: searches["none"],
                      tts_factory=lambda n, v=None: FakeTTS(0.05, adapter_id="chatterbox"))
    results = asyncio.run(harness.run_all())
    analysis = report.analyse(spec, results)
    arm = analysis["_arm_objects"][0]
    assert arm.bottleneck()[0] == "synthesis"
    assert "bottleneck" in report.render(spec, analysis).lower()


def test_a_simulated_run_is_labelled_and_recommends_nothing(tmp_path):
    spec = ExperimentSpec(name="sim", trials=3, minutes=1, queries=["q"],
                          arms=[Arm("a", search="exa", tts="chatterbox")])
    results = asyncio.run(_fake_harness(spec).run_all())
    analysis = report.analyse(spec, results)
    text = report.render(spec, analysis)
    assert "SIMULATED" in text
    assert "None — this run was simulated." in text
    assert "Actual, from recorded usage" not in text


def test_a_confounded_comparison_says_so(tmp_path):
    spec = ExperimentSpec(name="confound", trials=5, minutes=1, queries=["q"],
                          arms=[Arm("a", search="exa", tts="chatterbox"),
                                Arm("b", search="anthropic_web_search", tts="piper")])
    results = asyncio.run(_fake_harness(spec).run_all())
    analysis = report.analyse(spec, results)
    assert analysis["comparisons"]
    assert analysis["comparisons"][0]["dimensions"] == ["search", "tts"] or \
           set(analysis["comparisons"][0]["dimensions"]) == {"search", "tts"}


# --------------------------------------------------------------------------
# History
# --------------------------------------------------------------------------
def test_runs_are_never_overwritten(tmp_path):
    spec = ExperimentSpec(name="same name", arms=[Arm("a")], queries=["q"], trials=3)
    first = store.create(spec, root=tmp_path)
    second = store.create(spec, root=tmp_path)
    assert first.path != second.path
    assert len(store.list_runs(tmp_path)) == 2


def test_the_previous_run_of_the_same_name_is_found(tmp_path):
    spec = ExperimentSpec(name="repeatable", arms=[Arm("a")], queries=["q"], trials=3)
    old = store.create(spec, root=tmp_path)
    new = store.create(spec, root=tmp_path)
    found = store.find_previous("repeatable", exclude=new.id, root=tmp_path)
    assert found is not None and found.id == old.id


def test_a_truncated_trial_line_does_not_lose_the_rest(tmp_path):
    spec = ExperimentSpec(name="trunc", arms=[Arm("a")], queries=["q"], trials=3)
    run = store.create(spec, root=tmp_path)
    run.append_trial({"index": 1, "ok": True})
    with (run.path / "trials.jsonl").open("a") as handle:
        handle.write('{"index": 2, "ok": tr')
    trials = run.trials()
    assert len(trials) == 2
    assert trials[0]["index"] == 1
    assert "truncated" in trials[1]["error"]


def test_report_compares_against_the_previous_run():
    spec = ExperimentSpec(name="hist", trials=3, minutes=1, queries=["q"],
                          arms=[Arm("a", search="none", tts="none")])
    results = asyncio.run(_fake_harness(spec).run_all())
    analysis = report.analyse(spec, results)
    previous = {"arms": [{"name": "a", "headline": {"median": 9.99}}]}
    text = report.render(spec, analysis, previous=previous)
    assert "Against the previous run" in text
    assert "9.99s" in text


# --------------------------------------------------------------------------
# Cost is shown before it is spent
# --------------------------------------------------------------------------
def test_estimate_is_itemised_by_service():
    spec = ExperimentSpec(name="c", trials=10, minutes=3, queries=["q"],
                          arms=[Arm("a", search="exa", tts="chatterbox")])
    est = cost_mod.estimate(spec)
    assert est.anthropic > 0 and est.exa > 0 and est.gpu > 0
    assert est.total == pytest.approx(est.anthropic + est.exa + est.gpu)
    assert any("never creates" in n for n in est.notes)


def test_a_local_only_experiment_estimates_no_gpu_cost():
    spec = ExperimentSpec(name="c", trials=5, minutes=1, queries=["q"],
                          arms=[Arm("a", search="none", tts="piper")])
    assert cost_mod.estimate(spec).gpu == 0.0


def test_actual_cost_uses_real_token_counts():
    assert cost_mod.model_cost("claude-sonnet-5", 1_000_000, 0) == pytest.approx(2.0)
    assert cost_mod.model_cost("claude-sonnet-5", 0, 1_000_000) == pytest.approx(10.0)


# --------------------------------------------------------------------------
# The engine does not touch production
# --------------------------------------------------------------------------
def test_the_experiment_package_never_writes_to_the_script_cache():
    root = pathlib.Path(__file__).resolve().parent.parent / "experiments"
    for path in root.rglob("*.py"):
        text = path.read_text()
        assert ".put(" not in text, f"{path.name} appears to write to a cache"


def test_the_real_generator_disables_the_cache():
    source = (pathlib.Path(__file__).resolve().parent.parent / "experiments" / "generate.py").read_text()
    assert "cache_enabled=False" in source


def test_a_search_only_experiment_still_gets_a_headline():
    """Exa vs current search has no speech stage; it must still be comparable."""
    spec = ExperimentSpec(name="search only", trials=5, minutes=1, queries=["q"],
                          arms=[Arm("exa", search="exa", tts="none"),
                                Arm("baseline", search="anthropic_web_search", tts="none")])
    results = asyncio.run(_fake_harness(spec).run_all())
    analysis = report.analyse(spec, results)
    assert analysis["headline_metric"] == "first_chunk"
    assert all(a["headline"] is not None for a in analysis["arms"])
    text = report.render(spec, analysis)
    assert "Time to first speakable chunk" in text
    assert analysis["comparisons"], "two arms with data must produce a comparison"


def test_speech_runs_are_still_judged_on_first_audio():
    spec = ExperimentSpec(name="with speech", trials=3, minutes=1, queries=["q"],
                          arms=[Arm("a", search="none", tts="piper")])
    results = asyncio.run(_fake_harness(spec).run_all())
    analysis = report.analyse(spec, results)
    assert analysis["headline_metric"] == "first_audio"
    assert "Time to first audio" in report.render(spec, analysis)
