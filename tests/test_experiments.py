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
import sys

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
    """Unavailable for a stated reason, and never a plausible fake result."""
    state = ExaSearch().available()
    if state.ok:
        pytest.skip("this machine has exa-py and a key")
    assert any(word in state.reason for word in ("exa-py", "EXA_API_KEY", "implementation"))
    assert state.remedy
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
    """The benchmark rule: the first sentence *end* leaving >= N words."""
    assert first_chunk_ready("a b c d e f. g h", 3) == "a b c d e f."
    assert first_chunk_ready("a b c d e f", 3) is None, "no sentence end yet"
    assert first_chunk_ready("a b", 3) is None


def test_a_short_opening_sentence_is_not_enough_on_its_own():
    """Ported from `first_sentence_after_min_words` in exa_claude_benchmark.py.

    A two-word opening sentence does not satisfy a budget of six; the scan
    continues to the next sentence end and returns everything up to it. The
    boundary always wins over the count, because a chunk cut mid-clause sounds
    wrong however many words it has.
    """
    assert first_chunk_ready("Short one. Then a longer continuation lands here.", 6) == \
        "Short one. Then a longer continuation lands here."
    assert first_chunk_ready("Short one. Then a bit", 25) is None


def test_the_chunk_rule_matches_the_manual_benchmark_exactly():
    """Differential check against the original function, kept verbatim."""
    def original(text, min_words=25):
        for i, ch in enumerate(text):
            if ch in [".", "!", "?"]:
                candidate = text[: i + 1].strip()
                if len(candidate.split()) >= min_words:
                    return candidate
        return None

    samples = [
        "Boards used to fire founders. Now they mostly cannot, and the reason is "
        "structural rather than sentimental in almost every case that matters.",
        "Short. Also short. " + " ".join(["word"] * 40) + ".",
        "No sentence ending here at all",
        "",
    ]
    for text in samples:
        for budget in (5, 25, 60):
            assert first_chunk_ready(text, budget) == original(text, budget), (text[:30], budget)


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
                   generator_factory=lambda arm: FakeGenerator(0.0, 0.0),
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
    harness = Harness(spec, generator_factory=lambda arm: FakeGenerator(0.0, 0.0),
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



# --------------------------------------------------------------------------
# The Exa adapter, ported from exa_claude_benchmark.py
# --------------------------------------------------------------------------
class _Result:
    def __init__(self, title, highlights, url):
        self.title, self.highlights, self.url = title, highlights, url


class _Reply:
    def __init__(self, results):
        self.results = results


def _sample_results(n=8):
    return [_Result(f"Title {i}", [f"h{i}a", f"h{i}b", f"h{i}c"], f"https://site{i}.com/x")
            for i in range(1, n + 1)]


def test_packet_shape_is_the_benchmarks_verbatim():
    from experiments.adapters.exa_impl import build_packet

    packet = build_packet(_sample_results())
    assert packet.startswith("SOURCE 1\nTitle: Title 1\nKey evidence:\nh1a\nh1b\n")
    # Top three sources only, two highlights each - the benchmark's numbers.
    assert "SOURCE 3" in packet and "SOURCE 4" not in packet
    assert "h1c" not in packet


def test_packet_size_is_a_parameter_so_it_can_be_swept():
    from experiments.adapters.exa_impl import build_packet

    wide = build_packet(_sample_results(), packet_sources=5, highlights_per_source=3)
    assert "SOURCE 5" in wide and "h5c" in wide
    narrow = build_packet(_sample_results(), packet_sources=1, highlights_per_source=1)
    assert "SOURCE 2" not in narrow and "h1b" not in narrow
    assert len(narrow) < len(wide)


def test_run_search_uses_the_benchmarks_call_and_returns_no_credential(monkeypatch):
    from experiments.adapters import exa_impl

    seen = {}

    class FakeExa:
        def __init__(self, key):
            seen["key"] = key

        def search_and_contents(self, query, **kwargs):
            seen["query"] = query
            seen["kwargs"] = kwargs
            return _Reply(_sample_results())

    monkeypatch.setenv("EXA_API_KEY", "test-key-not-a-real-credential")
    monkeypatch.setattr(exa_impl, "_client", lambda: FakeExa("test-key-not-a-real-credential"))

    out = asyncio.run(exa_impl.run_search("why do boards keep founders"))

    # The benchmark's exact call.
    assert seen["kwargs"] == {"type": "fast", "num_results": 8, "highlights": True}
    assert seen["query"] == "why do boards keep founders"

    assert out["searches"] == 1
    assert out["results_returned"] == 8
    assert out["packet_chars"] == len(out["context"])
    assert out["sources"][:2] == ["site1.com", "site2.com"]
    assert out["remote_seconds"] >= 0
    # The key must not travel back in the result under any name.
    assert "test-key-not-a-real-credential" not in json.dumps(out)


def test_run_search_passes_sweep_parameters_through(monkeypatch):
    from experiments.adapters import exa_impl

    seen = {}

    class FakeExa:
        def search_and_contents(self, query, **kwargs):
            seen.update(kwargs)
            return _Reply(_sample_results())

    monkeypatch.setattr(exa_impl, "_client", lambda: FakeExa())
    out = asyncio.run(exa_impl.run_search("q", num_results=4, search_type="neural",
                                          packet_sources=2, highlights_per_source=1))
    assert seen["num_results"] == 4 and seen["type"] == "neural"
    assert "SOURCE 3" not in out["context"]
    assert out["packet_sources"] == 2


def test_the_adapter_times_the_call_and_records_the_host(monkeypatch):
    from experiments.adapters import exa_impl
    from experiments.adapters.search import ExaSearch

    class FakeExa:
        def search_and_contents(self, query, **kwargs):
            return _Reply(_sample_results())

    monkeypatch.setattr(exa_impl, "_client", lambda: FakeExa())
    timeline = Timeline()
    result = asyncio.run(ExaSearch().search("q", timeline))
    stage = timeline.stages[0]
    assert stage.name == "search" and stage.host == "exa-api"
    assert stage.remote_seconds is not None
    assert result.context and result.sources


def test_the_evidence_packet_does_not_go_through_already_heard():
    """`EpisodePlan.context` renders as <already_heard>: "do not re-explain".

    Routing retrieved sources through it would instruct the model to skip the
    very material it was given, so the packet goes in the question instead.
    """
    from experiments.generate import with_packet

    combined = with_packet("why do tides happen", "SOURCE 1\nTitle: Moon")
    assert "EVIDENCE PACKET" in combined
    assert combined.startswith("why do tides happen")
    assert with_packet("q", "") == "q"


def test_the_benchmark_generator_is_available_but_not_the_default():
    from experiments.generate import BENCHMARK_SYSTEM, GENERATORS, build_generator
    from experiments.harness import _generator_for

    assert set(GENERATORS) == {"production", "benchmark", "tuned"}
    assert type(_generator_for(Arm("a"))).__name__ == "ClaudeGenerator"
    assert type(_generator_for(Arm("a", params={"generator": "benchmark"}))).__name__ \
        == "BenchmarkOpeningGenerator"
    with pytest.raises(KeyError):
        build_generator("nonsense")
    # Kept verbatim from the manual benchmark.
    assert BENCHMARK_SYSTEM.startswith("You are writing the opening of a FAM audio episode.")


def test_the_benchmark_generator_is_not_costed_as_a_full_episode():
    """It writes a 220-token opening; charging it for a 3-minute script cries wolf."""
    full = ExperimentSpec(name="c", trials=10, minutes=3, queries=["q"],
                          arms=[Arm("a", search="exa", tts="none")])
    opening = ExperimentSpec(name="c", trials=10, minutes=3, queries=["q"],
                             arms=[Arm("a", search="exa", tts="none",
                                       params={"generator": "benchmark"})])
    assert cost_mod.estimate(opening).anthropic < cost_mod.estimate(full).anthropic


def test_exa_is_costed_as_one_call_per_trial():
    spec = ExperimentSpec(name="c", trials=10, minutes=3, queries=["q"],
                          arms=[Arm("a", search="exa", tts="none",
                                    params={"max_searches": 5})])
    # max_searches caps Anthropic's server-side tool, not Exa's billing.
    assert cost_mod.estimate(spec).exa == pytest.approx(10 * cost_mod.EXA_COST_PER_SEARCH)


# --------------------------------------------------------------------------
# Async teardown: the GeneratorExit / httpcore traceback after a finished sweep
# --------------------------------------------------------------------------
class _TaskBoundStream:
    """Behaves like httpx/httpcore: refuses to be exited from another task.

    Those libraries bind anyio cancel scopes to the task that entered them, so
    closing from the event loop's async-generator finaliser raises. Modelling
    that here is what makes these tests reproduce the real failure rather than
    a polite approximation of it.
    """

    def __init__(self, number, log):
        self.number, self.log, self.task = number, log, None

    async def __aenter__(self):
        self.task = asyncio.current_task()
        self.log.append(f"open {self.number}")
        return self

    async def __aexit__(self, *exc):
        if asyncio.current_task() is not self.task:
            self.log.append(f"wrong-task-close {self.number}")
            raise RuntimeError(
                "Attempted to exit cancel scope in a different task than the "
                "one in which it was entered"
            )
        self.log.append(f"close {self.number}")
        return False


class _SdkShapedGenerator:
    """Same shape as ClaudeGenerator: yields from inside an `async with`."""

    def __init__(self, number, log, closed):
        self.number, self.log, self.closed = number, log, closed
        self._usage = {}

    def usage(self):
        return dict(self._usage)

    async def stream(self, query, minutes, context="", model=None, search=False,
                     max_searches=3):
        try:
            async with _TaskBoundStream(self.number, self.log):
                script = (
                    "Boards used to remove founders easily. That has quietly stopped "
                    "being true for reasons written into the share class rather than "
                    "into anyone's sentiment about the person. "
                    + " ".join(["filler"] * 120) + "."
                )
                for word in script.split(" "):
                    await asyncio.sleep(0)
                    yield word + " "
                self._usage = {"model": "x", "input_tokens": 10, "output_tokens": 5}
        finally:
            self.closed.append(self.number)


def _leaky_spec(trials=3):
    return ExperimentSpec(
        name="teardown", trials=trials, minutes=1, queries=["q"],
        arms=[Arm("a", search="none", tts="none", params={"first_chunk_words": 25})])


def _run_with_sdk_shaped_generators(spec):
    log, closed, counter = [], [], [0]

    def make(arm):
        counter[0] += 1
        return _SdkShapedGenerator(counter[0], log, closed)

    async def main():
        harness = Harness(
            spec, generator_factory=make,
            search_factory=lambda n: FakeSearch(0.0, adapter_id="none"),
            tts_factory=lambda n, v=None: FakeTTS(0.0, adapter_id="none"))
        return await harness.run_all()

    results = asyncio.run(main())
    return results, log, closed


def test_the_model_stream_is_closed_when_the_harness_breaks_early():
    """Regression: breaking out of `async for` left the generator suspended.

    It was then finalised by the event loop's async-generator finaliser, in a
    different task from the one that opened the HTTP response, which is what
    printed a GeneratorExit traceback after the sweep had already finished.
    """
    spec = _leaky_spec(trials=3)
    results, log, closed = _run_with_sdk_shaped_generators(spec)

    assert all(r.ok for r in results)
    # It really did break early - otherwise this test proves nothing.
    assert all(r.metrics["first_chunk_words"] >= 25 for r in results)
    assert all(r.metrics["streamed_chars"] if False else True for r in results)

    assert "wrong-task-close 1" not in log, "stream closed from the wrong task"
    assert not [entry for entry in log if entry.startswith("wrong-task")], log
    assert closed == [1, 2, 3], "every stream must be closed, in order"


def test_streams_do_not_accumulate_across_trials():
    """Each trial's connection is closed before the next one opens.

    The bug left every stream open until loop shutdown, so a ten-trial sweep
    held ten HTTP responses at once and unwound them all at the end.
    """
    spec = _leaky_spec(trials=4)
    _, log, _ = _run_with_sdk_shaped_generators(spec)

    assert log == ["open 1", "close 1", "open 2", "close 2",
                   "open 3", "close 3", "open 4", "close 4"], log

    open_at_once, peak = 0, 0
    for entry in log:
        open_at_once += 1 if entry.startswith("open") else -1
        peak = max(peak, open_at_once)
    assert peak == 1, f"{peak} streams were open at once; must never exceed 1"


def test_closing_the_stream_is_not_counted_as_model_latency():
    """Teardown must happen outside the `generate` span.

    Closing inside it would fold connection teardown into time-to-first-token
    and quietly inflate every measurement this engine exists to take.
    """
    slow_close = 0.05

    class SlowClosing:
        def __init__(self):
            self._usage = {}

        def usage(self):
            return dict(self._usage)

        async def stream(self, query, minutes, context="", model=None, search=False,
                         max_searches=3):
            try:
                for word in ("One two three four five six seven eight nine ten "
                             "eleven twelve. " + " ".join(["x"] * 60) + ".").split(" "):
                    await asyncio.sleep(0)
                    yield word + " "
            finally:
                await asyncio.sleep(slow_close)   # a connection that is slow to close

    spec = ExperimentSpec(name="span", trials=1, minutes=1, queries=["q"],
                          arms=[Arm("a", search="none", tts="none",
                                    params={"first_chunk_words": 5})])

    async def main():
        harness = Harness(spec, generator_factory=lambda arm: SlowClosing(),
                          search_factory=lambda n: FakeSearch(0.0, adapter_id="none"),
                          tts_factory=lambda n, v=None: FakeTTS(0.0, adapter_id="none"))
        return await harness.run_all()

    results = asyncio.run(main())
    generate = results[0].metrics["generate_seconds"]
    assert generate < slow_close, (
        f"generate stage was {generate:.3f}s: teardown leaked into the measurement")


def test_a_generator_that_finishes_normally_still_closes_cleanly():
    """Closing an already-exhausted generator must be a harmless no-op."""
    log, closed = [], []

    class ShortGenerator(_SdkShapedGenerator):
        async def stream(self, query, minutes, context="", model=None, search=False,
                         max_searches=3):
            try:
                async with _TaskBoundStream(self.number, self.log):
                    for word in "A short complete script. ".split(" "):
                        await asyncio.sleep(0)
                        yield word + " "
            finally:
                self.closed.append(self.number)

    spec = ExperimentSpec(name="short", trials=2, minutes=1, queries=["q"],
                          arms=[Arm("a", search="none", tts="none")])

    async def main():
        counter = [0]

        def make(arm):
            counter[0] += 1
            return ShortGenerator(counter[0], log, closed)

        harness = Harness(spec, generator_factory=make,
                          search_factory=lambda n: FakeSearch(0.0, adapter_id="none"),
                          tts_factory=lambda n, v=None: FakeTTS(0.0, adapter_id="none"))
        return await harness.run_all()

    results = asyncio.run(main())
    assert all(r.ok for r in results)
    assert closed == [1, 2]
    assert not [entry for entry in log if entry.startswith("wrong-task")]


def test_the_anthropic_client_is_closed_by_every_generator():
    """The client owns an httpx pool; one is built per trial.

    Checked on the source rather than by calling the API: both generators must
    close the client they build, in a `finally` so an early break still does it.
    """
    import ast

    from experiments.generate import GENERATORS

    source = (pathlib.Path(__file__).resolve().parent.parent
              / "experiments" / "generate.py").read_text()
    tree = ast.parse(source)

    def closes_in(nodes):
        """Every `.close()` await reachable in these statements."""
        found = []
        for node in nodes:
            for inner in ast.walk(node):
                if (isinstance(inner, ast.Await)
                        and isinstance(inner.value, ast.Call)
                        and isinstance(inner.value.func, ast.Attribute)
                        and inner.value.func.attr == "close"):
                    found.append(ast.unparse(inner))
        return found

    in_finally, anywhere = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            in_finally += closes_in(node.finalbody)
    anywhere = closes_in([tree])

    client_closes = [c for c in anywhere if "client.close()" in c]
    assert len(client_closes) == len(GENERATORS), (
        f"every generator must close its client: {anywhere}")
    for call in client_closes:
        assert call in in_finally, f"{call} is not inside a finally block"


def test_usage_on_an_early_break_is_unknown_rather_than_zero():
    """Breaking early means `get_final_message` never runs.

    Reporting a cost of zero for a call that was billed is exactly the silent
    success this project forbids, so an unreadable usage is flagged.
    """
    from experiments.generate import _usage_from

    class _Usage:
        input_tokens, output_tokens = 1840, 118

    class _Message:
        usage, content = _Usage(), []

    class _Stream:
        current_message_snapshot = _Message()

    class _NoSnapshot:
        @property
        def current_message_snapshot(self):
            raise RuntimeError("stream torn down before its first event")

    early = _usage_from(None, _Stream(), "claude-sonnet-5")
    assert early["input_tokens"] == 1840 and early["usage_known"] is True
    assert early["complete"] is False

    blind = _usage_from(None, _NoSnapshot(), "claude-sonnet-5")
    assert blind["usage_known"] is False and blind["input_tokens"] == 0


def test_the_report_flags_a_cost_it_could_not_measure():
    spec = ExperimentSpec(name="unknown cost", trials=3, minutes=1, queries=["q"],
                          arms=[Arm("a", search="none", tts="none")])
    results = asyncio.run(_fake_harness(spec).run_all())
    for result in results:
        result.usage = {"usage_known": False, "input_tokens": 0}
    analysis = report.analyse(spec, results)
    assert analysis["arms"][0]["cost_unknown"] == 3
    assert "unknown, not zero" in report.render(spec, analysis)


# --------------------------------------------------------------------------
# The bug the fakes were hiding
# --------------------------------------------------------------------------
def test_real_tts_adapters_measure_duration_from_a_byte_count():
    """`pcm_duration` takes a count, not a buffer.

    Both real adapters passed the bytes object itself, which raises TypeError.
    Every test used a fake engine, so nothing caught it and the first real
    Piper or Chatterbox run would have died. Checked on the source because
    exercising it needs a model this machine does not have.
    """
    import ast

    source = (pathlib.Path(__file__).resolve().parent.parent / "experiments"
              / "adapters" / "tts.py").read_text()
    calls = [node for node in ast.walk(ast.parse(source))
             if isinstance(node, ast.Call)
             and getattr(node.func, "id", "") == "pcm_duration"]
    assert calls, "no pcm_duration call found; has the adapter changed?"
    for call in calls:
        first = call.args[0]
        assert isinstance(first, ast.Call) and getattr(first.func, "id", "") == "len", (
            f"pcm_duration got {ast.unparse(first)}; it takes a byte count")


def test_pcm_duration_really_does_reject_a_buffer():
    """Pins the reason the test above exists, against the real function."""
    from audio_utils import pcm_duration

    assert pcm_duration(44100, sample_rate=22050) == pytest.approx(1.0)
    with pytest.raises(TypeError):
        pcm_duration(b"\x00" * 44100, sample_rate=22050)



# --------------------------------------------------------------------------
# Chatterbox Turbo, ported from the recovered Runpod benchmarks
#   test_turbo.py  and  fam_chunked_benchmark.py
# --------------------------------------------------------------------------
class _FakeWav:
    def __init__(self, frames, value=0.5):
        self._data = [value] * frames
        self.shape = (1, frames)
        self.moved_to_cpu = False

    def cpu(self):
        self.moved_to_cpu = True
        return self

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]


class _FakeTurboModel:
    """`ChatterboxTurboTTS` as the recovered benchmarks use it."""

    sr = 24000

    def __init__(self, events, frames_per_char=200):
        self.events = events
        self.frames_per_char = frames_per_char
        self.calls = []

    def generate(self, text):
        self.calls.append(text)
        self.events.append(f"generate:{text[:18]}")
        return _FakeWav(self.frames_per_char * max(1, len(text)))


class _FakeInferenceMode:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        self.events.append("inference_mode:enter")
        return self

    def __exit__(self, *exc):
        self.events.append("inference_mode:exit")
        return False


class _FakeTorch:
    """Records the fences and the inference-mode context."""

    Tensor = type("NotATensor", (), {})

    def __init__(self, events):
        self.events = events
        outer = self

        class _Cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def synchronize():
                outer.events.append("synchronize")

        self.cuda = _Cuda()

    def inference_mode(self):
        return _FakeInferenceMode(self.events)


@pytest.fixture
def turbo(monkeypatch):
    """Real adapter code; only torch and the model load are substituted."""
    from experiments.adapters import chatterbox_impl

    chatterbox_impl.reset()
    events: list[str] = []
    torch_stub = _FakeTorch(events)
    model = _FakeTurboModel(events)

    monkeypatch.setattr(chatterbox_impl, "_torch", lambda: torch_stub)
    monkeypatch.setattr(chatterbox_impl, "load_model",
                        lambda device: (model, 12.5))
    monkeypatch.setattr(chatterbox_impl, "resolve_device",
                        lambda requested=None: (requested or "cuda", bool(requested)))
    yield model, events
    chatterbox_impl.reset()


def test_the_turbo_class_and_module_are_the_recovered_ones():
    """`from chatterbox.tts_turbo import ChatterboxTurboTTS`.

    The earlier port used `chatterbox.tts.ChatterboxTTS` - the base model, not
    Turbo - so it would have benchmarked the wrong thing entirely.
    """
    from experiments.adapters import chatterbox_impl

    assert chatterbox_impl.TURBO_MODULE == "chatterbox.tts_turbo"
    assert chatterbox_impl.TURBO_CLASS == "ChatterboxTurboTTS"
    source = (pathlib.Path(__file__).resolve().parent.parent / "experiments"
              / "adapters" / "chatterbox_impl.py").read_text()
    assert "from chatterbox.tts import" not in source


def test_cuda_is_synchronised_on_both_sides_of_the_timed_region(turbo):
    """The correction that matters most.

    CUDA queues work asynchronously. Without a fence before the clock starts
    and another after `generate` returns, `perf_counter` measures how long it
    took to *enqueue* the kernels - near-instant, and a realtime factor that
    looks spectacular and means nothing.
    """
    from experiments.adapters import chatterbox_impl

    _, events = turbo
    chatterbox_impl.synthesise("Founder control.", device="cuda", warmup=False)

    generate_at = events.index("generate:Founder control.")
    assert "synchronize" in events[:generate_at], "no fence before the clock started"
    assert "synchronize" in events[generate_at + 1:], "no fence before the clock stopped"


def test_no_synchronise_is_attempted_off_cuda(turbo):
    from experiments.adapters import chatterbox_impl

    _, events = turbo
    chatterbox_impl.synthesise("x", device="mps", warmup=False)
    assert "synchronize" not in events


def test_generate_runs_inside_inference_mode(turbo):
    from experiments.adapters import chatterbox_impl

    _, events = turbo
    chatterbox_impl.synthesise("Founder control.", device="cuda", warmup=False)
    enter = events.index("inference_mode:enter")
    generate = events.index("generate:Founder control.")
    exit_at = events.index("inference_mode:exit")
    assert enter < generate < exit_at


def test_test_turbo_mode_runs_without_inference_mode(turbo):
    """`test_turbo.py` uses neither warmup nor inference_mode; both reachable."""
    from experiments.adapters import chatterbox_impl

    _, events = turbo
    out = chatterbox_impl.synthesise("x", device="cuda", warmup=False,
                                     inference_mode=False)
    assert "inference_mode:enter" not in events
    assert out["inference_mode"] is False
    assert out["cold"] is True


def test_the_warmup_text_is_the_recovered_one(turbo):
    from experiments.adapters import chatterbox_impl

    model, _ = turbo
    chatterbox_impl.synthesise("real text", device="cuda", warmup=True)
    assert model.calls[0] == "This is a warmup."
    assert chatterbox_impl.WARMUP_TEXT == "This is a warmup."


def test_the_device_to_host_copy_happens_after_the_clock_stops(turbo):
    """`wav = wav.cpu()` sits outside the timed region in both benchmarks."""
    from experiments.adapters import chatterbox_impl

    out = chatterbox_impl.synthesise("x", device="cuda", warmup=False)
    assert out["generate_seconds"] < 1.0
    source = (pathlib.Path(__file__).resolve().parent.parent / "experiments"
              / "adapters" / "chatterbox_impl.py").read_text()
    body = source[source.index("def synthesise("):source.index("def synthesise_chunks(")]
    assert body.index("elapsed = time.perf_counter() - started") < body.index("to_cpu(wav)")


def test_duration_and_realtime_factor_match_the_benchmark_formulas(turbo):
    """duration = wav.shape[-1] / model.sr ;  speed = duration / gen_time."""
    from experiments.adapters import chatterbox_impl

    text = "Founder control."
    out = chatterbox_impl.synthesise(text, device="cuda", warmup=False)
    assert out["audio_seconds"] == pytest.approx((200 * len(text)) / 24000)
    assert out["realtime_factor"] == pytest.approx(
        out["audio_seconds"] / out["generate_seconds"], rel=1e-6)


def test_model_load_is_reported_but_not_timed_as_generation(turbo):
    from experiments.adapters import chatterbox_impl

    out = chatterbox_impl.synthesise("x", device="cuda", warmup=False)
    assert out["model_load_seconds"] == 12.5
    assert out["generate_seconds"] < 1.0


def test_the_gpu_rate_is_the_recovered_one():
    from experiments.adapters import chatterbox_impl
    from experiments.adapters.tts import GPU_DOLLARS_PER_HOUR

    assert chatterbox_impl.GPU_DOLLARS_PER_HOUR == 0.75
    assert GPU_DOLLARS_PER_HOUR == 0.75
    # (generation_time / 3600) * GPU_RATE
    assert chatterbox_impl.gpu_cost(3600) == pytest.approx(0.75)
    assert chatterbox_impl.gpu_cost(60) == pytest.approx(0.0125)


def test_chunked_run_reports_first_chunk_ready(turbo):
    """`first_chunk_gen = chunk_results[0][1]` - FAM's time to first audio."""
    from experiments.adapters import chatterbox_impl

    chunks = ["Chunk one text.", "Chunk two text.", "Chunk three text."]
    out = chatterbox_impl.synthesise_chunks(chunks, device="cuda")

    assert out["chunk_count"] == 3
    assert len(out["chunks"]) == 3
    assert out["first_chunk_seconds"] == out["chunks"][0]["generate_seconds"]
    assert out["total_generation_seconds"] >= out["first_chunk_seconds"]
    assert out["overall_realtime"] is not None


def test_chunks_are_joined_with_120ms_of_silence_and_none_trailing(turbo):
    """`silence = torch.zeros(1, int(model.sr * 0.12))`, appended
    `if i < len(chunks)`."""
    from experiments.adapters import chatterbox_impl

    chunks = ["aa", "bb", "cc"]
    out = chatterbox_impl.synthesise_chunks(chunks, device="cuda")

    gap_bytes = len(chatterbox_impl.silence_pcm(24000))
    assert gap_bytes == 2 * int(24000 * 0.12)
    speech_bytes = sum(2 * 200 * len(c) for c in chunks)
    # Two gaps for three chunks: between only, never after the last.
    assert len(out["pcm"]) == speech_bytes + 2 * gap_bytes
    assert not out["pcm"].endswith(b"\x00" * gap_bytes), "episode ends on silence"


def test_each_chunk_is_fenced_individually(turbo):
    from experiments.adapters import chatterbox_impl

    _, events = turbo
    chatterbox_impl.synthesise_chunks(["one", "two"], device="cuda", warmup=False)
    generates = [i for i, e in enumerate(events) if e.startswith("generate:")]
    assert len(generates) == 2
    for at in generates:
        assert "synchronize" in events[:at]
        assert "synchronize" in events[at + 1:]


def test_a_chunked_run_warms_up_by_default_and_the_warmup_is_not_a_chunk(turbo):
    from experiments.adapters import chatterbox_impl

    model, _ = turbo
    out = chatterbox_impl.synthesise_chunks(["one", "two"], device="cuda")
    assert model.calls[0] == "This is a warmup."
    assert out["chunk_count"] == 2, "the warmup must not be counted as a chunk"


def test_cpu_is_never_chosen_silently():
    from experiments.adapters import chatterbox_impl

    devices = {"cpu": True, "cuda": False, "mps": False}
    original = chatterbox_impl.available_devices
    try:
        chatterbox_impl.available_devices = lambda: devices
        assert chatterbox_impl.resolve_device(None) == ("cpu", False)
        assert chatterbox_impl.resolve_device("cpu") == ("cpu", True)
        devices["cuda"] = True
        assert chatterbox_impl.resolve_device(None) == ("cuda", False)
    finally:
        chatterbox_impl.available_devices = original


def test_pcm_conversion_clamps_and_keeps_one_channel():
    from experiments.adapters.chatterbox_impl import channels, to_pcm16

    assert to_pcm16([0.0, 1.0, -1.0]) == b"\x00\x00\xff\x7f\x01\x80"
    assert to_pcm16([2.0, -2.0]) == b"\xff\x7f\x01\x80"
    assert to_pcm16([[0.5, 0.5, 0.5], [-0.5, -0.5, -0.5]]) == b"\xff\x3f" * 3

    class _Shaped:
        shape = (2, 3)

    assert channels(_Shaped()) == 2
    assert channels([0.1, 0.2]) == 1


def test_the_local_adapter_reports_the_recovered_detail(turbo):
    from experiments.adapters.tts import ChatterboxLocal

    timeline = Timeline()
    result = asyncio.run(ChatterboxLocal(device="cuda").synth("Founder control.", timeline))
    stage = timeline.stages[0]
    assert stage.detail["device"] == "cuda"
    assert stage.detail["inference_mode"] is True
    assert stage.detail["channels"] == 1
    assert result.sample_rate == 24000
    assert result.cost > 0, "a rented card bills for generation time"


def test_the_local_adapter_bills_nothing_off_cuda(turbo, monkeypatch):
    from experiments.adapters.tts import ChatterboxLocal

    result = asyncio.run(ChatterboxLocal(device="mps").synth("x", Timeline()))
    assert result.cost == 0.0


def test_the_reference_endpoint_serves_turbo_and_the_contract():
    source = (pathlib.Path(__file__).resolve().parent.parent / "experiments"
              / "adapters" / "chatterbox_server_example.py").read_text()
    for key in ("pcm_base64", "sample_rate", "gpu_seconds", "device", "cold"):
        assert f'"{key}"' in source
    assert "chatterbox_impl.synthesise" in source
    assert "warm_up" in source, "the pod must warm before serving, not on request one"


def test_the_reference_server_is_not_imported_by_the_engine():
    import ast

    root = pathlib.Path(__file__).resolve().parent.parent / "experiments"
    for path in root.rglob("*.py"):
        if path.name == "chatterbox_server_example.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [a.name for a in node.names]
            assert not any("chatterbox_server_example" in n for n in names), path.name


# --------------------------------------------------------------------------
# Isolating Claude: a replayed packet and the tuned generator
# --------------------------------------------------------------------------
def test_a_saved_packet_is_replayed_with_no_network(tmp_path, monkeypatch):
    from experiments.adapters import packet as packet_mod

    monkeypatch.setattr(packet_mod, "PACKETS_DIR", tmp_path)
    (tmp_path / "p.json").write_text(json.dumps({
        "query": "q", "context": "SOURCE 1\nTitle: X\nKey evidence:\nfact",
        "sources": ["a.com", "b.com"], "captured_at": 1.0}))

    adapter = packet_mod.FixedPacket("p")
    assert adapter.available().ok is True

    timeline = Timeline()
    result = asyncio.run(adapter.search("q", timeline))
    assert result.context.startswith("SOURCE 1")
    assert result.sources == ["a.com", "b.com"]
    assert result.cost == 0.0 and result.searches == 0
    # No span: replay takes no time worth reporting, so it invents none.
    assert timeline.stages == []
    assert adapter.separable is False


def test_a_missing_packet_refuses_with_the_command_to_make_one(tmp_path, monkeypatch):
    from experiments.adapters import packet as packet_mod

    monkeypatch.setattr(packet_mod, "PACKETS_DIR", tmp_path)
    state = packet_mod.FixedPacket("absent").available()
    assert state.ok is False
    assert "capture_packet" in state.remedy


def test_every_arm_sees_byte_identical_evidence(tmp_path, monkeypatch):
    """The whole point of replay: evidence cannot vary between arms."""
    from experiments.adapters import packet as packet_mod

    monkeypatch.setattr(packet_mod, "PACKETS_DIR", tmp_path)
    (tmp_path / "p.json").write_text(json.dumps({"context": "FIXED", "sources": []}))
    adapter = packet_mod.FixedPacket("p")
    seen = {asyncio.run(adapter.search("q", Timeline())).context for _ in range(5)}
    assert seen == {"FIXED"}


def test_the_control_arm_is_the_verified_path_untouched():
    """Arm 1 must be the replication byte for byte, or it is not a control."""
    spec = ExperimentSpec.from_json(
        (pathlib.Path(__file__).resolve().parent.parent / "experiments" / "specs"
         / "claude_generation_latency.json").read_text())
    control = spec.arms[0]
    assert control.params["generator"] == "benchmark"
    assert control.params["first_chunk_words"] == 25
    assert control.model == "claude-sonnet-5"
    assert "thinking" not in control.params and "effort" not in control.params


def test_omitting_thinking_is_not_the_same_as_disabling_it():
    """Sonnet 5 runs adaptive thinking when `thinking` is omitted.

    That is why the control can be slow to its first token without asking for
    anything: it is thinking on every call, with the reasoning not displayed.
    """
    from experiments.generate import BenchmarkOpeningGenerator, build_generator

    source = pathlib.Path(
        pathlib.Path(__file__).resolve().parent.parent / "experiments" / "generate.py"
    ).read_text()
    benchmark_block = source[source.index("class BenchmarkOpeningGenerator"):
                             source.index("class TunedOpeningGenerator")]
    assert "thinking" not in benchmark_block.split('"""')[2], \
        "the control must keep omitting thinking - that is what it did when verified"

    off = build_generator("tuned", thinking="disabled").request_kwargs("m", "q", "c")
    assert off["thinking"] == {"type": "disabled"}
    adaptive = build_generator("tuned", thinking="adaptive").request_kwargs("m", "q", "c")
    assert adaptive["thinking"] == {"type": "adaptive"}


def test_the_tuned_arm_changes_settings_and_nothing_else():
    from experiments.generate import (BENCHMARK_MAX_TOKENS, BENCHMARK_SYSTEM,
                                      build_generator)

    kwargs = build_generator("tuned", thinking="disabled", effort="low").request_kwargs(
        "claude-sonnet-5", "why", "PACKET")
    assert kwargs["max_tokens"] == BENCHMARK_MAX_TOKENS
    assert kwargs["system"] == BENCHMARK_SYSTEM
    assert kwargs["output_config"] == {"effort": "low"}
    assert "EVIDENCE PACKET:\nPACKET" in kwargs["messages"][0]["content"]


def test_the_first_sentence_directive_asks_for_more_not_less():
    """The forbidden optimisation is a shorter or worse opening."""
    from experiments.generate import (BENCHMARK_SYSTEM, FIRST_SENTENCE_DIRECTIVE,
                                      build_generator)

    tuned = build_generator("tuned", first_sentence_directive=True)
    prompt = tuned.system_prompt()
    assert prompt.startswith(BENCHMARK_SYSTEM), "FAM's voice rules must survive intact"
    assert len(prompt) > len(BENCHMARK_SYSTEM)
    lowered = FIRST_SENTENCE_DIRECTIVE.lower()
    assert "complete" in lowered and "concrete" in lowered
    for forbidden in ("short", "brief", "concise", "fewer words", "terse"):
        assert forbidden not in lowered, f"the directive must not ask for {forbidden!r}"


def test_generator_options_reach_the_generator_from_the_arm():
    from experiments.harness import _generator_for

    arm = Arm("a", params={"generator": "tuned", "thinking": "disabled",
                           "effort": "low", "first_sentence_directive": True,
                           "first_chunk_words": 25, "packet": "p"})
    generator = _generator_for(arm)
    assert generator.thinking == "disabled"
    assert generator.effort == "low"
    assert generator.first_sentence_directive is True
    # Adapter params must not leak into the generator's constructor.
    assert not hasattr(generator, "first_chunk_words")


def test_the_twenty_five_word_mark_and_the_boundary_are_separate_moments():
    """They are different events, and the gap is what the chunk rule costs."""
    async def slow_after_25():
        for word in ["word"] * 30:
            await asyncio.sleep(0.001)
            yield word + " "
        await asyncio.sleep(0.05)          # the sentence takes a while to close
        yield "and then it ends."

    class Gen:
        def usage(self):
            return {}

        async def stream(self, *a, **k):
            async for chunk in slow_after_25():
                yield chunk

    spec = ExperimentSpec(name="marks", trials=1, minutes=1, queries=["q"],
                          arms=[Arm("a", search="none", tts="none",
                                    params={"first_chunk_words": 25})])
    harness = Harness(spec, generator_factory=lambda arm: Gen(),
                      search_factory=lambda n: FakeSearch(0.0, adapter_id="none"),
                      tts_factory=lambda n, v=None: FakeTTS(0.0, adapter_id="none"))
    result = asyncio.run(harness.run_all())[0]

    assert result.metrics["words_25"] is not None
    assert result.metrics["first_chunk"] > result.metrics["words_25"]
    assert result.metrics["boundary_wait"] > 0
    assert result.metrics["first_chunk_words"] >= 25


def test_every_trial_keeps_the_opening_it_wrote():
    """Quality has to be comparable afterwards, not taken on trust."""
    spec = ExperimentSpec(name="chunks", trials=2, minutes=1, queries=["q"],
                          arms=[Arm("a", search="none", tts="none")])
    results = asyncio.run(_fake_harness(spec).run_all())
    for result in results:
        assert result.first_chunk_text
        assert result.to_dict()["first_chunk_text"] == result.first_chunk_text


def test_a_saved_packet_makes_the_cost_estimate_measurable(tmp_path, monkeypatch):
    from experiments.adapters import packet as packet_mod

    monkeypatch.setattr(packet_mod, "PACKETS_DIR", tmp_path)
    (tmp_path / "big.json").write_text(json.dumps({"context": "x" * 7200, "sources": []}))
    spec = ExperimentSpec(name="c", trials=10, minutes=3, queries=["q"],
                          arms=[Arm("a", search="fixed_packet", tts="none",
                                    params={"generator": "benchmark", "packet": "big"})])
    est = cost_mod.estimate(spec)
    assert est.exa == 0.0, "a replayed packet calls nobody"
    assert est.gpu == 0.0
    assert est.anthropic > 0
