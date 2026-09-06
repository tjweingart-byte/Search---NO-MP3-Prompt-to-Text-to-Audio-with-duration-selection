"""The threshold analyser: it must count, tie, and refuse to score."""
from __future__ import annotations

import ast
import importlib.util
import json
import pathlib

import pytest

MODULE = pathlib.Path(__file__).resolve().parent.parent / "tools" / "analyze_thresholds.py"


def _load():
    spec = importlib.util.spec_from_file_location("analyze_thresholds", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trial(index, texts, boundaries, first_token=0.97):
    return {
        "index": index, "ok": True, "threshold_texts": texts,
        "metrics": {
            "probe_thresholds": [5, 10, 15, 20, 25],
            "seg_dispatch_to_first_token": first_token,
            **{f"boundary_{t}_at": boundaries[t] for t in boundaries},
            **{f"boundary_wait_{t}": 0.01 for t in boundaries},
        },
    }


SHORT = "Boards moved on."
MID = "Boards moved on. That stopped being true in November 2023 for one reason."
LONG = MID + " Everything since has followed from it."


def _fixture(n=4):
    return [_trial(i,
                   {"5": SHORT, "10": MID, "15": MID, "20": MID, "25": LONG},
                   {5: 0.12, 10: 0.54, 15: 0.54, 20: 0.54, 25: 0.80})
            for i in range(1, n + 1)]


def test_it_does_not_emit_a_quality_score():
    """The user asked for judgement, not a laundered number."""
    module = _load()
    result = module.analyse(_fixture())

    # Behavioural, not textual: nothing it emits may be a score. (Checking the
    # words in the source instead would fail on a comment that disavows them.)
    def keys(value, seen=None):
        seen = seen if seen is not None else set()
        if isinstance(value, dict):
            for key, item in value.items():
                seen.add(str(key).lower())
                keys(item, seen)
        elif isinstance(value, list):
            for item in value:
                keys(item, seen)
        return seen

    emitted = keys(result)
    for banned in ("score", "rating", "grade", "verdict", "quality", "judgement"):
        assert not any(banned in key for key in emitted), \
            f"the analyser emits a {banned!r} field"

    # And no model is called to opine.
    source = MODULE.read_text()
    tree = ast.parse(source)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or "").split(".")[0])
    assert "anthropic" not in names


def test_ties_are_counted_exactly():
    module = _load()
    result = module.analyse(_fixture())
    rows = result["rows"]
    assert rows[10]["same_as"][15] == 1.0
    assert rows[10]["same_as"][20] == 1.0
    assert rows[10]["same_as"][25] == 0.0
    assert rows[5]["tied_any"] == 0.0, "the 5-word opening is its own text"
    assert rows[25]["tied_any"] == 0.0


def test_a_partial_tie_is_reported_as_a_fraction():
    module = _load()
    trials = _fixture(4)
    # One trial where 10 and 15 diverge.
    trials[3]["threshold_texts"]["15"] = LONG
    rows = module.analyse(trials)["rows"]
    assert rows[10]["same_as"][15] == pytest.approx(0.75)


def test_latency_columns_come_from_the_data():
    module = _load()
    rows = module.analyse(_fixture())["rows"]
    assert rows[10]["boundary_from_first_token"] == pytest.approx(0.54)
    assert rows[10]["from_dispatch"] == pytest.approx(1.51)
    assert rows[25]["from_dispatch"] == pytest.approx(1.77)
    assert rows[5]["words"] == 3


def test_structure_counts_only_what_is_countable():
    module = _load()
    out = module.structure("Boards moved on in November 2023.")
    assert out["ends_terminal"] is True
    assert out["starts_capital"] is True
    assert out["has_digit"] is True
    assert out["has_proper_noun"] is True
    assert out["hedged_open"] is False

    hedged = module.structure("There is a reason boards no longer act.")
    assert hedged["hedged_open"] is True
    assert hedged["has_digit"] is False

    fragment = module.structure("boards moved")
    assert fragment["ends_terminal"] is False
    assert fragment["starts_capital"] is False
    assert fragment["short"] is True


def test_the_openings_are_printed_for_reading():
    module = _load()
    text = module.render(module.analyse(_fixture()), examples=2)
    assert SHORT in text and MID in text and LONG in text
    assert "[same as 10w]" in text, "ties must be visible where they are read"
    assert "nothing above judges them" in text
    assert "trial 1" in text and "trial 2" in text


def test_a_threshold_never_reached_is_reported_as_missing():
    module = _load()
    trials = _fixture(2)
    for trial in trials:
        trial["threshold_texts"]["25"] = None
        trial["metrics"]["boundary_25_at"] = None
    rows = module.analyse(trials)["rows"]
    assert rows[25]["n"] == 0 and rows[25]["missing"] == 2
    assert rows[25]["boundary_from_first_token"] is None
    assert "not reached" in module.render(module.analyse(trials), examples=1)


def test_it_reads_a_run_directory_or_a_file(tmp_path):
    module = _load()
    path = tmp_path / "trials.jsonl"
    path.write_text("\n".join(json.dumps(t) for t in _fixture(2)))
    assert len(module.load_trials(tmp_path)) == 2
    assert len(module.load_trials(path)) == 2


def test_failed_trials_are_excluded(tmp_path):
    module = _load()
    rows = _fixture(2) + [{"index": 3, "ok": False, "error": "boom", "metrics": {}}]
    path = tmp_path / "trials.jsonl"
    path.write_text("\n".join(json.dumps(t) for t in rows))
    assert len(module.load_trials(path)) == 2


def _multi_topic_fixture():
    """Two topics: one where 10 words is clean, one where it is a fragment."""
    good = [_trial(i, {"5": SHORT, "10": MID, "15": MID, "20": MID, "25": LONG},
                   {5: 0.12, 10: 0.54, 15: 0.54, 20: 0.54, 25: 0.80})
            for i in range(1, 4)]
    for t in good:
        t["query"] = "how does a heat pump work"

    # The risk topic: at 5 and 10 the opening is a hedged fragment.
    bad_text = "there is a feeling"
    risky = [_trial(i, {"5": bad_text, "10": bad_text, "15": MID, "20": MID, "25": LONG},
                    {5: 0.10, 10: 0.20, 15: 0.60, 20: 0.60, 25: 0.85})
             for i in range(1, 4)]
    for t in risky:
        t["query"] = "what makes a piece of music feel nostalgic"
    return good + risky


def test_trials_are_grouped_by_topic():
    module = _load()
    groups = module.by_topic(_multi_topic_fixture())
    assert len(groups) == 2
    assert all(len(rows) == 3 for rows in groups.values())


def test_consistency_disqualifies_a_threshold_that_fails_anywhere():
    module = _load()
    groups = module.by_topic(_multi_topic_fixture())
    risky = module.analyse(groups["what makes a piece of music feel nostalgic"])
    cons = module.consistency(risky)

    assert cons[5]["clean"] < cons[5]["n"], "a hedged fragment must not count as clean"
    assert cons[10]["clean"] < cons[10]["n"]
    assert cons[15]["clean"] == cons[15]["n"]
    assert cons[25]["clean"] == cons[25]["n"]


def test_the_multi_topic_view_names_the_earliest_surviving_threshold():
    module = _load()
    text = module.render_multi_topic(module.by_topic(_multi_topic_fixture()), examples=1)

    assert "READY TIME" in text and "CHUNK WORDS" in text
    assert "STRUCTURALLY CLEAN" in text and "TIES" in text
    assert "earliest surviving threshold: 15w" in text
    assert "5w  disqualified" in text and "10w  disqualified" in text
    assert "filter, not a recommendation" in text
    # Both topics' openings are printed for reading.
    assert "how does a heat pump work" in text
    assert "what makes a piece of music feel nostalgic" in text


def test_one_bad_topic_is_not_averaged_away():
    """The topic that breaks a threshold is the only one that matters."""
    module = _load()
    trials = _multi_topic_fixture()
    # Nine clean topics would not rescue the tenth.
    for extra in range(3):
        for t in _multi_topic_fixture()[:3]:
            clone = dict(t)
            clone["query"] = f"clean topic {extra}"
            trials.append(clone)
    text = module.render_multi_topic(module.by_topic(trials), examples=0)
    assert "earliest surviving threshold: 15w" in text
