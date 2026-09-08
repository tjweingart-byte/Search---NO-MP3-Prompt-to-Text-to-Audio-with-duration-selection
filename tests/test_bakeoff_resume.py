"""One engine resident at a time, and a killed run that resumes.

Exercised with fake engines, so the control flow is tested without loading
anything large - which is the point: the failure being fixed is that loading
two real models at once got the process killed.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

numpy = pytest.importorskip("numpy")

from experiments import voice_bakeoff as bake
from tools import voice_bakeoff as runner

PASSAGES = [
    {"id": "intimate", "label": "intimate", "stresses": "warmth",
     "watch_for": "?", "text": "One."},
    {"id": "energetic", "label": "energetic", "stresses": "momentum",
     "watch_for": "?", "text": "Two."},
    {"id": "authoritative", "label": "news", "stresses": "numbers",
     "watch_for": "?", "text": "Three."},
]

#: Every engine that is currently loaded. The invariant under test is that this
#: never holds more than one entry.
LIVE: list = []
HIGH_WATER: list = []


class _Fake:
    """A stand-in engine that registers itself as resident while it exists."""

    def __init__(self, key, fail_on=None):
        self.key, self.fail_on = key, fail_on
        LIVE.append(key)
        HIGH_WATER.append(len(LIVE))

    def __call__(self, text):
        if self.fail_on and self.fail_on in text:
            raise RuntimeError("engine exploded")
        rate = 24000
        n = rate // 4
        return (numpy.sin(numpy.linspace(0, 60 * numpy.pi, n)).astype("float32"),
                rate)

    def __del__(self):
        try:
            LIVE.remove(self.key)
        except ValueError:
            pass


def _candidate(key, fail_on=None, load_error=None):
    def make():
        if load_error:
            raise RuntimeError(load_error)
        return _Fake(key, fail_on)
    return bake.Candidate(key=key, label=key.title(), synth=make)


@pytest.fixture(autouse=True)
def _reset():
    LIVE.clear()
    HIGH_WATER.clear()
    yield
    LIVE.clear()


def test_only_one_engine_is_resident_at_a_time(tmp_path):
    """The bug: every loaded model was kept in a dict for the whole run, so
    the second Chatterbox loaded on top of the first and macOS killed it."""
    candidates = [_candidate(f"engine_{i}") for i in range(4)]
    runner.generate_all(candidates, PASSAGES, tmp_path, "mps", None, False)
    assert max(HIGH_WATER) == 1, f"engines held together: high water {HIGH_WATER}"
    assert LIVE == []


def test_every_clip_is_persisted_as_it_is_made(tmp_path):
    runner.generate_all([_candidate("a")], PASSAGES, tmp_path, None, None, False)
    for passage in PASSAGES:
        assert bake.raw_path(tmp_path, "a", passage["id"]).exists()


def test_a_later_failure_does_not_cost_an_earlier_success(tmp_path):
    """The rule that matters after a kill: finished work is kept."""
    good, bad = _candidate("good"), _candidate("bad", load_error="killed")
    outcome = runner.generate_all([good, bad], PASSAGES, tmp_path, None, None, False)
    assert outcome["done"]["good"] == 3
    assert "bad" in outcome["failed"]
    for passage in PASSAGES:
        assert bake.raw_path(tmp_path, "good", passage["id"]).exists()


def test_resuming_does_not_regenerate_and_keeps_the_first_take(tmp_path):
    """A resume must not re-roll a clip that already exists."""
    runner.generate_all([_candidate("a")], PASSAGES, tmp_path, None, None, False)
    before = {p["id"]: bake.raw_path(tmp_path, "a", p["id"]).read_bytes()
              for p in PASSAGES}

    LIVE.clear()
    HIGH_WATER.clear()
    outcome = runner.generate_all([_candidate("a")], PASSAGES, tmp_path, None,
                                  None, False)
    assert HIGH_WATER == [], "the engine was loaded even though nothing was needed"
    assert outcome["done"]["a"] == 3
    for passage in PASSAGES:
        assert (bake.raw_path(tmp_path, "a", passage["id"]).read_bytes()
                == before[passage["id"]])


def test_force_does_regenerate(tmp_path):
    runner.generate_all([_candidate("a")], PASSAGES, tmp_path, None, None, False)
    LIVE.clear(); HIGH_WATER.clear()
    runner.generate_all([_candidate("a")], PASSAGES, tmp_path, None, None, True)
    assert HIGH_WATER == [1]


def test_a_partial_engine_resumes_only_what_is_missing(tmp_path):
    """The realistic kill: two passages done, the third in flight."""
    runner.generate_all([_candidate("a", fail_on="Three")], PASSAGES, tmp_path,
                        None, None, False)
    assert bake.raw_path(tmp_path, "a", "intimate").exists()
    assert not bake.raw_path(tmp_path, "a", "authoritative").exists()

    LIVE.clear(); HIGH_WATER.clear()
    outcome = runner.generate_all([_candidate("a")], PASSAGES, tmp_path, None,
                                  None, False)
    assert outcome["done"]["a"] == 3
    assert bake.raw_path(tmp_path, "a", "authoritative").exists()


def test_labelling_skips_an_incomplete_candidate(tmp_path, capsys):
    """A half-generated engine appearing on some passages and not others would
    tell the listener something the blinding is meant to hide."""
    whole = _candidate("whole")
    partial = _candidate("partial", fail_on="Three")
    runner.generate_all([whole, partial], PASSAGES, tmp_path, None, None, False)
    another = _candidate("another")
    runner.generate_all([another], PASSAGES, tmp_path, None, None, False)

    complete, letters, clips = runner.label(
        [whole, partial, another], PASSAGES, tmp_path, 7, None)
    keys = {c.key for c in complete}
    assert keys == {"whole", "another"}
    for passage in PASSAGES:
        assert set(letters[passage["id"]]) == keys
    assert len(clips) == len(PASSAGES) * 2


def test_labelling_normalises_and_stays_blind(tmp_path):
    quiet, loud = _candidate("quiet"), _candidate("loud")
    runner.generate_all([quiet, loud], PASSAGES, tmp_path, None, None, False)
    # Make one raw clip much quieter than the other, then label.
    for passage in PASSAGES:
        path = bake.raw_path(tmp_path, "quiet", passage["id"])
        samples, rate = bake.read_wav(path)
        bake.write_wav(path, samples * 0.05, rate)

    complete, letters, clips = runner.label([quiet, loud], PASSAGES, tmp_path,
                                            7, None)
    assert len(complete) == 2
    for passage in PASSAGES:
        levels = []
        for letter in letters[passage["id"]].values():
            samples, _ = bake.read_wav(
                tmp_path / "clips" / passage["id"] / f"{letter}.wav")
            levels.append(float(numpy.sqrt(numpy.mean(samples ** 2))))
        assert max(levels) / min(levels) == pytest.approx(1.0, abs=0.2)


def test_release_memory_never_raises_without_torch(monkeypatch):
    """It must not become the thing that fails a run."""
    import builtins

    real = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    bake.release_memory("mps")


def test_the_log_is_flushed_so_a_kill_does_not_lose_it(tmp_path):
    """`zsh: killed` means nothing buffered survives."""
    path = tmp_path / "progress.log"
    with path.open("a", encoding="utf-8") as handle:
        runner.log_line(handle, "[engine] GENERATING intimate")
        # Read it back while the handle is still open: it must already be there.
        assert "GENERATING intimate" in path.read_text(encoding="utf-8")
