"""Results that outlive the terminal.

Every run gets a directory, and the directory is the record::

    experiments/runs/2026-09-05T14-22-03-exa-vs-current-search/
        spec.json      the exact configuration, credential-free
        trials.jsonl   one line per trial, appended as it happens
        summary.json   statistics
        report.md      the readable report and the recommendation
        artifacts/     generated audio, scripts

`trials.jsonl` is appended **during** the sweep rather than written at the end,
because the run this matters most for is the one that dies at trial 17 of 20.
Seventeen trials on disk beat a traceback.

Two rules this module keeps:

* **Nothing is deleted.** There is no prune, no rotate, no overwrite. A run
  directory is created once and only grows. Clearing history is a human
  action with `rm`, taken deliberately.
* **Nothing credential-shaped is written.** Every write goes through
  `redact.scrub`, and `verify_clean` re-reads the directory afterwards to
  prove it.
"""
from __future__ import annotations

import json
import os
import pathlib
import time
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from experiments import redact

#: Runs live inside the repo so they are easy to find, and are git-ignored so
#: they are not committed by accident. Promote one deliberately to keep it.
RUNS_DIR = pathlib.Path(__file__).resolve().parent / "runs"
INDEX = RUNS_DIR / "index.jsonl"


def _now_slug() -> str:
    return time.strftime("%Y-%m-%dT%H-%M-%S", time.localtime())


@dataclass
class Run:
    """A single experiment run on disk."""

    path: pathlib.Path

    # -- identity -------------------------------------------------------
    @property
    def id(self) -> str:
        return self.path.name

    @property
    def artifacts_dir(self) -> pathlib.Path:
        return self.path / "artifacts"

    # -- writing --------------------------------------------------------
    def write_spec(self, spec) -> None:
        self._write_json("spec.json", spec.to_dict())

    def append_trial(self, trial: dict) -> None:
        """Append one trial immediately. Crash-safe by construction."""
        line = json.dumps(redact.scrub(trial), sort_keys=False)
        with (self.path / "trials.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def write_summary(self, summary: dict) -> None:
        self._write_json("summary.json", summary)

    def write_report(self, markdown: str) -> None:
        (self.path / "report.md").write_text(redact.scrub_text(markdown), encoding="utf-8")

    def write_artifact(self, name: str, data: bytes) -> pathlib.Path:
        """Store a generated artifact - raw PCM/WAV audio, a script.

        Names are flattened so an adapter cannot write outside the run.
        """
        safe = name.replace("/", "_").replace("..", "_").lstrip(".")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        target = self.artifacts_dir / safe
        target.write_bytes(data)
        return target

    def _write_json(self, name: str, payload: Any) -> None:
        text = json.dumps(redact.scrub(payload), indent=2, sort_keys=False)
        (self.path / name).write_text(text, encoding="utf-8")

    # -- reading --------------------------------------------------------
    def spec_dict(self) -> Optional[dict]:
        return self._read_json("spec.json")

    def summary(self) -> Optional[dict]:
        return self._read_json("summary.json")

    def report(self) -> str:
        path = self.path / "report.md"
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def trials(self) -> list[dict]:
        path = self.path / "trials.jsonl"
        if not path.exists():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    # A half-written final line means the process died mid-trial.
                    # Everything before it is still good; say so rather than raise.
                    out.append({"error": "truncated trial record", "raw": line[:200]})
        return out

    def _read_json(self, name: str) -> Optional[dict]:
        path = self.path / name
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    # -- guarantee ------------------------------------------------------
    def verify_clean(self) -> list[str]:
        """Re-read everything written and prove no credential got through.

        Returns the offending file names; empty means clean. A test asserts
        this is empty, so a key shape the scrubber does not know about fails
        in CI rather than in a commit.
        """
        bad = []
        for path in sorted(self.path.rglob("*")):
            if not path.is_file() or path.suffix in (".wav", ".pcm", ".raw"):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if redact.looks_like_secret(text):
                bad.append(str(path.relative_to(self.path)))
        return bad


def create(spec, root: Optional[pathlib.Path] = None) -> Run:
    """Make a new run directory for `spec` and record it in the index."""
    base = pathlib.Path(root) if root else RUNS_DIR
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{_now_slug()}-{spec.slug()}"
    suffix = 1
    while path.exists():           # never overwrite an existing run
        suffix += 1
        path = base / f"{_now_slug()}-{spec.slug()}-{suffix}"
    path.mkdir(parents=True)
    run = Run(path)
    run.write_spec(spec)
    _append_index(base, {"id": path.name, "name": spec.name, "created": time.time(),
                         "arms": [a.name for a in spec.arms], "trials": spec.total_trials})
    return run


def _append_index(base: pathlib.Path, entry: dict) -> None:
    index = base / "index.jsonl"
    with index.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(redact.scrub(entry)) + "\n")


def list_runs(root: Optional[pathlib.Path] = None) -> list[Run]:
    """Every run on disk, newest first."""
    base = pathlib.Path(root) if root else RUNS_DIR
    if not base.exists():
        return []
    dirs = [p for p in base.iterdir() if p.is_dir()]
    return [Run(p) for p in sorted(dirs, key=lambda p: p.name, reverse=True)]


def load(run_id: str, root: Optional[pathlib.Path] = None) -> Optional[Run]:
    base = pathlib.Path(root) if root else RUNS_DIR
    path = base / run_id
    return Run(path) if path.is_dir() else None


def find_previous(name: str, exclude: Optional[str] = None,
                  root: Optional[pathlib.Path] = None) -> Optional[Run]:
    """The most recent earlier run of the same experiment name.

    This is what makes "compare against previous experiments" automatic: a run
    named the same thing is the baseline for the next one.
    """
    for run in list_runs(root):
        if exclude and run.id == exclude:
            continue
        spec = run.spec_dict()
        if spec and spec.get("name") == name:
            return run
    return None


def iter_trials(run: Run) -> Iterator[dict]:
    yield from run.trials()
