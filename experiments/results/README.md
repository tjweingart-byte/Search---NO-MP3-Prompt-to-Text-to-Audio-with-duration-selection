# Preserved experiment results

`experiments/runs/` is git-ignored on purpose — it holds audio and raw API
records, and it is per-machine. This folder is the opposite: the part of a run
worth keeping, committed deliberately, so an experiment can be reread from any
checkout instead of being paid for twice.

Promote a run with:

    python tools/preserve_run.py --latest --as <slug>

That copies the per-trial numbers, the report and any text artifacts (never
audio), re-runs the redaction check on every file it writes, and refuses if
anything still looks like a credential.

Each folder holds:

| file | what it is |
|---|---|
| `spec.json` | the experiment exactly as run |
| `trials.jsonl` | one row per trial, numbers only |
| `report.md` | the report as generated |
| `openings_by_arm.md` | the text the model actually wrote, where captured |
| `ANALYSIS.md` | written by hand: what it means, and what it does not |

`ANALYSIS.md` is not generated and is never overwritten by the tool.
