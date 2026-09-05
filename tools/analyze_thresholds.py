"""Read a chunk-threshold run and lay the openings out for judgement.

    python tools/analyze_thresholds.py experiments/runs/<run-id>
    python tools/analyze_thresholds.py <path-to-trials.jsonl>

**It does not score quality.** Whether an opening sounds natural aloud, earns
its curiosity, or reads as deliberate rather than truncated is a judgement, and
a number invented for it would launder that judgement into something that looks
like a measurement. So this computes only what can be counted, and prints the
openings side by side for a person to read:

  * counted   - latency, word count, sentence count, terminal punctuation,
                and whether two thresholds returned the *same string*
  * flagged   - short openings and hedged first words, as things to look at
  * judged    - nothing

The tie percentage is the load-bearing number. Thresholds that return an
identical string are not alternatives; they are the same decision wearing
different labels, and knowing which are which collapses five options into the
two or three that exist.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

#: Openings that begin this way start at a distance from the subject. Flagged
#: for reading, never scored - plenty of good sentences start with "It is".
HEDGE_OPENERS = (
    "there is", "there are", "there's", "it is", "it's", "in today's",
    "many people", "we often", "one of the most", "in a world", "imagine",
    "picture this", "have you ever",
)

#: Below this a chunk is short enough that it is worth looking at before
#: trusting it as an opening. Not a verdict - the text is printed.
SHORT_WORDS = 8


def load_trials(target: pathlib.Path) -> list[dict]:
    path = target / "trials.jsonl" if target.is_dir() else target
    if not path.exists():
        raise SystemExit(f"No trials at {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return [r for r in rows if r.get("ok")]


def structure(text: str) -> dict:
    """Only what can be counted from the string itself."""
    words = text.split()
    lowered = text.lower().lstrip()
    return {
        "words": len(words),
        "chars": len(text),
        "ends_terminal": text.rstrip().endswith((".", "!", "?")),
        "starts_capital": bool(words) and words[0][:1].isupper(),
        # A rough sentence count: terminal marks followed by a space or the end.
        "sentences": sum(text.count(mark) for mark in ".!?"),
        "has_digit": any(c.isdigit() for c in text),
        # A capitalised word that is not the first is a weak proper-noun proxy.
        "has_proper_noun": any(w[:1].isupper() for w in words[1:]),
        "hedged_open": any(lowered.startswith(h) for h in HEDGE_OPENERS),
        "short": len(words) < SHORT_WORDS,
    }


def median(values):
    clean = [v for v in values if isinstance(v, (int, float))]
    return statistics.median(clean) if clean else None


def analyse(trials: list[dict]) -> dict:
    thresholds = next((t["metrics"].get("probe_thresholds") for t in trials
                       if t["metrics"].get("probe_thresholds")), [])
    if not thresholds:
        raise SystemExit("No probe_thresholds in this run - is it a threshold run?")

    first_token = median([t["metrics"].get("seg_dispatch_to_first_token") for t in trials])
    rows = {}
    for threshold in thresholds:
        key = str(threshold)
        texts = [(t["index"], (t.get("threshold_texts") or {}).get(key)) for t in trials]
        present = [(i, x) for i, x in texts if x]
        boundary = median([t["metrics"].get(f"boundary_{threshold}_at") for t in trials])
        rows[threshold] = {
            "n": len(present),
            "missing": len(texts) - len(present),
            "boundary_from_first_token": boundary,
            "from_dispatch": (first_token + boundary
                              if first_token is not None and boundary is not None else None),
            "words": median([len(x.split()) for _, x in present]),
            "rule_cost": median([t["metrics"].get(f"boundary_wait_{threshold}") for t in trials]),
            "texts": dict(present),
            "structure": [structure(x) for _, x in present],
        }

    # Ties: per trial, does this threshold return the same string as another?
    for threshold in thresholds:
        same_as: dict = {}
        for other in thresholds:
            if other == threshold:
                continue
            matches = 0
            compared = 0
            for trial in trials:
                mine = (trial.get("threshold_texts") or {}).get(str(threshold))
                theirs = (trial.get("threshold_texts") or {}).get(str(other))
                if mine and theirs:
                    compared += 1
                    matches += int(mine == theirs)
            if compared:
                same_as[other] = matches / compared
        rows[threshold]["same_as"] = same_as
        rows[threshold]["tied_any"] = (
            max(same_as.values()) if same_as else 0.0)
    return {"thresholds": thresholds, "rows": rows, "trials": len(trials),
            "first_token_median": first_token}


def render(result: dict, examples: int) -> str:
    rows, thresholds = result["rows"], result["thresholds"]
    out = [f"\n  {result['trials']} successful trials"
           f"   median dispatch -> first token: "
           f"{result['first_token_median']:.3f}s" if result["first_token_median"]
           else f"\n  {result['trials']} successful trials", ""]

    out.append(f"  {'thr':>4} {'ready (disp)':>13} {'from 1st tok':>13} "
               f"{'words':>6} {'rule cost':>10} {'same as other':>14}  flags")
    out.append("  " + "-" * 86)
    for threshold in thresholds:
        row = rows[threshold]
        struct = row["structure"]
        flags = []
        if struct:
            if any(s["short"] for s in struct):
                flags.append(f"short×{sum(s['short'] for s in struct)}")
            if any(not s["ends_terminal"] for s in struct):
                flags.append("no-terminal-punct")
            if any(not s["starts_capital"] for s in struct):
                flags.append("lowercase-start")
            if any(s["hedged_open"] for s in struct):
                flags.append(f"hedged×{sum(s['hedged_open'] for s in struct)}")
            concrete = sum(s["has_digit"] or s["has_proper_noun"] for s in struct)
            if concrete < len(struct):
                flags.append(f"no-concrete-marker×{len(struct) - concrete}")
        if row["missing"]:
            flags.append(f"missing×{row['missing']}")

        def fmt(value, suffix="s"):
            return f"{value:.3f}{suffix}" if isinstance(value, float) else "—"

        tied = row["tied_any"]
        tie_note = "—" if tied < 0.5 else f"{tied * 100:.0f}% identical"
        out.append(f"  {threshold:>4} {fmt(row['from_dispatch']):>13} "
                   f"{fmt(row['boundary_from_first_token']):>13} "
                   f"{row['words'] if row['words'] is not None else '—':>6} "
                   f"{fmt(row['rule_cost']):>10} {tie_note:>14}  "
                   f"{', '.join(flags) if flags else 'none counted'}")

    out += ["", "  Ties, threshold by threshold (share of trials returning an "
            "identical string):"]
    for threshold in thresholds:
        pairs = [f"{other}:{share * 100:.0f}%"
                 for other, share in sorted(rows[threshold]["same_as"].items())
                 if share > 0]
        out.append(f"    {threshold:>3} -> {'  '.join(pairs) if pairs else 'unique to itself'}")

    out += ["", "  Openings, side by side. Read these; nothing above judges them.", ""]
    indices = sorted(next(iter(rows.values()))["texts"].keys())[:examples]
    for index in indices:
        out.append(f"  --- trial {index} " + "-" * 60)
        for threshold in thresholds:
            text = rows[threshold]["texts"].get(index)
            if not text:
                out.append(f"    {threshold:>3}w  (not reached)")
                continue
            marker = ""
            for other in thresholds:
                if other < threshold and rows[other]["texts"].get(index) == text:
                    marker = f"  [same as {other}w]"
                    break
            out.append(f"    {threshold:>3}w ({len(text.split()):>2} words){marker}")
            out.append(f"         {text}")
        out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="a run directory, or a trials.jsonl")
    parser.add_argument("--examples", type=int, default=3,
                        help="how many trials to print openings for")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    trials = load_trials(pathlib.Path(args.run))
    result = analyse(trials)
    if args.json:
        printable = {k: v for k, v in result.items()}
        for row in printable["rows"].values():
            row.pop("structure", None)
        print(json.dumps(printable, indent=2, default=str))
    else:
        print(render(result, args.examples))
        print("\n  Quality is not scored here. The counted columns narrow the "
              "field;\n  the openings above decide it.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
