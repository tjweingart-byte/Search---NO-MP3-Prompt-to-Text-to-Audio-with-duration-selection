"""Turn trials into an answer, and say how much to trust it.

The headline question this exists to answer is "where is the bottleneck", and
that is mechanical: the stage with the largest median duration. Everything else
in the report is there to stop that answer being read for more than it is
worth - the spread, the failures, whether the arms differ in more than one
dimension, and whether the sample can support the comparison at all.

The recommendation is deliberately allowed to be "not enough evidence". A
report that always picks a winner is a report that will eventually pick a
wrong one.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Optional

from experiments import stats as stats_mod
from experiments.spec import ExperimentSpec

#: The measurement the product is judged on, when there is speech in the run.
HEADLINE = "first_audio"
#: With no speech stage - a search-only or script-only experiment - the last
#: honest checkpoint is the first speakable chunk: the moment the pipeline had
#: something it could hand to a voice. Falling back to it is what lets an
#: "Exa vs current search" comparison have a headline at all.
HEADLINE_NO_SPEECH = "first_chunk"
HEADLINE_LABEL = {
    "first_audio": "Time to first audio",
    "first_chunk": "Time to first speakable chunk",
}


def headline_key(spec: ExperimentSpec) -> str:
    """One measure for the whole run, so arms stay comparable.

    If any arm has no speech stage, every arm is judged on the chunk time -
    comparing one arm's first audio against another's first chunk would be
    comparing two different events.
    """
    if all(arm.tts not in ("none", "") for arm in spec.arms):
        return HEADLINE
    return HEADLINE_NO_SPEECH

STAGE_KEYS = [
    ("search_seconds", "search"),
    ("generate_seconds", "generate"),
    ("synthesis_seconds", "synthesis"),
]


@dataclass
class ArmReport:
    name: str
    trials: int
    failures: int
    simulated: bool
    headline: Optional[stats_mod.Summary]
    stages: dict = field(default_factory=dict)          # stage -> Summary
    stage_hosts: dict = field(default_factory=dict)     # stage -> host
    cost: float = 0.0
    #: Trials whose token usage the stream could not report. Their cost is
    #: unknown, not zero, and the report must not quietly average it as free.
    cost_unknown: int = 0
    errors: list[str] = field(default_factory=list)

    def bottleneck(self) -> Optional[tuple[str, stats_mod.Summary]]:
        real = {k: v for k, v in self.stages.items() if v is not None}
        if not real:
            return None
        name = max(real, key=lambda k: real[k].median)
        return name, real[name]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "trials": self.trials,
            "failures": self.failures,
            "simulated": self.simulated,
            "headline": self.headline.to_dict() if self.headline else None,
            "stages": {k: (v.to_dict() if v else None) for k, v in self.stages.items()},
            "stage_hosts": self.stage_hosts,
            "cost": round(self.cost, 6),
            "cost_unknown": self.cost_unknown,
            "errors": self.errors[:5],
        }


def analyse(spec: ExperimentSpec, results) -> dict:
    """Aggregate raw trials into per-arm statistics and comparisons."""
    dicts = [r.to_dict() if hasattr(r, "to_dict") else r for r in results]
    key = headline_key(spec)
    arms: list[ArmReport] = []

    for arm in spec.arms:
        mine = [d for d in dicts if d["arm"] == arm.name]
        good = [d for d in mine if d.get("ok")]
        failed = [d for d in mine if not d.get("ok")]
        headline = stats_mod.summarise([d["metrics"].get(key) for d in good
                                        if d["metrics"].get(key) is not None])
        stage_stats, hosts = {}, {}
        # Not `key`: that name holds the headline metric for the whole run, and
        # shadowing it here silently moved every comparison onto the last stage.
        for metric, label in STAGE_KEYS:
            values = [d["metrics"].get(metric) for d in good if d["metrics"].get(metric) is not None]
            stage_stats[label] = stats_mod.summarise(values)
            for d in good:
                for s in d.get("timeline", {}).get("stages", []):
                    if s["name"] == label:
                        hosts[label] = s.get("host", "local")
                        break
        arms.append(ArmReport(
            name=arm.name,
            trials=len(mine),
            failures=len(failed),
            simulated=any(d.get("simulated") for d in mine),
            headline=headline,
            stages=stage_stats,
            stage_hosts=hosts,
            cost=sum(d.get("cost", 0.0) for d in mine),
            cost_unknown=sum(
                1 for d in good
                if d.get("usage") and d["usage"].get("usage_known") is False
            ),
            errors=[d["error"] for d in failed if d.get("error")],
        ))

    comparisons = []
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            a, b = arms[i], arms[j]
            a_vals = [d["metrics"].get(key) for d in dicts
                      if d["arm"] == a.name and d.get("ok") and d["metrics"].get(key) is not None]
            b_vals = [d["metrics"].get(key) for d in dicts
                      if d["arm"] == b.name and d.get("ok") and d["metrics"].get(key) is not None]
            cmp = stats_mod.bootstrap_diff(a_vals, b_vals, a_label=a.name, b_label=b.name,
                                           seed=spec.seed)
            if cmp:
                spec_a = next(x for x in spec.arms if x.name == a.name)
                spec_b = next(x for x in spec.arms if x.name == b.name)
                comparisons.append({
                    "comparison": cmp.to_dict(),
                    "dimensions": spec_a.dimensions_vs(spec_b),
                })

    return {
        "spec": spec.to_dict(),
        "headline_metric": key,
        "arms": [a.to_dict() for a in arms],
        "comparisons": comparisons,
        "total_cost": round(sum(a.cost for a in arms), 6),
        "simulated": any(a.simulated for a in arms),
        "_arm_objects": arms,
    }


def render(spec: ExperimentSpec, analysis: dict, previous: Optional[dict] = None) -> str:
    """The markdown report: findings first, caveats attached to them."""
    arms: list[ArmReport] = analysis["_arm_objects"]
    out: list[str] = [f"# {spec.name}", ""]

    if analysis["simulated"]:
        out += ["> **SIMULATED RUN — these numbers are fabricated.** At least one",
                "> adapter was a stand-in. Nothing here is a measurement of real",
                "> Exa, Claude or Chatterbox behaviour.", ""]

    out += ["## What was run", "",
            f"- **Arms:** {', '.join(a.name for a in spec.arms)}",
            f"- **Trials:** {spec.trials} per arm per query, {spec.total_trials} total",
            f"- **Queries:** {len(spec.queries)}",
            f"- **Episode length:** {spec.minutes:g} min",
            f"- **Concurrency:** {spec.concurrency} (sequential; see harness.py)", ""]

    # -- the headline -------------------------------------------------
    measure = analysis.get("headline_metric", HEADLINE)
    out += [f"## {HEADLINE_LABEL.get(measure, measure)}", ""]
    if measure == HEADLINE_NO_SPEECH:
        out += ["*No speech stage in this run, so the measure is the moment the "
                "pipeline had a chunk ready to speak.*", ""]
    out += [
            "| arm | median | spread | p95 | n | failed |",
            "|---|---|---|---|---|---|"]
    for arm in arms:
        if not arm.headline:
            out.append(f"| {arm.name} | — | every trial failed | — | 0 | {arm.failures} |")
            continue
        h = arm.headline
        out.append(f"| {arm.name} | **{h.median:.2f}s** | {h.minimum:.2f}–{h.maximum:.2f}s "
                   f"| {h.p95:.2f}s | {h.n} | {arm.failures} |")
    out.append("")

    # -- the bottleneck -----------------------------------------------
    out += ["## Where the time goes", ""]
    for arm in arms:
        out.append(f"### {arm.name}")
        out.append("")
        bn = arm.bottleneck()
        rows = ["| stage | host | median | share |", "|---|---|---|---|"]
        total = sum(s.median for s in arm.stages.values() if s)
        for label, summary in arm.stages.items():
            if not summary:
                rows.append(f"| {label} | — | not separable | — |")
                continue
            share = (summary.median / total * 100) if total else 0
            mark = "  ←**bottleneck**" if bn and bn[0] == label else ""
            host = arm.stage_hosts.get(label, "local")
            rows.append(f"| {label} | `{host}` | {summary.median:.2f}s{mark} | {share:.0f}% |")
        out += rows + [""]
        if bn:
            out += [f"**Bottleneck: `{bn[0]}` at {bn[1].median:.2f}s median "
                    f"({bn[1].minimum:.2f}–{bn[1].maximum:.2f}).**", ""]
        if arm.failures:
            out += [f"{arm.failures} of {arm.trials} trials failed. First error: "
                    f"`{arm.errors[0] if arm.errors else 'unknown'}`", ""]

    # -- comparisons ---------------------------------------------------
    if analysis["comparisons"]:
        out += ["## Does the difference hold up?", ""]
        for entry in analysis["comparisons"]:
            cmp = entry["comparison"]
            out.append(f"- {cmp['verdict']}")
            if not cmp["significant"] and cmp.get("required_n"):
                out.append(f"  - About **{cmp['required_n']} trials per arm** would be "
                           f"needed to resolve a gap this size. This run had "
                           f"{spec.trials}.")
            dims = entry["dimensions"]
            if len(dims) > 1:
                out.append(f"  - ⚠️ These arms differ in **{len(dims)} dimensions** "
                           f"({', '.join(dims)}), so the difference cannot be "
                           f"attributed to any one of them.")
        out.append("")

    # -- against history -----------------------------------------------
    if previous:
        out += ["## Against the previous run", ""]
        prev_arms = {a["name"]: a for a in previous.get("arms", [])}
        any_row = False
        for arm in arms:
            prev = prev_arms.get(arm.name)
            if not prev or not prev.get("headline") or not arm.headline:
                continue
            before = prev["headline"]["median"]
            now = arm.headline.median
            delta = now - before
            arrow = "faster" if delta < 0 else "slower"
            out.append(f"- **{arm.name}**: {before:.2f}s → {now:.2f}s "
                       f"({abs(delta):.2f}s {arrow})")
            any_row = True
        if not any_row:
            out.append("- No arm from the previous run shares a name with this one.")
        out.append("")

    # -- cost ----------------------------------------------------------
    label = ("Simulated, and therefore meaningless as a cost"
             if analysis["simulated"] else "Actual, from recorded usage")
    out += ["## Cost", "", f"{label}: **${analysis['total_cost']:.4f}**", ""]
    for arm in arms:
        line = f"- {arm.name}: ${arm.cost:.4f}"
        if arm.cost_unknown:
            line += (f"  — **incomplete**: {arm.cost_unknown} trial(s) reported no "
                     f"token usage, so their model cost is unknown, not zero")
        out.append(line)
    out.append("")

    # -- recommendation -------------------------------------------------
    out += ["## Recommendation", "", _recommend(spec, analysis, arms), ""]
    return "\n".join(out)


def _recommend(spec: ExperimentSpec, analysis: dict, arms: list[ArmReport]) -> str:
    if analysis["simulated"]:
        return ("**None — this run was simulated.** Connect the real adapters and "
                "run it again. The only thing this proves is that the harness works.")

    usable = [a for a in arms if a.headline]
    if not usable:
        return "**None — every trial failed.** Fix the errors above before reading anything into this."

    lines = []
    fastest = min(usable, key=lambda a: a.headline.median)
    decisive = [e for e in analysis["comparisons"] if e["comparison"]["significant"]]
    confounded = [e for e in analysis["comparisons"] if len(e["dimensions"]) > 1]

    if len(usable) == 1:
        lines.append(f"Single arm: **{fastest.name}** reaches first audio in "
                     f"{fastest.headline.median:.2f}s (median).")
    elif decisive:
        lines.append(f"**{fastest.name}** is the fastest arm at "
                     f"{fastest.headline.median:.2f}s median to first audio, and the "
                     f"difference survives a bootstrap confidence interval.")
        if confounded:
            lines.append("Treat it as provisional: the winning arm differs from its "
                         "comparator in more than one dimension, so *which* change "
                         "won is not established. Vary one thing at a time next.")
    else:
        lines.append("**No arm is measurably faster.** The differences observed are "
                     "within noise at this sample size, so choosing between these "
                     "arms on latency is not yet justified by the data.")

    bn = fastest.bottleneck()
    if bn:
        share = ""
        total = sum(s.median for s in fastest.stages.values() if s)
        if total:
            share = f", {bn[1].median / total * 100:.0f}% of measured stage time"
        lines.append(f"The bottleneck in `{fastest.name}` is **{bn[0]}** "
                     f"({bn[1].median:.2f}s median{share}). That is where an "
                     f"optimisation would pay; work anywhere else is rounding error "
                     f"until it moves.")

    slowest_stage_over_budget = any(
        a.headline and a.headline.median > 1.0 for a in usable
    )
    if slowest_stage_over_budget:
        lines.append("Note against the product spec: FAM's one-sentence spec is audio "
                     "within about a second of the question. Arms above that line are "
                     "failing the spec, not merely losing a comparison.")
    return "\n\n".join(lines)
