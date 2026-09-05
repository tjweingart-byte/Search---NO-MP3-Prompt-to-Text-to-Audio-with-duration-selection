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

#: The forensic decomposition, in the order the time is actually spent.
#: The first three sum to the fourth; the fourth is the number a listener feels.
SEGMENT_KEYS = [
    ("seg_dispatch_to_first_token", "dispatch -> first token"),
    ("seg_first_token_to_25_words", "first token -> 25 words"),
    ("seg_25_words_to_boundary", "25 words -> sentence boundary"),
]
TOTAL_SEGMENT = ("seg_dispatch_to_boundary", "dispatch -> first speakable chunk")
#: HTTP phases, when the run carried the tracer. Each is bounded by two
#: httpcore events, so each is a measurement rather than a share of a bucket.
PHASE_KEYS = [
    ("phase_local_setup", "local setup + serialisation", True),
    ("phase_connect", "DNS + TCP connect", True),
    ("phase_tls", "TLS handshake", True),
    ("phase_upload", "request upload", True),
    ("phase_wait_for_headers", "wait for response headers", False),
    ("phase_dispatch_to_headers", "dispatch -> response headers (sum)", True),
]

EXTRA_SEGMENTS = [
    ("dispatch_to_stream_open", "  of which: transport (dispatch -> response headers)"),
    ("seg_dispatch_to_complete", "dispatch -> generation complete"),
    ("harness_first_token_lag", "  harness overhead on the first token"),
]

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
        "_trials": dicts,
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

    # -- forensics -----------------------------------------------------
    out += _segment_section(spec, analysis)
    out += _threshold_section(spec, analysis)

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


def _segment_section(spec: ExperimentSpec, analysis: dict) -> list[str]:
    """Where the wait to the first speakable chunk actually goes.

    Only rendered when a run carried the instrumentation, so ordinary
    experiments are unaffected.
    """
    trials = [t for t in analysis.get("_trials", []) if t.get("ok")]
    if not any(t["metrics"].get(TOTAL_SEGMENT[0]) is not None for t in trials):
        return []

    out = ["## Latency forensics: the wait to the first speakable chunk", ""]

    for arm in spec.arms:
        mine = [t for t in trials if t["arm"] == arm.name]
        if not mine:
            continue
        totals = [t["metrics"].get(TOTAL_SEGMENT[0]) for t in mine
                  if t["metrics"].get(TOTAL_SEGMENT[0]) is not None]
        total_median = stats_mod.summarise(totals).median if totals else None

        if len(spec.arms) > 1:
            out += [f"### {arm.name}", ""]
        out += ["| segment | median | p95 | min-max | IQR | share |",
                "|---|---|---|---|---|---|"]

        def row(key, label, share=True):
            values = [t["metrics"].get(key) for t in mine
                      if t["metrics"].get(key) is not None]
            summary = stats_mod.summarise(values)
            if not summary:
                return f"| {label} | not recorded | | | | |"
            portion = ""
            if share and total_median:
                portion = f"{summary.median / total_median * 100:.0f}%"
            return (f"| {label} | **{summary.median:.3f}s** | {summary.p95:.3f}s "
                    f"| {summary.minimum:.3f}-{summary.maximum:.3f}s "
                    f"| {summary.iqr:.3f}s | {portion} |")

        for key, label in SEGMENT_KEYS:
            out.append(row(key, label))
        out.append(row(*TOTAL_SEGMENT))
        for key, label in EXTRA_SEGMENTS:
            out.append(row(key, label, share=False))
        out.append("")

        # -- HTTP phases, when they were traced --
        traced = [t for t in mine if t["metrics"].get("http_trace") == "ok"]
        if traced:
            out += ["", "**HTTP phases** (httpcore trace; each bounded by two "
                    "real events)", "",
                    "| phase | median | p95 | min-max | measured? |",
                    "|---|---|---|---|---|"]
            for key, label, direct in PHASE_KEYS:
                values = [t["metrics"].get(key) for t in traced
                          if t["metrics"].get(key) is not None]
                summary = stats_mod.summarise(values)
                if not summary:
                    out.append(f"| {label} | not observed | | | |")
                    continue
                kind = "measured" if direct else "**bucket**"
                out.append(f"| {label} | **{summary.median * 1000:.1f} ms** "
                           f"| {summary.p95 * 1000:.1f} ms "
                           f"| {summary.minimum * 1000:.1f}-{summary.maximum * 1000:.1f} ms "
                           f"| {kind} |")
            reused = [t["metrics"].get("connection_reused") for t in traced]
            hits = sum(1 for r in reused if r is True)
            out += ["", f"Connection reused on **{hits} of {len(reused)}** traced "
                    f"trials." + ("" if hits else " Every trial paid a fresh DNS, "
                    "TCP and TLS setup."), ""]
            unavailable = [t for t in mine if t["metrics"].get("http_trace") == "unavailable"]
            if unavailable:
                out += [f"{len(unavailable)} trial(s) could not be traced; their "
                        f"phases are absent rather than assumed.", ""]

        # -- raw, every trial, so nothing rests on the summary alone --
        out += ["<details><summary>Raw per-trial values</summary>", "",
                "| # | dispatch->first token | ->25 words | ->boundary | total | complete | chunk words |",
                "|---|---|---|---|---|---|---|"]
        for trial in mine:
            m = trial["metrics"]

            def cell(key):
                value = m.get(key)
                return f"{value:.3f}" if isinstance(value, (int, float)) else "-"

            out.append(
                f"| {trial['index']} | {cell('seg_dispatch_to_first_token')} "
                f"| {cell('seg_first_token_to_25_words')} "
                f"| {cell('seg_25_words_to_boundary')} "
                f"| {cell(TOTAL_SEGMENT[0])} | {cell('seg_dispatch_to_complete')} "
                f"| {m.get('first_chunk_words', '-')} |")
        out += ["", "</details>", ""]

        # -- the reading, stated rather than left to be inferred --
        first_token = [t["metrics"].get("seg_dispatch_to_first_token") for t in mine
                       if t["metrics"].get("seg_dispatch_to_first_token") is not None]
        transport = [t["metrics"].get("dispatch_to_stream_open") for t in mine
                     if t["metrics"].get("dispatch_to_stream_open") is not None]
        if first_token and total_median:
            ft = stats_mod.summarise(first_token)
            share = ft.median / total_median * 100
            note = (f"**{share:.0f}% of the wait is over before the first token "
                    f"arrives.**")
            if transport:
                tr = stats_mod.summarise(transport)
                note += (f" Of that, {tr.median:.3f}s is transport - the request "
                         f"reaching the server and its headers coming back - so "
                         f"roughly {ft.median - tr.median:.3f}s is the model "
                         f"before it writes anything.")
            out += [note, ""]
    return out


def _threshold_section(spec: ExperimentSpec, analysis: dict) -> list[str]:
    """What each word threshold would have cost, and what it would have said."""
    trials = [t for t in analysis.get("_trials", []) if t.get("ok")]
    thresholds = next((t["metrics"].get("probe_thresholds") for t in trials
                       if t["metrics"].get("probe_thresholds")), None)
    if not thresholds:
        return []

    out = ["## Chunk thresholds: when speech could have started", ""]

    def summarise_key(key):
        values = [t["metrics"].get(key) for t in trials
                  if t["metrics"].get(key) is not None]
        return stats_mod.summarise(values)

    first_token = summarise_key("seg_dispatch_to_first_token")
    out += ["| threshold | boundary, from first token | p95 | min-max | IQR "
            "| from dispatch | chunk words | rule cost |",
            "|---|---|---|---|---|---|---|---|"]
    for threshold in thresholds:
        boundary = summarise_key(f"boundary_{threshold}_at")
        words = summarise_key(f"boundary_{threshold}_words")
        wait = summarise_key(f"boundary_wait_{threshold}")
        if not boundary:
            out.append(f"| {threshold} words | never reached | | | | | | |")
            continue
        from_dispatch = ("—" if not first_token
                         else f"{first_token.median + boundary.median:.3f}s")
        mark = " *(production)*" if threshold == 25 else ""
        out.append(
            f"| **{threshold} words**{mark} | **{boundary.median:.3f}s** "
            f"| {boundary.p95:.3f}s | {boundary.minimum:.3f}-{boundary.maximum:.3f}s "
            f"| {boundary.iqr:.3f}s | {from_dispatch} "
            f"| {words.median:.0f} | {wait.median:.3f}s |" if words and wait else
            f"| **{threshold} words**{mark} | **{boundary.median:.3f}s** | "
            f"{boundary.p95:.3f}s | {boundary.minimum:.3f}-{boundary.maximum:.3f}s "
            f"| {boundary.iqr:.3f}s | {from_dispatch} | — | — |")
    out.append("")

    # What dropping to a lower threshold would actually buy.
    production = summarise_key(f"boundary_{max(thresholds)}_at")
    if production:
        out += ["**What each threshold would save against the current rule:**", ""]
        for threshold in thresholds:
            if threshold == max(thresholds):
                continue
            boundary = summarise_key(f"boundary_{threshold}_at")
            words = summarise_key(f"boundary_{threshold}_words")
            if not boundary:
                continue
            saved = production.median - boundary.median
            out.append(f"- **{threshold} words**: {saved:+.3f}s earlier, "
                       f"{words.median:.0f} words in the chunk")
        out.append("")

        # Thresholds that land on the same sentence are the same decision.
        # Saying so is the difference between five options and the two or
        # three that actually exist.
        groups: dict = {}
        for threshold in thresholds:
            boundary = summarise_key(f"boundary_{threshold}_at")
            if not boundary:
                continue
            groups.setdefault(round(boundary.median, 2), []).append(threshold)
        tied = [members for members in groups.values() if len(members) > 1]
        if tied:
            for members in tied:
                joined = ", ".join(f"{m}" for m in members)
                out.append(f"- Thresholds {joined} resolve to **the same boundary**, "
                           f"so anything above {members[0]} buys nothing here: "
                           f"the sentence they are all waiting for is the same one.")
            out.append("")

    broken = [t for t in trials if t["metrics"].get("probe_monotonic") is False]
    if broken:
        out += [f"⚠️ {len(broken)} trial(s) recorded a lower threshold closing "
                f"*later* than a higher one. That is impossible under the rule, "
                f"so treat these numbers as suspect until it is explained.", ""]

    cost = summarise_key("probe_seconds")
    if cost:
        out += [f"*Probe overhead: {cost.median * 1000:.2f} ms median across the "
                f"whole stream — small enough not to move what it watches.*", ""]

    # -- the openings themselves --------------------------------------
    out += ["### What each threshold would have said", "",
            "*One trial shown per threshold; every trial's text is in "
            "`trials.jsonl` and `artifacts/candidates.md`.*", ""]
    sample = next((t for t in trials if t.get("threshold_texts")), None)
    if sample:
        for threshold in thresholds:
            text = (sample.get("threshold_texts") or {}).get(str(threshold))
            if not text:
                continue
            out += [f"**{threshold} words** ({len(text.split())} words):",
                    "", f"> {text}", ""]
    return out


def candidates_markdown(spec: ExperimentSpec, trials: list[dict]) -> str:
    """Every threshold's opening from every trial, grouped for reading.

    Latency is in the tables; this is the half that decides whether an earlier
    threshold is acceptable, and it is only judgeable by reading.
    """
    thresholds = next((t["metrics"].get("probe_thresholds") for t in trials
                       if t["metrics"].get("probe_thresholds")), [])
    lines = [f"# Candidate openings — {spec.name}", "",
             "Grouped by threshold. Read down a section to see whether that "
             "threshold's openings hold up as FAM writing, not only whether "
             "they arrive sooner.", ""]
    for threshold in thresholds:
        lines += [f"## {threshold} words", ""]
        for trial in trials:
            text = (trial.get("threshold_texts") or {}).get(str(threshold))
            if not text:
                continue
            lines.append(f"{trial['index']:>3}. ({len(text.split())}w) {text}")
        lines.append("")
    return "\n".join(lines)


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
