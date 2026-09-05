"""The FAM Experiment Engineer.

Describe an experiment in plain English; get a plan, a cost, and - after you
say yes - a persistent report.

    python tools/experiment.py plan "exa vs current search, 10 trials"
    python tools/experiment.py run  "full exa to chatterbox pipeline 10 times"
    python tools/experiment.py run  "..." --simulate     # no key, no spend
    python tools/experiment.py list
    python tools/experiment.py show <run-id>
    python tools/experiment.py compare <run-a> <run-b>

Three things this will not do, by construction:

* spend anything without printing the estimate and asking first (`--yes` skips
  the question, never the estimate);
* start, stop or pay for GPU infrastructure - an experiment needing a GPU that
  is not running stops and says so;
* write a credential anywhere, including into its own reports.
"""
from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments import cost as cost_mod           # noqa: E402
from experiments import nl, registry, report, store  # noqa: E402
from experiments.adapters.base import InfrastructureRequired  # noqa: E402
from experiments.harness import Harness            # noqa: E402
from experiments.spec import ExperimentSpec        # noqa: E402

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"


def _load_spec(text: str, queries_file: str | None, trials: int | None,
               minutes: float | None) -> tuple[ExperimentSpec, list[str]]:
    """Either a JSON spec file, or English."""
    path = pathlib.Path(text)
    if path.suffix == ".json" and path.exists():
        spec = ExperimentSpec.from_json(path.read_text(encoding="utf-8"))
        assumptions = [f"Loaded verbatim from {path}."]
    else:
        parsed = nl.compile_request(text)
        spec, assumptions = parsed.spec, parsed.assumptions
    if queries_file:
        lines = [l.strip() for l in pathlib.Path(queries_file).read_text().splitlines()]
        spec.queries = [l for l in lines if l and not l.startswith("#")]
        assumptions.append(f"Queries from {queries_file} ({len(spec.queries)}).")
    if trials:
        spec.trials = trials
    if minutes:
        spec.minutes = minutes
    return spec, assumptions


def _print_plan(spec: ExperimentSpec, assumptions: list[str], simulate: bool) -> tuple:
    print(f"\n{BOLD}{spec.name}{RESET}\n")
    print(f"  {'kind':<10} {spec.kind}")
    print(f"  {'trials':<10} {spec.trials} per arm per query  ->  {spec.total_trials} total")
    print(f"  {'episode':<10} {spec.minutes:g} min")
    print(f"  {'queries':<10} {len(spec.queries)}")
    print(f"\n  {BOLD}arms{RESET}")
    for arm in spec.arms:
        model = arm.model or "(config default)"
        print(f"    {arm.name:<24} search={arm.search:<22} tts={arm.tts:<12} model={model}")
        if arm.params:
            print(f"    {'':<24} params={arm.params}")

    problems = spec.validate()
    if problems:
        print(f"\n  {BOLD}spec problems{RESET}")
        for item in problems:
            print(f"    - {item}")

    if assumptions:
        print(f"\n  {BOLD}assumptions{RESET}{DIM}  (correct these by editing the spec){RESET}")
        for item in assumptions:
            print(f"    ~ {item}")

    print(f"\n  {BOLD}availability{RESET}")
    pre = registry.preflight(spec) if not simulate else None
    print("    simulated run: adapters replaced by stand-ins" if simulate else pre.render())

    estimate = cost_mod.estimate(spec)
    print(f"\n  {BOLD}estimated cost{RESET}")
    print("    $0.000   nothing is called in a simulated run" if simulate
          else "\n".join("  " + l for l in estimate.render().splitlines()))
    print()
    return pre, estimate, problems


def cmd_plan(args) -> int:
    spec, assumptions = _load_spec(args.request, args.queries, args.trials, args.minutes)
    _print_plan(spec, assumptions, args.simulate)
    if args.save:
        pathlib.Path(args.save).write_text(spec.to_json(), encoding="utf-8")
        print(f"  spec written to {args.save}\n")
    return 0


def cmd_run(args) -> int:
    spec, assumptions = _load_spec(args.request, args.queries, args.trials, args.minutes)
    pre, estimate, problems = _print_plan(spec, assumptions, args.simulate)

    if problems:
        print("  Refusing to run: fix the spec problems above.\n")
        return 2

    if not args.simulate:
        if pre.approvals:
            print(f"  {BOLD}This experiment requires infrastructure that is not running.{RESET}")
            for item in pre.approvals:
                print(f"    {item}")
            print("\n  Nothing has been started and nothing has been spent. This tool does")
            print("  not create or start paid GPU infrastructure. Start it yourself and")
            print("  set the endpoint, or run with --simulate to exercise the harness.\n")
            return 3
        if pre.blockers:
            print("  Refusing to run: the adapters above are not available.\n")
            return 4
        if not args.yes:
            reply = input(f"  Spend about ${estimate.total:.3f} on {spec.total_trials} trials? [y/N] ")
            if reply.strip().lower() not in ("y", "yes"):
                print("  Nothing run, nothing spent.\n")
                return 1

    run = store.create(spec)
    print(f"  {DIM}recording to {run.path}{RESET}\n")

    if args.simulate:
        from experiments.fakes import FakeGenerator, FakeSearch, FakeTTS

        delays = {"exa": 0.25, "anthropic_web_search": 0.0, "none": 0.0}

        def fake_search(name):
            adapter = FakeSearch(delays.get(name, 0.1), adapter_id=name)
            adapter.separable = name != "anthropic_web_search"
            return adapter

        def fake_tts(name, voice=None):
            speed = {"chatterbox": 0.35, "piper": 0.05, "none": 0.0}.get(name, 0.1)
            return FakeTTS(speed, adapter_id=name,
                           host="runpod-gpu" if name == "chatterbox" else "local")

        harness = Harness(spec, run=run,
                          generator_factory=lambda: FakeGenerator(first_token_delay=0.6),
                          search_factory=fake_search, tts_factory=fake_tts,
                          save_audio=args.save_audio)
    else:
        harness = Harness(spec, run=run, save_audio=args.save_audio)

    done = [0]

    def progress(result):
        done[0] += 1
        mark = "ok " if result.ok else "FAIL"
        first = result.metrics.get("first_audio") or result.metrics.get("first_chunk")
        shown = f"{first:.2f}s" if isinstance(first, float) else "-"
        print(f"    [{done[0]:>3}/{spec.total_trials}] {mark} {result.arm:<24} {shown}")
        if not result.ok:
            print(f"          {result.error}")

    try:
        results = asyncio.run(harness.run_all(progress=progress))
    except InfrastructureRequired as exc:
        print(f"\n  {BOLD}Stopped: infrastructure required{RESET}\n    {exc}\n")
        print(f"  Partial trials kept in {run.path}\n")
        return 3
    except KeyboardInterrupt:
        print(f"\n  Interrupted. {len(harness.results)} trials kept in {run.path}\n")
        return 130

    analysis = report.analyse(spec, results)
    previous_run = store.find_previous(spec.name, exclude=run.id)
    previous = previous_run.summary() if previous_run else None
    markdown = report.render(spec, analysis, previous=previous)

    storable = {k: v for k, v in analysis.items() if not k.startswith("_")}
    run.write_summary(storable)
    run.write_report(markdown)

    leaks = run.verify_clean()
    if leaks:
        print(f"  {BOLD}WARNING: possible credential in {', '.join(leaks)}{RESET}\n")

    print("\n" + markdown)
    print(f"\n{DIM}saved: {run.path}{RESET}\n")
    return 0


def cmd_list(args) -> int:
    runs = store.list_runs()
    if not runs:
        print("\n  No experiments recorded yet.\n")
        return 0
    print(f"\n  {BOLD}{'run':<44} {'arms':<28} trials{RESET}")
    for run in runs[: args.limit]:
        spec = run.spec_dict() or {}
        arms = ", ".join(a["name"] for a in spec.get("arms", []))[:26]
        print(f"  {run.id:<44} {arms:<28} {len(run.trials())}")
    print()
    return 0


def cmd_show(args) -> int:
    run = store.load(args.run_id)
    if not run:
        print(f"\n  No such run: {args.run_id}\n")
        return 1
    text = run.report()
    print("\n" + (text or "  (no report; the run may not have finished)") + "\n")
    return 0


def cmd_compare(args) -> int:
    a, b = store.load(args.run_a), store.load(args.run_b)
    if not a or not b:
        print("\n  One of those runs does not exist.\n")
        return 1
    print(f"\n  {BOLD}{a.id}  vs  {b.id}{RESET}\n")
    a_arms = {x["name"]: x for x in (a.summary() or {}).get("arms", [])}
    b_arms = {x["name"]: x for x in (b.summary() or {}).get("arms", [])}
    shared = [n for n in a_arms if n in b_arms]
    if not shared:
        print("  These runs share no arm names, so there is nothing to compare.\n")
        return 1
    print(f"  {'arm':<24} {'before':>10} {'after':>10} {'change':>12}")
    for name in shared:
        first = (a_arms[name].get("headline") or {}).get("median")
        second = (b_arms[name].get("headline") or {}).get("median")
        if first is None or second is None:
            print(f"  {name:<24} {'-':>10} {'-':>10} {'no data':>12}")
            continue
        delta = second - first
        print(f"  {name:<24} {first:>9.2f}s {second:>9.2f}s {delta:>+11.2f}s")
    print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="experiment", description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)

    def common(sub):
        sub.add_argument("request", help="plain English, or a path to a spec .json")
        sub.add_argument("--queries", help="file of questions, one per line")
        sub.add_argument("--trials", type=int, help="override the trial count")
        sub.add_argument("--minutes", type=float, help="override the episode length")
        sub.add_argument("--simulate", action="store_true",
                         help="run with stand-in adapters: no key, no GPU, no spend")

    p_plan = subs.add_parser("plan", help="show the plan and cost; run nothing")
    common(p_plan)
    p_plan.add_argument("--save", help="write the compiled spec to this .json")
    p_plan.set_defaults(func=cmd_plan)

    p_run = subs.add_parser("run", help="run the experiment and save a report")
    common(p_run)
    p_run.add_argument("--yes", action="store_true", help="skip the spend confirmation")
    p_run.add_argument("--save-audio", action="store_true", help="keep generated PCM")
    p_run.set_defaults(func=cmd_run)

    p_list = subs.add_parser("list", help="every experiment recorded so far")
    p_list.add_argument("--limit", type=int, default=25)
    p_list.set_defaults(func=cmd_list)

    p_show = subs.add_parser("show", help="print a saved report")
    p_show.add_argument("run_id")
    p_show.set_defaults(func=cmd_show)

    p_cmp = subs.add_parser("compare", help="two saved runs, side by side")
    p_cmp.add_argument("run_a")
    p_cmp.add_argument("run_b")
    p_cmp.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
