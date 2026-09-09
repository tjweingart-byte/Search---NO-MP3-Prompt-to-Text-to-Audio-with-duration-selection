"""What the product cost, who cost it, and what that implies for a price.

    python tools/usage_report.py                  the last 30 days
    python tools/usage_report.py --days 7
    python tools/usage_report.py --json           the same numbers, for a sheet
    python tools/usage_report.py --flagged        who to look at, and why
    python tools/usage_report.py --price 4.99     what a monthly price would earn

Reads the ledger `app.py` writes. Makes no model call and needs no API key, so
it runs anywhere the database is - including against a copy pulled off the
server.

**Why the layout is what it is.** A single "average cost per user" is the
number everyone asks for and the one most likely to be wrong: it is a mean over
a distribution that is not symmetrical, sitting on top of a fixed cost that has
nothing to do with users at all. So this prints, separately and in this order:

  1. what was actually billed        - marginal, real, per listener
  2. how that is distributed         - median against p99 against the worst one
  3. what the machine costs anyway   - fixed, and unaffected by any of the above
  4. what a price would have to be   - the two put together, at a stated
                                       listener count, marked as an estimate

Anything that rests on an assumption says so on the line where it appears.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import metering  # noqa: E402

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"


def money(value: float) -> str:
    """Dollars at a precision that does not lie about how small they are.

    An episode costs about a cent and a half; printed to two places most of
    this report would read $0.02, $0.02, $0.02 and the differences that matter
    - between a median listener and the worst one - would vanish into rounding.
    """
    if value and abs(value) < 0.01:
        return f"${value:.5f}"
    if abs(value) < 10:
        return f"${value:.4f}"
    return f"${value:,.2f}"


def price_str(value: float) -> str:
    """A price the operator typed, printed the way they typed it.

    `money` deliberately shows tiny costs to five places; a *price* is a round
    number a person chose, and "$4.9900/month" reads like a rounding error in
    something rather than a decision.
    """
    return f"${value:,.2f}"


def bar(value: float, peak: float, width: int = 24) -> str:
    if peak <= 0:
        return ""
    return "█" * max(1, int(round(value / peak * width))) if value else ""


def render(report: dict, price: float = 0.0, flagged: list | None = None) -> str:
    out: list[str] = []
    say = out.append
    totals, spread = report["totals"], report["per_listener"]
    days = max(report["window"]["days"], 1 / 24)

    say(f"\n{BOLD}FAM usage and cost{RESET}  ·  "
        f"{days:.1f} days  ·  {totals['episodes']} episodes  ·  "
        f"{totals['listeners']} listeners")

    if not totals["episodes"]:
        say(f"\n  {BOLD}The ledger is empty for this window.{RESET}")
        say(f"{DIM}  Nothing has been metered yet, which is not the same as nothing")
        say(f"  having cost anything. Play an episode and run this again.{RESET}\n")
        return "\n".join(out)

    # --- 1. what was billed ------------------------------------------------
    say(f"\n{BOLD}Billed{RESET}  {DIM}provider usage at published rates{RESET}")
    say(f"  total                {money(totals['cost_usd'])}"
        f"   ({money(totals['cost_usd'] / days)}/day)")
    say(f"  per episode          {money(totals['cost_per_episode'])}")
    say(f"  per listener         {money(spread['cost_usd']['mean'])}"
        f"   over {days:.1f} days")
    b = report["breakdown"]
    peak = max(b.values()) if b else 0
    for label, key in (("Claude", "claude_usd"), ("Exa", "exa_usd"),
                       ("GPU (marginal)", "gpu_marginal_usd")):
        share = b[key] / totals["cost_usd"] * 100 if totals["cost_usd"] else 0
        say(f"    {label:<18} {money(b[key]):>12}  {share:5.1f}%  {bar(b[key], peak)}")
    if report["unpriced_episodes"]:
        say(f"  {BOLD}{report['unpriced_episodes']} episode(s) on a model with no "
            f"published price{RESET} - "
            f"{', '.join(report['unpriced_models']) or 'unnamed'}")
        say(f"{DIM}    Counted, not costed. Add it to metering.PRICES.{RESET}")

    # --- 2. the distribution ----------------------------------------------
    say(f"\n{BOLD}Per listener{RESET}  {DIM}the shape, not just the middle{RESET}")
    say(f"  {'':<14}{'median':>11}{'mean':>11}{'p90':>11}{'p99':>11}{'max':>11}")
    c, e = spread["cost_usd"], spread["episodes"]
    say(f"  {'cost':<14}{money(c['median']):>11}{money(c['mean']):>11}"
        f"{money(c['p90']):>11}{money(c['p99']):>11}{money(c['max']):>11}")
    say(f"  {'episodes':<14}{e['median']:>11.1f}{e['mean']:>11.1f}"
        f"{e['p90']:>11.1f}{e['p99']:>11.1f}{e['max']:>11.0f}")
    a = spread["audio_seconds"]
    say(f"  {'audio (min)':<14}{a['median'] / 60:>11.1f}{a['mean'] / 60:>11.1f}"
        f"{a['p90'] / 60:>11.1f}{a['p99'] / 60:>11.1f}{a['max'] / 60:>11.1f}")
    if c["median"] > 0:
        say(f"{DIM}  The worst listener costs {c['max'] / c['median']:.0f}x the median. "
            f"That ratio, not the mean,{RESET}")
        say(f"{DIM}  is what decides whether a flat price needs a usage cap "
            f"behind it.{RESET}")

    say(f"\n{BOLD}Most expensive listeners{RESET}")
    top_cost = report["top_listeners"][0]["cost_usd"] if report["top_listeners"] else 0
    for u in report["top_listeners"]:
        say(f"  {u['user_id'][:12]:<14}{u['plan']:<6}{u['episodes']:>4} ep "
            f"{money(u['cost_usd']):>11}   {bar(u['cost_usd'], top_cost, 18)}")

    # --- paid against unpaid ----------------------------------------------
    say(f"\n{BOLD}By plan{RESET}")
    say(f"  {'':<8}{'listeners':>11}{'episodes':>10}{'cost':>13}{'per listener':>15}")
    for plan, row in report["by_plan"].items():
        say(f"  {plan:<8}{row['listeners']:>11}{row['episodes']:>10}"
            f"{money(row['cost_usd']):>13}{money(row['cost_per_listener']):>15}")
    if not report["by_plan"]["paid"]["listeners"]:
        say(f"{DIM}  Every listener is on 'free' because nothing in the app takes "
            f"payment yet.{RESET}")
        say(f"{DIM}  The column is live: accounts.set_plan(user, 'paid') fills it "
            f"from that point on.{RESET}")

    # --- the cache, which gets cheaper with scale --------------------------
    cache = report["cache"]
    say(f"\n{BOLD}Shared script cache{RESET}")
    say(f"  {cache['hits']} hits / {cache['hits'] + cache['misses']} episodes "
        f"({cache['hit_rate'] * 100:.0f}%)   saved ~{money(cache['estimated_saving_usd'])}")
    say(f"{DIM}  Estimated: hits times the mean writing cost of a miss. This is the "
        f"one cost line{RESET}")
    say(f"{DIM}  that improves as listeners are added - two people asking the same "
        f"thing pay once.{RESET}")

    # --- 3. the fixed floor ------------------------------------------------
    fixed = report["fixed"]
    say(f"\n{BOLD}Fixed{RESET}  {DIM}assumed - the machine, not the listeners{RESET}")
    say(f"  GPU over this window {money(fixed['gpu_usd'])}"
        f"   ({fixed['gpu_hours']:.0f}h at {money(fixed['usd_per_hour'])}/h)")
    say(f"{DIM}  Chatterbox runs in-process, so this is paid whether or not anyone "
        f"listens.{RESET}")
    say(f"{DIM}  Deliberately not divided into the per-listener numbers above: doing "
        f"that would say{RESET}")
    say(f"{DIM}  more about how many listeners there are than about what one "
        f"costs.{RESET}")

    # --- 4. what it implies for a price ------------------------------------
    total_cost = totals["cost_usd"] + fixed["gpu_usd"]
    per_month = 30.0 / days
    say(f"\n{BOLD}All in{RESET}  {DIM}billed + assumed{RESET}")
    say(f"  this window          {money(total_cost)}")
    say(f"  extrapolated /month  {money(total_cost * per_month)}"
        f"{DIM}   at this rate of use{RESET}")
    if totals["listeners"]:
        say(f"  per listener /month  "
            f"{money(total_cost * per_month / totals['listeners'])}")
    if price:
        marginal = spread["cost_usd"]["mean"] * per_month
        p99 = spread["cost_usd"]["p99"] * per_month
        worst = spread["cost_usd"]["max"] * per_month
        say(f"\n{BOLD}At {price_str(price)}/month{RESET}  {DIM}estimate - marginal cost "
            f"only, fixed cost recovered separately{RESET}")
        say(f"  mean listener        {money(price - marginal)} margin "
            f"({(1 - marginal / price) * 100:5.1f}%)")
        say(f"  p99 listener         {money(price - p99)} margin "
            f"({(1 - p99 / price) * 100:5.1f}%)")
        say(f"  worst listener       {money(price - worst)} margin "
            f"({(1 - worst / price) * 100:5.1f}%)")
        if fixed["gpu_usd"] > 0:
            need = fixed["gpu_usd"] * per_month / max(price - marginal, 1e-9)
            if need > 0:
                say(f"  {BOLD}break even at {need:,.0f} listeners{RESET}"
                    f"{DIM}   covering the GPU at the mean margin{RESET}")

    if flagged:
        say(f"\n{BOLD}Worth a look{RESET}  {DIM}thresholds, not verdicts - nothing is "
            f"blocked on this{RESET}")
        for u in flagged:
            say(f"  {u['user_id'][:12]:<14}{u['plan']:<6}{money(u['cost_usd']):>11}"
                f"   {'; '.join(u['reasons'])}")
    elif flagged is not None:
        say(f"\n{BOLD}Worth a look{RESET}  nobody crossed a threshold in the last hour.")

    say("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=float, default=30.0)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--price", type=float, default=0.0,
                    help="A monthly price to test the margin of")
    ap.add_argument("--flagged", action="store_true",
                    help="Also run the abuse thresholds over the last hour")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--db", default="", help="A ledger other than this machine's")
    args = ap.parse_args()

    store = metering.MeterStore(args.db or None)
    now = time.time()
    report = store.report(since=now - args.days * 86400, until=now, top=args.top)
    flagged = metering.suspects(store) if args.flagged else None
    if args.json:
        if flagged is not None:
            report["flagged"] = flagged
        print(json.dumps(report, indent=2))
        return 0
    print(render(report, price=args.price, flagged=flagged))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
