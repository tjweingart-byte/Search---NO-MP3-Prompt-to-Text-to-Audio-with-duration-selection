# What each listener costs

Anthropic bills this organisation. Exa bills this key. The GPU bills by the
hour. **None of them can say which listener produced which request** — so if
that question is not answered at the moment of spend it cannot be answered
later, and every question about pricing, per-user limits and abuse is that
question wearing a different hat.

One row per episode, written where the listener id is known, appended and never
updated.

```sh
python tools/usage_report.py                 # the last 30 days
python tools/usage_report.py --days 7 --price 4.99
python tools/usage_report.py --flagged       # who to look at, and why
python tools/usage_report.py --json          # the same numbers, for a sheet
```

The same report is `GET /api/usage` for a dashboard, so a spreadsheet and a
person at a terminal cannot disagree about what a month is.

---

## Reading the report

It prints four things in this order, and the order is the argument.

### 1. What was billed — real, marginal, per listener

Claude tokens (input, output, cache reads and writes at the published
multipliers) and Exa searches, from the providers' own usage figures rather
than estimated from the script. A word count could have guessed output tokens;
it could never have seen the prompt, the examples, an evidence packet, or a
cache read — and it would have been wrong in the direction that flatters the
bill.

### 2. How that is distributed — median, p90, p99, max

The number everyone asks for is the mean, and it is the one most likely to be
wrong: the distribution is not symmetrical. On a measured synthetic run the
worst listener cost **68× the median**. That ratio, not the mean, is what
decides whether a flat price needs a usage cap behind it, so it is printed
next to it.

Broken down by **plan** (free against paid) and by **surface** — Explore
replays and never writes a script, so an Explore-heavy listener costs a
fraction of a search-heavy one.

### 3. What the machine costs anyway — fixed, and nothing to do with listeners

Chatterbox runs in-process on a GPU that costs the same whether it is
synthesising or idle. Two consequences, both counterintuitive:

* **Marginal audio cost is almost nothing.** Synthesis runs at ~330× realtime,
  so a three-minute episode is under a second of card — about $0.0001. This is
  the arithmetic the prefetch plan rests on, and the report shows it holding.
* **The real GPU cost is a floor that exists before the first listener.** It is
  reported beside the marginal total and **never folded into a per-listener
  average**, because an average that includes it says more about how many
  listeners there are than about what a listener costs.

### 4. What that implies for a price

`--price 4.99` prints the margin at the mean, the p99 and the worst listener,
and the listener count at which the fixed floor is covered. Marked as an
estimate, because it is one.

---

## Every number is billed, priced, or assumed

The report says which, on the line where it appears.

| | means | example |
|---|---|---|
| **billed** | the provider's own usage figures | Claude's `usage`, Exa's reported `cost_dollars` |
| **priced** | billed quantities × a published rate | tokens × `metering.PRICES` |
| **assumed** | rests on configuration, not an invoice | the GPU allocation, the cache saving |

A model with no entry in `PRICES` is counted as **unpriced** and named, never
costed at zero. A silent $0 is the same failure shape this project keeps paying
for: a number that looks like an answer.

The cost is **stored at write time**, not recomputed. Prices change, and a row
that repriced itself would make "what did March cost" depend on when you ask.

---

## The one line that gets better with scale

Two people who ask the same thing pay for one script. Reporting only what was
spent would make the shared cache invisible, so the report estimates what the
hits avoided — hits × the mean writing cost of a miss, labelled assumed.

It matters for forecasting rather than for accounting: a projection built from
today's per-episode cost **overstates** tomorrow's bill, because hit rate rises
with listeners.

---

## Abuse

`--flagged` (or `?flagged=1`) runs two thresholds over the last hour, because
the two abuses look different:

* **volume** — a script hammering the endpoint: many cheap requests.
* **spend** — 10-minute researched episodes all afternoon: few requests, a lot
  of money.

Either alone misses the other.

**It reports; it never acts.** An automatic block on a metering heuristic
eventually locks out a real listener who has no way to tell anyone.
`_rate_limit` remains the thing that actually paces requests, and a test
asserts this module never grows a `ban`/`block`/`suspend` verb.

---

## Who can read it

`/api/usage` is every listener's history of what they asked for and what it
cost — the most sensitive thing this app stores after the password hashes, and
unlike those it is meant to be read.

```sh
FAM_ADMIN_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
```

Unset, the endpoint **does not exist** (404, not 401: an unconfigured
deployment should not advertise that it has a billing endpoint, and a wrong
token should not confirm the path). The comparison is constant-time. The token
belongs in the secrets provider with the others — see `CREDENTIALS.md`.

`tools/usage_report.py` reads the database directly and needs no token, because
anyone who can read the file has already won.

---

## Paid against unpaid

`accounts` carries a `plan` column, default `free`.

**Nothing in the app sets it to `paid`** — there is no payment route, and
CLAUDE.md's open question about what an account should *entitle* you to is
still open. The column exists so the split is available from the day something
does take payment, rather than being backfilled out of a log that never
recorded it. `ACCOUNTS.set_plan(user_id, "paid")` is the whole interface.

The plan is **stamped on each row at write time**. Someone who upgrades on the
20th did not cost paid-plan money on the 5th, and joining to today's plan would
rewrite what free users cost.

An anonymous listener is `free`, not `unknown`: they are a real listener costing
real money, and a third bucket would answer "what does a free user cost" with a
number that excluded most of them.

---

## Configuration

| variable | default | what it changes |
|---|---|---|
| `METERING_DB` | `metering.db` | where the ledger lives. **Not regenerable** — pin it to the mounted disk, as both Dockerfiles do |
| `FAM_ADMIN_TOKEN` | unset | gates `/api/usage`; unset means the endpoint 404s |
| `GPU_USD_PER_HOUR` | `0.60` | mid L4 on-demand. A reserved card or a neocloud is cheaper |
| `SYNTHESIS_REALTIME_FACTOR` | `330` | how much faster than realtime Chatterbox synthesises |
| `GPU_HOURS_PER_DAY` | `24` | hours the card is actually paid for |

Rates live in `metering.PRICES`, checked against the published card on
2026-09-09. Update them there when the card changes; past rows keep what they
cost.

---

## What this still does not do

* **It does not enforce anything.** No quota, no per-plan limit, no cutoff. The
  data to build one now exists; the policy does not, and inventing one here
  would be inventing a product decision.
* **It does not bill anyone.** There is no payment processor, no invoice, no
  Stripe. `plan` is a label.
* **It does not track bandwidth.** 2.65 MB/min uncompressed per listener is a
  real cost at scale (CLAUDE.md) and is not in these numbers. `audio_seconds`
  is the quantity it would be computed from when there is a CDN bill to
  compare against.
* **It has never run against a real bill.** Every figure in this document comes
  from the published rate card and a synthetic ledger. The first month of real
  usage should be reconciled against the Anthropic and Exa invoices, and this
  file corrected where they disagree.
