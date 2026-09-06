# First-token latency on a warm client — analysis

**Status of the numbers below: reported from the run, not yet reread from
`trials.jsonl` in this checkout.** The run happened on the author's Mac; the
run data is git-ignored there until `tools/preserve_run.py` promotes it. The
medians in the first table were transcribed from the report and are recorded
as such. Everything in the diagnosis section is derived from the code in this
repository and is independently checkable here.

Once the run is preserved, the numbers below can be recomputed from
`trials.jsonl` and this heading replaced.

## What was run

120 trials: 5 arms × 6 topics × 4 trials, `claude-sonnet-5`, one pooled client
shared by every arm, 25-word chunk rule, no TTS. Each arm differs from the
control by exactly one request key, verified against the real outgoing request.

## Reported medians — time to first speakable chunk

| arm | change from control | median |
|---|---|---|
| A-control | — | ~2.45s |
| B-thinking-off | `thinking` disabled | ~2.10s |
| C-effort-low | `output_config.effort = low` | ~1.98s |
| D-first-sentence | one line added to the system prompt | ~2.47s |
| E-max-tokens-96 | cap 220 → 96 | ~2.06s |

**The report found no arm measurably faster at this sample size, and that
finding stands.** With 24 trials per arm against model-latency variance, a
0.3–0.5s median gap is inside the noise. Nothing here is a winner.

What can be said without overreaching: the five arms fall into two groups —
A and D near 2.45s, and B, C and E near 2.0s. B, C and E have nothing in
common except that each, by a different mechanism, gives the model less to do
before it starts emitting text. D changes the prompt without changing that,
and lands with the control. That is a coherent story and it is **a hypothesis
to test, not a result**. The next run should test the grouping directly with
enough trials to resolve a ~0.4s difference, rather than re-running five arms.

## Design flaw to fix in the next run

Arms execute in fixed order A, B, C, D, E inside every (trial, topic) cycle
(`harness.run_all`). Arm position is therefore perfectly confounded with arm
identity: any systematic effect of going first — server-side or local — lands
entirely on A-control, which is one of the two slow arms. This does not
invalidate the run, because no difference was claimed. It would invalidate a
run that did claim one. **Randomise or rotate arm order within each cycle
before treating any arm gap as real.**

## The "control reused a connection on 96% of trials" warning

**A reporting bug. It does not compromise the comparison — it is evidence the
comparison was sound.**

The report's reuse section was written for the earlier connection-reuse A/B,
which had exactly two arms: one building a fresh client per request, one
pooled. It identifies the control as `min(reuse_rate)` and the treatment as
`max(reuse_rate)`, then warns if the "control" reused a connection, on the
assumption that a fresh-client arm never should.

In this experiment **every** arm is pooled onto one client (`pool_key:
"warm_first_token"`, keep-alive 300s). There is no fresh-client arm, so
`min(reuse_rate)` picks whichever arm happens to be lowest — and the arithmetic
of that is forced:

* `client_pool.acquire` builds one client per pool key, so the whole run has
  exactly **one** connection to open.
* `harness.run_all` iterates trial → query → arm, so the very first request of
  the sweep is (trial 1, topic 1, **A-control**). That request pays the single
  handshake.
* A-control therefore reports 23 of 24 reused = **95.8% ≈ 96%**; every other
  arm reports 24 of 24 = 100%.

96% is not a defect. It is the signature of exactly one connection serving all
120 trials, seen from the arm that went first. The warning inverted it into
"the control should never do this" and told you to treat the comparison as
unsound. The opposite is true: transport was held constant, and the ~0.4s
grouping above is generation-side, whatever else it is.

**Fixed** in `experiments/report.py`: when every arm is pooled, the section is
now a transport *check* rather than a reuse A/B. It counts connections opened
across the run, calls one connection the intended outcome, and only warns when
the count is higher — which would mean the pool expired mid-run and part of an
arm gap really is handshake cost. Three tests pin it, including one that keeps
the warning working for a genuine fresh-vs-pooled A/B.

No paid work was re-run. The report can be regenerated from the preserved run
data for free.

## What still needs preserving

`tools/preserve_run.py --latest --as warm_first_token` writes, into this
folder: `trials.jsonl` (per-trial first-token and first-chunk segments),
`report.md`, and `openings_by_arm.md` — the text each arm actually wrote. The
openings are also the input corpus for the Chatterbox benchmark, which is
built to read them rather than to use invented sentences.
