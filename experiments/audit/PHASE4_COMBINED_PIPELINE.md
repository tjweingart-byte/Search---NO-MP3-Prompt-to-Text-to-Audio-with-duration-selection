# Phase 4 — Chatterbox Base alone, and the whole FAM request, on one 4090

**Preserved before the pod is rented.** Everything here can be read, argued
with and changed while the meter is off.

Two experiments, one rental, results kept apart:

* **A. Chatterbox Base alone.** Cold model load, the first generation after
  that load, then warm generations across the short / medium / long buckets of
  the real first-chunk corpus. Audio duration, wall-clock, realtime factor,
  GPU memory.
* **B. The whole request.** `user request → Exa → Claude → Chatterbox Base →
  playable audio`, on one monotonic clock, cold and then immediately warm.

They share a pod because the expensive part of a GPU experiment is the twenty
minutes of setup, not the two minutes of measurement.

## What this is not

**Not production latency.** One machine, one request at a time, no concurrency,
no queueing, no other tenant. Every artefact this writes carries
`"is_production_latency": false` and the string *RTX 4090 / DEVELOPMENT
BENCHMARK* so a number lifted out of it arrives with its caveat attached.

**Not an optimisation.** No T3 work, no streaming rewrite, no production
tuning. Nothing in `script_generator.py`, `search.py` or the app changes.
`experiments/generate.py` already runs production's own request builder behind
a `dataclasses.replace` that is restored in a `finally`, which is the only
touch this takes on production behaviour and is the touch every earlier
experiment took.

**Not a Turbo comparison.** Turbo's numbers stay where they are. This writes to
a fresh timestamped directory and refuses to start if that directory already
holds the section it is about to write.

## Why A and B are separate processes

There is exactly one cold model load per process, and both experiments need
one. Running them in a single process would force one of them to report a
composed number — cold load plus a warm run, added up — and a composed number
in a latency waterfall is the kind of thing that gets quoted later without its
adjective.

So `tools/pod_combined.sh` invokes the runner twice, `--experiment a` then
`--experiment b`, into the same `--out` directory. The second load costs about
ten seconds of GPU time. That is the cheapest honesty available.

In B, the load is marked inside the run (`model_load_start` →
`model_load_complete`) and appears as its own waterfall row, before
`request_start`. So `search_to_first_listen` measures what a listener on a warm
server waits, and `cold_start_to_first_listen` measures the same thing with the
load included. Both are measured; neither is inferred from the other.

## The clock

`experiments/pipeline_probe.py`. One `time.perf_counter()` origin per run, named
marks hung off it, spans computed between marks that were actually recorded.

```
process_start
model_load_start ─────────────┐ cold run only
model_load_complete ──────────┘
request_start
exa_request_start ────────────┐ Exa retrieval
exa_complete ─────────────────┘
claude_request_start ─────────┐ prompt assembly is the gap above this
claude_first_token ───────────┤ Claude TTFT
first_speakable_chunk         │
claude_complete ──────────────┘ Claude total
tts_start ────────────────────┐ Chatterbox synthesis
first_playable_audio          │
audio_complete ───────────────┘
```

**Marks are recorded, never inferred.** A stage that cannot be observed is
written down as unavailable *with the reason*, because a plausible number in a
waterfall is worse than a gap: the gap gets investigated and the number gets
believed. `waterfall()` gives such a row `seconds: null` and excludes it from
the percentage share rather than letting the other rows absorb it.

Two stages are known unavailable before the run, and are declared up front:

| stage | why it cannot be marked |
|---|---|
| first Exa result | `exa_impl.run_search` makes one blocking `search_and_contents` call. There is no observable point between request and complete. |
| first playable audio, mid-generation | Chatterbox Base is one-shot — `generate()` returns a finished waveform and the package yields nothing. The first playable moment **is** generation-complete for the first chunk. Phase 1 established this; it is the finding, not a limitation of the probe. |

The second is why `first_playable_audio` and `audio_complete` sit a hair apart
rather than seconds apart. That gap is the prize a streaming engine would win,
and this run measures its absence precisely.

## What is spoken

The **first speakable chunk**, not the whole script: the first sentence ending
that leaves at least 25 words, which is the rule the whole benchmark corpus was
built under. Synthesising the entire three-minute script would measure a thing
the product does not do.

If the chunk rule never fires — a script with no sentence ending past the word
floor — the run records `first_speakable_chunk` as unavailable with that reason
and synthesises the whole script. It does **not** silently speak a raw buffer.
Production's fallback (`harness.py:237`) does exactly that and can hand a voice
a mid-sentence fragment; that is `PRODUCT_ISSUES` P8, and it is a finding to
report, not a behaviour to reproduce inside a measurement.

## Credentials

**Both are required and both are verified by use.** Preflight makes one real
Exa search and one real eight-token Claude completion — together about $0.006 —
because "a key is set" is not "the key works", which is the rule that cost this
project four sessions.

If either is missing the run stops and names it. **Nothing is stubbed.** A
stubbed retrieval or a canned script in a latency waterfall produces a number
that looks like a measurement and is not one.

The keys are exported into the pod shell by hand. `tools/pack_for_pod.py`
builds the bundle from `git archive` plus two named files and has never
contained a credential; the pod needs no git remote, no GitHub token and no SSH
key. Rotating both keys after terminating the pod is the cautious move and
costs nothing.

## The reference voice

`--reference` is required and has no default. A default would quietly benchmark
whichever voice happened to be first on disk, and the voice is the one thing in
this run that a listener would notice being wrong.

`tools/pack_for_pod.py --reference <wav>` ships exactly one audio file. It reads
the recording's rights record first and refuses unless consent, commercial use
and synthetic-voice clearance are all cleared — the same gate as
`check_reference_audio.py`, enforced again here, because a gate the packer can
step around is not a gate. The record itself stays on the Mac, and so does
`sources.json`: both name the people who recorded, and a rented machine is the
last place that mapping belongs. The pod sees `reference_N.wav` and a note
saying the gate passed.

Watermarking is applied by Chatterbox inside every `generate()` and is not
bypassed, disabled or stubbed. Preflight fails if `PerthImplicitWatermarker` is
`None` rather than proceeding without it — which is what the four-gigabyte
`TypeError: 'NoneType' object is not callable` was.

## Cost

| item | estimate |
|---|---|
| preflight Exa search | ~$0.005 |
| preflight Claude completion | <$0.001 |
| experiment B, two runs (Exa + a 3-minute script each) | ~$0.07 |
| experiment A | $0 — no API calls |
| GPU | whatever a 4090 costs for the ~25 minutes, most of it install |

Actual API cost is computed from the API's own usage figures and written into
`pipeline.json` as `cost_usd_total`. The estimate above is arithmetic and is
allowed to be wrong; only the measured one goes in the report.

## Output

One timestamped directory, `experiments/results/combined_4090_<UTC stamp>/`:

```
results.json                 both experiments, merged
ANALYSIS.md                  the waterfall, written out
bench_chatterbox_base.json   A on its own
pipeline.json                B, cold and warm, with cost
events.jsonl                 every mark, raw, one JSON object per line
audio/cold.wav, warm.wav     what it actually said
audio/cold.txt, warm.txt     the query, the spoken chunk, the full script
```

`--experiment analyse` rebuilds `results.json` and `ANALYSIS.md` from whichever
section files exist, so if B fails, A's numbers are still readable rather than
lost with it. The pod script runs it on B's failure path for that reason.

## Validated here before anything is rented

`tests/test_pipeline_probe.py` and `tests/test_combined_experiment.py` run with
no GPU, no key and no network. Between them they hold the harness to:

* spans come from recorded marks, and a missing mark is `None`, never a guess;
* an unmeasured waterfall row is excluded from the share, so the measured
  shares still sum to 1.0;
* an unmeasurable stage is declared with its reason;
* a corpus missing a length bucket stops the run instead of benchmarking two
  thirds of the range;
* a second run never overwrites the first;
* no reference voice means no run;
* `analyse` writes a report from whichever half finished;
* the rights gate holds in the packer, and an uncleared recording is refused
  before it can reach a pod.

Every one of those is a fault that would otherwise have been found on a billing
meter. Four separate harness faults — not the thing being measured — have cost
runs on this project already.

## The procedure

See `experiments/RUNPOD_RUNBOOK.md`. In short: pack on the Mac, copy across,
export two keys, run `bash tools/pod_combined.sh <reference.wav>`, copy the
results directory back, terminate the pod.
