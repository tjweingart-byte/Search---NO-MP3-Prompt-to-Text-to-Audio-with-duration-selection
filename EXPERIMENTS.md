# The FAM Experiment Engineer

Describe an experiment in plain English. Get a plan, a cost, and — after you
approve it — a report that is still there tomorrow.

    python tools/experiment.py plan "exa vs current search, 10 trials"
    python tools/experiment.py run  "full exa to chatterbox pipeline 10 times"
    python tools/experiment.py run  "..." --simulate     # no key, no GPU, no spend
    python tools/experiment.py list
    python tools/experiment.py show <run-id>
    python tools/experiment.py compare <run-a> <run-b>

## What it is

An experiment layer **around** FAM, not inside it. Nothing in `experiments/` is
imported by the app; the dependency points the other way, so what gets measured
is the code that ships rather than a copy of it. `tools/check_additive.py`
fails the build if a production module is modified, and runs in `./dev.sh check`.

Exa and Chatterbox are **additional arms, never replacements.** The current
Anthropic web search and Piper stay the defaults and are the baseline every
comparison runs against.

## The loop

```
plain English  ->  spec  ->  preflight  ->  cost estimate  ->  your approval
                                                                    |
        report + recommendation  <-  statistics  <-  trials  <-------+
                    |
              saved to experiments/runs/<timestamp>-<name>/
```

## What a run leaves behind

    experiments/runs/2026-09-05T14-22-03-exa-vs-current-search/
        spec.json      the exact configuration, credential-free
        trials.jsonl   one line per trial, appended as it happens
        summary.json   statistics
        report.md      the readable report and the recommendation
        artifacts/     generated audio, when asked for with --save-audio

`trials.jsonl` is appended **during** the sweep, so a run that dies at trial 17
of 20 still leaves 16 on disk. Nothing is ever deleted or overwritten; there is
no prune and no rotate. Results are git-ignored by default — promote one
deliberately if a decision should carry its evidence.

## Where the time goes

Every stage is measured on **one clock**, the orchestrator's, whichever machine
did the work. A stage that ran elsewhere records the host, and — when the
service reports it — what the remote believed it spent:

| stage | host | wall | remote | overhead |
|---|---|---|---|---|
| search | `exa-api` | 0.31s | 0.24s | 0.07s |
| generate | `anthropic-api` | 0.94s | — | — |
| synthesis | `runpod-gpu` | 0.82s | 0.54s | 0.28s |

That last column is the price of the stage being on another machine, and it is
what a single-machine benchmark cannot see. The remote's own number is recorded
*beside* the wall time and never instead of it.

**The bottleneck is the stage with the largest median duration**, and the report
names it.

## What it refuses to do

These are enforced by code and pinned by tests, not by remembering:

- **It never starts, stops or pays for GPU infrastructure.** There is no Runpod
  SDK and no lifecycle call anywhere in `experiments/` — a test greps for them.
  An experiment needing a GPU that is not running stops with exit code 3 and
  tells you what to start.
- **It never spends without showing the estimate first.** `--yes` skips the
  question, never the number.
- **It never writes a credential**, including into its own reports. Every write
  is scrubbed, and `verify_clean()` re-reads the directory to prove it.
- **It never calls noise a winner.** Comparisons use a bootstrap confidence
  interval on the difference of medians; when that interval includes zero the
  report says "no detectable difference" and estimates the trials it would take.
- **It never hides a failure.** A failed trial is written down with its error
  and excluded from the statistics, and the report says how many failed.
- **It never passes off a stand-in as a measurement.** A simulated run is
  banner-marked and recommends nothing. The local speech adapter refuses to
  time the debug tone and call it Piper.
- **It never reads or writes the script cache.** A cache hit returns in
  milliseconds and would look like a result.

## Exa

**Connected**, ported from `exa_claude_benchmark.py` rather than rewritten, so
the numbers stay comparable with the ones measured by hand.

Two things still needed before it runs:

    pip install -r experiments/requirements.txt   # exa-py
    export EXA_API_KEY=...                        # or add it to ~/.fam/env

What was preserved verbatim from the manual benchmark:

| | |
|---|---|
| the call | `search_and_contents(type="fast", num_results=8, highlights=True)` |
| the packet | top **3** results, **2** highlights each, `SOURCE n / Title: / Key evidence:` |
| the chunk rule | the first sentence *ending* leaving at least **25** words |

The three hard-coded numbers are now parameters — `num_results`,
`packet_sources`, `highlights_per_source` — because they are the packet-size
knobs worth sweeping. Their defaults reproduce the manual run, and a
differential test checks the chunk rule against the original function verbatim.

**Two generators, and the difference matters.** `production` (the default) uses
FAM's own request shape, so a result says something about what ships.
`benchmark` reproduces the manual run's short 220-token opening call under its
own system prompt — use it to check the engine agrees with the hand-measured
number, not to decide anything about the product. An arm picks one with
`params={"generator": "benchmark"}`.

The evidence packet goes into the **question**, matching the benchmark, and
deliberately not through `EpisodePlan.context` — that field renders as
`<already_heard>` and tells the model the material is known and must not be
re-explained, which is the exact opposite of what retrieved sources are for.

### Ready-made specs

    python tools/experiment.py run experiments/specs/exa_benchmark_replication.json
    python tools/experiment.py run experiments/specs/exa_vs_current_search.json

The first replicates the manual run ten times instead of once, so the number
has a spread. Run it before trusting anything else the engine says.

## Chatterbox Turbo

Ported from the recovered Runpod benchmarks (`test_turbo.py`,
`fam_chunked_benchmark.py`), which are the source of truth. See
`experiments/adapters/CHATTERBOX_UNKNOWNS.md` for what they settle and what
they do not.

**`chatterbox_local` — in-process.** `ChatterboxTurboTTS.from_pretrained(device="cuda")`.
Needs a machine with a card:

    pip install -r experiments/requirements-chatterbox.txt

**`chatterbox` — remote, over HTTP.** For a pod that is already running. This
tool never starts, stops or pays for one.

    export CHATTERBOX_ENDPOINT=https://<host>/synthesise

### What is reproduced exactly

| | |
|---|---|
| model | `from chatterbox.tts_turbo import ChatterboxTurboTTS` |
| timing | `torch.cuda.synchronize()` on **both** sides of `perf_counter` |
| autograd | `with torch.inference_mode():` |
| warmup | one `generate("This is a warmup.")`, then a fence |
| duration | `wav.shape[-1] / model.sr` |
| realtime | `duration / gen_time` |
| copy | `wav.cpu()` **after** the clock stops |
| cost | `generation_time / 3600 * 0.75` |
| chunks | 120 ms silence between, none trailing |
| headline | "first chunk ready in" |

**The synchronisation is the load-bearing part.** CUDA queues work
asynchronously, so timing `generate` without a fence on each side measures how
long it took to *enqueue* the kernels — near-instant, and a realtime factor
that looks spectacular and means nothing. A test fails if either fence is
removed.

Both recovered methodologies are reachable, because the two files deliberately
differ: `params={"warmup": false, "inference_mode": false}` reproduces
`test_turbo.py`; the defaults reproduce `fam_chunked_benchmark.py`.

### The endpoint contract

    POST {endpoint}
    {"text": str, "sample_rate": int}          # rate is a hint only
    -> {"pcm_base64": str,                     # 16-bit LE mono PCM, no header
        "sample_rate": int,                    # the MODEL's rate (model.sr)
        "gpu_seconds": float,                  # the fenced generate time alone
        "device": str,                         # optional, e.g. "cuda"
        "cold": bool}                          # optional, first generate?

A working endpoint is `experiments/adapters/chatterbox_server_example.py` —
copy it to the pod. It loads *and warms* at startup, so request one is not
paying for either, and it wraps the same `synthesise()` the local arm calls so
the two arms stay comparable. Nothing here imports or starts it, and a test
enforces that.

### The target pipeline

    python tools/experiment.py plan experiments/specs/fam_latency_pipeline.json

    query -> Exa -> Claude streaming -> first 25-word sentence-boundary chunk
          -> Chatterbox Turbo -> first playable audio

Every stage is reported separately, with the host it ran on, and the headline
is time to first audio.

## Statistics, and why they are cautious

Latency is skewed and these samples are small. A t-test on ten trials produces
a confident number from evidence that does not support one, and this repository
has already paid for one confident claim that turned out to be wrong
(`PROBLEMS.md` §21 → §46).

So: medians for the headline, the full spread always shown, and comparisons by
bootstrap CI on the difference of medians with a fixed seed, so re-rendering a
report cannot change its conclusion.

When two arms differ in more than one dimension the report says so, because
"Exa + Chatterbox beat the baseline" does not tell you which half won.

## Adding a component test

A component test is the same machinery with one stage instead of five. Compare
first-chunk sizes without touching an adapter:

    python tools/experiment.py plan "current search, 10 trials, first-chunk 12"

## Not in V1

**Concurrency.** The field is recorded so today's runs stay comparable with
tomorrow's, and the spec refuses any value above 1 — a parallel sweep measures
queueing and calls it latency. It needs a load model before it means anything.
