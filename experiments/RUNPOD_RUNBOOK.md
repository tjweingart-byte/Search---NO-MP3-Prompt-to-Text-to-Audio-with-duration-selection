# Runpod GPU benchmark — execution plan

The apples-to-apples counterpart to the MPS run. **Nothing in this repository
starts, stops, resizes or pays for a pod.** Every step below is a human action.

## 1. What has to get onto the pod

Three things, and only three.

**The repository**, on this branch. The local benchmark path imports nothing
from production FAM — verified by walking the import graph:

    tools/chatterbox_first_audio.py -> stdlib, torch, perth, experiments
    experiments/chatterbox_probe.py -> stdlib, experiments
    experiments/adapters/chatterbox_impl.py -> stdlib, torch

No `config`, no `tts.py`, no app. A clone is simplest; a copy of those three
files plus the corpus would also work.

**The corpus**, `experiments/chunks/first_chunks.json` — the 106 validated
chunks. This is **git-ignored**, so cloning does not bring it. It is
regenerated on the pod from `experiments/results/warm_first_token/`.

> **Before starting the pod, check that folder is actually pushed.** Preserving
> a run writes it into a tracked folder but does not commit it, and it was not
> committed the first time — only the hand-written `ANALYSIS.md` was on the
> branch, so the clone would have carried no openings file and the extract on
> the pod would have failed *after* the install and the 4 GB download. Verify
> from any machine:
>
>     git ls-tree -r --name-only origin/<branch> -- experiments/results/warm_first_token/
>
> It must list `openings_by_arm.md`. If it does not, push it from the Mac
> first.

Then, on the pod:

    python tools/extract_chunks.py experiments/results/warm_first_token
    python tools/extract_chunks.py --verify        # expect 106, CUT included = 0

Regenerating is preferable to copying: it re-runs the validation on the pod and
proves the same corpus, rather than trusting a file transfer.

**Dependencies** — but *after* the corpus check below, not before. Every check
that costs nothing runs first, so a preventable failure happens in seconds on a
laptop rather than in minutes on a rented card:

    pip install -r experiments/requirements-chatterbox.txt

That file already pins `setuptools<82`, which is what stops the perth
watermarker silently becoming `None` (P13). The ~4 GB weight download happens
once per pod unless the HF cache is on a persistent volume — **put
`HF_HOME` on the persistent volume if the pod has one**, or the download is
repaid every time the pod is recreated.

## 2. Which GPU the previous test used

**An NVIDIA RTX 4090.** This is recorded in the repository, not remembered:

* `experiments/adapters/chatterbox_impl.py:4` — "both run on an RTX 4090"
* `experiments/adapters/CHATTERBOX_UNKNOWNS.md` — Device: `cuda`, explicitly, on an RTX 4090
* `GPU_DOLLARS_PER_HOUR = 0.75`, from `GPU_RATE = 0.75` in the recovered scripts

Both recovered files (`test_turbo.py`, `fam_chunked_benchmark.py`) call
`from_pretrained(device="cuda")`. **Use a 4090 again** — a different card would
confound the GPU-versus-MPS comparison with a GPU-versus-GPU one.

## 3. Can the benchmark run there unchanged?

**Yes, with `--device cuda` and nothing else.**

* `resolve_device("cuda")` returns cuda when named, and `available_devices()`
  reports whether the machine really has it; the preflight checks the machine,
  not the flag.
* `synchronize()` already fences with `torch.cuda.synchronize()` — it is a no-op
  on MPS and active on CUDA, which is the correction the recovered code forced.
  Without it a CUDA run times kernel *queueing* and reports a fantasy.
* The runner re-reads `model.device` after loading and refuses if it is not what
  was asked for.

No code change. The command differs from the Mac run by one word.

## 4. What will be directly comparable

Same corpus, same buckets, same trial count, same model, same metric
definitions — so these compare one-to-one:

| metric | why it compares |
|---|---|
| `first_playable_seconds` p50 per bucket | the headline; identical definition both sides |
| `model_seconds` p50 | text in → completed waveform; fenced on both devices |
| `delivery_seconds` p50 | 0.000s in-process on MPS; expected 0.000s in-process on CUDA too |
| `realtime_factor_p50` | audio produced ÷ generation time |
| words → latency slope | same 106 texts, same three buckets and ranges |

Reported separately and **not** compared as request latency: cold model load and
warmup. They differ by machine, disk and cache state, and are excluded from
every trial figure on both sides.

Not comparable, because it is not being measured the same way: anything
involving HTTP. The GPU run below is **in-process on the pod**, exactly as the
MPS run was in-process on the Mac. A remote-endpoint run is a *third*
experiment, and it is the one where P11 (streamed delivery) finally matters.

## 5. Preserving the results

Same path as every other run:

    python tools/chatterbox_first_audio.py --local --device cuda --trials 3 \
      --out experiments/results/chatterbox_runpod_4090.json

Then get the file **off the pod** before it is destroyed — a pod is more
ephemeral than a laptop, and `experiments/results/` is committed, so:

    git add experiments/results/chatterbox_runpod_4090.json
    git commit -m "Chatterbox on a 4090: first-audio latency"
    git push

The JSON carries its own label (`LOCAL CUDA / DEVELOPMENT BENCHMARK`),
`is_production_latency: false`, the cold-start block, and every per-trial row.
Losing the pod after pushing costs nothing.

## 6. Why the old "~1.5s" figure may not compare

**First and most important: that number is not recorded anywhere in this
repository.** `CHATTERBOX_UNKNOWNS.md` says plainly — "No numbers were
recovered - only the code." The ~1.5s is a recollection of a manual session,
and by the standing rule that historical manual measurements are not verified
runs, **it is not a baseline.** The GPU run establishes the baseline; it does
not reproduce one.

If it is nonetheless compared, these differences would each move the number:

| difference | effect |
|---|---|
| **Metric definition.** `fam_chunked_benchmark.py` reports `first_chunk_seconds = results[0]["generate_seconds"]` — generation only. This benchmark reports generation **plus** PCM conversion. | Small: delivery measured 0.000s in-process on MPS. The two are near-identical in practice. |
| **Text.** The old run used whatever text was to hand. This uses 106 real FAM chunks of 25-59 words. | Potentially large. Latency scales with length — a short test sentence would be much faster than a 41-word chunk. |
| **Trials and statistic.** `test_turbo.py` is a single cold generate with no warmup. This is 3 trials × 106 chunks, warm, reported as a median. | Large. A single cold run and a warm median are different quantities. |
| **`inference_mode`.** `test_turbo.py` omits it; `fam_chunked_benchmark.py` uses it. This benchmark uses it. | Unknown, probably small. |
| **Package version.** The version used on the pod was never recovered. This pins nothing beyond what `chatterbox-tts` requires; today that resolves to **0.1.7**. | Unknown. Record the installed version with the results. |
| **Card and pod.** Same 4090 is intended, but pod-to-pod variation in clocks, thermals and neighbours is real. | Small but nonzero. |

The honest framing for the report: the GPU run is the **first recorded**
Chatterbox measurement on FAM text. The ~1.5s is context for expectations, not
a number to reconcile against.

## 7. Cost and approval

An RTX 4090 pod bills roughly **$0.75/hour** while running. 318 generations plus
a ~4 GB download and a cold start; if the 4090 is meaningfully faster than MPS
the sweep is well under an hour, so **on the order of $0.40-$0.75**, dominated
by however long the pod is up rather than by the compute.

**This tool will not start the pod.** The adapter has no Runpod SDK import and
no create/start/stop call anywhere. Starting, sizing and stopping it are yours.
