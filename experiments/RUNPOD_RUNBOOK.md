# Runpod GPU benchmark — execution plan

The apples-to-apples counterpart to the MPS run. **Nothing in this repository
starts, stops, resizes or pays for a pod.** Every step below is a human action.

## 1. What has to get onto the pod

**No credentials, ever.** A rented card is somewhere to run a benchmark, not
somewhere to leave a GitHub token or an SSH key. So nothing is cloned on the
pod and nothing is pushed from it: one tarball goes up, one JSON comes back.

On the Mac:

    python tools/pack_for_pod.py
    scp fam-pod.tar.gz root@<pod>:/workspace/

`pack_for_pod.py` writes `git archive` of the tracked tree at HEAD, plus the
one git-ignored file the benchmark needs — the validated chunk corpus — and a
`POD.txt` naming the revision and the corpus SHA-256. It **validates the corpus
before packing** and refuses to ship one containing a chunk that does not end at
a sentence boundary, because a wrong corpus would break comparability with the
Mac run silently and only after the card had been paid for. About 0.5 MB.

The benchmark path imports nothing from production FAM — verified by walking
the import graph:

    tools/chatterbox_first_audio.py -> stdlib, torch, perth, experiments
    experiments/chatterbox_probe.py -> stdlib, experiments
    experiments/adapters/chatterbox_impl.py -> stdlib, torch

No `config`, no `tts.py`, no app.

On the pod, free checks first — every preventable failure happens in seconds,
before the install and before the 4 GB download:

    cd /workspace && tar xzf fam-pod.tar.gz && cd FAM
    cat POD.txt
    shasum -a 256 experiments/chunks/first_chunks.json | cut -c1-16
    python tools/extract_chunks.py --verify

The digest must match the one in `POD.txt`, and `--verify` must report the same
chunk count, the same bucket ranges, and `CUT chunks included in corpus 0` as
the Mac. If any of that differs, the pod is not running the Mac's corpus and
the comparison is void — stop before installing anything.

Then the expensive parts:

    export HF_HOME=/workspace/hf     # persistent volume, or the 4 GB is repaid
    pip install -r experiments/requirements-chatterbox.txt

That file pins `setuptools<82`, which is what stops the perth watermarker
silently becoming `None` (P13).

Getting the result back, also without credentials — from the Mac:

    scp root@<pod>:/workspace/FAM/experiments/results/chatterbox_runpod_4090.json \
        experiments/results/

Copy it **before destroying the pod**. Committing happens on the Mac, where the
credentials already are.

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

On the pod:

    python tools/chatterbox_first_audio.py --local --device cuda --trials 3 \
      --out experiments/results/chatterbox_runpod_4090.json

Then, from the Mac, before the pod is destroyed:

    scp root@<pod>:/workspace/FAM/experiments/results/chatterbox_runpod_4090.json \
        experiments/results/
    git add experiments/results/chatterbox_runpod_4090.json
    git commit -m "Chatterbox on a 4090: first-audio latency"
    git push

The pod never authenticates to anything. A pod is more ephemeral than a laptop,
so copy first and destroy second.

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

---

# Phase 4 — the combined run: Chatterbox alone, and the whole request

Design and reasoning: `experiments/audit/PHASE4_COMBINED_PIPELINE.md`. This
section is the procedure only.

**Nothing in this repository starts, stops, resizes or pays for a pod.** Every
step below is a human action.

## 0. What changes from the earlier runs — read this first

Sections 1-7 above say *no credentials, ever*. **Phase 4 needs two API keys on
the pod**, and that is a deliberate, narrow exception rather than a drift.

The reason is the experiment: it measures `Exa → Claude → Chatterbox` end to
end, and a stubbed retrieval or a canned script would produce a waterfall that
looks like a measurement and is not one. Either the keys go up or the
experiment is not the experiment.

What still holds, and what it costs:

* **No GitHub token, no SSH key, no git remote on the pod.** One tarball up,
  one directory back, exactly as before.
* **The keys are exported into the pod shell by hand and never written to a
  file.** `pack_for_pod.py` builds the bundle from `git archive` plus two named
  files; it has never carried a credential and still does not.
* **Rotate both keys after terminating the pod.** It costs a minute and removes
  the only thing this exception leaves behind.

Also new: one reference recording goes up. Its rights record and
`sources.json` — the two files that name the people who recorded — stay on the
Mac. The pod sees `reference_N.wav` and a line in `POD.txt` saying the rights
gate passed at pack time.

## 1. Before renting anything, on the Mac

    cd ~/path/to/FAM
    git pull
    python3 -m pytest tests/test_pipeline_probe.py tests/test_combined_experiment.py -q

Fourteen tests, no GPU, no key, no network. They are the harness held to the
faults that have cost runs on this project before: an unmeasured stage
absorbing someone else's time, a corpus missing a bucket, a run overwriting the
last one, the rights gate not holding in the packer.

## 2. Pack, on the Mac

    python3 tools/pack_for_pod.py \
      --reference experiments/references/working/reference_N.wav

Substitute the voice the Phase 3 identity bake-off chose. There is no default
and the tool will not guess.

It prints the corpus SHA-256, the voice SHA-256, and the three rights fields it
cleared. If it refuses, it names the field — fix the record, not the check.

## 3. Start the pod

Yours to start, size and stop. **RTX 4090**, the same card as every earlier
Chatterbox run; a different card would make this a GPU comparison as well as a
pipeline one, and the script's first gate refuses one. A PyTorch template. A
network volume mounted at `/workspace` is worth it if you expect to run again —
`HF_HOME` points there and the ~4 GB of weights then survive the pod.

    scp fam-pod.tar.gz root@<pod>:/workspace/
    ssh root@<pod>
    cd /workspace && tar xzf fam-pod.tar.gz && cd FAM

## 4. The whole experiment, as one command

    export EXA_API_KEY='...'
    export ANTHROPIC_API_KEY='...'
    bash tools/pod_combined.sh experiments/references/working/reference_N.wav

That is the run. It stops at the first gate that fails and says what to do:

| gate | cost | what it settles |
|---|---|---|
| 1. RTX 4090 | free, instant | wrong card caught in seconds, not after a 4 GB download |
| 2. both keys exported | free, instant | the run will not silently become a stubbed one |
| 3. corpus and voice present, SHA-256 printed, corpus verified | free, seconds | the pod is running the Mac's inputs |
| 4. dependencies | ~5-15 min | the slow part; nothing after it can fail on a missing package |
| 5. `diagnose_chatterbox.py` | free | the watermarker `NoneType`, the torchvision/torch mismatch, the transformers lazy-module mask — each with its real cause named |
| 6. preflight | ~$0.006 | one **real** Exa search and one **real** Claude completion. "A key is set" is not "the key works" |

Then experiment A, then experiment B cold and warm. If B fails, A's numbers are
already written and the script runs `--experiment analyse` so they stay
readable rather than being lost with it.

Expect roughly 25 minutes of billed time, most of it the install.

## 5. Getting the results back, then terminating

The script prints the directory name. From the **Mac**, before the pod is
destroyed:

    scp -r root@<pod>:/workspace/FAM/experiments/results/combined_4090_<stamp> \
        experiments/results/

    open experiments/results/combined_4090_<stamp>/ANALYSIS.md
    afplay experiments/results/combined_4090_<stamp>/audio/cold.wav

Then, on the Mac:

    git add -f experiments/results/combined_4090_<stamp>
    git commit -m "Combined 4090 run: Chatterbox Base alone, and the whole request"
    git push -u origin claude/fam-repo-inventory-5lznba

`experiments/results/` is git-ignored by default, so `-f` is how a run is
promoted deliberately. **The run is not shared until that push completes** —
data has been left on a terminated pod on this project three times.

Copy first, destroy second, rotate the keys third.

## 6. What comes back

```
results.json                 both experiments, merged
ANALYSIS.md                  the waterfall, written out
bench_chatterbox_base.json   Chatterbox Base alone
pipeline.json                the whole request, cold and warm, with measured cost
events.jsonl                 every mark, raw
audio/cold.wav  warm.wav     what it actually said
audio/cold.txt  warm.txt     the query, the spoken chunk, the full script
```

Every artefact carries `is_production_latency: false` and the string *RTX 4090 /
DEVELOPMENT BENCHMARK*: one machine, one request at a time, no concurrency and
no queueing. The two numbers to read first are `search_to_first_listen` and
`search_to_complete_audio` in the cold run.
