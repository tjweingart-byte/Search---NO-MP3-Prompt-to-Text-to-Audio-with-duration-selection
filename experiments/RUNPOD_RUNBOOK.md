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

---

# Phase 5 — the concurrent pipeline: Claude and Chatterbox at once

Design and reasoning: `experiments/audit/PHASE5_CONCURRENT_PIPELINE.md`. This
section is the procedure only.

Phase 4 measured the three stages **in sequence** — it found the first
speakable chunk at ~2.4s and then waited for `claude_complete` before
synthesising. Phase 5 measures FAM's actual architecture: the first chunk goes
to the voice while the model is still writing. The overlap is **asserted**; a
run that did not overlap writes its results and exits non-zero.

## 1. On the Mac, free

    git pull
    python3 -m pytest tests/test_concurrent_pipeline.py -q     # 19 tests
    python3 tools/streaming_4090_experiment.py --dry-run --runs 1

The dry run exercises the whole path — the same queue, the same assertions, the
same audio writing — with no model, no GPU and no API calls. It writes to a
directory named `streaming_STUB_<stamp>` and stamps every artefact **NOT A
RESULT**.

## 2. Get the new code onto the pod

**A pod that ran Phase 4 is running an older tarball** and does not have
`experiments/concurrent_pipeline.py`. Gate 3 refuses in that case rather than
running something stale. Re-pack and copy across:

    python3 tools/pack_for_pod.py \
      --reference experiments/references/working/reference_3.wav
    scp fam-pod.tar.gz root@<pod>:/workspace/
    ssh root@<pod>
    cd /workspace && tar xzf fam-pod.tar.gz && cd FAM

The weights under `HF_HOME` and the installed packages survive; only the source
tree is replaced, so this costs seconds rather than the ten-minute install.

## 3. The whole experiment, as one command

    export EXA_API_KEY='...'
    export ANTHROPIC_API_KEY='...'
    bash tools/pod_streaming.sh experiments/references/working/reference_3.wav

Gates, in cost order:

| gate | cost | what it settles |
|---|---|---|
| 1. RTX 4090 | free, instant | the same card as every earlier run |
| 2. both keys exported | free, instant | the run will not silently become a stubbed one |
| 3. voice present, and the tarball is current | free, instant | an old tree is caught before the install, not after the run |
| 4. **the stub run overlaps** | free, ~5s | if the harness is wrong, it is wrong here rather than on the meter |
| 5. dependencies | seconds on a configured pod | a no-op after Phase 4 |
| 6. preflight | ~$0.006 | one **real** Exa search, one **real** Claude call, the watermarker, and production's chunker |

Then the run itself: cold, then warm.

## 4. Reading the result

The first thing in `ANALYSIS.md` is **"Did it actually overlap?"** — a yes with
the number of seconds by which TTS beat `claude_complete`, or a no with exactly
which assertion failed.

Then, per run: `claude_ttft`, `claude_to_first_chunk`,
`first_chunk_tts_seconds`, **`search_to_first_listen`**, `claude_total`,
`search_to_complete_audio`, `overlap_seconds`, `backpressure_seconds`; whether
synthesis kept ahead of playback and by how much headroom; and a per-chunk
table of words, ready / start / done, audio duration, generation time, realtime
factor and queue wait.

Compare `search_to_first_listen` against the Phase 4 sequential run. That
difference is what the architecture is worth, and it costs no extra compute —
only a different order.

**Read `claude_total` against Phase 4's too.** Chatterbox's T3 stage is a Python
autoregressive loop holding the GIL, so the overlapped Claude stream may run
slower than the sequential one. That is a real cost of the architecture, and it
should be read rather than assumed away.

## 5. Getting it back, then terminating

    scp -r root@<pod>:/workspace/FAM/experiments/results/streaming_4090_<stamp> \
        experiments/results/

    open experiments/results/streaming_4090_<stamp>/ANALYSIS.md
    afplay experiments/results/streaming_4090_<stamp>/audio/cold_episode.wav

    git add -f experiments/results/streaming_4090_<stamp>
    git commit -m "Concurrent pipeline on a 4090: Claude and Chatterbox at once"
    git push -u origin claude/fam-repo-inventory-5lznba

Copy first, destroy second, rotate the keys third.

---

# Phase 6 — Claude decoupled from Chatterbox

Design and reasoning: `experiments/audit/PHASE6_DECOUPLED_PIPELINE.md`. This
section is the procedure only.

Phase 5 proved the overlap and gave the baseline: **Search → First Listen =
4.659s warm**, zero stalls. It also reported `claude_total = 66.668s` with
`backpressure = 64.273s` inside it, because the TTS queue sat directly under
the Claude reader. Phase 6 separates them, batches later sentences into
speech-sized chunks, and asserts that TTS saturation never reaches upstream.

## 1. On the Mac, free

    git pull
    python3 -m pytest tests/test_speech_assembler.py \
                      tests/test_decoupled_pipeline.py \
                      tests/test_phase6_runner.py -q          # 50 tests

    python3 tools/phase6_experiment.py --dry-run --runs 1
    python3 tools/phase6_experiment.py --dry-run --coupled --runs 1

The first dry run must pass; the second must **fail its decoupling assertion**,
because it rebuilds Phase 5's coupling on purpose. An assertion that has never
rejected anything is decoration.

If you still have Phase 5's `results.json`, this is the moment to look at it —
it was never pushed, so nothing in this repository has seen its chunk data:

    python3 tools/fit_chunk_policy.py <phase5>/results.json

It prints the measured chunk distribution, fits generation time against word
count from that run's own timings, and replays the assembler over the same
sentences to show what Phase 6 would have produced.

## 2. Get the new code onto the pod

A pod that ran Phase 5 does not have `experiments/decoupled_pipeline.py` or
`experiments/speech_assembler.py`; Gate 3 refuses rather than running something
stale.

    python3 tools/pack_for_pod.py \
      --reference experiments/references/working/reference_3.wav
    scp fam-pod.tar.gz root@<pod>:/workspace/
    ssh root@<pod>
    cd /workspace && tar xzf fam-pod.tar.gz && cd FAM

Weights under `HF_HOME` and installed packages survive; only the source tree is
replaced, so this costs seconds.

## 3. The whole experiment, as one command

    export EXA_API_KEY='...'
    export ANTHROPIC_API_KEY='...'
    bash tools/pod_phase6.sh experiments/references/working/reference_3.wav

| gate | cost | what it settles |
|---|---|---|
| 1. RTX 4090 | free, instant | the same card as Phase 5, so the comparison is architecture only |
| 2. both keys exported | free, instant | the run will not silently become a stubbed one |
| 3. voice present, tarball current | free, instant | an old tree is caught before the install |
| 4a. the decoupled stub passes | free, ~10s | the harness is sound |
| 4b. **the coupled stub fails** | free, ~10s | the assertion has teeth |
| 5. dependencies | seconds on a Phase 5 pod | a no-op |
| 6. preflight | ~$0.006 | one real Exa search, one real Claude call, the watermarker, production's chunker |

Then cold, then warm.

## 4. What success looks like

`ANALYSIS.md` opens with six plain-English answers, before any table:

1. Search → first listen, warm, against the 5.0s budget and Phase 5's 4.659s
2. whether Claude stayed decoupled — `claude_reader_blocked_seconds` at 0.000s
   and the decoupling reported as **exercised**
3. how long Claude actually took, with our processing and our blocking split out
4. playback stalls, and minimum / median / maximum headroom
5. sentences in, TTS calls out, median words, how many under five and ten
6. whether Phase 6 improved on Phase 5, or *not proven*

A pass is: first listen ≤ 5.0s, `first_tts_start < claude_complete`, zero
stalls, no text loss, and the reader never blocked. **One request per
condition**, so a first-listen difference of a few hundred milliseconds against
Phase 5 is API variance, not a finding.

## 5. Getting it back, then terminating

    scp -r root@<pod>:/workspace/FAM/experiments/results/phase6_4090_<stamp> \
        experiments/results/

    open experiments/results/phase6_4090_<stamp>/ANALYSIS.md
    afplay experiments/results/phase6_4090_<stamp>/audio/warm_episode.wav

    git add -f experiments/results/phase6_4090_<stamp>
    git commit -m "Phase 6 on a 4090: Claude decoupled from Chatterbox"
    git push -u origin claude/fam-repo-inventory-5lznba

Copy first, destroy second, rotate the keys third. **Push the results** —
Phase 5's were not, so nothing in this repository has ever seen its chunk data.
