# Phase 1 — Chatterbox stage split: contained project, or model-level rewrite?

**Builds nothing.** It measures the existing pipeline so the fork decision is
made on numbers. Decision criteria are written here *before* the run, so the
answer cannot be fitted to whatever comes back.

## What it measures

`experiments/chatterbox_stages.py`, driven by
`tools/chatterbox_stage_split.py`. It calls the same public stage methods
`tts_turbo.generate()` calls, in the same order, with
`chatterbox_impl.synchronize` fencing each — and keeps `generate()`'s own
constants (`n_cfm_timesteps=2`, the 6561 OOV threshold), so the sum is
comparable with the 2.848s the benchmark already recorded.

| stage | call |
|---|---|
| text prep | `punc_norm` + tokenizer |
| **T3** | `t3.inference_turbo` — text → speech tokens |
| **Flow** | `s3gen.flow_inference(..., finalize=True)` — tokens → mels |
| **HiFiGAN** | `s3gen.hift_inference(mels, None)` — mels → waveform |
| **Watermark** | `watermarker.apply_watermark` |

Run on real FAM chunks across short / medium / long, so length is the variable.

## The three questions, and what would answer each

### 1. Which stage owns time-to-first-audio?

Reported as seconds and as a share of the total, per bucket.

* **T3 dominates (say >60%)** → incremental *vocoding* buys little. Streaming
  Chatterbox would mean streaming token generation, which is a model-level
  change. **Verdict: deeper rewrite.**
* **Flow + HiFiGAN dominate** → the `finalize` and `cache_source` parameters
  already present are most of the work. **Verdict: contained.**
* **Split roughly evenly** → the win is bounded by whichever half is not
  streamed; report the arithmetic rather than a verdict.

### 2. Can T3 yield useful partial token groups early enough?

Measured: total T3 time and the token count, giving a mean per-token cost, and
from it a projected time-to-first-N-tokens.

**The projection is guarded.** It assumes the autoregressive loop costs roughly
the same per step. The probe runs three lengths and reports
`max_over_min` across them; **above 1.5 the report says the projection must be
discarded**, because a non-linear loop makes the mean meaningless. This is
stated up front so a convenient number cannot be kept by ignoring the check.

### 3. Can Flow and HiFiGAN consume successive partials?

The part nobody has tested. Both modes are run and compared against the
one-shot waveform from the same tokens:

* **`recompute_prefix`** — Flow re-run over `tokens[:n]` each round. What a
  naive implementation does; cost grows quadratically.
* **`delta_only`** — only the new tokens, with `cache_source` carried forward.
  The cheap version, and the one that has to be shown to work.

Every chunk but the last passes `finalize=False`, which is what makes
`flow.py:170` trim the pre-lookahead tail.

Two failure modes, both measured:

* **Recomputation.** If `delta_only` errors or produces the wrong audio, Flow
  needs the full prefix and the saving evaporates on long chunks.
* **Discontinuity.** `seam_ratios` reports each join's first-difference against
  the signal's own median first-difference. ~1 is inaudible; a step
  discontinuity reads far higher — validated at **1.4x for a clean join and
  41.9x for an injected click**. A click at every seam is not shippable
  however fast it is.

*(That check had an off-by-one on first writing — `diff[k]` is
`wav[k+1] - wav[k]`, so the join at sample `i` is `diff[i-1]`. Reading
`diff[i]` stepped over the click and reported all-clear. Fixed and pinned by a
test, because a false green here is the worst possible outcome of this phase.)*

## The watermark, treated as its own blocker

**Nothing here bypasses, removes or weakens watermarking, and no design that
does should be considered.** Chatterbox applies Perth's implicit watermark over
the finished waveform, and it operates on a spectrogram — so chunking it has
STFT edge effects at every boundary.

Perth ships its own decoder, `PerthImplicitWatermarker.get_watermark`, which
turns "is per-chunk watermarking acceptable" from an opinion into a
measurement. `watermark_probe` reports:

* the mark detected on the whole-waveform version — the baseline
* the mark detected on a per-chunk-watermarked waveform, joined
* the mark detected on **each chunk individually** — the case that matters for
  streaming, where a listener may only ever receive some chunks
* seam ratios at the watermark boundaries
* max sample difference against the whole-waveform version

**What would have to change for incremental playback.** Three options, in
increasing order of what they ask of Resemble AI:

1. **Watermark per chunk as emitted.** Cheapest. Acceptable only if the probe
   shows the mark still decodes from individual chunks *and* from the join
   without seams.
2. **Watermark on a trailing window.** Emit audio unwatermarked, watermark a
   sliding window behind the playhead. Adds complexity, and means some audio
   reaches the listener before it is marked — likely unacceptable.
3. **Ask Resemble AI.** Whether per-chunk application preserves the
   watermark's guarantees is a question about their design and their licence
   terms, not one this repository can settle by measurement alone. The probe
   produces the evidence to ask with.

**If option 1 fails the probe, the watermark is a genuine blocker on
incremental Chatterbox** regardless of what the stage split says, and that
belongs in the verdict.

## Running it

    python tools/chatterbox_stage_split.py --device cuda \
        --out experiments/results/chatterbox_stage_split.json

One 4090 session. Small: a few chunks per bucket, two trials each, plus the
chunkability and watermark probes once per chunk. Cold load and warmup happen
before any measurement and are excluded. Every result carries
`LOCAL CUDA / DEVELOPMENT BENCHMARK` and `is_production_latency: false`.

## Standing constraints, restated

* **No fork is built in this phase**, whatever the numbers say.
* **No production code is modified.**
* **No unverified weight licence is treated as commercially usable** — see
  Phase 0. Chatterbox and Kokoro weight licences remain unconfirmed (P19), and
  XTTS-v2's weights are CPML non-commercial (P18). A favourable Phase 1 result
  does not make Chatterbox adoptable; it makes it *technically* adoptable
  pending that check.
