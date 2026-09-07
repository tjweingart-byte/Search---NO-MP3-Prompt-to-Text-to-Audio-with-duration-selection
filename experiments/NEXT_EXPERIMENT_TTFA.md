# Proposed: closing the 2.8s gap — fork Chatterbox, or replace it

## The question the measurements produced

Chatterbox on a 4090 reaches first audio in **2.848s** for a median FAM chunk.
The spec is ~1s and the shipped Piper build is 0.5s. The gap is architectural:
the path is one-shot, so first audio costs `chunk_audio / realtime_factor`, and
hardware would need another 3-6x on top of a 4090 to close it.

Two ways out, and they are not symmetric in cost:

* **A — make generation incremental.** Fork `chatterbox-tts` so audio is
  emitted while it is still being generated. First audio stops depending on
  chunk length.
* **B — use an engine that already does that.** Replace Chatterbox with a
  self-hosted engine exposing a native incremental interface.

**Recommendation: do not choose yet. Run the cheap parts of both, in order,
and let the first gate settle it.** A is a fork of a third-party model with an
unresolved watermarking question; that is weeks. B might be answered in an
afternoon. Committing to A before pricing B would be expensive guessing.

---

## Phase 0 — source audit of every candidate. Free. Hours.

The method that answered the Chatterbox streaming question in minutes: download
the package, read the inference path, look for an incremental interface. No
GPU, no weights, no account.

For each candidate, answer four questions from the source alone:

1. Is there a generator, callback or chunked entry point — anything that emits
   audio before generation completes?
2. If yes, what is the granularity — frames, tokens, sentences?
3. Is there a post-processing step over the whole waveform (a watermarker, a
   loudness normaliser) that would defeat incremental output?
4. What are the license and the dependency footprint?

**Shortlist to audit**, and the honest status of each — *these are candidates
to verify, not capabilities I have confirmed*:

| candidate | why it is on the list | what must be verified |
|---|---|---|
| **Kokoro-82M** | named in `CLAUDE.md` as the untried candidate; open weights, ships with the app, no quota | whether it exposes incremental output at all, and how it sounds |
| **Piper** | the shipped baseline, already measured at 0.5s | it is the number to beat; confirm how it achieves that |
| **XTTS-v2 / Coqui** | widely reported to have a streaming inference path | whether that path exists in a maintained, licensable build |
| **Chatterbox (fork)** | already audited: `hifigan` takes `cache_source`, `t3` appends tokens in a loop | whether the watermarker can go per-chunk |

Add or drop candidates freely — the audit is cheap, and the point is to reach
the GPU with a shortlist that is known to have the interface.

**Gate 0:** at least one candidate other than Chatterbox has a genuine
incremental interface. If none does, A is the only road and Phase 1 decides
whether it is worth walking.

## Phase 1 — the stage split. **DONE. Verified.**

**Result: T3 is 92.7% of generation time (max 93.1%); everything downstream is
7.3%.** The pre-registered criterion was T3 above 60% means a model-level
rewrite, and it was crossed by 32.7 points. Flow and HiFiGAN's `finalize` and
`cache_source` are real and irrelevant - the most they could remove is 0.208s
of a 2.848s wait.

**Gate 1 outcome: t3 dominates -> option A is a deeper rewrite, not a contained
change.** Recommendation taken: **option 3**, stop investing in the fork and
prioritise the bake-off. See `results/chatterbox_stage_split/ANALYSIS.md`.

Chatterbox stays in Phase 2 as a *quality* candidate (base vs Turbo). The fork
reopens only if its voice wins the listening test, its weight licence checks
out, and nothing else reaches sub-second.

<details><summary>Original Phase 1 plan, kept for the record</summary>

### One 4090 session, ~$0.75. Hours.

Before anyone forks anything, find out whether incremental *vocoding* would
even help. Time the two stages separately on the same 106 chunks:

    t3.inference_turbo   text -> speech tokens
    s3gen.inference      speech tokens -> waveform
    watermarker          waveform -> watermarked waveform

Fenced with `torch.cuda.synchronize()` on both sides of each, exactly as the
existing benchmark does.

**Gate 1, and this is the decision point for A:**

* If **s3gen dominates** — incremental vocoding through `cache_source` would
  deliver most of the win, and A is a contained change.
* If **t3 dominates** — the tokens must all exist before any audio can be made,
  so incremental vocoding buys little and A means making *token generation*
  stream too. Much larger, and probably not worth it against B.
* Whatever the watermarker costs is a hard addition to any incremental design,
  because today it runs over the finished waveform.

This is one script, no new infrastructure, and it reuses the corpus and the
harness unchanged.

</details>

## Phase 2 — **reordered: the listening test comes first, and it is free**

Phase 1 made the quality question decisive, so the blind bake-off runs *before*
any further latency benchmark and needs no GPU. Design preserved in
`audit/PHASE2_VOICE_BAKEOFF.md`; roster, passages and scoring fixed before any
audio existed. The latency bake-off below follows only for whatever survives.

### The latency bake-off, afterwards. One 4090 session, ~$0.75. A day.

Every candidate that passed Gate 0, on the **same 106 chunks, same three
buckets, same 3 trials, same metric definitions**, so results drop straight into
the existing comparison beside MPS and the 4090.

Measured per engine: time to first *playable* audio (not to completion), model
versus delivery split, realtime factor, words-to-latency slope, and cold start
separately. Plus, for the streaming ones, the marks that currently collapse —
`stream_begin` and `first_audio_bytes` — which should finally separate.

**Gate 2:** does any engine reach **first audio under ~1s on a 33-word chunk**?

## Phase 3 — the listening test. Free. The part that actually decides.

Everything above is latency. **FAM's problem with Piper was never speed, it was
that it sounds flat** (P2), and that has been true through this entire
sequence: nobody has heard Chatterbox on a FAM script, or Kokoro at all.

Synthesise the same three chunks on every engine that passed Gate 2 and listen.
An engine that hits 0.4s and sounds worse than Piper has solved nothing.

**Gate 3:** at least one engine is both fast enough and better than Piper.

---

## Cost and shape

| phase | cost | time | decides |
|---|---|---|---|
| 0 audit | $0 | hours | which engines have the interface |
| 1 stage split | ~$0.75 | hours | whether forking Chatterbox is contained |
| 2 bake-off | ~$0.75 | a day | whether anything is fast enough |
| 3 listening | $0 | an hour | whether anything is good enough |

Under $2 of GPU to answer both A and B properly. That is why it is worth
running both cheap halves rather than picking now.

## What is deliberately not in scope

* No production change. This stays in the experiment layer.
* No hosted APIs. Per-character billing breaks the "audio is nearly free"
  premise the prefetch plan rests on (`VOICE_OPTIONS.md`).
* No fork written before Phase 1 reports. Forking on a hunch is how a week
  disappears.

## Standing caveat

Every figure produced by Phases 1 and 2 is a **development benchmark** on one
machine with no concurrency or queueing, exactly like the two runs before it.
Production latency needs a deployment, and that is a separate question — the
one `CLAUDE.md` lists as still open under "Where does this deploy?".
