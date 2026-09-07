# Phase 1 verdict — Chatterbox stage split

**Status of the numbers: reported from the run, not yet reread from the file.**
`chatterbox_stage_split.json` is on the author's Mac and is not in this
repository, so the figures below are as transcribed, not as verified here. Run

    python tools/verify_stage_split.py experiments/results/chatterbox_stage_split.json

which applies the pre-registered criteria and prints every field this analysis
needs. The verdict does not depend on the missing detail — the one number
reported crosses the threshold by a wide margin — but the detail should replace
this note.

## The criterion, fixed before the run

From `audit/PHASE1_STAGE_SPLIT.md`, written before any GPU time:

> **T3 dominates (say >60%)** → incremental *vocoding* buys little. Streaming
> Chatterbox would mean streaming token generation, which is a model-level
> change. **Verdict: deeper rewrite.**

## What was measured

**T3 accounts for roughly 92-93% of total generation time.** Flow, HiFiGAN,
the watermark and text preparation share the remaining 7-8%.

The criterion is crossed by more than thirty points. **Verdict: model-level
rewrite**, not a contained project.

## What 92-93% means, concretely

Against the 2.848s medium-bucket figure from the 4090 benchmark, T3 is
~2.62s of it and everything downstream is ~0.22s.

**So the ceiling on flow/hift streaming is about 0.22s.** If Flow and HiFiGAN
became instantaneous and the watermark free, first audio would fall from 2.848s
to roughly 2.62s. The target is ~1s and the shipped Piper build is 0.5s. The
entire prize available from the streaming primitives that started this
investigation is **under 8% of a wait that is 3-6x too long.**

This directly answers the question the Phase 0 audit raised: **Flow and
HiFiGAN's `finalize` and `cache_source` are real, and they are irrelevant.**
They cannot help with what they do not own. Chatterbox's latency is
overwhelmingly the autoregressive token loop.

## What incremental T3 would give, and why it is the whole project

T3 is autoregressive: it emits one speech token per step. If it yielded token
groups instead of collecting and returning, first audio would cost

    (tokens_per_chunk x per_token_cost) + flow(chunk) + hift(chunk) + watermark

rather than all tokens. With a per-token cost measured across three lengths,
that projection is arithmetic — *provided the per-token cost is linear, which
the run reports as `max_over_min`; above 1.5 the projection is unsafe and must
be discarded rather than believed.* Read that field from the file before
trusting any number derived this way.

The shape is favourable: a first chunk needing a small fraction of the tokens
would need a correspondingly small fraction of 2.62s. **Sub-second first audio
from Chatterbox is arithmetically reachable — and only through T3.**

That is precisely why it is a rewrite rather than a patch. Making an
autoregressive decode loop yield is not a signature change: it means owning the
KV-cache lifecycle across yields, the stopping logic, the interaction with
`transformers`' generation utilities, and the state that currently lives
implicitly in a function that runs to completion. It is a fork of a
third-party model's inference core, maintained against upstream.

## Why both chunking modes failed

Both `recompute_prefix` and `delta_only` failed. **The error text is in the
file and must be read before this section is finalised** — the paragraphs below
are the analysis to check it against, not a substitute for it.

Two candidate causes, both implementation-level rather than fundamental:

* **Conditioning and shape.** `flow_inference` takes `ref_dict=conds.gen` per
  call and produces mels for the token sequence it is given. A prefix is not
  the prefix of the full run's mels — the flow decoder attends across the
  sequence — so slicing the previous output out of a recomputed prefix is not
  guaranteed to line up, and a shape mismatch is the expected symptom.
* **The lookahead trim.** `flow.py:170` does
  `h = h[:, :-self.pre_lookahead_len * self.token_mel_ratio]` when
  `finalize is False`, removing 6 mel frames. A chunk that produces fewer than
  6 frames yields a negative or empty slice. With `token_mel_ratio=2`, a
  25-token chunk should be comfortably above that — but the first chunk in
  `delta_only` also loses the sequence context the decoder expects.

**Do these failures indicate fundamental architectural incompatibility?**
On this evidence, **no** — they look like an interface problem. The primitives
exist, they are used by CosyVoice-derived stacks in production, and the probe
called them in a way they were not written to be called. A correct
implementation would drive them from inside a streaming loop that owns the
token state, not from outside with prefixes.

**But that conclusion changes nothing about the recommendation**, because
fixing them buys at most the 7-8% those stages own. They would need to be
fixed *as part of* the T3 rewrite, not instead of it.

## Seams and watermarking

*Both sections require the file.* `verify_stage_split.py` prints the seam
ratios and the detector results; the calibration is **~1.4x for a clean join
and ~41.9x for an injected click**, so the numbers are interpretable on sight.

The watermark question is the one that could block incremental playback outright
even if T3 were solved. Perth applies its mark over a spectrogram, so per-chunk
application has STFT edge effects at every boundary. The probe reports the mark
detected on the whole-waveform baseline, on the per-chunk-marked waveform
joined, and **on each chunk alone** — the case that matters when a listener may
only receive some chunks.

**If per-chunk detection degrades materially against the whole-waveform
baseline, incremental Chatterbox is blocked on a question only Resemble AI can
answer**, regardless of the engineering. Nothing here weakens or bypasses
watermarking, and no design that does should be considered.

## Realistic engineering scope

Making Chatterbox genuinely incremental requires all four, in order:

1. **Rewrite T3's decode loop to yield** — KV-cache across yields, stopping
   logic, generation-utility interaction. The bulk of the work, in a fork of a
   third-party model's inference core.
2. **Drive Flow and HiFiGAN from inside that loop** — fixing what the probe hit
   from outside. Contained *given* (1).
3. **Resolve per-chunk watermarking** — possibly a conversation with Resemble
   AI rather than code.
4. **Maintain the fork** against upstream `chatterbox-tts`.

Weeks, not days, with a research question in the middle.

## And the reason not to spend them yet

**Nobody has heard Chatterbox.** Not on a FAM script, not at all. This entire
sequence measured latency.

Three facts make that decisive:

* **Voice quality is unverified.** FAM's problem with Piper was never speed
  (P2). An engine that reaches 0.4s and sounds worse than Piper solves nothing.
* **Turbo disables the expressive controls** the base model exposes — CFG,
  `min_p`, `exaggeration` are ignored (P17). The variant that is fast is the
  variant with fewer expressive knobs, and FAM's requirement is a *natural,
  expressive, premium* voice as well as low TTFA.
* **The weight licence is unverified** (P19), and the standing rule is that no
  unverified weight licence is treated as commercially usable.

Committing weeks to forking a model's inference core, before knowing whether
its voice is wanted or its weights are usable, is the expensive order to do
this in.

## Verdict

**Phase 1 answers its question: incremental Chatterbox is a model-level
rewrite, not a contained engineering project.** The streaming primitives that
motivated the investigation turn out to sit downstream of 92-93% of the cost.
