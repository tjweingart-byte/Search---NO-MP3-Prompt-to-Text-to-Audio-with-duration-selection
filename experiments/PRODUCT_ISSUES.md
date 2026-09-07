# Product issues found by the experiment layer

Carried forward between reports. Newest section last. Nothing here has changed
production FAM; these are findings, not fixes.

Status: **open** unless marked otherwise.

---

## Carried forward, still open

### P1. The scripts are not good enough, and the rewrite is still untested
From `CLAUDE.md`. Both prompts were rewritten around what makes a briefing
worth hearing, and the endings were rewritten again to stop teasing the next
episode (PROBLEMS.md §48). **Nobody has heard an episode under the new ending
rules** — it was written without an API key. Unchanged by any work in this
sequence; every experiment since has been about latency, not writing.

### P2. Voice quality — reopened, and now measured on one candidate
Piper works and sounds flat. WellSaid was removed after two episodes exhausted
a month's quota (PROBLEMS.md §61). Kokoro-82M was named as the candidate to try
first and **still has not been heard**. Chatterbox has now been measured on
Apple silicon (below) but nobody has listened to its output either — this
sequence measured latency, not quality. *The listening test is still the
outstanding move on voice.*

### P3. Explore cannot populate itself
Explore replays other listeners' episodes and by design cannot generate one, so
a fresh database leaves it empty however much it is tapped. `tools/seed_demo.py`
exists for this. Unchanged.

### P4. "What your followers are listening to" has no follow graph
It ranks co-listener overlap. The heading promises a social network the app does
not have. Either build follows or rename it. Unchanged.

### P5. `rank_might_like` is written, tested, and not shown
It was the only signal offering anything outside an established taste. One line
in `SECTIONS` brings it back. Unchanged.

---

## Found during the warm-client / first-token work

### P6. Arm order is confounded with arm identity in every sweep
`harness.run_all` iterates trial → query → arm in fixed order, so the first arm
always executes first. Any systematic effect of going first lands entirely on
that arm. It did not invalidate the warm-client run because no difference was
claimed, but it would invalidate one that did. **Randomise or rotate arm order
before treating any arm gap as real.** Experiment-layer only.

### P7. Connections are never reused, in the experiment layer *or* production
`ScriptGenerator` builds a client per call, so every episode pays DNS, TCP and
TLS again. The A/B measured what pooling saves. Production is untouched and
still pays it.

### P8. The chunk-rule fallback records a raw, unspeakable buffer
`harness.py`:

    if first_chunk is None:
        first_chunk = buffer.strip()

When a stream ends before the chunk rule is ever satisfied, the raw buffer
becomes the "first chunk" — mid-sentence, and in the warm-client run three of
them were. **In production that string would be sent to a voice and spoken
mid-clause.** The experiment layer now excludes them; production has no such
guard. This is the most directly user-visible finding in this list.

### P9. First chunks contain newlines the model wrote
`first_chunk_ready` strips only the ends, so a paragraph break inside an opening
survives into the text handed to TTS. Harmless for Chatterbox (`punc_norm`
handles it) but it is unexamined for Piper, and it broke a parser here.

---

## Found during the Chatterbox work

### P10. Chatterbox Turbo's *API* is one-shot. Whether its *architecture* must
be is an open question — **restated, the earlier wording overreached.**
What is established from the source: `generate()` returns one completed tensor
and there is **no `yield` anywhere in `chatterbox-tts` 0.1.7**. Tokens are
accumulated in full, vocoded in one call, then watermarked across the whole
waveform. Streamed *delivery* is possible and is built.

What is **not** established, and what an earlier version of this entry wrongly
implied was settled: that incremental generation is impossible. Two of the
three stages look built from streaming-capable parts — `s3gen/hifigan.py`
accepts a `cache_source` (the chunked-vocoding hook in CosyVoice-derived
stacks), and `t3.inference_turbo` generates tokens in an appending Python loop.
The open question is the watermarker, which currently runs over the finished
waveform. See `CHATTERBOX_STREAMING.md`. **Unresolved, and it is the subject of
the next proposed experiment.**

### P11. The JSON/base64 endpoint contract makes first-audio equal full-audio
Measured against a local server: identical work, first playable at 0.879s under
JSON/base64 versus 0.190s under streaming PCM. **Only matters remotely** — the
MPS run measured delivery at 0.000s in-process. Fix is `/synthesise/stream`,
already built and non-production.

### P12. Chatterbox is 2.8s to first audio on a 4090 — better, still too slow
*Updated with the measured comparison.* MPS 10.0/12.1/18.4s versus 4090
2.181/2.848/3.610s by bucket: a 4.3-5.1x speedup, both runs verified at 318
rows with zero failures. Against a ~1s spec and a shipped Piper build at 0.5s,
the voice stage alone is 2.2-3.6s — four to seven times the whole budget.
The 4090 made the words-to-latency line about six times shallower without
making it flat. See `results/chatterbox_mps_vs_4090/ANALYSIS.md`.

### P13. A dependency can disable Chatterbox silently, after a 4 GB download
`perth/__init__.py` swallows an ImportError and sets `PerthImplicitWatermarker`
to `None`; Chatterbox then calls it and raises `TypeError` far from the cause.
Trigger here was setuptools ≥ 82 dropping `pkg_resources`. Pinned in
`requirements-chatterbox.txt`; `tools/diagnose_chatterbox.py` surfaces the real
error. **If Chatterbox is ever adopted, this needs a startup check in
production** — it is exactly the silent-success failure this project keeps
paying for.

### P14. Cold start is 26.5s on MPS
Model load 13.9s plus warmup 12.60s. Argues for a long-lived server over
per-request loading. Not on the request path once warm.

### P15. First-audio latency is bounded below by whole-chunk synthesis
Independent of device, and now measured on two. Because the path is one-shot,
the listener waits for the **entire chunk** before any sound is possible:

    first playable audio = chunk_audio_seconds / realtime_factor

Both benchmarks recorded `delivery_seconds = 0.000s` with `first_audio_bytes`
and `stream_begin` collapsed onto completion - the signature of exactly that,
observed independently on MPS and on CUDA.

Quantified: a 33-word chunk is ~13.2s of audio, so a 1.0s first-audio needs
~13x realtime and 0.5s needs ~26x. The 4090 delivers ~4.6x. **Hardware alone
would need roughly another 3-6x on top of a 4090**, and MPS -> 4090 was 4.6x -
there is no comparable jump left in commodity cards, and per-listener GPU
rental is not this product's cost model.

Under incremental generation the listener waits for the first *frames*: at
4.6x realtime, ~0.2s of audio would be ready in ~0.04s and first-audio would
stop depending on chunk length at all. **The gap is architectural, not
computational.** This is the finding of the whole sequence and the subject of
the next experiment.

### P16. A RunPod PyTorch template ships a torchvision that chatterbox breaks
Installing chatterbox downgrades torch to 2.6.0 and leaves the template's
torchvision (built for 2.8.0) in place. transformers imports torchvision on the
way to `LlamaModel`, and the pair dies at
`register_fake("torchvision::nms")` naming neither version. Pinned in
`requirements-chatterbox.txt` and checked in the preflight. *Closed for the
experiment layer; would need the same guard anywhere Chatterbox is deployed.*

---

## Found during the Phase 0 source audit

### P17. Chatterbox Turbo disables the expressive controls the base model has
`tts_turbo.py:266` warns that **CFG, `min_p` and `exaggeration` are ignored**
by Turbo. The base multilingual model exposes all three (`mtl_tts.py:249-260`).
FAM's requirement is a natural, expressive, premium voice *and* low TTFA, and
Turbo trades the first for the second. **Nobody has heard either variant on a
FAM script.** Base-vs-Turbo on quality and latency is now an open comparison
that the latency work did not anticipate.

### P18. The only engine with true incremental synthesis has non-commercial weights
XTTS-v2 (`coqui-tts`) genuinely streams intra-utterance - `xtts.py:592-676`
yields a waveform every ~20 GPT tokens. But `manage.py:305` marks the weights
**CPML** with `tos_required: True`, and the prompt at `manage.py:331-333` reads
"I have purchased a commercial license from Coqui" / "Otherwise, I agree to the
terms of the non-commercial CPML". **Disqualifying for a commercial product
unless a licence can be obtained**, and Coqui the company wound down, so who
can grant one is itself unresolved. Settle this before spending GPU time.

### P19. Weight licences are unverified for Chatterbox and Kokoro
**Standing rule while this is open: no unverified weight licence is treated as
commercially usable.** A favourable engineering result does not make an engine
adoptable; it makes it technically adoptable pending this check.

Both packages are permissively licensed as *code* - Chatterbox MIT, Kokoro
Apache-2.0 - but a TTS model's weights carry their own terms, as XTTS proves.
The model cards could not be reached from the build container. **Check both
before either is adopted.** Piper is GPL-3.0-or-later, which is a deliberate
decision for a commercial product even server-side.

---

## Found while building Phase 1

### P20. The seam detector was reading one sample past the join
`seam_ratios` indexed `diff[i]` for a join at sample `i`, but `diff[k]` is
`wav[k+1] - wav[k]`, so the join is `diff[i-1]`. It stepped over the
discontinuity and would have reported "no clicks" on genuinely clicking audio -
a false all-clear on the exact question Phase 1 exists to answer. Fixed before
any run, and pinned by a test that injects a step discontinuity: a clean join
now reads 1.4x and an injected click 41.9x. *Closed.*

Recorded because it is the class of fault this project keeps paying for: a
check that answers a cheaper question than the one being asked, and then
reports OK.

---

## Found by Phase 1 (stage split)

### P21. Chatterbox latency is the autoregressive token loop, 92.7% of it
**Verified.** Measured on a 4090 across three chunk lengths: T3 mean **92.7%**,
max 93.1%. Flow, HiFiGAN, the watermark and text prep share the remaining
**7.3%**. Against the 2.848s medium-bucket figure, T3 is **2.640s** and
everything downstream is **0.208s** - so removing the downstream cost entirely
still leaves 2.6x the ~1s target and 5.3x the shipped Piper build.

**Consequence: the `finalize` and `cache_source` primitives that motivated the
whole investigation are irrelevant to the outcome.** They are real, and they
sit downstream of the cost. The most they could ever remove is under 8% of a
wait that is 3-6x too long. Sub-second Chatterbox is reachable only by making
T3 itself incremental, which is a fork of a third-party model's inference core.
*Open; see `results/chatterbox_stage_split/ANALYSIS.md`.*

### P22. Both chunked flow/hift modes failed, and the cause looks fixable
`recompute_prefix` and `delta_only` both errored. Analysis points at
conditioning/shape and the `flow.py:170` lookahead trim rather than anything
fundamental - the primitives are used in production by CosyVoice-derived
stacks, and the probe drove them from outside a streaming loop that does not
exist. **Not a blocker in itself, and not worth fixing on its own**: it would
have to be done as part of the T3 rewrite, and it buys at most the 7-8% those
stages own. *Open, deprioritised.*

### P23. We have still never heard Chatterbox
Not on a FAM script, not at all. Every measurement in this sequence was
latency. Combined with P17 (Turbo disables the expressive controls) and P19
(weight licence unverified), this is the reason not to spend weeks forking it
yet: the fork would be committed before knowing whether the voice is wanted or
the weights are usable. **The listening test is now the highest-value
outstanding action on the whole voice question.**
