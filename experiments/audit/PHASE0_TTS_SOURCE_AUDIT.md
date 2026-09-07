# Phase 0 — source audit of candidate TTS engines

Method: download each package, read its inference path, and answer from the
code. No GPU, no weights, no accounts. Every claim below cites a file and line
in the version stated. **Nothing here is a listening judgement — nobody has
heard any of these on a FAM script.**

Versions audited: `chatterbox-tts` 0.1.7, `kokoro` 0.9.4, `piper-tts` 1.8.0,
`coqui-tts` 0.27.5.

## The distinction that decides everything

**True incremental synthesis** — the model emits playable audio *while it is
still generating the rest of the same utterance*. First audio arrives after a
fixed small amount of work, independent of how long the utterance is.

**Segment chunking** — the text is split, each segment is synthesised whole,
and segments are emitted as they complete. First audio still costs a whole
segment. This is what FAM's own 25-word chunk rule already does; an engine
doing it internally adds nothing FAM does not have.

**Streamed transport** — a finished waveform written to a socket in pieces.
Measured at 0.879s → 0.190s over HTTP, and **0.000s in-process**. Not a
generation property at all.

Only the first removes the `chunk_audio / realtime_factor` floor.

---

## Chatterbox Turbo — segment-only today, and the streaming machinery is
## already in the code

`tts_turbo.py:248-296`. `generate()` returns one completed tensor; no `yield`
anywhere in the package. That much was known.

**What this audit adds: all three stages already carry the CosyVoice streaming
protocol, and `generate()` simply does not use it.**

| stage | streaming support present? | evidence |
|---|---|---|
| T3 — text → speech tokens | **loop exists, does not yield** | `t3.py:415+` accumulates `generated_speech_tokens` in a Python list and returns at the end |
| Flow — tokens → mels | **`finalize` flag present and wired** | `s3gen.py:301-320` takes `finalize: bool` and passes it down; `flow.py:170` does `if finalize is False: h = h[:, :-self.pre_lookahead_len * self.token_mel_ratio]` |
| HiFiGAN — mels → waveform | **`cache_source` present, and it returns the next source** | `s3gen.py:324-327` `hift_inference(speech_feat, cache_source)`; `hifigan.py:463-475` `# use cache_source to avoid glitch`, returns `(generated_speech, s)` |
| Watermark | **whole-waveform only** | `tts_turbo.py:295` `self.watermarker.apply_watermark(wav, ...)` on the finished array |

`flow.py:52-53` sets `token_mel_ratio: int = 2` and `pre_lookahead_len: int = 3`
— the chunk-and-lookahead constants of the CosyVoice streaming design. The
trimming at `flow.py:170` is precisely the "this is a mid-stream chunk, hold
back the lookahead tail so the next one joins seamlessly" step.

**Where the first playable waveform could become available:** after the first
batch of speech tokens (a handful, not the whole utterance) → `flow_inference`
with `finalize=False` → `hift_inference` with the running `cache_source`. The
plumbing to do this exists and is passed `None` and `False` by
`s3gen.inference`, which calls `hift_inference(output_mels, None)` — always a
cold start, always one shot.

**The exact stages preventing it today, in order of difficulty:**

1. **`t3.inference_turbo` returns instead of yielding.** Mechanically small: the
   loop already appends per token. Making it a generator that emits every N
   tokens is a contained change.
2. **`s3gen.inference` hard-codes the one-shot path.** It never passes
   `finalize=False` and never carries `cache_source` forward, though both
   parameters exist and `flow`/`hift` honour them.
3. **The watermarker runs over the finished waveform.** This is the genuine
   unknown. Whether Perth's implicit watermark can be applied per chunk without
   audible seams or loss of detectability is a question for Resemble AI's
   design, not readable from this source.

**Verdict: incremental output looks technically feasible and materially easier
than a from-scratch fork.** Two of the three blockers are wiring that already
exists. The third is a real research question. *Not built — that is Phase 1's
decision, informed by the stage split.*

**Expressiveness caveat, and it is a significant one for FAM.**
`tts_turbo.py:266-267` warns that **CFG, `min_p` and `exaggeration` are not
supported by the Turbo version and will be ignored.** The base multilingual
model (`mtl_tts.py:249-260`) exposes `exaggeration`, `cfg_weight`,
`repetition_penalty`, `min_p` — Turbo drops the expressive controls in exchange
for speed. Voice cloning survives (`audio_prompt_path`), and the default voice
comes from a shipped `conds.pt`. **A product that wants an expressive premium
voice may be choosing the wrong Chatterbox variant.** Base-vs-Turbo on both
quality and latency is now an open question this audit did not anticipate.

---

## XTTS-v2 (coqui-tts) — the only true incremental synthesis found

`xtts.py:592-676`, `inference_stream()`. This is genuine intra-utterance
streaming:

```python
while not is_end:
    x, latent = next(gpt_generator)          # one GPT step at a time
    last_tokens += [x]; all_latents += [latent]
    if is_end or len(last_tokens) >= stream_chunk_size:      # default 20
        gpt_latents = torch.cat(all_latents, dim=0)[None, :]
        wav_gen = self.hifigan_decoder(gpt_latents, g=speaker_embedding)
        wav_chunk, wav_gen_prev, wav_overlap = self.handle_chunks(
            wav_gen.squeeze(), wav_gen_prev, wav_overlap, overlap_wav_len)
        yield wav_chunk
```

**First playable waveform: after ~20 GPT tokens**, not after the utterance.
`handle_chunks` with `overlap_wav_len=1024` does the seam handling. This is the
architecture Chatterbox lacks today.

Note it re-vocodes all latents each round and slices the new part out, so
vocoding cost grows across the utterance — but the *first* chunk is cheap,
which is what TTFA measures.

Expressiveness: zero-shot cloning from reference audio via `gpt_cond_latent`
and `speaker_embedding`, plus `temperature`, `repetition_penalty`, `speed`.

**Licensing — the blocker.** The library is **MPL-2.0**, but the XTTS-v2
*weights* are **CPML**: `manage.py:305` sets `"license": "CPML"` with
`"tos_required": True`, and `manage.py:331-333` prompts *"I have purchased a
commercial license from Coqui"* / *"Otherwise, I agree to the terms of the
non-commercial CPML"*. `.models.json` marks both `xtts_v2` and `xtts_v1.1` CPML.
**Non-commercial weights are disqualifying for a commercial product** unless a
commercial licence is obtained — and Coqui the company wound down, which makes
who can grant one a question to answer before any engineering.

---

## Kokoro-82M — segment chunking, but possibly fast enough not to care

`pipeline.py:358-386`. The pipeline *is* a generator, but it yields **per text
segment**: it splits on `split_pattern`, then `en_tokenize` (`pipeline.py:196-221`)
further splits at a 510-phoneme cap, and each piece goes through
`KPipeline.infer` → `model.forward_with_tokens` (`model.py:87-119`) which
returns a complete `audio` tensor.

**That is the same category as FAM's own chunking**, not intra-chunk
incremental. First audio still costs one whole segment.

**But the floor may not matter here.** Kokoro is 82M parameters and
non-autoregressive (StyleTTS2-style with an iSTFT vocoder, `istftnet.py`). If
whole-segment synthesis is fast enough, a segment-level floor can still land
far below 1s. **This is the single most important thing Phase 2 must measure**
— it is the candidate where architecture and outcome may disagree.

Expressiveness: voices are precomputed style vectors downloaded per name
(`pipeline.py:136-157`), blendable by weighted combination. **No zero-shot
cloning from arbitrary reference audio.** `speed` is the only prosody control
exposed. For a product wanting a distinctive premium voice, the palette is
fixed to what ships.

Licence: **Apache-2.0** on the code. Weights licence must be confirmed on the
model card.

---

## Piper — sentence chunking; the latency baseline, not a quality candidate

`voice.py:333-337`, docstring: *"Synthesize one audio chunk **per sentence**
from text."* The yield at `voice.py:433` follows a completed inference. Sentence
granularity, not incremental.

It stays in the comparison as the **measured 0.5s baseline**. Its voice quality
was already judged insufficient (P2), so it is the number to beat, not a
destination.

Licence: **GPL-3.0-or-later**. Worth a deliberate decision for a commercial
product even in a server-side deployment.

---

## Decision matrix

Legend: **established** = read from the source cited. *italic* = must be
verified experimentally or from a document this audit could not reach.

| | Chatterbox Turbo | Chatterbox (base) | XTTS-v2 | Kokoro-82M | Piper |
|---|---|---|---|---|---|
| **True incremental synthesis** | **No** — but `finalize` + `cache_source` present in flow/hift; T3 loop needs to yield | *unaudited; same s3gen stack* | **Yes** — `inference_stream`, yields every ~20 GPT tokens | **No** — per text segment | **No** — per sentence |
| **Where first waveform could appear** | after first token batch, *if* stages 1-3 are changed | *as Turbo* | **after ~20 GPT tokens, today** | after one whole segment | after one whole sentence |
| **Expected TTFA potential** | *2.848s measured today; sub-second only via the fork* | *unmeasured, likely slower than Turbo* | *low by construction — first chunk is small; unmeasured* | *unknown, possibly very low: 82M, non-AR* | **0.5s measured** |
| **Voice quality / expressiveness potential** | *unheard.* Expressive knobs **disabled**: CFG, `min_p`, `exaggeration` ignored (`tts_turbo.py:266`) | *unheard.* Exposes `exaggeration`, `cfg_weight`, `min_p` | *unheard.* Cloning + temperature/repetition controls | *unheard.* Fixed voice packs, `speed` only | **judged insufficient** (P2) |
| **Voice cloning / customisation** | reference audio via `audio_prompt_path`; default `conds.pt` | reference audio + exaggeration | **zero-shot from reference audio** | blendable preset style vectors; **no arbitrary cloning** | fixed downloaded voices |
| **Licensing** | **MIT, commercial use permitted** (P19, resolved) | **MIT, commercial use permitted** | code **MPL-2.0**; weights **CPML non-commercial, TOS-gated** — *blocker* | code **Apache-2.0**; *weights unverified* | **GPL-3.0-or-later** |
| **Self-hostable** | yes, runs today on our 4090 | yes | yes | yes | yes, ships in-app today |
| **Implementation difficulty** | **medium** — yield from T3, thread `finalize`/`cache_source`, solve watermarking | *unknown* | **low to integrate**; *licence may make it moot* | **low** | already integrated |
| **Must be verified experimentally** | stage split (t3 vs flow vs hift vs watermark); per-chunk watermarking; Turbo-vs-base quality | latency and quality | first-chunk TTFA; **commercial licence availability**; quality | **TTFA on a 4090 — the key unknown**; quality; weights licence | nothing; it is the baseline |

---

## What Phase 0 concludes

1. **Chatterbox incremental output is more feasible than assumed.** Two of the
   three blockers are unused parameters that already exist. The watermarker is
   the real question. This raises option A's standing.
2. **XTTS-v2 is the only engine with true streaming — and its weights are
   non-commercial.** Unless a commercial licence can be obtained, it cannot be
   the answer, however good the architecture is. **Settle the licence before
   spending GPU time on it.**
3. **Kokoro is the most interesting unknown.** Architecturally it is
   segment-chunked like everything else, but at 82M and non-autoregressive it
   may reach a low TTFA anyway. It is also the cheapest to try.
4. **A new question this audit surfaced:** Turbo disables the expressive
   controls the base model exposes. Given the user's requirement of a
   *natural, expressive, premium* voice, base-vs-Turbo now belongs in Phase 2
   on both axes.

**Gate 0: passed.** One candidate with true incremental synthesis (XTTS-v2,
licence-blocked), one plausible low-TTFA candidate without it (Kokoro), and a
credible path to incremental Chatterbox. Phase 1 (stage split) and Phase 2
(bake-off, including base-vs-Turbo) are both worth running.

## Explicitly not established here

* How any of these **sound**. No audio was generated. Every quality cell above
  is a capability note, not a judgement.
* ~~Weight licences for Chatterbox and Kokoro~~ — **Chatterbox is resolved**:
  MIT, commercial use permitted, per Resemble AI's official documentation and
  the MIT marking on the official `ResembleAI/chatterbox` distribution (P19).
  **Kokoro remains unverified.** Reference-voice rights are a separate
  requirement from the model licence and remain open (P19b).
* Whether XTTS's first chunk is actually fast in wall-clock terms.
* Whether Kokoro's segment floor lands under or over a second.
* Whether per-chunk watermarking is acceptable to Resemble AI's design.
