# Chatterbox Turbo — LOCAL MPS / DEVELOPMENT BENCHMARK

**Apple silicon, not the production deployment target. These are not production
latency figures.** They establish that the instrument works end to end on the
real model and real FAM text, and they set a floor that a GPU has to beat by a
wide margin.

Raw data: `chatterbox_local_mps.json` (carries `is_production_latency: false`).

## What ran

| | |
|---|---|
| model | `chatterbox.tts_turbo.ChatterboxTurboTTS`, `from_pretrained(device="mps")` |
| weights | `ResembleAI/chatterbox-turbo`, ~4.04 GB, 24000 Hz |
| voice | the checkpoint's built-in `conds.pt`; `generate(text)` takes no voice argument |
| input | 106 real FAM first chunks from `warm_first_token`, 3 trials each = **318 generations** |
| excluded | 3 chunks cut mid-sentence by the harness fallback (all `E-max-tokens-96`) |
| device check | `model.device` re-read after load; a CPU fallback would have refused |

## Result

Text submitted → first playable audio, median per bucket:

| bucket | n | median words | first playable | model | delivery |
|---|---|---|---|---|---|
| short | 102 | 28 | **10.048s** | 10.048s | 0.000s |
| medium | 99 | 33 | **12.148s** | 12.148s | 0.000s |
| long | 117 | 41 | **18.368s** | 18.368s | 0.000s |

Cold start, measured once and excluded from every figure above:

| | |
|---|---|
| model load | 13.9s |
| warmup generate | 12.60s |
| total before the first timed trial | 26.5s |

## What this says

**1. On MPS, first-audio latency is entirely model generation.** Delivery is
0.000s at every length. In-process there is no encode, no socket and no
transfer, so the JSON/base64 versus streaming-PCM question — which measured a
real 0.879s → 0.190s difference over HTTP — is worth exactly nothing here. It
is a *remote* problem. That is a genuine result, not an absence of one: it
locates the entire wait inside the model.

**2. Latency scales with text length, and roughly linearly.** 28 → 41 words
(+46%) costs 10.0 → 18.4s (+83%). Longer chunks are worse than proportionally
worse, which matters because FAM's chunk rule produces 25-59 word chunks.

**3. MPS runs at approximately real time, which is disqualifying.** At FAM's
`TARGET_WPM = 150`, a 28-word chunk is ~11.2s of audio produced in 10.0s
(~1.1x); a 41-word chunk is ~16.4s of audio produced in 18.4s (~0.9x — slower
than real time). *These ratios are derived from the word counts, not measured;
the file's own `realtime_factor_p50` is the measured figure and should replace
them here.* Either way, a voice that generates at about the speed it speaks
cannot serve a streaming product: the listener catches up with the generator.

**4. Against the product spec, this is not close.** FAM's one-sentence spec is
audio within about a second, and the current measured build is 0.5s to first
audio on Piper. Chatterbox on MPS is **10-18 seconds** — 20-37x the entire
existing budget, for the voice stage alone. Nothing about prompt tuning,
connection reuse or chunk thresholds is on this scale.

**5. Cold start is 26.5s and would be paid at every process boot**, which is an
argument for a long-lived server rather than per-request loading — but it is
not on the request path once warm, and it is excluded here.

## What this does NOT say

It does not say Chatterbox is too slow for FAM. It says *Chatterbox on Apple
silicon* is. The recovered Runpod benchmarks ran on an RTX 4090, and a 4090 is
expected to be very substantially faster than MPS. Whether it is fast enough is
the next measurement, and it is unanswered until it is made.

## Filling the diagram

    USER ASKS → EXA → CLAUDE → FIRST SPEAKABLE TEXT → CHATTERBOX → FIRST PLAYABLE AUDIO

    CHATTERBOX = 12.1s   (LOCAL MPS, development; medium bucket, 33 words)
                         model 12.1s + delivery 0.000s
                         NOT production latency

The production box stays **TBD** until the GPU run. This figure fills the
development column only.
