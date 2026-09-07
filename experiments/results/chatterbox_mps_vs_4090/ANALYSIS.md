# Chatterbox first-audio: Apple MPS vs RTX 4090

**Both runs are DEVELOPMENT BENCHMARKS.** In-process on one machine, no
network, no concurrency, no queueing. Neither is production latency.

Verified before comparison, both files: **318 rows, 318 ok, 0 failed,
verdict OK** — 106 distinct chunks x 3 trials, all rows on one device, all
three buckets populated, cold start recorded and excluded.

## The measurement

Text submitted -> first playable audio, median per bucket:

| bucket | words | MPS | RTX 4090 | speedup |
|---|---|---|---|---|
| short | 28 | 10.048s | **2.181s** | 4.6x |
| medium | 33 | 12.149s | **2.848s** | 4.3x |
| long | 41 | 18.368s | **3.610s** | 5.1x |

Same 106 real FAM chunks, same buckets, same 3 trials, same model, same metric
definitions. The only variable is the device.

*Distribution detail (p90, p95, min, max, IQR), the measured `audio_seconds` and
`realtime_factor` per bucket, the least-squares words-to-latency slope over all
318 rows, and the 4090 cold-start figures are all in the two JSON files and are
printed by `tools/compare_runs.py`. They are not transcribed here yet — only
the medians above and the verification counts were read back. Everything below
marked* derived *is arithmetic on those medians, not a figure from the files.*

## What the 4090 bought

**A real speedup, and not enough.**

*Derived* realtime factor, at FAM's `TARGET_WPM = 150` — the files carry the
measured `realtime_factor_p50`, which supersedes this:

| bucket | audio (derived) | MPS | 4090 |
|---|---|---|---|
| short | ~11.2s | ~1.1x | ~5.1x |
| medium | ~13.2s | ~1.1x | ~4.6x |
| long | ~16.4s | ~0.9x | ~4.5x |

MPS generated at roughly the speed it speaks — on the long bucket, *slower*
than real time. The 4090 generates at about 4.5-5x real time. That is the
difference between an engine that cannot serve a streaming product at all and
one that can, in principle, keep ahead of a listener.

*Derived* cost per word, from the three bucket medians (the file's
least-squares fit over 318 rows is the authoritative figure):

| | ms per word |
|---|---|
| MPS | ~653 |
| 4090 | ~109 |

**The slope fell by about six times, but it is still a slope.** That is the
whole finding: the 4090 made the line shallower without making it flat.

## Against the product spec

FAM's one-sentence spec is audio within about a second. The current shipped
build measures **0.5s to first audio on Piper**.

| | first audio |
|---|---|
| FAM spec | ~1s |
| FAM today (Piper) | 0.5s |
| Chatterbox, 4090, medium chunk | **2.848s** |

Chatterbox on a 4090 is **2.2-3.6s for the voice stage alone** — four to seven
times the entire existing budget, before search or generation is counted. It is
much better than MPS and it is still disqualifying at the target.

## Why more hardware does not close the gap

The path is one-shot: no audio exists until the whole chunk is synthesised. So

    first playable audio  =  chunk_audio_seconds / realtime_factor

*Derived* from that identity: a 33-word chunk is ~13.2s of audio, so reaching
1.0s requires **~13x** realtime and reaching 0.5s requires **~26x**. The 4090
delivers ~4.6x. Closing the gap by hardware alone would need roughly another
**3x on top of a 4090** for the weaker target and **6x** for parity with Piper.
Going MPS -> 4090 bought 4.6x; there is no comparable jump left in commodity
GPUs, and renting one per listener is not the cost model this product has.

**The gap is architectural, not computational.** Under incremental generation
the listener waits for the first *frames* of audio, not the last — at 4.6x
realtime, ~0.2s of audio would be ready in ~0.04s, and first-audio would stop
depending on chunk length at all. The distance between 2.848s and that is
entirely the one-shot structure.

## The evidence that the path is one-shot

Both runs record, on every trial:

* `delivery_seconds = 0.000s`
* `stream_begin` and `first_audio_bytes` **collapsed** onto completion

In-process there is no encode, no socket and no transfer, so delivery is
genuinely zero and the entire wait is inside `generate()`. That is the measured
signature of whole-chunk synthesis, on both devices independently.

It also settles a subsidiary question: the JSON/base64 versus streaming-PCM
contract, which measured 0.879s -> 0.190s over HTTP, is worth **nothing**
in-process. It is a remote-deployment concern only.

## What this proves, and what it does not

**Proved.** Our current implementation is one-shot. `chatterbox-tts` 0.1.7
exposes no incremental interface — `generate()` returns one completed tensor
and there is no `yield` anywhere in the package. First-audio latency is
therefore bounded below by whole-chunk synthesis on any device.

**Not proved, and previously overstated here.** That Chatterbox *cannot*
generate incrementally. Two of its three stages look built from
streaming-capable parts: `s3gen/hifigan.py` accepts a `cache_source` — the
chunked-vocoding hook in CosyVoice-derived stacks — and `t3.inference_turbo`
accumulates tokens in an appending Python loop. The open question is the
watermarker, which currently runs over the finished waveform. Nobody has tried
any of this. See `../../adapters/CHATTERBOX_STREAMING.md`.

## Anomalies

None reported by the comparison: no failed rows, no non-zero in-process
delivery, no impossible timings, and the collapsed marks matched across both
runs — which is itself the finding above, observed identically on two devices.

## Filling the diagram

    USER ASKS → EXA → CLAUDE → FIRST SPEAKABLE TEXT → CHATTERBOX → FIRST PLAYABLE AUDIO

    CHATTERBOX (MPS,  development) = 12.149s
    CHATTERBOX (4090, development) =  2.848s   model 2.848s + delivery 0.000s
                                                medium bucket, 33 words

**2.848s is the defensible measured number** for the box, with the caveats that
it is a development benchmark and that it is a floor set by architecture rather
than by the card.
