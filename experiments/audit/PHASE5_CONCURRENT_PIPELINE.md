# Phase 5 — the real FAM pipeline: Claude and Chatterbox running at once

**Preserved before the pod is touched.** Everything here can be read and argued
with while the meter is off.

## What the previous run got wrong

The combined run (Phase 4) measured `Exa → Claude → Chatterbox` **in
sequence**. It detected the first speakable chunk at about 2.4s and then waited
for `claude_complete` before handing anything to the voice. Every stage was
real, and the total was honest — but the shape was not FAM's.

FAM's shape is in `pipeline.py`:

```python
pump = self._start(self.generator.stream_sentences(plan, notes))   # bounded queue
async for sentence in pump: ...                                    # speak as they land
```

`_start` consumes the sentence stream **now**, into an `asyncio.Queue` bounded
at `QUEUE_DEPTH = 4`, and the speaking loop pulls from it. The model is still
writing sentence twelve while sentence one is being synthesised. That is the
architecture, and it is the thing that has never been measured.

So Phase 4's `search_to_first_listen` is a ceiling, not the product's number.
Phase 5 measures the product's number.

## The rule this experiment exists to enforce

> **The first complete speakable chunk goes to Chatterbox the instant it
> exists.** Not after Claude finishes. Not after the script is concatenated.

Stated as an assertion, in `concurrency_problems()`:

| check | fails when |
|---|---|
| overlap | `first_tts_start >= claude_complete` — a sequential pipeline |
| identity | the first synthesis text is not the first chunk the chunker emitted |
| no concatenation | the first synthesis text is the whole final script |
| ordering | chunks were synthesised out of order |

A failing run **writes all its results and then exits non-zero**. The evidence
for why it failed is the point; losing it to a crash would waste the rental.

`tests/test_concurrent_pipeline.py` includes a deliberately sequential run —
collect every sentence, then synthesise — and asserts that the guard rejects
it. A guard that has never rejected anything is not known to work.

## Reusing the real chunker, not a copy of it

The chunking is production's `ScriptGenerator.stream_sentences`, unmodified.
That matters because the chunker is not just a regex:

* `_SENTENCE_END` — `(?<=[.!?])["')\]]*\s+`, so quotes and brackets stay with
  their sentence;
* `clean_for_speech` — strips stage directions, markdown, `Host:` prefixes;
* the `<<` hold-back — everything from an unmatched `<<` is the Go Deeper
  marker, which can arrive split across deltas and must never be spoken;
* the 1.35× word safety valve.

Reimplementing that in the harness would measure a copy of production. So the
experiment calls the real method.

The one thing `stream_sentences` does not expose is time-to-first-*token*: it
yields sentences. Rather than fork the loop to see it, the harness wraps the
**client** (`_TimedStream`), marks the first text delta, and passes everything
through untouched. Production code is unchanged; only the object it streams
from is a proxy.

`QUEUE_DEPTH` and `SENTENCE_GAP` are imported from `pipeline.py` rather than
copied. A setting is settled only where it is copied — so it is not copied.

## Why the concurrency is real and not an artefact of asyncio

`model.generate()` is a blocking torch call. Awaiting it directly on the event
loop would stall the Claude stream and produce a pipeline that *looks*
concurrent in code and is sequential in fact.

So the harness hands it to a thread with `asyncio.to_thread` — which is exactly
what `tts.py:273` already does for Piper's blocking synth. The Claude stream is
network-bound and advances while the torch thread holds the GPU.

**The honest caveat, stated before the numbers exist:** Chatterbox's T3 stage is
an autoregressive Python loop, so it holds the GIL between CUDA launches. The
Claude stream will therefore advance somewhat more slowly than it would alone.
That is a real cost of this architecture, not a measurement error, and it is
visible: compare `claude_total` here against `claude_total` in the Phase 4
sequential run, same model and same query. If it has inflated, that is the
price of the overlap, and it is almost certainly worth paying — but it should
be read, not assumed away.

## Backpressure is measured, not designed out

Phase 1 measured Chatterbox at roughly 4-5× realtime, and Claude writes a
three-minute script in far less than three minutes. So the queue will fill, and
production's bound of 4 will throttle the model.

That is production behaviour and is kept. But it means `claude_complete` is not
purely "Claude finished": part of it is the model waiting for the queue. So
every blocked `put` is timed and reported as `backpressure_seconds`, separately
from the stream. Folding it in would make Claude look slower than it is.

## Falling behind the listener

The number that decides whether this architecture works is not throughput, it
is **dead air**. `playback_analysis()` runs a virtual playback clock: audio
starts the moment chunk 0 finishes synthesising, then plays in real time with
production's 0.12s between chunks. Any chunk whose synthesis finishes after the
listener has arrived at it is an underrun, reported with its index, the moment
it was needed, and the moment it was ready.

`kept_ahead: true` with positive `headroom_seconds` means the pipeline never
stalls. Anything else names exactly where it did.

## What is measured

Marks, on one monotonic clock, recorded and never inferred:

```
request_start, exa_start, exa_complete,
claude_start, claude_ttft,
chunk_ready:NN            (every chunk)
first_speakable_chunk_ready
tts_start:NN, tts_complete:NN   (every chunk)
first_tts_start, first_tts_complete, first_playable_audio
claude_complete, final_audio_complete
```

Derived, per chunk: words, audio duration, generation time, realtime factor,
time spent waiting in the queue. Derived, per run: peak queue depth, the full
queue-depth series, backpressure, overlap, `search_to_first_listen`,
`search_to_complete_audio`.

Two stages stay declared-unavailable with their reasons, as in Phase 4: Exa has
no observable intermediate point, and Chatterbox Base is one-shot so the first
playable moment is the first chunk's completion.

## Audio

Every chunk is written in order as `audio/<run>/chunk_NN.wav`, and one
`audio/<run>_episode.wav` concatenates them with production's `SENTENCE_GAP`
between — so what you hear is what the app would have played, not a tighter
edit of it. `audio/<run>_script.txt` lists the chunks as they were spoken.

## The stub mode, and why it is fenced

`--dry-run` runs the entire path — the same queue, the same assertions, the
same audio writing and the same analysis — with no model, no GPU and no API
calls. It is how the wiring is proved before renting, and Gate 4 of the pod
script runs it on the pod as a free check that the tarball is sound.

Demo modes have cost this project a session before, so this one is fenced hard:

* every artefact carries `stub: true` and a label reading **NOT A RESULT**;
* the output directory name *must* contain `STUB`, and a real run *may not*
  write into one — both enforced, both tested;
* the report title does not mention the card, and a test asserts the string
  `RTX 4090` never appears in a stub's `ANALYSIS.md`.

## Cost

| item | estimate |
|---|---|
| stub gate | $0 |
| preflight | ~$0.006 (one real Exa search, one 8-token Claude call) |
| two runs, cold and warm | ~$0.07 (one Exa search and one 3-minute script each) |
| GPU | a few minutes on a pod that is already configured |

Actual Exa cost comes from Exa's own response. `stream_sentences` does not
return a usage block, so Anthropic's actual spend is not reported here — that
is recorded as a note in the results rather than estimated and presented as
measured.

## Validated before renting

`python3 -m pytest tests/test_concurrent_pipeline.py -q` — 19 tests, no GPU, no
key, no network. They hold the harness to: TTS starting before Claude finishes;
a sequential run being rejected; the first synthesis being the first emitted
chunk; chunks in order; mark ordering; queue depth sampled on both sides;
backpressure measured when the queue is bounded; an underrun reported when the
voice is slower than playback; headroom reported when it is faster; the whole
runner path producing chunk files and a gapped concatenation; and the stub
fence holding in both directions.

## The command

See `experiments/RUNPOD_RUNBOOK.md`, Phase 5.
