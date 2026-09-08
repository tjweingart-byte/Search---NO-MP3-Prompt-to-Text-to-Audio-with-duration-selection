# Phase 6 — Claude decoupled from Chatterbox, and speech-sized chunks

**Preserved before the pod is touched.**

## What Phase 5 established, and the one number it could not mean

Phase 5 proved the architecture: the first speakable chunk reached Chatterbox
while Claude was still writing, playback never stalled, and

> **Search → First Listen = 4.659s warm.**

That number is the baseline and is what Phase 6 must not damage.

It also reported `claude_total = 66.668s` — with `backpressure = 64.273s`
inside it. Phase 5 bounded the TTS queue at production's `QUEUE_DEPTH = 4` and
put it **directly under the Claude reader**. When Chatterbox fell behind, the
queue filled and `await queue.put(...)` stopped the reader. So TTS throughput
decided when Claude appeared to finish.

**A caveat on the obvious arithmetic.** `66.668 − 64.273 = 2.395` is *not*
Claude's generation time. It is how long our consumer spent actually reading.
While the reader is blocked, the HTTP response is still arriving into the SDK's
buffers and the socket, or is being flow-controlled to a stop — and which of
those happened is not observable from the outside. Phase 5 simply cannot
answer how long Claude took. Phase 6 answers it by never blocking the reader,
which is the only honest way to get the number.

## The architecture

```
Claude stream
  -> reader            never touches the TTS queue
  -> script buffer     bounded by CHARACTERS (64,000 ~= 20 episodes)
  -> assembler         whole sentences -> speech-sized chunks
  -> TTS work queue    bounded at production's QUEUE_DEPTH = 4
  -> Chatterbox Base
  -> playback model
```

The assembler **may** block on a full TTS queue, and in a real run it will.
That is fine: it sits downstream of the buffer, so Claude keeps being read
regardless. The reader can only block on the script buffer, and a three-minute
FAM episode is about 2,700 characters, so 64,000 is roughly twenty of them —
a genuine bound that no realistic episode approaches. If it is ever reached,
`reader_blocked_seconds` records it and the run fails.

**Text is the cheap half.** That is the entire justification for splitting the
queue in two: buffering a whole script costs kilobytes, while buffering the
audio for it costs hundreds of megabytes. Phase 5 conflated them.

## The chunk policy, and where its numbers come from

`experiments/speech_assembler.py`. Every threshold has a derivation, and none
was chosen because a round number looked reasonable.

### The evidence available

The only Chatterbox-on-4090 measurements in this repository
(`experiments/results/chatterbox_mps_vs_4090/ANALYSIS.md`):

| words | generate |
|---|---|
| 28 | 2.181s |
| 33 | 2.848s |
| 41 | 3.610s |

Least squares: **0.1086 s/word, intercept −0.81s, R² 0.991.**

Two consequences, and the first is uncomfortable:

1. **A negative intercept means no measurable fixed per-invocation cost
   between 28 and 41 words.** Cost per word actually *rises* with length
   (78 → 86 → 88 ms/word). So **batching does not buy compute efficiency** in
   the measured range. The intuitive reason for batching is not supported by
   the data, and saying so is more useful than repeating it. The real reasons
   are delivery quality, fragmented pacing, and queue churn.
2. **The fit must not be extrapolated below 28 words** — it goes negative at
   seven. There is no 4090 measurement of a tiny chunk anywhere here, because
   the benchmark corpus was built with a 25-word floor. What a two-word
   synthesis costs is genuinely unknown.

`tools/fit_chunk_policy.py` closes that gap from the other end: give it a real
`results.json` and it fits generation time against word count **from that run's
own timings**, reports the measured distribution, and replays the assembler
over the same sentences to show what it would have produced.

### The first chunk: a latency path with no size rule at all

**No word floor.** The first complete, speakable thought `stream_sentences`
produces goes to Chatterbox on the offer that produced it, at eight words or at
twenty. Time to first listen is the highest-priority metric and no batching
consideration outranks it. The minimum, target and cap begin only *after* that
first release.

The single safeguard is semantic rather than dimensional: the pending text must
end on terminal punctuation - `.`, `!` or `?`, optionally inside a closing quote
or bracket, the same shape production's `_SENTENCE_END` looks for. A half-written
clause is not a speakable thought. In practice `stream_sentences` yields only
complete sentences, so this fires approximately never; it exists for the
degenerate stream, and even then the wait timer and the end-of-script flush
still release the text rather than holding it.

There is deliberately **no first-chunk word knob**, not even one defaulting to
zero. A knob left behind is an invitation to turn it back on.

#### What the removed floor was protecting, now measured instead

An earlier draft held the opening to twelve words, from a real constraint: one
TTS worker means chunk 2 is synthesised while chunk 1 plays, so
`audio(chunk 1) >= generate(chunk 2)`.

The band where that actually bites is narrower than it looks. At
`TARGET_WPM = 150` break-even is about **5.6 words** against a target-sized
follower (28 words, ~2.23s) and about **10.2 words** against the largest allowed
(45 words, ~4.07s). It takes an opening under roughly six words - or under
eleven against a cap-sized second chunk - to stall the first handoff at all.

The constraint has not gone away, so the run reports it rather than enforcing
it. `playback_report()["first_handoff"]` gives the opening's audio duration
against the second chunk's generation time, with the margin and a plain verdict,
and the executive summary states it. **A risk that is measured and named is a
decision; a risk silently designed out is a different product.**

#### The cost it does have, and where it shows up

On the calibrated stub, a nine-word opening covers the second chunk by 0.08s -
but leaves headroom so thin that the assembler's headroom rule fires for the
next five chunks, shipping 9-14 word chunks until it recovers at chunk seven.
So the fragmentation batching was meant to remove is not eliminated by removing
the floor; it moves from chunk 0 into the chunks just after it, and is paid in
delivery rather than in first-listen. That is the trade the hierarchy asks for,
and `chunks_forced_by_headroom` and `headroom_recovered_after_chunk` are
reported so it can be seen rather than inferred.

Read against Phase 5: its first chunk took 2.757s to synthesise, which the curve
puts at about 33 words. Nothing here would have changed it, so this still
predicts **no change to Phase 5's first-listen latency** - and now it cannot
lengthen it either, because there is nothing left to wait for.

### Later chunks

These begin only after the first chunk has been released.

| knob | default | why |
|---|---|---|
| `min_words` | 18 | below this a chunk is a fragment; above it a natural beat may end one |
| `target_words` | 28 | the *bottom* of the measured range — cost per word rises with length, so aiming high would be aiming at the worse end |
| `max_words` | 45 | just past the top of the measured range; beyond 41 words nothing has been measured on a 4090, so the cap stays near the evidence |
| `max_wait_seconds` | 2.0 | text is never held longer than this, whatever its size |
| `headroom_floor_seconds` | 3.0 | when the listener is about to catch up, batching stops mattering and the pending text ships however short |
| `prefer_break_after` | `?` `!` | a chunk past the minimum ends at a natural beat |

Release happens on the first of: the cap, low headroom, the target, a natural
beat past the minimum, the wait timer, or end of script. **Nothing is ever held
indefinitely, and a short final fragment always ships.**

### Not a second text-processing system

The assembler never splits a sentence, never reorders, and never rewrites. It
only decides **how many** of production's sentences travel together. All the
semantic work — sentence boundaries that keep quotes and brackets attached,
`clean_for_speech`, the `<<NEXT:` hold-back, the word safety valve — remains
`script_generator.stream_sentences`, unmodified. A sentence longer than
`max_words` is spoken whole and oversized rather than cut, because cutting it
would damage meaning.

That is also what makes assertion 6 checkable: the spoken text is exactly the
concatenation of production's own sentences, so the only transformations
between Claude's output and the audio are production's documented ones.

## The assertions

`decoupled_pipeline.phase6_problems()` — empty, or the run fails with a
non-zero exit after writing every artefact.

| # | fails when |
|---|---|
| 1 | `first_tts_start >= claude_complete`, when the script produced more than one chunk |
| 2 | the first synthesis is not the first assembled chunk |
| 3 | the first synthesis is the whole final script |
| 4 | **`reader_blocked_seconds` exceeds 50ms** — TTS saturation reached upstream |
| 5 | text was lost, duplicated, reordered, or a chunk synthesised twice |
| 6 | the spoken text is not exactly the sentences the chunker emitted |
| 7 | a stub run could be mistaken for a benchmark |

**Assertion 1 is scoped, deliberately.** A script producing a single chunk
releases it at flush, after the stream ends, so TTS *cannot* start before
`claude_complete`. Demanding overlap there would be a false failure, so the
evidence block reports `overlap_applicable: false` instead of inventing a pass.

**Assertion 4 needs a second half, or it is vacuous.** A run where Chatterbox
never fell behind would satisfy it without testing anything.
`decoupling_evidence()` therefore reports whether the decoupling was
**exercised** — the TTS queue actually saturated, or Claude finished with a
backlog waiting — and says "not exercised by this run" rather than claiming a
proof.

**And it needs teeth.** `run_decoupled(coupled=True)` rebuilds Phase 5's shape
on purpose: the reader writes straight onto the TTS queue, one sentence per
synthesis. `tools/phase6_experiment.py --coupled` runs it, and the test suite
requires it to *fail*. On the calibrated stub the contrast is exactly the Phase
5 complaint:

| | decoupled | coupled fixture |
|---|---|---|
| TTS invocations | 11 | 30 |
| median words | 33 | 10 |
| chunks under 10 words | 0 | 12 |
| reader blocked | 0.000s | 0.280s |
| assertion | passed | **failed** |

## The five timing concepts, kept apart

The report must make it impossible to confuse "Claude took X" with "we stopped
reading Claude for X".

| name | what it is |
|---|---|
| `claude_stream_seconds` | Claude's own stream, with nothing downstream holding it |
| `claude_local_processing_seconds` | our work inside the reader loop |
| `claude_reader_blocked_seconds` | time **our architecture** stopped reading Claude |
| `assembler_blocked_seconds` | downstream blocking — allowed, and not Claude's |
| `tts_total_seconds` | all Chatterbox compute |

Plus the playback headroom series, sampled at every chunk arrival.

## Playback, modelled as a listener

Playback begins the moment chunk 0 finishes synthesising and then runs in real
time with production's 0.12s `SENTENCE_GAP` between chunks. For every later
chunk the model records when the listener reaches it, when its audio was ready,
and the shortfall if any.

**Headroom is sampled *before* each arriving chunk is folded in.** Sampling
after would make an underrun impossible to see — the fold repairs the timeline,
so the minimum would never go negative. That was a real bug in the first draft,
caught by a stub test rather than by the meter.

Reported: `playback_stalls`, `total_stall_seconds`,
`minimum_playback_headroom`, `median_playback_headroom`,
`maximum_playback_headroom`, headroom at Claude complete and at final
synthesis.

## Cold and warm

Chatterbox's model load is measured once, inside the cold run, and labelled
**infrastructure, paid once per process, never by a listener on a warm
server**. Cold and warm first-listen are reported separately and never blended.

## Success criteria, and what would not count as success

Primary, warm: first listen ≤ 5.0s; `first_tts_start < claude_complete`; zero
playback stalls; no text loss; the Claude reader never blocked.

Secondary: materially fewer tiny TTS calls; natural boundaries; a clean
independent Claude number; Chatterbox comfortably ahead.

**One request per condition.** A first-listen difference of a few hundred
milliseconds against Phase 5 is inside API variance and is not evidence of
anything. The report says which side of the budget it landed on and does not
convert one sample into a trend.

## Validated before renting

`python3 -m pytest tests/test_speech_assembler.py tests/test_decoupled_pipeline.py tests/test_phase6_runner.py -q`
— 62 tests, no GPU, no key, no network. Among them: a complete first sentence
of two, five, eight, nine and fifteen words each released on the offer that
produced it, with nothing waited for; the same short opening reaching the voice
un-merged through the whole pipeline; a slow voice that does not
stop the reader; the coupled fixture being rejected; decoupling reported as
*untested* when Chatterbox never falls behind; a stall detected when it should
be; text loss, duplication and reordering each detected; nothing held
indefinitely; a short final fragment shipping; a sentence longer than the cap
never split; and the stub fence holding in both directions.

The stub's voice follows the measured 4090 curve, divided by ten so the
rehearsal takes seconds while the *ratios* — and therefore the queue behaviour,
the headroom and the stalls — stay real. The policy's two wall-clock rules are
divided by the same factor, or they would fire on a timeline ten times shorter
than the one they were set for.

## The command

See `experiments/RUNPOD_RUNBOOK.md`, Phase 6.
