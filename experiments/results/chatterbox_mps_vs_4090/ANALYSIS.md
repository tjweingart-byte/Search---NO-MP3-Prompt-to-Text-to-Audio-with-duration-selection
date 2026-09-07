# Chatterbox: MPS vs RTX 4090 — awaiting the measured numbers

**Status: the run is complete and both JSON files are on the author's Mac. They
are not in this repository, so no number is written here yet.** This file is
the frame; `tools/compare_runs.py` fills it from the files rather than from
description, and the numbers go in verbatim once it has run.

    python tools/compare_runs.py \
        experiments/results/chatterbox_local_mps.json \
        experiments/results/chatterbox_runpod_4090.json

## Verification the tool performs before comparing anything

A comparison of a truncated file is worse than no comparison: it looks
finished. So the tool checks, per file, and refuses to call the result sound if
any fails:

* row count against `distinct sources x distinct trials` read from the file —
  expected 106 x 3 = 318
* failed rows, counted and their first error printed
* every row on one device, and which
* all three buckets populated
* `simulated`, `smoke_run` and `is_production_latency` flags
* cold start present and marked excluded

## What it reports

Per bucket, per device: n, p50, p90, p95, min, max, IQR of first-playable;
model p50; delivery p50; audio p50; realtime factor as audio ÷ model. Speedup
as the ratio of p50s. Words-to-latency as a least-squares slope in ms per word
with its correlation. Cold start separately, never folded into a trial figure.

Anomalies checked explicitly: rows where delivery exceeds 1 ms in-process,
rows where playable precedes model completion (impossible; a clock fault), and
differing collapsed marks between the two runs.

## The architectural reading, which does not depend on the numbers

Both runs record `delivery_seconds = 0.000s` with `stream_begin` and
`first_audio_bytes` collapsed onto completion. That is the measured signature
of a **one-shot path**: no audio exists until the whole chunk is synthesised.

The consequence is a floor that hardware cannot remove:

    first playable audio  >=  chunk_audio_seconds / realtime_factor

A faster device raises the realtime factor and lowers the floor. It does not
change the shape: the listener still waits for the *entire* chunk, and the wait
still grows with chunk length. That is why the words-to-latency slope is
reported — it is the coefficient of the thing hardware does not fix.

**What this proves and what it does not.** It proves our *current
implementation* is one-shot, and that `chatterbox-tts` 0.1.7 exposes no
incremental interface. It does **not** prove Chatterbox cannot generate
incrementally. Two of its three stages look built from streaming-capable parts
(`s3gen/hifigan.py` accepts a `cache_source`; `t3.inference_turbo` appends
tokens in a Python loop). The open question is the watermarker, which currently
runs over the finished waveform. See `../../adapters/CHATTERBOX_STREAMING.md`.

## Filling the diagram

    USER ASKS → EXA → CLAUDE → FIRST SPEAKABLE TEXT → CHATTERBOX → FIRST PLAYABLE AUDIO

    CHATTERBOX (MPS,  development) = 12.148s   medium bucket, 33 words
    CHATTERBOX (4090, development) = pending the file being read

Both remain **development** figures. Neither is production latency: both are
in-process on a single machine with no network, no concurrency and no queueing.
