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

### P10. Chatterbox Turbo cannot stream, and no configuration changes that
`generate()` returns one completed tensor; there is **no `yield` anywhere in
`chatterbox-tts` 0.1.7**. Tokens are accumulated in full, vocoded in one call,
then watermarked across the whole waveform — three sequential whole-sequence
barriers. Streamed *delivery* is possible and is built; **true model streaming
is not available without forking the package.** See `CHATTERBOX_STREAMING.md`.

### P11. The JSON/base64 endpoint contract makes first-audio equal full-audio
Measured against a local server: identical work, first playable at 0.879s under
JSON/base64 versus 0.190s under streaming PCM. **Only matters remotely** — the
MPS run measured delivery at 0.000s in-process. Fix is `/synthesise/stream`,
already built and non-production.

### P12. Chatterbox on Apple silicon is 10-18s to first audio
See `results/chatterbox_local_mps/ANALYSIS.md`. Roughly real-time generation,
which cannot serve a streaming product. **Not yet known for a GPU.**

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
