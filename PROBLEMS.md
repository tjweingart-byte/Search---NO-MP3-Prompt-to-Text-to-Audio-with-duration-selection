# Problems encountered, and how each one is solved

This is the honest engineering log for the project: every problem that came up
while building it, the solution that shipped, and the ones that are still open.

---

## 1. "It converted the search to an MP3 file" — the problem you hit last time

**Why that happens.** The obvious design is: generate script → synthesise the
whole thing → encode → write `episode.mp3` → hand the browser a URL. Every step
is blocking, so the listener waits for the *entire* episode before hearing the
first word. A 10-minute podcast means a 10-minute-ish wait, plus disk writes,
plus an ffmpeg/LAME dependency.

**Solution — never materialise a file.** The pipeline is:

```
Claude tokens → sentence buffer → TTS subprocess → raw PCM → HTTP chunk → speakers
```

- The TTS engine writes **raw 16-bit PCM** (`piper --output_raw`; espeak's WAV
  header is stripped in `audio_utils.strip_wav_header`). No encoder is involved
  at any point — there is no ffmpeg, LAME or `pydub` dependency in this repo.
- Those bytes go straight into a `StreamingResponse` (`app.py`).
- The browser reads the response with the fetch streams API and schedules each
  chunk on a Web Audio clock (`static/app.js`). No `Blob`, no object URL, no
  file.

**Result:** audio starts as soon as the *first sentence* is written — about a
second or two — and the rest is produced while the listener is already
listening. Measured here at ~200x realtime for synthesis, so the model is the
only thing anyone waits on.

### "If that is not possible…" — it is, but here's the tradeoff you're buying

Uncompressed PCM is bulky: **~2.6 MB per minute** at 22.05 kHz mono 16-bit, so a
10-minute episode is ~26 MB on the wire (measured). That's the price of not
encoding. It is the right trade for a stream you play once and discard, and
poor for something you'd store or re-download. If bandwidth becomes the
constraint, see §9 — you can halve or better it *without* reintroducing a file.

---

## 2. "Write a 6-minute podcast" produces wildly variable lengths

**Problem.** Duration language in a prompt is close to useless as a length
control. The same prompt yields 400 or 1,300 words on different runs.

**Solution — three independent mechanisms**, because none is sufficient alone:

1. **Word budget** (`script_generator.plan_episode`). Minutes × 150 wpm becomes
   a hard word range, handed to the model with a per-section breakdown. Section
   *count* also scales with duration, so a 10-minute episode gets real structure
   instead of one padded monologue.
2. **Pacing controller** (`audio_utils.PaceController`). Before *every* sentence
   it re-plans: given the audio already emitted and the words still unspoken,
   what rate lands exactly on target? Small misses vanish invisibly. Corrections
   are clamped to 115–185 wpm so the voice never becomes a chipmunk.
3. **Trim / top-up** (`pipeline.PodcastPipeline`). Over-long scripts are cut at a
   sentence boundary once the remaining time can't fit the next sentence.
   Short scripts trigger a second, small Claude call (`ScriptGenerator.top_up`)
   that continues the episode — which arrives while the listener is still
   hearing earlier material, so the top-up is free in wall-clock terms.

**Measured result** (real espeak synthesis, `tests/test_pipeline.py` plus a
live end-to-end run):

| Requested | Produced | Drift |
|---|---|---|
| 1 min | 60.68 s | +0.68 s |
| 3 min | 180.00 s | 0.00 s |
| 10 min | 600.00 s | 0.00 s |

Tests also assert this holds when the model misses its budget by ±30%.

---

## 3. Naive pacing can't absorb a long script (found by a failing test)

The first implementation used only mechanisms 1 and 2 above. A script 30% over
budget needed 196 wpm to fit — past the 185 wpm clamp — so a 5-minute request
produced **6 min 18 s** of audio. The clamp is not negotiable (faster is
unlistenable), so the fix was the sentence-boundary trim. Cutting mid-sentence
sounds broken; cutting *between* sentences is why the whole pipeline works in
sentence units.

**Residual issue:** a trimmed episode can end slightly abruptly, since the
model's planned closing line may be the thing cut. `stats.truncated` records
when this happened. A future improvement is to reserve ~10 seconds of budget for
a pre-written sign-off sentence that is always spoken last.

---

## 4. Naive pacing can't absorb a *short* script either

Symmetric failure: a 30%-short script left **64 seconds of dead air**. Padding
that with silence is not an answer at that scale. Solution is the top-up call
(§2.3). Capped at `MAX_TOPUPS = 2` so a chronically under-writing model can't
fan out into unbounded API calls; any residual gap under 6 seconds is closed
with room tone, which reads as "the episode ended" rather than as a bug.

---

## 5. Sample-rate mismatch — the silent, nasty one

**Problem.** The app assumed 22,050 Hz. espeak-ng happens to emit exactly that,
so everything worked *by luck*. Piper voices ship at 16,000 **or** 22,050
depending on the model. Feed 16 kHz samples to a player told they're 22.05 kHz
and you get audio that is pitch-shifted and ~27% short — and no test catches it,
because the byte counts still look plausible.

**Solution.** The engine is authoritative about its own rate, never the config
file. `PiperEngine` reads the `sample_rate` from the voice's sidecar JSON;
`EspeakEngine` reads it from the WAV header of its first synthesis. That rate
flows into the WAV header, the `X-Sample-Rate` response header the player reads,
and the `PaceController`'s timing maths.

---

## 6. Streaming a WAV whose length isn't known yet

A normal WAV header must declare total byte count, which doesn't exist until the
episode is finished. `audio_utils.streaming_wav_header` writes `0xFFFFFFFF` for
both size fields — the conventional live-stream trick that tells players "read
until the socket closes". This is what lets `fmt=wav` work in a plain `<audio>`
tag. The `fmt=pcm` path used by the web player skips headers entirely.

---

## 7. Claude and the TTS engine run at wildly different speeds

The model produces text in bursts; the engine synthesises ~200x faster than
realtime. Coupling them directly means one repeatedly blocks the other.

**Solution.** A bounded `asyncio.Queue` (depth 4) between them. The model writes
ahead while the current sentence plays, and the small depth caps memory at a few
seconds of audio *regardless of episode length* — a 10-minute episode uses the
same memory as a 1-minute one.

---

## 8. Assorted smaller problems

| Problem | Solution |
|---|---|
| Model emits `**bold**`, `[intro music]`, `Host:` — all read aloud as literal noise | `clean_for_speech()` strips markdown, bracketed stage directions and speaker labels before synthesis |
| 16-bit samples straddle chunk boundaries; a split sample becomes a click | Player carries the odd byte over to the next chunk (`leftover` in `app.js`) |
| Network stall pushes the Web Audio play head into the past, silently dropping chunks | Schedule at `max(playHead, currentTime + 0.05)` |
| Browsers start `AudioContext` suspended | Created inside the click handler and explicitly `resume()`d |
| An error *after* streaming starts can't become an HTTP 500 | Logged and the stream closed cleanly; player keeps what it has. Errors *before* the first byte are still proper JSON errors |
| Listener closes the tab; generation keeps burning tokens | `request.is_disconnected()` checked between chunks, generation abandoned |
| One request holds a Claude stream + TTS process open for minutes — trivially abusable | Per-IP rate limit (`RATE_LIMIT_SECONDS`, default 3 s) |
| A model ignoring the budget entirely could emit an hour of audio | Hard safety valve at 1.35x the word budget in `stream_sentences` |
| `stop_reason: "refusal"` would otherwise yield silence | Detected and turned into a spoken explanation |
| Nothing works with no TTS installed, making the pipeline untestable | `DebugEngine` emits a duration-accurate tone, so timing, transport and player are all testable offline. `/api/health` reports it so it can't be mistaken for a real voice |

---

## 9. Reducing output time and compute further

Already in place: streaming-first architecture, `effort: "low"` (script writing
isn't a hard reasoning task, and effort costs time-to-first-token directly),
sentence-level pipelining, bounded memory, top-up calls that skip web search
since research is already done, and abandonment on disconnect.

If you need more:

**Where the time actually goes.** Measured: TTS is ~330x realtime (15 ms for a
5-second sentence) and the model outputs text ~16x faster than the listener
consumes it. So after the opening sentence, *nothing downstream is ever the
bottleneck* — the script finishes long before playback does. The only latency
that matters is **time to first audio**, and that is almost entirely the model:
web search (~2-6 s) + thinking tokens + the first sentence. Optimising anything
else is optimising ~2% of the wait. Two consequences:

- Parallelising script generation across sections would not help. The script is
  already far ahead of the listener.
- **Prompt caching is not worth it here** (correcting an earlier note): the
  stable prefix is ~172 tokens, well under the 512-4096 token minimum cacheable
  prefix, so it would never cache.

**Latency — the levers that actually move it**
- **Cold-open in parallel.** Have `claude-haiku-4-5` write a one-sentence opener
  with no tools while the main model researches. Speech starts in well under a
  second and the research latency disappears behind it. Biggest single win.
- **Skip web search when it isn't needed.** It is the largest share of
  time-to-first-audio. `ENABLE_WEB_SEARCH=0` for evergreen topics, or expose it
  as a "use live sources" toggle in the UI.
- **`claude-sonnet-5` or `claude-haiku-4-5`** for the script pass: cheaper and
  noticeably faster to first token. One env var (`MODEL`).
- **Pre-warm the TTS process.** Measured: subprocess spawn is 9.4 ms, 62% of
  espeak's per-sentence cost but only ~1.6 s spread across a whole 10-minute
  episode — so it is irrelevant for espeak. It is *essential* for piper, which
  reloads its ONNX voice on every spawn. Use a persistent worker fed over a pipe
  if you switch to piper.

**Throughput and cost per episode**
- **Cache scripts by `(query, minutes)`.** The model call is the only expensive
  part of an episode; re-synthesising the audio is essentially free.
- **Batch API (50% cheaper)** for pre-generating popular or scheduled episodes
  ahead of time. Not for interactive requests.

**Server capacity.** At ~330x realtime, one CPU core covers roughly 300
concurrent listeners' worth of synthesis. The app is I/O-bound; the TTS
subprocesses are the only real CPU load, and queue depth caps memory per stream
regardless of episode length.

**Bandwidth / compute** — all of these keep the no-file architecture:
- Drop to 16 kHz mono: ~1.9 MB/min instead of 2.6, with little quality loss for
  speech.
- Opus over WebSocket or MediaSource: ~0.2 MB/min, a ~13x reduction. This is
  still a *stream*, not a file — but it does add an encoder dependency, which is
  exactly what you asked to avoid, so it's opt-in rather than default.
- Cache generated scripts by `(query, minutes)`. The expensive part is the model
  call; re-synthesising audio is nearly free at 200x realtime.

**Cost**
- Web search dominates token cost. `ENABLE_WEB_SEARCH=0` for evergreen topics.
- `MAX_WEB_SEARCHES` caps the research budget per episode.

---

## 10. Known limitations / not yet done

- **No live API key was available in the build environment**, so the Claude path
  is exercised against a scripted fake generator and the espeak path against
  real synthesis. The SDK call shape follows the current Anthropic Python SDK
  (streaming, `output_config.effort`, `web_search_20260209`), but the
  first real run should be watched.
- **espeak-ng sounds robotic.** It is the zero-dependency default. Install a
  Piper voice for a natural one — the code path is already there and selected
  automatically.
- **The rate limiter is in-process**, so it resets on restart and doesn't work
  across multiple workers. Use Redis for a real deployment.
- **No authentication.** Anyone who can reach the server can spend your tokens.
- **Trimmed episodes can end abruptly** (§3).
- **No seeking or pause.** The player is a live stream; adding transport
  controls means buffering the whole episode client-side, which is a deliberate
  tradeoff against the current instant-start design.
