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
- **Shared script cache — implemented** (`cache.py`). See §11.
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


---

## 11. Sharing one output between different users

Implemented in `cache.py`. Two design decisions and three problems it raised.

**Cache the script, not the audio.** A 10-minute script is ~9 KB; the same
episode as PCM is ~26 MB — 2,900x larger. Re-synthesis runs at ~330x realtime,
so storing audio buys almost nothing and costs a great deal of disk. Audio also
can't be shared across durations, whereas the text is the part that cost money.

**SQLite, not an in-process dict.** The users are different people, so the hit
has to be visible to whichever worker serves the next request. SQLite in WAL
mode gives cross-process sharing and restart persistence with no new dependency.
Swap in Redis (same two methods) if you outgrow one machine.

### Problem: equivalent phrasings miss each other

"Give me a recap of week 5 of the NFL season" and "Week 5, NFL season — recap"
are the same request. Keys are therefore normalized: lowercased, punctuation
stripped, filler words removed, remaining tokens sorted.

This is lexical, so it has a real limit — "NFL week 5 recap" vs "NFL **season**
week 5 recap" differ by one meaningful word and still miss. There is a test
documenting exactly this. `CACHE_SEMANTIC_KEY=1` closes the gap with a small
model that canonicalises the topic first; the tradeoff is ~400 ms added to every
request, which is a win when traffic concentrates on popular topics and a waste
on a long tail of unique queries. Off by default.

### Problem: staleness is worse than slowness

Serving a cached "latest news" from six hours ago is a worse failure than making
someone wait. TTL is therefore query-dependent: queries containing volatile
markers ("latest", "today", "now", "breaking", "score") get 15 minutes; the rest
get 24 hours. A recap of a *completed* event is a perfect cache candidate; the
same query asked mid-event is not, which the volatile-word list only partly
catches. A classifier call would do better, and is the natural upgrade.

### Problem: an over-eager privacy filter silently disabled the cache

The first version treated "me", "I" and "we" as personal markers. That reads
sensibly and is badly wrong: "give **me** a recap of week 5" is not a personal
query, and the filter meant most real traffic bypassed the cache entirely. It
was caught by an end-to-end run showing two identical requests both missing, not
by the unit tests. The filter now matches only possessives ("my", "our"), email
addresses and phone numbers. There is a regression test for ordinary phrasing.

Anything matching is generated fresh and never stored — the bias is towards not
sharing.

### Measured

Three differently-worded requests for the same episode: **one model call, two
cache hits**. A fourth request containing "my" was regenerated and left out of
the store. On a hit the API cost is zero and time-to-first-audio drops from
seconds to the cost of one sentence of synthesis (~15 ms).

### Still open

- Cache keys include `minutes`, so a topic is generated up to 10 times. Serving
  shorter episodes by trimming a longer cached one would collapse that, at some
  cost to structure — a 3-minute episode is not a truncated 10-minute one.
- No cache warming. Pre-generating predictable episodes (this week's recap, the
  morning briefing) via the Batch API at 50% cost would make the common case a
  hit for everyone.
- `purge_expired()` exists but nothing calls it on a schedule.


---

## 12. "It generates an episode but I can't hear anything"

Reported from a real run, reproduced, and fixed. There were **two** ways the app
could produce a silent episode that every layer treated as a success.

**Cause 1: a failure before the first audio byte became a silent HTTP 200.**
The original code wrapped the whole stream in a catch-all whose comment read
"the response has already begun, so an error cannot become a 500". That is true
*after* the first byte — and false before it. So a rejected API key made the
Claude call raise, the handler logged it and closed the stream, and the browser
received `200 OK` with zero bytes. The player dutifully reported "Episode
complete" and played nothing.

Reproduced exactly:

```
$ curl -o /dev/null -w "%{http_code} %{size_download}\n" ".../api/audio?q=...&fmt=pcm"
200 0
```

**Fix.** The handler now pulls chunks until real audio exists *before* returning
a response. A failure during that window becomes a proper `502` with a message
naming the cause. The same request now answers:

```
502 {"error": "Claude rejected the credentials. Set ANTHROPIC_API_KEY in .env
     (or run `ant auth login`) and restart the server."}
```

`friendly_error()` maps auth, permission, model-not-found, rate-limit and
connection failures to something actionable rather than a stack trace.

**Cause 2: an empty script was padded into a silent "valid" episode.**
Found by a test written for cause 1. When the model returned nothing, the
end-of-episode room tone still ran, so the pipeline emitted a few seconds of
silence — enough bytes to look like a real episode to the priming check, the
`Content-Length`, and the player. The pipeline now refuses to pad an episode
containing no speech, and the endpoint rejects on `stats.sentences == 0` rather
than on a byte count, because silence is bytes but is not an episode.

**Also fixed, on the player side.** It now counts received bytes: zero bytes
raises a visible error instead of "Episode complete", and a stream ending under
half the expected length says so. The health check disables the Listen button
outright when the server reports no credentials, so the failure is visible
before anyone waits on a generation.

**Verified in a real browser** (Chromium, Web Audio instrumented): 106 buffers
scheduled, 180.2 s of audio, peak amplitude 0.84. The player was never the
problem — it was faithfully playing an empty stream.

**Regression tests** in `tests/test_app.py` cover both causes across both output
formats: a generator that raises, and a generator that yields nothing, must each
produce a 502 rather than a playable silence.

### Follow-up: the disabled button looked like a spinner

Reported with a screenshot. Two more UI problems, both mine:

1. **`button:disabled { cursor: progress }`** meant hovering the disabled Listen
   button showed a spinning wheel — which reads as "working on it" when it
   actually means "you cannot press this". Now `cursor: not-allowed`, with
   `progress` moved to a `.working` class applied only while an episode is
   genuinely being generated.
2. **Two health warnings overwrote each other.** `say()` replaces the status
   text, so when both the API key and the speech engine were missing, only the
   *second* message survived. The screenshot showed "No speech engine installed"
   while the button was disabled for an entirely different reason — the missing
   key. Notices are now collected and rendered as a list: blocking problems in
   red, warnings in amber, and the disabled button carries a `title` explaining
   itself on hover.

### If you still hear nothing

1. `curl localhost:8000/api/health` — check `api_key_configured` and
   `tts.selected`. If `selected` is `"debug"`, no speech engine is installed and
   you will hear a quiet placeholder tone, not a voice: install `espeak-ng`.
2. Watch the server log while you press Listen. Every episode logs a line with
   `words`, `sentences` and `cache`; `sentences: 0` means the script was empty.
3. Check the browser console and the status line under the form — real errors
   now surface there in red.
4. Check your system volume and that the tab is not muted. The stream is played
   through Web Audio, so it obeys the tab's mute state.
