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


---

## 13. You cannot evaluate the audio approach without an API key

The most useful failure of the whole build. Every blocker reported so far — the
silent episode, the disabled button, the spinner — had the same root: the app
required Claude credentials before it would produce a single second of sound. So
the question "does instant streaming audio actually work?" could not be answered
without first solving an unrelated setup problem.

That is backwards. The audio pipeline is the risky, novel part; the model call is
the routine part. The risky part should be the easiest to try.

**Demo mode** (`demo_script.py`). With no credentials the server now runs on a
built-in sample script instead of refusing. Everything downstream of the writer
is real: the same sentence streaming, the same pacing controller, the same
raw-PCM transport, the same player. `/api/health` reports `"mode": "demo"` and
the interface says so plainly, so nothing is mistaken for real output.

Verified with no key present: a 1-minute request produced 60.00 s and a 3-minute
request 180.00 s, and in Chromium the button was enabled, 86 buffers were
scheduled, and peak amplitude was 0.80.

**macOS `say`** (`tts.SayEngine`). Requiring a Homebrew install of espeak-ng
before hearing anything was a second unnecessary gate; every Mac already ships a
speech engine, and it sounds considerably better. It is now auto-detected ahead
of the placeholder tone. `say` needs a seekable destination for a WAV container,
so each sentence goes to a scratch file that is read and deleted immediately —
a per-sentence temporary of a second or two, not an episode file. No encoding
happens and nothing is assembled on disk, so the architecture is unchanged.

**A footgun this exposed.** `PodcastPipeline(cache=None)` used to mean "build the
default cache", so a caller passing None to switch caching *off* silently turned
it on. It caused two separate test failures — the second time as false passes,
where a cached script from an earlier test made a deliberately failing generator
return 200. `cache` now takes a store, or the explicit `AUTO` sentinel to build
the configured one, or `None` to disable. Ambiguous defaults that read as their
own opposite are worth deleting the moment they mislead once.

Untested caveat: `say` could not be exercised here (this build machine is
Linux). The engine falls back automatically if it fails, and the failure would
be visible in `/api/health`.


---

## 14. Demo mode was mistaken for a broken engine

Reported after the prototype integration: "I typed in a search, it did not
generate an episode, all it did was talk about the process." Everything was
working exactly as designed, which is what made the report so useful.

The server had no `ANTHROPIC_API_KEY`, so it was in demo mode and playing the
built-in sample script — which happens to describe the audio pipeline. From the
listener's seat that is indistinguishable from "the generator was never wired
in": you type a question and hear something that is not an answer to it.

The signal existed (a banner above the phone) but was in the wrong place. The
listening happens *inside* the phone, on the player screen, and that is where
the state needed to be visible. Demo mode is now labelled in three places: the
banner, a `SAMPLE SCRIPT` badge beside "Now playing", and the loading overlay,
which reads "Playing the built-in sample script…" instead of the normal text.

**Proving the live path, without credentials.** The deeper problem was that the
live path had never actually run — no key had existed in any environment all
session, so "the engine is wired in" was an inference rather than an
observation. A stand-in server that speaks the real Anthropic streaming wire
protocol now makes that testable: pointing `ANTHROPIC_BASE_URL` at it exercises
the genuine `ScriptGenerator`, the real SDK, real SSE streaming and real
sentence assembly.

Result: a request for "the 1969 moon landing" produced a script about the 1969
moon landing, at 150 words against a 150-word budget, and in the browser the
banner read "Live — briefings written by claude-opus-5" with no sample-script
badge. The engine is wired in; only the credentials were missing.

The lesson worth keeping: a fallback that is *useful* is also a fallback that is
*confusable*, and the label belongs where the user's attention is, not where it
was convenient to put it.


---

## 15. Five seconds of audio, then five seconds of dead air

Reported from the first real generated episode. Playback started almost
instantly, ran for about five seconds, went silent for about five, then resumed
and completed normally.

**Cause: buffer underrun between the cold open and the main script.** The
opener (§9) exists to cover research latency, but it was a single sentence -
roughly five seconds of speech - while a researched Claude call can take ten
seconds or more to produce its first sentence. When the opener ended, there was
nothing left to play, so the listener heard the shortfall as silence. Audio,
gap, audio sounds broken in a way that simply waiting does not.

**Reproduced deterministically** with a stand-in API that stalls a configurable
number of seconds before responding, and a detector that compares audio arrival
against audio consumption - a listener hears a gap exactly when the audio
delivered so far is shorter than the time elapsed since playback began:

```
BEFORE  research takes 15s
  time to first audio : 0.20s
  GAP: 9.9s of dead air, 5.2s into playback
```

**First attempt, rejected.** Holding the opener until the main script was ready
removed the gap completely - and pushed time to first audio from 0.2s to 15.3s.
That trades away the property the whole architecture exists for.

**The fix: an adaptive opener.** The opener is now written as up to four short
framing sentences, released one at a time, with a check before each one for
whether the main script has arrived. The moment it has, the opener stops and the
briefing takes over seamlessly. Slow research gets more introduction; fast
research gets almost none; unused sentences are discarded. Each sentence is
prompted to stand alone, since playback may cut away after any of them.

```
AFTER   research takes 2s / 8s / 15s
  time to first audio : 0.51s / 0.49s / 0.48s
  no gaps in any case
```

Three guards around it: at least one sentence always plays (the main script is
written on the understanding that the episode has already been opened, so
without it the briefing starts mid-thought); nothing is spoken at all until the
script has had a moment to fail, so a bad key still produces a clean 502 rather
than an introduction to an episode that never comes; and the opener will not run
past `COLD_OPEN_MAX_SECONDS`, after which a gap is better than endless preamble.

**Still open.** There is no client-side jitter buffer. On localhost the server
now guarantees continuity, but across a real network a stalled connection can
still starve the player. A pre-roll of a few hundred milliseconds in
`fam-audio.js` would absorb that, at a small cost to time-to-first-audio.


---

## 16. Transport controls: why skip needed a different player

Requested: working pause, playback speed, skip forward/back 15 seconds, length
changes from inside the player, and a working Go Deeper.

**Skip was the one that forced a redesign.** The player scheduled each incoming
chunk onto the audio clock and then forgot it, which is the cheapest way to play
a live stream and makes seeking impossible: there is nothing behind you to go
back to, and nothing to re-schedule differently.

The player now keeps every sample it has received in a growing buffer and drives
playback from a position cursor. Skip, scrub and speed all become the same
operation - stop what is queued, move the cursor, re-schedule from there.
Samples are kept as Int16 (~2.6 MB per minute) and converted to float only for
the quarter-second slice being scheduled, which halves what a long episode
holds.

Position is derived from the audio clock rather than counted separately, so it
stays correct across pauses for free: suspending an `AudioContext` stops its
clock advancing.

**Length is not a playback setting.** Speed changes what is already playing;
length changes what the episode *is*. Changing it from the player therefore
regenerates rather than adjusting anything locally.

**Go Deeper was already carrying the typed prompt** - the pipeline was just
ignoring it in favour of the display title. It also needed the parent topic
attached: "focus on the economics" is not a briefing request on its own, so the
query sent is now `<follow-up> (following up on: <parent topic>)`.

**Forward skip has a real limit** and says so. The episode is still being
written, so you cannot skip into audio that does not exist yet; skipping past
the end of what has arrived reports "that is as far as the episode has been
written" rather than silently doing nothing.

Verified in Chromium against a live server: pause froze position at 7.27s and
resume advanced it; back-15 from 8.71s clamped to 0s and forward-15 landed on
14.98s with playback continuing; 2x advanced 4.02s of audio in 2s of wall clock
and 1x advanced 2.01s; a length change from 2 min to 1 min regenerated and the
timer showed 1:00; and Go Deeper sent
`focus on the economics of early radio (following up on: the history of radio)`.


---

## 17. Five issues from the 29/08 field notes

**Skipping forward past the written script froze the player.** Reproduced with a
slow-streaming stand-in server: skipping to the edge of received audio left the
cursor exactly where no audio existed, so playback stopped dead and further
skips appeared to do nothing - position pinned at 27.29s across three
consecutive presses.

The cursor now stops two seconds short of the written edge while an episode is
still streaming, so there is always audio left to keep playing. After the fix
the same test clamps at 25.76s and *keeps advancing* (25.76 -> 28.16 -> 30.56)
as more arrives. The progress bar also grew a faint second track showing how
much has been written, so the limit is visible rather than only felt, and the
message changed from "that is as far as the episode has been written" to
"caught up with the writing - it will keep going", which is what actually
happens.

**A gap could still open between the opener and the script.** The adaptive
opener (§15) covers only as much time as the sentences it was given. If the
script took longer than that, silence returned. The opener is now refilled -
up to twice - when it runs dry while the script is still being written.

**The opener was generic and repetitive across episodes.** It was prompted only
to "frame what is about to be covered", which produces the same shape every
time; a listener working through several episodes hears a template. It is now
prompted to be specific to the topic - what is unsettled, why someone would ask
now - and the server keeps the last few openers and instructs the model not to
reuse their wording, rhythm, or opening move. "Here is your briefing" and
similar are explicitly banned.

**No sense of time scope.** Asking for "the Tour Championship update" could not
distinguish this morning's state from this moment's, because *the prompt never
said what time it was*. The model had no clock. Requests now carry the current
date and time, with instructions to prefer the newest information, say out loud
what the picture is as of, describe what changed during the day in order, and
state plainly when something is still in progress rather than implying a result.

**Go Deeper had the wrong scope.** The follow-up was being glued onto the query
as one run-on string, so the model treated it as a fresh topic and re-explained
the ground the listener had just heard. What was already covered now travels as
a separate `context` parameter, and the prompt tells the model to treat it as
known, not re-introduce the subject, and spend the whole episode on the narrower
point. It is also part of the cache key, since a follow-up is a different
episode from the same words asked cold.

**A bug found while fixing these.** The opener refill called `cold_open(plan)`
from a method that did not take `plan`. It did not raise cleanly - it hung the
test suite - which is a good argument for passing dependencies as arguments
rather than stashing them on `self`, which is what the code now does.


---

## 18. Pinning the Anthropic client to HTTP/1.1

Reported: `APIConnectionError: Connection error` on every live request, with
`curl` succeeding only under `--http1.1`, so HTTP/2 was suspected.

**What the code does now.** All Anthropic clients are built by
`anthropic_client.build_async_client()`, which passes `http2=False` to
`anthropic.DefaultAsyncHttpxClient`. The SDK subclass is used rather than a raw
`httpx2.AsyncClient` so its timeouts, connection limits and TCP keep-alive
settings survive; only the protocol version is overridden. `/api/health` reports
the version in force, and `ANTHROPIC_HTTP2=1` opts back in - raising a named
error if the optional `h2` package is missing, rather than failing as another
opaque connection error.

**A finding worth stating plainly, because it affects the diagnosis.**
`anthropic` 1.x runs on **httpx2**, not httpx, and `httpx2.AsyncClient` already
defaults to `http2=False`. The SDK never sets it. And `h2` is not installed by
this project, without which httpx2 cannot negotiate HTTP/2 at all. So the Python
client was almost certainly *already* using HTTP/1.1 before this change, even
though `curl` was negotiating HTTP/2 by default and failing.

That means pinning is worth doing - it states the requirement in code, survives
a future default change, and is now asserted by tests - but it should not be
assumed to be the fix. If the error persists, the cause lies elsewhere.

**So the change ships with a diagnostic.** `diagnose_api.py` reports library
versions, whether `h2` is present, the pinned protocol, the proxy and CA-bundle
environment, and - on a real request - the full `__cause__` chain that
`APIConnectionError` hides, with the common causes named: DNS failure, TLS
interception (`SSLCertVerificationError`, fixed with `SSL_CERT_FILE`), a proxy
refusing CONNECT, or a timeout indicating traffic is dropped rather than
refused.

**Tests** (`tests/test_http_client.py`) assert on the real transport pool
(`client._transport._pool._http2 is False`), not merely on the flag passed in;
that the generator used in production is built through the pinned path; that SDK
defaults are preserved rather than replaced; and that enabling HTTP/2 without
`h2` fails with a clear message.

Also corrected: `requirements.txt` listed `httpx`, which the Anthropic SDK does
not use. It is needed only by `fastapi.testclient`, and now says so.


---

## 19. Voice selection

The first step towards a shippable v1: the espeak fallback sounds robotic, and
for an audio product that is the first thing anyone judges.

`GET /api/voices` enumerates what the machine can actually speak. Each engine
reports its own: espeak offers a curated six accents (it exposes hundreds of
variants, and offering all of them is worse for a listener than offering a few
that sound genuinely different), macOS `say` parses `say -v ?` and puts the
long-form-friendly voices first, Piper reports its configured model. Voice ids
carry their engine (`say:Samantha`), so an id alone routes the request.

**Voice is not part of the script cache key, on purpose.** A voice changes the
audio, not the words, so the same script serves every voice. Measured: two
voices for one query produced different audio, identical duration, and **one
model call** - the second request was a cache hit. Switching voice on an
existing episode takes ~90 ms against ~1.2 s for a fresh topic, and costs
nothing.

An unavailable voice falls back to the best engine present rather than failing:
a listener whose chosen voice was uninstalled should still hear their episode.

**Untested here:** the macOS `say` voice list could not be exercised on this
Linux build machine. The parsing is defensive and the engine falls back if it
returns nothing, but the first Mac run should be watched.


---

## 20. Making the voice part of the app rather than the host

Until now the voice was whatever the machine happened to have: macOS `say` on a
laptop, espeak-ng on Linux if someone apt-installed it, a placeholder tone
otherwise. That is fine for a prototype and wrong for a product, and it breaks
in a specific way at exactly the wrong moment: **deploying to a Linux container
would have made the app sound worse than it does locally**, because `say` does
not exist there.

Piper is now a real dependency (`piper-tts` in `requirements.txt`) with voice
models installed into `voices/` by `python setup_voices.py`. The engine ships
with the app; only the model files are fetched, and they are fetched by a
command in the project rather than assumed to exist.

**Two things the implementation has to get right.**

*Load the model once.* A Piper voice takes about a second to load. The old
implementation shelled out to a `piper` binary per sentence, which would have
reloaded the model roughly 170 times in a ten-minute episode. Loaded voices are
now cached for the life of the process. (§9 measured subprocess spawn as 62% of
espeak's per-sentence cost while being irrelevant in absolute terms; for Piper
the same mistake would have been fatal.)

*Do not block the event loop.* Inference is synchronous CPU work. Run directly
it would stall every other listener for the duration of every sentence, so it
runs via `asyncio.to_thread`. There is a test asserting synthesis does not
happen on the main thread.

**Hosted neural voices were considered and not adopted.** They sound best, but
bill per character, which can dwarf the model cost, and add a network dependency
to the one part of the pipeline that is currently free and instant.

### What is not verified

The build machine cannot reach huggingface.co, where the voice models are
hosted, so **the ONNX inference itself has never been run here.** Everything
around it is tested against a stub: discovery, naming, routing, model caching,
rate control, sample rate, thread offloading, and fallback when a voice is
missing. `python verify_voice.py` closes the gap on a real machine - it
synthesises a sentence with every installed voice and reports duration, sample
rate and realtime factor, with `--save` to write a WAV to listen to.

If the models cannot be downloaded on the target machine either, the app keeps
working on the existing fallbacks and says so; it does not fail.


---

## 21. The gap, properly understood

Reported as a consistent 5-7 second pause near the start - "appears to the
viewer as a glitch". §15 and §17 had each improved this without eliminating it,
because both fixes were aimed at the wrong variable.

### Finding the real mechanism

Two hypotheses were tested and killed first. Piper is far slower to synthesise
than espeak (~10x realtime against ~330x), so that looked likely - but sweeping
simulated synthesis speed from 10x down to 3x produced no gaps at all, because
anything faster than realtime keeps the stream ahead. And time-to-first-audio,
while it did rise from 0.5s to ~2.8s with a neural voice, is a delay before
playback, not a hole in it.

The pipeline was then made to report on itself: per-sentence synthesis time, and
a running comparison of audio produced against wall clock consumed - which is
exactly when a listener hears silence. With a realistically short opener and a
30-second researched call, it named the mechanism immediately:

```
opener ran dry after 6.8s; refilling
opener ran dry after 13.7s; refilling     <- the second and last refill
STARVED after 31.2s: only 29.5s of audio made in 31.2s of wall clock
GAP: 7.9s of dead air, 20.4s into playback
```

`MAX_OPENER_REFILLS = 2` covered about twenty seconds. Research took thirty. The
listener heard the difference - and because the cap was fixed, so was the gap,
which is why it was *consistent* at 5-7 seconds.

### The control law

Removing the cap fixed the gap and created a worse problem: the opener produced
**75 seconds of preamble to cover a 30 second wait**. Audio is synthesised many
times faster than it is heard, so any wall-clock budget lets the opener sprint
ahead of the listener.

The opener is now paced to the listener rather than the clock. It speaks only
while the audio produced is less than `OPENER_HEADROOM_TARGET` (8s) ahead of the
wall clock, then waits. That is precisely the buffer needed never to fall
silent, and it self-regulates to roughly real time. Top-ups are triggered by
seconds of speech held rather than sentence count, and fetched *before* the
buffer drains, so an API call never interrupts speech.

Measured, at 8x-realtime synthesis:

| Research takes | Gaps | Opener fetches | Opener spoken |
|---|---|---|---|
| 10s | none | 0 | one batch |
| 30s | none | 3 | 36s |
| 45s | none | 4 | 49s |

### What this does not fix

**A slow researched call still means a long introduction.** If research takes
thirty seconds the listener must hear thirty seconds of *something*, and that
something is preamble. The gap is now structurally impossible up to a 60 second
ceiling, but the cure for a long opener is a faster script, not a longer opener:

- `MAX_WEB_SEARCHES` is now 3 rather than 5. Each search costs seconds.
- `ENABLE_WEB_SEARCH=0` removes the dominant cost for evergreen topics.
- A faster model (`MODEL=claude-sonnet-5`) reduces time to first sentence.
- On the browse surfaces, prefetching removes the wait entirely (`CLAUDE.md`).

Every episode now logs `first_audio_at`, `opener_fills`, `min_headroom` and
`starved`, so this is diagnosable from a log line rather than by ear.


---

## 22. Voice models were being re-downloaded for every new version

Each release was handed over as a fresh `fam-podcast N` folder, and voice models
lived in `voices/` *inside* it. So moving from version 6 to version 7 looked
like a fresh install: no voices, download sixty megabytes again. The models are
large and change far less often than the code, so tying their lifetime to the
code's was simply the wrong choice.

They now live once per user in `~/.fam/voices`, resolved in one place
(`voice_store.py`) that every entry point uses - the server, `setup_voices.py`,
`verify_voice.py` and the tests - so they cannot drift apart. `FAM_VOICES_DIR`
overrides it for deployments and tests; the older `VOICES_DIR` is still honoured.

**Existing downloads are adopted, not re-fetched.** On startup, if the shared
store is empty and an older project folder has voices, they are copied across.
Three properties make that safe to run automatically on every launch:

* **Copy, not move.** The old folder keeps working if anything goes wrong, and
  can be deleted whenever the user chooses. `--move-legacy` moves instead.
* **Written via a `.partial` name, then renamed.** An interrupted copy can never
  leave a half-written model that looks installed.
* **Idempotent, and never overwrites.** It only runs when the shared store is
  empty, and skips any voice already present. A model whose `.onnx.json` sidecar
  is missing is skipped rather than half-adopted, since Piper needs both.

The result is the workflow that was asked for: unzip, install requirements, run.
No voice step at all after the first time.


---

## 23. The opener was filler, and the scripts were worse

Reported bluntly and correctly: the first thirty seconds were "I'm telling you
what I'm going to tell you about", and the briefings themselves were "poor,
sometimes inaccurate, and not interesting to listen to".

### The opener

It was filler *by construction*. Its prompt said: "State NO facts, figures,
dates, names, results or opinions... Frame the question; never answer it." Given
that instruction there was no good version of it - and §21, which made it longer
so it could cover slower research, made the experience worse rather than better.
It is now off by default. Nothing plays until the real briefing does.

A concrete bug was also found while investigating: when an opener refill fails,
the pipeline fired up to twelve rapid retries and then went silent - one
sentence, then nothing, which is exactly what was reported. The burst may have
been causing the failures itself by tripping a rate limit.

### The scripts

The real problem, and the one that had gone unexamined while the plumbing got
all the attention. Three lines of the brief were doing the damage:

* **"Length contract - this is the most important requirement."** The model was
  told, in as many words, that hitting a word count mattered more than being
  worth hearing. So it padded.
* **"a one-line hook (about 146 words)."** Incoherent: an instruction to inflate
  a single line into a paragraph.
* **A fixed five-beat template** applied to every topic. For a golf recap, "the
  main debate or open question" is a section with nothing in it, so the model
  invented something. That is where inaccuracy came from - not a hallucinating
  model, but a prompt demanding content that did not exist.

Both prompts are rewritten. The system prompt now describes what makes a
briefing good - open on the most concrete thing you know, prefer one exact
detail to three general statements, cut anything the listener could have
guessed, let the material choose the shape, never fill a gap with something
plausible. The brief gives a length as "the listener's time, not a quota", with
an explicit instruction to finish early rather than pad.

### The tension this exposes

**Duration control and content quality were fighting each other.** Enforcing a
word count in both directions guarantees padding whenever a topic has less to
say than the slider asks for. The length is now a ceiling: over-runs are still
trimmed, but a short script ends rather than being topped up (`ALLOW_TOPUPS=1`
restores the old behaviour). A three-minute briefing worth hearing beats a
five-minute one stretched to fill the slider.

### Still unverified

The rewrite cannot be judged from here - it needs reading against real queries,
and the judgement is editorial rather than technical. `write.py` prints a script
in seconds without generating audio, which is the loop for improving it.


---

## 24. Optimising the wait instead of removing it

The product is one sentence: type a question, hear the answer within about a
second. Sections 15, 17, 21 and 23 are all elaborate machinery for *disguising*
a twenty to thirty second wait - an adaptive opener, refills, a pacing control
law, a headroom target. None of them asked why the wait existed.

It existed because every query went through the slowest model with live web
search attached. Web search front-loads 10-25 seconds before the model writes a
single word. No buffering strategy can hide that, and every attempt to hide it
made the product worse: first a gap, then filler, then more filler.

**What changed**

* **Web search is off by default**, opt-in per request (`search=1`). Most
  questions do not need today's facts, and the ones that do can wait knowingly.
* **The default model is `claude-sonnet-5`**, which answers from what it knows
  almost immediately, rather than `claude-opus-5`.
* **The voice model is loaded at startup**, not on the first listener - it was
  costing 1.5-3 seconds on the first episode, exactly where it showed most.
* **A 1.5 second pre-roll** before playback begins. Models stream in bursts, so
  starting on the very first sentence turns any stall into an audible hole a
  second in. At many-times-realtime synthesis this costs almost nothing.
* **The prompt bans preamble outright**, by name: "Here's what I can tell you
  about...", "Let's talk about...", "There's a lot to unpack here...". If the
  first sentence would survive having a different topic substituted into it, it
  is wrong.

**Measured, end to end, with a real speech engine:**

```
time to first audio : 0.69s / 0.53s / 0.51s
no gaps in any run
```

Against 20-30 seconds and a gap before.

**The lesson worth keeping.** Every fix from §15 onward optimised the machinery
around a wait that should not have existed. The question "why is this slow?"
was never asked, only "what can we play while it is slow?" - and the answer to
that question is always filler.


---

## 25. Two things: narrated timestamps, and prefetching the wait away

**"Here is what I have as of Sunday, August 30..."** was mine. §17 told the
script to say what its picture was current as of "the first time it matters",
and the model turned that into an opening disclaimer. A listener does not want a
timestamp read to them. The prompt now bans announcing currency outright, and
permits timing only where it changes the meaning - "the count is still going" -
said in passing rather than as a preamble.

**Prefetch, which was the user's idea and a better one than anything above.**
The observation: a shop estimates delivery when an item goes in the basket, not
at checkout. It does the slow work during a pause the customer is taking anyway.

Applied here: the expensive part of an episode is the script; the audio is
nearly free. So 800ms after someone stops typing, `/api/prefetch` writes the
script into the cache. When they press play, `/api/audio` finds it there and the
model wait has already happened.

Measured against a stand-in API that takes 18 seconds to answer:

```
cold press   : time to first audio 18.30s
prefetched   : time to first audio  0.12s
```

**This corrects something stated wrongly earlier in this log.** §24 framed live
search as costing 10-25 seconds, full stop. That is only true if generation
starts when the button is pressed. It does not have to. The wait is a scheduling
choice, not a property of the work.

Guards: only after a real pause (800ms) and a real question (12+ characters);
never the same script twice; never a personal query; a failed prefetch is
silent and the listener simply takes the normal path. An unused prefetch costs
one script and no audio.


---

## 26. A prefetch that could vanish mid-flight

Reported from a clean install on Python 3.12.6: `test_prefetch_puts_a_script_in_the_cache`
failed while everything else passed. It passed here, which was luck - running it
twenty times in a row failed **seven**.

**Cause.** The endpoint started its work with `asyncio.create_task(build())` and
kept no reference to the result. The event loop holds only a *weak* reference to
a task, so a fire-and-forget coroutine can be garbage collected part-way through
and simply stop. Python's own documentation warns about exactly this.

**Why it mattered more than a flaky test.** A failed prefetch is deliberately
silent - the listener just takes the normal path - so in production this would
have shown up only as the wait occasionally not being removed, at random, with
nothing in the logs. The feature built specifically to make the app feel instant
would have worked most of the time and quietly not worked the rest.

**Fix.** Starlette's `BackgroundTasks`. It runs after the response is sent, so
the browser is never kept waiting, and the task is owned by the request rather
than floating free. Twenty consecutive runs of the test now pass, as do five
consecutive full-suite runs.

**The test was strengthened, not weakened.** It had been polling with a retry
loop - which existed only to paper over this bug. That is gone: the assertion is
now immediate, because the test client runs background tasks to completion. A
second test parses the endpoint's AST and fails if `asyncio.create_task`
reappears, and a third proves end to end that a prefetched episode plays without
touching the model.

Verified unaffected: the shared voice store still resolves to `~/.fam/voices`,
`setup_voices.py --list` reads it, startup reports it, and prefetch still turns
an 18-second model call into 0.11s to first audio.


---

## 27. Prefetch removed

Removed at the user's request. `/api/prefetch`, the typing-pause trigger in the
interface, its bookkeeping and its tests are all gone - not disabled behind a
flag, deleted, so there is no dead path to reason about later.

What it did is worth keeping in mind rather than in code: writing the script
during the pause before someone presses play turned an 18.30s wait into 0.12s,
because the script is the slow part and the audio is nearly free. What it cost
was a speculative model call for every abandoned query, which on a search box is
most of them.

That trade is much better on the browse surfaces, where what someone might tap
is known well in advance and the hit rate would be far higher. The reasoning is
recorded in `CLAUDE.md`; the mechanism is not in the codebase.

The ordinary script cache is untouched and still does the cheap half of the same
job: a repeat of the same query is a hit. Measured after removal - 0.70s to
first audio cold, 0.11s on the repeat, no gaps.
