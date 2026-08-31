# Search → Podcast

Ask a question, pick a length from 1 to 10 minutes, and hear the answer.
**Audio starts in about half a second** and runs for the length you asked for.

Typing pauses are used: 800ms after you stop typing, the script is written into
the cache, so pressing play is usually a cache hit. Measured against an
18-second researched call, that turns 18.30s into 0.12s.

Speed is the product. The default answers from what the model already knows, so
there is nothing between pressing the button and hearing the answer. Add
`search=1` to a request when a question genuinely needs today's facts and the
listener will accept waiting 10-25 seconds for them.

Written by Claude (with live web search), read aloud straight off the script.
**No MP3, no audio file, no encoder** — the samples go from the speech engine to
your speakers over a single streaming HTTP response.

## Just want to check the audio approach works?

Run it with **no API key at all**. The server starts in demo mode: a built-in
sample script stands in for Claude, and everything downstream — the streaming,
the pacing, the duration matching, the playback — is the real thing.

```bash
pip install -r requirements.txt
python3 -m uvicorn app:app --port 8000     # no .env needed
```

Open http://localhost:8000, press Listen, and change the length slider. If audio
starts within a second or two and a 3-minute selection produces 3 minutes of
audio, the approach is working and the interface can be built against it.
`/api/health` reports `"mode": "demo"` so nothing is mistaken for real output.

On macOS the built-in `say` voice is used automatically — nothing to install.

## Quick start

```bash
pip install -r requirements.txt

# A voice. espeak-ng is instant and zero-config; piper sounds far better.
# A voice. macOS already has one (`say`) - nothing to do there.
sudo apt-get install espeak-ng        # Linux

cp .env.example .env                  # add your ANTHROPIC_API_KEY
./run.sh                              # http://localhost:8000
```

For a natural voice, download a [Piper](https://github.com/rhasspy/piper) voice
and set `PIPER_BIN` / `PIPER_MODEL`; it is selected automatically when present.

## How it works

```
query + minutes
      │
      ├─ plan_episode()      minutes → word budget + section outline
      │
      ├─ Claude (streaming, web search) ──→ sentences, as they are written
      │                                          │
      │                            PaceController re-plans the speaking
      │                            rate before every single sentence
      │                                          │
      ├─ TTS subprocess ────────────────────→ raw 16-bit PCM
      │
      └─ StreamingResponse ──→ fetch streams ──→ Web Audio ──→ speakers
```

Hitting the clock takes three mechanisms working together — a word budget, a
per-sentence pacing controller, and trim/top-up correction. Measured drift:

| Requested | Produced | Drift |
|---|---|---|
| 1 min | 60.68 s | +0.68 s |
| 3 min | 180.00 s | 0.00 s |
| 10 min | 600.00 s | 0.00 s |

## Files

| File | Purpose |
|---|---|
| `app.py` | FastAPI server and streaming endpoints |
| `pipeline.py` | Joins Claude → TTS → socket; owns duration correction |
| `script_generator.py` | Duration → word budget, streaming Claude calls |
| `tts.py` | Pluggable speech engines (piper / espeak / debug), all raw PCM |
| `audio_utils.py` | Live WAV header, silence, the pacing controller |
| `cache.py` | Shared script cache: key normalization, TTL policy, SQLite store |
| `demo_script.py` | Built-in sample script used when there are no credentials |
| `voice_store.py` | Resolves the one shared voice folder every entry point uses |
| `setup_voices.py` | Installs the neural voices into the shared folder |
| `verify_voice.py` | Proves the voices work on this machine, and how fast |
| `write.py` | Print a briefing as text in seconds, to judge the writing |
| `compare_models.py` | Generate one query on several models and compare cost, speed and text |
| `anthropic_client.py` | Builds the Anthropic client; pins the HTTP version |
| `diagnose_api.py` | Reports why the API is unreachable when it is |
| `config.py` | Every setting, overridable by environment variable |
| `static/index.html` | The FAM prototype interface, wired to live audio |
| `static/fam-audio.js` | Streaming player the prototype calls instead of speechSynthesis |
| `static/reference-ui.html` | The original minimal test interface, kept for debugging |
| `tests/test_pipeline.py` | Runs offline — no API key, no TTS needed |
| `PROBLEMS.md` | Every problem hit while building this, and its solution |

## Two optimisations worth knowing about

**Cold open (on by default).** `claude-haiku-4-5` writes one framing sentence
with no tools *while* the main model is still researching. The listener hears
speech in well under a second instead of waiting out web search. The opener is
forbidden from stating any fact, since it has done no research; it frames the
question and nothing more. Its words are deducted from the main script's budget
so the episode still lands on the clock.

**Shared script cache (on by default).** Different people asking the same thing
reuse one script — a cache hit costs zero API tokens. The cache stores the
*script*, not the audio: a 10-minute script is ~9 KB against ~26 MB of PCM, and
re-synthesis is nearly free. SQLite means every worker on the machine shares it.

Verified: three differently-worded requests for the same NFL recap produced one
model call and two cache hits, while a query containing "my" was regenerated and
never stored.

Keys are normalized (case, punctuation, word order, filler words), so "Give me a
recap of week 5 of the NFL season" and "Week 5, NFL season — recap" share an
entry. This is lexical, so a real synonym still misses; set
`CACHE_SEMANTIC_KEY=1` to have a small model canonicalise the topic first, which
raises the hit rate at the cost of ~400 ms in front of every request.

Two safeguards: queries containing possessives or contact details are never
shared or stored, and time-sensitive queries ("latest", "today") get a 15-minute
TTL instead of 24 hours.

## API

| Endpoint | Description |
|---|---|
| `GET /api/health` | Model, selected voice, sample rate, credential status |
| `GET /api/voices` | Voices this machine can speak in, best first |
| `POST /api/script` | `{query, minutes}` → the script as JSON, no audio |
| `GET /api/audio?q=…&minutes=N&fmt=pcm\|wav` | The episode, streamed live |

`/api/audio` also takes `voice=` (an id from `/api/voices`) and `context=` (the
topic a follow-up is deepening).

`fmt=wav` streams a WAV with an unknown-length header, so it works directly as
an `<audio src>`. `fmt=pcm` sends bare samples for the built-in player, which
starts sooner.

## Tests

```bash
python -m pytest tests/ -q      # 17 tests, no API key required
```

Claude is replaced by a scripted generator and speech by a duration-accurate
tone, so the part that actually breaks — length control — is tested directly,
including when the model misses its word budget by ±30%.

## Voices

`GET /api/voices` reports what the machine can actually speak, best-sounding
first, and the player has a picker under the speed/length pill.

**Voices are installed once per user, not once per copy of the project.** They
live in `~/.fam/voices`, outside any project folder, so unzipping a new version
of the app finds them already there.

First time only:

```bash
pip install -r requirements.txt
python setup_voices.py       # downloads into ~/.fam/voices
python verify_voice.py       # proves they work, and how fast
```

Every version after that:

```bash
pip install -r requirements.txt
./run.sh
```

`setup_voices.py` is safe to re-run: it downloads only what is missing, and it
first adopts any voices an older project folder already downloaded. Override the
location with `FAM_VOICES_DIR` if you need to.

| Engine | Quality | Where it comes from |
|---|---|---|
| **Piper** | Neural, natural | **Ships with the app** — a pip dependency plus models in `voices/` |
| macOS `say` | Good | Fallback: built into macOS only |
| espeak-ng | Robotic | Last resort: only if apt-installed |

Piper is the voice the product is meant to have, and it is deliberately part of
the project rather than something the host provides. espeak only exists if
someone installed it and macOS `say` does not exist on a Linux server at all, so
an app relying on either sounds different — and worse — once deployed.

`setup_voices.py` fetches four voices (two US, two UK). They are a few tens of
MB each and are never committed. For deployment, run it as a build step with
`FAM_VOICES_DIR` pointing somewhere on the image or a mounted volume.

Voice is deliberately **not** part of the script cache key: it changes the audio,
not the words. Switching voice reuses the cached script, so it costs no model
call and takes about 90 ms.

## Choosing a model

`MODEL` defaults to `claude-opus-5`. Writing a spoken briefing to a word budget
is not a hard reasoning task, so a cheaper model is likely indistinguishable —
but that is a judgement about *your* queries, so measure it rather than take
anyone's word:

```bash
python compare_models.py "recap of week 5 of the NFL season" --minutes 3
```

It reports time to first word, total time, how close each model landed to the
word budget, and real token cost from the API's own usage figures, then prints
the scripts so you can read them against each other.

## Configuration

See `.env.example`. Notable knobs: `MODEL` (defaults to `claude-opus-5`;
`claude-sonnet-5` is cheaper and faster), `ENABLE_WEB_SEARCH`, `TTS_ENGINE`,
`TARGET_WPM`, and `RATE_LIMIT_SECONDS`.

## How the prototype is wired

`static/index.html` is the FAM clickable prototype, unchanged visually. It used
`window.speechSynthesis` to read a canned caption aloud; that layer is now
`static/fam-audio.js`, which streams a real briefing from the server. Three
functions changed and nothing else:

| Prototype function | Now does |
|---|---|
| `speakText(title, ...)` | `FamAudio.play(title, selectedLengthMinutes, ...)` — the topic title becomes the query |
| `pauseSpeech` / `resumeSpeech` / `stopSpeech` | Suspend / resume / abort the audio stream |
| `generate()` | Holds the "FAMiliarizing you…" overlay until real audio arrives, instead of a fixed 900 ms |

The player's progress bar and timer, previously fixed at 44%, now track actual
playback position, and the transport controls are live:

| Control | Behaviour |
|---|---|
| Play / pause | Suspends the audio clock, so position freezes exactly. Pressing play on a finished episode replays it |
| Skip ±15s | Seeks within audio already received; forward is clamped to what has been written so far, and says so |
| Speed pill | 1x–2x, applied to audio already playing without restarting it |
| Length | Not a playback setting — changing it while listening regenerates the episode at the new length |
| Go Deeper | Generates a new episode from what you type, carrying the parent topic as context | `selectedLengthMinutes` already existed and is passed
straight through, so the 1–10 minute picker drives real episode length.

## Demo mode vs live mode

With no `ANTHROPIC_API_KEY`, the server runs in **demo mode**: every briefing is
the same built-in sample script describing how the audio pipeline works. It does
*not* answer your search. This is deliberate — it makes the audio half testable
before any key exists — but it is easy to mistake for a broken engine, so the
app labels it in three places: a banner above the phone, a `SAMPLE SCRIPT` badge
beside "Now playing", and the loading overlay text.

To get real briefings:

```bash
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env
./run.sh
```

Then `/api/health` reports `"mode": "live"` and the banner names the model.

## Troubleshooting

**`APIConnectionError` / 502 from `/api/audio`.** Run `python diagnose_api.py`.
It prints the HTTP configuration in force, the proxy and certificate
environment, and the full cause chain behind the error, which the SDK otherwise
hides. The client is pinned to HTTP/1.1 (`ANTHROPIC_HTTP2=0`) because some
proxies break HTTP/2 to the API; `/api/health` reports which version is in use.

**"It generates an episode but I can't hear anything."** Almost always the
server failing while the browser still gets a valid-looking response. Check
`curl localhost:8000/api/health`: `api_key_configured: false` means no
credentials, and `tts.selected: "debug"` means no speech engine is installed (you
will hear a quiet tone, not a voice — install `espeak-ng`). Both now surface in
the interface rather than playing silence. See `PROBLEMS.md` §12.

## Known limitations

See `PROBLEMS.md` §10 — in short: espeak sounds robotic (install piper), the
rate limiter is in-process, there is no authentication, and the player is a live
stream with no seeking.
