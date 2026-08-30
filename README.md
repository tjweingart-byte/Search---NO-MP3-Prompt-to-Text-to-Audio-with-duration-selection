# Search → Podcast

Ask a question, pick a length from 1 to 10 minutes, and hear a spoken briefing
that starts playing within a couple of seconds and runs for exactly the length
you asked for.

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
| `POST /api/script` | `{query, minutes}` → the script as JSON, no audio |
| `GET /api/audio?q=…&minutes=N&fmt=pcm\|wav` | The episode, streamed live |

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
playback position. `selectedLengthMinutes` already existed and is passed
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
