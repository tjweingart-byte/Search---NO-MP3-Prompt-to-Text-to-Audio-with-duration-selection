# Search → Podcast

Ask a question, pick a length from 1 to 10 minutes, and hear a spoken briefing
that starts playing within a couple of seconds and runs for exactly the length
you asked for.

Written by Claude (with live web search), read aloud straight off the script.
**No MP3, no audio file, no encoder** — the samples go from the speech engine to
your speakers over a single streaming HTTP response.

## Quick start

```bash
pip install -r requirements.txt

# A voice. espeak-ng is instant and zero-config; piper sounds far better.
sudo apt-get install espeak-ng        # or: brew install espeak-ng

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
| `config.py` | Every setting, overridable by environment variable |
| `static/` | Interface and the streaming Web Audio player |
| `tests/test_pipeline.py` | Runs offline — no API key, no TTS needed |
| `PROBLEMS.md` | Every problem hit while building this, and its solution |

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

## Known limitations

See `PROBLEMS.md` §10 — in short: espeak sounds robotic (install piper), the
rate limiter is in-process, there is no authentication, and the player is a live
stream with no seeking.
