# The voice bench

Branch: **`claude/voice-bench-search-only`**

Search and audio playback, and nothing else. One screen for trying a candidate
voice on real FAM output, without myFAM, DailyFAM, Explore, the profile,
echoes, mixes or attachments in the way.

    ./bench.sh          # then http://localhost:8000

## What it does

**Speak text** — type or paste a paragraph and hear it. No model call, no API
key needed. This is the half that makes an A/B fair: the same words through
every candidate, so the only thing that differs is the voice.

**Ask FAM** — the real pipeline. Claude writes a real FAM script and the voice
reads it. This is the honest test, because a voice has to carry *this
product's* prose, not a demo sentence. Asking the same question again in
another voice reuses the cached script, so the comparison is the same words and
the second listen costs nothing.

Each run reports what a listening test alone will not tell you:

| | |
|---|---|
| `first sentence` | how long until there was anything to play |
| `Nx realtime` | how far ahead of playback synthesis ran |
| `engine` | which engine actually spoke |

**Read the realtime figure.** The pipeline speaks while it writes, so a voice
has to stay ahead of playback with room to spare. Piper runs at roughly 330x.
A voice that sounds lovely at 0.8x cannot ship at any price, and that is
invisible if you only listen.

## The one thing this branch does *not* do

**It does not delete the app.** `app.py`, `static/index.html` and every module
they use are byte-identical to the main branch; this branch differs only by
files it adds (`bench_app.py`, `static/bench.html`, `bench.sh`, its tests and
this note).

That is deliberate, and it is the whole reason the branch is shaped this way.
The point of the bench is to find a replacement for Piper and then **bring it
home**. A branch that gutted the repo would make that merge miserable — the
voice work would arrive tangled up in hundreds of lines of removal that have
nothing to do with voices. As it is, a voice you decide to keep is a new
`TTSEngine` subclass in `tts.py` plus a line in `ENGINES`: a clean diff over
shared files, which merges to main without argument.

## Adding a candidate

1. Write the engine in `tts.py`, next to `PiperEngine`: a `synth(text, wpm,
   voice) -> PCM` method, a `sample_rate`, an `available()`, and a `voices()`
   returning ids prefixed with the engine name.
2. Add it to `ENGINES`.
3. `./bench.sh` — it appears in the picker with no other change, because the
   picker is just `/api/voices`.

Two guards to add deliberately if the candidate is hosted rather than local,
both of which were real bugs in the WellSaid attempt (see `VOICE_OPTIONS.md`):

* **A paid engine must never be reachable by default.** `default_voice()`
  returns the first offered voice, so merely registering one made it what
  every listener got on a machine without Piper installed.
* **A hosted engine must never fall back to a local one.** Substituting Piper
  when the hosted voice fails means judging one engine by another's output.

The bench already enforces the second rule on itself: asking for a voice this
machine does not have is a 400 with the reason, never a substitution. A bench
that quietly answers with a different voice than the one you asked for is a
machine for reaching confident wrong conclusions.

## Where to start

`VOICE_OPTIONS.md` has the shortlist and the arithmetic. The short version:
**Kokoro-82M first** — open weights, ships with the app like Piper, no key and
no quota — because it is the only candidate that fixes Piper's flatness without
breaking the "audio is nearly free" premise the prefetch plan rests on.
