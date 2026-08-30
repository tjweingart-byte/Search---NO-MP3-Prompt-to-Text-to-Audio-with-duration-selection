# FAM — working context

Read this before making changes. It records where the product is going, which
constraints are load-bearing, and which decisions are already settled, so work
does not drift or re-litigate them.

## Where this is going

Three surfaces, all backed by generated audio:

1. **searchFAM** — ask anything, hear a briefing of a chosen length. *Working
   today.* This is the only surface that fully works.
2. **myFAM** — a browse page of trending / recommended / for-you episodes.
   Tapping a tile generates and plays that episode.
3. **dailyFAM** — stories-style, infinitely swipeable, episodes generating as
   the listener moves through them.

myFAM and dailyFAM are **personalised**, driven by a per-user model that updates
as they interact with the app.

## The architectural consequence that matters most

Today every episode is generated **on demand**, which is why the app needs a
fast-model "cold open" to cover the wait, and why a gap can appear when that
opener runs out before the researched script arrives.

**On the browse surfaces, that whole problem is avoidable.** myFAM and dailyFAM
know what the listener might tap *before* they tap it. So:

> **Decouple script generation from speech synthesis in time.**
> The script is the expensive part (~$0.03, several seconds, cacheable text).
> The audio is nearly free (~330x realtime, milliseconds).
> Pre-generate *scripts* for likely-next episodes; synthesise audio on tap.

That yields instant playback with no cold open and no gap, and wastes only cheap
text when a prediction is wrong — not audio compute or bandwidth. The existing
script cache (`cache.py`) is already the right place to put pre-generated
scripts; it stores scripts, not audio, for exactly this reason.

Corollary: **the cold open is a workaround for on-demand latency.** Do not
extend it to the browse surfaces. Prefetch there instead.

## Open problems, in the order they hurt

1. **Voice quality.** espeak-ng sounds robotic. Piper (local, free, much better)
   is already supported — set `PIPER_MODEL`. macOS `say` is auto-detected.
   Hosted neural voices sound best but bill per character, which can dwarf the
   model cost; price it before adopting.
2. **Voice selection is a product requirement**, not just a config. The engine
   layer (`tts.py`) is already pluggable; this needs a `voice` parameter on
   `/api/audio`, a picker in the UI, and the voice folded into the cache key.
3. **The cold-open → script gap.** Mitigated by the adaptive opener and refill
   (PROBLEMS.md §15, §17), not eliminated. Prefetch removes it on browse
   surfaces; on search it is bounded by how fast the researched call returns.
4. **Personalisation needs state the app does not have**: user identity, an
   interaction log, and a recommender. Everything today is stateless.

## Constraints that are settled — do not undo without discussing

- **No MP3, no audio files.** Raw PCM streams from the TTS engine to the browser
  and is played as it arrives. This is the core of the product. Compression
  (Opus over a stream) is compatible with it and is the right answer at scale;
  writing a *file* is not.
- **Duration is a contract.** The selected length must be honoured to ~1s. Three
  mechanisms hold it: word budget, per-sentence pacing, and trim/top-up.
- **Failures must be visible.** Silent success (empty audio, a placeholder tone,
  demo mode mistaken for live) has caused more lost time on this project than
  any real bug. Every fallback must announce itself.

## Decisions that will shape the next phase

- **Where does this deploy?** Bandwidth is 2.65 MB/min uncompressed; that is
  fine on localhost and expensive at scale.
- **Is there a user account?** Personalisation cannot start without identity.
- **Local or hosted voices?** Changes the cost model more than the model choice
  does.
- **How much to prefetch?** Every speculative script costs money; every one not
  fetched costs a wait.

## Working notes

- `PROBLEMS.md` is the engineering log: every problem hit, its cause, its fix,
  and what is still open. Add to it rather than starting fresh notes.
- Tests run with no API key and no speech engine (`python -m pytest tests/ -q`).
- `diagnose_api.py` explains connection failures; `compare_models.py` compares
  cost, speed and output across models.
