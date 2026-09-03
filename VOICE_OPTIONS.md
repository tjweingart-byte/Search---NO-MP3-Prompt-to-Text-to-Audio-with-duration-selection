# Replacing Piper: what the WellSaid run actually taught us

Piper's problem is that it sounds flat. That is still true and still worth
fixing. But the first attempt at fixing it failed on something else entirely,
and the reason matters more than the shortlist.

## What went wrong, precisely

WellSaid was not rejected on quality. **Two episodes exhausted a month's
quota.** A 3-minute FAM episode is about **2,610 characters**, so those two
episodes spent roughly **5,200 characters** - which is a seat-and-quota
product being used as an API, not an API that was too expensive.

That is a *category* error rather than a *vendor* error, and it will repeat
with any provider sold per seat, per month, or per "credit" that does not
divide cleanly into characters. So the first question about a replacement is
not how it sounds:

> **Is it billed per character, with no monthly ceiling I can hit by using it?**

Anything that answers no is out before it is listened to.

## The number that decides the rest

CLAUDE.md's architecture rests on one measured claim:

> The script is the expensive part (~$0.03, several seconds, cacheable text).
> The audio is nearly free (~330x realtime, milliseconds).

Everything downstream depends on that. Pre-generating scripts for the browse
surfaces is affordable *because* speech is free once the words exist; if
speech costs real money per play, prefetch stops being a latency fix and
becomes a bill, and Explore replaying other listeners' episodes stops being
free.

Here is what each option does to that premise, per 3-minute episode:

| Voice | $/M chars | Audio | Episode total | vs script alone |
|---|---:|---:|---:|---:|
| **Kokoro-82M** (self-hosted) | 0 | $0.000 | $0.030 | 1.0x |
| **Piper** (today) | 0 | $0.000 | $0.030 | 1.0x |
| Amazon Polly standard | $4 | $0.010 | $0.040 | 1.3x |
| Polly neural / Azure neural / Google Neural2 | $16 | $0.042 | $0.072 | 2.4x |
| Azure Neural HD | $22 | $0.057 | $0.087 | 2.9x |
| Google Chirp3 HD / Polly generative / Deepgram Aura-2 | $30 | $0.078 | $0.108 | 3.6x |
| ElevenLabs Flash / Cartesia Sonic | $50 | $0.131 | $0.161 | 5.4x |

Prices as published in 2026; verify before committing to one.

**Read the last column, not the third.** The best-sounding hosted voices make
the audio *more* expensive than the writing - five times the script cost - and
"the audio is nearly free" stops being true. That does not forbid them. It
means choosing one is a decision to rebuild the browse-surface plan around a
per-play cost, and that should be decided deliberately rather than discovered
in a bill.

## The recommendation

**Try Kokoro-82M first.** It is the only option that fixes the actual
complaint without touching the cost model.

* Apache 2.0, 82M parameters, open weights - so it ships with the app exactly
  as Piper does, in `~/.fam/voices`, with `voice_store.py` unchanged.
* Reported MOS ~4.2 against Piper's noticeably flatter delivery. That is the
  gap being complained about.
* ~0.16 real-time factor on CPU - about 6x faster than playback. Slower than
  Piper's ~0.03, and still far inside the budget: a 3-minute episode
  synthesises in about 30 seconds of CPU, streamed sentence by sentence, so
  the first sentence is still out in well under a second.
* No key, no quota, no per-character cost, nothing leaving the machine.

The honest cost: it is roughly 5x Piper's CPU, which matters on a small server
with many concurrent listeners in a way it does not on a laptop. That is a
capacity question to measure, not a reason not to try it.

**If Kokoro is not good enough**, the next step is *not* the best-sounding
hosted voice. It is Polly/Azure/Google neural at $16/M - about $0.04 an
episode, doubling episode cost, with real per-character billing and no quota
cliff. Aura-2, Cartesia and ElevenLabs are better voices and are worth
hearing, but at 3.6-5.4x the script cost they change what the product can
afford to pre-generate, so they belong to a pricing decision rather than a
voice one.

## How to decide, cheaply

The mistake last time was integrating first and listening second. Reverse it:

1. Synthesise the **same** FAM script through each candidate, offline, with no
   app changes. `python write.py "<query>" --minutes 3` already prints a real
   script in seconds without generating audio - that is the input.
2. Listen to them back to back. Voice quality is a judgement call and cannot
   be settled by a benchmark table.
3. Only then wire the winner in, behind the existing `TTSEngine` interface -
   which the WellSaid experiment did prove is the right shape: a new voice was
   a new class and touched nothing in the script pipeline.

Two things the WellSaid work established that are worth keeping in mind for
whatever comes next, both learned the hard way:

* **A paid engine must never be reachable by default.** `default_voice()`
  returns the first offered voice, so simply *registering* a hosted engine on
  a machine without Piper installed silently made it what every listener got.
* **A hosted engine must never fall back to a local one.** Substituting Piper
  when the hosted voice fails means judging one engine by another's output.

Neither guard is in the code now - both were removed with WellSaid, on
purpose, because a knob left behind gets turned back on. They are written
here so the next engine adds them deliberately.

## Still unverified

Nobody has heard Kokoro on a FAM script. The MOS figure and the real-time
factor are from published comparisons, not from this project's own ears or
this project's own machine, and "reported MOS 4.2" is not "sounds right
reading a briefing". Step 1 above is the whole decision.
