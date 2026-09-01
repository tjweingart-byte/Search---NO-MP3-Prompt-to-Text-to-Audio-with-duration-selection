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
2b. **DailyFAM** (was playFAM) — named daily mixes. A mix holds topic ids or
   questions the listener typed, never audio, so it is fresh every morning.
3. **explore** (was dailyFAM) — a vertical feed of episodes *other listeners
   have already generated*. It never writes a script: cards come from the
   shared cache and playing one sends `cached_only`, which the pipeline
   refuses to satisfy by generating.

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

## The one-sentence spec

**Type a question, and within about a second audio starts giving the answer.**
Everything else is negotiable; this is not. Any change that puts seconds in
front of the first word is wrong, however clever the thing filling those
seconds is.

What that rules out, learned the hard way: live web search on every query (it
front-loads 10-25 seconds), the slowest model by default, and any form of
preamble used to disguise a wait. Search is now opt-in per request; the default
answers from what the model already knows, immediately.

Measured on the current build: **0.5s to first audio, no gaps.**

**A slow answer is a scheduling choice, not a property of the work.** The wait
only exists if generation starts when the button is pressed - measured, starting
earlier turned an 18.30s wait into 0.12s.

Prefetch-on-typing-pause was built to exploit that and then **removed** at the
user's request; it is not in the code. The same reasoning still applies to the
browse surfaces, where what someone might tap is known well before they tap it,
and where a speculative script is far more likely to be used than one triggered
by a keystroke pause. That is where to spend it.

## What makes a FAM episode different

**First, it satisfies the thing that brought them.** Someone searched, or tapped
a tile, or decided to keep listening in dailyFAM — each of those is a want, and
the episode's first duty is to meet it. They should finish knowing what they came
to find out, well enough to say it back in their own words. **Satisfied first,
curious second.** Everything below is about how that answer arrives and is worth
nothing without it.

That ordering is load-bearing, not a pleasantry: the curiosity is what makes
someone want another episode, but the satisfaction is what makes them believe
another episode is worth having. Get them the wrong way round and the second one
never gets tapped.

The failure this rules out, which the ending rules could otherwise produce:
**the episode must never withhold.** Withholding is not momentum, it is a bait
and switch, and a listener spots it instantly. Close the question they came
with, completely — and then stop.

**And it is a story; that is the product.** Not a briefing with storytelling
added — the narrative is how the information arrives. A listener asking about a
simple concept or a routine update should find themselves pulled along without
noticing why.

The distinction that matters, because getting it wrong is what produces the two
failure modes seen so far:

* **Narrative as structure** (right): facts arrive in an order that opens a
  question and closes it. Because / therefore / but. Invisible.
* **Storytelling as decoration** (wrong): "picture this", scene-setting,
  atmosphere. This is what makes a listener think *get to the point*.

The aim is **annexation**: not inviting the listener in, but absorbing them
before they decide to come, so that leaving takes a deliberate act. Two
mechanics carry it:

* **Speak from inside.** No orienting, no justifying the topic. Begin as though
  continuing a conversation they were already in.
* **Land it and stop.** The last line is the most concrete thing in the piece,
  and then it ends, mid-stride. No summary, no recap, no "so, to sum up" — each
  hands the listener their coat.

**Endings do not tease. (Reversed — this used to say the opposite.)** The rule
was once "endings widen, they do not conclude": leave one named thread standing
and end pointed at it. On paper that is momentum. Heard back to back it is a
hook at the end of every single episode, which is a tease, and it was asked for
to be removed twice. So: no dangling hook, no "but that raises another
question", no rhetorical question at the end, no forecasting. Anything genuinely
unresolved is said *inside* the piece — plainly, as unresolved, and then the
piece carries on.

The guard: every sentence must carry information. Atmosphere alone is cut. The
point should be arriving continuously, from the first line, inside the story.

**Go Deeper did not lose its suggestion — it stopped coming from the script.**
The model still writes a trailing `<<NEXT: ...>>` line, stripped before
synthesis and never spoken, but it is now a *prediction* rather than a promise:
having heard this episode, what would this listener most likely ask next, read
off what was actually covered. The likeliest follow-up, not the most obscure
one. The pipeline stores it beside the script, `GET /api/next` returns it for
free, and Go Deeper offers it as a one-tap chip — so the suggestion is waiting
afterwards for anyone who wants it, and costs nothing to anyone who does not.
The script is explicitly barred from gesturing at it.

Note what this costs, so it is a known trade rather than a surprise: the
browse surfaces no longer get their "the ending of one is the entry to the
next" pull for free. dailyFAM's infinite swipe and myFAM's tiles now have to
earn the next tap on their own, which is what the predicted follow-up and the
prefetch plan are for.

Note this **replaced an earlier rule** that said to open with the answer
immediately. That was news-writing — the inverted pyramid — and it is the
opposite of story structure. The opening should be concrete and open a question,
not state the conclusion.

## The problem that matters most right now

**The scripts are not good enough.** Not the voice - the writing. That is the
product, and it had received almost no attention next to the plumbing.

The prompt was the cause. It told the model that hitting a word count was "the
most important requirement", asked for "a one-line hook (about 146 words)", and
imposed the same five beats on every topic - so for a golf recap it had to
invent something to fill "the main debate or open question". Padding and
invention were being requested. Both prompts have been rewritten around what
makes a briefing worth hearing; that rewrite is untested against real output.

The endings were then rewritten again (PROBLEMS.md §48) to stop teasing the
next episode. **Also untested against real output** - it was written without an
API key, so nobody has yet heard an episode that ends under the new rules. This
is the first thing to check.

`python write.py "<query>" --minutes 3` prints a script in seconds without
generating audio. That is the loop for improving this, and it is a judgement
call rather than an engineering one.

**`examples/` is the strongest lever on the writing.** Briefings dropped in
there are shown to the model as the house voice. Rules describe a style loosely;
examples are matched closely, so two or three good ones move the output more
than any amount of further prompt wording. Prefer adding an example over adding
another rule.

## Open problems, in the order they hurt

1. ~~**Voice quality**~~ — *addressed, needs verifying on a real machine.* Piper
   is now a pip dependency (`piper-tts`) with voice models in a **shared
   per-user folder** (`~/.fam/voices`, see `voice_store.py`) installed by
   `python setup_voices.py`. They deliberately live outside the project so a new
   version of the app reuses them instead of re-downloading. It ships **with the app** rather than
   depending on the host OS: espeak only exists if apt-installed and macOS `say`
   does not exist on a Linux server, so relying on either means the deployed app
   sounds worse than the laptop it was built on. Hosted neural voices were not
   adopted: they bill per character, which can dwarf the model cost.
   *The ONNX inference could not be exercised on the build machine (the models
   are hosted somewhere it cannot reach) — `python verify_voice.py` is the check
   that closes that gap.*
2. ~~**Voice selection**~~ — *done*. `/api/voices` lists what the machine can
   speak; `voice=` on `/api/audio` selects one; the player has a picker.
   Note: voice is deliberately **not** part of the script cache key, because a
   voice changes the audio and not the words. Switching voice therefore reuses
   the cached script — measured at ~90 ms and zero API cost.
3. ~~**The cold-open → script gap**~~ — *fixed twice; measured this time*
   (PROBLEMS.md §21, then §46). §21 claimed silence was "structurally
   impossible up to a 60s ceiling". **That was wrong** - a rebound variable was
   orphaning opener sentences mid-flight, so on a 30s script 52 were written
   and 11 spoken with 11.5s of dead air. Fills are now kept in a list and
   drained oldest-first. `python tools/gap_probe.py` measures it with no API
   key and reports 0.00s at every latency from 2s to 30s; run it before
   believing any future claim about this.
   **What remains is not a bug but a consequence:** a 30s researched call means
   30s of preamble, because the listener must hear *something*, and the opener
   may state no facts - so the seam between it and the script is structural.
   The cure is a faster script - fewer web searches, a faster model, or
   prefetch on the browse surfaces - not a longer opener.
   Note the cold open is still **off** by default (`ENABLE_COLD_OPEN=0`).
4. **myFAM is built; the taste model is deliberately crude.** `topics.py` ranks
   a *shared* bank of ~28 topics three ways (trending / co-listener / history)
   from an append-only event log. Tags come from keyword matching, not a
   classifier. A fourth ranking, `rank_might_like` (adjacent to your taste),
   is written and tested but no longer shown - its section was removed from
   myFAM. Worth knowing what went with it: it was the only signal that offered
   anything *outside* an established taste, so the feed is now history,
   co-listeners and the crowd. One line in `SECTIONS` brings it back. The cost design is the load-bearing part: **one bank for
   everyone, personalisation in the ordering, not the inventory** - so two
   people tapping a tile share one script through `cache.py`.
5. **playFAM is built as its own tab.** `mixes.py` stores named daily mixes -
   a mix holds *topic ids*, never audio, so "At the gym" is the same subjects
   every day and a different set of episodes. Members are validated against the
   same shared bank, which is what keeps the cost design intact.
6. **"What your followers are listening to" has no follow graph behind it.**
   It ranks co-listener overlap. The heading promises a social network the app
   does not have; either build follows or rename it.
7. **Echoes are the social layer, and they generate nothing.** `social.py`
   stores an echo as a row pointing at a query whose script already exists, so
   pushing an episode to other listeners costs nothing. It changes what the
   Explore card says, not what has to be written. Mixes are private by default
   and appear on the profile once made public.
8. **Profile is a scaffold, deliberately.** `/api/profile` returns only what
   the event log actually holds - started, finished, open threads, subjects -
   because a profile page is the easiest place in an app to invent numbers,
   and every invented one is a promise to keep later.
9. **Attachments are built** (`attachments.py`, PROBLEMS.md §47). A search can
   carry documents, photos and links. Extraction happens when the thing is
   attached, never on the generation path, because a round-trip in front of the
   first word is the one cost this product refuses. Every failure is a sentence
   the listener can act on, and an attached episode is **never cached**, so it
   cannot reach another listener or Explore. Only PDF needs a package (pypdf,
   optional); .docx is read with `zipfile`.
10. **Personalisation needs state the app does not have**: user identity, an
   interaction log, and a recommender. Everything today is stateless.

## Constraints that are settled — do not undo without discussing

- **No MP3, no audio files.** Raw PCM streams from the TTS engine to the browser
  and is played as it arrives. This is the core of the product. Compression
  (Opus over a stream) is compatible with it and is the right answer at scale;
  writing a *file* is not.
- **Duration is a ceiling, not a quota.** *(Revised.)* The selected length still
  caps the episode and over-runs are trimmed, but a script that runs out of
  substance now ends early instead of being padded. Enforcing the number in both
  directions is what produced filler: it made the model pad. `ALLOW_TOPUPS=1`
  restores the old behaviour.
- **No filler, ever.** The cold open is off (`ENABLE_COLD_OPEN=0`). It was
  prompted to state no facts, which made it worthless by construction, and
  covering a long research wait meant 15-30 seconds of it. Nothing plays until
  the real briefing does; the interface shows an honest loading state.
- **Failures must be visible.** Silent success (empty audio, a placeholder tone,
  demo mode mistaken for live) has caused more lost time on this project than
  any real bug. Every fallback must announce itself. *(PROBLEMS.md §51: demo
  mode did announce itself, in an 8.5px chip, and still cost a whole session -
  and it was writing its canned script into the shared cache, so the failure
  outlived the run. Announcing is not enough if the thing keeps a record.)*
- **Verify, do not inspect.** *(PROBLEMS.md §52.)* Four consecutive failures on
  a real machine all had the same shape: a check answered a cheaper question
  than the one being asked and then reported OK. "A key is set" is not "the key
  works"; "a cache is configured" is not "this generator may write to it". The
  server now asks Claude at startup whether the credential is actually accepted
  and says so on every tab. Anything that reports readiness must perform the
  real action, not confirm that it was configured.
- **Per-machine state lives in `~/.fam/`, never in the project.** Voice models
  (`~/.fam/voices`) and the API key (`~/.fam/env`, written by
  `python setup_key.py`) are set once and found by every later copy of the app.
  A key in a project `.env` is lost on every new copy, and the workaround for
  that is pasting it again somewhere it should not go. The key is never written
  into source: a commit keeps it in history after the line is deleted.

## Decisions that will shape the next phase

- **Where does this deploy?** Bandwidth is 2.65 MB/min uncompressed; that is
  fine on localhost and expensive at scale.
- **Is there a user account?** Personalisation cannot start without identity.
- **Local or hosted voices?** Changes the cost model more than the model choice
  does.
- **How much to prefetch?** Every speculative script costs money; every one not
  fetched costs a wait.

## How to ship a change (standing instruction)

The loop matters as much as the code. Every change ends the same way, without
being asked:

1. `./dev.sh check` — tests, the interface-parses check, the preview build and
   its browser smoke test. All of it, every time.
2. Rebuild the phone preview and **republish it to the same artifact URL** so
   the link never changes:

       https://claude.ai/code/artifact/c8bd86aa-e61e-4262-a1c8-b9c8d8d6645e

   Publishing to the same file path within one conversation updates it in
   place; **from a new conversation, pass that URL as `url`** or you will
   create a second artifact and the link the phone has bookmarked will go
   stale. Read it first, then publish `preview/fam-artifact.html` to it.
3. Reply with a short summary of what changed and the preview URL. Not a zip,
   not a wall of files.
4. If something genuinely cannot be automated, say the exact command to run.

`DEVELOPMENT.md` documents the whole loop. The preview is the interface running
on fixtures - good for layout, flow and interaction on a real phone, useless
for writing quality or time-to-first-audio, which need the server.

**To show or judge the product rather than change it, `./demo.sh`** (PROBLEMS.md
§50). It reports what the machine will actually do before it starts - canned
script with no API key, placeholder tone with no voice model, dead Explore tab
with an empty cache - refuses to start quietly broken, offers to seed, and then
says where to press on each tab. `python tools/seed_demo.py` writes the history
the browse surfaces need: Explore replays other listeners' episodes and by
design cannot generate one, so on a fresh database it stays empty however much
you tap it.

## Picking this up in a new session

Everything is in the repo; nothing of consequence lives in a chat log. Branch:
`claude/search-podcast-audio-generator-ed4br1` — develop and push there, and do
not open a pull request unless asked.

Read in this order: this file for where it is going and what is settled,
`PROBLEMS.md` for every problem hit and its cause (newest last — §46-53 are the
most recent), `DEVELOPMENT.md` for the loop.

A fresh container has none of the dependencies installed. Setup is two lines,
and the second one is not optional:

    pip install -r requirements.txt
    pip install playwright        # or the browser smoke test skips itself

Then run `./dev.sh check` before changing anything, so you know the baseline is
green rather than assuming it. A complete run ends with `all checks passed` and
twelve named smoke behaviours; anything less means something was skipped, and
`dev.sh` now says so out loud (PROBLEMS.md §49).

What is true but not obvious from the code:

- There is **no API key** in the build container, so writing quality and
  time-to-first-audio cannot be verified here. Tests, the interface checks and
  the browser smoke test all run without one. Anything about *how the writing
  sounds* is unverified until someone runs it with a key.
- The checks answer "does it work", not "does it look right". `tools/shots.py`
  photographs all nine surfaces so a refactor can be proved neutral;
  `tools/gap_probe.py` measures dead air without a key. Both exist because a
  claim was once made without them and was wrong.
- Deleting CSS from `static/index.html` has broken this app twice. Use
  `tools/check_css.py` and `tools/shots.py`, not judgement.

## Working notes

- `PROBLEMS.md` is the engineering log: every problem hit, its cause, its fix,
  and what is still open. Add to it rather than starting fresh notes.
- Tests run with no API key and no speech engine (`python -m pytest tests/ -q`).
- `diagnose_api.py` explains connection failures; `compare_models.py` compares
  cost, speed and output across models.
