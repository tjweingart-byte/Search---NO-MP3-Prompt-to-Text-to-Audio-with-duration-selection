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

Today every episode is generated **on demand**. There used to be a fast-model
"cold open" covering that wait; it is gone (PROBLEMS.md §55). Nothing is spoken
until the real episode is.

**On the browse surfaces, that whole problem is avoidable.** myFAM and dailyFAM
know what the listener might tap *before* they tap it. So:

> **Decouple script generation from speech synthesis in time.**
> The script is the expensive part (~$0.03, several seconds, cacheable text).
> The audio is nearly free (~330x realtime, milliseconds).
> Pre-generate *scripts* for likely-next episodes; synthesise audio on tap.

That yields instant playback with no wait at all, and wastes only cheap
text when a prediction is wrong — not audio compute or bandwidth. The existing
script cache (`cache.py`) is already the right place to put pre-generated
scripts; it stores scripts, not audio, for exactly this reason.

The same "do it before the listener is waiting" logic is why matching happens
at write time too: `CACHE_VECTOR` embeds a question once when its script is
stored and compares locally on the next lookup, instead of `CACHE_SEMANTIC_KEY`'s
model call in front of every request. Off by default, and PROBLEMS.md §68 says
plainly what it is and is not currently buying.

Corollary: **latency is answered by starting earlier, never by filling the
gap.** The cold open tried to fill it and was removed. Prefetch on the browse
surfaces; on search, keep the work small enough that there is no gap to fill.

**On search, that is now literal** (PROBLEMS.md §56). A question that needs
today's facts starts two calls at once: one with no tools that begins writing
immediately, one with search that is still reading. The first is spoken while
the second works and hands over the moment research has a sentence. The wait is
covered by the answer rather than by filler - which is exactly what the cold
open could not be, since it was told to state no facts. `ANSWER_FIRST_SHARE`
caps the instant half at half the episode, because synthesis outruns research
and without a ceiling the from-knowledge half finishes the whole episode and the
research is never heard.

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

**Part of that is now paid back after the episode rather than inside it**
(PROBLEMS.md §70). When one finishes on the player, four recommendations appear
in a grid and the first starts itself in five seconds. The countdown tile
prefers the album's next episode, then the predicted `<<NEXT:>>` follow-up,
then the ranking - so the pull is there without a word of it being spoken, and
declining it is one tap. The tiles are the *feed's* ranking
(`topics.rank_next_up`), not a second one, so the popup and the shelves cannot
give a listener two different answers to the same question. It deliberately
does not fire on Explore or Explore New, which are already continuous.

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

1. **Voice quality — answered, and unheard.** **Chatterbox is the production
   voice and the only one.** Piper is gone: engine class, configuration,
   `piper-tts` dependency, `setup_voices.py` and the interim slot itself. It
   was removed rather than switched off because it reached listeners three ways
   nobody chose — `build_engine` fell through to it, `engine_for_voice` fell
   back to it, `list_voices` offered it whenever the production slot was empty
   — and an app that quietly sounds worse than intended is the failure this
   project has lost the most time to. A knob left behind is an invitation to
   turn it back on, and this one turned itself.
   There are now exactly two honest states: Chatterbox speaks, or nothing does
   and everything says so — `/api/health` reports `interim: true`,
   `build_engine` logs why Chatterbox was unavailable, `demo.sh` refuses to
   start quietly broken, and playback is a **placeholder tone**, not a lesser
   voice. A tone cannot be mistaken for FAM; a flat neural voice can.
   Hosted neural voices were not adopted: WellSaid was removed after **two
   episodes exhausted a month's quota** — a seat product used as an API, not a
   voice that was too expensive (PROBLEMS.md §61) — and per-character billing
   breaks the "audio is nearly free" premise the prefetch plan rests on
   (`VOICE_OPTIONS.md` has the arithmetic). Chatterbox has open weights and no
   quota, and clones a reference recording in the same shared per-user folder
   (`~/.fam/voices`, see `voice_store.py`) that Piper's models used, for the
   same reason: a new copy of the app must find it already there.
   *Nobody has heard a FAM episode in this voice.* The build container has no
   GPU, so `RUNPOD_PRODUCTION.md` is the procedure that closes that gap and
   `python verify_voice.py` is the check that says whether a given machine can
   speak at all. **That listening test is the next move.**
2. ~~**Voice selection**~~ — *done*. `/api/voices` lists what the machine can
   speak; `voice=` on `/api/audio` selects one; the player has a picker.
   Note: voice is deliberately **not** part of the script cache key, because a
   voice changes the audio and not the words. Switching voice therefore reuses
   the cached script — measured at ~90 ms and zero API cost.
3. ~~**The cold-open → script gap**~~ — *dissolved, not fixed* (PROBLEMS.md
   §55). Two sessions went into making the opener cover the research wait, and
   heard on a real machine it was 3-5 seconds of contentless speech in front of
   a 30-45 second silence. Five does not cover forty-five, and the opener was
   prompted to state no facts, so what it did cover was worthless. **The whole
   feature is deleted** - not switched off - along with `tools/gap_probe.py`,
   which existed only to measure it. The interface now shows an honest wait
   that names what it is waiting for and counts the seconds.
4. **myFAM is built; the taste model is deliberately crude.** `topics.py` ranks
   a *shared* bank of ~28 topics **four** ways (history / exploration /
   co-listener / trending) from an append-only event log. Tags come from
   keyword matching, not a classifier. `rank_might_like` (adjacent to your
   taste) is **back on myFAM as the Explore New rail**, and serves the Explore
   New screen behind it from the same ranking - one ranking, two views, so the
   rail and the surface it opens cannot disagree. It sits second, between
   "Made for you" and the crowd: it is the only signal offering anything
   *outside* an established taste, and without it the page is three ways of
   being told what you already like. The intro's chosen interests now seed `taste`, so "Made for you" is no
   longer honestly empty on a listener's first open. The cost design is the load-bearing part: **one bank for
   everyone, personalisation in the ordering, not the inventory** - so two
   people tapping a tile share one script through `cache.py`.
5. **playFAM is built as its own tab.** `mixes.py` stores named daily mixes -
   a mix holds *topic ids*, never audio, so "At the gym" is the same subjects
   every day and a different set of episodes. Members are validated against the
   same shared bank, which is what keeps the cost design intact.
6. ~~**"What your followers are listening to" has no follow graph behind it.**~~
   - *the graph is built* (`SHARING.md`). Follows are asymmetric, like the copy
   always said, and a **friend is the mutual case, derived and never stored** -
   no request, no accept, no pending state to get wrong. What is still true:
   the myFAM rail itself still ranks co-listener overlap rather than the graph.
   That is now a one-line change rather than a missing feature, and worth
   making deliberately - a new listener follows nobody, so a rail backed only
   by follows would be empty on the day it matters most.
7. **The social layer generates nothing, and now there is more of it.**
   `social.py` stores an echo as a row pointing at a query whose script already
   exists. `messages.py` does the same for a *directed* share - one person, one
   episode - and `sharing.py` for a link posted outside FAM. All three cost one
   row: sending an episode to ten people costs ten rows and not ten episodes,
   because their taps are what synthesise audio, from one cached script,
   against their own allowances. Mixes are private by default and appear on the
   profile once made public.
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
10. ~~**Personalisation needs state the app does not have**~~ - *identity is
   done; the recommender is still crude.* `accounts.py` gives every listener a
   server-minted session id in an HttpOnly cookie, and an account is *email and
   password attached to the id they already have* - so signing up keeps their
   history rather than starting a second listener beside it, and logging in on
   a phone reaches the same data. **Listening still works with no account at
   all** - search, myFAM, DailyFAM's episodes, Explore and Go Deeper - which is
   the constraint that stopped this becoming a login screen in front of the
   product. What an account now buys is durability: mixes, chosen interests and
   language, and the weekly recap are gated on having one (PROBLEMS.md §70, and
   the constraint above). Sign-up is now **email or phone**, and **Sign in with
   Google or Apple** attach the same way - one account, several routes in,
   keyed on `(provider, subject)` because Apple sends an address on the first
   authorization only. What is genuinely missing is one capability behind three
   gaps: **delivery**. Without it there is no **password reset**, no verified
   address and no verified number - so a phone number is an identifier rather
   than a second factor, and a forgotten password is still a lost account. Say
   so before anyone relies on it; the provider sign-ins have no such gap, which
   is a real argument for making them the prominent buttons in the app.

## Constraints that are settled — do not undo without discussing

- **No MP3, no audio files.** Raw PCM streams from the TTS engine to the browser
  and is played as it arrives. This is the core of the product. Compression
  (Opus over a stream) is compatible with it and is the right answer at scale;
  writing a *file* is not.
  **Downloads do not break this, and the reason is worth stating** *(SHARING.md).*
  The server still writes nothing: the episode streams exactly as it always
  does, and the *client* keeps the bytes it was already sent - IndexedDB in the
  browser, the app's container on iOS. No file exists server-side, nothing is
  cached as audio, and no URL serves a stored episode. `saved.py` holds a
  registry of what a listener claims to hold, never the audio - which is also
  why that registry can drift, and why releasing a slot is one tap.
- **Duration is a ceiling, not a quota.** *(Revised.)* The selected length still
  caps the episode and over-runs are trimmed, but a script that runs out of
  substance now ends early instead of being padded. Enforcing the number in both
  directions is what produced filler: it made the model pad. `ALLOW_TOPUPS=1`
  restores the old behaviour.
- **Transport: two gestures, and both stay.** *(PROBLEMS.md §71.)* The
  progress bar is draggable on all three listening surfaces, and the
  fifteen-second buttons are untouched. They answer different questions - the
  buttons "say that again", the drag "get me to roughly there" - so neither is
  a replacement for the other, and removing either would be a regression. The
  drag clamps at what has actually been written, because the episode is still
  being generated while it plays.
- **No filler, ever, and no setting for it.** The cold open was deleted, not
  disabled - a knob left behind is an invitation to turn it back on, and this
  one was turned back on by an example file. Nothing plays until the real
  briefing does. The interface says what it is waiting for and how long it has
  been waiting; a wait you were warned about is a different experience from the
  same wait unexplained.
- **Every episode is researched. (Reversed — this used to say the opposite.)**
  *(PROBLEMS.md §76.)* `SEARCH_MODE=always` is the production default and the
  question no longer gets a vote. The old rule was "search is opt-in, and the
  question opts in": `auto` read the query with the cache's freshness keywords
  and answered everything else from memory. That was correct arithmetic against
  research that cost **10-25 seconds** — the model's own `web_search` tool.
  `RESEARCH_BACKEND=exa` retrieves in about **half a second**, and at that price
  the guess only ever loses: a question it gets wrong is answered from memory
  that may be a year stale, and one it gets right saves nothing a listener can
  hear. Production proved it on `49ers game last night`, logged as *"nothing in
  it reads as time-sensitive"*. A keyword list can always be widened by one more
  word, and the next question it misses is already written.
  `auto` and `never` are kept and are **not production** — offline `write.py`,
  `tools/compare_search.py`, a deployment with no Exa key. `search=1`/`search=0`
  on a request still wins. This does not reopen the one-sentence spec: half a
  second is not seconds in front of the first word, and if that ever stops being
  true the answer is `ANSWER_FIRST=1`, not guessing again.
  **And a tool is not an instruction** *(§77).* When an episode is researched
  and no evidence packet came back — `RESEARCH_BACKEND=claude`, or Exa finding
  nothing usable — `_request_kwargs` attaches the `web_search` tool, and
  `build_prompt` must *ask the model to use it*. It did not, for as long as the
  tool has existed; always-on research turned that from a rare case into every
  episode, which is how it was finally seen. The packet and the tool stay
  alternatives, never both.
- **An account gates what is kept, never what is heard.** *(PROBLEMS.md §70.)*
  Saved mixes, chosen interests and language, and the weekly recap need an
  account; search, myFAM, DailyFAM's episodes, Explore, Go Deeper and the whole
  audio path do not. The interaction log is deliberately outside the gate - it
  is ambient personalisation rather than something the listener made and can
  point at, and gating it would mean an anonymous feed could never be ranked.
  `ACCOUNT_REQUIRED` in `app.py` holds the reasoning beside the code that
  enforces it. Nothing is lost by signing up late: a mix made before the gate
  is still under the same id and appears the moment credentials are attached.
- **A tier is what you may spend, never what you may reach.** *(ACCOUNTS.md.)*
  Three tiers - `free`, `plus`, `unlimited` - and the free one is a **daily
  ceiling on episodes**, not a smaller product: `entitlements.FEATURES` gates
  capabilities and today every tier has every one of them. The mechanism ships
  on and the policy ships empty, because taking away something every listener
  has always had is the "quietly worse than intended" failure again, with the
  twist that here they notice and are right. Moving a feature behind a tier is
  one line plus the test that fails when you do, which exists to make it a
  decision somebody wrote down.
  Two things counted separately, because they cost differently: an **episode**
  may write a script, an **Explore replay** provably cannot. A cache hit is
  still an episode - the listener heard one and the GPU made it. And the free
  quota is **a budget shaped like a limit, not a security control**: an
  anonymous session can be thrown away and a fresh allowance started, which is
  the price of not putting a login in front of the first word.
- **Save for later and download are different things, and stay different.**
  *(SHARING.md.)* Saving is a **pointer** - question, length, folder - and
  playing one needs the network like any other episode. Downloading is **the
  audio on the device** and plays with the network off. A download is an
  upgrade to a saved item rather than a second list, which is why saving asks
  the question and why one row carries both states. The limit is per tier and
  is a **standing capacity, not a rate** - a windowed counter would hand out a
  fresh download allowance every morning and never require anybody to delete
  anything. A full shelf is a 409 that **names what to clear**, least recently
  played first, because a limit without a remedy is a dead end on a phone.
- **FAM posts nothing to anybody's social account, and holds no token.**
  *(SHARING.md.)* Every external destination is reached from the phone: the
  share sheet, or a platform SDK hand-off where their app does the posting with
  the person watching. The server produces the link, the wording and - for
  Instagram and Snapchat, which cannot carry a link as text - the story card.
  This is the correct shape rather than a stage: no OAuth to maintain, no
  tokens to leak, and nothing that can post while somebody is asleep.
- **A listener id is never accepted from the client.** It arrives from an
  HttpOnly session cookie the server minted, and `?user=` is ignored wherever
  it still appears. This replaced `famUserId()`, which made an id up with
  `Math.random()` and put it in every query string - so anyone who read or
  guessed one could take over that listener. Anonymous listeners still get a
  full identity, because requiring a login to hear an episode would break the
  one-sentence spec. If you add an endpoint that touches per-listener data,
  take the id from `_listener(request)` and never from a parameter.
  **Two carriers now, one rule.** A native client cannot rely on a cookie jar
  iOS clears without asking, so the same server-minted token is also accepted
  as `Authorization: Bearer` and stored in the Keychain. Nothing about the
  trust changes - a bearer token is the same unforgeable, revocable string the
  cookie holds - and a browser must never ask for one, because reading it in
  script is what HttpOnly exists to prevent.
- **Failures must be visible.** Silent success (empty audio, a placeholder tone,
  demo mode mistaken for live) has caused more lost time on this project than
  any real bug. Every fallback must announce itself. *(PROBLEMS.md §51: demo
  mode did announce itself, in an 8.5px chip, and still cost a whole session -
  and it was writing its canned script into the shared cache, so the failure
  outlived the run. Announcing is not enough if the thing keeps a record.)*
- **A running server says which code it is running.** *(PROBLEMS.md §77.)*
  `/api/health` reports `build` (the commit, from `RENDER_GIT_COMMIT`,
  `FAM_COMMIT` or `git rev-parse`, and `"unknown"` rather than a guess) and
  `search_mode_source` (env var or code default). Both exist because a session
  went into inferring them: "the fix is pushed" and "the fix is live" are the
  same sentence from outside, and an env var beats a code default silently and
  outlives every push. Anything else that can be set in two places belongs
  here too.
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
- **What a listener costs is recorded when it is spent** *(§73).* The provider
  only ever sees one account, so "which listener produced which request" has to
  be answered at the moment of spend or not at all. `metering.py` appends one
  row per episode, tagged from `_listener(request)`; `python tools/usage_report.py`
  and the admin-gated `/api/usage` read it back. The load-bearing part is that
  it never reports one blended cost per user: Claude and Exa are **marginal**,
  the GPU is a **fixed floor** that exists before the first listener, and the
  shared cache is a **discount that grows with listeners** - averaged together
  they describe how many listeners there are rather than what one costs. Every
  number says whether it is billed, priced or assumed, and the median is printed
  next to the p99 and the max because on a measured run the worst listener cost
  68x the median. `METERING.md` is the whole of it - including what it
  deliberately does not do: no quota, no enforcement, no billing, and no
  automatic block on an abuse signal.
- **A credential is never something a human types** *(extends the above; §72).*
  `~/.fam/env` solved this for one machine, and the demo does not run on one
  machine — a pod, a container and a CI runner each arrive with an empty
  `~/.fam`. `FAM_SECRETS` names a place the app fetches its own credentials
  from (`file:` or `cmd:`, so every secrets manager works and none becomes a
  dependency), and it is the one *non-secret* line a new machine needs. The
  order is process env > `FAM_SECRETS` > project `.env` > `~/.fam/env`, an
  explicit variable always wins, and a provider that is set and broken says so
  at startup, on `/api/health` and in the preflight rather than falling through
  to the canned script. `CREDENTIALS.md` is the whole of it — including what it
  deliberately does *not* buy: Anthropic's rate limits are per organisation, so
  a pool of keys is failover and not headroom, and the real ceiling on
  concurrency is the GPU, not the credential.

## There is an iOS app coming, and it changes how to write everything else

`IOS_APP.md` is the whole of it. The short version, because it constrains work
that has nothing to do with the app:

**The app is a native client of this API, not a web view around
`static/index.html`.** A wrapper is rejected under guideline 4.2, and worse, iOS
suspends a `WKWebView`'s `AudioContext` when the phone locks - so the episode
would stop the moment it is most wanted.

Three consequences for ordinary changes, starting now:

- **Every feature is an API before it is a screen.** Behaviour that exists only
  in the interface is behaviour that has to be written a second time.
- **Nothing new on the audio path may assume a browser** - not `AudioContext`
  semantics, not a cookie riding along on its own, not a relative URL.
  `static/fam-audio.js` is the port's specification: the retained `Int16`
  buffer, the clock-derived cursor and `TAIL_MARGIN` were all paid for in bugs.
- **A listener id still never comes from the client, and `_listener` must be
  satisfiable by a header**, not only by the cookie. A server-minted bearer
  token in the Keychain keeps that rule exactly; `?user=` never does.

The two settled constraints the app pressures, and how they resolve: **no MP3,
no audio files still holds** - Opus over a stream is decoded as it arrives and
writes nothing, so compression was always compatible and is now a prerequisite
rather than a scale question (26 MB for a ten-minute episode on cellular). And
**account deletion is missing**, which is a hard rejection for any app that
creates accounts - and an interesting decision here, because the script cache
is shared, so a deleted listener's scripts are other listeners' Explore feed.

**The server side of that is now built** - `ACCOUNTS.md` is the whole of it.
Email-or-phone sign-up, Sign in with Google and Apple, account settings,
in-app deletion, three tiers with enforced per-window limits, bearer sessions
beside the cookie, and every endpoint reachable at `/api/v1/...`. What is not
built, and is not an oversight: **payment** (nothing sets a paid plan yet, and
the enforcement path is worth trusting before money moves) and **delivery**
(no email or SMS, so no password reset and no verified address or number).

What has to happen, in order: hear an episode in the production voice (open
problem #1 - everything else is scaffolding around an unlistened product);
deploy the API somewhere with a GPU that is up when a phone asks; then a
throwaway Swift spike that plays one streamed episode with the phone locked,
which is the go/no-go for all of it.

## Decisions that will shape the next phase

- **Where does this deploy?** Bandwidth is 2.65 MB/min uncompressed; that is
  fine on localhost and expensive at scale.
- ~~**Is there a user account, and what does it entitle you to?**~~ *Answered
  twice: §66 for the shape, §70 for the boundary.* An identity is a session; an
  account is credentials attached to one; and what an account buys is
  **durability** - the things the server keeps for you. The constraint above
  says exactly which. Still open, and more visible than it was: **password
  reset**, which needs email delivery, and which the sign-up screen now says
  out loud rather than letting anyone find out the hard way.
- **Local or hosted voices?** Changes the cost model more than the model choice
  does.
- **How much to prefetch?** Every speculative script costs money; every one not
  fetched costs a wait.
- **Is a local embedding model worth installing?** The near-match cache
  (PROBLEMS.md §68) is built, measured and off by default. It raises the share
  of re-phrasings that find an existing episode from 22% to 56% on a measured
  corpus - but the bench's own control line shows the *vector* earning none of
  that: a free token-overlap guard finds everything the lexical embedding
  finds. A real sentence model in `~/.fam/embed` is the only thing that changes
  that answer, and it is the same trade as the voices - ship a model with the
  app, or pay a service per call. Nobody has run one yet.

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
   stale. Read it first, then publish to it.

   **What lives at that URL is now `preview/fam-live-artifact.html`**, built by
   `python preview/build_live_preview.py` - the same interface, but running on
   a real database rather than fixtures, with the store shown beside it.
   Publish it with `capabilities: {"db": {}}`; without that declaration
   `claude.use("db")` resolves null in the viewer, the page falls back to
   memory, and the whole point of it is quietly gone. The fixture build
   (`preview/fam-artifact.html`) is still what `./dev.sh check` produces and
   smoke-tests; it is just no longer what the bookmarked link serves.
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
`PROBLEMS.md` for every problem hit and its cause (newest last — §68-73 are the
most recent), `DEVELOPMENT.md` for the loop, `CREDENTIALS.md` for how a
machine gets its API keys without anybody typing one, `METERING.md` for
what a listener costs and how the report says so, `ACCOUNTS.md` for identity,
tiers, quotas and the public API, `SHARING.md` for friends, sharing, saving
and downloads, and `IOS_APP.md` for the app version this is now being written
towards.

A fresh container has none of the dependencies installed. Setup is two lines,
and the second one is not optional:

    pip install -r requirements.txt
    pip install playwright        # or the browser smoke test skips itself

Then run `./dev.sh check` before changing anything, so you know the baseline is
green rather than assuming it. A complete run ends with `all checks passed` and
twenty-four named smoke behaviours; anything less means something was skipped, and
`dev.sh` now says so out loud (PROBLEMS.md §49).

What is true but not obvious from the code:

- There is **no API key** in the build container, so writing quality and
  time-to-first-audio cannot be verified here. Tests, the interface checks and
  the browser smoke test all run without one. Anything about *how the writing
  sounds* is unverified until someone runs it with a key.
- The checks answer "does it work", not "does it look right". `tools/shots.py`
  photographs all sixteen surfaces so a refactor can be proved neutral;
  `tools/stall_probe.py` measures browser stalls without a key, and
  `tools/compare_search.py` measures what research actually buys. Each exists
  because a claim was once made without it and was wrong.
- Deleting CSS from `static/index.html` has broken this app twice. Use
  `tools/check_css.py` and `tools/shots.py`, not judgement.
- **A setting is settled only where it is copied.** `.env.example` shipped the
  cold open and web search *on* while `config.py` had them off with the
  reasoning attached (PROBLEMS.md §54), so following the documented setup
  configured the product against its own spec. `tests/test_env_example.py` now
  fails on any disagreement between the two.

## Working notes

- `PROBLEMS.md` is the engineering log: every problem hit, its cause, its fix,
  and what is still open. Add to it rather than starting fresh notes.
- Tests run with no API key and no speech engine (`python -m pytest tests/ -q`).
- `diagnose_api.py` explains connection failures; `compare_models.py` compares
  cost, speed and output across models.
