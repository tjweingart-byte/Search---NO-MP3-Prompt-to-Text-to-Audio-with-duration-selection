# FAM on iOS — the building blocks

This file exists before the app does. Its job is to record which of FAM's
settled decisions survive contact with iOS, which ones iOS breaks, and what
has to be true before a build can go to TestFlight — so that work done between
now and then is app-shaped by default rather than retrofitted.

Nothing here asks for the app to be built yet. It asks for every change from
here on to be made as though it will be.

## What an app version is, and is not

**It is not a web view around `static/index.html`.** Two independent reasons,
either one fatal:

* Review guideline 4.2 rejects a repackaged website, and this would be one.
* More importantly, the audio would be Web Audio inside a `WKWebView`, and iOS
  suspends that `AudioContext` when the app is backgrounded or the phone is
  locked. The episode stops when the listener puts the phone in their pocket.
  That is the product failing at the exact moment it is most useful.

**It is a native client of the same HTTP API.** The server keeps doing what it
already does. What changes is that it now has two clients instead of one, and
therefore that the API — not `index.html` — is where behaviour lives.

## The three things iOS breaks

### 1. Playback stops being the browser's problem

`static/fam-audio.js` is not a file to port line by line, it is a **spec**, and
it was paid for in bugs. Keep its decisions:

* every sample retained as `Int16` in a growing buffer, because you cannot seek
  backwards through audio you threw away;
* a position cursor derived from the audio clock, not from wall time;
* `TAIL_MARGIN` — never seek within two seconds of what has arrived, or
  playback starves and further skips appear to do nothing.

What the browser was doing for free and iOS will not:

* `AVAudioSession` category `.playback`, and the `audio` background mode, or
  there is no lock-screen playback at all;
* `MPNowPlayingInfoCenter` and `MPRemoteCommandCenter` — lock screen, Control
  Centre, CarPlay, AirPods stems. The fifteen-second buttons and the drag are
  settled transport (CLAUDE.md); on a phone they are also hardware buttons.
* interruptions (a phone call) and route changes (AirPods pulled out). Neither
  exists in the browser build and both are ordinary on a phone.
* **Duration is unknown while streaming.** Now Playing wants a total; the
  episode is still being written. Report what has arrived, the same as the
  scrubber does, and update it.

The Swift side is `AVAudioEngine` + `AVAudioPlayerNode` with scheduled PCM
buffers. That is the same architecture as the Web Audio version, which is why
the port is a port and not a redesign.

### 2. Identity has to work without a browser

Today a listener is an HttpOnly cookie the server minted, and `?user=` is
ignored wherever it appears. **That invariant is not negotiable and is not the
thing iOS breaks** — a server-minted bearer token in the Keychain satisfies it
exactly as well as a cookie does. What iOS breaks is the *carrier*:
`URLSession` will hold the cookie in `HTTPCookieStorage`, but that store is
cleared by conditions the app does not control, and "the listener silently
became a different listener" is the worst possible failure for a product whose
whole personalisation model is an append-only log keyed on that id.

So, before the app: `/api/auth/*` should be able to hand back a session token
the client stores in the Keychain and sends as `Authorization: Bearer`, with
the cookie path untouched for the web. One session model, two carriers.

The rule for anything written between now and then, unchanged in spirit and
extended in wording: **take the listener from `_listener(request)`, never from
a parameter — and make sure `_listener` can be satisfied by a header.**

### 3. Bandwidth stops being a footnote

2.65 MB/min uncompressed. A ten-minute episode is 26 MB, on a cell connection,
with no file to resume from if it drops. CLAUDE.md already lists compression as
"the right answer at scale"; the app is what turns *at scale* into *now*.

The settled constraint survives intact: **no MP3, no audio files.** Opus in an
Ogg or WebM stream is still a stream — it is decoded as it arrives and nothing
is ever written to disk. That is compatible with the constraint and always was.
What is not compatible is producing a file and handing over a URL to it.

Note the ordering this implies: compression is a *prerequisite* of the app, not
a polish item after it.

## What the App Store requires that FAM does not have

Verify each against the current guidelines when the submission is real — they
move — but these are the ones that apply to this app specifically, and two of
them are hard rejections rather than notes.

* **In-app account deletion (5.1.1(v)) — hard.** `accounts.py` has sign-up,
  login, and password change. There is no delete. An app that creates accounts
  and cannot delete them is rejected. It is also more interesting here than
  usual: deleting a listener means deciding what happens to their events, their
  echoes, their public mixes, and their rows in the shared script cache — and
  the cache is *shared*, so their scripts are other listeners' Explore feed.
  The answer is almost certainly: delete the identity and its personal stores,
  keep the cached scripts unattributed. Decide it deliberately, once.
* **Generated and other-listener content (1.2) — hard.** FAM speaks an answer
  to an arbitrary typed question, and **Explore replays episodes other
  listeners generated**, with echoes carrying names. That is squarely the
  user-generated-content rule however the text was written. Needed: a way to
  report an episode from the player and from an Explore card, a way to block a
  listener whose echoes you keep seeing, a filter on the generation path, and a
  published contact route. Explore is the exposure surface; search alone would
  be a much smaller argument.
* **Age rating.** An app that will say arbitrary things in response to
  arbitrary questions rates conservatively. Assume 17+ unless the filter above
  is strong enough to argue otherwise.
* **A demo account and a server that is up (2.1).** Review will run against
  whatever the backend is doing that afternoon. If the GPU is cold or the key
  is unset, a reviewer hears the placeholder tone — and the placeholder tone is
  deliberately indistinguishable from a broken app, because that is what it is
  for. Reviewing FAM requires a warm, keyed, funded backend for the duration.
* **In-app purchase (3.1.1)** the moment anything is charged for. `metering.py`
  knows what a listener costs; it deliberately has no quota and no billing.
  Nothing needs to change until there is a price, but the shape of the answer
  is: the ledger is already per-listener, so entitlement is a lookup, not a
  rebuild.
* **Privacy nutrition labels, and the ledger.** The metering ledger is
  per-listener cost data, which METERING.md already calls the most sensitive
  thing the app stores after the password hashes. It gets declared.
* **Export compliance.** HTTPS only, standard exemption, but it is a question
  on every submission.

## The path, in the order it has to happen

**Stage 0 — the demo has to be true.** Open problem #1 in CLAUDE.md: nobody has
heard a FAM episode in the production voice. Everything below is scaffolding
around a product nobody has listened to. `RUNPOD_PRODUCTION.md`, then judge the
writing with `write.py`. Do not start Stage 1 until an episode has been heard
end to end and the endings are known to be good.

**Stage 1 — the server becomes a public API rather than a localhost app.**
A stable host with TLS and a name that does not change; a GPU that is up when a
phone asks; bearer-token sessions beside the cookie; CORS for exactly the
origins that need it; a version prefix so a shipped app is not broken by a
server deploy; and a real quota. `RATE_LIMIT_SECONDS=3.0` is a debounce, not a
budget — it stops double-taps and stops nothing else. A public beta with no
ceiling on episodes per listener per day is an uncapped bill against an
uncapped GPU.

**Stage 2 — the audio spike, and it is the go/no-go.** One throwaway Swift app
that plays one episode from the real server, from PCM over a live stream, and
keeps playing with the phone locked and the app backgrounded for ten minutes.
No screens, no login, no design. If this does not work, nothing above it
matters; if it does, everything else is ordinary app work.

**Stage 3 — compression.** Opus over the stream, both clients, measured against
the 0.5s to first audio that the one-sentence spec protects. If it costs
first-word latency it is wrong and gets reverted.

**Stage 4 — the app proper.** Every screen already has an endpoint:
search → `/api/audio`, myFAM → `/api/myfam`, DailyFAM → `/api/mixes`,
Explore → `/api/explore` with `cached_only`, Go Deeper → `/api/next`, profile →
`/api/profile`. That mapping is the reason this is a shell rather than a
rewrite, and keeping it true is the standing instruction below.

**Stage 5 — the compliance work above**, which is real feature work and not
paperwork: delete, report, block, filter.

**Stage 6 — TestFlight.** Internal testers first (no review), then external
(reviewed, and the 1.2 items must be in by then). Budget the beta as GPU-hours
against tester count, because concurrency is bounded by the card, not by the
API key — CREDENTIALS.md already says so.

## What this changes about every commit from here

1. **Every feature is an API before it is a screen.** Behaviour that lives only
   in `static/index.html` is behaviour the app will not have and will have to
   be written twice. When adding to the interface, ask what a second client
   would call.
2. **Nothing new on the audio path may assume a browser.** No dependency on
   `AudioContext` semantics, on cookies riding along automatically, or on a
   relative URL.
3. **Every new listener-scoped endpoint takes its id from `_listener`**, and
   must be satisfiable by a header rather than only a cookie.
4. **`/api/audio`'s wire format now has two consumers.** A change to the
   framing, the sample rate header, or the parameter set is a change to a
   shipped app that cannot be redeployed with the server.
5. **Everything in "Constraints that are settled" still applies.** The app does
   not get to reopen no-MP3, no-filler, duration-as-ceiling, or the listener-id
   rule. Compression is the one thing that looked like an exception and is not.
