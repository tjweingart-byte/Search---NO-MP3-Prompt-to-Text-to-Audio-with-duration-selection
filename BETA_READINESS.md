# Getting FAM to beta

What is actually built, what a beta needs, and the four ways to reach an
iPhone — with the reasoning for the one this recommends. Written so the
decision is made once, on facts, rather than re-argued later.

`IOS_APP.md` is the architecture. This is the schedule and the trade.

## The stack, measured

Verified in this repo, not recalled:

| | |
|---|---|
| Backend | Python / FastAPI — 35 modules, **62 endpoints** |
| Frontend | `static/index.html` — **308,787 bytes** of vanilla DOM, one `<script src>`, no framework |
| Build system | none — no `package.json`, no bundler |
| Swift | **0 files** — no `.swift`, no `.xcodeproj`, no `Package.swift` |
| Hybrid wrapper | none — no Capacitor, Cordova or Ionic config |
| Tests | **1288 passed, 5 skipped**; 24 named smoke behaviours on both previews |

**It is neither Swift nor React.** It is a Python API with one hand-written
HTML client. That is not a criticism — it is the fact that decides everything
below, because it means an iOS beta has no build to import and a web beta has
nothing to wait for.

## Two betas, and only one is available now

**Track A — web beta.** Deploy per `DEPLOY.md`, hand testers a URL. No Apple
involvement, no review, no 90-day expiry, fixes ship the moment they are
pushed. This tests the things that are actually unverified: the writing, the
voice, and time-to-first-audio.

**Track B — TestFlight.** Distributes `.ipa` builds. There is no Xcode project
to produce one from, so this is weeks out at best regardless of which option
below is chosen.

These are not competing. Track A is how the product gets judged; Track B is how
it gets distributed. Running A first makes B cheaper, because every writing fix
found on the web costs nothing to apply and would cost a build cycle later.

## The constraint that decides the iOS question

Everything about "can we import what we have" reduces to one mechanism.

`static/fam-audio.js` is the player. It is Web Audio: `new AudioContext()`, a
growing `Int16Array`, a cursor derived from `ctx.currentTime`, and a
`TAIL_MARGIN` of 2.0s that stops a seek starving playback.

**iOS suspends a `WKWebView`'s `AudioContext` when the phone locks or the app
backgrounds.** There is no flag, entitlement or configuration that changes it.

    web view   ▸ play ▸▸▸▸▸ [phone locks] ──── suspended, silence ────
    native     ▸ play ▸▸▸▸▸ [phone locks] ▸▸▸▸ keeps playing ▸▸▸▸▸▸▸▸

For a product whose entire premise is that audio starts in half a second and
keeps going, stopping when the phone goes in a pocket is not a bug at the edge.
It is the product failing at the moment it is most useful.

So: **the audio path is native in every option that ships to the App Store.**
What the options differ on is how much of the *other* 308KB gets reused.

Guideline 4.2 is the second reason, and the weaker one — a repackaged website
is rejected as minimum functionality. It is weaker because it is arguable;
the `AudioContext` suspension is not arguable, it is measurable.

## The four options

### 1. Native Swift client — what `IOS_APP.md` prescribes

Port `fam-audio.js` to `AVAudioEngine` + `AVAudioPlayerNode` with scheduled PCM
buffers, and rebuild the screens in SwiftUI against the 62 endpoints.

**For** — the audio works, including locked, backgrounded, on AirPods and in
CarPlay. `MPNowPlayingInfoCenter` and `MPRemoteCommandCenter` come with it, and
the settled fifteen-second buttons and drag become hardware buttons for free.
Zero 4.2 risk. Aligns with every Apple system the product will eventually want
— StoreKit 2, Sign in with Apple, background audio, Handoff.

**Against** — the most new code. The screens get written a second time, and
from then on every feature is built twice unless it stays an API-first change.

### 2. `WKWebView` wrapper

Ship `static/index.html` in a web view.

**For** — fastest to an `.ipa`. Days.

**Against** — **it does not work.** Audio stops when the phone locks. Plus a
live 4.2 rejection risk. This is the option that looks like the shortcut and is
actually the only one that fails at its own job. Rejected in `IOS_APP.md` and
the reasoning still holds.

### 3. Capacitor shell + native audio plugin

Keep `index.html` for the screens; write a Swift plugin that owns the PCM
stream, `AVAudioSession`, background playback and lock-screen transport.

**For** — reuses the 308KB of interface, which is the single largest asset. One
codebase for the screens, so a change lands in both clients at once. The audio
is genuinely native, so the fatal problem is solved. Capacitor apps do ship on
the App Store routinely.

**Against** — the hard part is not avoided, only relocated: the audio plugin is
most of the Swift work in option 1. What is saved is the screens, and the cost
of saving them is a JS↔native bridge on the hot path — every PCM chunk crosses
it — which is exactly where the 0.5s first-audio budget lives. 4.2 risk is real
but manageable given genuine native capability. Adds a Node build step to a repo
that deliberately has none.

### 4. PWA only — no App Store

Add a manifest and a service worker; testers "Add to Home Screen".

**For** — no Apple process at all. Ships today.

**Against** — same lock-screen death as option 2. Fine for Track A, not a
substitute for an app.

## Recommendation

**Track A now, option 1 for iOS, with a scoped v1.**

Two reasons, and the second is the one worth arguing about.

**First: build the audio spike before choosing anything.** Stage 2 in
`IOS_APP.md` — one throwaway Swift app, no screens, no login, that plays one
episode from the real server and keeps playing with the phone locked for ten
minutes. Options 1 and 3 share that work entirely. It is therefore not a
commitment to either, it is the measurement that tells you whether the product
can exist on iOS at all. Do it before writing a line of anything else.

**Second: cut Explore from v1.** `IOS_APP.md` notes that "Explore is the
exposure surface; search alone would be a much smaller argument" for guideline
1.2. Take that seriously and it changes the schedule. A v1 of search + player +
Go Deeper + myFAM, with no Explore and no echoes, means:

- no block-a-listener (you cannot see another listener),
- a much narrower 1.2 argument — AI-generated content still needs a filter, a
  report control and a published contact route, but not a social moderation
  stack,
- a plausible age rating argument below 17+, instead of assuming 17+,
- and less to get wrong in the first review, which is the review most likely
  to be rejected.

Explore ships in v1.1, once the app is through review once. This is the
difference between a compliance workstream in front of the first build and one
behind it.

## The path to TestFlight

Sequenced by dependency, not by preference. Stages 1–3 run in parallel with
each other; nothing after stage 4 can start early.

**0 — Hear an episode.** `RUNPOD_PRODUCTION.md`, then judge the writing with
`python write.py "<query>" --minutes 3`. Nobody has heard FAM in the production
voice, and the rewritten endings (PROBLEMS.md §48) have never been heard at all.
Everything below is scaffolding around a product nobody has listened to. This
gates the web beta too.

**1 — Enrol in the Apple Developer Program.** $99/yr. Individual is usually
same-day; **organisation enrolment needs a D-U-N-S number and can take 1–2
weeks**. Start it today regardless of which option wins — it is pure calendar
time and it blocks everything.

**2 — Deploy the API.** A stable host, TLS, a name that does not change, and a
GPU that is up when a phone asks. `DEPLOY.md` is the procedure;
`python tools/demo_preflight.py` is the check that it is true.

**3 — The audio spike.** The go/no-go. Ten minutes of locked-phone playback
from the real server, or the iOS plan changes shape.

**4 — Compression.** Opus over the stream. 2.65 MB/min uncompressed is 26 MB
for a ten-minute episode on cellular with no file to resume from.
`IOS_APP.md` calls this a prerequisite, not polish. Measured against the 0.5s
first-audio budget — if it costs first-word latency it gets reverted.

**5 — The app.** Every screen has an endpoint already: search → `/api/audio`,
myFAM → `/api/myfam`, DailyFAM → `/api/mixes`, Go Deeper → `/api/next`,
profile → `/api/profile`. Bearer sessions, `/api/v1`, tiers and quotas are
built and tested.

**6 — Compliance features.** Real work, not paperwork: a content filter on the
generation path, report-an-episode from the player, and a published contact
route. Block-a-listener only if Explore ships.

**7 — App Store Connect setup.** Bundle ID, app record, automatic signing.

**8 — Internal TestFlight.** Up to 100 App Store Connect users. **No Beta App
Review.** Testable as soon as the build finishes processing. Start here.

**9 — External TestFlight.** Up to 10,000 testers by email or public link.
Requires **Beta App Review** on the first build (typically 24–48h; later builds
usually pass automatically unless significantly changed). Needs a demo account,
a contact email, a working feedback route, and "what to test" notes. The 1.2
items from stage 6 must be in.

Per-build, every time: export compliance (`ITSAppUsesNonExemptEncryption` —
HTTPS-only is the standard exemption, but it is asked every submission), and
privacy nutrition labels. **Declare the metering ledger** — `METERING.md` calls
it the most sensitive thing stored after the password hashes.

Builds expire after **90 days**. Budget the beta as GPU-hours against tester
count: Chatterbox runs in-process, so concurrency is bounded by the card, not
by the API key.

## Where we already align with Apple, and where we do not

Genuinely ahead — none of this is a scramble later:

- **In-app account deletion** — `DELETE /api/account`, required by 5.1.1(v),
  and a hard rejection without it.
- **Sign in with Apple** — built. Note this is not optional: offering Google
  sign-in *requires* offering Apple's. Already satisfied.
- **A versioned API** — `/api/v1` as a rewrite, so a shipped app is not broken
  by a server deploy.
- **Bearer sessions** — a Keychain token, not a cookie jar iOS clears without
  asking. The listener-id rule survives intact.
- **Per-listener cost ledger** — `metering.py`. When there is a price, StoreKit
  2 entitlement is a lookup, not a rebuild.
- **Tiers and enforced quotas** — `entitlements.py`, `quotas.py`.

Genuinely missing:

- **Delivery** (email/SMS) — so no password reset, no verified address, no
  verified number. A forgotten password is a lost account. Say so before a
  tester relies on it. The provider sign-ins have no such gap, which is a real
  argument for making them the prominent buttons.
- **Payment** — nothing sets a paid plan. Not needed for a beta, and the
  enforcement path is worth trusting before money moves.
- **Report / block / filter** — stage 6. No such endpoints exist today.
- **A deployed server.** The largest single gap, and it blocks both tracks.

## Timeline

Ranges, and they assume the audio spike passes.

| | |
|---|---|
| Web beta live | **days** — gated on stage 0 and a deployed server |
| Developer Program enrolled | **same day to 2 weeks** — start now |
| Audio spike answered | **~1 week** after a deployed server |
| Internal TestFlight | **6–10 weeks** from the spike passing |
| External TestFlight | **+2–3 weeks** — stage 6 plus Beta App Review |

The critical path is not the app. It is stage 0 and stage 2 — hearing an
episode, and having a GPU that answers a phone.

## Next three things

1. **Start the Developer Program enrolment.** Calendar time that cannot be
   compressed, and it costs nothing to have it running.
2. **Stand up the server and listen to an episode.** `DEPLOY.md`, then
   `verify_voice.py`, then `write.py`. Judge the endings.
3. **Write the audio spike.** One Swift file. Ten minutes, locked. Everything
   else waits on that answer.
