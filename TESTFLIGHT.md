# FAM on TestFlight

How a build gets from this repository onto a tester's phone, and what has to be
true before it can. `IOS_APP.md` is why the client is native; `BETA_READINESS.md`
is the schedule and the options weighed. This is the procedure.

The shape is the one `DEPLOY.md` already uses for the server: **once, ever**,
then **every release**, then **what is deliberately still manual.**

## What is in the repository now

    ios/
      project.yml                  XcodeGen spec — generates FAM.xcodeproj
      ExportOptions.plist          app-store-connect, symbols uploaded
      Gemfile                      fastlane, pinned
      fastlane/
        Fastfile                   beta · external · certificates
        Appfile                    no Apple ID, no credential
        .env.example               every variable, none of the values
      Sources/FAM/
        FAMApp.swift  Config.swift
        Audio/   PCMStreamPlayer · AudioSessionController · NowPlayingController
        Net/     APIClient · AudioStream · SessionStore
        Models/  Models.swift
        UI/      RootView · PlayerView · DebugView · EpisodeModel
        Support/ Info.plist · FAM.entitlements
    .github/workflows/ios-testflight.yml

**The `.xcodeproj` is generated, not committed.** It is a directory of
machine-written XML that conflicts on every concurrent edit and cannot be
reviewed; `project.yml` is the reviewable version of the same thing and the one
place the bundle id, the background mode and the per-configuration server URL
are set.

### Three things in the client that are not negotiable

* **`UIBackgroundModes: audio` plus `AVAudioSession` category `.playback`.**
  Without them the episode stops when the phone locks. This is the whole reason
  the app is not a web view.
* **`PCMStreamPlayer` keeps every sample as `Int16` and derives its cursor from
  the audio clock**, with a two-second `tailMargin`. Those three decisions are
  ported from `static/fam-audio.js` because each was paid for in bugs there.
* **The listener id never comes from the client.** `SessionStore` holds a
  server-minted bearer token in the Keychain and `APIClient` sends it as
  `Authorization: Bearer`. There is no code path that sends `?user=`.

### What v1 deliberately leaves out

Search, the player and Go Deeper. **No Explore, no echoes** — Explore replays
other listeners' episodes and echoes carry names, which is squarely guideline
1.2 however the text was written. Leaving it out keeps the first review, the one
most likely to be rejected, a much smaller argument. Every other surface already
has an endpoint and is v1.1.

## Before any of this: three things must be true

TestFlight cannot paper over any of them.

1. **A deployed server with a GPU that is up when a phone asks.** `DEPLOY.md`,
   then `python tools/demo_preflight.py`. A build pointed at a cold server plays
   the placeholder tone, which is deliberately indistinguishable from a broken
   app.
2. **An Apple Developer Program membership.** $99/yr. Individual is usually
   same-day; **organisation enrolment needs a D-U-N-S number and can take 1–2
   weeks.** Start it first — it is pure calendar time and it blocks everything.
3. **Somebody has heard an episode in the production voice.** Open problem #1.
   Shipping an unlistened product to testers spends the one thing a beta is for.

## Once, ever

**1. Register the app.** In App Store Connect: My Apps → + → New App. Bundle id
`com.fam.app` (create the App ID first at
Certificates, Identifiers & Profiles, with **Sign in with Apple** enabled —
the entitlement is already in `FAM.entitlements`, and Apple's sign-in is not
optional once Google's is offered).

**2. Mint an App Store Connect API key.** Users and Access → Integrations →
App Store Connect API → +. Role **App Manager** is enough to upload and
distribute. Download the `.p8` **once** — Apple will not show it again.

This is the step that makes everything else unattended. There is no Apple ID
password anywhere in this setup, no app-specific password and no 2FA prompt to
babysit: the key authenticates as itself. Same principle as `FAM_SECRETS` on the
server — *a credential is never something a human types.*

**3. Put the credentials where CI can reach them.** GitHub → Settings →
Secrets and variables → Actions.

| Secret | What |
|---|---|
| `ASC_KEY_ID` | the key's id, e.g. `ABCD123456` |
| `ASC_ISSUER_ID` | the issuer UUID on the same page |
| `ASC_KEY_P8` | `base64 -i AuthKey_ABCD123456.p8 \| tr -d '\n'` |
| `FAM_TEAM_ID` | your ten-character team id |
| `FAM_REVIEW_*`, `FAM_DEMO_*` | external lane only — see below |

| Variable | What |
|---|---|
| `FAM_API_BASE_URL_RELEASE` | **the deployed server.** The workflow refuses to run without it |
| `FAM_APP_IDENTIFIER` | `com.fam.app` |
| `FAM_INTERNAL_GROUP` | e.g. `Internal` |
| `FAM_EXTERNAL_GROUP` | e.g. `Public Beta` |

`FAM_API_BASE_URL_RELEASE` is a variable rather than a default on purpose. A
build quietly pointed at the wrong server is the exact failure mode this project
has lost the most time to, so the workflow fails loudly instead of guessing.

**4. Create the TestFlight groups.** App Store Connect → your app → TestFlight.
Internal testers are App Store Connect users — up to 100, **no Beta App Review**,
and the build is testable as soon as processing finishes. That is where a beta
starts.

## Every release

One button. Actions → **iOS · TestFlight** → Run workflow, fill in what testers
should look at, choose `internal`.

What it does: checks out, pins Xcode, generates the project from `project.yml`,
sets the build number to the CI run number, archives Release, exports with
symbols, and uploads. Roughly 15–25 minutes, most of it Apple's processing.

Locally, the same thing:

    cd ios
    cp fastlane/.env.example fastlane/.env   # fill it in; .env is gitignored
    bundle install
    bundle exec fastlane beta

**The build number is the CI run number, never a human's guess.** Two builds
sharing a number is the most common way an upload is rejected *after* the slow
part has already run.

**Export compliance is already answered.** `ITSAppUsesNonExemptEncryption` is
`false` in `Info.plist` — HTTPS only, standard exemption — so App Store Connect
does not stop each build on the question. Revisit it the moment any non-exempt
cryptography is added.

## Going external

`distribute: external` runs the `external` lane. Up to 10,000 testers by email
or public link, and **Beta App Review** on the first build — typically 24–48
hours; later builds usually pass automatically unless significantly changed.

Do not run it until these are in, because they are what the review is looking
for:

- [ ] **A filter on the generation path.** The app answers arbitrary typed
      questions out loud.
- [ ] **Report an episode**, from the player.
- [ ] **A published contact route**, reachable from inside the app.
- [ ] **Privacy nutrition labels**, including the metering ledger — `METERING.md`
      calls it the most sensitive thing stored after the password hashes.
- [ ] **An age rating.** Without a strong filter, assume 17+.
- [ ] **A demo account that works**, and a backend that is warm, keyed and
      funded for the duration of the review.

The last one is not paperwork. Review runs against whatever the backend is doing
that afternoon; if the GPU is cold, the reviewer hears the placeholder tone.

`DELETE /api/account` already satisfies 5.1.1(v), which is a hard rejection
without it.

## What is deliberately still manual

**Starting and paying for the GPU.** Nothing in this repository starts, stops,
resizes or pays for a pod, and that stays true. This makes the release
reproducible; it does not make the infrastructure automatic.

**Deciding a build is worth a tester's attention.** The workflow is
`workflow_dispatch`, not `on: push`. A build on every commit burns the 90-day
expiry clock and trains testers to ignore the notification.

**The first run on a new machine.** Xcode must sign in once to create signing
assets, or set `FAM_USE_MATCH=1` and run `fastlane certificates`.

## What this costs to run

Builds expire after **90 days**. Budget the beta as GPU-hours against tester
count, not as API spend: Chatterbox runs in-process, so concurrency is bounded
by the card, not by the credential. `python tools/usage_report.py` reads back
what listeners actually cost, and the median is printed next to the p99 because
on a measured run the worst listener cost 68× the median — a beta's worth of
testers is exactly where that spread shows up.

## Honest status of the code in `ios/`

**None of the Swift has been compiled.** This repository's container is Linux;
`xcodebuild` exists only on macOS, so the first `xcodegen generate && fastlane
beta` on a Mac is also the first compile. Expect to fix small things there.

What *has* been checked here: `project.yml` and the workflow parse as YAML, the
plists and entitlements parse as plists, and the Fastfile passes `ruby -c`.

The one file worth reviewing by hand before trusting it is
`PCMStreamPlayer.swift`. It is the port of the spec, and the spec's own comments
record that its three decisions were each paid for in bugs. The go/no-go is
still what `IOS_APP.md` says it is: **one episode, from the real server, with
the phone locked, for ten minutes.**
