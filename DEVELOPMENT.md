# The loop

The goal is one command here, and one tap on your phone.

## Every change

```sh
./dev.sh
```

Runs the test suite, checks the interface parses, builds the phone preview,
smoke-tests it in a real browser, then serves the app and prints **the address
to open on your phone** — same wifi, real audio, real generation.

```
  On this machine   http://localhost:8000
  On your phone     http://192.168.1.24:8000   (same wifi)
```

Two shorter forms:

```sh
./dev.sh check     # tests + checks, no server
./dev.sh preview   # rebuild the preview file only
```

## Three ways to look at it, and what each one proves

| | Opens on a phone | Real scripts & audio | Needs |
|---|---|---|---|
| **Preview artifact** | anywhere, one tap | no — fixtures and silence | nothing |
| **`./dev.sh`** | same wifi only | yes | this machine running |
| **Hosted deploy** | anywhere | yes | a host + API key |

The preview is the interface running for real against fake data: taps, swipes,
transport controls, the topic picker, mixes and the reels are all live code.
What it cannot tell you is anything about writing quality or time-to-first-audio
— those need the server.

## Preview builds

```sh
python preview/build_preview.py
```

Produces two files, both gitignored because they are build output:

- `preview/fam-preview.html` — a complete page. Open it from anywhere, email
  it to yourself, drop it in a Slack message. No server, no install, no network.
- `preview/fam-artifact.html` — the same page minus the outer document tags,
  for publishing as an artifact.

Fixtures are built by importing `topics.py` and `mixes.py`, not by copying
them, so a preview cannot drift from the real bank.

## Hosted deploy (the only part that needs your account)

Once, to get a URL that works off your wifi and stays up:

```sh
# 1. Check the image builds (this repo has no other build step)
docker build -t fam .
docker run --rm -p 8000:8000 -e DEMO_MODE=1 fam        # then open localhost:8000

# 2. Deploy. render.yaml is committed, so this is the whole thing:
render blueprint launch          # or: point Render at the repo in the dashboard
```

Then set `ANTHROPIC_API_KEY` in the Render dashboard — never in the repo.
`autoDeploy: true` means every push redeploys, so after the first setup there
is nothing to run.

Any container host works; Render is in the repo because it reads a Dockerfile,
offers a persistent disk, and redeploys on push. Fly and Railway need only a
different config file.

**The disk matters.** `/data` holds the script cache, myFAM history and your
mixes. Without a mounted disk every deploy is a fresh start — fine for a
preview, wrong once anyone is actually listening.

## What runs in CI

`.github/workflows/ci.yml`, on every push:

1. `pytest tests/` — no API key, no speech engine needed
2. `tools/check_js.py` — the interface is one large HTML file with inline
   script, and a stray brace there is invisible to pytest and fatal on a phone
3. `tools/check_css.py` — the same file's styles. Fails when a class is used
   with no rule behind it, or when an element with `hidden` wears a class that
   forces it visible. Deleting a block of CSS breaks nothing that fails; it
   just makes the page wrong, which is how a lost stylesheet reached a phone
   twice
4. `preview/build_preview.py` — the preview must always build
5. `tools/smoke_preview.py` — drives the built preview in a browser: every
   tab renders, and Explore actually plays and advances

The built preview is attached to each run for 30 days.

Step 5 is the one that earns its place: all three Explore bugs — the infinite
auto-advance, the demo-mode cache, the play button that resumed instead of
pausing — were invisible to the Python tests and obvious in a browser.


## Proving a change did not move anything (`tools/shots.py`)

Everything above asks whether the app *works*. None of it can see a page that
looks wrong. `tools/shots.py` is the missing half: it photographs all nine
surfaces so a refactor can be shown to change nothing.

    python tools/shots.py before
    # make the change, then rebuild:
    python preview/build_preview.py
    python tools/shots.py after
    python tools/shots.py --compare before after

This is what made it safe to delete 8.5 KB of unreferenced CSS: eight surfaces
came back byte-identical.

Two surfaces move on their own — Explore rotates its reel, and the player's
scrubber advances — so a difference there is not by itself a regression.
Settle it by capturing the *same* build twice: if the unchanged build produces
the same difference, it is the clock, not the change.

It is deliberately not in CI, for that reason.
