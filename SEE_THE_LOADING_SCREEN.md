# Seeing the loading screen

Three ways, in the order of how little they need from you.

## 1. Look at it — no install, no key, no server

Open **`preview/loading-screen.html`** in any browser, on a laptop or a phone.
It is one self-contained file; double-clicking it is enough.

The screen is held still so it can be judged, with buttons for the four states
the status line actually takes:

| Button | When the app shows it |
|---|---|
| Instant answer | The question can be answered from what the model knows |
| Needs today's facts | Research is running underneath, with a counter |
| No API key | Demo mode — the canned script is playing |
| Key rejected | The key was refused at startup; nothing can be written |

**Show a cache hit (450 ms)** plays the shortest wait the app can produce.
That is the floor added after the smoke test measured a 3 ms flash; anything
faster read as a glitch rather than as speed.

This page is *extracted* from `static/index.html` at build time — markup, CSS,
palette and fonts — so it cannot drift from what ships. `tests/test_loading_demo.py`
fails if it does.

## 2. See it in the app, on fixtures — no key

    pip install -r requirements.txt
    python preview/build_preview.py
    open preview/fam-preview.html          # or just double-click it

Tap search, a myFAM tile, a DailyFAM mix. The screen appears on every one of
them — but the preview answers from fixtures, so it is gone in well under a
second. Good for checking it covers the right surfaces; useless for looking at.

Same thing on your phone, already built:
<https://claude.ai/code/artifact/c8bd86aa-e61e-4262-a1c8-b9c8d8d6645e>

## 3. See it doing its job — needs a key

This is the only one that shows a real wait, because the wait is real.

    pip install -r requirements.txt
    python setup_key.py                    # stored in ~/.fam/env, once, for good
    python setup_voices.py                 # optional; without it you get a tone
    ./run.sh

Then <http://localhost:8000>, and:

* Search something evergreen — *"how does a heat pump work"*. The screen shows
  **Writing your episode…** and clears in about half a second.
* Search something that moves — *"latest on the Fed"*. It shows **Answering now
  — checking sources underneath…** with the seconds counting, because that
  question opts into research.
* Tap a myFAM tile someone has already played. Cache hit: the 450 ms floor is
  the whole of it.

Explore deliberately does **not** show it. Its reel replays episodes already in
the shared cache and refuses to generate, so a loading screen there would
promise work that is not happening.
