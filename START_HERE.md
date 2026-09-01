# Run the demo

Two ways to look at this. They answer different questions, and only the second
one can tell you anything about the writing.

## 1. The interface, right now, no install

`fam-preview.html` — open it in any browser, or on your phone. Every tab, tap,
swipe and control is the real interface code running against fixtures.

It has **no model and no audio behind it**. Use it for layout, flow and
interaction. It cannot tell you whether an episode is any good.

## 2. The product, generating real episodes

Needs: Python 3.10+, an Anthropic API key, and about two minutes.

### Set up once per machine, not once per copy

```sh
python setup_key.py       # paste the key once; checked, then stored
python setup_voices.py    # the neural voice, ~60 MB
```

Both write to `~/.fam/` — **outside the project folder**. That is the point:
every new copy of the app finds them already there, so you never paste the key
again and never re-download the voice. `python setup_key.py --show` says which
file the key came from and whether Claude still accepts it.

The key is never written into the source. Source gets committed, and a key in a
commit has to be rotated — it stays in the history after the line is deleted.

`setup_key.py` refuses to store a key Claude rejects, so "bad key" is answered
while you are setting it up rather than mid-episode.

### Then, every time

```sh
./demo.sh
```

That is the whole thing. On a machine with nothing installed it offers to
create a `.venv`, install into it, download a voice model, and take your API
key — then it tells you what it is about to do and starts the server:

```
  On this machine   http://localhost:8000
  On your phone     http://192.168.1.24:8000   (same wifi)
```

Open the phone address. Same wifi, real generation, real audio.

### It will stop rather than pretend

Two things make a demo look like it works when it does not, and both are
checked before the server starts:

* **No API key** — every episode becomes the same built-in sample script. It
  reads fine. It tells you nothing about the model.
* **No voice model** — playback falls back to a placeholder tone.

`./demo.sh --anyway` starts regardless, if you only want to see the interface.

### Explore needs seeding, once

```sh
python tools/seed_demo.py
```

Explore replays episodes *other listeners generated* and refuses to generate
one itself — that guarantee is in the pipeline, not the interface — so on a
fresh install it is empty and tapping it will never fill it. Seeding writes
eight real episodes into the shared cache and records other listeners having
played them. Eight model calls, no audio. `--dry-run` shows what it would do
and spends nothing.

`./demo.sh` offers this when it sees an empty cache.

## Where to press

| tab | what it does | what it proves |
|---|---|---|
| **search** | type a question, pick a length | the writing. This is the one to judge |
| **myFAM** | tap a tile | same pipeline, question from the shared bank; a tile someone already played starts instantly |
| **DailyFAM** | open a starter mix, play it through | a mix holds topics, never audio — different episodes each day |
| **explore** | swipe the feed | replay only; a card no longer cached says so instead of quietly writing a new one |
| **profile** | what the event log holds | nothing invented. Thin until you have played a few |

**Listen to the last two sentences of any episode.** They should land on the
most concrete thing in the piece and stop — no hook, no rhetorical question, no
"to sum up". That rule was rewritten recently and has never been checked against
real output.

After an episode, **Go Deeper** offers the follow-up the model predicted while
writing. That line is never spoken.

## If something is wrong

```sh
./dev.sh check                   # tests, interface checks, browser smoke test
python tools/demo_preflight.py   # what this machine will actually do
python diagnose_api.py           # explains connection failures
python write.py "a question" --minutes 3   # a script, printed, no audio
```
