# Testing the WellSaid voices

Compare the **same FAM episode** in three voices: Piper, Chase J and Kai M.
Claude still writes every word — only the speaker changes.

## Where the API key goes

**Not in a file you edit, and never in the code.** Run this once:

    python setup_wellsaid.py

It asks you to paste the key (hidden as you type), asks WellSaid to speak one
short line with it, and only saves it if that works. It goes in `~/.fam/env` —
the same place as your Anthropic key, outside the project folder, readable only
by you. You will not be asked again on this machine.

Get the key from your WellSaid Labs account at studio.wellsaidlabs.com, under
**Settings → API**.

Check it any time:

    python setup_wellsaid.py --show     # is it there, and does it still work?
    python setup_wellsaid.py --remove   # forget it

## Running the test

    ./run.sh

Then open <http://localhost:8000>, search for something, and once it is
playing:

1. Tap the pill under the play button — it reads something like
   `1.2× · 3 min · Lessac (US, medium)`.
2. Tap **Voice**.
3. Pick **Chase J (WellSaid)** or **Kai M (WellSaid)**.

The episode restarts in the new voice within about a second. **It does not cost
another Claude call**: the script is already cached, and voice is deliberately
not part of the cache key, so you are hearing the identical words in a different
voice — which is the only way the comparison means anything.

Switch back to a Piper voice the same way.

## Checking it is really WellSaid

Three independent signals, so you never have to take it on trust:

* **The pill** on the player names the voice that is speaking right now.
* **The terminal** prints one line per sentence:

      fam.wellsaid: TTS provider=wellsaid voice='Chase J' speaker_id=35 chars=142
                    chunks=1 first_audio=0.31s total=0.42s audio=6.20s (15x realtime)

  If you do not see `provider=wellsaid` lines, you are hearing Piper.
* **Failures are loud.** If WellSaid rejects the key, rate-limits, or is
  unreachable, the episode stops and shows the real error. It never quietly
  falls back to Piper — an episode that silently changed voice would make the
  whole comparison worthless.

## What to expect

* **Slower to start than Piper.** Piper runs on your machine; WellSaid is a
  network round trip per sentence. That is the trade being measured.
* **No speed control.** WellSaid has no speaking-rate parameter, so the `1.2×`
  setting still works in the player but the pacing controller cannot stretch or
  compress the delivery to hit the clock exactly. Length is still capped.
* **It costs money per character.** Which is why it is never selected
  automatically — only when you pick it by name. Piper stays the default even
  on a machine where nothing else is installed.
* **If you see an ffmpeg error**, WellSaid served MP3 rather than WAV on your
  account. Install ffmpeg (`brew install ffmpeg` on a Mac) and it will decode
  in memory — still no file is ever written.

## Changing the speaker ids

The two under test are set in `.env.example` and can be overridden per machine:

    WELLSAID_CHASE_J_ID=35
    WELLSAID_KAI_M_ID=32
