# Reference recordings for the voice-identity bake-off

**Nothing goes in here that we do not have the right to clone.** This folder is
git-ignored: the recordings are somebody's voice, and they are not source code.

Four files, plus a rights record for each:

    magnetic.wav      magnetic.rights.json
    human.wav         human.rights.json
    storyteller.wav   storyteller.rights.json
    authority.wav     authority.rights.json

Then:

    python tools/check_reference_audio.py experiments/references

It refuses to pass a voice with no rights record, and it checks the technical
constraints that come from how Chatterbox actually reads a reference.

## What Chatterbox Base does with the file

Read from `tts.py:182-206`, not from documentation:

| conditioning | how much of the file it uses |
|---|---|
| s3gen reference (`embed_ref`) | **first 10 seconds** (`DEC_COND_LEN = 10 * 24000`) |
| T3 prompt speech tokens | **first 6 seconds** (`ENC_COND_LEN = 6 * 16000`), capped at 150 tokens |
| voice-encoder speaker embedding | **the whole file**, untruncated |

Because the third one is untruncated, **length is not a free variable**. Four
references of different lengths feed the speaker embedding unequal amounts of
each voice while feeding the other two the same. Make them the same length.

## The spec

| | |
|---|---|
| duration | **12-15 seconds**, and **all four within 2 seconds of each other** |
| format | any file `librosa` can open - wav, flac, mp3, m4a. **Prefer WAV** |
| sample rate | 24 kHz or higher (44.1/48 kHz is ideal). Never below 16 kHz |
| channels | mono preferred; stereo is downmixed automatically |
| level | peaks below -1 dBFS. **A clipped reference clones its distortion** |
| background | quiet room. Room tone, hiss and hum are cloned along with the voice |
| processing | none. No compression, EQ, de-noise, reverb or "podcast polish" |
| content | continuous natural speech, no long gaps, no music, one speaker only |
| **transcript** | **not required.** Nothing in the code reads one |

## What to say in the recording

This is the part that decides whether the experiment measures what it is meant
to. The reference carries **delivery as well as identity** — the T3 prompt is
made of *speech tokens* from the first six seconds, which encode how the person
was speaking, not only who they are.

So: **have all four speakers read the same neutral, conversational passage**,
in the same unremarkable way. Then the thing that differs between the four
files is the speaker, which is what this experiment is choosing.

If instead each speaker performs their assigned direction — one being magnetic,
one being a storyteller — the test measures performance as much as voice, and
a later delivery-optimisation pass cannot untangle them.

Suggested reference text (about 13 seconds at a natural pace):

> I have been thinking about this for a while, and I am still not sure I have
> it right. The part that keeps surprising me is how ordinary the explanation
> turns out to be, once you actually look at it. It is not complicated. It is
> just not what anyone expects.

Neutral register, some commas, a couple of natural pauses, no performance.

## The rights record

One `.rights.json` per voice. The checker requires every field to be answered:

```json
{
  "source": "where the recording came from",
  "speaker": "who is speaking",
  "consent": true,
  "commercial_use": true,
  "synthetic_voice_cleared": true,
  "notes": "what the agreement actually says, and where it is filed"
}
```

`synthetic_voice_cleared` is its own field on purpose. Permission to *use a
recording* is not permission to *synthesise a voice from it*, and a standard
voiceover agreement often covers the first and says nothing about the second.

**Not legal advice.** The checker verifies a record exists and is filled in. It
cannot verify the rights are sufficient - that is for the project owner and
their counsel.
