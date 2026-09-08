# Reference recordings for the voice-identity bake-off

**Nothing goes in here that we do not have the right to clone.** This folder is
git-ignored: the recordings are somebody's voice, and they are not source code.

Put the three recordings in here **under whatever names they already have** -
nothing renames, moves or converts them - then:

    python tools/check_reference_audio.py experiments/references --adopt
    python tools/equalise_references.py experiments/references
    python tools/check_reference_audio.py experiments/references

That writes `sources.json`, mapping neutral ids to the real files, and creates
a rights template for each. The assignment is shuffled rather than
alphabetical, and the mapping file is git-ignored along with the audio.

**Neutral ids on purpose.** A filename like `magnetic.wav` would assert which
speaker is the magnetic one - a mapping nobody has verified - and would prime
whoever listens. Speaker names are worse: they identify the voice outright.
Nothing downstream ever sees a source filename except `KEY.json`, which is the
reveal.

    sources.json               reference_1 -> whichever file
    reference_1.rights.json    rights for that recording
    reference_2.rights.json
    reference_3.rights.json

The four qualities being listened for (magnetic, human, storyteller, modern
authority) are recorded in the experiment design, unattached to any file. The
order of the numbers means nothing.

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

Because the third one is untruncated, **length is not a free variable**.
References of different lengths feed the speaker embedding unequal amounts of
each voice while feeding the other two the same. Make them the same length.

## The spec

| | |
|---|---|
| duration | **12-15 seconds**. They need not match exactly - `tools/equalise_references.py` derives equal-length working copies without touching the originals |
| format | any file `librosa` can open - **wav, m4a, mp3, flac**. No conversion needed; Chatterbox calls `librosa.load` itself |
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

So: **have all three speakers read the same neutral, conversational passage**,
in the same unremarkable way. Then the thing that differs between the three
files is the speaker, which is what this experiment is choosing.

If instead each speaker performs a different character — one being magnetic,
one being a storyteller — the test measures performance as much as voice, and a
later delivery-optimisation pass cannot untangle them. This is also why the
files are numbered rather than named after the directions.

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
