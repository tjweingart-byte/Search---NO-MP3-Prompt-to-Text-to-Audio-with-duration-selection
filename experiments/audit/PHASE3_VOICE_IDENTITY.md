# Phase 3 — FAM voice identity, Chatterbox Base only

**Preserved before any reference audio exists.** Engine selection is finished:
Phase 2 chose Chatterbox Base on voice quality, especially how human it sounded.
This experiment chooses the *voice*, and deliberately does not re-open the
engine question. No other engine appears.

**Three candidates, blind as A/B/C**, from the three speakers recorded. The
recordings keep their own filenames on disk; `sources.json` maps them to
`reference_1`, `reference_2`, `reference_3`, and that mapping is git-ignored.
Nothing downstream ever writes a source filename - not the listening page, not
the choice sheet, not the generated clips, and not `progress.log`, which
persists while the judging happens. Only `KEY.json` names them, and that is the
reveal.

The ids are neutral rather than named after the four directions. A filename asserting which speaker is "the magnetic one" would
claim a mapping nobody has verified and would prime the listener toward hearing
it, which is the opposite of a blind test.

No GPU rental. No latency work. This runs on the Mac.

## What Chatterbox Base actually needs, read from the source

`tts.py:182-206`. Not from documentation, and not from memory.

```python
s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)      # 24 kHz, mono
ref_16k_wav = librosa.resample(s3gen_ref_wav, S3GEN_SR, S3_SR) # 16 kHz
s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]              # first 10 s
... s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], ...)   # first 6 s
ve_embed = self.ve.embeds_from_wavs([ref_16k_wav], ...)        # WHOLE file
```

| conditioning | how much of the reference it consumes |
|---|---|
| s3gen reference (`embed_ref`) | **first 10 s** — `DEC_COND_LEN = 10 * 24000` |
| T3 prompt speech tokens | **first 6 s** — `ENC_COND_LEN = 6 * 16000`, capped at 150 tokens |
| voice-encoder speaker embedding | **the entire file**, untruncated |

**No transcript is required.** Nothing in the code reads one.

**Format is whatever `librosa` opens** — WAV, FLAC, MP3, M4A — at any sample
rate, stereo downmixed automatically. WAV at 44.1/48 kHz is the safe choice.

**Because the speaker embedding is untruncated, length is not a free
variable.** References of different lengths feed that embedding unequal amounts
while feeding the other two the same. All three must match.

## Is this a clean test of speaker identity? No — and here is exactly why

Asked directly, so answered directly. Holding every generation parameter equal
while swapping the reference is the *best available* test, and it is not a
clean one. Four confounds, three of which can be reduced and one that cannot.

**1. The reference carries delivery, not only timbre. Unavoidable.**
`cond_prompt_speech_tokens` are *speech tokens* taken from the first six
seconds — they encode how the person was speaking: rate, contour, energy. A
reference recorded slowly and warmly biases the output slow and warm.
**Speaker identity and delivery are entangled by construction in this model.**

This bites hardest because the qualities being sought - magnetic, human,
storyteller, modern authority - are descriptions of *performance* as much as of
voice. If each speaker performs a different character, the test measures the
performance and a later delivery-tuning pass cannot untangle it. It is also why
the references are numbered rather than named after the directions.

*Mitigation, and it is the most important instruction in this document:* have
**all three speakers read the same neutral, conversational passage in the same
unremarkable way**. Then what differs between the three files is the speaker,
which is what is being chosen. The suggested text is in
`experiments/references/README.md`.

**2. Recording conditions are cloned too. Reducible.**
Room, microphone, EQ, compression and noise all pass into the embedding and the
s3gen reference. A better-recorded voice can win for reasons that have nothing
to do with the voice. *Mitigation:* record all three in similar conditions,
unprocessed, and the checker reports each one's noise floor and dynamic range
so a mismatched recording is visible before it is used.

**3. Length asymmetry. Removable.** Covered above. *Mitigation:* the checker
fails the set if the references differ by more than 2 seconds.

**4. Sampling is stochastic. Removable, and removed.**
`temperature=0.8` with `min_p` and `top_p` means identical settings do not give
identical output. Two draws of the same identity differ, so without control a
voice could win on a luckier sample. *Mitigation:* the seed is fixed **per
passage and shared by every identity** — same passage, same random stream,
three references. A test asserts it.

**What this experiment therefore measures**, stated honestly: *the voice that
results from a given reference recording under fixed settings.* That is the
unit that actually ships, so it is the right thing to choose on — but it is not
a pure timbre comparison, and a winner should be understood as "this reference
produces the FAM voice", not "this person's vocal cords are the FAM voice".

## The qualities being listened for

Recorded here because they are what the listening is *for* - and kept
deliberately unattached to any reference file:

| | |
|---|---|
| **Magnetic** | intimate, intriguing, slightly restrained, sophisticated; pulls the listener toward it rather than demanding attention |
| **Human** | exceptionally conversational and natural; a smart person beside you, not a narrator, announcer or assistant |
| **Storyteller** | warm, emotionally intelligent, dynamic; timing and movement without theatricality |
| **Modern Authority** | confident, composed, intelligent, sophisticated, still human; authoritative without becoming a news anchor |

Three speakers against four qualities is not a mismatch to fix. The question is
which recorded speaker best carries FAM, not which speaker fills which slot.

## Controls held

| | |
|---|---|
| engine | Chatterbox Base only, one model load, `chatterbox.tts.ChatterboxTTS` |
| settings | `exaggeration 0.5, cfg_weight 0.5, temperature 0.8, repetition_penalty 1.2, min_p 0.05, top_p 1.0` — the model's own defaults, pinned by a test |
| seed | fixed per passage, shared across all three identities |
| passages | the same three from Phase 2, unchanged, so this stays anchored to what was already heard |
| blinding | Voice A/B/C, randomised **separately per passage** |
| loudness | every clip normalised to -23 LUFS before labelling |
| first takes | a clip already on disk is never regenerated |
| checkpointing | every clip written as it is made; a kill costs only the clip in flight |

**Delivery is deliberately not tuned here.** `exaggeration`, `cfg_weight`,
`min_p` and pacing stay at defaults precisely so that no voice wins on
settings. Tuning them is the next experiment, after a speaker is chosen.

## What you record and score

Four questions, not forty — the eight-axis scorecard is not used here:

* best voice for each of the three passages
* overall favourite
* would you ship this as FAM's voice? **Yes / Maybe / No**
* optional freeform note

## The rights gate

**The runner refuses to synthesise a voice that has no rights record.** Every
reference needs a `<name>.rights.json` beside it, with every field answered:
`source`, `speaker`, `consent`, `commercial_use`, `synthetic_voice_cleared`,
`notes`. A `false` in any of the three clearance fields blocks that voice.

`synthetic_voice_cleared` is separate from `commercial_use` on purpose.
Permission to *use a recording* is not permission to *synthesise a voice from
it*, and a standard voiceover agreement often covers the first while saying
nothing about the second.

`experiments/references/` is git-ignored. Someone's voice is not source code.

**None of this is legal advice.** The checker verifies a record exists and is
filled in. Whether the rights are sufficient is for the project owner and their
counsel.

## Sequence

1. Obtain three reference recordings and put them in
   `experiments/references/` under any names.
2. `python tools/check_reference_audio.py experiments/references --adopt`
3. Fill in the three `.rights.json` files.
4. `python tools/check_reference_audio.py experiments/references`
5. `python tools/voice_identity_bakeoff.py --device mps --out experiments/results/identity`
6. Open `listen.html`, record choices in `choices.md`, then open `KEY.json`.

**Step 1 is yours.** No reference audio is sourced, invented or downloaded by
this repository.

## Constraints held

No production code modified. No GPU rented. No T3 rewrite, no Kokoro latency
benchmark, no production integration. The design, controls and confounds were
written down before any audio existed.
