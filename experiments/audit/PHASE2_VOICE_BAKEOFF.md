# Phase 2 — blind FAM voice bake-off

**Preserved before any audio exists.** The passages, the roster, the scoring
axes and the blinding rules are fixed here so the result cannot be fitted to a
preference formed while listening.

No GPU rental. No latency benchmark. This runs on the Mac.

## Why this comes before more latency work

Four experiments have measured latency. None has answered the question that
started the voice thread: **Piper's problem was never speed** (P2). An engine
that reaches 0.4s and sounds worse than Piper has solved nothing, and we have
still never heard Chatterbox at all (P23).

Phase 1 settled that incremental Chatterbox is a model-level rewrite. Whether
that rewrite is worth weeks depends entirely on whether Chatterbox's voice is
one FAM wants — which costs nothing to find out.

## The roster, and why each is on it

| candidate | licence | why |
|---|---|---|
| **Chatterbox (base)** | MIT (P19, resolved) | exposes `exaggeration`, `cfg_weight`, `min_p` |
| **Chatterbox Turbo** | MIT (P19, resolved) | the fast variant, which **ignores** those controls (P17) |
| **Kokoro-82M** | Apache-2.0 code; **weights unverified** (P19) | the untried candidate named in `CLAUDE.md`; 82M, non-autoregressive |
| **Piper** | GPL-3.0-or-later | the incumbent, entered **unnamed** as a blind anchor |

Base against Turbo is the comparison Phase 0 surfaced and nobody planned: the
faster variant is the one with fewer expressive knobs, and FAM needs both low
TTFA *and* an expressive premium voice. If base sounds materially better, that
reframes the whole latency question.

Piper is in the test blind on purpose. It is the thing to beat, and a candidate
that cannot beat it in a blind comparison has settled the matter.

### Excluded, with reasons

| candidate | reason |
|---|---|
| **XTTS-v2** | weights are CPML, non-commercial and TOS-gated (P18). The only engine with true incremental synthesis, and unusable. |
| **F5-TTS** | code is MIT, but its own README states the pre-trained models are **CC-BY-NC** "due to the training data Emilia". Same blocker. |
| **Parler-TTS** | no readable licence in the published distribution, and no incremental interface found. Not supportable on the audit's evidence. |

That is why the roster is four and not six: the audit supported two additions
beyond the three required, and both failed on licensing.

## The passages

Three, in `experiments/passages/fam_voice_passages.json`, each 57-63 words —
matched to the real FAM chunk range of 25-59 words so the test hears what the
product would actually send. Written in FAM's register per `CLAUDE.md`.

**1. Intimate / curious storytelling** — an old Babylonian lullaby that turns
out to be a threat. Small dynamic range, a turn that needs timing, a dry
closing line. *Does it lean in, or announce? Does the last line land or get
flattened?*

**2. Energetic explanatory narration** — heat pumps moving heat rather than
making it. A colon, an em-dash, a three-verb run, and an emphasis that must
fall on "move it". *Does the run accelerate or plod?*

**3. Authoritative / news-style** — market movement with proper nouns,
decimals, a large number in words, percentages, and a mid-sentence turn.
*Are "zero point four percent" and "three point eight" read naturally? Does
"Nasdaq" survive? Does "Beneath that flatness" carry a turn?*

**The numbers in passage 3 are synthetic**, written to exercise number prosody.
The JSON marks them `synthetic_numbers: true`. They are not market data and
must never be quoted as such.

## How the test is kept honest

Three properties, enforced in code and pinned by tests rather than trusted:

**Blind.** Outputs are relabelled Voice A/B/C/D with a **separate randomisation
per passage**, so Voice A on one passage is not Voice A on another and nobody
learns the ordering. A test asserts the arrangements differ across passages,
that the first-listed candidate is not reliably Voice A, and that neither the
player page nor the scorecard contains any engine name.

**Loudness-matched.** Every clip is normalised to -23 LUFS (pyloudnorm where
available, RMS otherwise) before labelling. Louder reliably wins blind tests
for the wrong reason. A test asserts a 0.05-gain and a 0.9-gain tone come out
within 15% of the same RMS.

**Same words, first take.** Every candidate speaks the identical passage.
Nothing is re-prompted, re-rolled or hand-picked; the first generation is the
one scored.

The mapping lives in one file, `KEY.json`, which is not opened until scoring is
finished. The player page carries that warning at the top.

## Scoring

Eight axes, 1-5, 3 meaning "fine, unremarkable":

| axis | the question asked on the sheet |
|---|---|
| naturalness | Could this be a person? Or is something off? |
| warmth | Talking to you, or reading at you? |
| intrigue | Does it make you want to keep listening? |
| authority | Would you believe a fact from this voice? |
| expressiveness | Does emphasis land on the right words? Any range? |
| pacing | Prosody, phrasing, breaths, and the handling of numbers. |
| artificial | How artificial does it sound? **(1 = very, 5 = not at all)** |
| **is_fam** | **Does this feel like FAM?** |

`artificial` is reversed deliberately so every row reads "higher is better" and
a column can be added up without a sign error.

`is_fam` is the one that decides. The other seven explain it.

Per passage: a best-on-this-passage line and free notes. At the end: overall
winner, would-you-ship-it, and **better than Piper?** — asked while Piper is
still unnamed.

## Running it

On the Mac, with the Chatterbox environment already working:

    pip install kokoro                       # the only new dependency
    python tools/voice_bakeoff.py --device mps --out experiments/results/bakeoff

It prints its roster first, saying which candidates can run and why any cannot.
**A missing candidate weakens the test rather than failing it** — but the
comparison then cannot speak for that engine, and the tool says so.

Then:

1. Open `experiments/results/bakeoff/listen.html` in a browser. No network
    needed; nothing is loaded from a CDN.
2. Score in `scorecard.md` while listening. One passage at a time, every voice,
    first impressions before replays.
3. **Only then** open `KEY.json`.

Expect roughly 12 generations (4 candidates x 3 passages). On MPS the
Chatterbox variants run near real time, so budget ten to fifteen minutes of
generation for about three and a half minutes of audio.

## What the result decides

* **A Chatterbox variant wins** → its voice justifies the latency work, and the
  Phase 1 verdict (model-level rewrite) becomes a cost worth weighing rather
  than an argument to stop.
* **Kokoro wins** → measure its TTFA next. Architecturally it chunks per
  segment like everything else, but at 82M and non-autoregressive it may land
  under a second anyway — and its weight licence then has to be verified before
  anything else (P19).
* **Piper wins, or nothing clearly beats it** → the voice search has not found
  a replacement yet, and the honest conclusion is to widen the search rather
  than to fork anything.

## Constraints held

No production code modified. No GPU rented. No latency benchmark started. The
roster, passages and scoring were fixed before a single clip existed.
