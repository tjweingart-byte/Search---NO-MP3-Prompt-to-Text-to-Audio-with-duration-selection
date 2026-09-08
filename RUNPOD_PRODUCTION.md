# The real-world validation gate

Everything about FAM's voice up to this point has been measured in pieces: a
bakeoff that chose `reference_3`, a stage split that found where the time goes,
a Phase 6 experiment that proved Claude and the voice can run at once. None of
those was FAM. They were harnesses.

This runs **the production server** — the real `app.py`, the real pipeline, the
real prompt, the real engine — on a real GPU, and asks it real questions. It is
the first time anyone hears what the product actually sounds like end to end.

**Nothing in this repository starts, stops, resizes or pays for a pod.** Every
step below is a human action, taken deliberately, on hardware billed by the
minute.

---

## What is being tested, and what is not

Tested here:

* Does production select Chatterbox, or does it quietly fall back to Piper?
* Does the credential work, from this process, on this machine?
* How long until the first byte reaches a client — the one-sentence spec?
* Does synthesis really begin before Claude finishes, on real hardware?
* Does a three-minute episode come in under its ceiling?
* **How does it sound.** The numbers cannot answer this and do not try to.

Not tested here, deliberately: the browser, the player, the browse surfaces,
deployment. Those are unchanged by this work and a pod is a poor place to
judge them.

## One divergence to hold in mind

The reference-voice bakeoff and the 4090 streaming runs used **Chatterbox
Base** — `from chatterbox.tts import ChatterboxTTS` — which is what
`tts.py` imports, so the voice you hear here is the voice that was chosen.

The *stage-split* work used **Chatterbox Turbo**, a different module and class.
Its per-stage timings describe Turbo, not this. If a number from that work is
quoted against this run, say which model it came from.

---

## 1. On the Mac: pack the bundle

    python tools/pack_for_pod.py
    scp fam-pod.tar.gz root@<pod>:/workspace/

`pack_for_pod.py` writes `git archive` of the tracked tree at HEAD plus two
files that are deliberately not in the repository: the reference recording, and
a **minimal** rights record for it.

The record is *regenerated*, not copied. The original names the person who
recorded it; the pod gets the three answers `ChatterboxEngine` gates on and
nothing else. Identity does not leave the Mac.

The rights gate runs before packing. A recording whose record does not clear
consent, commercial use and synthetic voice is refused here — a gate the packer
can step around is not a gate.

It also refuses a dirty working tree, because `git archive` ships commits and
the uncommitted changes would silently not be in the bundle. `--allow-dirty`
if you mean it.

**No credentials, ever.** No git remote, no token, no SSH key goes up.
`ANTHROPIC_API_KEY` is exported into the pod shell by hand and is never written
into the bundle.

## 2. On the pod: extract and run

    cd /workspace && tar xzf fam-pod.tar.gz && cd FAM
    cat POD.txt
    export ANTHROPIC_API_KEY='...'
    bash tools/pod_production_test.sh

Seven gates, cheapest first, so every preventable failure happens in seconds
rather than after ten minutes of installing and a 4 GB download:

| Gate | Costs | Fails when |
| --- | --- | --- |
| 1 the card | nothing | no GPU. Chatterbox refuses CPU; an episode would starve mid-sentence |
| 2 the credential is exported | nothing | `ANTHROPIC_API_KEY` is not in the shell (its value is never printed) |
| 3 the bundle | nothing | no `POD.txt`, no recording, or no rights record beside it |
| 4 the tests | ~30s | the suite fails here — the numbers would be about a broken build |
| 5 dependencies | ~10 min cold | `requirements-chatterbox.txt` will not install |
| 6 **production selects Chatterbox** | weights load | `engine_report()` says `interim: true` |
| 7 the credential is accepted | a few cents | `warm_up()` finishes with no resident model |

Gate 6 is the one that matters most, and it is the reason this is a script and
not a list of commands. *"Chatterbox is installed"* is not *"this server will
speak with Chatterbox"*. A pod that fell back to Piper would produce a
complete, plausible, entirely worthless set of numbers about the wrong engine —
the exact failure shape PROBLEMS.md §52 is about.

Gate 7 is the same discipline applied to the key: `warm_up()` swallows
exceptions by design so a bad voice cannot stop the server booting, which means
"warm-up ran" is not "the model is resident". It asserts residency.

## 3. What the run produces

Four episodes: two questions × two pipelines, `legacy` first so `phase6` is
read against a baseline taken on the same card in the same session rather than
a remembered one. Caching is off — a cached script would report a Claude time
of nearly zero and the comparison would be between two replays.

    pod-results/<stamp>/
      server.log                  every request's complete timeline
      legacy_known/episode.wav    LISTEN TO THESE
      legacy_known/episode.json   listener view, headers and marks
      phase6_known/…
      legacy_fresh/…  phase6_fresh/…

`tools/pod_episode.py` reports each episode three ways, because each answers
something the others cannot:

* **listener** — measured from the client. First byte, last byte, seconds of
  audio. This is what someone holding the app gets, and the only view that
  includes the network.
* **preroll** — the `X-*` response headers: the server's view up to the moment
  it decided there was enough audio to answer. It stops there because a
  `StreamingResponse` fixes its headers before the body runs. That is a
  property of streaming, so the log is read rather than the header made to lie.
* **episode** — the complete timeline, read back from the log: `claude_complete`,
  `speaking_complete`, the backlog when the script finished, and
  `claude_decoupled` — true only when synthesis began strictly before Claude
  finished writing. That is Phase 6's entire claim, and it is checked rather
  than assumed. An episode whose timeline could not be read is reported as
  **unverified**, never as a pass.

## 4. Getting it back

    scp -r root@<pod>:/workspace/FAM/pod-results/<stamp> ./pod-results/

**Copy before terminating the pod.** Then terminate it. Committing happens on
the Mac, where the credentials already are.

## 5. The question the numbers cannot answer

Play `phase6_known/episode.wav` and `legacy_known/episode.wav`.

The scripts have been rewritten twice — around what makes a briefing worth
hearing, and again to stop the endings teasing the next episode — and **neither
rewrite has ever been heard**. Both were written without an API key. This is
the first opportunity to judge them, and it is the more important half of this
run: a fast episode that sounds wrong is not the product.

## When something goes wrong

    python tools/diagnose_chatterbox.py

Two packages on this chain hide their real import errors behind a try/except
and then fail much later naming neither the module nor the version. `perth`
turns a real `ImportError` into `PerthImplicitWatermarker = None`, and you see
`TypeError: 'NoneType' object is not callable` after a 4 GB download.
`transformers` raises `Could not import module 'LlamaModel'` and discards the
cause. The tool re-runs both imports unmasked and prescribes the smallest fix.

Every pin in `requirements-chatterbox.txt` has a comment saying which silent
failure it prevents. Read the comment before moving the pin.

Do not stub the watermarker out to get past it. It runs inside every
`generate()`, so removing it changes the audio, changes the latency this run
exists to measure, and means FAM is shipping unwatermarked synthetic speech.
