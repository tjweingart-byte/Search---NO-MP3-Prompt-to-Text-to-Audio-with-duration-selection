# What the Chatterbox port does not know

`test_chatterbox.py` is twelve lines run on a Mac (`device="mps"`, and the
text itself says "running locally"). The Runpod RTX 4090 work was done in a
Jupyter environment that has not been recovered, so the adapter is built on
what that one local file proves plus explicit, listed assumptions.

This file exists so those assumptions are auditable rather than buried in
code. Anything here that turns out to be wrong invalidates a comparison, so
check it before trusting a Chatterbox number.

## Blocking — a wrong guess here makes the numbers meaningless

| Unknown | What the adapter assumes | Why it matters |
|---|---|---|
| **Is "Turbo" this model?** | plain `ChatterboxTTS.from_pretrained(device=...)`, no checkpoint argument | The local file names no variant. If Turbo is a different package, class or checkpoint, the adapter benchmarks the wrong model and the result is not about Turbo at all. |
| **`generate()` arguments** | `model.generate(text)` and nothing else | Chatterbox exposes voice-prompt and expressiveness controls. If the Runpod run passed any — a reference voice especially — output quality and speed both differ, and the local file cannot show it. |
| **One-shot or streaming?** | one call returns the whole waveform | The engine measures "first playable audio" as the end of one `generate`. If Chatterbox was used through a streaming API, first-audio is a different and much earlier event, and the pipeline stage would need restructuring rather than reconfiguring. |
| **Package and version** | `chatterbox-tts` from PyPI, unpinned | A guess from the import path. The Jupyter install command is the ground truth. |

## Material — affects reproducibility and cost, not correctness

| Unknown | Assumed | Note |
|---|---|---|
| CUDA / torch build | whatever `pip install torch` gives | A cu121 vs cu124 mismatch is a common cause of a silent CPU fallback; the adapter refuses unasked CPU, so this surfaces as a refusal rather than a bad number. |
| Weight cache location | default Hugging Face cache | On Runpod, a container-disk cache re-downloads gigabytes on every pod start. That is load time, not synthesis time, and the adapter already reports them separately. |
| Gated weights / `HF_TOKEN` | not required | Unverified. If loading needs auth, `from_pretrained` fails at load. |
| Pod image / Python version | none | Only matters for reproducing the environment exactly. |
| Existing server on the pod | none — the contract in `chatterbox_server_example.py` is this repo's design | If a server was already written in Jupyter, its shape almost certainly differs from this contract. |
| Runpod proxy URL shape | any HTTPS URL in `CHATTERBOX_ENDPOINT` | Runpod exposes ports through a proxy host; the adapter does not care, but the URL is not guessable. |

## Confirmed by the file, not assumed

* the model class and its `from_pretrained(device=...)` loading
* `model.generate(text)` returning a waveform
* the sample rate coming from `model.sr`
* that the local run was `mps` — a Mac, not the 4090

## Not in the file at all — derived here, and therefore new

No timing, no duration arithmetic, no realtime factor. Twelve lines that save
a WAV and print "Done" contain no measurement. Duration
(`frames / model.sr`), realtime factor (`audio / wall`), model-load time and
cold/warm state are all computed by this engine and have no manual counterpart
to be checked against.

## Assumptions the code makes visible rather than hiding

* **Mono.** Channel 0 is taken explicitly; the channel count is recorded on
  every trial. Flattening a stereo tensor would have played left then right at
  twice the length, silently.
* **Float waveform in [-1, 1].** Clamped before scaling, so an overshoot clips
  instead of wrapping into noise.
* **CPU is never chosen silently**, because Chatterbox on CPU is slower than
  realtime and would measure the machine rather than the model.
