# Chatterbox: what is now known, and what is not

The Runpod code was recovered. `test_turbo.py` and `fam_chunked_benchmark.py`
are the source of truth and the adapter is ported from them. This file used to
list guesses; most are now answered, and the few that remain are below.

## Answered by the recovered code

| Question | Answer |
|---|---|
| Which model is "Turbo"? | `from chatterbox.tts_turbo import ChatterboxTurboTTS` - its own module and class, **not** a checkpoint of the base model |
| Device | `cuda`, explicitly, on an RTX 4090 |
| `generate()` arguments | none - `model.generate(text)` and nothing else |
| One-shot or streaming? | **one-shot.** `generate` returns the whole waveform; latency is managed by chunking the *text*, not by streaming the audio |
| Timing | `torch.cuda.synchronize()` on both sides of `perf_counter`, around `generate` alone |
| Autograd | `with torch.inference_mode():` in the chunked run; `test_turbo.py` omits it |
| Warmup | one throwaway `generate("This is a warmup.")`, then a fence - chunked run only |
| Duration | `wav.shape[-1] / model.sr` |
| Realtime factor | `duration / gen_time` |
| Device-to-host copy | `wav.cpu()` **after** the clock stops |
| GPU rate | `GPU_RATE = 0.75` dollars/hour; cost = `generation_time / 3600 * rate` |
| Chunk joining | 120 ms silence between chunks, none after the last |
| Headline metric | "First chunk ready in" - which is FAM's time to first audio |
| Output | `torchaudio.save(...)` to `/workspace/*.wav` |

The two files differ from each other on purpose and both are reproduced:
`SINGLE` (one generate, cold, no inference_mode) and `CHUNKED` (warmed,
inference_mode, per-chunk timing).

## Still unknown

| Unknown | Impact |
|---|---|
| **Package version** | The install command from the Jupyter environment was not recovered. `chatterbox-tts` is unpinned, so a future version could change `tts_turbo` under us. |
| **Measured values** | No numbers were recovered - only the code. There is still nothing to validate a ported run against; the first real run establishes the baseline rather than reproducing one. |
| **Weight cache location** | On a fresh pod this decides whether `from_pretrained` re-downloads gigabytes. It is load time, reported separately, so it cannot contaminate a synthesis measurement. |
| **CUDA / torch build** | Unknown. A mismatch usually shows up as a CPU fallback, which the adapter refuses rather than silently timing. |
| **Gated weights / `HF_TOKEN`** | Unverified. If loading needs auth, `from_pretrained` fails at load, loudly. |
| **Any server already written on the pod** | None was recovered. `chatterbox_server_example.py` is this repo's design; if one exists, prefer it and adapt the adapter to it. |

## Assumptions the code still makes, made visible

* **Mono.** Channel 0 taken explicitly; the channel count is recorded on every
  trial. `silence = torch.zeros(1, ...)` in the recovered code agrees.
* **Float waveform in [-1, 1]**, clamped before scaling so an overshoot clips
  rather than wrapping into noise.
* **PCM, not WAV.** The benchmarks save files; FAM streams raw PCM and writes
  none. The samples are identical either way.
* **CPU is never chosen silently**, because it is slower than realtime and
  would measure the machine rather than the model.
