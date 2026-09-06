# Can Chatterbox Turbo stream? No. Read this before saying otherwise.

Settled by reading the package source, not by inference: `chatterbox-tts`
**0.1.7**, `src/chatterbox/tts_turbo.py`.

## The evidence

`ChatterboxTurboTTS.generate()` ends:

```python
wav = wav.squeeze(0).detach().cpu().numpy()
watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
return torch.from_numpy(watermarked_wav).unsqueeze(0)
```

One return, one completed tensor. There is **no `yield` anywhere in the
package** — not in `tts_turbo.py`, not in `t3.py`, not in `s3gen/`. No
callback parameter, no generator, no `generate_stream`.

Three whole-sequence barriers, in order, each of which must finish before the
next begins:

1. `t3.inference_turbo` accumulates every speech token in a Python list and
   returns the finished sequence.
2. `s3gen.inference` vocodes that entire token sequence in one call.
3. `watermarker.apply_watermark` runs over the complete waveform.

Even if 1 and 2 were made incremental, 3 currently operates on the whole array.

## The distinction that must be kept

| | what happens | available? |
|---|---|---|
| **True model streaming** | generation → incremental audio frames → playback begins while generation continues | **No.** Not in this package. |
| **Post-generation streamed delivery** | generation → completed waveform → written to the socket in pieces, playable before the last piece lands | **Yes.** `/synthesise/stream`. |

`experiments/adapters/chatterbox_server_example.py` exposes the second and says
so in a response header, `X-Streaming-Kind: post-generation-delivery-only`, so
a client cannot mistake one for the other even without reading this file.

**What streamed delivery can and cannot buy.** It removes base64 encoding, JSON
assembly and the wait for the final byte. It is strictly additive on top of
model time and cannot reduce it. If generation takes 3 seconds, the listener
waits at least 3 seconds — streamed delivery only stops them waiting 3 seconds
*plus* the encode-and-transfer tail.

## The route to true streaming, if it is ever wanted

`s3gen/hifigan.py` takes a `cache_source` argument, which is the standard
machinery for chunked vocoding in CosyVoice-derived stacks. Turbo does not use
it that way and does not expose it. Making it incremental would mean forking
the package, and the watermarker would need to move to a per-chunk basis. That
is a project, not a configuration change, and nothing here has costed it.

Until then, FAM's lever on time-to-first-audio is **chunking the text** — which
is exactly what `fam_chunked_benchmark.py` did and what FAM's first-chunk rule
already produces.

## Other facts established from the source, worth having

* Sample rate is **24000 Hz** (`S3GEN_SR`), fixed.
* `from_pretrained(device="mps")` **silently falls back to CPU** if MPS is
  unavailable — it prints and continues. The benchmark re-reads `model.device`
  after loading and refuses to run rather than label a CPU run as MPS.
* Weights come from the HF repo `ResembleAI/chatterbox-turbo`, via
  `snapshot_download`, honouring `HF_TOKEN` if set. First load downloads.
* Default voice: `conds.pt` shipped in the checkout. `generate(text)` with no
  `audio_prompt_path` uses it, which is what the recovered benchmarks did, so
  there is no voice to select and no reference sample to supply.
* The package pins **`torch==2.6.0`** and requires Python **>=3.10**.
