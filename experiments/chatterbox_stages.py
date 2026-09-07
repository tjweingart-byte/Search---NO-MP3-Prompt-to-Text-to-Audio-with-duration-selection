"""Chatterbox Turbo, timed one stage at a time, and probed for chunkability.

Phase 1 answers two questions and builds nothing:

**Which stage owns time-to-first-audio?** `tts_turbo.generate()` runs four
things back to back and reports one number. If T3 dominates, incremental
vocoding buys little and streaming Chatterbox means streaming *token
generation* - a much larger project. If Flow or HiFiGAN dominate, the
`finalize` and `cache_source` parameters already present are most of the work.

**Can Flow and HiFiGAN actually consume partial input?** Both carry the
CosyVoice streaming protocol, but nobody has fed them a prefix and looked at
what came out. Two failure modes matter and are measured, not assumed:

* **Recomputation.** If Flow must be re-run over the whole prefix for each
  chunk, cost grows quadratically and the saving evaporates on long chunks.
  Both variants are run - full-prefix recompute and delta-only - and compared.
* **Discontinuity.** A chunked waveform that clicks at every seam is not
  shippable however fast it is. Seams are measured against the signal's own
  local roughness rather than eyeballed.

Nothing here modifies the installed package. It calls the same public stage
methods `generate()` calls, in the same order, with fences between them.
`chatterbox_impl.synchronize` is used on both sides of every stage, because a
CUDA timing without it measures kernel *queueing*.

**Watermarking is never skipped.** It is timed as its own stage, and its
per-chunk behaviour is measured with Perth's own detector. Whether per-chunk
watermarking is acceptable is Resemble AI's call; this only reports what
happens.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

#: `tts_turbo.generate` passes this to the flow decoder. Kept identical, or the
#: stage split would be timing a different computation from the real one.
N_CFM_TIMESTEPS = 2

#: `speech_tokens[speech_tokens < 6561]` in generate() - dropping out-of-
#: vocabulary tokens before the vocoder sees them.
OOV_THRESHOLD = 6561

#: Speech tokens per streaming chunk. XTTS uses 20 GPT tokens; Chatterbox's
#: token rate differs, so this is a knob the probe sweeps rather than a
#: constant anyone should trust.
DEFAULT_CHUNK_TOKENS = 25


@dataclass
class StageTiming:
    """One generate(), with a fence between every stage."""

    text: str
    words: int
    device: str
    text_prep: float = 0.0
    t3: float = 0.0
    flow: float = 0.0
    hift: float = 0.0
    watermark: float = 0.0
    tokens: int = 0
    mel_frames: int = 0
    samples: int = 0
    sample_rate: int = 0
    detail: dict = field(default_factory=dict)

    @property
    def total(self) -> float:
        return self.text_prep + self.t3 + self.flow + self.hift + self.watermark

    @property
    def seconds_per_token(self) -> Optional[float]:
        return self.t3 / self.tokens if self.tokens else None

    def shares(self) -> dict:
        total = self.total or 1.0
        return {name: getattr(self, name) / total for name in
                ("text_prep", "t3", "flow", "hift", "watermark")}

    def as_dict(self) -> dict:
        out = {
            "text_words": self.words, "device": self.device,
            "stage_text_prep": self.text_prep, "stage_t3": self.t3,
            "stage_flow": self.flow, "stage_hift": self.hift,
            "stage_watermark": self.watermark, "total_seconds": self.total,
            "tokens": self.tokens, "mel_frames": self.mel_frames,
            "samples": self.samples, "sample_rate": self.sample_rate,
            "seconds_per_token": self.seconds_per_token,
            "audio_seconds": (self.samples / self.sample_rate
                              if self.sample_rate else None),
        }
        out.update({f"share_{k}": v for k, v in self.shares().items()})
        out.update(self.detail)
        return out


def split_once(model, text: str, device: str) -> StageTiming:
    """One synthesis, with every stage on its own clock.

    Mirrors `ChatterboxTurboTTS.generate` step for step - `punc_norm`, tokenize,
    `t3.inference_turbo`, OOV drop plus trailing silence, `flow_inference`,
    `hift_inference`, `apply_watermark` - so the sum is comparable with the
    single number the benchmark already records.
    """
    import torch
    from chatterbox.models.s3gen.const import S3GEN_SIL
    from chatterbox.tts_turbo import punc_norm

    from experiments.adapters.chatterbox_impl import synchronize

    timing = StageTiming(text=text, words=len(text.split()), device=device)

    def fence():
        synchronize(device)
        return time.perf_counter()

    start = fence()
    normalised = punc_norm(text)
    text_tokens = model.tokenizer(normalised, return_tensors="pt",
                                  padding=True, truncation=True)
    text_tokens = text_tokens.input_ids.to(model.device)
    timing.text_prep = fence() - start

    start = fence()
    with torch.inference_mode():
        speech_tokens = model.t3.inference_turbo(
            t3_cond=model.conds.t3, text_tokens=text_tokens,
            temperature=0.8, top_k=1000, top_p=0.95, repetition_penalty=1.2)
    timing.t3 = fence() - start

    speech_tokens = speech_tokens[speech_tokens < OOV_THRESHOLD]
    speech_tokens = speech_tokens.to(model.device)
    silence = torch.tensor([S3GEN_SIL] * 3).long().to(model.device)
    speech_tokens = torch.cat([speech_tokens, silence])
    timing.tokens = int(speech_tokens.shape[-1])

    start = fence()
    with torch.inference_mode():
        mels = model.s3gen.flow_inference(
            speech_tokens, ref_dict=model.conds.gen,
            n_cfm_timesteps=N_CFM_TIMESTEPS, finalize=True)
    timing.flow = fence() - start
    timing.mel_frames = int(mels.shape[-1])

    start = fence()
    with torch.inference_mode():
        wavs, _source = model.s3gen.hift_inference(mels, None)
    timing.hift = fence() - start

    wav = wavs.squeeze(0).detach().cpu().numpy()
    timing.sample_rate = int(getattr(model, "sr", 0) or 0)
    timing.samples = int(wav.shape[-1])

    start = fence()
    model.watermarker.apply_watermark(wav, sample_rate=timing.sample_rate)
    timing.watermark = fence() - start

    return timing


def first_chunk_projection(timing: StageTiming, chunk_tokens: int) -> dict:
    """When the first chunk's tokens would be ready, if T3 yielded them.

    Derived from the measured mean per-token cost. It assumes the
    autoregressive loop costs roughly the same per step, which is why the
    probe runs three chunk lengths: if the assumption fails, seconds_per_token
    will not agree across them and this projection must be discarded rather
    than believed.
    """
    per_token = timing.seconds_per_token
    if not per_token:
        return {}
    tokens = min(chunk_tokens, timing.tokens)
    return {
        "projected_t3_to_first_chunk": per_token * tokens,
        "projected_chunk_tokens": tokens,
        "projection_basis": "mean per-token cost; check it agrees across lengths",
    }


@dataclass
class ChunkedResult:
    """One chunked run: what it cost, and whether it matches the one-shot audio."""

    mode: str                       # "recompute_prefix" | "delta_only"
    chunk_tokens: int
    ok: bool = True
    error: str = ""
    first_chunk_seconds: Optional[float] = None
    total_seconds: Optional[float] = None
    chunk_seconds: list = field(default_factory=list)
    samples: int = 0
    baseline_samples: int = 0
    max_abs_diff: Optional[float] = None
    seam_ratios: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "mode": self.mode, "chunk_tokens": self.chunk_tokens, "ok": self.ok,
            "error": self.error,
            "first_chunk_seconds": self.first_chunk_seconds,
            "total_seconds": self.total_seconds,
            "chunks": len(self.chunk_seconds),
            "samples": self.samples, "baseline_samples": self.baseline_samples,
            "max_abs_diff": self.max_abs_diff,
            "worst_seam_ratio": max(self.seam_ratios) if self.seam_ratios else None,
            "seam_ratios": self.seam_ratios,
        }


def seam_ratios(wav, boundaries: list) -> list:
    """How abrupt each join is, relative to the signal's own roughness.

    A click is a first-difference far larger than the neighbourhood's typical
    one. Reporting the ratio rather than the raw jump makes it comparable
    across chunks of different loudness. Around 1 is inaudible; large is a
    click.
    """
    import numpy as np

    diffs = np.abs(np.diff(wav))
    if diffs.size == 0:
        return []
    typical = float(np.median(diffs)) or 1e-9
    out = []
    for index in boundaries:
        # `diffs[k]` is wav[k+1] - wav[k], so the join *at* sample `index` -
        # where one piece ends and the next begins - is diffs[index - 1].
        # Reading diffs[index] is one sample late and steps straight over the
        # click, which is a false all-clear on the question this exists to
        # answer.
        step = index - 1
        if 0 <= step < diffs.size:
            out.append(float(diffs[step] / typical))
    return out


def chunked_flow_probe(model, speech_tokens, chunk_tokens: int, device: str,
                       mode: str, baseline_wav=None) -> ChunkedResult:
    """Feed Flow and HiFiGAN successive partials and see what comes back.

    `mode="recompute_prefix"` re-runs Flow over `tokens[:n]` each round, which
    is what a naive implementation would do and is quadratic.
    `mode="delta_only"` passes just the new tokens, carrying `cache_source`
    forward - the cheap version, and the one that has to be shown to work.

    Every chunk but the last uses `finalize=False`, which is what makes
    `flow.py` trim its pre-lookahead tail.
    """
    import numpy as np
    import torch

    from experiments.adapters.chatterbox_impl import synchronize

    result = ChunkedResult(mode=mode, chunk_tokens=chunk_tokens)
    total = int(speech_tokens.shape[-1])
    bounds = list(range(chunk_tokens, total, chunk_tokens)) + [total]

    pieces, cache, boundaries = [], None, []
    started = time.perf_counter()
    try:
        for position, end in enumerate(bounds):
            last = end == total
            if mode == "recompute_prefix":
                window = speech_tokens[:end]
            else:
                start_index = 0 if position == 0 else bounds[position - 1]
                window = speech_tokens[start_index:end]

            synchronize(device)
            with torch.inference_mode():
                mels = model.s3gen.flow_inference(
                    window, ref_dict=model.conds.gen,
                    n_cfm_timesteps=N_CFM_TIMESTEPS, finalize=last)
                wav, cache = model.s3gen.hift_inference(mels, cache)
            synchronize(device)

            audio = wav.squeeze(0).detach().cpu().numpy()
            if mode == "recompute_prefix":
                # Each round contains everything so far; keep only the new tail.
                kept = sum(p.shape[-1] for p in pieces)
                audio = audio[kept:]
            if pieces:
                boundaries.append(sum(p.shape[-1] for p in pieces))
            pieces.append(audio)
            if position == 0:
                result.first_chunk_seconds = time.perf_counter() - started
            result.chunk_seconds.append(time.perf_counter() - started)
    except Exception as exc:
        result.ok = False
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    result.total_seconds = time.perf_counter() - started
    joined = np.concatenate(pieces) if pieces else np.zeros(0)
    result.samples = int(joined.shape[-1])
    result.seam_ratios = seam_ratios(joined, boundaries)

    if baseline_wav is not None:
        result.baseline_samples = int(baseline_wav.shape[-1])
        length = min(result.samples, result.baseline_samples)
        if length:
            result.max_abs_diff = float(
                np.max(np.abs(joined[:length] - baseline_wav[:length])))
    return result


def watermark_probe(model, wav, sample_rate: int, pieces: int = 4) -> dict:
    """Does a per-chunk watermark still decode once the chunks are joined?

    The watermark is applied over a spectrogram, so chunking it has edge
    effects at the STFT boundaries. Perth ships its own detector
    (`get_watermark`), which turns "is per-chunk watermarking acceptable" from
    an opinion into a measurement.

    **This never removes or weakens the watermark.** It compares two ways of
    applying it and reports what the detector says about each.
    """
    import numpy as np

    out: dict = {"pieces": pieces}
    whole = model.watermarker.apply_watermark(wav, sample_rate=sample_rate)
    out["whole_detected"] = _detect(model, whole, sample_rate)

    splits = np.array_split(wav, pieces)
    marked = [model.watermarker.apply_watermark(p, sample_rate=sample_rate)
              for p in splits]
    joined = np.concatenate(marked)
    out["per_chunk_detected"] = _detect(model, joined, sample_rate)
    out["per_chunk_each_detected"] = [
        _detect(model, p, sample_rate) for p in marked]

    boundaries = list(np.cumsum([len(p) for p in marked[:-1]]))
    out["per_chunk_seam_ratios"] = seam_ratios(joined, boundaries)
    length = min(len(whole), len(joined))
    out["max_abs_diff_vs_whole"] = (
        float(np.max(np.abs(whole[:length] - joined[:length]))) if length else None)
    return out


def _detect(model, wav, sample_rate: int):
    """Perth's own decoder, or a recorded reason it could not run."""
    try:
        mark = model.watermarker.get_watermark(wav, sample_rate=sample_rate)
        import numpy as np

        return {"ok": True, "mean": float(np.mean(mark)),
                "length": int(np.size(mark))}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
