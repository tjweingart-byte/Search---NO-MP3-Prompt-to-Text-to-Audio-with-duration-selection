#!/usr/bin/env python3
"""Phase 1: which Chatterbox stage owns time-to-first-audio, and can it chunk?

Builds nothing. Answers whether an incremental Chatterbox is a contained
engineering project or a model-level rewrite, by measuring instead of arguing:

1. **Stage split.** T3, Flow, HiFiGAN and the watermark, each fenced, on real
   FAM chunks across the three length buckets.
2. **Chunkability.** Flow and HiFiGAN fed successive partials, both
   recompute-prefix and delta-only, compared against the one-shot waveform for
   sample-level agreement and for clicks at the joins.
3. **Watermark.** Applied per chunk and detected with Perth's own decoder.
   Never skipped, never weakened.

    python tools/chatterbox_stage_split.py --device cuda \\
        --out experiments/results/chatterbox_stage_split.json

Cold model load and warmup happen before any measurement and are excluded.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments import chatterbox_stages as stages          # noqa: E402
from experiments.adapters import chatterbox_impl as impl     # noqa: E402

DEFAULT_CHUNKS = "experiments/chunks/first_chunks.json"
BUCKETS = ("short", "medium", "long")


def pick(chunks: list[dict], per_bucket: int) -> list[dict]:
    """A few real chunks from each bucket - length is the variable that matters."""
    seen, out = {}, []
    for chunk in chunks:
        bucket = chunk.get("bucket")
        if seen.get(bucket, 0) < per_bucket:
            seen[bucket] = seen.get(bucket, 0) + 1
            out.append(chunk)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", help="cuda / mps; never cpu unless named")
    parser.add_argument("--chunks", default=DEFAULT_CHUNKS)
    parser.add_argument("--per-bucket", type=int, default=2, dest="per_bucket")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--chunk-tokens", type=int, dest="chunk_tokens",
                        default=stages.DEFAULT_CHUNK_TOKENS)
    parser.add_argument("--out")
    args = parser.parse_args()

    resolved, explicit = impl.resolve_device(args.device)
    devices = impl.available_devices()
    if not devices.get(resolved):
        raise SystemExit(f"device {resolved!r} is not available here "
                         f"(have: {', '.join(k for k, v in devices.items() if v)})")
    if resolved == "cpu" and not explicit:
        raise SystemExit("refusing CPU silently; name it if you mean it")

    corpus = pathlib.Path(args.chunks)
    if not corpus.exists():
        raise SystemExit(f"no chunk corpus at {corpus}")
    chunks = pick(json.loads(corpus.read_text(encoding="utf-8"))["chunks"],
                  args.per_bucket)
    print(f"\nLOCAL {resolved.upper()} / DEVELOPMENT BENCHMARK - not production "
          f"latency\n{len(chunks)} chunks x {args.trials} trials\n")

    model, load_seconds = impl.load_model(resolved)
    actual = str(getattr(model, "device", resolved))
    if actual.split(":")[0] != resolved:
        raise SystemExit(f"asked for {resolved!r}, model loaded on {actual!r}")
    impl.warm_up(model, resolved)
    print(f"cold start (excluded): model load {load_seconds:.1f}s\n")

    rows, chunked_rows, watermark_rows = [], [], []
    for chunk in chunks:
        for trial in range(1, args.trials + 1):
            timing = stages.split_once(model, chunk["text"], resolved)
            row = {"bucket": chunk["bucket"], "trial": trial,
                   "source": chunk.get("source"), **timing.as_dict()}
            row.update(stages.first_chunk_projection(timing, args.chunk_tokens))
            rows.append(row)
            share = timing.shares()
            print(f"  {chunk['bucket']:<7}{timing.words:>3}w t{trial}  "
                  f"total {timing.total:6.3f}s   "
                  f"t3 {timing.t3:6.3f}s ({share['t3']*100:4.1f}%)  "
                  f"flow {timing.flow:6.3f}s ({share['flow']*100:4.1f}%)  "
                  f"hift {timing.hift:6.3f}s ({share['hift']*100:4.1f}%)  "
                  f"wm {timing.watermark:6.3f}s ({share['watermark']*100:4.1f}%)")

        # Chunkability and watermarking, once per chunk rather than per trial.
        import torch
        from chatterbox.models.s3gen.const import S3GEN_SIL
        from chatterbox.tts_turbo import punc_norm

        with torch.inference_mode():
            ids = model.tokenizer(punc_norm(chunk["text"]), return_tensors="pt",
                                  padding=True, truncation=True).input_ids.to(model.device)
            tokens = model.t3.inference_turbo(
                t3_cond=model.conds.t3, text_tokens=ids, temperature=0.8,
                top_k=1000, top_p=0.95, repetition_penalty=1.2)
            tokens = tokens[tokens < stages.OOV_THRESHOLD].to(model.device)
            tokens = torch.cat([tokens,
                                torch.tensor([S3GEN_SIL] * 3).long().to(model.device)])
            mels = model.s3gen.flow_inference(
                tokens, ref_dict=model.conds.gen,
                n_cfm_timesteps=stages.N_CFM_TIMESTEPS, finalize=True)
            baseline, _ = model.s3gen.hift_inference(mels, None)
        baseline_wav = baseline.squeeze(0).detach().cpu().numpy()

        for mode in ("recompute_prefix", "delta_only"):
            probe = stages.chunked_flow_probe(
                model, tokens, args.chunk_tokens, resolved, mode, baseline_wav)
            chunked_rows.append({"bucket": chunk["bucket"],
                                 "source": chunk.get("source"), **probe.as_dict()})
            worst = probe.as_dict()["worst_seam_ratio"]
            print(f"    chunked/{mode:<17} "
                  + (f"first {probe.first_chunk_seconds:.3f}s  "
                     f"total {probe.total_seconds:.3f}s  "
                     f"worst seam {worst:.1f}x  "
                     f"max diff {probe.max_abs_diff}"
                     if probe.ok else f"FAILED: {probe.error}"))

        marks = stages.watermark_probe(
            model, baseline_wav, int(getattr(model, "sr", 24000)))
        watermark_rows.append({"bucket": chunk["bucket"], **marks})
        print(f"    watermark  whole {marks['whole_detected']}  "
              f"per-chunk {marks['per_chunk_detected']}")

    summary = summarise(rows, chunked_rows)
    report(summary)

    if args.out:
        path = pathlib.Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "label": f"LOCAL {resolved.upper()} / DEVELOPMENT BENCHMARK",
            "is_production_latency": False, "device": resolved,
            "cold_start": {"load_seconds": load_seconds,
                           "excluded_from_trials": True},
            "chunk_tokens": args.chunk_tokens,
            "summary": summary, "stages": rows,
            "chunked": chunked_rows, "watermark": watermark_rows,
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {path}")
    return 0


def summarise(rows: list[dict], chunked: list[dict]) -> dict:
    out: dict = {"buckets": {}}
    for bucket in BUCKETS:
        mine = [r for r in rows if r["bucket"] == bucket]
        if not mine:
            continue
        def med(key):
            values = [r[key] for r in mine if r.get(key) is not None]
            return statistics.median(values) if values else None
        out["buckets"][bucket] = {
            "n": len(mine), "words": med("text_words"),
            "total": med("total_seconds"), "t3": med("stage_t3"),
            "flow": med("stage_flow"), "hift": med("stage_hift"),
            "watermark": med("stage_watermark"), "tokens": med("tokens"),
            "seconds_per_token": med("seconds_per_token"),
            "projected_t3_to_first_chunk": med("projected_t3_to_first_chunk"),
        }
    per_token = [r["seconds_per_token"] for r in rows if r.get("seconds_per_token")]
    if per_token:
        out["seconds_per_token_spread"] = {
            "min": min(per_token), "max": max(per_token),
            "median": statistics.median(per_token),
            # If T3 is linear per token this ratio is ~1. Far from 1 means the
            # projection below must not be believed.
            "max_over_min": max(per_token) / min(per_token),
        }
    for mode in ("recompute_prefix", "delta_only"):
        mine = [c for c in chunked if c["mode"] == mode and c["ok"]]
        if mine:
            seams = [c["worst_seam_ratio"] for c in mine
                     if c.get("worst_seam_ratio") is not None]
            out[mode] = {
                "runs": len(mine),
                "first_chunk_median": statistics.median(
                    [c["first_chunk_seconds"] for c in mine
                     if c.get("first_chunk_seconds") is not None] or [0]),
                "total_median": statistics.median(
                    [c["total_seconds"] for c in mine
                     if c.get("total_seconds") is not None] or [0]),
                "worst_seam_ratio": max(seams) if seams else None,
            }
        failures = [c for c in chunked if c["mode"] == mode and not c["ok"]]
        if failures:
            out.setdefault(mode, {})["failed"] = len(failures)
            out[mode]["first_error"] = failures[0]["error"]
    return out


def report(summary: dict) -> None:
    print(f"\n{'bucket':<8}{'words':>6}{'total':>9}{'T3':>9}{'Flow':>9}"
          f"{'HiFi':>9}{'wmark':>9}{'tokens':>8}{'T3 share':>10}")
    for bucket, stat in summary["buckets"].items():
        total = stat["total"] or 1.0
        print(f"{bucket:<8}{stat['words'] or 0:>6.0f}{stat['total']:>8.3f}s"
              f"{stat['t3']:>8.3f}s{stat['flow']:>8.3f}s{stat['hift']:>8.3f}s"
              f"{stat['watermark']:>8.3f}s{stat['tokens'] or 0:>8.0f}"
              f"{stat['t3'] / total * 100:>9.1f}%")

    spread = summary.get("seconds_per_token_spread")
    if spread:
        print(f"\nper-token cost  {spread['median'] * 1000:.2f} ms "
              f"(min {spread['min'] * 1000:.2f}, max {spread['max'] * 1000:.2f}, "
              f"max/min {spread['max_over_min']:.2f})")
        if spread["max_over_min"] > 1.5:
            print("  T3 is NOT linear per token across lengths; the projected "
                  "time-to-first-chunk below is unsafe and must be discarded.")

    for mode in ("recompute_prefix", "delta_only"):
        stat = summary.get(mode)
        if not stat:
            continue
        if stat.get("failed"):
            print(f"\n{mode}: {stat['failed']} run(s) FAILED - {stat['first_error']}")
            continue
        print(f"\n{mode}: first chunk {stat['first_chunk_median']:.3f}s, "
              f"total {stat['total_median']:.3f}s, "
              f"worst seam {stat['worst_seam_ratio']}")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
