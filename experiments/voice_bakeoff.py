"""Blind A/B/C listening test across TTS candidates on FAM's own passages.

Latency has had four experiments. This one asks the question latency cannot:
does it sound like FAM? An engine that reaches 0.4s and sounds worse than
Piper has solved nothing, and Piper's voice is why this whole line of work
exists.

Three things make a listening test worth trusting, and all three are enforced
here rather than left to good intentions:

**Blind.** Outputs are relabelled Voice A, B, C... with a per-passage
randomisation, so the letters do not carry across passages and a listener
cannot learn "C is always the fast one". The mapping is written to a single
key file that the listener does not open until scoring is finished.

**Loudness-matched.** Louder reliably sounds better in a blind test. Every
clip is normalised to the same integrated loudness before it is labelled, so
the comparison is of voice rather than of gain.

**Same words.** Every candidate speaks the identical passage. Nothing is
re-prompted, re-rolled or hand-picked; the first generation is the one that
gets scored.
"""
from __future__ import annotations

import hashlib
import json
import random
import struct
import wave
from dataclasses import dataclass, field
from typing import Callable, Optional

#: Everything is resampled/emitted at the model's own rate; this is only the
#: rate used when a candidate does not report one.
FALLBACK_RATE = 24000

#: Integrated loudness every clip is normalised to. -23 LUFS is the broadcast
#: convention and leaves headroom for the peaks a lively read produces.
TARGET_LUFS = -23.0

#: Fallback when pyloudnorm is unavailable: match RMS to this dBFS instead.
TARGET_DBFS = -20.0


@dataclass
class Candidate:
    """One engine under test, and how to make it speak."""

    key: str                       # stable id, appears only in the key file
    label: str                     # human name, revealed after scoring
    synth: Callable                # (text) -> (float32 samples, sample_rate)
    notes: str = ""
    available: bool = True
    reason: str = ""


@dataclass
class Clip:
    passage_id: str
    candidate_key: str
    letter: str = ""
    seconds: float = 0.0
    sample_rate: int = 0
    generate_seconds: float = 0.0
    loudness_before: Optional[float] = None
    loudness_after: Optional[float] = None
    detail: dict = field(default_factory=dict)


def normalise(samples, sample_rate: int) -> tuple:
    """Match integrated loudness so gain cannot masquerade as quality.

    Uses pyloudnorm when present - chatterbox already depends on it - and falls
    back to RMS matching, which is cruder but still removes the gross level
    differences that decide blind tests for the wrong reason.
    """
    import numpy as np

    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        return samples, None, None
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sample_rate)
        before = float(meter.integrated_loudness(samples))
        if not np.isfinite(before):
            raise ValueError("non-finite loudness")
        out = pyln.normalize.loudness(samples, before, TARGET_LUFS)
        after = float(meter.integrated_loudness(out))
    except Exception:
        rms = float(np.sqrt(np.mean(samples ** 2))) or 1e-9
        before = 20.0 * np.log10(rms)
        gain = 10.0 ** ((TARGET_DBFS - before) / 20.0)
        out = samples * gain
        after = TARGET_DBFS
    # Clip rather than wrap: a normalised lively read can exceed full scale.
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out = out / peak * 0.999
    return out.astype(np.float32), before, after


def write_wav(path, samples, sample_rate: int) -> None:
    """16-bit mono PCM. A wav, not a stream - this one is for listening to."""
    import numpy as np

    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm)


def assign_letters(candidate_keys: list, passage_id: str, seed: int) -> dict:
    """Letters, randomised per passage so they do not carry across.

    Seeded by (seed, passage_id) so a run is reproducible - a listener who
    wants a second opinion later gets the same arrangement - while still being
    unguessable from the ordering of the candidate list.
    """
    digest = hashlib.sha256(f"{seed}:{passage_id}".encode()).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    shuffled = list(candidate_keys)
    rng.shuffle(shuffled)
    letters = [chr(ord("A") + index) for index in range(len(shuffled))]
    return dict(zip(shuffled, letters))


SCORES = [
    ("naturalness", "Could this be a person? Or is something off?"),
    ("warmth", "Does it feel like someone talking to you, or reading at you?"),
    ("intrigue", "Does it make you want to keep listening?"),
    ("authority", "Would you believe a fact from this voice?"),
    ("expressiveness", "Does emphasis land on the right words? Any range?"),
    ("pacing", "Prosody, phrasing, breaths, and the handling of numbers."),
    ("artificial", "How artificial does it sound? (1 = very, 5 = not at all)"),
    ("is_fam", "Does this feel like FAM?"),
]


def scorecard_markdown(passages: list, letters_by_passage: dict) -> str:
    """One sheet, filled in while listening, before any name is revealed."""
    lines = [
        "# FAM voice bake-off — scorecard", "",
        "Score each voice **1-5** on each row. 3 is 'fine, unremarkable'.",
        "",
        "**Do not open `KEY.json` until every sheet below is filled in.** The",
        "whole value of this test is that you do not know which engine you are",
        "listening to while you decide.",
        "",
        "Suggested order: listen to every voice on one passage before moving to",
        "the next, and note a first impression before replaying. Second",
        "listens reward familiarity, which is not the same as quality.",
        "",
        "| | meaning |", "|---|---|",
    ]
    lines += [f"| **{name}** | {question} |" for name, question in SCORES]
    lines.append("")
    lines.append("Note that **artificial** is reversed on purpose: 5 means it "
                 "does *not* sound artificial, so every row reads 'higher is "
                 "better'.")
    lines.append("")

    for passage in passages:
        letters = sorted(letters_by_passage[passage["id"]].values())
        lines += [f"## {passage['label']}", "",
                  f"*{passage['stresses']}*", "",
                  f"**Listen for:** {passage['watch_for']}", "",
                  "> " + passage["text"].replace("\n", " "), "",
                  "| | " + " | ".join(f"Voice {letter}" for letter in letters) + " |",
                  "|---|" + "---|" * len(letters)]
        for name, _ in SCORES:
            lines.append(f"| {name} | " + " | ".join([" "] * len(letters)) + " |")
        lines += ["", "Notes:", "", "- ", "", "**Best on this passage:** ____",
                  "", "---", ""]

    lines += ["## After all three passages", "",
              "**Overall winner:** ____", "",
              "**Would you ship it?** ____", "",
              "**Better than Piper?** ____   *(Piper is in this test, unnamed.)*",
              "",
              "Only now open `KEY.json`.", ""]
    return "\n".join(lines)


def player_html(passages: list, letters_by_passage: dict) -> str:
    """A local page with the clips side by side. No network, no CDN."""
    rows = []
    for passage in passages:
        letters = sorted(letters_by_passage[passage["id"]].values())
        players = "".join(
            f'<div class="v"><h3>Voice {letter}</h3>'
            f'<audio controls preload="none" '
            f'src="clips/{passage["id"]}/{letter}.wav"></audio></div>'
            for letter in letters)
        rows.append(
            f'<section><h2>{passage["label"]}</h2>'
            f'<p class="stress">{passage["stresses"]}</p>'
            f'<p class="watch"><strong>Listen for:</strong> {passage["watch_for"]}</p>'
            f'<blockquote>{passage["text"]}</blockquote>'
            f'<div class="voices">{players}</div></section>')
    return f"""<!doctype html>
<meta charset="utf-8"><title>FAM voice bake-off</title>
<style>
 body{{font:16px/1.55 -apple-system,BlinkMacSystemFont,sans-serif;
      max-width:52rem;margin:2rem auto;padding:0 1.2rem;color:#1a1a1a}}
 h1{{font-size:1.5rem}} h2{{font-size:1.15rem;margin-top:2.4rem}}
 h3{{font-size:.95rem;margin:0 0 .35rem}}
 blockquote{{border-left:3px solid #ddd;margin:1rem 0;padding:.2rem 0 .2rem 1rem;
             color:#333}}
 .stress,.watch{{color:#555;font-size:.9rem;margin:.3rem 0}}
 .voices{{display:grid;gap:1rem;grid-template-columns:repeat(auto-fit,minmax(15rem,1fr))}}
 .v{{background:#f6f6f6;border-radius:8px;padding:.9rem}}
 audio{{width:100%}}
 .warn{{background:#fff5d6;border:1px solid #e6d089;border-radius:8px;
        padding:.8rem 1rem;font-size:.92rem}}
</style>
<h1>FAM voice bake-off — blind</h1>
<p class="warn"><strong>Do not open <code>KEY.json</code> until the scorecard is
filled in.</strong> Letters are randomised separately for each passage, so
Voice A on one passage is not Voice A on another.</p>
<p>Score in <code>scorecard.md</code> as you go. Listen to every voice on one
passage before moving on.</p>
{''.join(rows)}
"""
