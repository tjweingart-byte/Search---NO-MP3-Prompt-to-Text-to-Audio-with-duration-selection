"""Blind bake-off across four speaker identities on one engine.

Phase 2 chose the engine. This chooses the voice, and the two are deliberately
not asked at once: Chatterbox Base is held fixed and only the reference
recording changes.

**Generation settings are held identical across all four**, including the
random seed. Sampling is stochastic - `temperature=0.8`, `min_p`, `top_p` - so
without a fixed seed two identities would differ partly by luck of the draw,
and the luckier draw would win a blind test. Seeding per (identity, passage)
gives every identity the same random stream for the same passage.

The scoring is deliberately thinner than the engine bake-off's eight axes. The
question here is not *why* a voice is better; it is *which one is FAM*.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

#: The four directions being compared. The keys are the reference filenames.
IDENTITIES = (
    ("magnetic", "Magnetic",
     "intimate, intriguing, slightly restrained, sophisticated; pulls the "
     "listener toward it rather than demanding attention"),
    ("human", "Human",
     "exceptionally conversational and natural; a smart person beside you, "
     "not a narrator, announcer or assistant"),
    ("storyteller", "Storyteller",
     "warm, emotionally intelligent, dynamic; timing and movement without "
     "theatricality"),
    ("authority", "Modern Authority",
     "confident, composed, intelligent, sophisticated, still human; "
     "authoritative without becoming a news anchor"),
)

#: Chatterbox Base defaults, pinned so a later run cannot drift. These are the
#: values `generate()` uses when nothing is passed, and they stay untouched for
#: this experiment: identity is the only variable. Delivery tuning
#: (exaggeration, cfg_weight, min_p, pacing) is the *next* experiment.
GENERATION = {
    "exaggeration": 0.5,
    "cfg_weight": 0.5,
    "temperature": 0.8,
    "repetition_penalty": 1.2,
    "min_p": 0.05,
    "top_p": 1.0,
}


@dataclass
class IdentityClip:
    passage_id: str
    identity_key: str
    letter: str = ""
    seconds: float = 0.0
    sample_rate: int = 0
    generate_seconds: float = 0.0
    seed: int = 0
    loudness_before: Optional[float] = None
    loudness_after: Optional[float] = None
    detail: dict = field(default_factory=dict)


def seed_for(passage_id: str, base_seed: int) -> int:
    """One seed per passage, shared by every identity.

    Shared rather than per-identity on purpose: the four voices should differ
    because the reference differs, not because one of them drew a better
    sample. Same passage, same random stream, four references.
    """
    import hashlib

    digest = hashlib.sha256(f"{base_seed}:{passage_id}".encode()).hexdigest()
    return int(digest[:8], 16)


def apply_seed(seed: int) -> None:
    """Pin torch's RNG so the sampler makes the same choices for every identity."""
    try:
        import torch

        torch.manual_seed(seed)
        if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
            try:
                torch.mps.manual_seed(seed)
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def choice_sheet(passages: list, letters_by_passage: dict) -> str:
    """Four questions, not forty. Which voice, and would you ship it."""
    lines = [
        "# FAM voice identity — choices", "",
        "One engine (Chatterbox Base), four reference voices, identical",
        "generation settings and the same random seed per passage.",
        "",
        "**Do not open `KEY.json` until this sheet is finished.** Letters are",
        "randomised separately for each passage, so Voice A on one passage is",
        "not Voice A on another.",
        "",
        "---", "",
    ]
    for passage in passages:
        letters = sorted(letters_by_passage[passage["id"]].values())
        lines += [
            f"## {passage['label']}", "",
            "> " + passage["text"].replace("\n", " "), "",
            f"Voices: {', '.join(letters)}", "",
            "**Best voice for this passage:** ____", "",
            "---", "",
        ]
    lines += [
        "## Overall", "",
        "**Overall favourite:** ____", "",
        "**Would you ship this as FAM's voice?**  Yes / Maybe / No", "",
        "**Notes** (optional):", "",
        "> ", "",
        "---", "",
        "Only now open `KEY.json`.", "",
        "Whichever wins, its *delivery* is not yet tuned. `exaggeration`,",
        "`cfg_weight`, `min_p` and pacing were all held at their defaults so",
        "that no voice won on settings. Tuning them is the next experiment.",
        "",
    ]
    return "\n".join(lines)


def identity_player_html(passages: list, letters_by_passage: dict) -> str:
    """The listening page. Same shape as the engine bake-off, thinner prompts."""
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
            f'<blockquote>{passage["text"]}</blockquote>'
            f'<div class="voices">{players}</div>'
            f'<p class="ask"><strong>Best voice for this passage?</strong></p>'
            f'</section>')
    return f"""<!doctype html>
<meta charset="utf-8"><title>FAM voice identity — blind</title>
<style>
 body{{font:16px/1.55 -apple-system,BlinkMacSystemFont,sans-serif;
      max-width:52rem;margin:2rem auto;padding:0 1.2rem;color:#1a1a1a}}
 h1{{font-size:1.5rem}} h2{{font-size:1.15rem;margin-top:2.4rem}}
 h3{{font-size:.95rem;margin:0 0 .35rem}}
 blockquote{{border-left:3px solid #ddd;margin:1rem 0;padding:.2rem 0 .2rem 1rem;
             color:#333}}
 .voices{{display:grid;gap:1rem;
          grid-template-columns:repeat(auto-fit,minmax(15rem,1fr))}}
 .v{{background:#f6f6f6;border-radius:8px;padding:.9rem}}
 audio{{width:100%}}
 .ask{{margin-top:1rem;color:#444}}
 .warn{{background:#fff5d6;border:1px solid #e6d089;border-radius:8px;
        padding:.8rem 1rem;font-size:.92rem}}
 .note{{color:#555;font-size:.92rem}}
</style>
<h1>FAM voice identity — blind</h1>
<p class="warn"><strong>Do not open <code>KEY.json</code> until you have chosen.</strong>
Letters are randomised separately for each passage.</p>
<p class="note">One engine, four reference voices. Generation settings and the
random seed are identical across all four, so no voice is winning on settings —
delivery tuning comes after a voice is chosen.</p>
<p>Record your choices in <code>choices.md</code>.</p>
{''.join(rows)}
<section><h2>Overall</h2>
<p><strong>Overall favourite?</strong><br>
<strong>Would you ship this as FAM's voice?</strong> Yes / Maybe / No<br>
<strong>Any notes?</strong></p></section>
"""
