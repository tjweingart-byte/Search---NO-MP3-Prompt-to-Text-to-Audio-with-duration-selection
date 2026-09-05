"""What a sweep will cost before it runs, and what it did cost after.

Two numbers, kept apart on purpose.

The **estimate** is arithmetic over published prices and an assumed episode
size. It is shown in the plan so nothing spends without the number being seen
first, and it is allowed to be wrong.

The **actual** comes from the API's own usage figures per trial. Only that one
goes in the report as a cost.

GPU time is never estimated as though this tool might start a pod. A remote GPU
stage costs what it costs while *someone else* has it running; the estimate
says so rather than implying this tool would rent one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: Dollars per million tokens (input, output), from the published price list.
#: Kept in step with compare_models.py, which uses the same table.
PRICES = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

#: Assumed shape of one episode, for the estimate only.
ASSUMED_INPUT_TOKENS = 3000
ASSUMED_OUTPUT_TOKENS_PER_MINUTE = 260
#: A searched call reads its results back in, which dominates input tokens.
SEARCH_INPUT_TOKEN_MULTIPLIER = 4.0

EXA_COST_PER_SEARCH = 0.005


def model_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Actual dollars for one call, from real usage numbers."""
    price_in, price_out = PRICES.get(model, PRICES["claude-sonnet-5"])
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000.0


@dataclass
class Estimate:
    """A pre-run cost breakdown, itemised so the big line is obvious."""

    anthropic: float = 0.0
    exa: float = 0.0
    #: Dollars of GPU time the run would *consume* on infrastructure that is
    #: already running. Zero when no GPU arm is involved.
    gpu: float = 0.0
    trials: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def total(self) -> float:
        return self.anthropic + self.exa + self.gpu

    def to_dict(self) -> dict:
        return {
            "anthropic": round(self.anthropic, 4),
            "exa": round(self.exa, 4),
            "gpu": round(self.gpu, 4),
            "total": round(self.total, 4),
            "trials": self.trials,
            "notes": list(self.notes),
        }

    def render(self) -> str:
        lines = [f"  {'Anthropic':<12} ${self.anthropic:>7.3f}"]
        if self.exa:
            lines.append(f"  {'Exa':<12} ${self.exa:>7.3f}")
        if self.gpu:
            lines.append(f"  {'GPU time':<12} ${self.gpu:>7.3f}   (on infrastructure you are already running)")
        lines.append(f"  {'TOTAL':<12} ${self.total:>7.3f}   across {self.trials} trials")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


def estimate(spec) -> Estimate:
    """Estimated cost of running `spec` once, itemised by service."""
    from experiments.adapters.tts import GPU_DOLLARS_PER_HOUR

    est = Estimate(trials=spec.total_trials)
    per_arm_trials = len(spec.queries) * spec.trials

    for arm in spec.arms:
        model = arm.model or _default_model()
        searched = arm.search not in ("none", "")
        input_tokens = ASSUMED_INPUT_TOKENS
        if arm.search == "anthropic_web_search":
            input_tokens = int(input_tokens * SEARCH_INPUT_TOKEN_MULTIPLIER)
        elif arm.search == "exa":
            # Exa context is pasted in, so it also inflates input tokens.
            input_tokens = int(input_tokens * SEARCH_INPUT_TOKEN_MULTIPLIER)
        output_tokens = int(ASSUMED_OUTPUT_TOKENS_PER_MINUTE * spec.minutes)
        est.anthropic += model_cost(model, input_tokens, output_tokens) * per_arm_trials

        if arm.search == "exa":
            caps = int(arm.params.get("max_searches", 3) or 3)
            est.exa += EXA_COST_PER_SEARCH * caps * per_arm_trials

        if arm.tts == "chatterbox":
            # Assume synthesis runs comfortably faster than realtime; the pod
            # bills wall-clock regardless, so this is time consumed, not rented.
            audio_seconds = spec.minutes * 60.0
            gpu_seconds = audio_seconds / 20.0
            est.gpu += gpu_seconds / 3600.0 * GPU_DOLLARS_PER_HOUR * per_arm_trials
            est.notes.append(
                "GPU cost assumes a pod you have already started. This tool "
                "never creates, starts or stops GPU infrastructure."
            )

    if any(a.search == "exa" for a in spec.arms):
        est.notes.append(
            "Exa pricing is the published per-search rate; the real figure is "
            "recorded from the API response once the adapter is connected."
        )
    return est


def _default_model() -> str:
    try:
        from config import settings

        return settings.model
    except Exception:
        return "claude-sonnet-5"
