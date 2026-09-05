"""Statistics that refuse to overclaim.

Latency is skewed and these samples are small - ten trials, sometimes three. A
t-test on that is a way of producing a confident number from evidence that does
not support one, and this project has already paid for one confident claim that
turned out to be wrong.

So: medians rather than means for the headline, the full spread always shown,
and comparisons by **bootstrap confidence interval on the difference of
medians**. When that interval straddles zero the answer is "no detectable
difference at this sample size", and `required_n` estimates what it would take
to see one. That sentence is a real result and the report prints it as such.
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, asdict
from typing import Optional, Sequence


@dataclass
class Summary:
    """The distribution of one measurement, for one arm."""

    n: int
    median: float
    mean: float
    stdev: float
    minimum: float
    maximum: float
    p95: float
    iqr: float

    def to_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in asdict(self).items()}

    def brief(self) -> str:
        return f"{self.median:.2f}s (n={self.n}, {self.minimum:.2f}-{self.maximum:.2f})"


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile. `q` in 0..1."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[int(pos)]
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def summarise(values: Sequence[float]) -> Optional[Summary]:
    """None for an empty sample - an arm where every trial failed has no stats."""
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return None
    return Summary(
        n=len(clean),
        median=statistics.median(clean),
        mean=statistics.fmean(clean),
        stdev=statistics.stdev(clean) if len(clean) > 1 else 0.0,
        minimum=min(clean),
        maximum=max(clean),
        p95=percentile(clean, 0.95),
        iqr=percentile(clean, 0.75) - percentile(clean, 0.25),
    )


@dataclass
class Comparison:
    """Whether two arms actually differ, stated as conservatively as the data allows."""

    a_label: str
    b_label: str
    a_median: float
    b_median: float
    difference: float          # a - b, seconds. Negative means a is faster.
    ci_low: float
    ci_high: float
    confidence: float
    significant: bool
    verdict: str
    required_n: Optional[int] = None

    def to_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    *,
    a_label: str = "A",
    b_label: str = "B",
    iterations: int = 10000,
    confidence: float = 0.95,
    seed: Optional[int] = 1234,
) -> Optional[Comparison]:
    """Confidence interval on the difference of medians, by resampling.

    Nonparametric on purpose: it assumes nothing about the shape of the
    distribution, which matters because latency has a long right tail and a
    handful of samples.

    `seed` is fixed by default so a report is reproducible from its stored
    trials - re-rendering a run must not change its conclusion.
    """
    a_clean = [v for v in a if v is not None and not math.isnan(v)]
    b_clean = [v for v in b if v is not None and not math.isnan(v)]
    if len(a_clean) < 2 or len(b_clean) < 2:
        return None

    rng = random.Random(seed)
    a_med = statistics.median(a_clean)
    b_med = statistics.median(b_clean)
    observed = a_med - b_med

    diffs = []
    for _ in range(iterations):
        ra = [a_clean[rng.randrange(len(a_clean))] for _ in a_clean]
        rb = [b_clean[rng.randrange(len(b_clean))] for _ in b_clean]
        diffs.append(statistics.median(ra) - statistics.median(rb))
    diffs.sort()
    tail = (1.0 - confidence) / 2.0
    low = percentile(diffs, tail)
    high = percentile(diffs, 1.0 - tail)
    significant = (low > 0) or (high < 0)

    if significant:
        faster, slower = (a_label, b_label) if observed < 0 else (b_label, a_label)
        verdict = (
            f"{faster} is faster than {slower} by {abs(observed):.2f}s "
            f"(median), {int(confidence * 100)}% CI [{low:+.2f}, {high:+.2f}]."
        )
        needed = None
    else:
        verdict = (
            f"No detectable difference between {a_label} and {b_label} at "
            f"n={len(a_clean)}/{len(b_clean)}: the {int(confidence * 100)}% CI "
            f"[{low:+.2f}, {high:+.2f}] includes zero. The observed gap of "
            f"{observed:+.2f}s is within noise."
        )
        needed = required_n(a_clean, b_clean)

    return Comparison(
        a_label=a_label,
        b_label=b_label,
        a_median=a_med,
        b_median=b_med,
        difference=observed,
        ci_low=low,
        ci_high=high,
        confidence=confidence,
        significant=significant,
        verdict=verdict,
        required_n=needed,
    )


def required_n(a: Sequence[float], b: Sequence[float], power: float = 0.8) -> Optional[int]:
    """Roughly how many trials per arm would be needed to resolve this gap.

    A normal-approximation sample-size estimate. It is an order-of-magnitude
    guide for "is another run worth it", not a promise - which is why the
    report words it as "about N".
    """
    if len(a) < 2 or len(b) < 2:
        return None
    effect = abs(statistics.fmean(a) - statistics.fmean(b))
    if effect <= 0:
        return None
    pooled = math.sqrt((statistics.variance(a) + statistics.variance(b)) / 2.0)
    if pooled <= 0:
        return None
    # z(0.975) + z(0.80), the standard two-sided 95%/80%-power constants.
    n = 2 * ((1.96 + 0.84) * pooled / effect) ** 2
    return max(3, math.ceil(n))
