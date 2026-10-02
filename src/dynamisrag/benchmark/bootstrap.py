"""The paired bootstrap over per-query metrics, frozen at seed 138.

RES-138's second selection step is "compare the top two by paired bootstrap; if
the 95% CI for the nDCG@10 difference excludes 0, they are not tied". That is
only a decision procedure if the interval is a property of the evidence rather
than of the run, so every parameter of it is named once:

* **seed 138** — the resampling stream. Drawn from ``random.Random`` rather than
  from NumPy's ``Generator``, whose stream is explicitly not guaranteed stable
  across library versions; a benchmark artifact has to be reproducible from a
  fixed result set on a machine that has never run this code.
* **10,000 replicates** — enough for a 95% percentile interval to be stable to a
  few thousandths, and a number rather than a convergence tolerance so the cost
  is bounded and identical on every machine.
* **confidence 0.95** — the level, reported alongside the interval so a reader
  never has to assume it.

**Paired, and pairing is the whole point.** The same resampled query positions
are applied to both candidates within each workload, because both candidates
answered the *same* queries on the *same* corpus. An unpaired resample would
measure the variance of each candidate separately and throw away the covariance
between them, which for two models ranking the same 323 queries is most of the
signal: two candidates that agree on 95% of queries would still get wide
overlapping intervals from an unpaired design.

Within each replicate every workload is resampled independently, each workload's
macro mean is recomputed from its own resample, and the replicate's difference is
the difference of the unweighted across-workload means — the same statistic the
observed difference reports, so the interval is an interval around *that*
number.

**The function is pure.** It takes per-query values and returns an interval. It
reads no clock, no environment, no file, and it returns its own parameters, so
the same result set always yields the same interval.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from random import Random
from typing import Final

from dynamisrag.benchmark.contracts import (
    RES138_BOOTSTRAP_CONFIDENCE,
    RES138_BOOTSTRAP_SAMPLES,
    RES138_BOOTSTRAP_SEED,
    require_exact_int,
)
from dynamisrag.benchmark.errors import BenchmarkContractError

__all__ = [
    "RES138_BOOTSTRAP_PARAMETERS",
    "BootstrapEstimate",
    "BootstrapParameters",
    "paired_bootstrap",
]


@dataclass(frozen=True)
class BootstrapParameters:
    """The three frozen parameters, validated together.

    Kept as one value rather than three function arguments because they travel
    together into an artifact: a result that recorded a seed but not a sample
    count would have an interval nobody could reproduce.
    """

    seed: int = RES138_BOOTSTRAP_SEED
    samples: int = RES138_BOOTSTRAP_SAMPLES
    confidence: float = RES138_BOOTSTRAP_CONFIDENCE

    def __post_init__(self) -> None:
        # `bool` is an `int`, and `seed=True` would be seed 1 rather than a flag. The
        # shared primitive gate owns that rule so it cannot drift from the one the
        # workload contracts apply.
        require_exact_int(
            self.seed,
            kind="seed",
            operation="bootstrap_parameters",
            minimum=0,
            because="A seed names a position in a reproducible stream, so it is a number.",
        )
        require_exact_int(
            self.samples,
            kind="sample count",
            operation="bootstrap_parameters",
            minimum=2,
            because="Two replicates are the minimum that can produce two distinct order "
            "statistics, so a smaller count cannot produce an interval at all.",
        )
        if not 0.0 < self.confidence < 1.0:
            raise BenchmarkContractError(
                f"bootstrap confidence {self.confidence!r} is not strictly between 0 and 1. A "
                "confidence of 0 or 1 produces an interval that is either empty or the whole "
                "range, neither of which is an interval.",
                operation="bootstrap_parameters",
            )

    def payload(self) -> dict[str, object]:
        """The hashed description of these parameters."""
        return {"seed": self.seed, "samples": self.samples, "confidence": self.confidence}


RES138_BOOTSTRAP_PARAMETERS: Final[BootstrapParameters] = BootstrapParameters()


@dataclass(frozen=True)
class BootstrapEstimate:
    """One paired difference and its interval, with the parameters that produced it."""

    metric: str
    observed_difference: float
    lower: float
    upper: float
    samples: int
    seed: int
    confidence: float
    workloads: tuple[str, ...]
    resampling_unit: str = "query-within-workload"

    def excludes_zero(self) -> bool:
        """Whether the interval excludes 0 — RES-138's step-2 tie test."""
        return self.lower > 0.0 or self.upper < 0.0

    def payload(self) -> dict[str, object]:
        """The hashed description of this estimate."""
        return {
            "metric": self.metric,
            "observed_difference": self.observed_difference,
            "lower": self.lower,
            "upper": self.upper,
            "samples": self.samples,
            "seed": self.seed,
            "confidence": self.confidence,
            "workloads": list(self.workloads),
            "resampling_unit": self.resampling_unit,
        }


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    """Linear interpolation between order statistics (the type-7 convention).

    Stated rather than inherited: the percentile method changes the third decimal
    of an interval, so an interval reported without saying how its endpoints were
    computed is not reproducible. ``ordered`` must already be ascending.
    """
    if not ordered:
        raise BenchmarkContractError(
            "a percentile of an empty sample is undefined.",
            operation="paired_bootstrap",
        )
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


def _macro(values: Sequence[float], indices: Sequence[int]) -> float:
    """Mean of the resampled positions, in plain float arithmetic.

    Plain ``sum`` rather than ``math.fsum``: the summation order is fixed by the
    drawn indices, so the result is deterministic, and ``fsum`` would buy exact
    rounding at a cost paid on every one of 10,000 replicates for no benefit at
    float64 precision.
    """
    total = 0.0
    for index in indices:
        total += values[index]
    return total / len(indices)


def paired_bootstrap(
    *,
    candidate_a: Mapping[str, Mapping[str, float]],
    candidate_b: Mapping[str, Mapping[str, float]],
    metric: str,
    parameters: BootstrapParameters = RES138_BOOTSTRAP_PARAMETERS,
) -> BootstrapEstimate:
    """Interval for ``macro(a) - macro(b)``, resampling queries within each workload.

    ``candidate_a`` and ``candidate_b`` map a workload name to that candidate's
    per-query values. They must hold **the same workload names and the same query
    ids** in each, because the pairing is positional: the difference is only
    meaningful if both candidates were asked the same question. A mismatch is
    refused rather than intersected, because silently dropping a query would give
    the two candidates different denominators and the interval would then describe
    a comparison nobody ran.
    """
    if not candidate_a:
        raise BenchmarkContractError(
            "a paired bootstrap needs at least one workload. An empty candidate has nothing to "
            "resample.",
            operation="paired_bootstrap",
        )
    if sorted(candidate_a) != sorted(candidate_b):
        raise BenchmarkContractError(
            "the two candidates were measured on different workloads "
            f"({sorted(candidate_a)} against {sorted(candidate_b)}). A paired comparison needs "
            "both to have answered the same questions.",
            operation="paired_bootstrap",
        )
    workloads = tuple(sorted(candidate_a))
    columns: list[tuple[list[float], list[float]]] = []
    for workload in workloads:
        left = candidate_a[workload]
        right = candidate_b[workload]
        if sorted(left) != sorted(right):
            raise BenchmarkContractError(
                f"workload {workload!r} was scored for the two candidates over different query "
                "sets. A paired bootstrap compares the same query positions for both, so the "
                "query sets must be identical.",
                operation="paired_bootstrap",
                workload=workload,
            )
        query_ids = sorted(left)
        if not query_ids:
            raise BenchmarkContractError(
                f"workload {workload!r} contributes no scored queries, so it cannot be resampled. "
                "A workload whose queries are all excluded is absent from the comparison rather "
                "than present with nothing in it.",
                operation="paired_bootstrap",
                workload=workload,
            )
        columns.append(
            (
                [left[query_id] for query_id in query_ids],
                [right[query_id] for query_id in query_ids],
            )
        )

    observed = _macro_average(columns, [list(range(len(values))) for _, values in columns])
    # Not cryptographic and not random: a seeded Mersenne Twister is exactly what
    # a reproducible resampling stream needs, and a CSPRNG here would make every
    # replicate unreproducible. Bandit flags the constructor, not the intent.
    generator = Random(parameters.seed)  # noqa: S311
    differences: list[float] = []
    sizes = [len(values) for values, _ in columns]
    for _ in range(parameters.samples):
        indices = [[generator.randrange(size) for _ in range(size)] for size in sizes]
        differences.append(_macro_average(columns, indices))

    differences.sort()
    tail = (1.0 - parameters.confidence) / 2.0
    return BootstrapEstimate(
        metric=metric,
        observed_difference=observed,
        lower=_percentile(differences, tail),
        upper=_percentile(differences, 1.0 - tail),
        samples=parameters.samples,
        seed=parameters.seed,
        confidence=parameters.confidence,
        workloads=workloads,
    )


def _macro_average(
    columns: Sequence[tuple[list[float], list[float]]], indices: Sequence[Sequence[int]]
) -> float:
    """Difference of the two unweighted across-workload means for one resample.

    ``indices`` is per workload and holds positions into both candidates'
    columns, which is what makes the comparison paired.
    """
    per_workload: list[float] = []
    for (left, right), sampled in zip(columns, indices, strict=True):
        per_workload.append(_macro(left, sampled) - _macro(right, sampled))
    return sum(per_workload) / len(per_workload)
