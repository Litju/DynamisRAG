"""The sealed RES-138 Stage A result, loaded strictly, as Stage B's only input.

Stage B does not re-run Stage A and does not interpret it. It loads the sealed
bundle and checks every binding before any Stage B work happens, so that a Stage B
number is always a number about *this* Stage A result and about no other.

**The seal.** :data:`RES138_STAGE_A_SEAL` is the identity of one finished Stage A
run, as declared by the run that produced it and reviewed when it was accepted:
the bundle digest, the full-run digest, the code commit, the pinned model
revisions, the Stage A input policy, the frozen BEIR source digests, and the
shortlist Stage A handed to Stage B. It is data, not a derivation: recomputing it
from the bundle would be circular, because the point is that the *expected*
identity is known before the bundle is read.

**No partial or best-effort loading.** :func:`load_sealed_stage_a` either returns a
:class:`SealedStageA` whose every field was verified against the seal, or raises.
It has no ``strict=False``, no ``ignore=`` and no "warnings" list, for the same
reason :func:`~dynamisrag.benchmark.bundle.verify_run_bundle` has none: a Stage B
run that proceeded on a bundle it could not fully verify would be measuring a
reference it cannot name, and the qualification payload would bind a digest that
describes nothing.

**What Stage A decided and what Stage B may still decide.** Stage A decided the
*quality* ranking and, from it, the shortlist:
:data:`RES138_STAGE_B_MODEL_IDS` is one model, so the Stage B shortlist is that
model at 512 and at 1024. Voyage 4 Nano remains Stage A evidence — its macro
metrics, rankings and bootstrap intervals are in the sealed bundle and are not
modified, hidden or re-read here — but it is not qualified, so
:func:`require_stage_b_shortlist` refuses it. That exclusion is a property of this
sealed result, not a preference encoded in the selection rule: the rule still
compares two candidates, and it still reads every number from the artifacts.

**Quality evidence, read not remembered.** :meth:`SealedStageA.quality` returns the
macro metrics Stage A persisted for a shortlisted configuration, so nDCG@10 and
Recall@100 that reach the selection table are the sealed run's own numbers. They
are never restated in this repository, which is what makes a drift between the
selection table and the sealed bundle impossible rather than merely unlikely.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Final, cast

from dynamisrag.benchmark.artifacts import (
    Res138RunManifest,
    file_sha256,
    read_artifact,
)
from dynamisrag.benchmark.bundle import BundleVerification, verify_run_bundle
from dynamisrag.benchmark.contracts import (
    RES138_BEIR_SOURCES,
    RES138_CANDIDATE_DIMENSIONS,
    RES138_INPUT_MAX_TOKENS,
    RES138_INPUT_TRUNCATION_DIRECTION,
    RES138_MODEL_CANDIDATES,
    RES138_MODEL_IDS,
    RES138_WORKLOAD_NAMES,
    require_candidate_dimension,
    require_code_sha,
    require_exact_str,
)
from dynamisrag.benchmark.errors import BenchmarkArtifactError, BenchmarkContractError
from dynamisrag.benchmark.production import StageAReference
from dynamisrag.benchmark.res138 import generation_semantics_sha256
from dynamisrag.benchmark.truncation import INPUT_POLICY_REVISION, input_policy_payload
from dynamisrag.embedding.contracts import canonical_json

__all__ = [
    "RES138_STAGE_A_SEAL",
    "RES138_STAGE_B_MODEL_IDS",
    "RES138_STAGE_B_SHORTLIST",
    "SealedStageA",
    "StageAQuality",
    "StageASeal",
    "load_sealed_stage_a",
    "require_stage_b_shortlist",
]

RES138_STAGE_B_MODEL_IDS: Final[tuple[str, ...]] = ("Qwen/Qwen3-Embedding-0.6B",)
"""The models the sealed Stage A result advanced to Stage B, in a fixed order.

**This is the sealed Stage A outcome, stated once.** Stage A measured both frozen
candidates over the three frozen workloads and produced this shortlist; the
admission decision was Stage A's, taken on quality evidence, and Stage B inherits
it rather than re-opening it. A candidate Stage A did not advance cannot be
introduced downstream, which is why
:func:`~dynamisrag.benchmark.production.ProductionQualification` refuses a
qualification naming a configuration outside the reference shortlist and why
:func:`require_stage_b_shortlist` refuses it here.

Voyage 4 Nano is absent because Stage A did not advance it, not because it is
unmeasured: its macro metrics, per-query rows and bootstrap intervals remain in
the sealed bundle and are what Stage A advanced the shortlist against.
"""

RES138_STAGE_B_SHORTLIST: Final[tuple[tuple[str, int], ...]] = tuple(
    (model_id, dimension)
    for model_id in RES138_STAGE_B_MODEL_IDS
    for dimension in RES138_CANDIDATE_DIMENSIONS
)
"""The exact Stage B shortlist: the advanced model at every frozen dimension.

Derived from :data:`RES138_STAGE_B_MODEL_IDS` and
:data:`~dynamisrag.benchmark.contracts.RES138_CANDIDATE_DIMENSIONS` rather than
written out, so a candidate dimension added to the frozen tuple later fails
:func:`require_stage_b_shortlist` instead of silently leaving one dimension out of
the shortlist and letting the selection rule compare a single pair.
"""


@dataclass(frozen=True)
class StageASeal:
    """One accepted Stage A result's identity, as declared outside the bundle.

    Every field is a *precondition*: the loader compares what the bundle says
    against what this says and refuses on any difference, including a difference
    that would otherwise look harmless. ``source_digests`` and ``input_policy`` are
    included for the same reason as the digests — a Stage B run has to be able to
    state the whole identity of the reference it qualifies, not only its digest.
    """

    bundle_sha256: str
    full_run_sha256: str
    code_sha: str
    model_revisions: tuple[tuple[str, str], ...]
    input_policy_revision: str
    input_max_tokens: int
    truncate: bool
    truncation_direction: str
    source_digests: tuple[tuple[str, str], ...]
    shortlist: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        require_code_sha(self.code_sha, operation="stage_a_seal")
        for name, digest in (
            ("bundle_sha256", self.bundle_sha256),
            ("full_run_sha256", self.full_run_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise BenchmarkContractError(
                    f"the Stage A seal {name} is {digest!r}, which is not 64 lowercase "
                    "hexadecimal characters. A seal made of digests has to be made of digests.",
                    operation="stage_a_seal",
                )
        if self.input_policy_revision != INPUT_POLICY_REVISION:
            raise BenchmarkContractError(
                f"the Stage A seal declares input policy {self.input_policy_revision!r}, not the "
                f"frozen {INPUT_POLICY_REVISION!r}. Stage B qualifies a reference produced under "
                "one declared input policy, and a qualification of vectors produced under another "
                "would reproduce a different experiment.",
                operation="stage_a_seal",
            )
        if self.input_max_tokens != RES138_INPUT_MAX_TOKENS:
            raise BenchmarkContractError(
                f"the Stage A seal declares input boundary {self.input_max_tokens}, not the frozen "
                f"Stage A reference boundary {RES138_INPUT_MAX_TOKENS}. Longer context is the "
                "optional Stage C benchmark and cannot replace the Stage B semantic boundary.",
                operation="stage_a_seal",
            )
        if self.truncate is not True:
            raise BenchmarkContractError(
                "the Stage A seal must declare truncate=true. Stage B reproduces Stage A's "
                "semantic input policy exactly, and a reference that refused an over-long input "
                "measured a different function than a production path that truncates it.",
                operation="stage_a_seal",
            )
        if self.truncation_direction != RES138_INPUT_TRUNCATION_DIRECTION:
            raise BenchmarkContractError(
                "the Stage A seal declares truncation_direction "
                f"{self.truncation_direction!r}, not the frozen "
                f"{RES138_INPUT_TRUNCATION_DIRECTION!r}. Truncating the other end keeps "
                "a different part of an over-long input and is not the reference policy.",
                operation="stage_a_seal",
            )
        if {name for name, _ in self.source_digests} != set(RES138_WORKLOAD_NAMES):
            raise BenchmarkContractError(
                f"the Stage A seal covers workloads "
                f"{sorted(name for name, _ in self.source_digests)}, not the frozen "
                f"{list(RES138_WORKLOAD_NAMES)}. Stage B qualifies a reference measured "
                "over every frozen workload.",
                operation="stage_a_seal",
            )
        if dict(self.model_revisions) != {
            candidate.model_id: candidate.revision for candidate in RES138_MODEL_CANDIDATES
        }:
            raise BenchmarkContractError(
                "the Stage A seal does not bind every frozen candidate at its frozen revision. A "
                "seal that names a subset cannot establish that the excluded candidate's evidence "
                "is the one Stage A advanced the shortlist against.",
                operation="stage_a_seal",
            )
        if self.source_digests != tuple(sorted(self.source_digests)):
            raise BenchmarkContractError(
                "the Stage A seal source digests are not in canonical (sorted) order. Canonical "
                "order is what makes a hashed identity reproducible.",
                operation="stage_a_seal",
            )
        require_stage_b_shortlist(self.shortlist, operation="stage_a_seal")

    def payload(self) -> Mapping[str, object]:
        """The hashed description of the seal."""
        return {
            "bundle_sha256": self.bundle_sha256,
            "full_run_sha256": self.full_run_sha256,
            "code_sha": self.code_sha,
            "model_revisions": [list(pair) for pair in self.model_revisions],
            "input_policy": {
                "policy_revision": self.input_policy_revision,
                "input_max_tokens": self.input_max_tokens,
                "truncate": self.truncate,
                "truncation_direction": self.truncation_direction,
            },
            "source_digests": [list(pair) for pair in self.source_digests],
            "shortlist": [list(pair) for pair in self.shortlist],
        }


def require_stage_b_shortlist(
    candidates: Sequence[tuple[str, int]], *, operation: str
) -> tuple[tuple[str, int], ...]:
    """Require the shortlist to be exactly the sealed Stage B admissions.

    Three refusals, each for a different reason a shortlist can be wrong:

    * a model the sealed Stage A result did not advance — including Voyage 4 Nano,
      which remains Stage A evidence and is not qualified;
    * a dimension outside the frozen candidate dimensions, which no later stage may
      introduce;
    * an incomplete or extended shortlist, because the frozen selection rule
      compares two candidates and a third configuration would silently change
      which pair is compared.
    """
    selected = tuple(candidates)
    if not selected:
        raise BenchmarkContractError(
            "a Stage B shortlist must name at least one candidate-configuration.",
            operation=operation,
        )
    expected = set(RES138_STAGE_B_SHORTLIST)
    for model_id, dimension in selected:
        require_exact_str(model_id, kind="Stage B shortlist model id", operation=operation)
        if model_id not in RES138_MODEL_IDS:
            raise BenchmarkContractError(
                f"the Stage B shortlist names {model_id!r}, which is not a frozen candidate. "
                "Stage B qualifies a Stage A candidate; it cannot introduce one.",
                operation=operation,
                model_id=model_id,
            )
        if model_id not in RES138_STAGE_B_MODEL_IDS:
            raise BenchmarkContractError(
                f"the Stage B shortlist names {model_id!r}, which the sealed Stage A result "
                f"did not advance. The Stage B admissions are "
                f"{list(RES138_STAGE_B_MODEL_IDS)}. Stage A "
                "measured this candidate and its evidence remains in the sealed bundle, but a "
                "configuration Stage A did not advance cannot enter production qualification.",
                operation=operation,
                model_id=model_id,
            )
        require_candidate_dimension(dimension, operation=operation)
        if (model_id, dimension) not in expected:
            raise BenchmarkContractError(
                f"the Stage B shortlist names {model_id}@{dimension}, which is not one of the "
                f"sealed admissions {sorted(expected)}.",
                operation=operation,
                model_id=model_id,
            )
    if set(selected) != expected:
        raise BenchmarkContractError(
            f"the Stage B shortlist is {sorted(selected)}, not the sealed admissions "
            f"{sorted(expected)}. The frozen selection rule compares two candidates; a partial or "
            "extended shortlist would silently change which pair it compares.",
            operation=operation,
        )
    if len(selected) != len(set(selected)):
        raise BenchmarkContractError(
            "the Stage B shortlist repeats a candidate-configuration.", operation=operation
        )
    return selected


RES138_STAGE_A_SEAL: Final[StageASeal] = StageASeal(
    bundle_sha256="85f3d7b14db4aa2b4ccbd83b0e4b3f4dba2b57f6c7a4bc5515b15070d300b196",
    full_run_sha256="16386ad1c4a2e65f4ed2c72e951b16f88cbabfb32fbff829ccabe133e8c6d357",
    code_sha="1339dea8c0c06a90f0a073b1e5371c22c3418ee6",
    model_revisions=tuple(
        (candidate.model_id, candidate.revision) for candidate in RES138_MODEL_CANDIDATES
    ),
    input_policy_revision=INPUT_POLICY_REVISION,
    input_max_tokens=RES138_INPUT_MAX_TOKENS,
    truncate=True,
    truncation_direction=RES138_INPUT_TRUNCATION_DIRECTION,
    source_digests=tuple(
        sorted((source.workload, source.sha256) for source in RES138_BEIR_SOURCES)
    ),
    shortlist=RES138_STAGE_B_SHORTLIST,
)
"""The sealed Stage A result Stage B is bound to.

Two digests name the bytes: the bundle manifest digest is the whole run directory,
and the full-run digest is the summary that declares the quality evidence complete,
production qualification ``not_run`` and the production default ``not_configured``.
The code commit is the third, and it is what makes the plan, the generation
semantics and the shard identities in that bundle reproducible from this source
tree rather than merely present.
"""


@dataclass(frozen=True)
class StageAQuality:
    """One shortlisted configuration's macro quality metrics, as Stage A persisted them.

    Read from ``results/macro/<candidate>/<dimension>.json``, so the numbers the
    selection table consumes are the sealed run's own numbers rather than a
    transcription of them. ``workloads`` keeps the per-workload means so a reviewer
    can see that the macro average is over the three frozen workloads and not over
    a weighted corpus.
    """

    model_id: str
    model_revision: str
    dimension: int
    workload_count: int
    queries_scored: int
    ndcg_at_10: float
    recall_at_10: float
    recall_at_100: float
    workloads: tuple[tuple[str, float, float, float], ...]

    @property
    def label(self) -> str:
        """``model@dimension``, the label every Stage B table row uses."""
        return f"{self.model_id}@{self.dimension}"

    def payload(self) -> Mapping[str, object]:
        """The hashed description of this configuration's quality evidence."""
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "dimension": self.dimension,
            "workload_count": self.workload_count,
            "queries_scored": self.queries_scored,
            "ndcg_at_10": self.ndcg_at_10,
            "recall_at_10": self.recall_at_10,
            "recall_at_100": self.recall_at_100,
            "workloads": [list(row) for row in self.workloads],
        }


def _candidate_key(model_id: str) -> str:
    """The on-disk directory name for one candidate, as the full run writes it."""
    return model_id.replace("/", "__")


def _require_number(value: object, *, label: str, operation: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkArtifactError(
            f"{label} is {value!r}, which is not a number.", operation=operation
        )
    number = float(value)
    if not isfinite(number):
        raise BenchmarkArtifactError(f"{label} is not finite.", operation=operation)
    return number


def _require_count(value: object, *, label: str, operation: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BenchmarkArtifactError(
            f"{label} is {value!r}, which is not a non-negative integer count.",
            operation=operation,
        )
    return value


def _require_rows(
    value: object, *, label: str, operation: str
) -> tuple[tuple[str, float, float, float], ...]:
    """Decode the per-workload metric rows of one macro-metrics artifact."""
    if not isinstance(value, list):
        raise BenchmarkArtifactError(f"{label} holds no per-workload rows.", operation=operation)
    rows: list[tuple[str, float, float, float]] = []
    names: list[str] = []
    for item in cast("list[object]", value):
        if not isinstance(item, Mapping):
            raise BenchmarkArtifactError(
                f"{label} holds a {type(item).__name__} where a workload row belongs.",
                operation=operation,
            )
        row = cast("Mapping[str, object]", item)
        name = require_exact_str(
            row.get("workload"), kind="macro row workload", operation=operation
        )
        names.append(name)
        rows.append(
            (
                name,
                _require_number(
                    row.get("ndcg_at_10"), label=f"{label} {name} ndcg@10", operation=operation
                ),
                _require_number(
                    row.get("recall_at_10"), label=f"{label} {name} recall@10", operation=operation
                ),
                _require_number(
                    row.get("recall_at_100"),
                    label=f"{label} {name} recall@100",
                    operation=operation,
                ),
            )
        )
    if tuple(names) != tuple(sorted(RES138_WORKLOAD_NAMES)):
        raise BenchmarkArtifactError(
            f"{label} covers workloads {names}, not every frozen workload exactly once in the "
            "sorted order macro_across_workloads writes them.",
            operation=operation,
        )
    return tuple(rows)


@dataclass(frozen=True)
class SealedStageA:
    """A Stage A bundle that was loaded and verified against the seal in full.

    ``reference`` is what a Stage B qualification binds: the four digests and the
    shortlist, as
    :class:`~dynamisrag.benchmark.production.StageAReference` requires them.
    ``verification`` is the existing bundle verification report, kept so an operator
    can print what was checked rather than being told it happened.
    """

    root: Path
    seal: StageASeal
    verification: BundleVerification
    reference: StageAReference
    quality: tuple[StageAQuality, ...]
    plan_sha256: str
    generation_semantics_sha256: str

    def quality_for(self, model_id: str, dimension: int) -> StageAQuality:
        """The Stage A macro metrics for one shortlisted configuration."""
        for item in self.quality:
            if item.model_id == model_id and item.dimension == dimension:
                return item
        raise BenchmarkContractError(
            f"the sealed Stage A result holds no quality evidence for "
            f"{model_id}@{dimension}, which the Stage B shortlist requires. A "
            "shortlisted configuration without macro metrics "
            "cannot enter the selection table.",
            operation="sealed_stage_a",
            model_id=model_id,
        )

    @property
    def calibration_items(self) -> tuple[dict[str, object], ...]:
        """The Stage A calibration set's item identities, read from the sealed preflight.

        Read on demand rather than stored, so there is one reader for one artifact and a
        second copy cannot drift from the file the sealed bundle actually holds.
        """
        from dynamisrag.benchmark.gpu_evidence import stage_a_calibration_items

        return stage_a_calibration_items(self.root, operation="sealed_stage_a")

    @property
    def labels(self) -> tuple[str, ...]:
        """``model@dimension`` for each shortlisted configuration, in shortlist order."""
        return tuple(f"{model_id}@{dimension}" for model_id, dimension in self.reference.candidates)

    def payload(self) -> Mapping[str, object]:
        """The hashed description of the loaded reference."""
        return {
            "stage": "reference-quality",
            "root": str(self.root),
            "seal": dict(self.seal.payload()),
            "reference": dict(self.reference.payload()),
            "bundle_verification_sha256": self.verification.sha256,
            "bundle_file_count": self.verification.file_count,
            "bundle_shard_count": self.verification.shard_count,
            "bundle_row_count": self.verification.row_count,
            "plan_sha256": self.plan_sha256,
            "generation_semantics_sha256": self.generation_semantics_sha256,
            "quality": [dict(item.payload()) for item in self.quality],
        }


def _require_bundle_digest(path: Path, *, expected: str, label: str, operation: str) -> str:
    observed = file_sha256(path)
    if observed != expected:
        raise BenchmarkArtifactError(
            f"the Stage A {label} at {path.name} hashed {observed}, not the sealed "
            f"{expected}. Stage B is bound to one Stage A result; a bundle that is not "
            "it cannot be a reference for a production qualification, however similar "
            "it looks.",
            operation=operation,
            expected=expected,
            observed=observed,
        )
    return observed


def _require_stage_a_completion(payload: Mapping[str, object], *, operation: str) -> None:
    """Require the sealed full-run summary to be a completed Stage A quality result.

    The two states it checks are what make Stage B necessary rather than
    redundant: Stage A must have finished its quality evidence, and it must not have
    already run production qualification or configured a production default. A
    bundle whose summary claims a configured production default describes a
    deployment decision, not a reference, and Stage B would be deciding after the
    fact.
    """
    production = payload.get("production_qualification")
    if (
        not isinstance(production, Mapping)
        or cast("Mapping[str, object]", production).get("status") != "not_run"
    ):
        raise BenchmarkArtifactError(
            "the sealed Stage A full-run summary does not declare production_qualification "
            f"status='not_run' (observed {production!r}). Stage B is the stage that runs "
            "production qualification; a bundle that already claims it is not the Stage A "
            "reference Stage B is defined against.",
            operation=operation,
        )
    default = payload.get("production_default")
    if (
        not isinstance(default, Mapping)
        or cast("Mapping[str, object]", default).get("status") != "not_configured"
    ):
        raise BenchmarkArtifactError(
            "the sealed Stage A full-run summary does not declare production_default "
            f"status='not_configured' (observed {default!r}). No production default may be "
            "configured until the Stage B selection evidence is complete, so a bundle that already "
            "names one is not this stage's reference.",
            operation=operation,
        )


def _require_input_policy(payload: Mapping[str, object], *, operation: str) -> None:
    """Require the sealed preflight to declare the frozen Stage A input policy exactly."""
    declared = payload.get("input_policy")
    if canonical_json(cast("object", declared)) != canonical_json(input_policy_payload()):
        raise BenchmarkArtifactError(
            "the sealed Stage A preflight does not declare the frozen input policy "
            f"({input_policy_payload()}). Stage B reproduces Stage A's semantic input policy "
            "exactly, so a reference produced under a refusal, a left truncation, a "
            "longer boundary or no declared boundary is not the reference Stage B qualifies.",
            operation=operation,
        )


def _require_model_revisions(
    manifest: Res138RunManifest, seal: StageASeal, *, operation: str
) -> None:
    """Require the bundle's recorded model revisions to be the sealed, frozen ones."""
    observed = dict(manifest.model_revisions)
    expected = dict(seal.model_revisions)
    if observed != expected:
        differing = sorted(
            model_id
            for model_id in set(observed) | set(expected)
            if observed.get(model_id) != expected.get(model_id)
        )
        raise BenchmarkArtifactError(
            f"the Stage A bundle records candidate revisions that differ from the seal for "
            f"{differing}. Stage B qualifies vectors produced at the sealed revisions; a bundle "
            "recorded under another revision is a different experiment.",
            operation=operation,
            expected=str(sorted(expected.items()))[:64],
            observed=str(sorted(observed.items()))[:64],
        )


def _require_source_digests(
    manifest: Res138RunManifest, seal: StageASeal, *, operation: str
) -> None:
    """Require the bundle's dataset digests to be the sealed, frozen BEIR digests."""
    observed = tuple(sorted(manifest.dataset_digests))
    if observed != seal.source_digests:
        raise BenchmarkArtifactError(
            "the Stage A bundle records source digests that are not the frozen BEIR workloads the "
            "seal declares. Stage B's ANN recall is measured against Stage A's exact rankings for "
            "these corpora, so a bundle measured over other corpora is not a comparable reference.",
            operation=operation,
            expected=str(list(seal.source_digests))[:64],
            observed=str(list(observed))[:64],
        )


def _require_generation_semantics(manifest: Res138RunManifest, *, operation: str) -> str:
    """Require the bundle's generation semantics to be this tree's frozen semantics.

    The recorded digest names the exact ``(model, side, dimension)`` generation
    configurations — normalisation, truncation, direction, prompt, dimension — that
    produced the vectors. Recomputing it from this source tree and comparing is what
    makes "the reference is the function this repository measures" a statement that
    holds, rather than a digest that was copied along with the bundle.
    """
    expected = generation_semantics_sha256()
    if manifest.generation_semantics_sha256 != expected:
        raise BenchmarkArtifactError(
            "the Stage A bundle's generation semantics digest differs from the frozen semantics of "
            "this source tree. The vectors would have been produced under different normalisation, "
            "truncation, prompt or dimension semantics, so they are not the reference a Stage B "
            "equivalence gate can be measured against.",
            operation=operation,
            expected=expected,
            observed=manifest.generation_semantics_sha256,
        )
    return expected


def _load_quality(
    root: Path, shortlist: Sequence[tuple[str, int]], *, operation: str
) -> tuple[StageAQuality, ...]:
    """Read the sealed macro metrics for every shortlisted configuration."""
    frozen = {candidate.model_id: candidate for candidate in RES138_MODEL_CANDIDATES}
    quality: list[StageAQuality] = []
    for model_id, dimension in shortlist:
        candidate = frozen[model_id]
        relative = Path("results") / "macro" / _candidate_key(model_id) / f"{dimension}.json"
        path = root / relative
        if not path.exists():
            raise BenchmarkArtifactError(
                f"the Stage A bundle holds no macro metrics for {model_id}@{dimension} at "
                f"{relative.as_posix()}. A configuration on the Stage B shortlist without macro "
                "quality metrics cannot enter the selection table.",
                operation=operation,
                model_id=model_id,
            )
        envelope = read_artifact(path, name="macro_metrics")
        if envelope.sha256 != file_sha256(path):
            raise BenchmarkArtifactError(
                f"{relative.as_posix()} is not canonical JSON, so its recorded metrics cannot be "
                "the metrics the sealed run wrote.",
                operation=operation,
                model_id=model_id,
            )
        payload = envelope.payload
        if (
            payload.get("model_id") != candidate.model_id
            or payload.get("model_revision") != candidate.revision
            or payload.get("dimension") != dimension
        ):
            raise BenchmarkArtifactError(
                f"{relative.as_posix()} names a different candidate-configuration than the path "
                "it occupies.",
                operation=operation,
                model_id=model_id,
            )
        metrics = payload.get("metrics")
        if not isinstance(metrics, Mapping):
            raise BenchmarkArtifactError(
                f"{relative.as_posix()} carries no metrics object, so the macro quality evidence "
                "this configuration contributes cannot be read.",
                operation=operation,
                model_id=model_id,
            )
        macro = cast("Mapping[str, object]", metrics)
        quality.append(
            StageAQuality(
                model_id=candidate.model_id,
                model_revision=candidate.revision,
                dimension=dimension,
                workload_count=_require_count(
                    macro.get("workload_count"),
                    label=f"{relative.as_posix()} workload_count",
                    operation=operation,
                ),
                queries_scored=_require_count(
                    macro.get("queries_scored"),
                    label=f"{relative.as_posix()} queries_scored",
                    operation=operation,
                ),
                ndcg_at_10=_require_number(
                    macro.get("ndcg_at_10"),
                    label=f"{relative.as_posix()} ndcg_at_10",
                    operation=operation,
                ),
                recall_at_10=_require_number(
                    macro.get("recall_at_10"),
                    label=f"{relative.as_posix()} recall_at_10",
                    operation=operation,
                ),
                recall_at_100=_require_number(
                    macro.get("recall_at_100"),
                    label=f"{relative.as_posix()} recall_at_100",
                    operation=operation,
                ),
                workloads=_require_rows(
                    macro.get("workloads"),
                    label=f"{relative.as_posix()} workloads",
                    operation=operation,
                ),
            )
        )
    return tuple(quality)


def load_sealed_stage_a(
    root: Path,
    *,
    seal: StageASeal = RES138_STAGE_A_SEAL,
    operation: str = "load_sealed_stage_a",
) -> SealedStageA:
    """Load the sealed Stage A bundle, or refuse it.

    The order is deliberate. The existing bundle verification runs first and is the
    expensive, thorough one: it re-hashes every declared file, checks the shard
    graph and canonical order, re-runs exact retrieval from the persisted matrices,
    recomputes every metric and every bootstrap estimate, and reconciles the
    reference execution timings. Only once the bundle is internally complete does
    the *external* identity check run — the digests, the commit, the revisions, the
    input policy, the source digests and the two Stage A completion states.

    That order matters because the two layers answer different questions. Internal
    completeness says the bundle is a whole Stage A run; external identity says it
    is *this* Stage A run. A bundle can satisfy the first and fail the second, and
    only the second makes it the reference Stage B is allowed to qualify against.
    """
    if not root.is_dir():
        raise BenchmarkArtifactError(
            f"the Stage A bundle root {root} is not a directory.",
            operation=operation,
        )
    verification = verify_run_bundle(root, expect_code_sha=seal.code_sha, operation=operation)
    if verification.code_sha != seal.code_sha:
        raise BenchmarkArtifactError(
            f"the Stage A bundle was produced by commit {verification.code_sha}, not the sealed "
            f"{seal.code_sha}. Evidence for one commit is evidence for that commit only.",
            operation=operation,
            expected=seal.code_sha,
            observed=verification.code_sha,
        )
    _require_bundle_digest(
        root / "bundle-manifest.json",
        expected=seal.bundle_sha256,
        label="bundle manifest",
        operation=operation,
    )
    _require_bundle_digest(
        root / "full-run.json",
        expected=seal.full_run_sha256,
        label="full-run summary",
        operation=operation,
    )

    manifest = Res138RunManifest.read(root / "run-manifest.json")
    _require_model_revisions(manifest, seal, operation=operation)
    _require_source_digests(manifest, seal, operation=operation)
    semantics = _require_generation_semantics(manifest, operation=operation)

    plan = read_artifact(root / "benchmark-plan.json", name="plan")
    if (
        plan.sha256 != manifest.plan_sha256
        or file_sha256(root / "benchmark-plan.json") != plan.sha256
    ):
        raise BenchmarkArtifactError(
            "the Stage A bundle's plan digest does not match its run manifest, or the plan file is "
            "not canonical JSON.",
            operation=operation,
            expected=manifest.plan_sha256,
            observed=file_sha256(root / "benchmark-plan.json"),
        )
    preflight = read_artifact(root / "preflight.json", name="preflight")
    _require_input_policy(preflight.payload, operation=operation)
    full_run = read_artifact(root / "full-run.json", name="full_run")
    _require_stage_a_completion(full_run.payload, operation=operation)

    shortlist = require_stage_b_shortlist(seal.shortlist, operation=operation)
    quality = _load_quality(root, shortlist, operation=operation)

    reference = StageAReference(
        bundle_sha256=seal.bundle_sha256,
        full_run_sha256=seal.full_run_sha256,
        plan_sha256=manifest.plan_sha256,
        generation_semantics_sha256=semantics,
        input_policy_revision=seal.input_policy_revision,
        input_max_tokens=seal.input_max_tokens,
        candidates=shortlist,
    )
    return SealedStageA(
        root=root,
        seal=seal,
        verification=verification,
        reference=reference,
        quality=quality,
        plan_sha256=manifest.plan_sha256,
        generation_semantics_sha256=semantics,
    )
