"""The Stage-B GPU preflight manifest: the digest that authorizes the full corpus pass.

Spending an A100 hour on the full corpus before the equivalence gate is known to
hold would be measuring throughput for a configuration that may be disqualified.
So the GPU lane has two modes and this artifact is the join between them:

* **preflight** verifies the sealed Stage A reference, the Stage-B plan, the
  deployment floor, ``/health``, ``/info`` and both dimension request paths, then
  re-embeds only the frozen Stage A calibration set and writes one artifact per
  dimension carrying ``metrics = null``. The workstation verifies those artifacts
  locally with ``verify-gpu-evidence`` — the numerical and query-to-document
  ranking equivalence gate is recomputed from bytes there — and this manifest is
  the single deterministic digest covering both dimensions.
* **full** refuses to measure the production corpus without the approved
  manifest digest, and refuses unless the Stage-B plan, the model revision, the
  TEI server identity, the precision, the backend and the input semantics are all
  identical to the ones the preflight proved. Nothing about the preflight is
  re-derived from memory: the manifest is re-read and compared field by field.

The manifest is deterministic by construction: it carries digests, dimensions and
the semantics under test, and it carries no clock, no hostname and no path outside
the artifact directory. Two preflights of the same configuration produce the same
digest, which is what makes "approved SHA" meaningful.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from dynamisrag.benchmark.contracts import require_candidate_dimension, require_exact_str
from dynamisrag.benchmark.errors import (
    BenchmarkArtifactError,
    BenchmarkContractError,
    BenchmarkPreflightError,
)
from dynamisrag.benchmark.stage_b import StageBPlan
from dynamisrag.benchmark.tei_server import TeiServerInfo
from dynamisrag.embedding.contracts import canonical_json

__all__ = [
    "RES138_GPU_PREFLIGHT_FILENAME",
    "RES138_GPU_PREFLIGHT_REVISION",
    "GpuPreflightDimension",
    "GpuPreflightManifest",
    "read_gpu_preflight",
    "require_approved_preflight_digest",
]

RES138_GPU_PREFLIGHT_REVISION: Final[str] = "res138-gpu-preflight-v1"
"""Revision of the GPU preflight manifest."""

RES138_GPU_PREFLIGHT_FILENAME: Final[str] = "gpu-preflight.json"
"""The name the manifest is written under inside the evidence directory."""


def _require_sha256(value: object, *, label: str, operation: str) -> str:
    text = require_exact_str(value, kind=label, operation=operation)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise BenchmarkArtifactError(
            f"the GPU preflight manifest {label} is {text!r}, which is not 64 lowercase "
            "hexadecimal characters.",
            operation=operation,
        )
    return text


def _require_bare_npy(value: object, *, label: str, operation: str) -> str:
    name = require_exact_str(value, kind=label, operation=operation)
    if Path(name).name != name or not name.endswith(".npy"):
        raise BenchmarkArtifactError(
            f"the GPU preflight manifest {label} is {name!r}, which is not a bare .npy file name. "
            "The vector file travels beside the artifact, not at an arbitrary path.",
            operation=operation,
        )
    return name


def _require_bare_json(value: object, *, label: str, operation: str) -> str:
    name = require_exact_str(value, kind=label, operation=operation)
    if Path(name).name != name or not name.endswith(".json"):
        raise BenchmarkArtifactError(
            f"the GPU preflight manifest {label} is {name!r}, which is not a bare .json file name.",
            operation=operation,
        )
    return name


@dataclass(frozen=True)
class GpuPreflightDimension:
    """One dimension's preflight artifact and the vectors it binds, by digest."""

    dimension: int
    evidence_file: str
    evidence_sha256: str
    vector_file: str
    vector_sha256: str
    rows: int

    def __post_init__(self) -> None:
        require_candidate_dimension(self.dimension, operation="gpu_preflight_dimension")
        if isinstance(self.rows, bool) or self.rows < 1:
            raise BenchmarkContractError(
                f"the GPU preflight dimension {self.dimension} records {self.rows!r} rows, which "
                "is not a positive integer count.",
                operation="gpu_preflight_dimension",
            )
        _require_bare_json(
            self.evidence_file, label="evidence file", operation="gpu_preflight_dimension"
        )
        _require_bare_npy(
            self.vector_file, label="vector file", operation="gpu_preflight_dimension"
        )
        _require_sha256(
            self.evidence_sha256, label="evidence digest", operation="gpu_preflight_dimension"
        )
        _require_sha256(
            self.vector_sha256, label="vector digest", operation="gpu_preflight_dimension"
        )

    def payload(self) -> dict[str, object]:
        """The hashed description of this dimension's preflight evidence."""
        return {
            "dimension": self.dimension,
            "evidence_file": self.evidence_file,
            "evidence_sha256": self.evidence_sha256,
            "vector_file": self.vector_file,
            "vector_sha256": self.vector_sha256,
            "rows": self.rows,
        }


@dataclass(frozen=True)
class GpuPreflightManifest:
    """The deterministic digest of one GPU preflight, over both dimensions.

    The fields are exactly the ones full mode must re-prove: the Stage-B plan
    digest (which binds the code commit), the serving identity digest, the model
    and its revision, the declared precision and backend, and one record per
    dimension covering the artifact and vector digests the local workstation
    verified.
    """

    stage_b_plan_sha256: str
    tei_server_sha256: str
    model_id: str
    model_revision: str
    precision: str
    backend: str
    dimensions: tuple[GpuPreflightDimension, ...]

    def __post_init__(self) -> None:
        _require_sha256(
            self.stage_b_plan_sha256, label="Stage B plan digest", operation="gpu_preflight"
        )
        _require_sha256(
            self.tei_server_sha256, label="TEI server digest", operation="gpu_preflight"
        )
        require_exact_str(self.model_id, kind="preflight model id", operation="gpu_preflight")
        require_exact_str(
            self.model_revision, kind="preflight model revision", operation="gpu_preflight"
        )
        require_exact_str(self.precision, kind="preflight precision", operation="gpu_preflight")
        require_exact_str(self.backend, kind="preflight backend", operation="gpu_preflight")
        if not self.dimensions:
            raise BenchmarkContractError(
                "a GPU preflight manifest must cover at least one dimension.",
                operation="gpu_preflight",
            )
        covered = [record.dimension for record in self.dimensions]
        if len(set(covered)) != len(covered):
            raise BenchmarkContractError(
                f"the GPU preflight manifest repeats a dimension ({covered}). One dimension has "
                "one artifact, and two records for it would describe two runs.",
                operation="gpu_preflight",
            )

    def record_for(self, dimension: int) -> GpuPreflightDimension:
        """The preflight record for one dimension, or a refusal naming it."""
        for record in self.dimensions:
            if record.dimension == dimension:
                return record
        raise BenchmarkArtifactError(
            f"the GPU preflight manifest covers dimensions "
            f"{[record.dimension for record in self.dimensions]}, not {dimension}.",
            operation="gpu_preflight",
        )

    def payload(self) -> dict[str, object]:
        """The hashed description of this preflight."""
        return {
            "artifact_revision": RES138_GPU_PREFLIGHT_REVISION,
            "stage_b_plan_sha256": self.stage_b_plan_sha256,
            "tei_server_sha256": self.tei_server_sha256,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "precision": self.precision,
            "backend": self.backend,
            "dimensions": [record.payload() for record in self.dimensions],
        }

    @property
    def sha256(self) -> str:
        """SHA-256 over the canonical preflight payload."""
        return hashlib.sha256(canonical_json(self.payload()).encode("utf-8")).hexdigest()

    def envelope(self) -> dict[str, object]:
        """The payload with its own digest, as written to disk."""
        payload = self.payload()
        payload["preflight_sha256"] = self.sha256
        return payload

    def write(self, path: Path) -> str:
        """Write the manifest as canonical JSON and return its digest."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.tmp")
        temporary.write_bytes(canonical_json(self.envelope()).encode("utf-8"))
        temporary.replace(path)
        return self.sha256

    def require_matches(
        self,
        *,
        plan: StageBPlan,
        server_info: TeiServerInfo,
        precision: str,
        backend: str,
        operation: str,
    ) -> None:
        """Require the current run to be the one this preflight authorized.

        Every comparison is a refusal naming the condition that changed. The plan
        digest carries the code commit, so a different Stage-B implementation
        cannot inherit this authorization even if it types the same precision.
        """
        if self.stage_b_plan_sha256 != plan.sha256:
            raise BenchmarkContractError(
                f"the approved GPU preflight was produced under Stage B plan "
                f"{self.stage_b_plan_sha256}, not this plan {plan.sha256}. Because the plan binds "
                "the code commit, this also refuses evidence produced by a different Stage-B "
                "implementation.",
                operation=operation,
                expected=plan.sha256,
                observed=self.stage_b_plan_sha256,
            )
        if self.tei_server_sha256 != server_info.sha256:
            raise BenchmarkContractError(
                "the TEI server identity changed after the preflight "
                f"({self.tei_server_sha256} vs {server_info.sha256}). The preflight proved "
                "equivalence against one served model, dtype and boundary; a changed server is a "
                "different function and the full pass would measure a configuration nobody "
                "qualified.",
                operation=operation,
                expected=self.tei_server_sha256,
                observed=server_info.sha256,
            )
        if self.model_id != plan.model_ids[0] or self.model_revision != plan.model_revision:
            raise BenchmarkContractError(
                f"the approved GPU preflight binds {self.model_id}@{self.model_revision}, not the "
                f"plan's {plan.model_ids[0]}@{plan.model_revision}.",
                operation=operation,
            )
        if self.precision != precision:
            raise BenchmarkContractError(
                f"the approved GPU preflight declares precision {self.precision!r}, not this run's "
                f"{precision!r}. The production precision is chosen before launch; it is never "
                "changed after a benchmark has been seen.",
                operation=operation,
                expected=self.precision,
                observed=precision,
            )
        if self.backend != backend:
            raise BenchmarkContractError(
                f"the approved GPU preflight declares backend {self.backend!r}, not this run's "
                f"{backend!r}.",
                operation=operation,
                expected=self.backend,
                observed=backend,
            )
        covered = tuple(record.dimension for record in self.dimensions)
        if covered != plan.dimensions:
            raise BenchmarkContractError(
                f"the approved GPU preflight covers dimensions {list(covered)}, not the plan's "
                f"{list(plan.dimensions)}. Full mode measures exactly the dimensions the preflight "
                "proved, and no more.",
                operation=operation,
            )


def _rows(value: object, *, operation: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list) or not value:
        raise BenchmarkArtifactError(
            "the GPU preflight manifest records no dimensions.", operation=operation
        )
    rows: list[Mapping[str, object]] = []
    for item in cast("list[object]", value):
        if not isinstance(item, Mapping):
            raise BenchmarkArtifactError(
                "a GPU preflight dimension record is not an object.", operation=operation
            )
        rows.append(cast("Mapping[str, object]", item))
    return tuple(rows)


def read_gpu_preflight(
    path: Path, *, operation: str = "read_gpu_preflight"
) -> GpuPreflightManifest:
    """Read a written preflight manifest, refusing anything that does not rebuild.

    The manifest is rebuilt from its own records and compared canonically, so a
    hand-edited plan digest, server digest or vector digest cannot authorize a
    full run.
    """
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise BenchmarkArtifactError(
            f"the GPU preflight manifest at {path.name} could not be read "
            f"({type(error).__name__}).",
            operation=operation,
        ) from None
    except ValueError as error:
        raise BenchmarkArtifactError(
            f"the GPU preflight manifest at {path.name} is not valid JSON ({error}).",
            operation=operation,
        ) from None
    if not isinstance(decoded, Mapping):
        raise BenchmarkArtifactError(
            "the GPU preflight manifest is not an object.", operation=operation
        )
    raw = dict(cast("Mapping[str, object]", decoded))
    declared_digest = _require_sha256(
        raw.pop("preflight_sha256", None), label="preflight digest", operation=operation
    )
    if raw.get("artifact_revision") != RES138_GPU_PREFLIGHT_REVISION:
        raise BenchmarkArtifactError(
            f"the GPU preflight manifest declares revision {raw.get('artifact_revision')!r}, not "
            f"{RES138_GPU_PREFLIGHT_REVISION!r}.",
            operation=operation,
        )
    dimensions = tuple(
        GpuPreflightDimension(
            dimension=cast("int", row.get("dimension")),
            evidence_file=require_exact_str(
                row.get("evidence_file"), kind="preflight evidence file", operation=operation
            ),
            evidence_sha256=_require_sha256(
                row.get("evidence_sha256"),
                label="preflight evidence digest",
                operation=operation,
            ),
            vector_file=require_exact_str(
                row.get("vector_file"), kind="preflight vector file", operation=operation
            ),
            vector_sha256=_require_sha256(
                row.get("vector_sha256"), label="preflight vector digest", operation=operation
            ),
            rows=cast("int", row.get("rows")),
        )
        for row in _rows(raw.get("dimensions"), operation=operation)
    )
    rebuilt = GpuPreflightManifest(
        stage_b_plan_sha256=_require_sha256(
            raw.get("stage_b_plan_sha256"), label="preflight plan digest", operation=operation
        ),
        tei_server_sha256=_require_sha256(
            raw.get("tei_server_sha256"), label="preflight server digest", operation=operation
        ),
        model_id=require_exact_str(
            raw.get("model_id"), kind="preflight model id", operation=operation
        ),
        model_revision=require_exact_str(
            raw.get("model_revision"), kind="preflight model revision", operation=operation
        ),
        precision=require_exact_str(
            raw.get("precision"), kind="preflight precision", operation=operation
        ),
        backend=require_exact_str(
            raw.get("backend"), kind="preflight backend", operation=operation
        ),
        dimensions=dimensions,
    )
    if rebuilt.sha256 != declared_digest:
        raise BenchmarkArtifactError(
            "the GPU preflight manifest's declared digest does not match its own records. A "
            "manifest that does not rebuild is not the one an operator approved.",
            operation=operation,
            expected=rebuilt.sha256,
            observed=declared_digest,
        )
    if canonical_json(rebuilt.payload()) != canonical_json(raw):
        raise BenchmarkArtifactError(
            "the GPU preflight manifest differs from its own records: a field was changed, removed "
            "or re-typed after it was written.",
            operation=operation,
        )
    return rebuilt


def require_approved_preflight_digest(approved: str, *, manifest: GpuPreflightManifest) -> None:
    """Require the operator's approved digest to be this manifest's digest."""
    approved_text = require_exact_str(
        approved, kind="approved GPU preflight digest", operation="require_approved_preflight"
    )
    if approved_text != manifest.sha256:
        raise BenchmarkPreflightError(
            f"the approved GPU preflight digest {approved_text!r} is not the manifest's "
            f"{manifest.sha256!r}. Full mode runs only the preflight an operator approved.",
            operation="require_approved_preflight",
            expected=manifest.sha256,
            observed=approved_text,
        )
