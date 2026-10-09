"""Offline score and compare workflows for sealed RES-140 run inputs."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

from dynamisrag.ir.artifacts import (
    IrBundleReceipt,
    read_ir_inputs,
    read_verified_ir_evaluation,
    write_ir_bundle,
)
from dynamisrag.ir.contracts import IrContractError, canonical_ir_json
from dynamisrag.ir.scoring import IrEvaluation

__all__ = [
    "IR_COMPARISON_REVISION",
    "IrAggregateDelta",
    "IrComparison",
    "IrComparisonReceipt",
    "IrQueryDelta",
    "compare_ir_bundles",
    "compare_ir_evaluations",
    "score_ir_inputs",
    "verify_ir_comparison",
    "write_ir_comparison",
]

IR_COMPARISON_REVISION: Final[str] = "ir-comparison-v1"
_COMPARISON_BUNDLE_REVISION: Final[str] = "ir-comparison-bundle-v1"
_PER_QUERY_DELTA_REVISION: Final[str] = "ir-query-delta-v1"
_AGGREGATE_DELTA_REVISION: Final[str] = "ir-aggregate-delta-v1"
_PER_QUERY_METRICS: Final[tuple[tuple[str, str], ...]] = (
    ("nDCG@10", "ndcg_at_10"),
    ("Recall@10", "recall_at_10"),
    ("MAP", "map"),
    ("MRR", "mrr"),
)
_DIFF_FILES: Final[tuple[str, ...]] = (
    "comparison.json",
    "per-query-delta.json",
    "per-query-delta.parquet",
    "aggregate-delta.json",
    "aggregate-delta.parquet",
)


@dataclass(frozen=True)
class IrQueryDelta:
    query_id: str
    measure: str
    baseline: float
    candidate: float
    difference: float

    def payload(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "measure": self.measure,
            "baseline": self.baseline,
            "candidate": self.candidate,
            "difference": self.difference,
        }


@dataclass(frozen=True)
class IrAggregateDelta:
    measure: str
    baseline: float
    candidate: float
    difference: float
    query_denominator: int

    def payload(self) -> dict[str, object]:
        return {
            "measure": self.measure,
            "baseline": self.baseline,
            "candidate": self.candidate,
            "difference": self.difference,
            "query_denominator": self.query_denominator,
        }


@dataclass(frozen=True)
class IrComparison:
    baseline_evaluation_sha256: str
    candidate_evaluation_sha256: str
    baseline_run_sha256: str
    candidate_run_sha256: str
    dataset_sha256: str
    projection_sha256: str
    passage_mapping_sha256: str | None
    evaluation_depth: int
    metric_policy_sha256: str
    scoring_engine: tuple[tuple[str, str], ...]
    per_query: tuple[IrQueryDelta, ...]
    aggregate: tuple[IrAggregateDelta, ...]

    def payload(self) -> dict[str, object]:
        return {
            "revision": IR_COMPARISON_REVISION,
            "baseline_evaluation_sha256": self.baseline_evaluation_sha256,
            "candidate_evaluation_sha256": self.candidate_evaluation_sha256,
            "baseline_run_sha256": self.baseline_run_sha256,
            "candidate_run_sha256": self.candidate_run_sha256,
            "dataset_sha256": self.dataset_sha256,
            "projection_sha256": self.projection_sha256,
            "passage_mapping_sha256": self.passage_mapping_sha256,
            "evaluation_depth": self.evaluation_depth,
            "metric_policy_sha256": self.metric_policy_sha256,
            "scoring_engine": dict(self.scoring_engine),
            "per_query_delta": [row.payload() for row in self.per_query],
            "aggregate_delta": [row.payload() for row in self.aggregate],
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_ir_json(self.payload())).hexdigest()


@dataclass(frozen=True)
class IrComparisonReceipt:
    root: Path
    manifest_sha256: str
    comparison_sha256: str


def score_ir_inputs(
    inputs_root: Path,
    destination: Path,
    *,
    expected_run_sha256: str,
) -> IrBundleReceipt:
    """Score canonical run JSON offline and publish a complete result bundle."""
    inputs = read_ir_inputs(inputs_root, expected_run_sha256=expected_run_sha256)
    return write_ir_bundle(
        destination,
        dataset=inputs.dataset,
        config=inputs.config,
        run=inputs.run,
        passage_mapping=inputs.passage_mapping,
    )


def _comparable(baseline: IrEvaluation, candidate: IrEvaluation) -> None:
    for label, first, second in (
        ("corpus, queries or judgments", baseline.dataset_sha256, candidate.dataset_sha256),
        ("index snapshot", baseline.projection_sha256, candidate.projection_sha256),
        ("passage mapping", baseline.passage_mapping_sha256, candidate.passage_mapping_sha256),
        ("evaluation depth", baseline.evaluation_depth, candidate.evaluation_depth),
        ("metric policy", baseline.metric_policy_sha256, candidate.metric_policy_sha256),
        ("scoring engine", baseline.scoring_engine, candidate.scoring_engine),
    ):
        if first != second:
            raise IrContractError(f"candidate runs are not comparable: {label} differs")


def compare_ir_evaluations(baseline: IrEvaluation, candidate: IrEvaluation) -> IrComparison:
    """Compute deltas only where every scientific evaluation boundary matches."""
    _comparable(baseline, candidate)
    baseline_queries = {row.query_id: row for row in baseline.per_query}
    candidate_queries = {row.query_id: row for row in candidate.per_query}
    if tuple(baseline_queries) != tuple(candidate_queries):
        raise IrContractError("candidate runs are not comparable: query rows differ")
    per_query = tuple(
        IrQueryDelta(
            query_id=base.query_id,
            measure=measure,
            baseline=float(getattr(base, attribute)),
            candidate=float(getattr(candidate_queries[base.query_id], attribute)),
            difference=float(getattr(candidate_queries[base.query_id], attribute))
            - float(getattr(base, attribute)),
        )
        for base in baseline.per_query
        for measure, attribute in _PER_QUERY_METRICS
    )
    baseline_aggregate = {row.measure: row for row in baseline.aggregate}
    candidate_aggregate = {row.measure: row for row in candidate.aggregate}
    if tuple(baseline_aggregate) != tuple(candidate_aggregate):
        raise IrContractError("candidate runs are not comparable: aggregate measures differ")
    aggregate = tuple(
        IrAggregateDelta(
            measure=row.measure,
            baseline=row.value,
            candidate=candidate_aggregate[row.measure].value,
            difference=candidate_aggregate[row.measure].value - row.value,
            query_denominator=row.query_denominator,
        )
        for row in baseline.aggregate
    )
    return IrComparison(
        baseline_evaluation_sha256=baseline.sha256,
        candidate_evaluation_sha256=candidate.sha256,
        baseline_run_sha256=baseline.run_sha256,
        candidate_run_sha256=candidate.run_sha256,
        dataset_sha256=baseline.dataset_sha256,
        projection_sha256=baseline.projection_sha256,
        passage_mapping_sha256=baseline.passage_mapping_sha256,
        evaluation_depth=baseline.evaluation_depth,
        metric_policy_sha256=baseline.metric_policy_sha256,
        scoring_engine=baseline.scoring_engine,
        per_query=per_query,
        aggregate=aggregate,
    )


def _parquet(
    rows: list[dict[str, object]], *, revision: str, fields: tuple[tuple[str, str], ...]
) -> bytes:
    import pyarrow as pa
    import pyarrow.parquet as pq

    arrow = cast(Any, pa)
    parquet = cast(Any, pq)
    types = {"string": arrow.string(), "float64": arrow.float64(), "int64": arrow.int64()}
    schema = arrow.schema(
        [arrow.field(name, types[field_type], nullable=False) for name, field_type in fields],
        metadata={
            b"dynamisrag.schema_revision": revision.encode("ascii"),
            b"dynamisrag.pyarrow_version": str(arrow.__version__).encode("ascii"),
        },
    )
    table = arrow.Table.from_pylist(rows, schema=schema)
    sink = arrow.BufferOutputStream()
    parquet.write_table(
        table,
        sink,
        version="2.6",
        data_page_version="1.0",
        compression=None,
        use_dictionary=False,
        write_statistics=False,
        write_page_index=False,
        write_page_checksum=False,
        row_group_size=65_536,
    )
    return sink.getvalue().to_pybytes()


def _comparison_contents(comparison: IrComparison) -> dict[str, bytes]:
    per_query = [row.payload() for row in comparison.per_query]
    aggregate = [row.payload() for row in comparison.aggregate]
    return {
        "comparison.json": canonical_ir_json(comparison.payload()),
        "per-query-delta.json": canonical_ir_json(
            {"schema_revision": _PER_QUERY_DELTA_REVISION, "rows": per_query}
        ),
        "per-query-delta.parquet": _parquet(
            per_query,
            revision=_PER_QUERY_DELTA_REVISION,
            fields=(
                ("query_id", "string"),
                ("measure", "string"),
                ("baseline", "float64"),
                ("candidate", "float64"),
                ("difference", "float64"),
            ),
        ),
        "aggregate-delta.json": canonical_ir_json(
            {"schema_revision": _AGGREGATE_DELTA_REVISION, "rows": aggregate}
        ),
        "aggregate-delta.parquet": _parquet(
            aggregate,
            revision=_AGGREGATE_DELTA_REVISION,
            fields=(
                ("measure", "string"),
                ("baseline", "float64"),
                ("candidate", "float64"),
                ("difference", "float64"),
                ("query_denominator", "int64"),
            ),
        ),
    }


def _comparison_manifest(comparison: IrComparison, contents: dict[str, bytes]) -> bytes:
    return canonical_ir_json(
        {
            "revision": _COMPARISON_BUNDLE_REVISION,
            "comparison_sha256": comparison.sha256,
            "baseline_evaluation_sha256": comparison.baseline_evaluation_sha256,
            "candidate_evaluation_sha256": comparison.candidate_evaluation_sha256,
            "files": [
                {
                    "name": name,
                    "size_bytes": len(contents[name]),
                    "sha256": hashlib.sha256(contents[name]).hexdigest(),
                }
                for name in _DIFF_FILES
            ],
        }
    )


def write_ir_comparison(
    root: Path, *, baseline: IrEvaluation, candidate: IrEvaluation
) -> IrComparisonReceipt:
    comparison = compare_ir_evaluations(baseline, candidate)
    destination = root.resolve(strict=False)
    if destination.exists():
        raise IrContractError("refusing to overwrite an existing IR comparison")
    contents = _comparison_contents(comparison)
    manifest = _comparison_manifest(comparison, contents)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".ir-diff-stage-", dir=destination.parent))
    try:
        for name, data in contents.items():
            (stage / name).write_bytes(data)
        (stage / "manifest.json").write_bytes(manifest)
        if destination.exists():
            raise IrContractError("another process published this comparison already")
        stage.rename(destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return IrComparisonReceipt(destination, hashlib.sha256(manifest).hexdigest(), comparison.sha256)


def compare_ir_bundles(
    baseline_root: Path,
    candidate_root: Path,
    destination: Path,
    *,
    baseline_run_sha256: str,
    candidate_run_sha256: str,
) -> IrComparisonReceipt:
    baseline = read_verified_ir_evaluation(baseline_root, expected_run_sha256=baseline_run_sha256)
    candidate = read_verified_ir_evaluation(
        candidate_root, expected_run_sha256=candidate_run_sha256
    )
    return write_ir_comparison(destination, baseline=baseline, candidate=candidate)


def verify_ir_comparison(  # noqa: PLR0912, PLR0915 - keep each comparison integrity gate explicit
    root: Path,
    *,
    baseline: IrEvaluation,
    candidate: IrEvaluation,
    expected_comparison_sha256: str,
) -> IrComparisonReceipt:
    if not root.is_dir() or root.is_symlink():
        raise IrContractError("IR comparison root is not a regular directory")
    names = {path.name for path in root.iterdir()}
    if names != set(_DIFF_FILES) | {"manifest.json"}:
        raise IrContractError("IR comparison inventory differs from the closed contract")
    if any((root / name).is_symlink() or not (root / name).is_file() for name in names):
        raise IrContractError("IR comparison entries must be regular files")
    try:
        manifest_bytes = (root / "manifest.json").read_bytes()
        manifest: object = json.loads(manifest_bytes)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise IrContractError("IR comparison manifest cannot be read") from error
    data = cast("dict[str, Any]", manifest) if isinstance(manifest, dict) else {}
    if canonical_ir_json(data) != manifest_bytes or set(data) != {
        "revision",
        "comparison_sha256",
        "baseline_evaluation_sha256",
        "candidate_evaluation_sha256",
        "files",
    }:
        raise IrContractError("IR comparison manifest differs from the closed contract")
    comparison = compare_ir_evaluations(baseline, candidate)
    if data.get("revision") != _COMPARISON_BUNDLE_REVISION:
        raise IrContractError("IR comparison bundle revision is incompatible")
    if (
        data.get("comparison_sha256") != expected_comparison_sha256
        or comparison.sha256 != expected_comparison_sha256
    ):
        raise IrContractError("IR comparison identity does not match caller expectations")
    if (
        data.get("baseline_evaluation_sha256") != baseline.sha256
        or data.get("candidate_evaluation_sha256") != candidate.sha256
    ):
        raise IrContractError("IR comparison inputs differ from caller expectations")
    content: dict[str, bytes] = {}
    entries = data.get("files")
    if not isinstance(entries, list) or len(cast("list[object]", entries)) != len(_DIFF_FILES):
        raise IrContractError("IR comparison file inventory is invalid")
    for name, raw_entry in zip(_DIFF_FILES, cast("list[object]", entries), strict=True):
        entry = cast("dict[str, Any]", raw_entry) if isinstance(raw_entry, dict) else {}
        if set(entry) != {"name", "size_bytes", "sha256"} or entry.get("name") != name:
            raise IrContractError("IR comparison file inventory is invalid")
        size_bytes = entry.get("size_bytes")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int):
            raise IrContractError("IR comparison file size is invalid")
        payload = (root / name).read_bytes()
        if size_bytes != len(payload) or entry.get("sha256") != hashlib.sha256(payload).hexdigest():
            raise IrContractError("IR comparison file digest does not match its manifest")
        content[name] = payload
    if _comparison_contents(comparison) != content:
        raise IrContractError("IR comparison rows differ from independently recomputed deltas")
    for json_name, parquet_name, revision in (
        ("per-query-delta.json", "per-query-delta.parquet", _PER_QUERY_DELTA_REVISION),
        ("aggregate-delta.json", "aggregate-delta.parquet", _AGGREGATE_DELTA_REVISION),
    ):
        try:
            raw_table: object = json.loads(content[json_name])
        except (UnicodeDecodeError, ValueError) as error:
            raise IrContractError("IR comparison JSON table is invalid") from error
        if not isinstance(raw_table, dict):
            raise IrContractError("IR comparison JSON table is not an object")
        table = cast("dict[str, object]", raw_table)
        rows = table.get("rows")
        if set(table) != {"schema_revision", "rows"} or table.get("schema_revision") != revision:
            raise IrContractError("IR comparison JSON table schema is incompatible")
        if not isinstance(rows, list):
            raise IrContractError("IR comparison JSON table rows are invalid")
        if _read_diff_parquet(content[parquet_name], revision) != cast("list[object]", rows):
            raise IrContractError("IR comparison JSON and Parquet rows differ")
    return IrComparisonReceipt(
        root,
        hashlib.sha256(manifest_bytes).hexdigest(),
        comparison.sha256,
    )


def _read_diff_parquet(content: bytes, revision: str) -> list[dict[str, object]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    arrow = cast(Any, pa)
    parquet = cast(Any, pq)
    try:
        table = parquet.read_table(arrow.BufferReader(content))
    except Exception as error:
        raise IrContractError("IR comparison Parquet file is invalid") from error
    metadata = cast("dict[bytes, bytes]", table.schema.metadata or {})
    if metadata.get(b"dynamisrag.schema_revision") != revision.encode("ascii"):
        raise IrContractError("IR comparison Parquet schema revision is incompatible")
    return cast("list[dict[str, object]]", table.to_pylist())
