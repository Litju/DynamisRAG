"""Atomic, inspectable RES-140 IR JSON/TREC bundles with integrity verification.

An evaluation bundle contains original graded judgments, explicit ranked hits,
the immutable evaluation dataset, and a full semantic experiment fingerprint.
No service or model runtime is contacted during writing or verification.

Machine-local paths and timestamps are excluded from manifest content and digests.
The staging directory is unique but not part of a persisted artifact.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

from dynamisrag.ir.contracts import (
    IR_CONTRACT_REVISION,
    IrContractError,
    IrDataset,
    IrExperimentConfig,
    IrHit,
    IrMetricPolicy,
    IrPassageMapEntry,
    IrPassageMapping,
    IrQrel,
    IrQuery,
    IrRun,
    canonical_ir_json,
    trec_qrels,
    trec_run,
)

if TYPE_CHECKING:
    from dynamisrag.ir.scoring import IrEvaluation

__all__ = [
    "IR_BUNDLE_REVISION",
    "IrBundleReceipt",
    "IrRunInputs",
    "read_ir_inputs",
    "read_verified_ir_evaluation",
    "verify_ir_bundle",
    "write_ir_bundle",
]

IR_BUNDLE_REVISION: Final[str] = "ir-bundle-v2"
_MANIFEST_NAME: Final[str] = "manifest.json"
_PAYLOAD_NAMES: Final[tuple[str, ...]] = (
    "config.json",
    "dataset.json",
    "passage-mapping.json",
    "qrels.trec",
    "run.json",
    "run.trec",
    "evaluation.json",
    "per-query.json",
    "per-query.parquet",
    "aggregate.json",
    "aggregate.parquet",
    "ranked-run.json",
    "ranked-run.parquet",
)
_QUERY_SCHEMA_REVISION: Final[str] = "ir-per-query-v1"
_AGGREGATE_SCHEMA_REVISION: Final[str] = "ir-aggregate-v1"
_RUN_SCHEMA_REVISION: Final[str] = "ir-ranked-run-v1"
_INPUT_NAMES: Final[tuple[str, ...]] = (
    "config.json",
    "dataset.json",
    "passage-mapping.json",
    "run.json",
)


@dataclass(frozen=True)
class IrBundleReceipt:
    """The verified hashes and destination; path itself is never an identity."""

    root: Path
    manifest_sha256: str
    dataset_sha256: str
    config_sha256: str
    run_sha256: str
    evaluation_sha256: str
    passage_mapping_sha256: str


@dataclass(frozen=True)
class IrRunInputs:
    """Validated canonical JSON inputs for an offline score run."""

    dataset: IrDataset
    config: IrExperimentConfig
    run: IrRun
    passage_mapping: IrPassageMapping


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _table_json(revision: str, rows: list[dict[str, object]]) -> bytes:
    return canonical_ir_json({"schema_revision": revision, "rows": rows})


def _parquet_schema(revision: str, fields: tuple[tuple[str, str, bool], ...]) -> object:
    import pyarrow as pa

    arrow = cast(Any, pa)
    types = {"string": arrow.string(), "int64": arrow.int64(), "float64": arrow.float64()}
    return arrow.schema(
        [
            arrow.field(name, types[type_name], nullable=nullable)
            for name, type_name, nullable in fields
        ],
        metadata={
            b"dynamisrag.schema_revision": revision.encode("ascii"),
            b"dynamisrag.pyarrow_version": str(arrow.__version__).encode("ascii"),
        },
    )


def _parquet_bytes(
    rows: list[dict[str, object]], *, revision: str, fields: tuple[tuple[str, str, bool], ...]
) -> bytes:
    import pyarrow as pa
    import pyarrow.parquet as pq

    arrow = cast(Any, pa)
    parquet = cast(Any, pq)
    table = arrow.Table.from_pylist(rows, schema=_parquet_schema(revision, fields))
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


def _parquet_rows(content: bytes, *, revision: str) -> list[dict[str, object]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    arrow = cast(Any, pa)
    parquet = cast(Any, pq)
    try:
        table = parquet.read_table(arrow.BufferReader(content))
    except Exception as error:
        raise IrContractError("IR Parquet table is invalid") from error
    metadata = cast("dict[bytes, bytes]", table.schema.metadata or {})
    if metadata.get(b"dynamisrag.schema_revision") != revision.encode("ascii"):
        raise IrContractError("IR Parquet schema revision is incompatible")
    return cast("list[dict[str, object]]", table.to_pylist())


def _contents(
    dataset: IrDataset,
    config: IrExperimentConfig,
    run: IrRun,
    passage_mapping: IrPassageMapping,
) -> tuple[dict[str, bytes], IrEvaluation]:
    from dynamisrag.ir.scoring import evaluate_ir_run

    run.validate_against(dataset, config)
    evaluation = evaluate_ir_run(dataset, config, run, passage_mapping=passage_mapping)
    per_query = [row.payload() for row in evaluation.per_query]
    aggregate = [row.payload() for row in evaluation.aggregate]
    ranked_run = [hit.payload() for hit in run.hits]
    query_fields = (
        ("query_id", "string", False),
        ("qrel_document_count", "int64", False),
        ("positive_qrel_document_count", "int64", False),
        ("retrieved_document_count", "int64", False),
        ("judged_retrieved_document_count", "int64", False),
        ("unjudged_retrieved_document_count", "int64", False),
        ("ndcg_at_10", "float64", False),
        ("recall_at_10", "float64", False),
        ("map", "float64", False),
        ("mrr", "float64", False),
    )
    aggregate_fields = (
        ("measure", "string", False),
        ("value", "float64", False),
        ("numerator", "float64", False),
        ("query_denominator", "int64", False),
        ("queries_without_qrels", "int64", False),
        ("zero_positive_queries", "int64", False),
    )
    run_fields = (
        ("query_id", "string", False),
        ("document_id", "string", False),
        ("rank", "int64", False),
        ("raw_score", "float64", False),
        ("source_passage_id", "string", True),
    )
    return {
        "config.json": canonical_ir_json(config.payload()),
        "dataset.json": canonical_ir_json(dataset.payload()),
        "passage-mapping.json": canonical_ir_json(passage_mapping.payload()),
        "qrels.trec": trec_qrels(dataset).encode("utf-8"),
        "run.json": canonical_ir_json(run.payload()),
        "run.trec": trec_run(run).encode("utf-8"),
        "evaluation.json": canonical_ir_json(evaluation.payload()),
        "per-query.json": _table_json(_QUERY_SCHEMA_REVISION, per_query),
        "per-query.parquet": _parquet_bytes(
            per_query, revision=_QUERY_SCHEMA_REVISION, fields=query_fields
        ),
        "aggregate.json": _table_json(_AGGREGATE_SCHEMA_REVISION, aggregate),
        "aggregate.parquet": _parquet_bytes(
            aggregate, revision=_AGGREGATE_SCHEMA_REVISION, fields=aggregate_fields
        ),
        "ranked-run.json": _table_json(_RUN_SCHEMA_REVISION, ranked_run),
        "ranked-run.parquet": _parquet_bytes(
            ranked_run, revision=_RUN_SCHEMA_REVISION, fields=run_fields
        ),
    }, evaluation


def _manifest_payload(
    *,
    contents: dict[str, bytes],
    dataset_sha256: str,
    config_sha256: str,
    run_sha256: str,
    evaluation_sha256: str,
    passage_mapping_sha256: str,
) -> dict[str, object]:
    return {
        "revision": IR_BUNDLE_REVISION,
        "dataset_sha256": dataset_sha256,
        "config_sha256": config_sha256,
        "run_sha256": run_sha256,
        "evaluation_sha256": evaluation_sha256,
        "passage_mapping_sha256": passage_mapping_sha256,
        "files": [
            {
                "name": name,
                "size_bytes": len(contents[name]),
                "sha256": _sha256(contents[name]),
            }
            for name in _PAYLOAD_NAMES
        ],
    }


def write_ir_bundle(
    root: Path,
    *,
    dataset: IrDataset,
    config: IrExperimentConfig,
    run: IrRun,
    passage_mapping: IrPassageMapping | None = None,
) -> IrBundleReceipt:
    """Stage a complete bundle, then atomically rename it into an absent path.

    Never overwrite a prior scientific result. A partial write leaves no named
    result. The caller supplies the exact run; this function does not execute it.
    """
    destination = root.resolve(strict=False)
    if destination.exists():
        raise IrContractError("refusing to overwrite an existing IR bundle")
    mapping = passage_mapping or IrPassageMapping(())
    contents, evaluation = _contents(dataset, config, run, mapping)
    manifest = canonical_ir_json(
        _manifest_payload(
            contents=contents,
            dataset_sha256=dataset.sha256,
            config_sha256=config.sha256,
            run_sha256=run.sha256,
            evaluation_sha256=evaluation.sha256,
            passage_mapping_sha256=mapping.sha256,
        )
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".ir-stage-", dir=destination.parent))
    try:
        for name in _PAYLOAD_NAMES:
            (stage / name).write_bytes(contents[name])
        (stage / _MANIFEST_NAME).write_bytes(manifest)
        # The final artifact path appears only after every file exists.
        if destination.exists():
            raise IrContractError("another process published this bundle already")
        stage.rename(destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return IrBundleReceipt(
        root=destination,
        manifest_sha256=_sha256(manifest),
        dataset_sha256=dataset.sha256,
        config_sha256=config.sha256,
        run_sha256=run.sha256,
        evaluation_sha256=evaluation.sha256,
        passage_mapping_sha256=mapping.sha256,
    )


def _parse_manifest(data: bytes) -> dict[str, object]:
    try:
        value: object = json.loads(data)
    except (ValueError, UnicodeDecodeError) as error:
        raise IrContractError("IR manifest is not valid JSON") from error
    if not isinstance(value, dict):
        raise IrContractError("IR manifest is not an object")
    manifest = cast("dict[str, object]", value)
    if canonical_ir_json(manifest) != data:
        raise IrContractError("IR manifest bytes are not canonical")
    if set(manifest) != {
        "revision",
        "dataset_sha256",
        "config_sha256",
        "run_sha256",
        "evaluation_sha256",
        "passage_mapping_sha256",
        "files",
    }:
        raise IrContractError("IR manifest fields differ from the closed contract")
    return manifest


def _restore_json(payload: object, *, kind: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise IrContractError(f"IR {kind} payload must be a JSON object")
    return cast("dict[str, Any]", payload)


def _restore_dataset(raw: object) -> IrDataset:
    data = _restore_json(raw, kind="dataset")
    if data.get("revision") != IR_CONTRACT_REVISION:
        raise IrContractError("IR dataset contract revision is incompatible")
    queries = data.get("queries")
    qrels = data.get("qrels")
    if not isinstance(queries, list) or not isinstance(qrels, list):
        raise IrContractError("IR dataset contains invalid query/qrel lists")
    typed_queries = cast("list[object]", queries)
    typed_qrels = cast("list[object]", qrels)
    return IrDataset(
        source_id=cast("str", data.get("source_id")),
        source_revision=cast("str", data.get("source_revision")),
        corpus_sha256=cast("str", data.get("corpus_sha256")),
        queries=tuple(IrQuery(**_restore_json(q, kind="query")) for q in typed_queries),
        qrels=tuple(IrQrel(**_restore_json(q, kind="qrel")) for q in typed_qrels),
    )


def _restore_config(raw: object) -> IrExperimentConfig:
    data = _restore_json(raw, kind="configuration")
    if data.get("revision") != IR_CONTRACT_REVISION:
        raise IrContractError("IR configuration contract revision is incompatible")
    if data.get("metric_policy") != IrMetricPolicy().payload():
        raise IrContractError("IR metric policy is incompatible")
    return IrExperimentConfig(
        dataset_sha256=cast("str", data.get("dataset_sha256")),
        code_sha=cast("str", data.get("code_sha")),
        retrieval_revision=cast("str", data.get("retrieval_revision")),
        projection_sha256=cast("str", data.get("projection_sha256")),
        parameters_json=cast("str", data.get("parameters_json")),
    )


def _restore_mapping(raw: object) -> IrPassageMapping:
    data = _restore_json(raw, kind="passage mapping")
    if set(data) != {"revision", "entries"} or data.get("revision") != "ir-passage-mapping-v1":
        raise IrContractError("IR passage mapping revision or fields are incompatible")
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise IrContractError("IR passage mapping entries are invalid")
    return IrPassageMapping(
        tuple(
            IrPassageMapEntry(**_restore_json(entry, kind="passage mapping entry"))
            for entry in cast("list[object]", entries)
        )
    )


def read_ir_inputs(root: Path, *, expected_run_sha256: str) -> IrRunInputs:
    """Load a closed input directory and require the caller's sealed run SHA."""
    if not root.is_dir() or root.is_symlink():
        raise IrContractError("IR input root is not a regular directory")
    names = {path.name for path in root.iterdir()}
    if names != set(_INPUT_NAMES):
        raise IrContractError("IR input file inventory differs from the closed contract")
    if any((root / name).is_symlink() or not (root / name).is_file() for name in names):
        raise IrContractError("IR input entries must be regular files")
    documents: dict[str, object] = {}
    contents: dict[str, bytes] = {}
    for name in _INPUT_NAMES:
        content = (root / name).read_bytes()
        try:
            document: object = json.loads(content)
        except (UnicodeDecodeError, ValueError) as error:
            raise IrContractError(f"IR input {name} is invalid JSON") from error
        if canonical_ir_json(document) != content:
            raise IrContractError(f"IR input {name} is not canonical JSON")
        documents[name] = document
        contents[name] = content
    dataset = _restore_dataset(documents["dataset.json"])
    config = _restore_config(documents["config.json"])
    run = _restore_run(documents["run.json"])
    passage_mapping = _restore_mapping(documents["passage-mapping.json"])
    run.validate_against(dataset, config)
    passage_mapping.validate_run(run)
    if (
        _sha256(contents["dataset.json"]) != dataset.sha256
        or _sha256(contents["config.json"]) != config.sha256
        or _sha256(contents["run.json"]) != run.sha256
        or _sha256(contents["passage-mapping.json"]) != passage_mapping.sha256
        or run.sha256 != expected_run_sha256
    ):
        raise IrContractError("IR input identities do not match the caller's expected run")
    return IrRunInputs(dataset, config, run, passage_mapping)


def _restore_run(raw: object) -> IrRun:
    data = _restore_json(raw, kind="run")
    if data.get("revision") != IR_CONTRACT_REVISION:
        raise IrContractError("IR run contract revision is incompatible")
    hits = data.get("hits")
    query_ids = data.get("query_ids")
    source_exhausted_query_ids = data.get("source_exhausted_query_ids", [])
    if (
        not isinstance(hits, list)
        or not isinstance(query_ids, list)
        or not isinstance(source_exhausted_query_ids, list)
    ):
        raise IrContractError("IR run contains invalid query/hit lists")
    typed_hits = cast("list[object]", hits)
    return IrRun(
        dataset_sha256=cast("str", data.get("dataset_sha256")),
        config_sha256=cast("str", data.get("config_sha256")),
        query_ids=tuple(cast("list[str]", query_ids)),
        hits=tuple(IrHit(**_restore_json(hit, kind="hit")) for hit in typed_hits),
        evaluation_depth=cast("int", data.get("evaluation_depth")),
        passage_mapping_sha256=cast("str | None", data.get("passage_mapping_sha256")),
        source_exhausted_query_ids=tuple(cast("list[str]", source_exhausted_query_ids)),
    )


def _require_closed_inventory(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        raise IrContractError("IR bundle root is not a regular directory")
    names = {path.name for path in root.iterdir()}
    if names != set(_PAYLOAD_NAMES) | {_MANIFEST_NAME}:
        raise IrContractError("IR bundle file inventory differs from the closed contract")
    if any((root / name).is_symlink() for name in names):
        raise IrContractError("IR bundle does not permit symbolic links")
    if any(not (root / name).is_file() for name in names):
        raise IrContractError("IR bundle entries must be regular files")


def _verified_payloads(root: Path, manifest: dict[str, object]) -> dict[str, bytes]:
    raw_entries: object = manifest.get("files")
    if not isinstance(raw_entries, list):
        raise IrContractError("IR bundle declares an invalid artifact inventory")
    entries = cast("list[object]", raw_entries)
    if len(entries) != len(_PAYLOAD_NAMES):
        raise IrContractError("IR bundle declares an invalid artifact inventory")
    contents: dict[str, bytes] = {}
    for name, entry in zip(_PAYLOAD_NAMES, entries, strict=True):
        info = _restore_json(entry, kind="artifact inventory entry")
        if set(info) != {"name", "size_bytes", "sha256"}:
            raise IrContractError("IR bundle inventory entry has unsupported fields")
        if info.get("name") != name:
            raise IrContractError("IR bundle file name or ordering is invalid")
        content = (root / name).read_bytes()
        size_bytes = info.get("size_bytes")
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes != len(content)
            or info.get("sha256") != _sha256(content)
        ):
            raise IrContractError("IR bundle content disagrees with its manifest")
        contents[name] = content
    return contents


def _verified_json(
    contents: dict[str, bytes],
    *,
    dataset_sha256: str,
    config_sha256: str,
    run_sha256: str,
    evaluation_sha256: str,
    passage_mapping_sha256: str,
) -> dict[str, object]:
    documents: dict[str, object] = {}
    for name, expected in (
        ("dataset.json", dataset_sha256),
        ("config.json", config_sha256),
        ("run.json", run_sha256),
        ("passage-mapping.json", passage_mapping_sha256),
        ("evaluation.json", evaluation_sha256),
    ):
        content = contents[name]
        try:
            document: object = json.loads(content)
        except (UnicodeDecodeError, ValueError) as error:
            raise IrContractError("IR bundle contains invalid JSON") from error
        if canonical_ir_json(document) != content or _sha256(content) != expected:
            raise IrContractError("IR bundle semantic identity does not match its JSON content")
        documents[name] = document
    return documents


def _verify_semantics(
    documents: dict[str, object],
    contents: dict[str, bytes],
    *,
    dataset_sha256: str,
    config_sha256: str,
    run_sha256: str,
    evaluation_sha256: str,
    passage_mapping_sha256: str,
) -> None:
    try:
        dataset = _restore_dataset(documents["dataset.json"])
        config = _restore_config(documents["config.json"])
        run = _restore_run(documents["run.json"])
        passage_mapping = _restore_mapping(documents["passage-mapping.json"])
        run.validate_against(dataset, config)
        passage_mapping.validate_run(run)
    except (TypeError, KeyError) as error:
        raise IrContractError("IR bundle contains malformed typed payloads") from error
    if (
        dataset.sha256 != dataset_sha256
        or config.sha256 != config_sha256
        or run.sha256 != run_sha256
        or passage_mapping.sha256 != passage_mapping_sha256
    ):
        raise IrContractError("IR bundle typed semantic identity disagrees with its manifest")
    if contents["qrels.trec"] != trec_qrels(dataset).encode("utf-8"):
        raise IrContractError("IR qrels TREC text differs from the canonical dataset")
    if contents["run.trec"] != trec_run(run).encode("utf-8"):
        raise IrContractError("IR run TREC text differs from the canonical ranking")
    _verify_table_parity(contents)
    regenerated, evaluation = _contents(dataset, config, run, passage_mapping)
    if (
        evaluation.sha256 != evaluation_sha256
        or contents["evaluation.json"] != canonical_ir_json(evaluation.payload())
        or regenerated != contents
    ):
        raise IrContractError("IR result tables disagree with independently regenerated scores")


def _verify_table_parity(contents: dict[str, bytes]) -> None:
    for json_name, parquet_name, revision in (
        ("per-query.json", "per-query.parquet", _QUERY_SCHEMA_REVISION),
        ("aggregate.json", "aggregate.parquet", _AGGREGATE_SCHEMA_REVISION),
        ("ranked-run.json", "ranked-run.parquet", _RUN_SCHEMA_REVISION),
    ):
        try:
            raw: object = json.loads(contents[json_name])
        except (UnicodeDecodeError, ValueError) as error:
            raise IrContractError("IR result table JSON is invalid") from error
        table = _restore_json(raw, kind="result table")
        if set(table) != {"schema_revision", "rows"} or table.get("schema_revision") != revision:
            raise IrContractError("IR result table JSON schema revision is incompatible")
        rows = table.get("rows")
        if not isinstance(rows, list) or canonical_ir_json(table) != contents[json_name]:
            raise IrContractError("IR result table JSON rows are invalid or noncanonical")
        if _parquet_rows(contents[parquet_name], revision=revision) != rows:
            raise IrContractError("IR JSON and Parquet table rows differ")


def verify_ir_bundle(root: Path, *, expected_run_sha256: str) -> IrBundleReceipt:
    """Recompute every contract, cross-link and derivative from trusted run SHA.

    A manifest is not a trust anchor. The caller must supply the expected run
    identity out of band. Qrels and TREC are regenerated from verified JSON:
    rewriting a manifest to bless a substituted TREC file cannot pass.
    """
    _require_closed_inventory(root)
    data = (root / _MANIFEST_NAME).read_bytes()
    manifest = _parse_manifest(data)
    if manifest.get("revision") != IR_BUNDLE_REVISION:
        raise IrContractError("IR bundle revision is incompatible")
    dataset_sha256 = manifest.get("dataset_sha256")
    config_sha256 = manifest.get("config_sha256")
    run_sha256 = manifest.get("run_sha256")
    evaluation_sha256 = manifest.get("evaluation_sha256")
    passage_mapping_sha256 = manifest.get("passage_mapping_sha256")
    if (
        not isinstance(dataset_sha256, str)
        or not isinstance(config_sha256, str)
        or not isinstance(run_sha256, str)
        or not isinstance(evaluation_sha256, str)
        or not isinstance(passage_mapping_sha256, str)
        or run_sha256 != expected_run_sha256
    ):
        raise IrContractError("IR bundle identities do not match caller expectations")
    contents = _verified_payloads(root, manifest)
    documents = _verified_json(
        contents,
        dataset_sha256=dataset_sha256,
        config_sha256=config_sha256,
        run_sha256=run_sha256,
        evaluation_sha256=evaluation_sha256,
        passage_mapping_sha256=passage_mapping_sha256,
    )
    _verify_semantics(
        documents,
        contents,
        dataset_sha256=dataset_sha256,
        config_sha256=config_sha256,
        run_sha256=run_sha256,
        evaluation_sha256=evaluation_sha256,
        passage_mapping_sha256=passage_mapping_sha256,
    )
    return IrBundleReceipt(
        root=root,
        manifest_sha256=_sha256(data),
        dataset_sha256=dataset_sha256,
        config_sha256=config_sha256,
        run_sha256=run_sha256,
        evaluation_sha256=evaluation_sha256,
        passage_mapping_sha256=passage_mapping_sha256,
    )


def read_verified_ir_evaluation(root: Path, *, expected_run_sha256: str) -> IrEvaluation:
    """Return score rows only after independently verifying the full bundle."""
    from dynamisrag.ir.scoring import IrAggregateScore, IrEvaluation, IrQueryScore

    verify_ir_bundle(root, expected_run_sha256=expected_run_sha256)
    try:
        data: object = json.loads((root / "evaluation.json").read_bytes())
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise IrContractError("IR evaluation JSON cannot be read") from error
    payload = _restore_json(data, kind="evaluation")
    per_query = payload.get("per_query")
    aggregate = payload.get("aggregate")
    engine = payload.get("scoring_engine")
    if (
        not isinstance(per_query, list)
        or not isinstance(aggregate, list)
        or not isinstance(engine, dict)
    ):
        raise IrContractError("IR evaluation payload has invalid result tables")
    engine_entries = cast("dict[str, object]", engine)
    return IrEvaluation(
        dataset_sha256=cast("str", payload.get("dataset_sha256")),
        config_sha256=cast("str", payload.get("config_sha256")),
        projection_sha256=cast("str", payload.get("projection_sha256")),
        run_sha256=cast("str", payload.get("run_sha256")),
        metric_policy_sha256=cast("str", payload.get("metric_policy_sha256")),
        evaluation_depth=cast("int", payload.get("evaluation_depth")),
        passage_mapping_sha256=cast("str | None", payload.get("passage_mapping_sha256")),
        scoring_engine=tuple(
            (name, cast("str", version)) for name, version in engine_entries.items()
        ),
        per_query=tuple(
            IrQueryScore(**_restore_json(row, kind="per-query score"))
            for row in cast("list[object]", per_query)
        ),
        aggregate=tuple(
            IrAggregateScore(**_restore_json(row, kind="aggregate score"))
            for row in cast("list[object]", aggregate)
        ),
    )
