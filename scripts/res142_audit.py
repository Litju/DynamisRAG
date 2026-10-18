"""Offline RES-142 baseline and B2 transfer evidence verification/export."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import stat
import sys
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

from dynamisrag.ir.artifacts import read_verified_ir_evaluation
from dynamisrag.ir.contracts import IrContractError

__all__ = [
    "export_audit_package",
    "verify_audit_package",
    "verify_baseline_evidence",
    "verify_candidate_transfer",
]

SOURCE_ROOT: Final[str] = ".tmp/res142-scifact-colab-baseline-b748cfc"
TRANSFER_ROOT: Final[str] = ".tmp/res142-b2-transfer-20261016"
SCIFACT_ROOT: Final[str] = ".tmp/res142-scifact-beir-test-20261015"
RETRIEVAL_EVIDENCE: Final[str] = ".tmp/res142-colab-res140-evidence.zip"
PUBLISHED: Final[dict[str, str]] = {
    "source_archive": "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165",
    "source_manifest": "13346b186c531fa7dfc6e4fbb9ed3606de4e1ad10ae24564ac742ab7ee37aa15",
    "source_dataset": "4e7123751dc926ebd35c4f2f681565b7dfa0fed405bdf7f762ca85f1f46240d4",
    "source_pilot_dataset": "70c07ebaaac9d7244540c8a92e8605df9d2dbaaa7002b4e1021a7962800784d7",
    "pilot_selection": "9448b2cccf9d5755c8029b70f73c83da924a6b16f68779a5dddaf0b6ddd67aa6",
    "selected_query_ids": "6f84f7bb46976a5baafbc5d9be958f300dde35ddc74a23c51940f46045f5db3c",
    "candidate_snapshot": "c4320cebfca3350a8c0d2070c6ed3f20a68692cfdc9b3fc4f583b0a92dd5310e",
    "passage_mapping": "f2489a8dd194849921c81fec8fbd2959567913ef1b58bb56e6d27c9af37854ad",
    "projection": "f8483d00a654a1a2c3d5a8d5f712676ba79edc006ae513bf0a3398ce963a0473",
    "source_document_map": "c43b1020ddf783edc9759e0b6abd5a250beeb6d5a37ffca8e47e82eb347193ab",
    "evaluation_dataset": "d73f7463c9338ce8134ddb5ff0bbebaeae8e390edf768e092e803fb961d59f12",
    "baseline_run": "e2831be83856c98d752c570aad93d0f4734e20598be7e563b33ba21866dc3453",
    "baseline_manifest": "13661d791a9a49c3313c98c247f78814c1aea074d21d84b1169a0b33ff6ab1fc",
    "baseline_evaluation": "0ffe6d4811ae42b754e9b04ccda8c079e0efbbbc3f60d0d75d11698e5c3a2e66",
    "qwen3_config": "9d3565ea1fb7ae863cc38b3119a9384a4a0327120b6c5272428450113c65f756",
    "retrieval_evidence": "4be0b44075836ff68580ec6a6de19b74283b9afa74c152179c3b91bb5d2a258a",
    "retrieval_summary": "4e2474efd2833f22841b304be110ec172ee439405066b90ef337f3caddcef83c",
    "retrieval_validation": "76f8ce9daba2171859ea988bef9df843c9cf865979b4876966daff4f58fa8c81",
    "candidate_zip": "c98d26a7aaf38f35b66203f0d27a73f618800218b1a90c4e322a02ab2c4100ac",
    "rights_sidecar": "65980a1e19e6a01444193df55e98fd90a941b4c14b4c83efbd32db3564449d92",
}
BASELINE_METRICS: Final[dict[str, float]] = {
    "nDCG@10": 0.7254108326542275,
    "Recall@10": 0.8625,
    "MAP": 0.6765476190476191,
    "MRR": 0.6883333333333334,
}
_CANDIDATE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "NOTICES.txt",
        "RES142_TRANSFER_DECISION.txt",
        "candidate-input.json",
        "manifest.json",
        "pilot-query-selection.json",
        "source-provenance.json",
    }
)
_PACKAGE_FILES: Final[dict[str, str]] = {
    "baseline/aggregate.json": f"{SOURCE_ROOT}/baseline/aggregate.json",
    "baseline/aggregate.parquet": f"{SOURCE_ROOT}/baseline/aggregate.parquet",
    "baseline/config.json": f"{SOURCE_ROOT}/baseline/config.json",
    "baseline/dataset.json": f"{SOURCE_ROOT}/baseline/dataset.json",
    "baseline/evaluation.json": f"{SOURCE_ROOT}/baseline/evaluation.json",
    "baseline/manifest.json": f"{SOURCE_ROOT}/baseline/manifest.json",
    "baseline/passage-mapping.json": f"{SOURCE_ROOT}/baseline/passage-mapping.json",
    "baseline/per-query.json": f"{SOURCE_ROOT}/baseline/per-query.json",
    "baseline/per-query.parquet": f"{SOURCE_ROOT}/baseline/per-query.parquet",
    "baseline/qrels.trec": f"{SOURCE_ROOT}/baseline/qrels.trec",
    "baseline/ranked-run.json": f"{SOURCE_ROOT}/baseline/ranked-run.json",
    "baseline/ranked-run.parquet": f"{SOURCE_ROOT}/baseline/ranked-run.parquet",
    "baseline/run.json": f"{SOURCE_ROOT}/baseline/run.json",
    "baseline/run.trec": f"{SOURCE_ROOT}/baseline/run.trec",
    "source/beir-scifact-test.zip": f"{SCIFACT_ROOT}.zip",
    "source/manifest.json": f"{SCIFACT_ROOT}/manifest.json",
    "source/pilot-query-selection.json": f"{SCIFACT_ROOT}/pilot-query-selection.json",
    "source/source-dataset.json": f"{SOURCE_ROOT}/source-dataset.json",
    "source/source-pilot-dataset.json": f"{SOURCE_ROOT}/source-pilot-dataset.json",
    "identity/qrel-source-document-identity-map.json": (
        f"{SOURCE_ROOT}/qrel-source-document-identity-map.json"
    ),
    "identity/selected-qrel-identity-projection.json": (
        ".tmp/res142-scifact-colab-baseline-20261016/selected-qrel-identity-projection.json"
    ),
    "provenance/candidate-snapshot.json": f"{SOURCE_ROOT}/candidate-snapshot.json",
    "provenance/qwen3-reranker-config.json": f"{SOURCE_ROOT}/qwen3-reranker-config.json",
    "retrieval/evidence.zip": RETRIEVAL_EVIDENCE,
    "b2/res142-b2-candidate-input-v3.zip": (f"{TRANSFER_ROOT}/res142-b2-candidate-input-v3.zip"),
    "b2/rights-confirmation-v3.json": f"{TRANSFER_ROOT}/rights-confirmation-v3.json",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(data: bytes, label: str) -> dict[str, Any]:
    try:
        value: object = json.loads(data)
    except (UnicodeDecodeError, ValueError) as error:
        raise IrContractError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise IrContractError(f"{label} must be a JSON object")
    return cast("dict[str, Any]", value)


def _require_digest(data: bytes, expected: str, label: str) -> str:
    observed = _sha256(data)
    if observed != expected:
        raise IrContractError(f"{label} SHA-256 differs from the published identity")
    return observed


# This parser enforces the complete rights, inventory, and identity boundary in one pass.
def _read_candidate_archive(  # noqa: PLR0912, PLR0915
    archive_path: Path,
    *,
    expected_archive_sha256: str | None,
    expected_sidecar_sha256: str | None,
    rights_sidecar_path: Path,
    source_to_canonical: Mapping[str, str],
) -> dict[str, Any]:
    archive_bytes = archive_path.read_bytes()
    archive_sha256 = _sha256(archive_bytes)
    if expected_archive_sha256 is not None and archive_sha256 != expected_archive_sha256:
        raise IrContractError("B2 candidate ZIP SHA-256 differs from the expected identity")
    sidecar_bytes = rights_sidecar_path.read_bytes()
    sidecar_sha256 = _sha256(sidecar_bytes)
    if expected_sidecar_sha256 is not None and sidecar_sha256 != expected_sidecar_sha256:
        raise IrContractError("B2 rights sidecar SHA-256 differs from the expected identity")
    sidecar = _read_json(sidecar_bytes, "B2 rights sidecar")
    if set(sidecar) != {
        "candidate_transfer_zip_sha256",
        "candidate_snapshot_sha256",
        "drive_destination",
        "rights_status",
        "schema_revision",
        "source_input_zip_sha256",
        "transfer_status",
        "user_confirmation",
    }:
        raise IrContractError("B2 rights sidecar fields differ from its closed contract")

    try:
        bundle = zipfile.ZipFile(archive_path)
    except (OSError, zipfile.BadZipFile) as error:
        raise IrContractError("B2 candidate input is not a valid ZIP archive") from error
    with bundle:
        infos = bundle.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or set(names) != _CANDIDATE_NAMES:
            raise IrContractError("B2 candidate ZIP inventory differs from its closed contract")
        if any(
            info.is_dir() or PurePosixPath(info.filename).name != info.filename for info in infos
        ):
            raise IrContractError("B2 candidate ZIP contains a directory or unsafe member path")
        damaged = bundle.testzip()
        if damaged is not None:
            raise IrContractError(f"B2 candidate ZIP member failed CRC validation: {damaged}")
        contents = {name: bundle.read(name) for name in names}

    manifest = _read_json(contents["manifest.json"], "B2 candidate manifest")
    if manifest.get("schema_revision") != "res142-scifact-b2-candidate-transfer-manifest-v1":
        raise IrContractError("B2 candidate manifest revision is incompatible")
    if set(manifest.get("files", {})) != _CANDIDATE_NAMES - {"manifest.json"}:
        raise IrContractError("B2 candidate manifest file inventory is not closed")
    for name, item in manifest["files"].items():
        if not isinstance(item, dict) or set(item) != {"sha256", "size_bytes"}:
            raise IrContractError(f"B2 candidate manifest entry is invalid: {name}")
        payload = contents[name]
        if item["size_bytes"] != len(payload) or item["sha256"] != _sha256(payload):
            raise IrContractError(f"B2 candidate member digest differs: {name}")

    candidate_input = _read_json(contents["candidate-input.json"], "B2 candidate input")
    selection = _read_json(contents["pilot-query-selection.json"], "B2 query selection")
    query_rows = candidate_input.get("queries_in_selection_order")
    query_ids = selection.get("selected_query_ids")
    if not isinstance(query_rows, list) or not isinstance(query_ids, list):
        raise IrContractError("B2 candidate queries or selected IDs are invalid")
    if (
        len(query_rows) != 20
        or candidate_input.get("query_count") != 20
        or manifest.get("query_count") != 20
        or candidate_input.get("candidate_limit") != 50
        or manifest.get("candidate_limit") != 50
        or candidate_input.get("candidate_count") != 1000
        or manifest.get("candidate_count") != 1000
        or len(query_ids) != 20
        or manifest.get("candidate_snapshot_sha256") != PUBLISHED["candidate_snapshot"]
        or manifest.get("passage_mapping_sha256") != PUBLISHED["passage_mapping"]
        or manifest.get("projection_sha256") != PUBLISHED["projection"]
        or manifest.get("qrel_identity_mapping_sha256") != PUBLISHED["source_document_map"]
        or manifest.get("evaluation_dataset_sha256") != PUBLISHED["evaluation_dataset"]
    ):
        raise IrContractError("B2 input must contain 20 queries and exactly 1,000 candidates")
    ordered_ids = [row.get("query_id") for row in query_rows if isinstance(row, dict)]
    if len(ordered_ids) != 20 or ordered_ids != query_ids or len(set(ordered_ids)) != 20:
        raise IrContractError("B2 query IDs differ from the frozen selection order")

    candidate_total = 0
    query_text_sha256: dict[str, str] = {}
    for query in query_rows:
        if not isinstance(query, dict):
            raise IrContractError("B2 query row is not an object")
        query_id = query.get("query_id")
        query_text = query.get("query_text")
        candidates = query.get("candidates")
        if (
            not isinstance(query_id, str)
            or not isinstance(query_text, str)
            or not query_text.strip()
            or not isinstance(candidates, list)
            or len(candidates) != 50
            or query.get("candidate_count") != 50
            or query.get("query_text_sha256") != _sha256(query_text.encode("utf-8"))
        ):
            raise IrContractError(f"B2 query row is incomplete or malformed: {query_id}")
        query_text_sha256[query_id] = query["query_text_sha256"]
        passages: set[str] = set()
        for rank, candidate in enumerate(candidates, start=1):
            if not isinstance(candidate, dict):
                raise IrContractError(f"B2 candidate row is invalid: {query_id}/{rank}")
            text = candidate.get("text")
            retrieval = candidate.get("original_retrieval")
            if not isinstance(text, str) or not text.strip() or not isinstance(retrieval, dict):
                raise IrContractError(
                    f"B2 candidate text or identity is missing: {query_id}/{rank}"
                )
            passage_id = candidate.get("passage_id")
            passage_key = retrieval.get("passage_key")
            provenance = retrieval.get("provenance")
            candidate_sha256 = _sha256(text.encode("utf-8"))
            if (
                not isinstance(passage_id, str)
                or passage_id != passage_key
                or passage_id in passages
                or candidate.get("query_id") != query_id
                or candidate.get("original_rrf_rank") != rank
                or retrieval.get("original_rrf_rank") != rank
                or candidate.get("candidate_text_sha256") != candidate_sha256
                or retrieval.get("candidate_text_sha256") != candidate_sha256
                or not isinstance(provenance, dict)
            ):
                raise IrContractError(
                    f"B2 candidate identity or rank is invalid: {query_id}/{rank}"
                )
            source_id = provenance.get("source_external_id")
            canonical_id = provenance.get("document_canonical_key")
            if (
                not isinstance(source_id, str)
                or not isinstance(canonical_id, str)
                or source_to_canonical.get(source_id) != canonical_id
            ):
                raise IrContractError(
                    "B2 candidate source identity is absent from the 5,183-row map: "
                    f"{query_id}/{rank}"
                )
            passages.add(passage_id)
            candidate_total += 1
    if candidate_total != 1000:
        raise IrContractError("B2 candidate population is not exactly 1,000")

    sidecar_expected = {
        "schema_revision": "res142-rights-confirmation-v1",
        "source_input_zip_sha256": manifest.get("source_input_zip_sha256"),
        "candidate_transfer_zip_sha256": archive_sha256,
        "candidate_snapshot_sha256": manifest.get("candidate_snapshot_sha256"),
        "rights_status": "requester-confirmed-for-this-run",
        "transfer_status": "prepared-locally; not uploaded",
    }
    expected_drive_destination = (
        "/content/drive/MyDrive/DynamisRAG/RES-142/retrieval/runs/"
        "20261016-10bce56f6ea04d949355af34fba2698c/b2"
    )
    if (
        any(sidecar.get(key) != value for key, value in sidecar_expected.items())
        or sidecar.get("user_confirmation") != "we have the rights"
        or sidecar.get("drive_destination") != expected_drive_destination
    ):
        raise IrContractError("B2 rights sidecar does not bind the candidate ZIP and status")
    if (
        candidate_input.get("candidate_snapshot_sha256")
        != manifest.get("candidate_snapshot_sha256")
        or candidate_input.get("passage_mapping_sha256") != manifest.get("passage_mapping_sha256")
        or candidate_input.get("projection_sha256") != manifest.get("projection_sha256")
        or manifest.get("qrels_included") is not False
        or manifest.get("full_corpus_included") is not False
        or manifest.get("rights_status") != "requester-confirmed-for-this-private-run"
    ):
        raise IrContractError("B2 candidate manifest identity or transfer boundary is invalid")
    return {
        "archive_sha256": archive_sha256,
        "archive_size_bytes": len(archive_bytes),
        "rights_sidecar_sha256": sidecar_sha256,
        "query_count": 20,
        "candidate_count": candidate_total,
        "query_ids_in_selection_order": ordered_ids,
        "query_text_sha256": query_text_sha256,
        "candidate_snapshot_sha256": manifest["candidate_snapshot_sha256"],
        "passage_mapping_sha256": manifest["passage_mapping_sha256"],
        "projection_sha256": manifest["projection_sha256"],
        "qrels_included": False,
        "full_corpus_included": False,
        "internal_members": {
            name: {"size_bytes": len(data), "sha256": _sha256(data)}
            for name, data in sorted(contents.items())
        },
    }


def verify_candidate_transfer(
    archive_path: Path,
    rights_sidecar_path: Path,
    source_identity_map_path: Path,
    *,
    expected_archive_sha256: str | None = PUBLISHED["candidate_zip"],
    expected_sidecar_sha256: str | None = PUBLISHED["rights_sidecar"],
) -> dict[str, Any]:
    """Check the published candidate archive, sidecar, closed inventory and identities."""
    map_bytes = source_identity_map_path.read_bytes()
    _require_digest(map_bytes, PUBLISHED["source_document_map"], "Source identity map")
    identity_payload = _read_json(map_bytes, "Source identity map")
    entries = identity_payload.get("entries")
    if not isinstance(entries, list) or len(entries) != 5183:
        raise IrContractError("Source-to-canonical identity map must contain 5,183 rows")
    identity: dict[str, str] = {}
    for row in entries:
        if not isinstance(row, dict):
            raise IrContractError("Source-to-canonical identity map contains an invalid row")
        source_id = row.get("source_document_id")
        canonical_id = row.get("canonical_document_id")
        if (
            not isinstance(source_id, str)
            or not isinstance(canonical_id, str)
            or source_id in identity
        ):
            raise IrContractError("Source-to-canonical identity map contains duplicate IDs")
        identity[source_id] = canonical_id
    if len(set(identity.values())) != 5183:
        raise IrContractError("Source-to-canonical identity map contains duplicate canonical IDs")
    return _read_candidate_archive(
        archive_path,
        expected_archive_sha256=expected_archive_sha256,
        expected_sidecar_sha256=expected_sidecar_sha256,
        rights_sidecar_path=rights_sidecar_path,
        source_to_canonical=identity,
    )


def _check_source_lineage(
    *,
    source_dataset_bytes: bytes,
    pilot_dataset_bytes: bytes,
    selection_bytes: bytes,
    map_bytes: bytes,
    selected_projection_bytes: bytes,
    source_archive_bytes: bytes,
    canonical_dataset_bytes: bytes,
) -> dict[str, Any]:
    source = _read_json(source_dataset_bytes, "Full source dataset")
    pilot = _read_json(pilot_dataset_bytes, "Source pilot dataset")
    selection = _read_json(selection_bytes, "Frozen pilot selection")
    identity_payload = _read_json(map_bytes, "Source identity map")
    projected = _read_json(selected_projection_bytes, "Selected qrel identity projection")
    canonical_dataset = _read_json(canonical_dataset_bytes, "Canonical evaluation dataset")
    if (
        projected.get("evaluation_dataset_sha256") != PUBLISHED["evaluation_dataset"]
        or projected.get("qrel_identity_mapping_sha256") != PUBLISHED["source_document_map"]
    ):
        raise IrContractError(
            "Selected qrel projection references a different identity map or dataset"
        )
    if len(source.get("queries", [])) != 300 or len(source.get("qrels", [])) != 339:
        raise IrContractError("Full SciFact source dataset must preserve 300 queries and 339 qrels")
    if len(pilot.get("queries", [])) != 20 or len(pilot.get("qrels", [])) != 24:
        raise IrContractError("SciFact source pilot must preserve 20 queries and 24 source qrels")
    entries = identity_payload.get("entries")
    selected_ids = selection.get("selected_query_ids")
    if not isinstance(entries, list) or len(entries) != 5183 or not isinstance(selected_ids, list):
        raise IrContractError("Source map or frozen query selection has the wrong population")
    source_to_canonical = {
        str(row["source_document_id"]): str(row["canonical_document_id"])
        for row in entries
        if isinstance(row, dict)
    }
    source_queries = {row["query_id"]: row for row in source["queries"]}
    pilot_queries = pilot["queries"]
    if pilot_queries != [source_queries[query_id] for query_id in sorted(selected_ids)]:
        raise IrContractError("Source pilot queries differ from the unchanged source dataset")
    source_qrels = source["qrels"]
    selected_qrels = [row for row in source_qrels if row["query_id"] in set(selected_ids)]
    if pilot["qrels"] != selected_qrels:
        raise IrContractError(
            "Source pilot qrels or relevance labels differ from the source dataset"
        )
    expected_projection = sorted(
        [
            {
                "query_id": row["query_id"],
                "source_document_id": row["document_id"],
                "canonical_document_id": source_to_canonical[row["document_id"]],
                "relevance": row["relevance"],
            }
            for row in pilot["qrels"]
        ],
        key=lambda row: (row["query_id"], row["source_document_id"]),
    )
    if projected.get("rows") != expected_projection:
        raise IrContractError("Selected canonical qrel projection changes source IDs or labels")
    expected_canonical_qrels = sorted(
        [
            {
                "query_id": row["query_id"],
                "document_id": source_to_canonical[row["document_id"]],
                "relevance": row["relevance"],
            }
            for row in pilot["qrels"]
        ],
        key=lambda row: (row["query_id"], row["document_id"]),
    )
    if (
        canonical_dataset.get("queries") != pilot_queries
        or canonical_dataset.get("qrels") != expected_canonical_qrels
    ):
        raise IrContractError("RES-140 dataset does not preserve canonicalized source qrels")

    with zipfile.ZipFile(io.BytesIO(source_archive_bytes)) as source_zip:
        if "scifact/corpus.jsonl" not in source_zip.namelist():
            raise IrContractError("Pinned BEIR SciFact source archive lacks its corpus")
        corpus_rows = [
            json.loads(line)
            for line in source_zip.read("scifact/corpus.jsonl").splitlines()
            if line.strip()
        ]
    corpus_ids = {str(row["_id"]) for row in corpus_rows}
    if len(corpus_ids) != 5183 or corpus_ids != set(source_to_canonical):
        raise IrContractError("5,183-row identity map does not cover the source corpus IDs")
    if selection.get("selected_query_ids_sha256") != PUBLISHED["selected_query_ids"]:
        raise IrContractError("Frozen selected-query ID digest differs from its published identity")
    return {
        "source_query_count": 300,
        "source_qrel_count": 339,
        "pilot_query_count": 20,
        "pilot_source_qrel_count": 24,
        "source_document_map_rows": len(source_to_canonical),
        "source_corpus_ids_match_map": True,
        "source_qrels_and_labels_preserved": True,
    }


def verify_baseline_evidence(baseline_root: Path, source_root: Path) -> dict[str, Any]:
    """Verify the complete RES-140 bundle and reproduce its four aggregate metrics."""
    from dynamisrag.ir.artifacts import verify_ir_bundle

    source_dataset = (source_root / "source-dataset.json").read_bytes()
    pilot_dataset = (source_root / "source-pilot-dataset.json").read_bytes()
    snapshot = (source_root / "candidate-snapshot.json").read_bytes()
    identity_map = (source_root / "qrel-source-document-identity-map.json").read_bytes()
    qwen_config = (source_root / "qwen3-reranker-config.json").read_bytes()
    manifest = (baseline_root / "manifest.json").read_bytes()
    run = (baseline_root / "run.json").read_bytes()
    evaluation = (baseline_root / "evaluation.json").read_bytes()
    passage_mapping = (baseline_root / "passage-mapping.json").read_bytes()
    canonical_dataset = (baseline_root / "dataset.json").read_bytes()
    for data, key, name in (
        (source_dataset, "source_dataset", "Full source dataset"),
        (pilot_dataset, "source_pilot_dataset", "Source pilot dataset"),
        (snapshot, "candidate_snapshot", "Candidate snapshot"),
        (identity_map, "source_document_map", "Source-to-canonical map"),
        (qwen_config, "qwen3_config", "Prepared Qwen configuration"),
        (manifest, "baseline_manifest", "Baseline manifest"),
        (run, "baseline_run", "Baseline run"),
        (evaluation, "baseline_evaluation", "Baseline evaluation"),
        (passage_mapping, "passage_mapping", "Passage mapping"),
        (canonical_dataset, "evaluation_dataset", "Canonical evaluation dataset"),
    ):
        _require_digest(data, PUBLISHED[key], name)

    receipt = verify_ir_bundle(baseline_root, expected_run_sha256=PUBLISHED["baseline_run"])
    evaluation_result = read_verified_ir_evaluation(
        baseline_root, expected_run_sha256=PUBLISHED["baseline_run"]
    )
    metrics = {row.measure: row.value for row in evaluation_result.aggregate}
    if set(metrics) != set(BASELINE_METRICS) or any(
        abs(metrics[name] - expected) > 1e-12 for name, expected in BASELINE_METRICS.items()
    ):
        raise IrContractError(
            "Reproduced RES-140 baseline metrics differ from the published values"
        )
    return {
        "manifest_sha256": receipt.manifest_sha256,
        "run_sha256": receipt.run_sha256,
        "evaluation_sha256": receipt.evaluation_sha256,
        "dataset_sha256": receipt.dataset_sha256,
        "config_sha256": receipt.config_sha256,
        "passage_mapping_sha256": receipt.passage_mapping_sha256,
        "metrics": metrics,
        "query_count": len(evaluation_result.per_query),
    }


def _published_sources(repo_root: Path) -> dict[str, Path]:
    return {name: repo_root / source for name, source in _PACKAGE_FILES.items()}


def _package_payload(repo_root: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    source_root = repo_root / SOURCE_ROOT
    baseline_root = source_root / "baseline"
    source_archive = (repo_root / f"{SCIFACT_ROOT}.zip").read_bytes()
    materialized_manifest = (repo_root / f"{SCIFACT_ROOT}/manifest.json").read_bytes()
    selection_bytes = (repo_root / f"{SCIFACT_ROOT}/pilot-query-selection.json").read_bytes()
    source_dataset = (source_root / "source-dataset.json").read_bytes()
    pilot_dataset = (source_root / "source-pilot-dataset.json").read_bytes()
    map_bytes = (source_root / "qrel-source-document-identity-map.json").read_bytes()
    selected_projection = (
        repo_root / _PACKAGE_FILES["identity/selected-qrel-identity-projection.json"]
    ).read_bytes()
    for data, key, name in (
        (source_archive, "source_archive", "Source archive"),
        (materialized_manifest, "source_manifest", "Source manifest"),
        (selection_bytes, "pilot_selection", "Pilot selection"),
        (source_dataset, "source_dataset", "Full source dataset"),
        (pilot_dataset, "source_pilot_dataset", "Source pilot dataset"),
        (map_bytes, "source_document_map", "Source-to-canonical map"),
    ):
        _require_digest(data, PUBLISHED[key], name)
    selected_ids = _read_json(selection_bytes, "Pilot selection").get("selected_query_ids")
    if not isinstance(selected_ids, list):
        raise IrContractError("Frozen query selection is invalid")
    selection_payload = _read_json(selection_bytes, "Pilot selection")
    if selection_payload.get("selected_query_ids_sha256") != PUBLISHED["selected_query_ids"]:
        raise IrContractError("Selected query ID digest differs from its published identity")
    lineage = _check_source_lineage(
        source_dataset_bytes=source_dataset,
        pilot_dataset_bytes=pilot_dataset,
        selection_bytes=selection_bytes,
        map_bytes=map_bytes,
        selected_projection_bytes=selected_projection,
        source_archive_bytes=source_archive,
        canonical_dataset_bytes=(baseline_root / "dataset.json").read_bytes(),
    )
    baseline = verify_baseline_evidence(baseline_root, source_root)

    map_payload = _read_json(map_bytes, "Source-to-canonical map")
    identity = {
        str(row["source_document_id"]): str(row["canonical_document_id"])
        for row in cast("list[dict[str, Any]]", map_payload["entries"])
    }
    candidate_zip = repo_root / _PACKAGE_FILES["b2/res142-b2-candidate-input-v3.zip"]
    sidecar = repo_root / _PACKAGE_FILES["b2/rights-confirmation-v3.json"]
    candidate = _read_candidate_archive(
        candidate_zip,
        expected_archive_sha256=PUBLISHED["candidate_zip"],
        expected_sidecar_sha256=PUBLISHED["rights_sidecar"],
        rights_sidecar_path=sidecar,
        source_to_canonical=identity,
    )
    retrieval = repo_root / RETRIEVAL_EVIDENCE
    _require_digest(retrieval.read_bytes(), PUBLISHED["retrieval_evidence"], "Retrieval evidence")
    with zipfile.ZipFile(retrieval) as evidence_zip:
        summary = evidence_zip.read("retrieval-summary.json")
        validation = evidence_zip.read("validation/saved-rrf-validation.json")
    _require_digest(summary, PUBLISHED["retrieval_summary"], "Retrieval summary")
    _require_digest(validation, PUBLISHED["retrieval_validation"], "Saved RRF validation")
    with zipfile.ZipFile(candidate_zip) as candidate_archive:
        candidate_manifest = _read_json(candidate_archive.read("manifest.json"), "B2 manifest")
    if candidate_manifest.get("retrieval_validation_sha256") != PUBLISHED["retrieval_validation"]:
        raise IrContractError("B2 candidate does not bind the published saved-RRF validation")

    payload = {name: path.read_bytes() for name, path in _published_sources(repo_root).items()}
    payload["retrieval/retrieval-summary.json"] = summary
    payload["retrieval/saved-rrf-validation.json"] = validation
    files = {
        name: {"size_bytes": len(data), "sha256": _sha256(data)}
        for name, data in sorted(payload.items())
    }
    receipt = {
        "schema_revision": "res142-final-audit-package-v1",
        "source_head_sha": "85024771215947e0315f37a6dd67d1e8d0575eff",
        "source_build_code_sha": "b748cfcf3c3f6aff9c8aa167864d80fa34f4be1a",
        "published_identities": PUBLISHED,
        "source_lineage": lineage,
        "baseline": baseline,
        "candidate_transfer": candidate,
        "files": files,
    }
    payload["audit-receipt.json"] = _canonical_json(receipt)
    return payload, receipt


def _canonical_json(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _publish_create_only(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise IrContractError("refusing to overwrite an existing audit artifact")
        os.link(temp_path, path)
        temp_path.unlink()
        path.chmod(stat.S_IREAD)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def export_audit_package(repo_root: Path, output_path: Path) -> dict[str, Any]:
    """Export a self-contained, create-only ZIP and detached audit receipt."""
    destination = output_path.resolve(strict=False)
    receipt_path = destination.with_suffix(destination.suffix + ".receipt.json")
    if destination.exists() or receipt_path.exists():
        raise IrContractError("refusing to overwrite an existing audit package or receipt")
    payload, receipt = _package_payload(repo_root.resolve())
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w+b") as raw:
            with zipfile.ZipFile(raw, "w") as archive:
                for name, data in sorted(payload.items()):
                    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o444 << 16
                    archive.writestr(info, data)
            raw.flush()
            os.fsync(raw.fileno())
        package_bytes = temp_path.read_bytes()
        package_sha256 = _sha256(package_bytes)
        _publish_create_only(destination, package_bytes)
        external_receipt = {
            "schema_revision": "res142-final-audit-receipt-v1",
            "audit_package": str(destination),
            "audit_package_sha256": package_sha256,
            "embedded_receipt_sha256": _sha256(payload["audit-receipt.json"]),
            "baseline_metrics": receipt["baseline"]["metrics"],
            "source_document_map_rows": receipt["source_lineage"]["source_document_map_rows"],
            "candidate_zip_sha256": receipt["candidate_transfer"]["archive_sha256"],
            "published_identities": PUBLISHED,
        }
        _publish_create_only(receipt_path, _canonical_json(external_receipt))
    finally:
        temp_path.unlink(missing_ok=True)
    return external_receipt


def verify_audit_package(package_path: Path) -> dict[str, Any]:  # noqa: PLR0912, PLR0915
    """Verify a sealed package without access to the original working directory."""
    with zipfile.ZipFile(package_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or any(
            info.is_dir()
            or PurePosixPath(info.filename).is_absolute()
            or ".." in PurePosixPath(info.filename).parts
            for info in infos
        ):
            raise IrContractError("Audit package contains duplicate or unsafe member paths")
        if "audit-receipt.json" not in names:
            raise IrContractError("Audit package has no embedded receipt")
        receipt_bytes = archive.read("audit-receipt.json")
        receipt = _read_json(receipt_bytes, "Embedded audit receipt")
        if _canonical_json(receipt) != receipt_bytes:
            raise IrContractError("Embedded audit receipt is not canonical JSON")
        files = receipt.get("files")
        if not isinstance(files, dict) or set(names) != set(files) | {"audit-receipt.json"}:
            raise IrContractError("Audit package inventory differs from the embedded receipt")
        if set(files) != set(_PACKAGE_FILES) | {
            "retrieval/retrieval-summary.json",
            "retrieval/saved-rrf-validation.json",
        }:
            raise IrContractError("Audit package is missing required baseline or provenance files")
        payload: dict[str, bytes] = {}
        for name, expected in files.items():
            data = archive.read(name)
            if (
                not isinstance(expected, dict)
                or expected.get("size_bytes") != len(data)
                or expected.get("sha256") != _sha256(data)
            ):
                raise IrContractError(f"Audit package member digest differs: {name}")
            payload[name] = data
    if receipt.get("published_identities") != PUBLISHED:
        raise IrContractError("Audit package published identity table differs from the verifier")
    with tempfile.TemporaryDirectory(prefix="res142-audit-verify-") as temp_name:
        root = Path(temp_name)
        for name, data in payload.items():
            target = root.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        baseline_inputs = root / "baseline-inputs"
        baseline_inputs.mkdir()
        for source_name, package_name in (
            ("source-dataset.json", "source/source-dataset.json"),
            ("source-pilot-dataset.json", "source/source-pilot-dataset.json"),
            (
                "qrel-source-document-identity-map.json",
                "identity/qrel-source-document-identity-map.json",
            ),
            ("candidate-snapshot.json", "provenance/candidate-snapshot.json"),
            ("qwen3-reranker-config.json", "provenance/qwen3-reranker-config.json"),
        ):
            (baseline_inputs / source_name).write_bytes(payload[package_name])
        baseline = verify_baseline_evidence(root / "baseline", baseline_inputs)
        candidate = verify_candidate_transfer(
            root / "b2/res142-b2-candidate-input-v3.zip",
            root / "b2/rights-confirmation-v3.json",
            root / "identity/qrel-source-document-identity-map.json",
        )
        source_archive_data = (root / "source/beir-scifact-test.zip").read_bytes()
        source_manifest = (root / "source/manifest.json").read_bytes()
        selection = (root / "source/pilot-query-selection.json").read_bytes()
        source_dataset = (root / "source/source-dataset.json").read_bytes()
        pilot_dataset = (root / "source/source-pilot-dataset.json").read_bytes()
        identity_map = (root / "identity/qrel-source-document-identity-map.json").read_bytes()
        projection = (root / "identity/selected-qrel-identity-projection.json").read_bytes()
        for data, key, label in (
            (source_archive_data, "source_archive", "Source archive"),
            (source_manifest, "source_manifest", "Source manifest"),
            (selection, "pilot_selection", "Pilot selection"),
            (source_dataset, "source_dataset", "Full source dataset"),
            (pilot_dataset, "source_pilot_dataset", "Source pilot dataset"),
            (identity_map, "source_document_map", "Source identity map"),
            (payload["provenance/candidate-snapshot.json"], "candidate_snapshot", "Snapshot"),
            (payload["provenance/qwen3-reranker-config.json"], "qwen3_config", "Qwen config"),
            (payload["retrieval/retrieval-summary.json"], "retrieval_summary", "Retrieval summary"),
            (
                payload["retrieval/saved-rrf-validation.json"],
                "retrieval_validation",
                "Saved RRF validation",
            ),
        ):
            _require_digest(data, PUBLISHED[key], label)
        lineage = _check_source_lineage(
            source_dataset_bytes=source_dataset,
            pilot_dataset_bytes=pilot_dataset,
            selection_bytes=selection,
            map_bytes=identity_map,
            selected_projection_bytes=projection,
            source_archive_bytes=source_archive_data,
            canonical_dataset_bytes=(root / "baseline/dataset.json").read_bytes(),
        )
    if baseline["metrics"] != receipt.get("baseline", {}).get("metrics"):
        raise IrContractError("Reproduced metrics differ from the embedded audit receipt")
    if candidate["archive_sha256"] != receipt.get("candidate_transfer", {}).get(
        "archive_sha256"
    ) or lineage["source_document_map_rows"] != receipt.get("source_lineage", {}).get(
        "source_document_map_rows"
    ):
        raise IrContractError("Reproduced input identities differ from the embedded audit receipt")
    detached_receipt_path = package_path.with_suffix(package_path.suffix + ".receipt.json")
    if detached_receipt_path.is_file():
        detached = _read_json(detached_receipt_path.read_bytes(), "Detached audit receipt")
        if detached.get("audit_package_sha256") != _sha256(
            package_path.read_bytes()
        ) or detached.get("embedded_receipt_sha256") != _sha256(receipt_bytes):
            raise IrContractError("Detached audit receipt does not authenticate this package")
    return {
        "audit_package_sha256": _sha256(package_path.read_bytes()),
        "source_document_map_rows": lineage["source_document_map_rows"],
        "baseline_metrics": baseline["metrics"],
        "candidate_zip_sha256": candidate["archive_sha256"],
        "status": "verified",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    export = subcommands.add_parser("export", help="create the local immutable audit package")
    export.add_argument("--repo-root", type=Path, default=Path.cwd())
    export.add_argument("--out", type=Path, required=True)
    verify = subcommands.add_parser("verify", help="verify an existing audit package offline")
    verify.add_argument("package", type=Path)
    args = parser.parse_args()
    try:
        result = (
            export_audit_package(args.repo_root, args.out)
            if args.command == "export"
            else verify_audit_package(args.package)
        )
    except (IrContractError, OSError, zipfile.BadZipFile) as error:
        parser.exit(1, f"res142-audit: {type(error).__name__}: {error}\n")
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
