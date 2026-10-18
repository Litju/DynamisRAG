"""Offline audit-package input validation tests; no retrieval or GPU calls."""

from __future__ import annotations

import hashlib
import json
import runpy
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

from dynamisrag.ir.contracts import IrContractError

_AUDIT_NAMESPACE = runpy.run_path(str(Path(__file__).parents[2] / "scripts" / "res142_audit.py"))
PUBLISHED = cast("dict[str, str]", _AUDIT_NAMESPACE["PUBLISHED"])
_read_candidate_archive = cast(
    "Callable[..., dict[str, Any]]", _AUDIT_NAMESPACE["_read_candidate_archive"]
)


def _write_candidate_archive(
    tmp_path: Path,
    *,
    bad_source_identity: bool = False,
    bad_rank: bool = False,
) -> tuple[Path, Path, dict[str, str]]:
    selected_ids = [f"q{index:02d}" for index in range(20)]
    identity = {"source-1": "canonical-1"}
    query_rows: list[dict[str, Any]] = []
    for query_id in selected_ids:
        query_text = f"claim {query_id}"
        candidates: list[dict[str, Any]] = []
        for rank in range(1, 51):
            passage_id = f"{query_id}-p{rank:02d}"
            text = f"abstract {query_id} {rank}"
            text_sha256 = hashlib.sha256(text.encode()).hexdigest()
            canonical_id = "wrong-canonical" if bad_source_identity and rank == 1 else "canonical-1"
            provenance = {
                "document_canonical_key": canonical_id,
                "passage_key": passage_id,
                "source_external_id": "source-1",
            }
            original_rank = 2 if bad_rank and query_id == "q00" and rank == 1 else rank
            original = {
                "candidate_text_sha256": text_sha256,
                "original_rrf_rank": original_rank,
                "original_rrf_score": 1 / rank,
                "passage_key": passage_id,
                "provenance": provenance,
            }
            candidates.append(
                {
                    "candidate_text_sha256": text_sha256,
                    "original_retrieval": original,
                    "original_rrf_rank": original_rank,
                    "passage_id": passage_id,
                    "query_id": query_id,
                    "text": text,
                }
            )
        query_rows.append(
            {
                "candidate_count": 50,
                "candidates": candidates,
                "query_id": query_id,
                "query_text": query_text,
                "query_text_sha256": hashlib.sha256(query_text.encode()).hexdigest(),
            }
        )
    candidate_input = {
        "candidate_count": 1000,
        "candidate_limit": 50,
        "candidate_snapshot_sha256": PUBLISHED["candidate_snapshot"],
        "passage_mapping_sha256": PUBLISHED["passage_mapping"],
        "projection_sha256": PUBLISHED["projection"],
        "queries_in_selection_order": query_rows,
        "query_count": 20,
    }
    contents = {
        "NOTICES.txt": b"SciFact and S2ORC source notices retained.\n",
        "RES142_TRANSFER_DECISION.txt": b"Requester-confirmed private-run transfer.\n",
        "candidate-input.json": json.dumps(candidate_input, sort_keys=True).encode(),
        "pilot-query-selection.json": json.dumps(
            {"selected_query_ids": selected_ids}, sort_keys=True
        ).encode(),
        "source-provenance.json": json.dumps({"corpus_document_count": 5183}).encode(),
    }
    manifest = {
        "candidate_count": 1000,
        "candidate_limit": 50,
        "candidate_snapshot_sha256": PUBLISHED["candidate_snapshot"],
        "evaluation_dataset_sha256": PUBLISHED["evaluation_dataset"],
        "files": {
            name: {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
            for name, data in contents.items()
        },
        "full_corpus_included": False,
        "passage_mapping_sha256": PUBLISHED["passage_mapping"],
        "projection_sha256": PUBLISHED["projection"],
        "qrel_identity_mapping_sha256": PUBLISHED["source_document_map"],
        "qrels_included": False,
        "query_count": 20,
        "rights_status": "requester-confirmed-for-this-private-run",
        "schema_revision": "res142-scifact-b2-candidate-transfer-manifest-v1",
        "source_input_zip_sha256": "d" * 64,
    }
    archive_path = tmp_path / "candidate.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
        archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True).encode())
    archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    rights = {
        "candidate_snapshot_sha256": PUBLISHED["candidate_snapshot"],
        "candidate_transfer_zip_sha256": archive_sha256,
        "drive_destination": (
            "/content/drive/MyDrive/DynamisRAG/RES-142/retrieval/runs/"
            "20261016-10bce56f6ea04d949355af34fba2698c/b2"
        ),
        "rights_status": "requester-confirmed-for-this-run",
        "schema_revision": "res142-rights-confirmation-v1",
        "source_input_zip_sha256": "d" * 64,
        "transfer_status": "prepared-locally; not uploaded",
        "user_confirmation": "we have the rights",
    }
    sidecar_path = tmp_path / "rights.json"
    sidecar_path.write_text(json.dumps(rights, sort_keys=True), encoding="utf-8")
    return archive_path, sidecar_path, identity


def _verify_fixture(
    tmp_path: Path, *, bad_source_identity: bool = False, bad_rank: bool = False
) -> dict[str, Any]:
    archive, sidecar, identity = _write_candidate_archive(
        tmp_path, bad_source_identity=bad_source_identity, bad_rank=bad_rank
    )
    return _read_candidate_archive(
        archive,
        expected_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        expected_sidecar_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        rights_sidecar_path=sidecar,
        source_to_canonical=identity,
    )


def test_candidate_transfer_validation_accepts_exact_mock_population(tmp_path: Path) -> None:
    report = _verify_fixture(tmp_path)
    assert report["query_count"] == 20
    assert report["candidate_count"] == 1000
    assert report["qrels_included"] is False
    assert report["full_corpus_included"] is False


def test_candidate_transfer_rejects_source_identity_mismatch(tmp_path: Path) -> None:
    archive, sidecar, identity = _write_candidate_archive(tmp_path, bad_source_identity=True)
    with pytest.raises(IrContractError, match="source identity"):
        _read_candidate_archive(
            archive,
            expected_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            expected_sidecar_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            rights_sidecar_path=sidecar,
            source_to_canonical=identity,
        )


def test_candidate_transfer_rejects_changed_original_rank(tmp_path: Path) -> None:
    archive, sidecar, identity = _write_candidate_archive(tmp_path, bad_rank=True)
    with pytest.raises(IrContractError, match="rank is invalid"):
        _read_candidate_archive(
            archive,
            expected_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            expected_sidecar_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            rights_sidecar_path=sidecar,
            source_to_canonical=identity,
        )


def test_candidate_transfer_rejects_reported_archive_digest_mismatch(tmp_path: Path) -> None:
    archive, sidecar, identity = _write_candidate_archive(tmp_path)
    with pytest.raises(IrContractError, match="ZIP SHA-256"):
        _read_candidate_archive(
            archive,
            expected_archive_sha256="0" * 64,
            expected_sidecar_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            rights_sidecar_path=sidecar,
            source_to_canonical=identity,
        )
