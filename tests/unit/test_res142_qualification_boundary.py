"""Synthetic offline boundary tests: no private fixtures, network, GPU or inference."""

from __future__ import annotations

import json
import runpy
import shutil
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from dynamisrag.ir.contracts import IrContractError

qualification = cast(
    Any,
    SimpleNamespace(
        **runpy.run_path(
            str(Path(__file__).parents[2] / "scripts/res142_scientific_qualification.py")
        )
    ),
)


def _write(path: Path, value: object) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(data)
    return data


def _receipt(path: str, data: bytes) -> dict[str, object]:
    digest = qualification._sha256(data)
    return {
        "path": path,
        "size_bytes": len(data),
        "sha256": digest,
        "readback_sha256": digest,
        "status": "verified",
    }


@pytest.fixture
def evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, Any]]:
    root = tmp_path / "evidence"
    drive = root / "drive-readback"
    queries = [{"query_id": f"q{i:02d}"} for i in range(20)]
    results = _write(root / "results.json", {"queries": queries})
    monkeypatch.setitem(
        qualification.qualify_res142.__globals__, "_RESULTS_SHA256", qualification._sha256(results)
    )
    persisted = {"scores/results.json": _receipt("scores/results.json", results)}
    for directory, names in qualification._DRIVE_READBACK_FILES.items():
        for name in names:
            relative = f"{directory}/{name}"
            persisted[relative] = _receipt(relative, _write(drive / relative, {}))
    persisted["manifest-initial.json"] = _receipt(
        "manifest-initial.json", _write(drive / "manifest-initial.json", {})
    )
    for index, query in enumerate(queries, 1):
        relative = f"queries/{index:02d}-{query['query_id']}.json"
        persisted[relative] = _receipt(relative, _write(drive / relative, query))
    checkpoints: list[dict[str, object]] = []
    for ordinal in range(1, 27):
        relative = f"checkpoints/{ordinal:04d}-test.json"
        entry = _receipt(
            relative, _write(root / relative, {"checkpoint": ordinal, "run_id": "run"})
        )
        persisted[relative] = entry
        checkpoints.append(entry)
    manifest: dict[str, Any] = {
        "schema_revision": "res142-colab-b2-manifest-v1",
        "status": "success",
        "pair_count": 1000,
        "query_count": 20,
        "request_count": 140,
        "retry_count": 0,
        "results_sha256": qualification._sha256(results),
        "run_id": "run",
        "drive_storage": {
            "persisted_artifacts": persisted,
            "checkpoint_count": 26,
            "checkpoints": checkpoints,
        },
    }
    manifest_bytes = _write(root / "manifest-final.json", manifest)
    with zipfile.ZipFile(root / "evidence.zip", "w") as archive:
        archive.writestr("manifest-final.json", manifest_bytes)
        archive.writestr("scores/results.json", results)
        archive.writestr("vllm.log", b"synthetic")
        for member, relative in (
            ("import-smoke.json", "environment/import-smoke.json"),
            ("pip-freeze.txt", "environment/pip-freeze.txt"),
            ("model-snapshot.json", "model/model-snapshot.json"),
        ):
            archive.writestr(member, (drive / relative).read_bytes())
        for path in (drive / "queries").iterdir():
            archive.writestr(f"scores/queries/{path.name}", path.read_bytes())
    relative = "checkpoints/0027-test.json"
    checkpoint = {
        "checkpoint": 27,
        "run_id": "run",
        "stage": "b2-results-complete",
        "persistence_status": "verified",
        "persisted_artifacts": persisted,
        "payload": {
            "candidate_count": 1000,
            "request_count": 140,
            "results_sha256": qualification._sha256(results),
        },
    }
    final_map = {**persisted, relative: _receipt(relative, _write(root / relative, checkpoint))}
    for name in ("manifest-final.json", "evidence.zip"):
        final_map[name] = _receipt(name, (root / name).read_bytes())
    _write(
        root / "checkpoints/0028-test.json",
        {
            "checkpoint": 28,
            "run_id": "run",
            "stage": "final-outcome",
            "persistence_status": "verified",
            "persisted_artifacts": final_map,
            "payload": {
                "errors": [],
                "persistence_status": "complete",
                "run_id": "run",
                "persisted": [final_map[name] for name in ("manifest-final.json", "evidence.zip")],
            },
        },
    )
    return root, manifest


def test_synthetic_evidence_passes_real_zip_and_drive_verifiers(
    evidence: tuple[Path, dict[str, Any]],
) -> None:
    root, manifest = evidence
    drive = root / "drive-readback"
    assert qualification._verify_drive_readback(root, drive, manifest)["checkpoint_count"] == 28
    assert (
        qualification._verify_evidence_zip(
            root,
            drive,
            (root / "manifest-final.json").read_bytes(),
            (root / "results.json").read_bytes(),
            qualification._query_result_names(drive),
        )["zip_crc_pass"]
        is True
    )


@pytest.mark.parametrize(
    "attack",
    ["result-bytes", "drive-receipt", "missing-checkpoint", "duplicate-checkpoint", "corrupt-zip"],
)
def test_qualification_fails_closed_before_publication(
    evidence: tuple[Path, dict[str, Any]],
    tmp_path: Path,
    attack: str,
) -> None:
    root, manifest = evidence
    if attack == "result-bytes":
        with (root / "results.json").open("ab") as handle:
            handle.write(b" ")
    elif attack == "drive-receipt":
        manifest["drive_storage"]["persisted_artifacts"]["manifest-initial.json"]["sha256"] = (
            "0" * 64
        )
        raw = _write(root / "manifest-final.json", manifest)
        # Keep the ZIP copy consistent so the real receipt verifier must reject it.
        with zipfile.ZipFile(root / "evidence.zip") as archive:
            members = {name: archive.read(name) for name in archive.namelist()}
        members["manifest-final.json"] = raw
        with zipfile.ZipFile(root / "evidence.zip", "w") as archive:
            for name, data in members.items():
                archive.writestr(name, data)
    elif attack == "missing-checkpoint":
        (root / "checkpoints/0028-test.json").unlink()
    elif attack == "duplicate-checkpoint":
        shutil.copyfile(
            root / "checkpoints/0027-test.json", root / "checkpoints/0027-duplicate.json"
        )
        (root / "checkpoints/0028-test.json").unlink()
    else:
        (root / "evidence.zip").write_bytes(b"corrupt ZIP")
    audit = tmp_path / "audit.zip"
    audit.write_bytes(b"not reached")
    output = tmp_path / "public"
    private = tmp_path / "private"
    messages = {
        "result-bytes": "pinned successful",
        "drive-receipt": "persisted receipt",
        "missing-checkpoint": "26 indexed and two closure",
        "duplicate-checkpoint": "missing or duplicated ordinal",
        "corrupt-zip": "File is not a zip file",
    }
    with pytest.raises((IrContractError, zipfile.BadZipFile), match=messages[attack]):
        qualification.qualify_res142(root, audit, output, private_output=private)
    assert not output.exists()
    assert not private.exists()


def test_changed_frozen_candidate_identity_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "candidate.zip"

    def candidate_archive(passage_id: str) -> bytes:
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("candidate-input.json", json.dumps({"passage_id": passage_id}))
            archive.writestr("pilot-query-selection.json", "{}")
        return archive_path.read_bytes()

    frozen = candidate_archive("frozen-passage")
    monkeypatch.setitem(
        qualification._read_candidate_input.__globals__,
        "_CANDIDATE_ZIP_SHA256",
        qualification._sha256(frozen),
    )
    assert qualification._read_candidate_input(frozen)[0]["passage_id"] == "frozen-passage"
    with pytest.raises(IrContractError, match="frozen transfer"):
        qualification._read_candidate_input(candidate_archive("changed-passage"))


def test_mismatched_baseline_authority_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    def mismatched_receipt(_: Path) -> dict[str, object]:
        return {
            "status": "verified",
            "candidate_zip_sha256": qualification._CANDIDATE_ZIP_SHA256,
            "source_document_map_rows": 5183,
            "baseline_metrics": {"MAP": 999.0},
        }

    monkeypatch.setitem(
        qualification._verify_baseline_and_b2.__globals__,
        "_verify_audit_package",
        mismatched_receipt,
    )
    with pytest.raises(IrContractError, match="pinned study authority"):
        qualification._verify_baseline_and_b2(
            Path("unused"), Path("unused"), {}, {}, seed=1422026, bootstrap_samples=1
        )


@pytest.mark.parametrize("existing", ["empty-directory", "populated-directory", "file"])
def test_publication_never_replaces_existing_destination(tmp_path: Path, existing: str) -> None:
    output = tmp_path / "output"
    if existing == "file":
        output.write_bytes(b"owner")
    else:
        output.mkdir()
        if existing == "populated-directory":
            (output / "owner").write_bytes(b"owner")
    with pytest.raises(IrContractError, match="overwrite"):
        qualification._publish_create_only(output, {"evaluation.json": b"new"})
    assert (
        output.read_bytes() == b"owner"
        if output.is_file()
        else not (output / "evaluation.json").exists()
    )


def test_publication_rejects_concurrent_empty_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    original_mkdir = Path.mkdir

    def race(path: Path, *args: Any, **kwargs: Any) -> None:
        if path == output:
            original_mkdir(path)  # A concurrent publisher wins immediately before reservation.
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", race)
    with pytest.raises(IrContractError, match="appeared during creation"):
        qualification._publish_create_only(output, {"evaluation.json": b"new"})
    assert output.is_dir()
    assert list(output.iterdir()) == []
    assert sorted(path.name for path in tmp_path.iterdir()) == ["output"]


def test_publication_and_public_sanitization(tmp_path: Path) -> None:
    evaluation: dict[str, Any] = dict.fromkeys(
        (
            "schema_revision",
            "source_identities",
            "protocol",
            "decision",
            "query_count",
            "metrics",
            "ranking_summary",
            "candidate_pool_ceiling",
            "uncertainty_method",
        )
    )
    evaluation["paired_query_outcomes"] = {"positive_qrels": 1, "rrf_positive_relevant_ranks": [1]}
    public = qualification._public_evaluation(evaluation)
    assert "paired_query_outcomes" not in public
    assert "paired_query_outcomes" in evaluation
    output = tmp_path / "output"
    qualification._publish_create_only(output, {"evaluation.json": b"verified"})
    assert (output / "evaluation.json").read_bytes() == b"verified"


@pytest.mark.parametrize("private_name", ["public", "public/private", "."])
def test_public_and_private_outputs_must_be_separate_trees(
    tmp_path: Path,
    private_name: str,
) -> None:
    with pytest.raises(IrContractError, match="separate trees"):
        qualification.qualify_res142(
            tmp_path / "unused",
            tmp_path / "unused.zip",
            tmp_path / "public",
            private_output=tmp_path / private_name,
        )
    assert not (tmp_path / "public").exists()


def test_committed_public_report_is_sanitized_and_labels_nominal_intervals() -> None:
    root = Path(__file__).parents[2] / "reports/res142-b2-scientific-review"
    raw = (root / "evaluation.json").read_text(encoding="utf-8")
    evaluation = json.loads(raw)
    assert "paired_query_outcomes" not in evaluation
    for field in (
        "positive_qrels",
        "relevant_documents_in_candidate_pool",
        "rrf_positive_relevant_ranks",
        "qwen3_positive_relevant_ranks",
    ):
        assert field not in raw
    report = (root / "scientific-review.md").read_text(encoding="utf-8")
    assert "nominal, unadjusted 95%" in report
    assert "populations. No metric" in report
    assert evaluation["metrics"]["nDCG@10"]["exact_sign_flip_test_two_sided"]["p_value"] == 0.03125
    assert (
        evaluation["metrics"]["nDCG@10"]["exact_sign_flip_test_two_sided"]["holm_adjusted_p_value"]
        == 0.125
    )
