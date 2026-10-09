"""RES-140: atomically publish and independently verify deterministic IR evidence."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq
import pytest

from dynamisrag.ir import (
    IrContractError,
    IrDataset,
    IrExperimentConfig,
    IrHit,
    IrQrel,
    IrQuery,
    IrRun,
    canonical_ir_json,
    verify_ir_bundle,
    write_ir_bundle,
)

_PARQUET = cast(Any, pq)


def _inputs() -> tuple[IrDataset, IrExperimentConfig, IrRun]:
    dataset = IrDataset(
        source_id="scifact/test",
        source_revision="freeze-v1",
        corpus_sha256="a" * 64,
        queries=(IrQuery("q1", "Scientific evidence?"), IrQuery("q2", "Other study")),
        qrels=(IrQrel("q1", "d1", 2),),
    )
    config = IrExperimentConfig(
        dataset_sha256=dataset.sha256,
        code_sha="b" * 40,
        retrieval_revision="hybrid-rrf-v1",
        projection_sha256="c" * 64,
        parameters_json='{"k":60}',
    )
    run = IrRun(
        dataset_sha256=dataset.sha256,
        config_sha256=config.sha256,
        query_ids=("q1", "q2"),
        hits=(IrHit("q1", "d1", 1, 1.0), IrHit("q1", "d2", 2, -2.0)),
        evaluation_depth=50,
    )
    return dataset, config, run


def test_bundle_is_reproducible_across_paths_and_verifiable(tmp_path: Path) -> None:
    ds, cfg, run = _inputs()
    first = write_ir_bundle(tmp_path / "bundle-a", dataset=ds, config=cfg, run=run)
    second = write_ir_bundle(tmp_path / "bundle-b", dataset=ds, config=cfg, run=run)
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.run_sha256 == run.sha256
    assert verify_ir_bundle(first.root, expected_run_sha256=run.sha256) == first
    expected_inventory = {
        "manifest.json",
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
    }
    assert {path.name for path in first.root.iterdir()} == expected_inventory
    for stem, revision in (
        ("per-query", "ir-per-query-v1"),
        ("aggregate", "ir-aggregate-v1"),
        ("ranked-run", "ir-ranked-run-v1"),
    ):
        parquet_path = first.root / f"{stem}.parquet"
        table: Any = _PARQUET.read_table(parquet_path)
        json_table = json.loads((first.root / f"{stem}.json").read_bytes())
        assert table.to_pylist() == json_table["rows"]
        assert table.schema.metadata[b"dynamisrag.schema_revision"].decode() == revision
        assert table.schema.metadata[b"dynamisrag.pyarrow_version"] == b"25.0.1"
        assert parquet_path.read_bytes() == (second.root / parquet_path.name).read_bytes()
    assert (first.root / "run.trec").read_text(encoding="utf-8") == (
        "q1 Q0 d1 1 -1 dynamisrag\nq1 Q0 d2 2 -2 dynamisrag\n"
    )
    assert (first.root / "qrels.trec").read_text(encoding="utf-8") == "q1 0 d1 2\n"


def test_source_exhaustion_evidence_is_sealed_with_the_run(tmp_path: Path) -> None:
    dataset, config, original_run = _inputs()
    run = dataclasses.replace(original_run, source_exhausted_query_ids=("q1",))
    receipt = write_ir_bundle(tmp_path / "bundle", dataset=dataset, config=config, run=run)

    assert json.loads((receipt.root / "run.json").read_bytes())["source_exhausted_query_ids"] == [
        "q1"
    ]
    assert verify_ir_bundle(receipt.root, expected_run_sha256=run.sha256) == receipt


def test_existing_bundle_cannot_be_overwritten(tmp_path: Path) -> None:
    ds, cfg, run = _inputs()
    path = tmp_path / "bundle"
    write_ir_bundle(path, dataset=ds, config=cfg, run=run)
    with pytest.raises(IrContractError, match="overwrite"):
        write_ir_bundle(path, dataset=ds, config=cfg, run=run)


def test_bundle_detects_replaced_bytes_even_if_json_is_still_valid(tmp_path: Path) -> None:
    ds, cfg, run = _inputs()
    path = tmp_path / "bundle"
    write_ir_bundle(path, dataset=ds, config=cfg, run=run)
    (path / "run.json").write_bytes(b'{"changed":true}\n')
    with pytest.raises(IrContractError, match="content"):
        verify_ir_bundle(path, expected_run_sha256=run.sha256)


def test_manifest_and_caller_identity_both_required(tmp_path: Path) -> None:
    ds, cfg, run = _inputs()
    path = tmp_path / "bundle"
    write_ir_bundle(path, dataset=ds, config=cfg, run=run)
    with pytest.raises(IrContractError, match="expectations"):
        verify_ir_bundle(path, expected_run_sha256="f" * 64)
    (path / "rogue.txt").write_text("untracked")
    with pytest.raises(IrContractError, match="inventory"):
        verify_ir_bundle(path, expected_run_sha256=run.sha256)


def _replace_manifested_json(path: Path, name: str, payload: dict[str, object]) -> None:
    content = canonical_ir_json(payload)
    (path / name).write_bytes(content)
    manifest = json.loads((path / "manifest.json").read_bytes())
    entry = next(item for item in manifest["files"] if item["name"] == name)
    entry["size_bytes"] = len(content)
    entry["sha256"] = hashlib.sha256(content).hexdigest()
    (path / "manifest.json").write_bytes(canonical_ir_json(manifest))


@pytest.mark.parametrize("name", ["config.json", "dataset.json", "run.json"])
def test_bundle_rejects_changed_payload_revision_even_with_resigned_manifest(
    tmp_path: Path, name: str
) -> None:
    dataset, config, run = _inputs()
    path = tmp_path / "bundle"
    write_ir_bundle(path, dataset=dataset, config=config, run=run)
    payload = json.loads((path / name).read_bytes())
    payload["revision"] = "future-ir-contract"
    _replace_manifested_json(path, name, payload)

    with pytest.raises(IrContractError):
        verify_ir_bundle(path, expected_run_sha256=run.sha256)


def test_bundle_rejects_unrecognized_manifest_and_inventory_fields(tmp_path: Path) -> None:
    dataset, config, run = _inputs()
    path = tmp_path / "bundle"
    write_ir_bundle(path, dataset=dataset, config=config, run=run)
    manifest = json.loads((path / "manifest.json").read_bytes())
    manifest["host"] = "agent-local"
    (path / "manifest.json").write_bytes(canonical_ir_json(manifest))
    with pytest.raises(IrContractError, match="manifest fields"):
        verify_ir_bundle(path, expected_run_sha256=run.sha256)

    del manifest["host"]
    manifest["files"][0]["path"] = "C:/private/run.json"
    (path / "manifest.json").write_bytes(canonical_ir_json(manifest))
    with pytest.raises(IrContractError, match="inventory entry"):
        verify_ir_bundle(path, expected_run_sha256=run.sha256)


def test_bundle_rejects_directory_in_place_of_payload_file(tmp_path: Path) -> None:
    dataset, config, run = _inputs()
    path = tmp_path / "bundle"
    write_ir_bundle(path, dataset=dataset, config=config, run=run)
    (path / "config.json").unlink()
    (path / "config.json").mkdir()

    with pytest.raises(IrContractError, match="regular files"):
        verify_ir_bundle(path, expected_run_sha256=run.sha256)


def test_bundle_rejects_swapped_parquet_even_when_file_digest_is_resigned(tmp_path: Path) -> None:
    dataset, config, first_run = _inputs()
    first_path = tmp_path / "first"
    write_ir_bundle(first_path, dataset=dataset, config=config, run=first_run)
    second_run = IrRun(
        dataset_sha256=dataset.sha256,
        config_sha256=config.sha256,
        query_ids=("q1", "q2"),
        hits=(IrHit("q1", "d2", 1, 1.0), IrHit("q1", "d1", 2, -2.0)),
        evaluation_depth=50,
    )
    second_path = tmp_path / "second"
    write_ir_bundle(second_path, dataset=dataset, config=config, run=second_run)

    replacement = (second_path / "per-query.parquet").read_bytes()
    (first_path / "per-query.parquet").write_bytes(replacement)
    manifest = json.loads((first_path / "manifest.json").read_bytes())
    entry = next(item for item in manifest["files"] if item["name"] == "per-query.parquet")
    entry["size_bytes"] = len(replacement)
    entry["sha256"] = hashlib.sha256(replacement).hexdigest()
    (first_path / "manifest.json").write_bytes(canonical_ir_json(manifest))

    with pytest.raises(IrContractError, match="rows differ"):
        verify_ir_bundle(first_path, expected_run_sha256=first_run.sha256)


def test_incompatible_inputs_cannot_publish_a_partial_bundle(tmp_path: Path) -> None:
    ds, cfg, run = _inputs()
    invalid = dataclasses.replace(cfg, retrieval_revision="other")
    root = tmp_path / "bundle"
    with pytest.raises(IrContractError):
        write_ir_bundle(root, dataset=ds, config=invalid, run=run)
    assert not root.exists()
