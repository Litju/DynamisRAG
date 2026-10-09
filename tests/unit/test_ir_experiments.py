"""RES-140 offline scoring, comparability and deterministic result diffs."""

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
    read_verified_ir_evaluation,
    verify_ir_bundle,
    write_ir_bundle,
)
from dynamisrag.ir.experiments import (
    compare_ir_evaluations,
    score_ir_inputs,
    verify_ir_comparison,
    write_ir_comparison,
)

_RUN_SHA = "a64f361ed29b25ce57db9554dcb403d810ed4b5f94a5a7d55fc5f63066eb30f2"
_PARQUET = cast(Any, pq)


def _candidates() -> tuple[IrDataset, IrExperimentConfig, IrRun, IrExperimentConfig, IrRun]:
    dataset = IrDataset(
        "synthetic:comparison",
        "synthetic-v1",
        "a" * 64,
        (IrQuery("q1", "graded query"), IrQuery("q2", "zero-positive query")),
        (IrQrel("q1", "d1", 2), IrQrel("q1", "d2", 0), IrQrel("q2", "d3", -1)),
    )
    baseline_config = IrExperimentConfig(
        dataset.sha256,
        "b" * 40,
        "bm25-v1",
        "c" * 64,
        '{"depth":50}',
    )
    candidate_config = dataclasses.replace(baseline_config, retrieval_revision="dense-v1")
    baseline = IrRun(
        baseline_config.sha256,
        dataset.sha256,
        ("q1", "q2"),
        (IrHit("q1", "d1", 1, 0.1), IrHit("q1", "d2", 2, 900.0)),
        evaluation_depth=50,
    )
    candidate = IrRun(
        candidate_config.sha256,
        dataset.sha256,
        ("q1", "q2"),
        (IrHit("q1", "d2", 1, 700.0), IrHit("q1", "d1", 2, 0.2)),
        evaluation_depth=50,
    )
    return dataset, baseline_config, baseline, candidate_config, candidate


def test_complete_static_fixture_scores_offline_and_verifies(tmp_path: Path) -> None:
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "ir-res140"
    destination = tmp_path / "scored"

    receipt = score_ir_inputs(fixture, destination, expected_run_sha256=_RUN_SHA)

    assert receipt.run_sha256 == _RUN_SHA
    assert verify_ir_bundle(destination, expected_run_sha256=_RUN_SHA) == receipt
    evaluation = read_verified_ir_evaluation(destination, expected_run_sha256=_RUN_SHA)
    assert [row.measure for row in evaluation.aggregate] == ["nDCG@10", "Recall@10", "MAP", "MRR"]
    assert evaluation.per_query[0].qrel_document_count == 2
    assert {
        name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
        for name in ("per-query.parquet", "aggregate.parquet", "ranked-run.parquet")
    } == {
        "per-query.parquet": "f676ab9050bc5ca2dfa6696fbf2cc3436e1ff4f062ce409e2539841c33e8ce69",
        "aggregate.parquet": "6291f2cb9e693e2414dadcbfb37e9bb2c51be789d39726724b4abaa0fcfdeedb",
        "ranked-run.parquet": "76eeda62ba46ffa5fd1f7283d6e82f096a778d202fca58e633d5701917a8e49a",
    }
    with pytest.raises(IrContractError, match="expected run"):
        score_ir_inputs(fixture, tmp_path / "wrong-run", expected_run_sha256="f" * 64)


def test_candidate_comparison_requires_matching_scientific_boundary(tmp_path: Path) -> None:
    dataset, base_config, base_run, candidate_config, candidate_run = _candidates()
    base_path = tmp_path / "baseline"
    candidate_path = tmp_path / "candidate"
    write_ir_bundle(base_path, dataset=dataset, config=base_config, run=base_run)
    write_ir_bundle(candidate_path, dataset=dataset, config=candidate_config, run=candidate_run)
    baseline = read_verified_ir_evaluation(base_path, expected_run_sha256=base_run.sha256)
    candidate = read_verified_ir_evaluation(
        candidate_path, expected_run_sha256=candidate_run.sha256
    )

    comparison = compare_ir_evaluations(baseline, candidate)
    first = write_ir_comparison(tmp_path / "diff-a", baseline=baseline, candidate=candidate)
    second = write_ir_comparison(tmp_path / "diff-b", baseline=baseline, candidate=candidate)

    assert first.comparison_sha256 == comparison.sha256
    assert first.manifest_sha256 == second.manifest_sha256
    assert (
        verify_ir_comparison(
            first.root,
            baseline=baseline,
            candidate=candidate,
            expected_comparison_sha256=comparison.sha256,
        )
        == first
    )
    assert comparison.aggregate[0].difference < 0
    json_rows = json.loads((first.root / "per-query-delta.json").read_bytes())["rows"]
    parquet_rows: Any = _PARQUET.read_table(first.root / "per-query-delta.parquet").to_pylist()
    assert json_rows == parquet_rows
    assert (first.root / "per-query-delta.parquet").read_bytes() == (
        second.root / "per-query-delta.parquet"
    ).read_bytes()

    with pytest.raises(IrContractError, match="index snapshot differs"):
        compare_ir_evaluations(baseline, dataclasses.replace(candidate, projection_sha256="d" * 64))


def test_comparison_verifier_rejects_resigned_swapped_rows(tmp_path: Path) -> None:
    dataset, base_config, base_run, candidate_config, candidate_run = _candidates()
    base_path = tmp_path / "baseline"
    candidate_path = tmp_path / "candidate"
    write_ir_bundle(base_path, dataset=dataset, config=base_config, run=base_run)
    write_ir_bundle(candidate_path, dataset=dataset, config=candidate_config, run=candidate_run)
    baseline = read_verified_ir_evaluation(base_path, expected_run_sha256=base_run.sha256)
    candidate = read_verified_ir_evaluation(
        candidate_path, expected_run_sha256=candidate_run.sha256
    )
    receipt = write_ir_comparison(tmp_path / "diff", baseline=baseline, candidate=candidate)

    changed = canonical_ir_json({"schema_revision": "ir-query-delta-v1", "rows": []})
    (receipt.root / "per-query-delta.json").write_bytes(changed)

    with pytest.raises(IrContractError):
        verify_ir_comparison(
            receipt.root,
            baseline=baseline,
            candidate=candidate,
            expected_comparison_sha256=receipt.comparison_sha256,
        )
