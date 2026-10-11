"""The ArguAna self-document rule at the *public* evaluation boundary (RES-141).

``beir.validate_run_protocol`` and ``beir.exclude_identical_document_hits`` are
correct in isolation, but they are opt-in helpers: nothing in ``ir score`` knew
the dataset declared the BEIR ignore-identical-ids policy, so a sealed ArguAna
run that retrieved a query's own document at rank 1 was scored like any other
run and could be reported as BEIR-comparable.

These tests drive the public entry points - ``ir score`` and
``datasets score-retrieval`` - rather than the helpers, because the boundary that
was missing is exactly the one a user crosses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from dynamisrag import __main__ as cli
from dynamisrag.datasets.beir import IGNORE_IDENTICAL_IDS_POLICY, DanglingQrelPolicy
from dynamisrag.datasets.errors import DatasetContractError
from dynamisrag.datasets.primitives import canonical_bytes
from dynamisrag.ir.contracts import IrExperimentConfig, IrHit, IrRun, canonical_ir_json
from tests.unit.dataset_support import (
    DATASET_FIXTURES,
    beir_source,
    synthetic_beir_spec,
    synthetic_split,
)

_DEPTH = 10
_DEEP = DATASET_FIXTURES / "beir-arguana-deep-mini" / "arguana"
_DEEP_MEMBERS = ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv")
_DEEP_CANDIDATES: dict[str, list[str]] = {
    "t1": ["t1", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10", "a11"],
    "t2": ["a12", "t2", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10"],
}


def _deep_source() -> Any:
    return beir_source(
        source_id="beir.arguana",
        prefix="arguana",
        fixture="beir-arguana-deep-mini",
        members=_DEEP_MEMBERS,
    )


def _deep_spec() -> Any:
    return synthetic_beir_spec(
        source_id="beir.arguana",
        splits={
            "test": synthetic_split(
                documents=14,
                documents_without_text=0,
                queries_in_archive=2,
                queries=2,
                qrels=2,
                min_relevance=1,
                max_relevance=1,
                dangling_policy=DanglingQrelPolicy.REFUSE,
            )
        },
        self_document_policy=IGNORE_IDENTICAL_IDS_POLICY,
    )


def _slice(tmp_path: Path) -> Path:
    """Write the deep ArguAna fixture as a sealed slice directory."""
    from dynamisrag.datasets.beir import build_beir_artifacts
    from dynamisrag.datasets.slices import write_slice_bundle

    artifacts = build_beir_artifacts(
        source=_deep_source(),
        spec=_deep_spec(),
        split="test",
        files={name: _DEEP.joinpath(*name.split("/")) for name in _DEEP_MEMBERS},
    )
    write_slice_bundle(tmp_path / "arguana-slice", artifacts.bundle())
    return tmp_path / "arguana-slice"


def _dataset_of(root: Path) -> Any:
    from dynamisrag.datasets.slices import (
        _restore_dataset_bytes,  # pyright: ignore[reportPrivateUsage]
    )

    return _restore_dataset_bytes((root / "dataset.json").read_bytes(), name="dataset.json")


def _write_inputs(
    root: Path,
    work: Path,
    *,
    candidates: dict[str, list[str]],
    run_documents: dict[str, list[str]],
    source_exhausted: tuple[str, ...] = (),
) -> str:
    """Write the closed IR input directory for one sealed run."""
    dataset = _dataset_of(root)
    config = IrExperimentConfig(
        dataset_sha256=dataset.sha256,
        code_sha="1" * 40,
        retrieval_revision="synthetic-retrieval-v1",
        projection_sha256="2" * 64,
        parameters_json="{}",
    )
    hits: list[IrHit] = []
    for query_id in sorted(run_documents):
        hits.extend(
            IrHit(
                query_id=query_id,
                document_id=document_id,
                rank=rank,
                raw_score=float(-rank),
            )
            for rank, document_id in enumerate(run_documents[query_id], start=1)
        )
    run = IrRun(
        config_sha256=config.sha256,
        dataset_sha256=dataset.sha256,
        query_ids=tuple(query.query_id for query in dataset.queries),
        hits=tuple(hits),
        evaluation_depth=_DEPTH,
        source_exhausted_query_ids=source_exhausted,
    )
    work.mkdir(parents=True, exist_ok=True)
    (work / "dataset.json").write_bytes((root / "dataset.json").read_bytes())
    (work / "config.json").write_bytes(canonical_ir_json(config.payload()))
    (work / "run.json").write_bytes(canonical_ir_json(run.payload()))
    (work / "passage-mapping.json").write_bytes(
        canonical_ir_json({"revision": "ir-passage-mapping-v1", "entries": []})
    )
    return run.sha256


def _candidate_evidence(root: Path, work: Path, candidates: dict[str, list[str]]) -> Path:
    """Write the complete untruncated candidate prefix the adapter retrieved."""
    path = work / "candidates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        canonical_bytes(
            {
                "artifact_revision": "res141-candidate-evidence-v1",
                "policy": IGNORE_IDENTICAL_IDS_POLICY,
                "dataset_sha256": _dataset_of(root).sha256,
                "evaluation_depth": _DEPTH,
                "candidates": candidates,
            }
        )
    )
    return path


def _reference_run_documents(candidates: dict[str, list[str]]) -> dict[str, list[str]]:
    """What the reference BEIR rule plus depth truncation would evaluate."""
    return {
        query_id: [document_id for document_id in documents if document_id != query_id][:_DEPTH]
        for query_id, documents in sorted(candidates.items())
    }


def _score_retrieval(
    root: Path,
    inputs: Path,
    run_sha256: str,
    out: Path,
    candidates: Path,
    *extra: str,
) -> list[str]:
    return [
        "datasets",
        "score-retrieval",
        "--slice",
        str(root),
        "--inputs",
        str(inputs),
        "--run-sha256",
        run_sha256,
        "--out",
        str(out),
        "--candidates",
        str(candidates),
        *extra,
    ]


def _ir_score(inputs: Path, run_sha256: str, out: Path) -> list[str]:
    return [
        "ir",
        "score",
        "--inputs",
        str(inputs),
        "--run-sha256",
        run_sha256,
        "--out",
        str(out),
    ]


def test_a_generic_ir_score_refuses_a_dataset_that_declares_the_protocol(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Before: any sealed ArguAna run scored through `ir score`, self-hit included."""
    root = _slice(tmp_path)
    run_sha256 = _write_inputs(
        root,
        tmp_path / "inputs",
        candidates=_DEEP_CANDIDATES,
        run_documents={"t1": ["t1", "a1"], "t2": ["a12"]},
    )
    exit_code = cli.main(_ir_score(tmp_path / "inputs", run_sha256, tmp_path / "scored"))
    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert "score-retrieval" in stderr
    assert not (tmp_path / "scored").exists()


def test_the_qualified_workflow_scores_a_genuinely_filtered_candidate_prefix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Filtering happened on twelve untruncated candidates, before depth truncation."""
    root = _slice(tmp_path)
    documents = _reference_run_documents(_DEEP_CANDIDATES)
    run_sha256 = _write_inputs(
        root,
        tmp_path / "inputs",
        candidates=_DEEP_CANDIDATES,
        run_documents=documents,
    )
    evidence = _candidate_evidence(root, tmp_path, _DEEP_CANDIDATES)
    assert (
        cli.main(
            _score_retrieval(root, tmp_path / "inputs", run_sha256, tmp_path / "scored", evidence)
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    protocol = payload["protocol"]
    assert protocol["eligibility"] == "beir-protocol-comparable"
    assert protocol["policy"] == IGNORE_IDENTICAL_IDS_POLICY
    assert protocol["self_document_candidates_excluded"] == 2
    assert protocol["candidate_evidence_sha256"]
    assert payload["slice_verification"]["verification"] == "self-consistency"
    assert (tmp_path / "scored" / "manifest.json").is_file()


def test_the_qualified_workflow_refuses_candidates_truncated_before_filtering(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A cleaned top-10 alone does not prove the rule ran before truncation."""
    root = _slice(tmp_path)
    truncated = {query: documents[:_DEPTH] for query, documents in _DEEP_CANDIDATES.items()}
    documents = _reference_run_documents(truncated)
    run_sha256 = _write_inputs(
        root,
        tmp_path / "inputs",
        candidates=truncated,
        run_documents=documents,
    )
    evidence = _candidate_evidence(root, tmp_path, truncated)
    exit_code = cli.main(
        _score_retrieval(root, tmp_path / "inputs", run_sha256, tmp_path / "scored", evidence)
    )
    assert exit_code == 1
    assert "score-retrieval" in capsys.readouterr().err
    assert not (tmp_path / "scored").exists()


def test_source_exhaustion_is_accepted_instead_of_candidate_overflow(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A corpus smaller than the depth is complete when the run says it is exhausted."""
    root = _slice(tmp_path)
    candidates = {"t1": ["t1", "a1", "a2"], "t2": ["a12"]}
    documents = _reference_run_documents(candidates)
    run_sha256 = _write_inputs(
        root,
        tmp_path / "inputs",
        candidates=candidates,
        run_documents=documents,
        source_exhausted=("t1", "t2"),
    )
    evidence = _candidate_evidence(root, tmp_path, candidates)
    assert (
        cli.main(
            _score_retrieval(root, tmp_path / "inputs", run_sha256, tmp_path / "scored", evidence)
        )
        == 0
    )
    protocol = json.loads(capsys.readouterr().out)["protocol"]
    assert protocol["eligibility"] == "beir-protocol-comparable"
    assert protocol["self_document_candidates_excluded"] == 1


def test_a_self_hit_in_the_sealed_run_is_refused_even_with_candidate_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _slice(tmp_path)
    documents = _reference_run_documents(_DEEP_CANDIDATES)
    run_sha256 = _write_inputs(
        root,
        tmp_path / "inputs",
        candidates=_DEEP_CANDIDATES,
        run_documents={**documents, "t1": ["t1", *documents["t1"][:1]]},
    )
    evidence = _candidate_evidence(root, tmp_path, _DEEP_CANDIDATES)
    exit_code = cli.main(
        _score_retrieval(root, tmp_path / "inputs", run_sha256, tmp_path / "scored", evidence)
    )
    assert exit_code == 1
    assert "score-retrieval" in capsys.readouterr().err


def test_the_qualified_workflow_refuses_a_run_whose_prefix_is_not_the_reference_filter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A different order at equal depth means the run was not produced by the rule."""
    root = _slice(tmp_path)
    documents = _reference_run_documents(_DEEP_CANDIDATES)
    reordered = {**documents, "t1": list(reversed(documents["t1"]))}
    run_sha256 = _write_inputs(
        root,
        tmp_path / "inputs",
        candidates=_DEEP_CANDIDATES,
        run_documents=reordered,
    )
    evidence = _candidate_evidence(root, tmp_path, _DEEP_CANDIDATES)
    exit_code = cli.main(
        _score_retrieval(root, tmp_path / "inputs", run_sha256, tmp_path / "scored", evidence)
    )
    assert exit_code == 1
    assert "score-retrieval" in capsys.readouterr().err


def test_candidate_evidence_bound_to_another_dataset_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _slice(tmp_path)
    documents = _reference_run_documents(_DEEP_CANDIDATES)
    run_sha256 = _write_inputs(
        root,
        tmp_path / "inputs",
        candidates=_DEEP_CANDIDATES,
        run_documents=documents,
    )
    evidence = tmp_path / "candidates.json"
    evidence.write_bytes(
        canonical_bytes(
            {
                "artifact_revision": "res141-candidate-evidence-v1",
                "policy": IGNORE_IDENTICAL_IDS_POLICY,
                "dataset_sha256": "0" * 64,
                "evaluation_depth": _DEPTH,
                "candidates": _DEEP_CANDIDATES,
            }
        )
    )
    exit_code = cli.main(
        _score_retrieval(root, tmp_path / "inputs", run_sha256, tmp_path / "scored", evidence)
    )
    assert exit_code == 1
    assert "score-retrieval" in capsys.readouterr().err


def test_candidate_evidence_that_repeats_a_document_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _slice(tmp_path)
    documents = _reference_run_documents(_DEEP_CANDIDATES)
    run_sha256 = _write_inputs(
        root,
        tmp_path / "inputs",
        candidates=_DEEP_CANDIDATES,
        run_documents=documents,
    )
    repeated = {**{q: list(d) for q, d in _DEEP_CANDIDATES.items()}, "t1": ["t1", "t1", "a1"]}
    evidence = _candidate_evidence(root, tmp_path, repeated)
    exit_code = cli.main(
        _score_retrieval(root, tmp_path / "inputs", run_sha256, tmp_path / "scored", evidence)
    )
    assert exit_code == 1
    assert "score-retrieval" in capsys.readouterr().err


def test_a_dataset_without_the_policy_scores_through_the_qualified_workflow(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SciFact declares no self-document rule; nothing extra is demanded of it."""
    root = _scifact_slice(tmp_path)
    run_sha256 = _write_scifact_inputs(root, tmp_path / "inputs")
    evidence = _candidate_evidence(root, tmp_path, {"q1": ["d1"], "q2": ["d2"]})
    assert (
        cli.main(
            _score_retrieval(root, tmp_path / "inputs", run_sha256, tmp_path / "scored", evidence)
        )
        == 0
    )
    protocol = json.loads(capsys.readouterr().out)["protocol"]
    assert protocol["eligibility"] == "no-declared-protocol"
    assert protocol["policy"] is None


def _scifact_slice(tmp_path: Path) -> Path:
    """A BEIR slice that declares no self-document protocol."""
    from dynamisrag.datasets.beir import build_beir_artifacts
    from dynamisrag.datasets.slices import write_slice_bundle

    members = ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv")
    root = DATASET_FIXTURES / "beir-scifact-mini" / "scifact"
    artifacts = build_beir_artifacts(
        source=beir_source(
            source_id="beir.scifact",
            prefix="scifact",
            fixture="beir-scifact-mini",
            members=members,
        ),
        spec=synthetic_beir_spec(
            source_id="beir.scifact",
            splits={
                "test": synthetic_split(
                    documents=3,
                    documents_without_text=0,
                    queries_in_archive=3,
                    queries=2,
                    qrels=3,
                    min_relevance=0,
                    max_relevance=1,
                )
            },
        ),
        split="test",
        files={name: root.joinpath(*name.split("/")) for name in members},
    )
    write_slice_bundle(tmp_path / "scifact-slice", artifacts.bundle())
    return tmp_path / "scifact-slice"


def _write_scifact_inputs(root: Path, work: Path) -> str:
    dataset = _dataset_of(root)
    config = IrExperimentConfig(
        dataset_sha256=dataset.sha256,
        code_sha="1" * 40,
        retrieval_revision="synthetic-retrieval-v1",
        projection_sha256="2" * 64,
        parameters_json="{}",
    )
    run = IrRun(
        config_sha256=config.sha256,
        dataset_sha256=dataset.sha256,
        query_ids=("q1", "q2"),
        hits=(
            IrHit(query_id="q1", document_id="d1", rank=1, raw_score=1.0),
            IrHit(query_id="q2", document_id="d2", rank=1, raw_score=1.0),
            IrHit(query_id="q2", document_id="d3", rank=2, raw_score=0.5),
        ),
        evaluation_depth=_DEPTH,
    )
    work.mkdir(parents=True, exist_ok=True)
    (work / "dataset.json").write_bytes((root / "dataset.json").read_bytes())
    (work / "config.json").write_bytes(canonical_ir_json(config.payload()))
    (work / "run.json").write_bytes(canonical_ir_json(run.payload()))
    (work / "passage-mapping.json").write_bytes(
        canonical_ir_json({"revision": "ir-passage-mapping-v1", "entries": []})
    )
    return run.sha256


def test_the_dataset_aware_workflow_refuses_an_unverified_slice(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _slice(tmp_path)
    documents = _reference_run_documents(_DEEP_CANDIDATES)
    run_sha256 = _write_inputs(
        root,
        tmp_path / "inputs",
        candidates=_DEEP_CANDIDATES,
        run_documents=documents,
    )
    evidence = _candidate_evidence(root, tmp_path, _DEEP_CANDIDATES)
    (root / "manifest.json").unlink()
    exit_code = cli.main(
        _score_retrieval(root, tmp_path / "inputs", run_sha256, tmp_path / "scored", evidence)
    )
    assert exit_code == 1
    assert "DatasetArtifactError" in capsys.readouterr().err


def test_the_dataset_aware_workflow_refuses_inputs_from_another_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The scored dataset must be the verified slice's dataset, not merely similar."""
    root = _slice(tmp_path)
    documents = _reference_run_documents(_DEEP_CANDIDATES)
    _write_inputs(
        root,
        tmp_path / "inputs",
        candidates=_DEEP_CANDIDATES,
        run_documents=documents,
    )
    other = _mini_slice(tmp_path)
    other_run_sha256 = _write_inputs(
        other,
        tmp_path / "other-inputs",
        candidates={"t1": ["a1", "a2", "a3"]},
        run_documents={"t1": ["a1", "a2"]},
    )
    evidence = _candidate_evidence(other, tmp_path, {"t1": ["a1", "a2", "a3"]})
    exit_code = cli.main(
        _score_retrieval(
            root, tmp_path / "other-inputs", other_run_sha256, tmp_path / "scored", evidence
        )
    )
    assert exit_code == 1
    assert "verified slice" in capsys.readouterr().err


def _mini_slice(tmp_path: Path) -> Path:
    """A second, differently shaped ArguAna slice: same family, different dataset."""
    from dynamisrag.datasets.beir import build_beir_artifacts
    from dynamisrag.datasets.slices import write_slice_bundle

    members = ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv")
    mini_root = DATASET_FIXTURES / "beir-arguana-mini" / "arguana"
    artifacts = build_beir_artifacts(
        source=beir_source(
            source_id="beir.arguana",
            prefix="arguana",
            fixture="beir-arguana-mini",
            members=members,
        ),
        spec=synthetic_beir_spec(
            source_id="beir.arguana",
            splits={
                "test": synthetic_split(
                    documents=4,
                    documents_without_text=0,
                    queries_in_archive=1,
                    queries=1,
                    qrels=1,
                    min_relevance=1,
                    max_relevance=1,
                    dangling_qrels=1,
                    dangling_policy=DanglingQrelPolicy.EXCLUDE_AND_DECLARE,
                )
            },
            self_document_policy=IGNORE_IDENTICAL_IDS_POLICY,
        ),
        split="test",
        files={name: mini_root.joinpath(*name.split("/")) for name in members},
    )
    write_slice_bundle(tmp_path / "arguana-mini-slice", artifacts.bundle())
    return tmp_path / "arguana-mini-slice"


def test_a_refused_qualification_names_the_qualified_workflow() -> None:
    """The refusal message must tell a user where the protocol-aware path is."""
    from dynamisrag.datasets.protocol import require_no_declared_protocol

    class _Dataset:
        source_id = "beir.arguana"
        source_revision = f"2021.03.1.test.{IGNORE_IDENTICAL_IDS_POLICY}"
        sha256 = "0" * 64
        queries: tuple[Any, ...] = ()

    with pytest.raises(DatasetContractError) as raised:
        require_no_declared_protocol(cast(Any, _Dataset()))
    assert "score-retrieval" in str(raised.value)
