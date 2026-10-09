"""The materialization facade: verification, refusal, replay and integration.

The materializer is where every other guarantee becomes an on-disk fact. These
tests drive it through both input modes (an extracted directory and the official
archive shape), prove the two produce byte-identical slices, and then attack the
result: drift one member, flip one archive byte, reject a rights decision, add
an extra file to a sealed slice. Finally, one test feeds a materialized slice to
the RES-140 ``ir score`` command, because a canonical dataset that no scorer
accepts is not an integration.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dynamisrag import __main__ as cli
from dynamisrag.datasets.errors import (
    DatasetArtifactError,
    DatasetContractError,
    DatasetRightsError,
    DatasetSourceError,
)
from dynamisrag.datasets.pipeline import (
    AdapterOverrides,
    MaterializeRequest,
    materialize,
    materialize_source,
)
from dynamisrag.datasets.qasper import QasperSplitExpectation
from dynamisrag.datasets.rights import RightsOutcome
from dynamisrag.datasets.scifact_open import ScifactOpenExpectation
from dynamisrag.datasets.slices import verify_slice
from dynamisrag.datasets.sources import (
    FAMILY_SCIFACT_OPEN,
    FrozenDatasetSource,
    source_by_id,
)
from dynamisrag.ir.contracts import IrExperimentConfig, IrHit, IrRun, canonical_ir_json
from tests.unit.dataset_support import (
    DATASET_FIXTURES,
    beir_source,
    dataset_source,
    qasper_source,
    scifact_open_source,
    synthetic_beir_spec,
    synthetic_rights,
    synthetic_split,
)

_SCIFACT_FIXTURE = DATASET_FIXTURES / "beir-scifact-mini"
_SCIFACT_MEMBERS = ("scifact/corpus.jsonl", "scifact/queries.jsonl", "scifact/qrels/test.tsv")

_SCIFACT_SPEC = synthetic_beir_spec(
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
)

_SCIFACT_OVERRIDES = AdapterOverrides(beir_specs={"beir.scifact": _SCIFACT_SPEC})

_QASPER_EXPECTATION = QasperSplitExpectation(
    papers=2,
    questions=4,
    annotations=5,
    unanswerable=1,
    text_evidence=6,
    resolved_evidence=4,
    ambiguous_evidence=1,
    unmatched_evidence=1,
    float_evidence=1,
)

_SCIFACT_OPEN_EXPECTATION = ScifactOpenExpectation(
    claims=2,
    evidence_links=3,
    evidence_documents=3,
    citation_links=1,
    pooling_links=2,
    support_links=2,
    contradict_links=1,
    metadata_records=2,
    candidate_documents=4,
    pool_pairs=4,
    pool_union_documents=3,
    full_corpus_documents=5,
    evidence_links_in_pool=2,
    evidence_links_outside_pool=1,
)


def _scifact_source(*, archive: bool, tmp_path: Path) -> FrozenDatasetSource:
    if not archive:
        return beir_source(
            source_id="beir.scifact",
            prefix="scifact",
            fixture="beir-scifact-mini",
            members=("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"),
        )
    return dataset_source(
        source_id="beir.scifact",
        family="beir",
        members=_SCIFACT_MEMBERS,
        fixture_dir=_SCIFACT_FIXTURE,
        archive_name="synthetic-scifact.zip",
        archive_dir=tmp_path,
    )


def _request(
    out: Path, *, archives: tuple[Path, ...] = (), source_dir: Path | None = None
) -> MaterializeRequest:
    return MaterializeRequest(
        source_id="beir.scifact",
        split="test",
        out=out,
        archives=archives,
        source_dir=source_dir,
    )


def _slice_bytes(root: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(root.iterdir())}


def test_directory_mode_materializes_a_verifiable_slice(tmp_path: Path) -> None:
    receipt = materialize_source(
        _scifact_source(archive=False, tmp_path=tmp_path),
        _request(tmp_path / "slice", source_dir=_SCIFACT_FIXTURE),
        overrides=_SCIFACT_OVERRIDES,
    )
    verified = verify_slice(tmp_path / "slice")
    assert receipt.manifest_sha256 == verified.manifest_sha256
    assert receipt.dataset_sha256 == verified.dataset_sha256
    assert receipt.source_id == "beir.scifact"
    assert receipt.split == "test"


def test_archive_mode_produces_the_identical_slice(tmp_path: Path) -> None:
    source = _scifact_source(archive=True, tmp_path=tmp_path)
    archive = tmp_path / str(source.artifacts[0].archive_name)
    directory_receipt = materialize_source(
        source,
        _request(tmp_path / "dir-slice", source_dir=_SCIFACT_FIXTURE),
        overrides=_SCIFACT_OVERRIDES,
    )
    archive_receipt = materialize_source(
        source,
        _request(tmp_path / "zip-slice", archives=(archive,)),
        overrides=_SCIFACT_OVERRIDES,
    )
    assert directory_receipt.manifest_sha256 == archive_receipt.manifest_sha256
    assert _slice_bytes(tmp_path / "dir-slice") == _slice_bytes(tmp_path / "zip-slice")


def test_tar_gz_mode_produces_the_identical_slice(tmp_path: Path) -> None:
    source = dataset_source(
        source_id="qasper",
        family="qasper",
        members=("qasper-dev-v0.3.json", "qasper-test-v0.3.json", "qasper-train-v0.3.json"),
        fixture_dir=DATASET_FIXTURES / "qasper-mini",
        archive_name="synthetic-qasper.tgz",
        archive_dir=tmp_path,
    )
    overrides = AdapterOverrides(qasper_expectations={"test": _QASPER_EXPECTATION})
    directory_receipt = materialize_source(
        source,
        MaterializeRequest(
            source_id="qasper",
            split="test",
            out=tmp_path / "dir",
            source_dir=DATASET_FIXTURES / "qasper-mini",
        ),
        overrides=overrides,
    )
    archive_receipt = materialize_source(
        source,
        MaterializeRequest(
            source_id="qasper",
            split="test",
            out=tmp_path / "tar",
            archives=(tmp_path / "synthetic-qasper.tgz",),
        ),
        overrides=overrides,
    )
    assert directory_receipt.task_sha256 == archive_receipt.task_sha256
    assert _slice_bytes(tmp_path / "dir") == _slice_bytes(tmp_path / "tar")


def test_two_source_directory_paths_replay_to_identical_bytes(tmp_path: Path) -> None:
    first = tmp_path / "one"
    second = tmp_path / "two"
    for root in (first, second):
        (root / "scifact").mkdir(parents=True)
    import shutil

    shutil.copytree(_SCIFACT_FIXTURE / "scifact", first / "scifact", dirs_exist_ok=True)
    shutil.copytree(_SCIFACT_FIXTURE / "scifact", second / "scifact", dirs_exist_ok=True)
    source = _scifact_source(archive=False, tmp_path=tmp_path)
    a = materialize_source(
        source, _request(tmp_path / "a", source_dir=first), overrides=_SCIFACT_OVERRIDES
    )
    b = materialize_source(
        source, _request(tmp_path / "b", source_dir=second), overrides=_SCIFACT_OVERRIDES
    )
    assert a.manifest_sha256 == b.manifest_sha256
    assert _slice_bytes(tmp_path / "a") == _slice_bytes(tmp_path / "b")


def test_an_existing_output_directory_is_never_overwritten(tmp_path: Path) -> None:
    out = tmp_path / "slice"
    out.mkdir()
    with pytest.raises(DatasetArtifactError):
        materialize_source(
            _scifact_source(archive=False, tmp_path=tmp_path),
            _request(out, source_dir=_SCIFACT_FIXTURE),
            overrides=_SCIFACT_OVERRIDES,
        )


def test_a_member_that_drifted_after_pinning_is_refused(tmp_path: Path) -> None:
    import shutil

    source = _scifact_source(archive=False, tmp_path=tmp_path)
    root = tmp_path / "scifact"
    shutil.copytree(_SCIFACT_FIXTURE / "scifact", root / "scifact", dirs_exist_ok=True)
    corpus = root / "scifact" / "corpus.jsonl"
    corpus.write_bytes(corpus.read_bytes() + b'{"_id": "d9", "title": "t", "text": "x"}\n')
    with pytest.raises(DatasetSourceError) as raised:
        materialize_source(
            source, _request(tmp_path / "slice", source_dir=root), overrides=_SCIFACT_OVERRIDES
        )
    assert "pinned" in str(raised.value)


def test_an_archive_that_drifted_after_pinning_is_refused(tmp_path: Path) -> None:
    source = _scifact_source(archive=True, tmp_path=tmp_path)
    archive = tmp_path / str(source.artifacts[0].archive_name)
    content = bytearray(archive.read_bytes())
    content[-20] ^= 0xFF
    archive.write_bytes(bytes(content))
    with pytest.raises(DatasetSourceError):
        materialize_source(
            source, _request(tmp_path / "slice", archives=(archive,)), overrides=_SCIFACT_OVERRIDES
        )


def test_an_archive_that_matches_no_pin_is_refused(tmp_path: Path) -> None:
    source = _scifact_source(archive=False, tmp_path=tmp_path)
    stranger = tmp_path / "stranger.zip"
    stranger.write_bytes(b"not a frozen distribution")
    with pytest.raises(DatasetSourceError):
        materialize_source(
            source, _request(tmp_path / "slice", archives=(stranger,)), overrides=_SCIFACT_OVERRIDES
        )


def test_a_missing_declared_member_is_refused(tmp_path: Path) -> None:
    source = dataset_source(
        source_id="beir.scifact",
        family="beir",
        members=("scifact/corpus.jsonl", "scifact/queries.jsonl"),
        fixture_dir=_SCIFACT_FIXTURE,
    )
    with pytest.raises(DatasetContractError):
        materialize_source(
            source,
            MaterializeRequest(
                source_id="beir.scifact",
                split="test",
                out=tmp_path / "slice",
                source_dir=_SCIFACT_FIXTURE,
            ),
        )


def test_both_input_modes_at_once_are_refused(tmp_path: Path) -> None:
    source = _scifact_source(archive=True, tmp_path=tmp_path)
    with pytest.raises(DatasetContractError):
        materialize_source(
            source,
            _request(
                tmp_path / "slice",
                archives=(tmp_path / str(source.artifacts[0].archive_name),),
                source_dir=_SCIFACT_FIXTURE,
            ),
            overrides=_SCIFACT_OVERRIDES,
        )


def test_neither_input_mode_is_refused(tmp_path: Path) -> None:
    with pytest.raises(DatasetContractError):
        materialize_source(
            _scifact_source(archive=False, tmp_path=tmp_path),
            _request(tmp_path / "slice"),
            overrides=_SCIFACT_OVERRIDES,
        )


def test_a_rejected_rights_decision_stops_materialization_before_any_read(tmp_path: Path) -> None:
    source = dataset_source(
        source_id="beir.scifact",
        family="beir",
        members=_SCIFACT_MEMBERS,
        fixture_dir=_SCIFACT_FIXTURE,
        rights=synthetic_rights(outcome=RightsOutcome.REJECTED),
    )
    with pytest.raises(DatasetRightsError):
        materialize_source(
            source,
            _request(tmp_path / "slice", source_dir=_SCIFACT_FIXTURE),
            overrides=_SCIFACT_OVERRIDES,
        )


def test_the_registry_lookup_refuses_an_unknown_source(tmp_path: Path) -> None:
    with pytest.raises(DatasetContractError):
        materialize(
            MaterializeRequest(source_id="beir.unknown", split="test", out=tmp_path / "slice")
        )


def test_scifact_open_materializes_through_the_facade(tmp_path: Path) -> None:
    receipt = materialize_source(
        scifact_open_source(),
        MaterializeRequest(
            source_id="scifact-open",
            split="test",
            out=tmp_path / "slice",
            source_dir=DATASET_FIXTURES / "scifact-open-mini",
        ),
        overrides=AdapterOverrides(scifact_open_expectation=_SCIFACT_OPEN_EXPECTATION),
    )
    assert receipt.dataset_sha256 is not None
    assert (tmp_path / "slice" / "evidence-provenance.json").is_file()


def test_qasper_materializes_through_the_facade(tmp_path: Path) -> None:
    receipt = materialize_source(
        qasper_source(),
        MaterializeRequest(
            source_id="qasper",
            split="test",
            out=tmp_path / "slice",
            source_dir=DATASET_FIXTURES / "qasper-mini",
        ),
        overrides=AdapterOverrides(qasper_expectations={"test": _QASPER_EXPECTATION}),
    )
    assert receipt.task_sha256 is not None
    assert (tmp_path / "slice" / "task.json").is_file()
    assert not (tmp_path / "slice" / "dataset.json").exists()


def test_verification_refuses_an_extra_file(tmp_path: Path) -> None:
    materialize_source(
        _scifact_source(archive=False, tmp_path=tmp_path),
        _request(tmp_path / "slice", source_dir=_SCIFACT_FIXTURE),
        overrides=_SCIFACT_OVERRIDES,
    )
    (tmp_path / "slice" / "extra.json").write_text("{}", encoding="utf-8")
    with pytest.raises(DatasetArtifactError):
        verify_slice(tmp_path / "slice")


def test_verification_refuses_a_swapped_dataset(tmp_path: Path) -> None:
    materialize_source(
        _scifact_source(archive=False, tmp_path=tmp_path),
        _request(tmp_path / "slice", source_dir=_SCIFACT_FIXTURE),
        overrides=_SCIFACT_OVERRIDES,
    )
    (tmp_path / "slice" / "dataset.json").write_text("{}", encoding="utf-8")
    with pytest.raises(DatasetArtifactError):
        verify_slice(tmp_path / "slice")


def test_verification_refuses_a_noncanonical_manifest(tmp_path: Path) -> None:
    materialize_source(
        _scifact_source(archive=False, tmp_path=tmp_path),
        _request(tmp_path / "slice", source_dir=_SCIFACT_FIXTURE),
        overrides=_SCIFACT_OVERRIDES,
    )
    manifest = tmp_path / "slice" / "manifest.json"
    manifest.write_bytes(manifest.read_bytes().replace(b"{", b"{ ", 1))
    with pytest.raises(DatasetArtifactError):
        verify_slice(tmp_path / "slice")


def _write_ir_inputs(slice_dir: Path, work: Path) -> str:
    from dynamisrag.datasets.slices import (
        _restore_dataset_bytes,  # pyright: ignore[reportPrivateUsage]
    )

    dataset = _restore_dataset_bytes((slice_dir / "dataset.json").read_bytes(), name="dataset.json")
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
        evaluation_depth=10,
    )
    work.mkdir(parents=True)
    (work / "dataset.json").write_bytes((slice_dir / "dataset.json").read_bytes())
    (work / "config.json").write_bytes(canonical_ir_json(config.payload()))
    (work / "run.json").write_bytes(canonical_ir_json(run.payload()))
    (work / "passage-mapping.json").write_bytes(
        canonical_ir_json({"revision": "ir-passage-mapping-v1", "entries": []})
    )
    return run.sha256


def test_a_materialized_slice_scores_through_ir_score(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The adapter's output is exactly what RES-140 consumes; no translation layer."""
    materialize_source(
        _scifact_source(archive=False, tmp_path=tmp_path),
        _request(tmp_path / "slice", source_dir=_SCIFACT_FIXTURE),
        overrides=_SCIFACT_OVERRIDES,
    )
    run_sha256 = _write_ir_inputs(tmp_path / "slice", tmp_path / "ir-inputs")
    exit_code = cli.main(
        [
            "ir",
            "score",
            "--inputs",
            str(tmp_path / "ir-inputs"),
            "--run-sha256",
            run_sha256,
            "--out",
            str(tmp_path / "scored"),
        ]
    )
    capsys.readouterr()
    assert exit_code == 0
    assert (tmp_path / "scored" / "manifest.json").is_file()


def test_the_cli_lists_the_registry(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["datasets", "list"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {entry["source_id"] for entry in payload["sources"]} >= {
        "beir.scifact",
        "scifact-open",
        "qasper",
    }
    assert len(payload["shortlist"]) == 4


def test_the_cli_materializes_and_verifies_with_registry_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _scifact_source(archive=False, tmp_path=tmp_path)

    def fake_source_by_id(_source_id: str) -> FrozenDatasetSource:
        return source

    monkeypatch.setattr("dynamisrag.datasets.pipeline.source_by_id", fake_source_by_id)
    monkeypatch.setattr("dynamisrag.datasets.pipeline.SCIFACT_BEIR_SPEC", _SCIFACT_SPEC)
    assert (
        cli.main(
            [
                "datasets",
                "materialize",
                "--source",
                "beir.scifact",
                "--split",
                "test",
                "--source-dir",
                str(_SCIFACT_FIXTURE),
                "--out",
                str(tmp_path / "cli-slice"),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["dataset_sha256"]
    assert cli.main(["datasets", "verify", str(tmp_path / "cli-slice")]) == 0
    capsys.readouterr()


def test_the_cli_refuses_an_unknown_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        cli.main(
            [
                "datasets",
                "materialize",
                "--source",
                "beir.unknown",
                "--split",
                "test",
                "--source-dir",
                str(_SCIFACT_FIXTURE),
                "--out",
                str(tmp_path / "slice"),
            ]
        )
        == 1
    )
    assert "DatasetContractError" in capsys.readouterr().err


def test_the_registry_source_by_id_still_resolves_after_overrides() -> None:
    assert source_by_id("scifact-open").family == FAMILY_SCIFACT_OPEN
