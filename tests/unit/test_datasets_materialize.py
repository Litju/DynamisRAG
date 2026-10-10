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
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

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
from dynamisrag.datasets.primitives import canonical_bytes, digest
from dynamisrag.datasets.qasper import QasperSplitExpectation
from dynamisrag.datasets.rights import RightsOutcome
from dynamisrag.datasets.scifact_open import ScifactOpenExpectation
from dynamisrag.datasets.slices import bytes_sha256, verify_slice
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
    candidate_documents=5,
    pool_pairs=4,
    pool_union_documents=3,
    full_corpus_documents=6,
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


def _materialize_scifact(tmp_path: Path, name: str = "slice") -> Path:
    materialize_source(
        _scifact_source(archive=False, tmp_path=tmp_path),
        _request(tmp_path / name, source_dir=_SCIFACT_FIXTURE),
        overrides=_SCIFACT_OVERRIDES,
    )
    return tmp_path / name


def _mutate_manifest(root: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    path = root / "manifest.json"
    payload: Any = json.loads(path.read_bytes())
    mutate(payload)
    path.write_bytes(canonical_bytes(payload))


def _resign_manifest_file(root: Path, payload: dict[str, Any], name: str) -> None:
    content = (root / name).read_bytes()
    entries: list[dict[str, Any]] = payload["files"]
    for entry in entries:
        if entry["name"] == name:
            entry["size_bytes"] = len(content)
            entry["sha256"] = bytes_sha256(content)
            return
    raise AssertionError(f"no manifest entry for {name}")


def test_verification_recomputes_the_manifest_statistics(tmp_path: Path) -> None:
    root = _materialize_scifact(tmp_path)
    receipt = verify_slice(root)
    assert receipt.verification == "self-consistency"
    assert receipt.trusted_source_sha256 is None
    assert "qrel-statistics" in receipt.verified_claims
    assert "query-statistics" in receipt.verified_claims
    assert "generated-rights-notice" in receipt.verified_claims
    assert "corpus-identity" in receipt.attested_claims
    assert "source-archive-and-member-pins" in receipt.attested_claims


def test_verification_refuses_a_resigned_false_qrel_count(tmp_path: Path) -> None:
    root = _materialize_scifact(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["qrels"]["count"] = 99

    _mutate_manifest(root, mutate)
    with pytest.raises(DatasetArtifactError) as raised:
        verify_slice(root)
    assert "qrel" in str(raised.value)


def test_verification_refuses_a_resigned_false_query_count(tmp_path: Path) -> None:
    root = _materialize_scifact(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["queries"]["count"] = 99

    _mutate_manifest(root, mutate)
    with pytest.raises(DatasetArtifactError) as raised:
        verify_slice(root)
    assert "query" in str(raised.value)


def test_verification_refuses_a_resigned_false_relevance_distribution(tmp_path: Path) -> None:
    root = _materialize_scifact(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["qrels"]["positive_count"] = 0

    _mutate_manifest(root, mutate)
    with pytest.raises(DatasetArtifactError):
        verify_slice(root)


def test_verification_refuses_a_changed_rights_notice_even_when_resigned(tmp_path: Path) -> None:
    root = _materialize_scifact(tmp_path)
    rights = root / "rights.txt"
    rights.write_bytes(rights.read_bytes().replace(b"decision: accepted", b"decision: rejected"))

    def mutate(payload: dict[str, Any]) -> None:
        _resign_manifest_file(root, payload, "rights.txt")

    _mutate_manifest(root, mutate)
    with pytest.raises(DatasetArtifactError) as raised:
        verify_slice(root)
    assert "rights" in str(raised.value)


def test_verification_refuses_changed_manifest_source_metadata(tmp_path: Path) -> None:
    root = _materialize_scifact(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["source"]["rights"]["dataset_license"] = "MIT"

    _mutate_manifest(root, mutate)
    with pytest.raises(DatasetArtifactError) as raised:
        verify_slice(root)
    assert "rights" in str(raised.value)


def test_a_self_consistent_forgery_still_requires_a_trust_anchor(tmp_path: Path) -> None:
    """A forged source can regenerate its own notice; only a trust anchor catches it."""
    root = _materialize_scifact(tmp_path)
    rights = root / "rights.txt"
    rights.write_bytes(
        rights.read_bytes().replace(b"dataset_license: synthetic-fixture", b"dataset_license: MIT")
    )
    manifest_path = root / "manifest.json"
    payload: Any = json.loads(manifest_path.read_bytes())
    payload["source"]["rights"]["dataset_license"] = "MIT"
    _resign_manifest_file(root, payload, "rights.txt")
    manifest_path.write_bytes(canonical_bytes(payload))
    verify_slice(root)
    trusted = _scifact_source(archive=False, tmp_path=tmp_path)
    with pytest.raises(DatasetArtifactError):
        verify_slice(root, trusted_source=trusted)


def test_verification_refuses_a_mismatched_split(tmp_path: Path) -> None:
    root = _materialize_scifact(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["split"] = "train"

    _mutate_manifest(root, mutate)
    with pytest.raises(DatasetArtifactError):
        verify_slice(root)


def test_verification_refuses_a_mismatched_dataset_source(tmp_path: Path) -> None:
    root = _materialize_scifact(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["source"]["source_id"] = "beir.nfcorpus"

    _mutate_manifest(root, mutate)
    with pytest.raises(DatasetArtifactError):
        verify_slice(root)


def test_verification_refuses_an_untrusted_source_pin(tmp_path: Path) -> None:
    root = _materialize_scifact(tmp_path)
    with pytest.raises(DatasetArtifactError):
        verify_slice(root, trusted_source=source_by_id("beir.nfcorpus"))


def test_registered_source_verification_refuses_a_synthetic_slice(tmp_path: Path) -> None:
    """The synthetic fixture is not the registered distribution, whatever its source id."""
    root = _materialize_scifact(tmp_path)
    with pytest.raises(DatasetArtifactError):
        verify_slice(root, registered_source=True)


def test_qualified_verification_with_an_expected_manifest_digest(tmp_path: Path) -> None:
    root = _materialize_scifact(tmp_path)
    digest = verify_slice(root).manifest_sha256
    receipt = verify_slice(root, expected_manifest_sha256=digest)
    assert receipt.verification == "qualified"
    assert receipt.expected_manifest_sha256 == digest
    with pytest.raises(DatasetArtifactError):
        verify_slice(root, expected_manifest_sha256="0" * 64)


def test_verification_refuses_a_resigned_false_qasper_manifest_count(tmp_path: Path) -> None:
    materialize_source(
        qasper_source(),
        MaterializeRequest(
            source_id="qasper",
            split="test",
            out=tmp_path / "qasper-slice",
            source_dir=DATASET_FIXTURES / "qasper-mini",
        ),
        overrides=AdapterOverrides(qasper_expectations={"test": _QASPER_EXPECTATION}),
    )
    root = tmp_path / "qasper-slice"

    def mutate(payload: dict[str, Any]) -> None:
        payload["counts"]["questions"] = 99

    _mutate_manifest(root, mutate)
    with pytest.raises(DatasetArtifactError):
        verify_slice(root)


def _materialize_qasper(tmp_path: Path, name: str = "qasper-slice") -> Path:
    materialize_source(
        qasper_source(),
        MaterializeRequest(
            source_id="qasper",
            split="test",
            out=tmp_path / name,
            source_dir=DATASET_FIXTURES / "qasper-mini",
        ),
        overrides=AdapterOverrides(qasper_expectations={"test": _QASPER_EXPECTATION}),
    )
    return tmp_path / name


def _empty_rankings(root: Path, work: Path) -> Path:
    """Rank every question with no anchor; scoring still runs and still fails closed."""
    task: Any = json.loads((root / "task.json").read_bytes())
    questions = cast("list[dict[str, Any]]", task["questions"])
    rankings: dict[str, list[str]] = {str(question["question_id"]): [] for question in questions}
    path = work / "rankings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(rankings))
    return path


def _score_evidence(root: Path, rankings: Path, out: Path, *extra: str) -> list[str]:
    return [
        "datasets",
        "score-evidence",
        "--slice",
        str(root),
        "--rankings",
        str(rankings),
        "--out",
        str(out),
        *extra,
    ]


def test_score_evidence_reports_the_verified_slice_identity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _materialize_qasper(tmp_path)
    receipt = verify_slice(root)
    out = tmp_path / "evaluation.json"
    assert cli.main(_score_evidence(root, _empty_rankings(root, tmp_path), out)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verification"] == "self-consistency"
    assert payload["source_id"] == "qasper"
    assert payload["split"] == "test"
    assert payload["task_sha256"] == receipt.task_sha256
    assert payload["manifest_sha256"] == receipt.manifest_sha256
    assert payload["trusted_source_sha256"] is None
    assert payload["expected_manifest_sha256"] is None
    assert "qasper-task-identity" in payload["verified_claims"]
    assert json.loads(out.read_bytes())["aggregate"]["task_sha256"] == receipt.task_sha256


def test_score_evidence_refuses_a_tampered_task_with_an_unchanged_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Before: the command parsed task.json directly, so rewritten gold scored cleanly."""
    root = _materialize_qasper(tmp_path)
    rankings = _empty_rankings(root, tmp_path)
    task_path = root / "task.json"
    task: Any = json.loads(task_path.read_bytes())
    task["questions"][0]["question"] = "a rewritten question that never existed"
    task_path.write_bytes(canonical_bytes(task))
    out = tmp_path / "evaluation.json"
    assert cli.main(_score_evidence(root, rankings, out)) == 1
    assert "DatasetArtifactError" in capsys.readouterr().err
    assert not out.exists()


def test_score_evidence_refuses_a_missing_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _materialize_qasper(tmp_path)
    rankings = _empty_rankings(root, tmp_path)
    (root / "manifest.json").unlink()
    out = tmp_path / "evaluation.json"
    assert cli.main(_score_evidence(root, rankings, out)) == 1
    assert "DatasetArtifactError" in capsys.readouterr().err
    assert not out.exists()


def test_score_evidence_refuses_a_rogue_payload_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _materialize_qasper(tmp_path)
    rankings = _empty_rankings(root, tmp_path)
    (root / "extra.json").write_bytes(b"{}")
    out = tmp_path / "evaluation.json"
    assert cli.main(_score_evidence(root, rankings, out)) == 1
    assert "DatasetArtifactError" in capsys.readouterr().err
    assert not out.exists()


def test_score_evidence_refuses_a_symlinked_task(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _materialize_qasper(tmp_path)
    rankings = _empty_rankings(root, tmp_path)
    elsewhere = tmp_path / "swapped.json"
    elsewhere.write_bytes((root / "task.json").read_bytes())
    task_path = root / "task.json"
    task_path.unlink()
    try:
        task_path.symlink_to(elsewhere)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlinks are unavailable on this platform: {type(error).__name__}")
    out = tmp_path / "evaluation.json"
    assert cli.main(_score_evidence(root, rankings, out)) == 1
    assert "DatasetArtifactError" in capsys.readouterr().err
    assert not out.exists()


def test_a_resigned_task_forgery_scores_only_as_self_consistent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A re-signed forgery is internally consistent; only a trust anchor can reject it.

    The command must therefore never describe such a result as qualified.
    """
    root = _materialize_qasper(tmp_path)
    rankings = _empty_rankings(root, tmp_path)
    task_path = root / "task.json"
    task: Any = json.loads(task_path.read_bytes())
    task["questions"][0]["question"] = "a rewritten question that never existed"
    task_path.write_bytes(canonical_bytes(task))

    def mutate(payload: dict[str, Any]) -> None:
        payload["task_sha256"] = bytes_sha256(task_path.read_bytes())
        _resign_manifest_file(root, payload, "task.json")

    _mutate_manifest(root, mutate)
    out = tmp_path / "evaluation.json"
    assert cli.main(_score_evidence(root, rankings, out)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verification"] == "self-consistency"
    assert payload["trusted_source_sha256"] is None
    assert payload["expected_manifest_sha256"] is None
    assert "corpus-identity" not in payload["verified_claims"]


def test_score_evidence_refuses_a_wrong_expected_manifest_digest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _materialize_qasper(tmp_path)
    rankings = _empty_rankings(root, tmp_path)
    out = tmp_path / "evaluation.json"
    assert cli.main(_score_evidence(root, rankings, out, "--expect-manifest-sha256", "0" * 64)) == 1
    assert "DatasetArtifactError" in capsys.readouterr().err
    assert not out.exists()


def test_score_evidence_qualifies_against_a_trusted_manifest_digest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _materialize_qasper(tmp_path)
    rankings = _empty_rankings(root, tmp_path)
    digest = verify_slice(root).manifest_sha256
    out = tmp_path / "evaluation.json"
    assert cli.main(_score_evidence(root, rankings, out, "--expect-manifest-sha256", digest)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verification"] == "qualified"
    assert payload["expected_manifest_sha256"] == digest


def test_score_evidence_refuses_a_document_retrieval_slice(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A retrieval slice has no evidence-selection task; it is not a scoring input."""
    root = _materialize_scifact(tmp_path)
    rankings = tmp_path / "rankings.json"
    rankings.write_bytes(canonical_bytes({"q1": []}))
    out = tmp_path / "evaluation.json"
    assert cli.main(_score_evidence(root, rankings, out)) == 1
    assert "DatasetArtifactError" in capsys.readouterr().err
    assert not out.exists()


def _resign_sidecar(root: Path, payload: dict[str, Any]) -> None:
    """Re-sign a mutated sidecar: its file entry and the digest the manifest declares."""
    sidecar = root / "evidence-provenance.json"
    sidecar.write_bytes(canonical_bytes(payload))

    def mutate(manifest: dict[str, Any]) -> None:
        _resign_manifest_file(root, manifest, sidecar.name)
        manifest["diagnostics"]["provenance_sidecar_sha256"] = digest(payload)

    _mutate_manifest(root, mutate)


def _link_mutations() -> tuple[tuple[str, Callable[[dict[str, Any]], None]], ...]:
    """Every link field whose shape the sidecar must enforce, and a violation of it.

    ``sentences`` and ``model_ranks`` were previously accepted in any shape, so
    most of these are the defect itself rather than a guard.
    """

    def _first(payload: dict[str, Any]) -> dict[str, Any]:
        links = cast("list[dict[str, Any]]", payload["links"])
        return links[0]

    def _second(payload: dict[str, Any]) -> dict[str, Any]:
        return cast("list[dict[str, Any]]", payload["links"])[1]

    return (
        ("descending-sentences", lambda payload: _first(payload).update(sentences=[1, 0])),
        ("negative-sentence", lambda payload: _first(payload).update(sentences=[-1])),
        ("repeated-sentence", lambda payload: _first(payload).update(sentences=[0, 0])),
        ("non-integer-sentence", lambda payload: _first(payload).update(sentences=["0"])),
        ("boolean-sentence", lambda payload: _first(payload).update(sentences=[True])),
        ("missing-sentences", lambda payload: _first(payload).pop("sentences")),
        ("citation-with-model-ranks", lambda payload: _first(payload).update(model_ranks={"m": 1})),
        ("pooling-without-model-ranks", lambda payload: _second(payload).update(model_ranks=None)),
        ("pooling-with-empty-model-ranks", lambda payload: _second(payload).update(model_ranks={})),
        (
            "negative-model-rank",
            lambda payload: _second(payload).update(model_ranks={"model_a": -1, "model_b": 3}),
        ),
        (
            "non-integer-model-rank",
            lambda payload: _second(payload).update(model_ranks={"model_a": "1", "model_b": 3}),
        ),
        (
            "blank-model-key",
            lambda payload: _second(payload).update(model_ranks={"": 1, "model_b": 3}),
        ),
        (
            "non-boolean-pool-membership",
            lambda payload: _first(payload).update(in_released_pool="yes"),
        ),
        ("non-binary-relevance", lambda payload: _first(payload).update(relevance=2)),
        ("unknown-provenance", lambda payload: _first(payload).update(provenance="crowd")),
    )


@pytest.mark.parametrize(("case", "mutate"), _link_mutations(), ids=lambda value: str(value)[:40])
def test_verification_refuses_a_malformed_sidecar_link(
    tmp_path: Path, case: str, mutate: Callable[[dict[str, Any]], None]
) -> None:
    root = _materialize_scifact_open(tmp_path)
    payload: Any = json.loads((root / "evidence-provenance.json").read_bytes())
    mutate(cast("dict[str, Any]", payload))
    _resign_sidecar(root, cast("dict[str, Any]", payload))
    with pytest.raises(DatasetArtifactError):
        verify_slice(root)


def test_the_scifact_open_sidecar_is_mandatory_even_when_resigned(
    tmp_path: Path,
) -> None:
    """Before: deleting the sidecar and its manifest entry passed verification."""
    root = _materialize_scifact_open(tmp_path)
    (root / "evidence-provenance.json").unlink()

    def mutate(payload: dict[str, Any]) -> None:
        payload["files"] = [
            entry for entry in payload["files"] if entry["name"] != "evidence-provenance.json"
        ]

    _mutate_manifest(root, mutate)
    with pytest.raises(DatasetArtifactError) as raised:
        verify_slice(root)
    assert "evidence-provenance.json" in str(raised.value.expected)
    assert "evidence-provenance.json" not in str(raised.value.observed)


def test_the_scifact_open_sidecar_is_mandatory_for_both_corpus_variants(
    tmp_path: Path,
) -> None:
    root = materialize_source(
        scifact_open_source(),
        MaterializeRequest(
            source_id="scifact-open",
            split="test",
            out=tmp_path / "open-full",
            source_dir=DATASET_FIXTURES / "scifact-open-mini",
            corpus_variant="full",
        ),
        overrides=AdapterOverrides(scifact_open_expectation=_SCIFACT_OPEN_EXPECTATION),
    ).root
    assert (root / "evidence-provenance.json").is_file()
    (root / "evidence-provenance.json").unlink()

    def mutate(payload: dict[str, Any]) -> None:
        payload["files"] = [
            entry for entry in payload["files"] if entry["name"] != "evidence-provenance.json"
        ]

    _mutate_manifest(root, mutate)
    with pytest.raises(DatasetArtifactError):
        verify_slice(root)


def test_a_beir_slice_refuses_a_foreign_provenance_sidecar(tmp_path: Path) -> None:
    root = _materialize_scifact(tmp_path)
    sidecar = root / "evidence-provenance.json"
    sidecar.write_bytes(canonical_bytes({"artifact_revision": "foreign"}))

    def mutate(payload: dict[str, Any]) -> None:
        content = sidecar.read_bytes()
        payload["files"] = [
            *payload["files"],
            {"name": sidecar.name, "size_bytes": len(content), "sha256": bytes_sha256(content)},
        ]

    _mutate_manifest(root, mutate)
    with pytest.raises(DatasetArtifactError) as raised:
        verify_slice(root)
    assert "evidence-provenance.json" in str(raised.value.observed)


def test_a_qasper_slice_refuses_a_foreign_provenance_sidecar(tmp_path: Path) -> None:
    root = _materialize_qasper(tmp_path)
    sidecar = root / "evidence-provenance.json"
    sidecar.write_bytes(canonical_bytes({"artifact_revision": "foreign"}))

    def mutate(payload: dict[str, Any]) -> None:
        content = sidecar.read_bytes()
        payload["files"] = [
            *payload["files"],
            {"name": sidecar.name, "size_bytes": len(content), "sha256": bytes_sha256(content)},
        ]

    _mutate_manifest(root, mutate)
    with pytest.raises(DatasetArtifactError):
        verify_slice(root)


def test_the_closed_inventory_is_a_verified_claim(tmp_path: Path) -> None:
    receipt = verify_slice(_materialize_scifact_open(tmp_path))
    assert "source-family-closed-inventory" in receipt.verified_claims
    assert "scifact-open-provenance-sidecar" in receipt.verified_claims
    assert "source-sentence-pointers" in receipt.attested_claims


def _materialize_scifact_open(tmp_path: Path) -> Path:
    materialize_source(
        scifact_open_source(),
        MaterializeRequest(
            source_id="scifact-open",
            split="test",
            out=tmp_path / "open-slice",
            source_dir=DATASET_FIXTURES / "scifact-open-mini",
        ),
        overrides=AdapterOverrides(scifact_open_expectation=_SCIFACT_OPEN_EXPECTATION),
    )
    return tmp_path / "open-slice"


def test_verification_refuses_a_resigned_false_sidecar_count(tmp_path: Path) -> None:
    root = _materialize_scifact_open(tmp_path)
    sidecar = root / "evidence-provenance.json"
    payload: Any = json.loads(sidecar.read_bytes())
    payload["counts"]["citation_links"] = 99
    sidecar.write_bytes(canonical_bytes(payload))

    def mutate(manifest: dict[str, Any]) -> None:
        _resign_manifest_file(root, manifest, "evidence-provenance.json")

    _mutate_manifest(root, mutate)
    with pytest.raises(DatasetArtifactError):
        verify_slice(root)


def test_verification_refuses_a_changed_sidecar_link_even_when_resigned(tmp_path: Path) -> None:
    root = _materialize_scifact_open(tmp_path)
    sidecar = root / "evidence-provenance.json"
    payload: Any = json.loads(sidecar.read_bytes())
    payload["links"][0]["document_id"] = "404"
    sidecar.write_bytes(canonical_bytes(payload))

    def mutate(manifest: dict[str, Any]) -> None:
        _resign_manifest_file(root, manifest, "evidence-provenance.json")

    _mutate_manifest(root, mutate)
    with pytest.raises(DatasetArtifactError):
        verify_slice(root)


def test_the_cli_verifies_with_a_manifest_digest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _materialize_scifact(tmp_path)
    digest = verify_slice(root).manifest_sha256
    assert cli.main(["datasets", "verify", str(root), "--expect-manifest-sha256", digest]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verification"] == "qualified"
    assert payload["expected_manifest_sha256"] == digest


def test_the_cli_registered_source_flag_refuses_a_synthetic_slice(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _materialize_scifact(tmp_path)
    assert cli.main(["datasets", "verify", str(root), "--registered-source"]) == 1
    assert "DatasetArtifactError" in capsys.readouterr().err


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
