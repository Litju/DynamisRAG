"""Shared builders for the RES-141 dataset adapter tests.

Every fixture under ``tests/fixtures/datasets`` is synthetic, so a test that
wants the *verification* pipeline needs a source whose pins describe those
fixture bytes. These helpers compute the pins from the fixture files, build
deterministic archives from them, and assemble registry-shaped sources. No
helper is imported by production code.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, cast

from dynamisrag.datasets.beir import BeirSliceSpec, BeirSplitExpectation, DanglingQrelPolicy
from dynamisrag.datasets.rights import (
    LicenseScope,
    Redistribution,
    RightsDecision,
    RightsOutcome,
)
from dynamisrag.datasets.sources import (
    FAMILY_BEIR,
    FAMILY_QASPER,
    FAMILY_SCIFACT_OPEN,
    FrozenDatasetSource,
    SourceArtifact,
    SourceFormat,
    SourceMember,
)
from tests._support import FIXTURES_ROOT

__all__ = [
    "DATASET_FIXTURES",
    "beir_source",
    "build_tar_gz",
    "build_zip",
    "dataset_source",
    "qrels_rows",
    "read_jsonl",
    "synthetic_rights",
]

DATASET_FIXTURES: Final[Path] = FIXTURES_ROOT / "datasets"
"""Synthetic RES-141 fixtures; see ``tests/fixtures/datasets/README.md``."""


def synthetic_rights(
    *,
    outcome: RightsOutcome = RightsOutcome.ACCEPTED,
    scope: LicenseScope = LicenseScope.OPEN,
    redistribution: Redistribution = Redistribution.UNVERIFIED,
) -> RightsDecision:
    """An accepted (or deliberately rejected) rights decision for a fixture."""
    return RightsDecision(
        dataset_license="synthetic-fixture",
        license_scope=scope,
        license_source="https://example.invalid/synthetic-fixture-license",
        underlying_content="entirely synthetic content written for this repository",
        redistribution=redistribution,
        attribution="DynamisRAG RES-141 test fixtures",
        outcome=outcome,
        basis="synthetic fixture used only in the test suite",
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value: object = json.loads(line)
        assert isinstance(value, dict)
        document = cast("dict[object, object]", value)
        records.append({str(key): item for key, item in document.items()})
    return records


def qrels_rows(path: Path) -> list[tuple[str, str, int]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [
        (columns[0], columns[1], int(columns[2]))
        for line in lines[1:]
        if line.strip()
        for columns in [line.split("\t")]
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _member(source_dir: Path, name: str) -> SourceMember:
    path = source_dir.joinpath(*name.split("/"))
    return SourceMember(name=name, size_bytes=path.stat().st_size, sha256=_sha256(path))


def dataset_source(
    *,
    source_id: str,
    family: str,
    members: Sequence[str],
    fixture_dir: Path,
    archive_name: str | None = None,
    archive_dir: Path | None = None,
    archive_format: SourceFormat = SourceFormat.TAR_GZ,
    rights: RightsDecision | None = None,
) -> FrozenDatasetSource:
    """A registry-shaped source pinned to one fixture directory.

    When ``archive_name`` is given the artifact's archive pin is derived from a
    deterministic archive built from the same members, so the same source object
    works for directory-mode and archive-mode materialization. ``archive_dir``
    keeps that archive out of the fixture tree.
    """
    pinned = tuple(_member(fixture_dir, name) for name in sorted(members))
    if archive_name is None:
        artifact = SourceArtifact(
            url=f"https://example.invalid/{source_id}.zip",
            archive_name="fixture.zip",
            format=archive_format,
            size_bytes=1,
            sha256="0" * 64,
            members=pinned,
        )
    else:
        parent = archive_dir if archive_dir is not None else fixture_dir.parent
        if archive_format is SourceFormat.TAR_GZ:
            archive = build_tar_gz(fixture_dir, members, parent / archive_name)
        else:
            archive = build_zip(fixture_dir, members, parent / archive_name)
        artifact = SourceArtifact(
            url=f"https://example.invalid/{archive_name}",
            archive_name=archive_name,
            format=archive_format,
            size_bytes=archive.stat().st_size,
            sha256=_sha256(archive),
            members=pinned,
        )
    return FrozenDatasetSource(
        source_id=source_id,
        family=family,
        revision="synthetic-v1",
        documentation="https://example.invalid/synthetic-fixture",
        content_note="synthetic fixture written for the RES-141 test suite",
        artifacts=(artifact,),
        rights=rights if rights is not None else synthetic_rights(),
    )


def build_zip(source_dir: Path, members: Sequence[str], destination: Path) -> Path:
    """Build a byte-deterministic ZIP from declared members, and return it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, source_dir.joinpath(*name.split("/")).read_bytes())
    return destination


def build_tar_gz(source_dir: Path, members: Sequence[str], destination: Path) -> Path:
    """Build a byte-deterministic gzip tar from declared members, and return it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.GNU_FORMAT) as archive:
        for name in sorted(members):
            content = source_dir.joinpath(*name.split("/")).read_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(content))
    with (
        destination.open("wb") as handle,
        gzip.GzipFile(fileobj=handle, mode="wb", mtime=0) as compressed,
    ):
        compressed.write(buffer.getvalue())
    return destination


def beir_source(
    *,
    source_id: str,
    prefix: str,
    fixture: str,
    members: Sequence[str],
    archive_name: str | None = None,
    rights: RightsDecision | None = None,
) -> FrozenDatasetSource:
    """A BEIR-shaped synthetic source over ``tests/fixtures/datasets/<fixture>``."""
    return dataset_source(
        source_id=source_id,
        family=FAMILY_BEIR,
        members=[f"{prefix}/{name}" for name in members],
        fixture_dir=DATASET_FIXTURES / fixture,
        archive_name=archive_name,
        rights=rights,
    )


def qasper_source(
    *, fixture: str = "qasper-mini", archive_name: str | None = None
) -> FrozenDatasetSource:
    return dataset_source(
        source_id="qasper",
        family=FAMILY_QASPER,
        members=(
            "qasper-dev-v0.3.json",
            "qasper-test-v0.3.json",
            "qasper-train-v0.3.json",
        ),
        fixture_dir=DATASET_FIXTURES / fixture,
        archive_name=archive_name,
    )


def scifact_open_source(
    *, fixture: str = "scifact-open-mini", archive_name: str | None = None
) -> FrozenDatasetSource:
    return dataset_source(
        source_id="scifact-open",
        family=FAMILY_SCIFACT_OPEN,
        members=(
            "data/claims.jsonl",
            "data/claims_metadata.jsonl",
            "data/corpus.jsonl",
            "data/corpus_candidates.jsonl",
            "prediction/retrievals.jsonl",
        ),
        fixture_dir=DATASET_FIXTURES / fixture,
        archive_name=archive_name,
    )


def synthetic_beir_spec(
    *,
    source_id: str,
    splits: Mapping[str, BeirSplitExpectation],
    role: str = "synthetic",
    domain: str = "synthetic retrieval",
    self_document_policy: str | None = None,
) -> BeirSliceSpec:
    return BeirSliceSpec(
        source_id=source_id,
        role=role,
        domain=domain,
        projection_note="synthetic fixture projection",
        splits=tuple((split, splits[split]) for split in sorted(splits)),
        self_document_policy=self_document_policy,
    )


def synthetic_split(
    *,
    documents: int,
    documents_without_text: int,
    queries_in_archive: int,
    queries: int,
    qrels: int,
    min_relevance: int,
    max_relevance: int,
    dangling_qrels: int = 0,
    dangling_policy: DanglingQrelPolicy = DanglingQrelPolicy.REFUSE,
) -> BeirSplitExpectation:
    return BeirSplitExpectation(
        documents=documents,
        documents_without_text=documents_without_text,
        queries_in_archive=queries_in_archive,
        queries=queries,
        qrels=qrels,
        min_relevance=min_relevance,
        max_relevance=max_relevance,
        dangling_qrels=dangling_qrels,
        dangling_policy=dangling_policy,
    )
