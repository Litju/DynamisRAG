"""The materialization facade: verified sources in, sealed slices out (RES-141).

One entry point, :func:`materialize`, with three explicit phases and no ambient
state:

1. **Rights gate.** The source's recorded decision must be accepted before an
   archive is opened.
2. **Verified resolution.** Either a caller-supplied extracted directory or
   caller-supplied archives matching the registry's pinned digests. In archive
   mode only the declared members the adapter will read are extracted, each is
   re-hashed after extraction, and the scratch tree is removed afterwards. In
   directory mode every read member is re-hashed where it lies and nothing is
   written or deleted.
3. **Family adapter.** BEIR, SciFact-Open or QASPER turns the verified files
   into a canonical slice bundle, which is then staged and atomically renamed
   into place. Nothing here contacts a network.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Final

from dynamisrag.datasets.beir import BeirSliceSpec, build_beir_artifacts
from dynamisrag.datasets.errors import DatasetContractError, DatasetSourceError
from dynamisrag.datasets.primitives import (
    extract_members,
    file_sha256,
    verify_file_against,
    verify_member_pin,
)
from dynamisrag.datasets.qasper import (
    QASPER_SPLIT_FILES,
    QasperSplitExpectation,
    build_qasper_task_artifacts,
)
from dynamisrag.datasets.scifact import SCIFACT_BEIR_SPEC
from dynamisrag.datasets.scifact_open import (
    CORPUS_VARIANTS,
    SCIFACT_OPEN_EXPECTATION,
    ScifactOpenExpectation,
    build_scifact_open_artifacts,
)
from dynamisrag.datasets.shortlist import R141_BEIR_SHORTLIST, shortlist_spec
from dynamisrag.datasets.slices import SliceBundle, SliceReceipt, write_slice_bundle
from dynamisrag.datasets.sources import (
    FAMILY_BEIR,
    FAMILY_QASPER,
    FAMILY_SCIFACT_OPEN,
    RES141_SOURCES,
    FrozenDatasetSource,
    source_by_id,
)

__all__ = [
    "AdapterOverrides",
    "MaterializeRequest",
    "materialize",
    "materialize_source",
    "registry_summary",
]

_BEIR_PREFIXES: Final[dict[str, str]] = {
    "beir.scifact": "scifact",
    "beir.nfcorpus": "nfcorpus",
    "beir.scidocs": "scidocs",
    "beir.arguana": "arguana",
    "beir.fiqa": "fiqa",
}


@dataclass(frozen=True)
class MaterializeRequest:
    """Everything one materialization needs, with nothing inferred from the CWD."""

    source_id: str
    split: str
    out: Path
    archives: tuple[Path, ...] = ()
    source_dir: Path | None = None
    scratch_dir: Path | None = None
    corpus_variant: str = "candidates"


@dataclass(frozen=True)
class AdapterOverrides:
    """Explicit adapter policies for a source the registry does not pin.

    The registry's sources use their own counted pins; a synthetic fixture is
    not the official distribution and must state its expectation rather than
    borrow the official one. This is the only way an override enters the
    materializer, and it is never read by the CLI.
    """

    beir_specs: Mapping[str, BeirSliceSpec] = field(default_factory=dict[str, BeirSliceSpec])
    scifact_open_expectation: ScifactOpenExpectation | None = None
    qasper_expectations: Mapping[str, QasperSplitExpectation] | None = None


def _beir_spec(source: FrozenDatasetSource, overrides: AdapterOverrides) -> BeirSliceSpec:
    if source.source_id in overrides.beir_specs:
        return overrides.beir_specs[source.source_id]
    if source.source_id == "beir.scifact":
        return SCIFACT_BEIR_SPEC
    return shortlist_spec(source.source_id)


def _wanted_members(
    source: FrozenDatasetSource, *, split: str, corpus_variant: str
) -> tuple[str, ...]:
    """The exact registry members this materialization will read.

    Selective on purpose: the SciFact-Open ``candidates`` variant never touches
    the 889 MB full corpus, and no adapter ever reads a member it did not
    declare.
    """
    if source.family == FAMILY_BEIR:
        prefix = _BEIR_PREFIXES[source.source_id]
        return (
            f"{prefix}/corpus.jsonl",
            f"{prefix}/queries.jsonl",
            f"{prefix}/qrels/{split}.tsv",
        )
    if source.family == FAMILY_SCIFACT_OPEN:
        if corpus_variant not in CORPUS_VARIANTS:
            raise DatasetContractError(
                f"unknown SciFact-Open corpus variant {corpus_variant!r}.",
                operation="materialize",
                source_id=source.source_id,
                item_id=corpus_variant,
                expected=str(list(CORPUS_VARIANTS)),
            )
        members = [
            "data/claims.jsonl",
            "data/claims_metadata.jsonl",
            "data/corpus_candidates.jsonl",
            "prediction/retrievals.jsonl",
        ]
        if corpus_variant == "full":
            members.append("data/corpus.jsonl")
        return tuple(sorted(members))
    if source.family == FAMILY_QASPER:
        if split not in QASPER_SPLIT_FILES:
            raise DatasetContractError(
                f"QASPER declares no split {split!r}.",
                operation="materialize",
                source_id=source.source_id,
                split=split,
                expected=str(sorted(QASPER_SPLIT_FILES)),
            )
        return (QASPER_SPLIT_FILES[split],)
    raise DatasetContractError(
        f"unknown source family {source.family!r}.",
        operation="materialize",
        source_id=source.source_id,
    )


def _resolve_from_directory(
    source: FrozenDatasetSource, members: tuple[str, ...], source_dir: Path
) -> dict[str, Path]:
    """Resolve and re-hash every wanted member inside a caller-provided directory."""
    if not source_dir.is_dir():
        raise DatasetSourceError(
            "the provided source directory is not a regular directory.",
            operation="materialize",
            source_id=source.source_id,
            item_id=str(source_dir.name),
        )
    resolved: dict[str, Path] = {}
    for name in members:
        member = source.member(name)
        path = source_dir.joinpath(*PurePosixPath(name).parts)
        verify_member_pin(path, member=name, size_bytes=member.size_bytes, sha256=member.sha256)
        resolved[name] = path
    return resolved


def _resolve_from_archives(
    source: FrozenDatasetSource,
    members: tuple[str, ...],
    archives: tuple[Path, ...],
    *,
    scratch: Path,
) -> dict[str, Path]:
    """Verify provided archives against the registry and extract only what is read."""
    remaining = set(members)
    resolved: dict[str, Path] = {}
    for position, archive in enumerate(archives):
        observed = file_sha256(archive)
        artifact = source.artifact_for_digest(observed)
        if artifact is None:
            raise DatasetSourceError(
                "a provided archive matches none of this source's pinned digests. It is not "
                "the frozen distribution, whatever its name says.",
                operation="materialize",
                source_id=source.source_id,
                item_id=archive.name,
                observed=observed,
            )
        verify_file_against(
            archive,
            size_bytes=artifact.size_bytes,
            sha256=artifact.sha256,
            operation="materialize",
            source_id=source.source_id,
        )
        wanted = [member.name for member in artifact.members if member.name in remaining]
        if not wanted:
            continue
        destination = scratch / f"{position:02d}-{artifact.archive_name}"
        extracted = extract_members(
            archive,
            archive_format=artifact.format.value,
            members=wanted,
            destination=destination,
        )
        for name, path in extracted.items():
            member = source.member(name)
            verify_member_pin(path, member=name, size_bytes=member.size_bytes, sha256=member.sha256)
            resolved[name] = path
            remaining.discard(name)
    if remaining:
        raise DatasetSourceError(
            "the provided archives do not cover every member this slice reads.",
            operation="materialize",
            source_id=source.source_id,
            count=len(remaining),
            item_id=sorted(remaining)[0],
        )
    return resolved


def _bundle_for(
    source: FrozenDatasetSource,
    split: str,
    variant: str,
    files: dict[str, Path],
    overrides: AdapterOverrides,
) -> SliceBundle:
    """Dispatch to the family adapter over already-verified files."""
    if source.family == FAMILY_BEIR:
        spec = _beir_spec(source, overrides)
        prefix = _BEIR_PREFIXES[source.source_id]
        relative = {
            name.split("/", 1)[1]: path
            for name, path in files.items()
            if name.startswith(f"{prefix}/")
        }
        return build_beir_artifacts(source=source, spec=spec, split=split, files=relative).bundle()
    if source.family == FAMILY_SCIFACT_OPEN:
        return build_scifact_open_artifacts(
            source=source,
            split=split,
            variant=variant,
            files=files,
            expectation=overrides.scifact_open_expectation or SCIFACT_OPEN_EXPECTATION,
        ).bundle()
    return build_qasper_task_artifacts(
        source=source,
        split=split,
        files=files,
        expectations=overrides.qasper_expectations,
    )


def _splits_for(source: FrozenDatasetSource) -> tuple[str, ...]:
    if source.family == FAMILY_BEIR:
        return _beir_spec(source, AdapterOverrides()).split_names
    if source.family == FAMILY_SCIFACT_OPEN:
        return ("test",)
    return tuple(sorted(QASPER_SPLIT_FILES))


def registry_summary() -> dict[str, object]:
    """The machine-readable registry view printed by ``datasets list``.

    Identities, split names, cardinality pins and the rights decision only: no
    archive URL is fetched, and no member is read.
    """
    sources: list[dict[str, object]] = []
    for source in RES141_SOURCES:
        sources.append(
            {
                "source_id": source.source_id,
                "family": source.family,
                "revision": source.revision,
                "splits": list(_splits_for(source)),
                "rights_outcome": source.rights.outcome.value,
                "dataset_license": source.rights.dataset_license,
                "license_scope": source.rights.license_scope.value,
                "redistribution": source.rights.redistribution.value,
                "artifacts": [
                    {
                        "archive_name": artifact.archive_name,
                        "format": artifact.format.value,
                        "size_bytes": artifact.size_bytes,
                        "sha256": artifact.sha256,
                    }
                    for artifact in source.artifacts
                ],
            }
        )
    shortlist = [
        {
            "source_id": entry.spec.source_id,
            "category": entry.category,
            "domain": entry.spec.domain,
            "rationale": entry.rationale,
        }
        for entry in R141_BEIR_SHORTLIST
    ]
    return {"sources": sources, "shortlist": shortlist}


def materialize(request: MaterializeRequest) -> SliceReceipt:
    """Verify a registered frozen source and write one sealed slice, or refuse."""
    return materialize_source(source_by_id(request.source_id), request)


def materialize_source(
    source: FrozenDatasetSource,
    request: MaterializeRequest,
    *,
    overrides: AdapterOverrides | None = None,
) -> SliceReceipt:
    """The full materialization pipeline over an explicitly supplied source.

    The CLI always resolves through the registry; this entry point exists so a
    synthetic fixture can exercise the identical pipeline with its own declared
    pins instead of pretending to be the official distribution.
    """
    policies = overrides if overrides is not None else AdapterOverrides()
    source.rights.require_accepted(source_id=source.source_id)
    if request.source_dir is not None and request.archives:
        raise DatasetContractError(
            "supply either an extracted source directory or archives, never both.",
            operation="materialize",
            source_id=source.source_id,
        )
    if request.source_dir is None and not request.archives:
        raise DatasetContractError(
            "supply --source-dir or at least one --archive.",
            operation="materialize",
            source_id=source.source_id,
        )
    members = _wanted_members(source, split=request.split, corpus_variant=request.corpus_variant)
    if request.source_dir is not None:
        files = _resolve_from_directory(source, members, request.source_dir)
        return write_slice_bundle(
            request.out,
            _bundle_for(source, request.split, request.corpus_variant, files, policies),
        )
    scratch = Path(tempfile.mkdtemp(prefix="res141-sources-", dir=request.scratch_dir))
    try:
        files = _resolve_from_archives(source, members, request.archives, scratch=scratch)
        bundle = _bundle_for(source, request.split, request.corpus_variant, files, policies)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return write_slice_bundle(request.out, bundle)
