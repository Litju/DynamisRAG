"""Canonical slice artifacts: corpus identity, manifests and verified receipts.

A **slice** is one frozen evaluation unit: one source distribution, one split,
one task, and the canonical RES-140 dataset inputs derived from them. This
module owns the artifact shape and its integrity rules; the family adapters own
what each source's records mean.

Three rules are structural rather than advisory:

* a corpus identity is the digest of the *sorted* ``(document_id,
  content_sha256)`` pairs, so file order, path and platform cannot change it;
* a manifest lists every payload file with its size and digest, and verification
  recomputes the typed dataset from the bytes rather than trusting the manifest;
* the dataset digest is the RES-140 :class:`~dynamisrag.ir.contracts.IrDataset`
  digest, so a slice's ``dataset.json`` is directly the file ``ir score`` consumes.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Final, cast

from dynamisrag.datasets.errors import (
    DatasetArtifactError,
    DatasetContractError,
    DatasetRightsError,
)
from dynamisrag.datasets.primitives import (
    canonical_bytes,
    canonical_sequence_digest,
    digest,
    ordered_ids_sha256,
)
from dynamisrag.datasets.rights import (
    LicenseScope,
    Redistribution,
    RightsDecision,
    RightsOutcome,
)
from dynamisrag.datasets.sources import (
    FAMILY_SCIFACT_OPEN,
    FrozenDatasetSource,
    SourceArtifact,
    SourceFormat,
    SourceMember,
    source_by_id,
)
from dynamisrag.ir.contracts import IrDataset, IrQrel, IrQuery

__all__ = [
    "ATTESTED_CORPUS_IDENTITY",
    "ATTESTED_SENTENCE_POINTERS",
    "ATTESTED_SOURCE_PINS",
    "DATASET_FILENAME",
    "MANIFEST_FILENAME",
    "RIGHTS_FILENAME",
    "SLICE_REVISION",
    "TASK_DOCUMENT_RETRIEVAL",
    "TASK_EVIDENCE_SELECTION",
    "VERIFICATION_ARTIFACT_PROVENANCE_QUALIFIED",
    "VERIFICATION_SELF_CONSISTENCY",
    "VERIFICATION_SOURCE_REGISTRY_MATCHED",
    "CorpusIdentity",
    "CorpusIdentityBuilder",
    "RetrievalArtifacts",
    "SliceBundle",
    "SliceReceipt",
    "build_retrieval_artifacts",
    "bytes_sha256",
    "rights_notice",
    "verify_slice",
    "write_slice_bundle",
]

SLICE_REVISION: Final[str] = "res141-dataset-slice-v1"
TASK_DOCUMENT_RETRIEVAL: Final[str] = "document-retrieval"
TASK_EVIDENCE_SELECTION: Final[str] = "within-document-evidence-selection"
DATASET_FILENAME: Final[str] = "dataset.json"
MANIFEST_FILENAME: Final[str] = "manifest.json"
RIGHTS_FILENAME: Final[str] = "rights.txt"

VERIFICATION_SELF_CONSISTENCY: Final[str] = "self-consistency"
"""Everything recomputable from the sealed bytes was recomputed; the corpus
identity and the source archive/member pins cannot be recomputed without the
original bytes and are reported as attested, not verified."""
VERIFICATION_SOURCE_REGISTRY_MATCHED: Final[str] = "source-registry-matched"
"""Self-consistency plus a registry source pin. This authenticates the *source
identity* - the archive and member digests and the rights decision the manifest
declares are the registered ones. It says nothing about the derived bytes: a
forged corpus identity or a forged qrel set, wrapped in authentic source
metadata, still passes, and is still reported as attested."""
VERIFICATION_ARTIFACT_PROVENANCE_QUALIFIED: Final[str] = "artifact-provenance-qualified"
"""Self-consistency plus an out-of-band expected whole-slice manifest digest, so
every derived claim in that manifest - corpus identity, dataset, judgments,
rights notice, provenance sidecar - is authenticated against a value the caller
obtained outside this slice. This is the only state in which derived artifacts
may be called qualified."""

ATTESTED_SOURCE_PINS: Final[str] = "source-archive-and-member-pins"
ATTESTED_CORPUS_IDENTITY: Final[str] = "corpus-identity"
ATTESTED_SENTENCE_POINTERS: Final[str] = "source-sentence-pointers"
"""SciFact-Open sentence indexes and model ranks are checked for *shape* only.
Whether sentence 3 of document 101 really exists in the S2ORC abstract cannot be
recomputed without the source corpus, so the claim is attested, never verified."""

_CLAIM_INVENTORY: Final[str] = "manifest-file-inventory"
_CLAIM_CLOSED_INVENTORY: Final[str] = "source-family-closed-inventory"
_CLAIM_RIGHTS_NOTICE: Final[str] = "generated-rights-notice"
_CLAIM_DATASET_IDENTITY: Final[str] = "typed-dataset-identity"
_CLAIM_QUERY_STATISTICS: Final[str] = "query-statistics"
_CLAIM_QREL_STATISTICS: Final[str] = "qrel-statistics"
_CLAIM_SOURCE_SPLIT_IDENTITY: Final[str] = "source-split-identity"
_CLAIM_TASK_IDENTITY: Final[str] = "qasper-task-identity"
_CLAIM_TASK_STATISTICS: Final[str] = "qasper-task-statistics"
_CLAIM_PROVENANCE_SIDECAR: Final[str] = "scifact-open-provenance-sidecar"
_VERIFIED_SOURCE_REGISTRY_IDENTITY: Final[str] = "source-registry-identity"
_VERIFIED_DERIVED_ARTIFACT_PROVENANCE: Final[str] = "derived-artifact-provenance"
_SCIFACT_OPEN_SIDECAR: Final[str] = "evidence-provenance.json"


@dataclass(frozen=True)
class CorpusIdentity:
    """The reproducible identity of one retrieval corpus snapshot."""

    policy: str
    document_count: int
    documents_without_text: int
    documents_without_text_ids_sha256: str | None
    sha256: str

    def payload(self) -> dict[str, object]:
        """The hashed description of the corpus snapshot."""
        return {
            "policy": self.policy,
            "document_count": self.document_count,
            "documents_without_text": self.documents_without_text,
            "documents_without_text_ids_sha256": self.documents_without_text_ids_sha256,
            "sha256": self.sha256,
        }


class CorpusIdentityBuilder:
    """Accumulate ``(document_id, content_sha256)`` pairs and finalise once.

    Streaming rather than list-first because SciFact-Open's full corpus is
    500,000 documents / 889 MB: the builder keeps only IDs and digests, never
    document text.
    """

    def __init__(self, *, policy: str) -> None:
        self._policy: Final[str] = policy
        self._entries: dict[str, str] = {}
        self._without_text: list[str] = []

    def add(self, document_id: str, content_sha256: str) -> None:
        """Record one document; a duplicate identifier is refused."""
        if document_id in self._entries:
            raise DatasetContractError(
                f"the corpus declares the document id {document_id!r} more than once.",
                operation="build_corpus_identity",
                item_id=document_id,
            )
        self._entries[document_id] = content_sha256

    def mark_without_text(self, document_id: str) -> None:
        """Record that a document has no embeddable text, without excluding it.

        The document still belongs to the corpus identity; the count exists so a
        manifest can say how many corpus slots carry no text rather than
        pretending the corpus is uniformly usable.
        """
        self._without_text.append(document_id)

    def finalize(self) -> CorpusIdentity:
        """The order-independent identity of every recorded document."""
        ordered: list[dict[str, str]] = [
            {"document_id": document_id, "content_sha256": self._entries[document_id]}
            for document_id in sorted(self._entries)
        ]
        excluded_ids: Iterable[str] = self._without_text
        return CorpusIdentity(
            policy=self._policy,
            document_count=len(ordered),
            documents_without_text=len(self._without_text),
            documents_without_text_ids_sha256=ordered_ids_sha256(excluded_ids),
            sha256=canonical_sequence_digest(ordered),
        )


@dataclass(frozen=True)
class SliceReceipt:
    """The verified identities and claim status of one materialized slice.

    ``verified_claims`` were recomputed from the sealed bytes. ``attested_claims``
    cannot be recomputed without the original corpus bytes and stay attested in
    every mode - what changes is *who vouches for them*:

    * ``source_identity_verified`` - a registry source pin matched, so the
      declared archive/member digests and rights decision are the registered
      ones. It authenticates the source identity only.
    * ``derived_artifacts_authenticated`` - an out-of-band expected manifest
      digest matched, so every derived claim in that manifest is pinned by a
      value obtained outside this slice.

    Neither is implied by the other, and neither is implied by
    ``verification == self-consistency``.
    """

    root: Path
    source_id: str
    split: str
    manifest_sha256: str
    dataset_sha256: str | None
    task_sha256: str | None
    verification: str
    source_identity_verified: bool
    derived_artifacts_authenticated: bool
    trusted_source_sha256: str | None
    expected_manifest_sha256: str | None
    verified_claims: tuple[str, ...]
    attested_claims: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalArtifacts:
    """A complete document-retrieval slice, before it is written to disk."""

    dataset: IrDataset
    dataset_bytes: bytes
    rights_bytes: bytes
    sidecar_files: dict[str, bytes]
    manifest: dict[str, object]

    @property
    def manifest_bytes(self) -> bytes:
        """Canonical manifest bytes."""
        return canonical_bytes(self.manifest)

    @property
    def manifest_sha256(self) -> str:
        """SHA-256 of the canonical manifest."""
        return digest(self.manifest)

    def bundle(self) -> SliceBundle:
        """Attach the document-retrieval task identity and payload files."""
        payloads = {
            DATASET_FILENAME: self.dataset_bytes,
            RIGHTS_FILENAME: self.rights_bytes,
            **self.sidecar_files,
        }
        return SliceBundle(task=TASK_DOCUMENT_RETRIEVAL, manifest=self.manifest, payloads=payloads)


def _qrel_statistics(qrels: Sequence[IrQrel]) -> dict[str, object]:
    relevances = [qrel.relevance for qrel in qrels]
    return {
        "count": len(qrels),
        "negative_count": sum(1 for value in relevances if value < 0),
        "zero_count": sum(1 for value in relevances if value == 0),
        "positive_count": sum(1 for value in relevances if value > 0),
        "min_relevance": min(relevances) if relevances else 0,
        "max_relevance": max(relevances) if relevances else 0,
        "pairs_sha256": digest(
            [[qrel.query_id, qrel.document_id, qrel.relevance] for qrel in qrels]
        ),
    }


def _query_statistics(queries: Sequence[IrQuery]) -> dict[str, object]:
    return {
        "count": len(queries),
        "ids_sha256": ordered_ids_sha256(query.query_id for query in queries),
        "texts_sha256": digest([query.payload() for query in queries]),
    }


def bytes_sha256(content: bytes) -> str:
    """SHA-256 of exact bytes, without a temporary file."""
    return hashlib.sha256(content).hexdigest()


def _payload_files(entries: Mapping[str, bytes]) -> list[dict[str, object]]:
    return [
        {"name": name, "size_bytes": len(content), "sha256": bytes_sha256(content)}
        for name, content in sorted(entries.items())
    ]


def rights_notice(source: FrozenDatasetSource) -> bytes:
    """The human-readable rights notice written beside every slice.

    Generated from the registry so it cannot drift from the accepted decision,
    and deliberately plain text so it survives being copied anywhere.
    """
    rights = source.rights
    lines = [
        "DynamisRAG RES-141 dataset slice rights notice",
        "",
        f"source_id: {source.source_id}",
        f"source_revision: {source.revision}",
        f"documentation: {source.documentation}",
        "",
        f"dataset_license: {rights.dataset_license}",
        f"license_scope: {rights.license_scope.value}",
        f"license_source: {rights.license_source}",
        f"underlying_content: {rights.underlying_content}",
        f"redistribution: {rights.redistribution.value}",
        f"attribution: {rights.attribution}",
        "",
        f"decision: {rights.outcome.value}",
        f"basis: {rights.basis}",
        "",
        "Corpus bytes are never copied into this slice; only identifiers, digests and",
        "judgments derived from the source are written here.",
        "This notice is generated from the frozen source registry and is not legal advice.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def build_retrieval_artifacts(
    *,
    source: FrozenDatasetSource,
    split: str,
    dataset_revision: str,
    corpus: CorpusIdentity,
    queries: Sequence[IrQuery],
    qrels: Sequence[IrQrel],
    diagnostics: Mapping[str, object],
    projection_note: str,
    corpus_variant: str | None = None,
    sidecar_files: Mapping[str, bytes] | None = None,
) -> RetrievalArtifacts:
    """Assemble the canonical dataset, sidecars and manifest for one slice.

    The manifest lists every payload file including the rights notice, so the
    closed inventory on disk is exactly what the manifest describes.
    """
    try:
        dataset = IrDataset(
            source_id=source.source_id,
            source_revision=dataset_revision,
            corpus_sha256=corpus.sha256,
            queries=tuple(sorted(queries, key=lambda query: query.query_id)),
            qrels=tuple(sorted(qrels, key=lambda qrel: (qrel.query_id, qrel.document_id))),
        )
    except ValueError as error:
        raise DatasetContractError(
            f"the derived dataset for {source.source_id!r} {split} violates the canonical IR "
            f"contract ({type(error).__name__}).",
            operation="build_retrieval_artifacts",
            source_id=source.source_id,
            split=split,
        ) from None
    dataset_bytes = canonical_bytes(dataset.payload())
    rights_bytes = rights_notice(source)
    sidecars: dict[str, bytes] = dict(sidecar_files or {})
    files = _payload_files(
        {
            DATASET_FILENAME: dataset_bytes,
            RIGHTS_FILENAME: rights_bytes,
            **sidecars,
        }
    )
    manifest: dict[str, object] = {
        "artifact_revision": SLICE_REVISION,
        "source": source.payload(),
        "split": split,
        "task": TASK_DOCUMENT_RETRIEVAL,
        "dataset_revision": dataset_revision,
        "corpus_variant": corpus_variant,
        "projection_note": projection_note,
        "corpus": corpus.payload(),
        "queries": _query_statistics(dataset.queries),
        "qrels": _qrel_statistics(dataset.qrels),
        "diagnostics": dict(diagnostics),
        "dataset_sha256": dataset.sha256,
        "files": files,
    }
    return RetrievalArtifacts(
        dataset=dataset,
        dataset_bytes=dataset_bytes,
        rights_bytes=rights_bytes,
        sidecar_files=sidecars,
        manifest=manifest,
    )


def _required_string(source: Mapping[str, object], key: str) -> str:
    value = source.get(key)
    if not isinstance(value, str):
        raise DatasetArtifactError(
            "the slice dataset does not contain the required field.",
            operation="verify_slice",
            item_id=key,
        )
    return value


def _object_rows(value: object, *, field: str) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise DatasetArtifactError(
            "the slice dataset does not contain the required list.",
            operation="verify_slice",
            item_id=field,
        )
    rows: list[dict[str, object]] = []
    for row in cast("list[object]", value):
        if not isinstance(row, dict):
            raise DatasetArtifactError(
                "the slice dataset contains a non-object row.",
                operation="verify_slice",
                item_id=field,
            )
        rows.append({str(key): item for key, item in cast("dict[object, object]", row).items()})
    return rows


@dataclass(frozen=True)
class SliceBundle:
    """Every payload file, the manifest that lists them, and the task identity.

    The manifest is *not* part of ``payloads``: it lists every other file with
    its size and digest, so including it in its own inventory would be circular.
    """

    task: str
    manifest: dict[str, object]
    payloads: dict[str, bytes]

    @property
    def manifest_bytes(self) -> bytes:
        """Canonical manifest bytes."""
        return canonical_bytes(self.manifest)

    @property
    def manifest_sha256(self) -> str:
        """SHA-256 of the canonical manifest."""
        return digest(self.manifest)


def write_slice_bundle(out: Path, bundle: SliceBundle) -> SliceReceipt:
    """Stage a bundle, then atomically rename it into an absent path.

    A previous scientific result is never overwritten, and a failed write leaves
    no slice behind: the final directory name appears only after every file and
    the manifest exist.
    """
    destination = out.resolve(strict=False)
    if destination.exists():
        raise DatasetArtifactError(
            "refusing to overwrite an existing slice directory.",
            operation="write_slice_bundle",
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".res141-stage-", dir=destination.parent))
    try:
        for name, content in bundle.payloads.items():
            (stage / name).write_bytes(content)
        (stage / MANIFEST_FILENAME).write_bytes(bundle.manifest_bytes)
        if destination.exists():
            raise DatasetArtifactError(
                "another process published this slice already.",
                operation="write_slice_bundle",
            )
        stage.rename(destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return verify_slice(destination)


def _restore_dataset_bytes(content: bytes, *, name: str) -> IrDataset:
    documents = _parse_canonical_object(content, name=name)
    try:
        return IrDataset(
            source_id=_required_string(documents, "source_id"),
            source_revision=_required_string(documents, "source_revision"),
            corpus_sha256=_required_string(documents, "corpus_sha256"),
            queries=tuple(
                IrQuery(
                    query_id=_required_string(row, "query_id"),
                    text=_required_string(row, "text"),
                )
                for row in _object_rows(documents.get("queries"), field="queries")
            ),
            qrels=tuple(
                IrQrel(
                    query_id=_required_string(row, "query_id"),
                    document_id=_required_string(row, "document_id"),
                    relevance=_required_relevance(row),
                )
                for row in _object_rows(documents.get("qrels"), field="qrels")
            ),
        )
    except (TypeError, ValueError) as error:
        raise DatasetArtifactError(
            f"the slice dataset is not a valid canonical IR dataset ({type(error).__name__}).",
            operation="verify_slice",
            item_id=name,
        ) from None


def _required_relevance(row: Mapping[str, object]) -> int:
    value = row.get("relevance")
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetArtifactError(
            "a slice qrel holds a non-integer relevance.",
            operation="verify_slice",
            item_id="relevance",
        )
    return value


def _parse_canonical_object(content: bytes, *, name: str) -> dict[str, object]:
    try:
        value: object = json.loads(content)
    except (UnicodeDecodeError, ValueError) as error:
        raise DatasetArtifactError(
            f"the slice file {name!r} is not valid JSON ({type(error).__name__}).",
            operation="verify_slice",
            item_id=name,
        ) from None
    if not isinstance(value, dict):
        raise DatasetArtifactError(
            f"the slice file {name!r} is not a JSON object.",
            operation="verify_slice",
            item_id=name,
        )
    document = cast("dict[object, object]", value)
    if canonical_bytes(document) != content:
        raise DatasetArtifactError(
            f"the slice file {name!r} is not canonical JSON.",
            operation="verify_slice",
            item_id=name,
        )
    return {str(key): item for key, item in document.items()}


def _load_manifest(root: Path) -> tuple[bytes, dict[str, object]]:
    if not root.is_dir() or root.is_symlink():
        raise DatasetArtifactError(
            "a slice root must be a regular directory.",
            operation="verify_slice",
        )
    try:
        manifest_bytes = (root / MANIFEST_FILENAME).read_bytes()
    except OSError as error:
        raise DatasetArtifactError(
            f"the slice has no readable {MANIFEST_FILENAME} ({type(error).__name__}).",
            operation="verify_slice",
        ) from None
    manifest = _parse_canonical_object(manifest_bytes, name=MANIFEST_FILENAME)
    if manifest.get("artifact_revision") != SLICE_REVISION:
        raise DatasetArtifactError(
            "the slice manifest revision is incompatible.",
            operation="verify_slice",
            expected=SLICE_REVISION,
            observed=str(manifest.get("artifact_revision")),
        )
    return manifest_bytes, manifest


def _manifest_identity(manifest: dict[str, object]) -> tuple[str, str]:
    raw_source = manifest.get("source")
    if not isinstance(raw_source, dict):
        raise DatasetArtifactError(
            "the slice manifest does not describe its source.",
            operation="verify_slice",
        )
    source = cast("dict[str, object]", raw_source)
    source_id = source.get("source_id")
    if not isinstance(source_id, str):
        raise DatasetArtifactError(
            "the slice manifest source has no source_id.",
            operation="verify_slice",
        )
    return source_id, str(manifest.get("split"))


def required_payload_names(source: FrozenDatasetSource, *, task: str) -> tuple[str, ...]:
    """The closed payload inventory one source family and task must carry.

    A manifest that lists its own inventory cannot decide what *should* be there:
    deleting a required file and re-signing the manifest would otherwise pass,
    which is exactly how a provenance sidecar could disappear from a SciFact-Open
    slice. The inventory is therefore a function of the declared source family
    and the declared task, and it is closed in both directions - a missing
    required file and a foreign extra file are both refusals.

    The family is itself part of what the manifest declares, so this closes the
    inventory *relative to that declaration*; authenticating the declaration
    itself is what ``--registered-source`` and ``--expect-manifest-sha256`` are
    for.
    """
    if task == TASK_DOCUMENT_RETRIEVAL:
        names = {DATASET_FILENAME, RIGHTS_FILENAME}
        if source.family == FAMILY_SCIFACT_OPEN:
            names.add(_SCIFACT_OPEN_SIDECAR)
        return tuple(sorted(names))
    if task == TASK_EVIDENCE_SELECTION:
        from dynamisrag.datasets.qasper import TASK_FILENAME

        return (TASK_FILENAME, RIGHTS_FILENAME)
    raise DatasetArtifactError(
        f"the slice declares an unknown task {task!r}.",
        operation="verify_slice",
        source_id=source.source_id,
        observed=task,
    )


def _verify_inventory(
    root: Path,
    manifest: dict[str, object],
    *,
    source: FrozenDatasetSource,
    task: str,
    split: str,
) -> list[dict[str, object]]:
    source_id = source.source_id
    raw_entries = manifest.get("files")
    if not isinstance(raw_entries, list):
        raise DatasetArtifactError(
            "the slice manifest does not list its payload files.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
        )
    entries = cast("list[object]", raw_entries)
    listed = [cast("dict[str, object]", entry) for entry in entries if isinstance(entry, dict)]
    names = [str(entry.get("name")) for entry in listed]
    if len(listed) != len(entries) or sorted(set(names)) != sorted(names):
        raise DatasetArtifactError(
            "the slice manifest file inventory is invalid.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
        )
    required = set(required_payload_names(source, task=task))
    if set(names) != required:
        raise DatasetArtifactError(
            "the slice payload inventory is not the closed inventory this source family "
            "and task require.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
            expected=str(sorted(required)),
            observed=str(sorted(names)),
        )
    expected_names = {MANIFEST_FILENAME, *names}
    actual_names = {path.name for path in root.iterdir()}
    if expected_names != actual_names:
        raise DatasetArtifactError(
            "the slice directory inventory differs from the manifest.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
            expected=str(sorted(expected_names)),
            observed=str(sorted(actual_names)),
        )
    if any(path.is_symlink() or not path.is_file() for path in root.iterdir()):
        raise DatasetArtifactError(
            "every slice entry must be a regular file.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
        )
    for entry in listed:
        name = str(entry.get("name"))
        size_bytes = entry.get("size_bytes")
        sha256 = entry.get("sha256")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int):
            raise DatasetArtifactError(
                "a manifest file entry has an invalid size.",
                operation="verify_slice",
                source_id=source_id,
                item_id=name,
            )
        content = (root / name).read_bytes()
        observed = bytes_sha256(content)
        if len(content) != size_bytes or observed != str(sha256):
            raise DatasetArtifactError(
                f"the slice file {name!r} disagrees with its manifest entry.",
                operation="verify_slice",
                source_id=source_id,
                item_id=name,
                expected=str(sha256),
                observed=observed,
            )
    return listed


def _manifest_object(
    manifest: dict[str, object], *, field: str, source_id: str, split: str
) -> dict[str, object]:
    value = manifest.get(field)
    if not isinstance(value, dict):
        raise DatasetArtifactError(
            f"the slice manifest {field!r} is not an object.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
        )
    return cast("dict[str, object]", value)


def _payload_text(payload: Mapping[str, object], field: str, *, item_id: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise DatasetArtifactError(
            f"the slice manifest source field {field!r} must be non-blank text.",
            operation="verify_slice",
            item_id=item_id,
        )
    return value


def _payload_int(payload: Mapping[str, object], field: str, *, item_id: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetArtifactError(
            f"the slice manifest source field {field!r} must be an integer.",
            operation="verify_slice",
            item_id=item_id,
        )
    return value


def _member_from_payload(raw: object, *, source_id: str) -> SourceMember:
    if not isinstance(raw, dict):
        raise DatasetArtifactError(
            "a slice manifest source member is not an object.",
            operation="verify_slice",
            source_id=source_id,
        )
    member = cast("dict[str, object]", raw)
    return SourceMember(
        name=_payload_text(member, "name", item_id=source_id),
        size_bytes=_payload_int(member, "size_bytes", item_id=source_id),
        sha256=_payload_text(member, "sha256", item_id=source_id),
    )


def _artifact_from_payload(raw: object, *, source_id: str) -> SourceArtifact:
    if not isinstance(raw, dict):
        raise DatasetArtifactError(
            "a slice manifest source artifact is not an object.",
            operation="verify_slice",
            source_id=source_id,
        )
    artifact = cast("dict[str, object]", raw)
    members_raw = artifact.get("members")
    if not isinstance(members_raw, list):
        raise DatasetArtifactError(
            "a slice manifest source artifact does not list its members.",
            operation="verify_slice",
            source_id=source_id,
        )
    return SourceArtifact(
        url=_payload_text(artifact, "url", item_id=source_id),
        archive_name=_payload_text(artifact, "archive_name", item_id=source_id),
        format=SourceFormat(_payload_text(artifact, "format", item_id=source_id)),
        size_bytes=_payload_int(artifact, "size_bytes", item_id=source_id),
        sha256=_payload_text(artifact, "sha256", item_id=source_id),
        members=tuple(
            _member_from_payload(entry, source_id=source_id)
            for entry in cast("list[object]", members_raw)
        ),
    )


def _rights_from_payload(raw: object, *, source_id: str) -> RightsDecision:
    if not isinstance(raw, dict):
        raise DatasetArtifactError(
            "the slice manifest source does not describe its rights decision.",
            operation="verify_slice",
            source_id=source_id,
        )
    rights = cast("dict[str, object]", raw)
    return RightsDecision(
        dataset_license=_payload_text(rights, "dataset_license", item_id=source_id),
        license_scope=LicenseScope(_payload_text(rights, "license_scope", item_id=source_id)),
        license_source=_payload_text(rights, "license_source", item_id=source_id),
        underlying_content=_payload_text(rights, "underlying_content", item_id=source_id),
        redistribution=Redistribution(_payload_text(rights, "redistribution", item_id=source_id)),
        attribution=_payload_text(rights, "attribution", item_id=source_id),
        outcome=RightsOutcome(_payload_text(rights, "outcome", item_id=source_id)),
        basis=_payload_text(rights, "basis", item_id=source_id),
    )


def _source_from_payload(payload: object, *, source_id: str, split: str) -> FrozenDatasetSource:
    """Rebuild the manifest's source description, refusing anything malformed.

    This proves internal consistency of the source metadata and the rights
    decision; it is not a trust anchor. Qualified verification supplies the trust
    anchor (a registry source pin or an expected manifest digest) separately.
    """
    if not isinstance(payload, dict):
        raise DatasetArtifactError(
            "the slice manifest does not describe its source.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
        )
    source = cast("dict[str, object]", payload)
    try:
        artifacts_raw = source.get("artifacts")
        if not isinstance(artifacts_raw, list):
            raise DatasetArtifactError(
                "the slice manifest source does not list its artifacts.",
                operation="verify_slice",
                source_id=source_id,
                split=split,
            )
        return FrozenDatasetSource(
            source_id=_payload_text(source, "source_id", item_id=source_id),
            family=_payload_text(source, "family", item_id=source_id),
            revision=_payload_text(source, "revision", item_id=source_id),
            documentation=_payload_text(source, "documentation", item_id=source_id),
            content_note=_payload_text(source, "content_note", item_id=source_id),
            artifacts=tuple(
                _artifact_from_payload(entry, source_id=source_id)
                for entry in cast("list[object]", artifacts_raw)
            ),
            rights=_rights_from_payload(source.get("rights"), source_id=source_id),
        )
    except DatasetArtifactError:
        raise
    except (DatasetContractError, DatasetRightsError, ValueError, TypeError) as error:
        raise DatasetArtifactError(
            f"the slice manifest source description is invalid ({type(error).__name__}).",
            operation="verify_slice",
            source_id=source_id,
            split=split,
        ) from None


def _declared_revision_candidates(
    manifest: dict[str, object], *, source: FrozenDatasetSource, split: str
) -> set[str]:
    """Every dataset revision the manifest's own declarations can justify.

    The base form is ``source.revision.split``; a corpus variant or a declared
    protocol policy appends one token. A revision outside this set is a
    mismatched split or source identity, not an acceptable alias.
    """
    base = f"{source.revision}.{split}"
    candidates = {base}
    variant = manifest.get("corpus_variant")
    if isinstance(variant, str) and variant:
        candidates.add(f"{base}.{variant}")
    diagnostics = manifest.get("diagnostics")
    if isinstance(diagnostics, dict):
        policy = cast("dict[str, object]", diagnostics).get("self_document_policy")
        if isinstance(policy, str) and policy:
            candidates.add(f"{base}.{policy}")
    return candidates


def _verify_document_retrieval(
    root: Path,
    manifest: dict[str, object],
    *,
    source: FrozenDatasetSource,
    source_id: str,
    split: str,
) -> IrDataset:
    """Recompute every dataset claim the sealed payload can prove."""
    dataset_bytes = (root / DATASET_FILENAME).read_bytes()
    dataset = _restore_dataset_bytes(dataset_bytes, name=DATASET_FILENAME)
    if dataset.sha256 != bytes_sha256(dataset_bytes):
        raise DatasetArtifactError(
            "the slice dataset bytes are not the canonical payload of the typed dataset.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
        )
    if dataset.sha256 != str(manifest.get("dataset_sha256")):
        raise DatasetArtifactError(
            "the slice dataset identity disagrees with the manifest.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
        )
    if dataset.source_id != source_id:
        raise DatasetArtifactError(
            "the slice dataset names a different source than the manifest.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
        )
    variant = manifest.get("corpus_variant")
    if variant is not None and not isinstance(variant, str):
        raise DatasetArtifactError(
            "the slice corpus variant is not text.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
        )
    candidates = _declared_revision_candidates(manifest, source=source, split=split)
    if dataset.source_revision not in candidates:
        raise DatasetArtifactError(
            "the slice dataset revision does not match the manifest source and split.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
            expected=str(sorted(candidates)),
            observed=dataset.source_revision,
        )
    if manifest.get("dataset_revision") != dataset.source_revision:
        raise DatasetArtifactError(
            "the slice manifest dataset revision disagrees with the dataset payload.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
        )
    if manifest.get("queries") != _query_statistics(dataset.queries):
        raise DatasetArtifactError(
            "the slice manifest query statistics disagree with the dataset payload.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
            item_id="queries",
        )
    if manifest.get("qrels") != _qrel_statistics(dataset.qrels):
        raise DatasetArtifactError(
            "the slice manifest qrel statistics disagree with the dataset payload.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
            item_id="qrels",
        )
    corpus_claim = manifest.get("corpus")
    if not isinstance(corpus_claim, dict):
        raise DatasetArtifactError(
            "the slice manifest does not describe its corpus identity.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
        )
    if cast("dict[str, object]", corpus_claim).get("sha256") != dataset.corpus_sha256:
        raise DatasetArtifactError(
            "the slice corpus identity disagrees with the dataset payload.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
            item_id="corpus",
        )
    return dataset


@dataclass(frozen=True)
class _SidecarStats:
    """Recoverable statistics of one SciFact-Open provenance sidecar."""

    pairs: tuple[tuple[str, str], ...]
    document_ids: frozenset[str]
    citation: int
    pooling: int
    support: int
    contradict: int
    in_pool: int


def _sidecar_sentences(raw: object, *, source_id: str, split: str) -> tuple[int, ...]:
    """Validate one link's sentence indexes: non-negative, ascending, unique.

    Shape only. That sentence N of this abstract exists is a source fact no
    sealed slice can prove, and the receipt says so through
    :data:`ATTESTED_SENTENCE_POINTERS`.
    """
    if not isinstance(raw, list):
        raise DatasetArtifactError(
            "a SciFact-Open provenance link has a non-list sentences field.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
        )
    indexes: list[int] = []
    for value in cast("list[object]", raw):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DatasetArtifactError(
                "a SciFact-Open provenance sentence index is not a non-negative integer.",
                operation="verify_slice",
                source_id=source_id,
                split=split,
            )
        indexes.append(value)
    if any(later <= earlier for earlier, later in pairwise(indexes)):
        raise DatasetArtifactError(
            "a SciFact-Open provenance link lists its sentence indexes out of order or twice.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
        )
    return tuple(indexes)


def _sidecar_model_ranks(
    raw: object, *, provenance: str, source_id: str, split: str
) -> tuple[tuple[str, int], ...] | None:
    """Validate one link's model ranks against its declared provenance.

    ``citation`` evidence was written by a human and carries no model rank;
    ``pooling`` evidence came from a retrieval model and must carry one per
    judging model. Empty, negative, non-integer and unnamed ranks are refusals,
    because a pooling link without ranks cannot be interpreted as pooled
    evidence at all.
    """
    if provenance == "citation":
        if raw is not None:
            raise DatasetArtifactError(
                "SciFact-Open citation evidence must carry null model ranks.",
                operation="verify_slice",
                source_id=source_id,
                split=split,
            )
        return None
    if not isinstance(raw, dict) or not raw:
        raise DatasetArtifactError(
            "SciFact-Open pooling evidence must carry model ranks.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
        )
    pairs: list[tuple[str, int]] = []
    for model, rank in cast("dict[object, object]", raw).items():
        if (
            not isinstance(model, str)
            or not model.strip()
            or isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank < 0
        ):
            raise DatasetArtifactError(
                "a SciFact-Open model rank is not a non-negative integer under a named model.",
                operation="verify_slice",
                source_id=source_id,
                split=split,
                item_id=str(model),
            )
        pairs.append((model, rank))
    return tuple(sorted(pairs))


def _parse_sidecar_links(raw_links: list[object], *, source_id: str, split: str) -> _SidecarStats:
    """Validate every sidecar link and recompute its statistics, or refuse."""
    pairs: list[tuple[str, str]] = []
    document_ids: set[str] = set()
    citation = pooling = support = contradict = in_pool = 0
    for raw_link in raw_links:
        if not isinstance(raw_link, dict):
            raise DatasetArtifactError(
                "a SciFact-Open provenance link is not an object.",
                operation="verify_slice",
                source_id=source_id,
                split=split,
            )
        link = {str(key): value for key, value in cast("dict[object, object]", raw_link).items()}
        query_id = link.get("query_id")
        document_id = link.get("document_id")
        relevance = link.get("relevance")
        provenance = link.get("provenance")
        label = link.get("label")
        released = link.get("in_released_pool")
        if (
            not isinstance(query_id, str)
            or not query_id
            or not isinstance(document_id, str)
            or not document_id
            or isinstance(relevance, bool)
            or relevance != 1
            or not isinstance(released, bool)
            or provenance not in {"citation", "pooling"}
            or label not in {"SUPPORT", "CONTRADICT"}
        ):
            raise DatasetArtifactError(
                "a SciFact-Open provenance link is malformed.",
                operation="verify_slice",
                source_id=source_id,
                split=split,
            )
        _sidecar_sentences(link.get("sentences"), source_id=source_id, split=split)
        _sidecar_model_ranks(
            link.get("model_ranks"),
            provenance=cast("str", provenance),
            source_id=source_id,
            split=split,
        )
        pair = (query_id, document_id)
        if pairs and pair <= pairs[-1]:
            raise DatasetArtifactError(
                "the SciFact-Open provenance links are not unique and ascending.",
                operation="verify_slice",
                source_id=source_id,
                split=split,
            )
        pairs.append(pair)
        document_ids.add(document_id)
        citation += provenance == "citation"
        pooling += provenance == "pooling"
        support += label == "SUPPORT"
        contradict += label == "CONTRADICT"
        in_pool += released
    return _SidecarStats(
        pairs=tuple(pairs),
        document_ids=frozenset(document_ids),
        citation=citation,
        pooling=pooling,
        support=support,
        contradict=contradict,
        in_pool=in_pool,
    )


def _verify_scifact_open_sidecar(
    root: Path,
    manifest: dict[str, object],
    dataset: IrDataset,
    *,
    source: FrozenDatasetSource,
    source_id: str,
    split: str,
) -> None:
    """Recompute every provenance-sidecar statistic the sealed bytes can prove."""
    from dynamisrag.datasets.scifact_open import PROJECTION_REVISION

    payload = _parse_canonical_object(
        (root / _SCIFACT_OPEN_SIDECAR).read_bytes(), name=_SCIFACT_OPEN_SIDECAR
    )
    if payload.get("artifact_revision") != PROJECTION_REVISION:
        raise DatasetArtifactError(
            "the SciFact-Open provenance sidecar revision is incompatible.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
        )
    if payload.get("source_id") != source_id or payload.get("source_revision") != source.revision:
        raise DatasetArtifactError(
            "the SciFact-Open provenance sidecar names a different source.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
        )
    if payload.get("corpus_variant") != manifest.get("corpus_variant"):
        raise DatasetArtifactError(
            "the SciFact-Open provenance sidecar names a different corpus variant.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
        )
    if payload.get("judgement_status") != "pooled-partial":
        raise DatasetArtifactError(
            "the SciFact-Open provenance sidecar declares an unknown judgment status.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
        )
    raw_links = payload.get("links")
    if not isinstance(raw_links, list):
        raise DatasetArtifactError(
            "the SciFact-Open provenance sidecar does not list its links.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
        )
    stats = _parse_sidecar_links(cast("list[object]", raw_links), source_id=source_id, split=split)
    if stats.pairs != tuple((qrel.query_id, qrel.document_id) for qrel in dataset.qrels):
        raise DatasetArtifactError(
            "the SciFact-Open provenance links disagree with the dataset qrels.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
        )
    if any(qrel.relevance != 1 for qrel in dataset.qrels):
        raise DatasetArtifactError(
            "a SciFact-Open qrel is not binary evidence-presence relevance.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
        )
    expected_counts: dict[str, object] = {
        "citation_links": stats.citation,
        "pooling_links": stats.pooling,
        "support_links": stats.support,
        "contradict_links": stats.contradict,
    }
    if payload.get("counts") != expected_counts:
        raise DatasetArtifactError(
            "the SciFact-Open provenance sidecar counts disagree with its links.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
        )
    pool = payload.get("pool")
    if not isinstance(pool, dict):
        raise DatasetArtifactError(
            "the SciFact-Open provenance sidecar does not describe its pool.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
        )
    pool_claim = cast("dict[str, object]", pool)
    outside = len(stats.pairs) - stats.in_pool
    if pool_claim.get("evidence_links_in_pool") != stats.in_pool or (
        pool_claim.get("evidence_links_outside_pool") != outside
    ):
        raise DatasetArtifactError(
            "the SciFact-Open provenance pool block disagrees with its links.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
        )
    diagnostics = _manifest_object(manifest, field="diagnostics", source_id=source_id, split=split)
    expected_diagnostics: tuple[tuple[str, object], ...] = (
        ("claims", len(dataset.queries)),
        ("evidence_links", len(stats.pairs)),
        ("evidence_documents", len(stats.document_ids)),
        ("citation_links", stats.citation),
        ("pooling_links", stats.pooling),
        ("support_links", stats.support),
        ("contradict_links", stats.contradict),
        ("evidence_links_in_pool", stats.in_pool),
        ("evidence_links_outside_pool", outside),
        ("pool_pairs", pool_claim.get("pairs")),
        ("pool_union_documents", pool_claim.get("union_documents")),
        ("evidence_document_ids_sha256", ordered_ids_sha256(stats.document_ids)),
        ("provenance_sidecar_sha256", digest(payload)),
    )
    for field, expected in expected_diagnostics:
        if diagnostics.get(field) != expected:
            raise DatasetArtifactError(
                f"the slice manifest diagnostic {field!r} disagrees with the payload.",
                operation="verify_slice",
                source_id=source_id,
                split=split,
                item_id=field,
            )


def _verify_qasper(
    root: Path,
    manifest: dict[str, object],
    *,
    source_id: str,
    split: str,
) -> str:
    """Recompute every QASPER task claim the sealed payload can prove."""
    from dynamisrag.datasets.qasper import ANCHOR_POLICY, TASK_FILENAME, read_task_bytes

    task = read_task_bytes(
        (root / TASK_FILENAME).read_bytes(),
        expected_sha256=str(manifest.get("task_sha256")),
        source_id=source_id,
        split=split,
    )
    if task.split != split:
        raise DatasetArtifactError(
            "the QASPER task declares a different split than the manifest.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
        )
    if manifest.get("counts") != task.counts:
        raise DatasetArtifactError(
            "the slice manifest QASPER counts disagree with the task payload.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
            item_id="counts",
        )
    if manifest.get("expected") != dict(task.expected):
        raise DatasetArtifactError(
            "the slice manifest QASPER expectations disagree with the task payload.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
            item_id="expected",
        )
    if manifest.get("scoring") != dict(task.scoring):
        raise DatasetArtifactError(
            "the slice manifest QASPER scoring policy disagrees with the task payload.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
            item_id="scoring",
        )
    if manifest.get("anchor_policy") != ANCHOR_POLICY:
        raise DatasetArtifactError(
            "the slice manifest declares an unknown QASPER anchor policy.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
        )
    if manifest.get("source") != dict(task.source_payload):
        raise DatasetArtifactError(
            "the QASPER task source identity disagrees with the manifest.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
        )
    return task.sha256


def _registry_anchor(
    source: FrozenDatasetSource,
    *,
    source_id: str,
    split: str,
    trusted_source: FrozenDatasetSource | None,
    verified: list[str],
) -> FrozenDatasetSource | None:
    """Apply the source-registry trust anchor, authenticating the source identity only.

    Returns the trusted source so the receipt can report its digest. A registry
    match says the manifest describes the registered distribution's pins and
    rights decision; it says nothing about the derived bytes, so the derived
    artifacts stay attested and the receipt is ``source-registry-matched``.
    """
    if trusted_source is None:
        return None
    if trusted_source.source_id != source_id or trusted_source.payload() != source.payload():
        raise DatasetArtifactError(
            "the slice manifest source does not match the caller-trusted source identity.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
        )
    verified.append(_VERIFIED_SOURCE_REGISTRY_IDENTITY)
    return trusted_source


def _manifest_anchor(
    manifest_bytes: bytes,
    *,
    source_id: str,
    split: str,
    expected_manifest_sha256: str | None,
    verified: list[str],
) -> str | None:
    """Apply the out-of-band whole-slice digest anchor, or return ``None``.

    The digest covers every derived claim in the manifest, so a match is the only
    state in which the derived artifacts may be called qualified.
    """
    if expected_manifest_sha256 is None:
        return None
    observed = bytes_sha256(manifest_bytes)
    if observed != expected_manifest_sha256:
        raise DatasetArtifactError(
            "the slice manifest does not match the caller-supplied expected digest.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
            expected=expected_manifest_sha256,
            observed=observed,
        )
    verified.append(_VERIFIED_DERIVED_ARTIFACT_PROVENANCE)
    return expected_manifest_sha256


def verify_slice(
    root: Path,
    *,
    trusted_source: FrozenDatasetSource | None = None,
    registered_source: bool = False,
    expected_manifest_sha256: str | None = None,
) -> SliceReceipt:
    """Verify a materialized slice, or refuse it.

    Self-consistency mode (the default) recomputes everything the sealed bytes
    can prove: the closed inventory, the typed dataset or QASPER task identity,
    query and qrel statistics, source/split identity, the generated rights notice
    and any provenance-sidecar statistics. Corpus identity counts and the
    original archive/member pins cannot be recomputed without the source bytes
    and are returned as *attested*, never as verified.

    Two trust anchors are supported and they authenticate different things:

    * ``trusted_source`` (or ``registered_source`` to resolve the registry pin by
      the manifest's own source id) requires the manifest source description to
      equal the trusted source exactly, and reports
      ``source-registry-matched``. That authenticates the **source identity** -
      the archive and member pins and the rights decision. It authenticates
      nothing about the derived corpus identity, judgments or evidence, which
      stay attested, so it must never be reported as qualified derived data.
    * ``expected_manifest_sha256`` requires the canonical manifest digest to
      match an out-of-band value and reports ``artifact-provenance-qualified``.
      Because the digest covers every derived claim in the manifest, this is the
      state in which derived artifacts may be called qualified.

    There is no trust-on-first-use: a manifest that matches nothing trusted stays
    self-consistency-only, and the receipt says exactly which of the two anchors
    was used.
    """
    manifest_bytes, manifest = _load_manifest(root)
    source_id, split = _manifest_identity(manifest)
    source = _source_from_payload(manifest.get("source"), source_id=source_id, split=split)
    if registered_source:
        if trusted_source is not None:
            raise DatasetContractError(
                "supply either a trusted source object or the registry lookup, never both.",
                operation="verify_slice",
                source_id=source_id,
                split=split,
            )
        trusted_source = source_by_id(source_id)
    verified: list[str] = []
    task = str(manifest.get("task"))
    listed = _verify_inventory(root, manifest, source=source, task=task, split=split)
    verified.extend((_CLAIM_INVENTORY, _CLAIM_CLOSED_INVENTORY))
    trusted_source = _registry_anchor(
        source,
        source_id=source_id,
        split=split,
        trusted_source=trusted_source,
        verified=verified,
    )
    expected_manifest = _manifest_anchor(
        manifest_bytes,
        source_id=source_id,
        split=split,
        expected_manifest_sha256=expected_manifest_sha256,
        verified=verified,
    )
    if (root / RIGHTS_FILENAME).read_bytes() != rights_notice(source):
        raise DatasetArtifactError(
            "the slice rights notice is not the notice generated from the manifest source.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
        )
    verified.append(_CLAIM_RIGHTS_NOTICE)
    attested: list[str] = [ATTESTED_SOURCE_PINS]
    dataset_sha256: str | None = None
    task_sha256: str | None = None
    if task == TASK_DOCUMENT_RETRIEVAL:
        dataset = _verify_document_retrieval(
            root, manifest, source=source, source_id=source_id, split=split
        )
        dataset_sha256 = dataset.sha256
        verified.extend(
            (
                _CLAIM_DATASET_IDENTITY,
                _CLAIM_QUERY_STATISTICS,
                _CLAIM_QREL_STATISTICS,
                _CLAIM_SOURCE_SPLIT_IDENTITY,
            )
        )
        attested.append(ATTESTED_CORPUS_IDENTITY)
        if _SCIFACT_OPEN_SIDECAR in {str(entry.get("name")) for entry in listed}:
            _verify_scifact_open_sidecar(
                root, manifest, dataset, source=source, source_id=source_id, split=split
            )
            verified.append(_CLAIM_PROVENANCE_SIDECAR)
            attested.append(ATTESTED_SENTENCE_POINTERS)
    elif task == TASK_EVIDENCE_SELECTION:
        task_sha256 = _verify_qasper(root, manifest, source_id=source_id, split=split)
        verified.extend((_CLAIM_TASK_IDENTITY, _CLAIM_TASK_STATISTICS))
    else:
        raise DatasetArtifactError(
            f"the slice declares an unknown task {task!r}.",
            operation="verify_slice",
            source_id=source_id,
            split=split,
            observed=task,
        )
    if expected_manifest is not None:
        verification = VERIFICATION_ARTIFACT_PROVENANCE_QUALIFIED
    elif trusted_source is not None:
        verification = VERIFICATION_SOURCE_REGISTRY_MATCHED
    else:
        verification = VERIFICATION_SELF_CONSISTENCY
    return SliceReceipt(
        root=root,
        source_id=source_id,
        split=split,
        manifest_sha256=bytes_sha256(manifest_bytes),
        dataset_sha256=dataset_sha256,
        task_sha256=task_sha256,
        verification=verification,
        source_identity_verified=trusted_source is not None,
        derived_artifacts_authenticated=expected_manifest is not None,
        trusted_source_sha256=trusted_source.sha256 if trusted_source is not None else None,
        expected_manifest_sha256=expected_manifest,
        verified_claims=tuple(verified),
        attested_claims=tuple(attested),
    )
