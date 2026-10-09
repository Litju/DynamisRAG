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
from pathlib import Path
from typing import Final, cast

from dynamisrag.datasets.errors import DatasetArtifactError, DatasetContractError
from dynamisrag.datasets.primitives import (
    canonical_bytes,
    canonical_sequence_digest,
    digest,
    ordered_ids_sha256,
)
from dynamisrag.datasets.sources import FrozenDatasetSource
from dynamisrag.ir.contracts import IrDataset, IrQrel, IrQuery

__all__ = [
    "DATASET_FILENAME",
    "MANIFEST_FILENAME",
    "RIGHTS_FILENAME",
    "SLICE_REVISION",
    "TASK_DOCUMENT_RETRIEVAL",
    "TASK_EVIDENCE_SELECTION",
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
    """The verified identities of one materialized slice."""

    root: Path
    source_id: str
    split: str
    manifest_sha256: str
    dataset_sha256: str | None
    task_sha256: str | None


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


def _verify_inventory(
    root: Path, manifest: dict[str, object], *, source_id: str, split: str
) -> list[dict[str, object]]:
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


def _verify_task_identity(
    root: Path, manifest: dict[str, object], *, source_id: str, split: str
) -> tuple[str | None, str | None]:
    task = str(manifest.get("task"))
    if task == TASK_DOCUMENT_RETRIEVAL:
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
        return dataset.sha256, None
    if task == TASK_EVIDENCE_SELECTION:
        from dynamisrag.datasets.qasper import TASK_FILENAME, verify_task_bytes

        task_sha256 = verify_task_bytes(
            (root / TASK_FILENAME).read_bytes(),
            expected_sha256=str(manifest.get("task_sha256")),
            source_id=source_id,
            split=split,
        )
        return None, task_sha256
    raise DatasetArtifactError(
        f"the slice declares an unknown task {task!r}.",
        operation="verify_slice",
        source_id=source_id,
        split=split,
        observed=task,
    )


def verify_slice(root: Path) -> SliceReceipt:
    """Verify a materialized slice against its manifest, or refuse it.

    A manifest is not a trust anchor: the dataset is re-parsed into the typed
    RES-140 contract, its digest recomputed from bytes, and every listed file
    matched against its pinned size and digest.
    """
    manifest_bytes, manifest = _load_manifest(root)
    source_id, split = _manifest_identity(manifest)
    _verify_inventory(root, manifest, source_id=source_id, split=split)
    dataset_sha256, task_sha256 = _verify_task_identity(
        root, manifest, source_id=source_id, split=split
    )
    return SliceReceipt(
        root=root,
        source_id=source_id,
        split=split,
        manifest_sha256=bytes_sha256(manifest_bytes),
        dataset_sha256=dataset_sha256,
        task_sha256=task_sha256,
    )
