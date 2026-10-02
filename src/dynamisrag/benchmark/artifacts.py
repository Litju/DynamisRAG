"""Deterministic benchmark artifacts, shard sidecars and run manifests.

Everything this benchmark writes is canonical JSON plus a SHA-256, and every
semantic artifact declares the revision that defines its own schema. Two rules
make the artifacts checkable rather than merely present:

**No semantic timestamps.** A run directory's creation time, a session id, a
hostname, a GPU serial and a wall-clock duration are all recorded — in a
non-semantic run manifest — and none of them is in a hashed payload. An artifact
whose digest changes when nothing semantic changed cannot be compared with itself,
and a resumed run that recomputed a plan digest would produce a different plan
SHA than the one a human reviewed.

**Canonical ordering is checked, not documented.** A shard sidecar carries the
complete ordered id list of its shard plus the digest of that list, so a verifier
holding nothing but the bundle can prove that shard 7 holds rows 28,672-32,767 of
the canonical corpus order. A sidecar that recorded only a row count would let
one shard file be swapped for another of the same shape and pass every check.

The write path is the same everywhere: build under local scratch, close the file,
hash the finished bytes, copy to the durable store, verify the copy's digest, and
only then call the shard complete. :func:`copy_verified` is that last three steps
in one function, because a benchmark that trusts a copy it has not re-hashed is
trusting a network filesystem.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, Self, cast

import numpy as np
from numpy.typing import NDArray

from dynamisrag.benchmark.contracts import (
    RES138_ARTIFACT_REVISIONS,
    RES138_CANDIDATE_DIMENSIONS,
    RES138_MODEL_CANDIDATES,
    RES138_RUN_ID_PREFIX,
    RES138_SHARD_SIZE,
    ModelCandidateSpec,
    RetrievalWorkload,
    ordered_ids_sha256,
    require_code_sha,
    require_exact_int,
    require_exact_str,
    require_shard_size,
)
from dynamisrag.benchmark.errors import BenchmarkArtifactError, BenchmarkContractError
from dynamisrag.benchmark.retrieval import RES138_SCORE_DTYPE, require_normalised_matrix
from dynamisrag.embedding.contracts import canonical_json

__all__ = [
    "READ_CHUNK_BYTES",
    "RES138_NORMALIZATION",
    "RES138_RUN_MANIFEST_REVISION",
    "ArtifactEnvelope",
    "Res138JsonValue",
    "Res138RunManifest",
    "ShardKind",
    "ShardSidecar",
    "copy_verified",
    "file_sha256",
    "read_artifact",
    "require_run_resumable",
    "write_artifact",
]

READ_CHUNK_BYTES: Final[int] = 1 << 20
"""Read granularity for file digests: 1 MiB.

Large enough that a 700 MB TREC-COVID matrix is a few hundred reads, small
enough that hashing stays a streaming operation rather than a second copy in
memory.
"""

RES138_NORMALIZATION: Final[str] = "l2-float32-v1"
"""The normalisation semantics recorded on every shard.

``l2`` unit rows in ``float32``. Named and versioned because a shard that stored
unnormalised float64 rows would be the same shape, the same row count and a
different set of vectors, and nothing else in the sidecar would say so.
"""

RES138_RUN_MANIFEST_REVISION: Final[str] = "res138-run-manifest-v1"
"""Revision of the per-run manifest written into a Drive run directory.

Separate from the nine semantic artifacts because it is the one document whose
whole job is to be *compared* — against another attempt to resume the same run —
and because it may carry non-semantic fields (a start time, a session label)
that a hashed payload must not.
"""

type Res138JsonValue = (
    str | int | float | bool | Sequence["Res138JsonValue"] | Mapping[str, "Res138JsonValue"] | None
)
"""The JSON value domain an artifact payload may hold.

The array and object members are the ``Sequence``/``Mapping`` protocols rather
than concrete ``list``/``dict``, so a freshly built payload — a comprehension
returning ``list[str]``, for instance — is still a ``Res138JsonValue`` without a
cast. That mirrors the JSON value domain
:mod:`dynamisrag.embedding.contracts` already declares, so a value crossing the
embedding boundary and a value crossing this one are checked the same way.
"""


def require_artifact_revision(name: str, *, operation: str) -> str:
    """The declared revision for one logical artifact name.

    ``name`` is a key of :data:`RES138_ARTIFACT_REVISIONS`, never a revision
    string typed by a caller: the map is the single place where "what schema does
    this document have" is answered, so a typo cannot invent a schema.
    """
    revision = RES138_ARTIFACT_REVISIONS.get(name)
    if revision is None:
        raise BenchmarkArtifactError(
            f"unknown benchmark artifact {name!r}. The declared artifacts are "
            f"{sorted(RES138_ARTIFACT_REVISIONS)}.",
            operation=operation,
            expected=str(sorted(RES138_ARTIFACT_REVISIONS)),
            observed=name,
        )
    return revision


def file_sha256(path: Path) -> str:
    """SHA-256 over a file's bytes, streamed.

    The only authority on what a stored file contains. A name, a size and a
    modification time are all forgeable and are never compared.
    """
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(READ_CHUNK_BYTES):
                digest.update(chunk)
    except OSError as error:
        raise BenchmarkArtifactError(
            f"could not read {path.name} to hash it ({type(error).__name__}). Every artifact is "
            "verified by its digest, so an unreadable artifact cannot be accepted as anything.",
            operation="file_sha256",
        ) from None
    return digest.hexdigest()


@dataclass(frozen=True)
class ArtifactEnvelope:
    """One canonical JSON artifact: a declared revision and a JSON object payload."""

    artifact_revision: str
    payload: Mapping[str, Res138JsonValue]

    @property
    def canonical_json(self) -> str:
        """The exact bytes that are hashed, as text.

        Sorted keys, compact separators, ``ensure_ascii=False`` — the same
        canonicalisation the embedding manifests use, so a digest taken over a
        benchmark artifact and one taken over an embedding manifest are computed
        the same way.
        """
        return canonical_json({"artifact_revision": self.artifact_revision, **self.payload})

    @property
    def artifact_bytes(self) -> bytes:
        """Canonical serialization, UTF-8 encoded."""
        return self.canonical_json.encode("utf-8")

    @property
    def sha256(self) -> str:
        """SHA-256 of :attr:`artifact_bytes`."""
        return hashlib.sha256(self.artifact_bytes).hexdigest()

    def write(self, path: Path) -> str:
        """Write the artifact atomically and return its digest.

        Written to a sibling ``.tmp`` and then replaced, so a process killed
        mid-write leaves either the previous artifact or none — never a
        half-written file that a later verifier would have to guess about.
        """
        temporary = path.with_name(f"{path.name}.tmp")
        temporary.write_bytes(self.artifact_bytes)
        temporary.replace(path)
        return self.sha256


def build_artifact(
    name: str, payload: Mapping[str, Res138JsonValue], *, operation: str
) -> ArtifactEnvelope:
    """Build one artifact from a payload, attaching its declared revision.

    The revision is looked up rather than passed in, so a payload cannot claim a
    schema it was not written for.
    """
    return ArtifactEnvelope(
        artifact_revision=require_artifact_revision(name, operation=operation), payload=payload
    )


def write_artifact(path: Path, *, name: str, payload: Mapping[str, Res138JsonValue]) -> str:
    """Build and write one artifact, returning its digest."""
    envelope = build_artifact(name, payload, operation="write_artifact")
    return envelope.write(path)


def read_artifact(path: Path, *, name: str) -> ArtifactEnvelope:
    """Read one artifact and require the revision it declares to be the expected one.

    The revision is checked on read as well as on write: a bundle assembled from
    two runs of different harness revisions must not verify, and the cheapest
    place to notice is the first field.
    """
    expected = require_artifact_revision(name, operation="read_artifact")
    decoded: object = _load_json_object(path, name="benchmark artifact", operation="read_artifact")
    revision = decoded.get("artifact_revision")
    if revision != expected:
        raise BenchmarkArtifactError(
            f"benchmark artifact {path.name} declares revision {revision!r}, which is not "
            f"{expected!r}. An artifact whose identity does not name its own schema cannot be "
            "compared with, or replaced by, another one.",
            operation="read_artifact",
            expected=expected,
            observed=str(revision),
        )
    payload: dict[str, Res138JsonValue] = {
        key: cast("Res138JsonValue", value)
        for key, value in decoded.items()
        if key != "artifact_revision"
    }
    return ArtifactEnvelope(artifact_revision=expected, payload=payload)


def _load_json_object(path: Path, *, name: str, operation: str) -> dict[str, object]:
    """Read a JSON object from disk, or refuse with a named reason.

    One loader for every JSON document the harness reads, because "the file is
    unreadable", "the file is not JSON" and "the file is not an object" are three
    different operator problems and a bare ``json.JSONDecodeError`` from one of
    them reports none of the context.
    """
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise BenchmarkArtifactError(
            f"{name} {path.name} could not be read ({type(error).__name__}).",
            operation=operation,
        ) from None
    except ValueError as error:
        raise BenchmarkArtifactError(
            f"{name} {path.name} is not valid JSON ({error}). A document that cannot be parsed "
            "cannot be compared with anything.",
            operation=operation,
        ) from None
    if not isinstance(decoded, dict):
        raise BenchmarkArtifactError(
            f"{name} {path.name} is a JSON {type(decoded).__name__}, not an object.",
            operation=operation,
        )
    return cast("dict[str, object]", decoded)


def copy_verified(source: Path, destination: Path) -> str:
    """Copy a finished artifact and verify the copy's digest, returning it.

    The last three steps of the shard workflow in one place: build and close
    locally, hash the local file, copy to durable storage, re-hash the copy. A
    mounted Drive is a network filesystem, and a copy that was never re-hashed is
    a claim rather than a fact.
    """
    expected = file_sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.partial")
    temporary.write_bytes(source.read_bytes())
    temporary.replace(destination)
    observed = file_sha256(destination)
    if observed != expected:
        raise BenchmarkArtifactError(
            f"the copy of {source.name} into durable storage hashed {observed}, not the "
            f"{expected} of the file that was written. The durable copy is not the artifact that "
            "was produced, so the shard is not complete.",
            operation="copy_verified",
            expected=expected,
            observed=observed,
        )
    return observed


class ShardKind(StrEnum):
    """Which side of a workload a shard holds.

    An enum rather than a bare ``str`` because the value decides the prompt
    applied to the rows, and a typo would produce a shard of documents embedded
    with the query prompt — a well-formed artifact holding the wrong vectors.
    """

    DOCUMENTS = "documents"
    QUERIES = "queries"

    @property
    def prompt_name(self) -> str:
        """The model-native prompt name for this side, as the pinned repos declare it.

        Deliberately **not** :attr:`value`: the artifact labels are plural
        (``documents``, ``queries``) because a directory holds a set, while the
        prompt keys in ``config_sentence_transformers.json`` are singular
        (``document``, ``query``). Passing the directory name straight to the
        prompt table would look up a key that does not exist, which is why the
        mapping is written down here instead of inferred.
        """
        return "document" if self is ShardKind.DOCUMENTS else "query"


@dataclass(frozen=True)
class ShardSidecar:
    """The metadata that makes one ``.npy`` shard identifiable.

    Binds the artifact revision, the model id **and** revision, the prompt pair's
    digest, the workload, the kind, the dimension, the dtype, the normalisation,
    the complete ordered id list of the shard with its digest, the matrix's own
    digest and size, and the code and runtime identities that produced it.

    The id list is here rather than in a separate document because it is what
    makes the shard verifiable on its own: concatenating the shard id lists in
    ordinal order must reproduce the canonical workload order, and that is
    checkable from the bundle alone, with no trust in a filename.
    """

    artifact_revision: str
    model_id: str
    model_revision: str
    prompt_sha256: str
    workload: str
    kind: ShardKind
    dimension: int
    dtype: str
    normalization: str
    shard_index: int
    shard_size: int
    first_id: str
    last_id: str
    row_count: int
    ordered_ids_sha256: str
    ids: tuple[str, ...]
    matrix_sha256: str
    matrix_byte_size: int
    code_sha: str
    runtime_sha256: str

    def payload(self) -> dict[str, Res138JsonValue]:
        """The hashed payload, including the id list."""
        return {
            "artifact_revision": self.artifact_revision,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "prompt_sha256": self.prompt_sha256,
            "workload": self.workload,
            "kind": self.kind.value,
            "dimension": self.dimension,
            "dtype": self.dtype,
            "normalization": self.normalization,
            "shard_index": self.shard_index,
            "shard_size": self.shard_size,
            "first_id": self.first_id,
            "last_id": self.last_id,
            "row_count": self.row_count,
            "ordered_ids_sha256": self.ordered_ids_sha256,
            "ids": list(self.ids),
            "matrix_sha256": self.matrix_sha256,
            "matrix_byte_size": self.matrix_byte_size,
            "code_sha": self.code_sha,
            "runtime_sha256": self.runtime_sha256,
        }

    @property
    def sha256(self) -> str:
        """SHA-256 over this sidecar's canonical payload."""
        return hashlib.sha256(canonical_json(self.payload()).encode("utf-8")).hexdigest()

    def write(self, path: Path) -> str:
        """Write the sidecar atomically and return its digest."""
        temporary = path.with_name(f"{path.name}.tmp")
        temporary.write_bytes(canonical_json(self.payload()).encode("utf-8"))
        temporary.replace(path)
        return self.sha256

    def __post_init__(self) -> None:
        expected = RES138_ARTIFACT_REVISIONS["shard"]
        if self.artifact_revision != expected:
            raise BenchmarkArtifactError(
                f"a shard sidecar declares revision {self.artifact_revision!r}, which is not "
                f"{expected!r}.",
                operation="shard_sidecar",
            )
        require_shard_size(self.shard_size, operation="shard_sidecar")
        if not self.ids:
            raise BenchmarkArtifactError(
                f"shard {self.shard_index} of {self.workload}/{self.kind.value} carries no ids. A "
                "shard with no ids has no first id, no last id and no rows to bind a matrix to.",
                operation="shard_sidecar",
                workload=self.workload,
            )
        if self.row_count != len(self.ids):
            raise BenchmarkArtifactError(
                f"shard {self.shard_index} of {self.workload}/{self.kind.value} declares "
                f"{self.row_count} rows but carries {len(self.ids)} ids. The sidecar's whole job "
                "is to say which rows the matrix holds.",
                operation="shard_sidecar",
                workload=self.workload,
                expected=str(self.row_count),
                observed=str(len(self.ids)),
            )
        if ordered_ids_sha256(self.ids) != self.ordered_ids_sha256:
            raise BenchmarkArtifactError(
                f"shard {self.shard_index} of {self.workload}/{self.kind.value} carries id lists "
                "whose digest is not the one it declares.",
                operation="shard_sidecar",
                workload=self.workload,
                expected=self.ordered_ids_sha256,
                observed=ordered_ids_sha256(self.ids),
            )
        for position in range(1, len(self.ids)):
            if not self.ids[position - 1] < self.ids[position]:
                raise BenchmarkArtifactError(
                    f"shard {self.shard_index} of {self.workload}/{self.kind.value} holds "
                    f"{self.ids[position]!r} at row {position} after {self.ids[position - 1]!r}. A "
                    "shard's own rows must be in canonical order, because the shard boundary is a "
                    "slice of that order.",
                    operation="shard_sidecar",
                    workload=self.workload,
                    item_id=self.ids[position],
                )
        if self.ids[0] != self.first_id or self.ids[-1] != self.last_id:
            raise BenchmarkArtifactError(
                f"shard {self.shard_index} of {self.workload}/{self.kind.value} declares "
                f"first/last ids {(self.first_id, self.last_id)} which are not the ends of its own "
                "id list.",
                operation="shard_sidecar",
                workload=self.workload,
            )
        if self.dtype != RES138_SCORE_DTYPE.__name__:
            raise BenchmarkArtifactError(
                f"shard sidecar declares dtype {self.dtype!r}; benchmark matrices are "
                f"{RES138_SCORE_DTYPE.__name__}.",
                operation="shard_sidecar",
                workload=self.workload,
            )
        if self.normalization != RES138_NORMALIZATION:
            raise BenchmarkArtifactError(
                f"shard sidecar declares normalization {self.normalization!r}; the frozen "
                f"semantics are {RES138_NORMALIZATION!r}.",
                operation="shard_sidecar",
                workload=self.workload,
            )


def build_shard_sidecar(
    *,
    candidate: ModelCandidateSpec,
    workload: RetrievalWorkload,
    kind: ShardKind,
    dimension: int,
    ids: Sequence[str],
    matrix_path: Path,
    shard_index: int,
    code_sha: str,
    runtime_sha256: str,
    operation: str,
) -> ShardSidecar:
    """Describe one finished shard from its matrix and its id range.

    Every binding value is taken from the frozen contract or observed from the
    file: the model identity from the candidate spec, the prompt digest from the
    candidate's own model-native prompt for this ``kind``, the dimension from the
    run's declared semantics, and the matrix digest and byte size from the
    finished file. Nothing here is inferred from the shard's position in a
    directory, and the prompt is derived rather than passed in so a caller cannot
    describe a documents shard with the query prompt.
    """
    prompt = candidate.prompt(kind=kind.prompt_name)
    if dimension not in RES138_CANDIDATE_DIMENSIONS:
        raise BenchmarkContractError(
            f"shard dimension {dimension!r} is not one of the frozen "
            f"{list(RES138_CANDIDATE_DIMENSIONS)}.",
            operation=operation,
            workload=workload.name,
        )
    if not ids:
        raise BenchmarkContractError(
            "a shard must hold at least one row. An empty shard would have no first id, no last id "
            "and no vectors.",
            operation=operation,
            workload=workload.name,
        )
    return ShardSidecar(
        artifact_revision=RES138_ARTIFACT_REVISIONS["shard"],
        model_id=candidate.model_id,
        model_revision=candidate.revision,
        prompt_sha256=prompt.content_sha256,
        workload=workload.name,
        kind=kind,
        dimension=dimension,
        dtype=RES138_SCORE_DTYPE.__name__,
        normalization=RES138_NORMALIZATION,
        shard_index=shard_index,
        shard_size=RES138_SHARD_SIZE,
        first_id=ids[0],
        last_id=ids[-1],
        row_count=len(ids),
        ordered_ids_sha256=ordered_ids_sha256(ids),
        ids=tuple(ids),
        matrix_sha256=file_sha256(matrix_path),
        matrix_byte_size=matrix_path.stat().st_size,
        code_sha=require_code_sha(code_sha, operation=operation),
        runtime_sha256=runtime_sha256,
    )


def verify_shard_matrix(matrix_path: Path, sidecar: ShardSidecar) -> NDArray[np.float32]:
    """Verify one shard's bytes against its sidecar and return the matrix.

    Digest first, then shape, dtype, finiteness and normalisation. The order
    matters only for cost: hashing a 700 MB file is cheaper than loading it, so a
    corrupt shard is rejected before it is parsed.
    """
    observed_digest = file_sha256(matrix_path)
    if observed_digest != sidecar.matrix_sha256:
        raise BenchmarkArtifactError(
            f"shard {sidecar.shard_index} of {sidecar.workload}/{sidecar.kind} hashed "
            f"{observed_digest}, not the {sidecar.matrix_sha256} its sidecar declares.",
            operation="verify_shard_matrix",
            workload=sidecar.workload,
            expected=sidecar.matrix_sha256,
            observed=observed_digest,
        )
    if matrix_path.stat().st_size != sidecar.matrix_byte_size:
        raise BenchmarkArtifactError(
            f"shard {sidecar.shard_index} of {sidecar.workload}/{sidecar.kind} is "
            f"{matrix_path.stat().st_size} bytes, not the {sidecar.matrix_byte_size} its sidecar "
            "declares.",
            operation="verify_shard_matrix",
            workload=sidecar.workload,
        )
    matrix = np.load(matrix_path, allow_pickle=False)
    if matrix.dtype != np.float32:
        raise BenchmarkArtifactError(
            f"shard matrix has dtype {matrix.dtype}, not float32.",
            operation="verify_shard_matrix",
            workload=sidecar.workload,
        )
    expected_shape = (sidecar.row_count, sidecar.dimension)
    if matrix.shape != expected_shape:
        raise BenchmarkArtifactError(
            f"shard matrix has shape {matrix.shape}, not the {expected_shape} its sidecar "
            "declares.",
            operation="verify_shard_matrix",
            workload=sidecar.workload,
            expected=str(expected_shape),
            observed=str(matrix.shape),
        )
    if not bool(np.all(np.isfinite(matrix))):
        raise BenchmarkArtifactError(
            "shard matrix holds a non-finite component. Every distance to a non-finite vector is "
            "undefined, which would corrupt the whole corpus ranking rather than one row.",
            operation="verify_shard_matrix",
            workload=sidecar.workload,
        )
    try:
        require_normalised_matrix(matrix, name="shard matrix")
    except BenchmarkContractError as error:
        # Re-raised as an artifact failure: inside bundle verification every
        # refusal means the same thing — this shard cannot be used — and a caller
        # catching one benchmark error type should not have to know which half of
        # the package raised it.
        raise BenchmarkArtifactError(
            str(error), operation="verify_shard_matrix", workload=sidecar.workload
        ) from None
    return matrix


@dataclass(frozen=True)
class Res138RunManifest:
    """The identity of one Drive run directory, and the condition for resuming it.

    A run may be resumed only when this manifest matches the one already in the
    directory **exactly** — code commit, runtime fingerprint, model revisions,
    dataset digests, generation semantics, shard revision and shard size. Any
    difference means the resumed shards would be produced under conditions that
    do not match the shards already in the folder, and a partially-resumed run
    under two identities is the one outcome worse than starting again.

    Non-semantic facts — a start time, a session label, the Colab runtime
    version — live beside this manifest and are deliberately not in its identity
    payload, so re-running the same code on the same runtime produces the same
    identity.
    """

    manifest_revision: str
    run_id: str
    code_sha: str
    runtime_sha256: str
    plan_sha256: str
    generation_semantics_sha256: str
    model_revisions: tuple[tuple[str, str], ...]
    dataset_digests: tuple[tuple[str, str], ...]
    shard_revision: str
    shard_size: int

    def __post_init__(self) -> None:
        if self.manifest_revision != RES138_RUN_MANIFEST_REVISION:
            raise BenchmarkArtifactError(
                f"a run manifest declares revision {self.manifest_revision!r}, which is not "
                f"{RES138_RUN_MANIFEST_REVISION!r}.",
                operation="res138_run_manifest",
            )
        require_code_sha(self.code_sha, operation="res138_run_manifest")
        require_shard_size(self.shard_size, operation="res138_run_manifest")
        if not self.run_id.startswith(f"{RES138_RUN_ID_PREFIX}-"):
            raise BenchmarkArtifactError(
                f"run id {self.run_id!r} does not start with {RES138_RUN_ID_PREFIX!r}, so a Drive "
                "run folder could not be recognised as one.",
                operation="res138_run_manifest",
                observed=self.run_id,
            )
        for name, digest in (
            ("runtime fingerprint", self.runtime_sha256),
            ("plan", self.plan_sha256),
            ("generation semantics", self.generation_semantics_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise BenchmarkArtifactError(
                    f"run manifest {name} digest {digest!r} is not 64 lowercase hexadecimal "
                    "characters. A run identity made of digests has to be made of digests.",
                    operation="res138_run_manifest",
                )
        if not self.model_revisions or not self.dataset_digests:
            raise BenchmarkArtifactError(
                "a run manifest must bind the candidate model revisions and the dataset digests it "
                "was produced under. A run whose identity omits either cannot be compared with "
                "another run at all.",
                operation="res138_run_manifest",
            )
        self._require_unique_pairs("model_revisions", self.model_revisions)
        self._require_unique_pairs("dataset_digests", self.dataset_digests)

    @staticmethod
    def _require_unique_pairs(name: str, pairs: Sequence[tuple[str, str]]) -> None:
        keys = [key for key, _ in pairs]
        if len(set(keys)) != len(keys):
            raise BenchmarkArtifactError(
                f"run manifest {name} repeats a key. Two entries for one key are two claims about "
                "the same thing, and which one would win is an accident of iteration order.",
                operation="res138_run_manifest",
            )

    def identity_payload(self) -> dict[str, Res138JsonValue]:
        """The hashed description of this run identity, with no ``run_id`` in it.

        ``run_id`` is excluded because it is a function of the code commit and the
        runtime fingerprint: including it would add a display string to the digest
        without adding information.
        """
        return {
            "manifest_revision": self.manifest_revision,
            "code_sha": self.code_sha,
            "runtime_sha256": self.runtime_sha256,
            "plan_sha256": self.plan_sha256,
            "generation_semantics_sha256": self.generation_semantics_sha256,
            "model_revisions": [[key, value] for key, value in self.model_revisions],
            "dataset_digests": [[key, value] for key, value in self.dataset_digests],
            "shard_revision": self.shard_revision,
            "shard_size": self.shard_size,
        }

    @property
    def identity_sha256(self) -> str:
        """SHA-256 over the identity payload."""
        return hashlib.sha256(canonical_json(self.identity_payload()).encode("utf-8")).hexdigest()

    def to_payload(self) -> dict[str, Res138JsonValue]:
        """The full on-disk payload: the identity plus its ``run_id``."""
        return {
            **self.identity_payload(),
            "run_id": self.run_id,
            "identity_sha256": self.identity_sha256,
        }

    def write(self, path: Path) -> str:
        """Write the manifest atomically and return the identity digest."""
        temporary = path.with_name(f"{path.name}.tmp")
        temporary.write_bytes(canonical_json(self.to_payload()).encode("utf-8"))
        temporary.replace(path)
        return self.identity_sha256

    @classmethod
    def read(cls, path: Path) -> Self:
        """Read a manifest from a run directory, re-verifying its own digest.

                The digest is recomputed rather than trusted: a manifest whose recorded
                ``identity_sha256`` disagrees with its own contents describes a run that
        never existed.
        """
        decoded = _load_json_object(path, name="run manifest", operation="read_run_manifest")
        fields = {
            "manifest_revision": require_exact_str(
                decoded.get("manifest_revision"), kind="run manifest revision", operation="read"
            ),
            "run_id": require_exact_str(decoded.get("run_id"), kind="run id", operation="read"),
            "code_sha": require_exact_str(
                decoded.get("code_sha"), kind="code sha", operation="read"
            ),
            "runtime_sha256": require_exact_str(
                decoded.get("runtime_sha256"), kind="runtime digest", operation="read"
            ),
            "plan_sha256": require_exact_str(
                decoded.get("plan_sha256"), kind="plan digest", operation="read"
            ),
            "generation_semantics_sha256": require_exact_str(
                decoded.get("generation_semantics_sha256"),
                kind="generation semantics digest",
                operation="read",
            ),
            "shard_revision": require_exact_str(
                decoded.get("shard_revision"), kind="shard revision", operation="read"
            ),
            "shard_size": require_exact_int(
                decoded.get("shard_size"),
                kind="shard size",
                operation="read",
                minimum=1,
                because="A run that recorded no shard size could not have produced its shards.",
            ),
            "model_revisions": _decode_pairs(decoded.get("model_revisions"), "model_revisions"),
            "dataset_digests": _decode_pairs(decoded.get("dataset_digests"), "dataset_digests"),
        }
        manifest = cls(**fields)  # pyright: ignore[reportArgumentType]
        recorded = decoded.get("identity_sha256")
        if recorded != manifest.identity_sha256:
            raise BenchmarkArtifactError(
                f"run manifest {path.name} records identity {recorded!r} but its own contents hash "
                f"to {manifest.identity_sha256!r}.",
                operation="read_run_manifest",
                expected=manifest.identity_sha256,
                observed=str(recorded),
            )
        return manifest


def _decode_pairs(value: object, name: str) -> tuple[tuple[str, str], ...]:
    """Decode a ``[[key, value], ...]`` pair list, refusing anything else."""
    if not isinstance(value, list):
        raise BenchmarkArtifactError(
            f"run manifest field {name} is a {type(value).__name__}, not a list of pairs.",
            operation="read_run_manifest",
        )
    pairs: list[tuple[str, str]] = []
    for entry in cast("list[object]", value):
        if (
            not isinstance(entry, list)
            or len(cast("list[object]", entry)) != 2
            or not all(isinstance(item, str) for item in cast("list[object]", entry))
        ):
            raise BenchmarkArtifactError(
                f"run manifest field {name} holds {entry!r}, which is not a [key, value] string "
                "pair.",
                operation="read_run_manifest",
            )
        pair = cast("list[str]", entry)
        pairs.append((pair[0], pair[1]))
    return tuple(pairs)


def require_run_resumable(existing: Res138RunManifest, current: Res138RunManifest) -> None:
    """Refuse to continue a run whose recorded identity differs, field by field.

    Reported one field at a time rather than as two digests, because the useful
    answer to an operator is *which* condition changed — a new GPU session, a
    re-cloned commit, a re-pinned model — and a digest comparison would make them
    work that out by hand.
    """
    checks: tuple[tuple[str, str, str], ...] = (
        ("code commit", existing.code_sha, current.code_sha),
        ("runtime fingerprint", existing.runtime_sha256, current.runtime_sha256),
        ("benchmark plan", existing.plan_sha256, current.plan_sha256),
        (
            "generation semantics",
            existing.generation_semantics_sha256,
            current.generation_semantics_sha256,
        ),
        ("shard revision", existing.shard_revision, current.shard_revision),
        ("shard size", str(existing.shard_size), str(current.shard_size)),
        (
            "candidate model revisions",
            canonical_json([list(pair) for pair in existing.model_revisions]),
            canonical_json([list(pair) for pair in current.model_revisions]),
        ),
        (
            "dataset digests",
            canonical_json([list(pair) for pair in existing.dataset_digests]),
            canonical_json([list(pair) for pair in current.dataset_digests]),
        ),
    )
    for label, recorded, observed in checks:
        if recorded == observed:
            continue
        raise BenchmarkArtifactError(
            f"this run directory was created with {label} {recorded!r} and the current run has "
            f"{observed!r}. Quality shards may be resumed only when code, model, dataset, "
            "generation and runtime identity match exactly; anything less would mix vectors "
            "produced under two identities in one folder. Start a new run instead.",
            operation="require_run_resumable",
            expected=recorded[:64],
            observed=observed[:64],
        )


def build_run_manifest(
    *,
    run_id: str,
    code_sha: str,
    runtime_sha256: str,
    plan_sha256: str,
    generation_semantics_sha256: str,
    dataset_digests: Sequence[tuple[str, str]],
) -> Res138RunManifest:
    """Build a run manifest from the frozen candidates and the verified datasets."""
    return Res138RunManifest(
        manifest_revision=RES138_RUN_MANIFEST_REVISION,
        run_id=run_id,
        code_sha=code_sha,
        runtime_sha256=runtime_sha256,
        plan_sha256=plan_sha256,
        generation_semantics_sha256=generation_semantics_sha256,
        model_revisions=tuple(
            (candidate.model_id, candidate.revision) for candidate in RES138_MODEL_CANDIDATES
        ),
        dataset_digests=tuple(sorted(dataset_digests)),
        shard_revision=RES138_ARTIFACT_REVISIONS["shard"],
        shard_size=RES138_SHARD_SIZE,
    )


def resolve_run_directory(runs_root: Path, run_id: str) -> Path:
    """Where a run directory lives, refusing a ``run_id`` that is not one.

    Paths are joined under a root the caller supplies — the mounted Drive on
    Colab, a temporary directory in a test — and the run id is validated rather
    than escaped, because a run id reaches this function from a notebook cell and
    a ``../`` in it would write outside the runs folder.
    """
    require_exact_str(run_id, kind="run id", operation="resolve_run_directory")
    if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in run_id):
        raise BenchmarkArtifactError(
            f"run id {run_id!r} holds a character outside lowercase ASCII letters, digits and '-'. "
            "A run id becomes a directory name, so it is constrained rather than escaped.",
            operation="resolve_run_directory",
            observed=run_id,
        )
    return runs_root / run_id


def open_drive_run(runs_root: Path, manifest: Res138RunManifest, *, operation: str) -> Path:
    """Create or re-open a Drive run directory, refusing an incompatible one.

    Absent: create it and write the manifest. Present and identical: resume it.
    Present and different: refuse, naming the differing field, rather than
    appending to a folder that already holds shards from another identity.

    There is deliberately no "best effort" mode. A partially-resumed run under two
    identities is worse than no resume at all, because it looks complete.
    """
    directory = resolve_run_directory(runs_root, manifest.run_id)
    manifest_path = directory / "run-manifest.json"
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=False)
        manifest.write(manifest_path)
        return directory
    if not directory.is_dir():
        raise BenchmarkArtifactError(
            f"{directory} exists and is not a directory.",
            operation=operation,
        )
    if not manifest_path.exists():
        raise BenchmarkArtifactError(
            f"run directory {directory} exists but holds no run manifest. A run directory without "
            "its identity cannot be verified or resumed, and adopting it would adopt shards whose "
            "identity nobody recorded.",
            operation=operation,
        )
    require_run_resumable(Res138RunManifest.read(manifest_path), manifest)
    return directory


def shard_paths(directory: Path, *, workload: str, kind: ShardKind, dimension: int) -> Path:
    """The directory holding one (workload, kind, dimension) shard set."""
    return directory / workload / kind.value / str(dimension)
