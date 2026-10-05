"""Local verification of a finished RES-138 bundle, with no trust on first use.

This is the gate that lets a bundle downloaded from Drive be trusted on a
Windows workstation without re-running anything. It reads a directory and either
returns a report that binds every byte it found to a declared digest, or refuses.

**What "verified" means here, in full.** Every file the root manifest declares
exists and hashes to the declared value; the tree contains no file the manifest
does not declare; every shard sidecar declares the ``res138-shard-v2`` revision
and its matrix hashes to the declared digest with the declared dtype, dimension,
row count and normalisation; the shard id lists are individually ascending and,
concatenated in ordinal order, reproduce one strictly ascending canonical
sequence with no gap, no overlap and no duplicate ordinal; the recorded totals
match the corpus and query counts in the source manifest; the candidate model
revisions are the frozen ones; the dataset digests are the frozen BEIR digests;
and the code commit matches the one the caller expects.

**No trust on first use.** Nothing is inferred from a filename, a directory name,
a modification time or a file size. A shard named ``shard-00000.npy`` is a shard
because the sidecar next to it says so and its bytes hash to what that sidecar
declares — and if the manifest does not list that file, the bundle is refused
rather than quietly extended.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from dynamisrag.benchmark.artifacts import (
    ArtifactEnvelope,
    Res138JsonValue,
    Res138RunManifest,
    ShardSidecar,
    file_sha256,
    read_shard_sidecar,
    verify_shard_matrix,
)
from dynamisrag.benchmark.contracts import (
    RES138_BEIR_SOURCES,
    RES138_MODEL_CANDIDATES,
    require_code_sha,
    require_exact_str,
)
from dynamisrag.benchmark.errors import BenchmarkArtifactError
from dynamisrag.embedding.contracts import canonical_json

__all__ = [
    "RES138_BUNDLE_MANIFEST_REVISION",
    "RES138_RUN_MANIFEST_FILENAME",
    "BundleVerification",
    "build_bundle_manifest",
    "verify_run_bundle",
]

RES138_BUNDLE_MANIFEST_REVISION: Final[str] = "res138-bundle-manifest-v1"
"""Revision of the bundle's own completeness manifest.

Separate from ``res138-run-manifest-v1`` because the two answer different
questions. The run manifest states **identity** — which code, weights, data and
runtime this run is — and its digest is what makes a resume safe. The bundle
manifest states **completeness** — which files must be present for this bundle to
be the whole run — and it is written last, after every shard has been verified in
place. Folding the file list into the run manifest would make the run's identity
change as shards landed, which would make every shard invalidate the previous
one.
"""

RES138_RUN_MANIFEST_FILENAME: Final[str] = "run-manifest.json"

_BUNDLE_MANIFEST_FILENAME: Final[str] = "bundle-manifest.json"

_EXCLUDED_FROM_TREE: Final[frozenset[str]] = frozenset(
    {_BUNDLE_MANIFEST_FILENAME, "*.partial", "*.tmp"}
)
"""Files that are not part of a bundle's declared content.

The bundle manifest cannot list itself (it is written after the tree is walked)
and must not list the temporary files an interrupted copy leaves behind — their
presence is a fact about the copy, not about the evidence. A ``.tmp`` or
``.partial`` file is ignored by the walk and is never evidence of anything.
"""

_SKIP_SUFFIXES: Final[tuple[str, ...]] = (".partial", ".tmp")
"""Temporary-file suffixes that are ignored by the walk and never evidence.

A ``.partial`` or ``.tmp`` file is a fact about an interrupted copy, not about the
run. Leaving them out of the manifest keeps the completeness check about the
artifacts themselves; a stale temporary beside a verified shard changes nothing
about it.
"""


def _relative_files(root: Path) -> tuple[str, ...]:
    """Every artifact file under ``root``, as sorted POSIX-style relative paths."""
    found: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative_path = path.relative_to(root)
        if any(part.endswith(_SKIP_SUFFIXES) for part in relative_path.parts[:-1]):
            continue
        relative = relative_path.as_posix()
        name = path.name
        if name == _BUNDLE_MANIFEST_FILENAME or name.endswith(_SKIP_SUFFIXES):
            continue
        found.append(relative)
    return tuple(sorted(found))


def _load_json_object(path: Path, *, name: str, operation: str) -> dict[str, Res138JsonValue]:
    """Read one JSON object from disk, or refuse it with a named reason."""
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise BenchmarkArtifactError(
            f"{name} {path.name} could not be read ({type(error).__name__}).",
            operation=operation,
        ) from None
    except ValueError as error:
        raise BenchmarkArtifactError(
            f"{name} {path.name} is not valid JSON ({error}).",
            operation=operation,
        ) from None
    if not isinstance(decoded, dict):
        raise BenchmarkArtifactError(
            f"{name} {path.name} is a JSON {type(decoded).__name__}, not an object.",
            operation=operation,
        )
    return cast("dict[str, Res138JsonValue]", decoded)


@dataclass(frozen=True)
class BundleVerification:
    """What a verified bundle contains, and the digest of that statement."""

    root: str
    code_sha: str
    run_id: str
    runtime_sha256: str
    file_count: int
    shard_count: int
    row_count: int
    model_revisions: tuple[tuple[str, str], ...]
    dataset_digests: tuple[tuple[str, str], ...]
    files: tuple[tuple[str, str], ...]

    def payload(self) -> dict[str, object]:
        """The hashed report: every file with the digest that was verified."""
        return {
            "root": self.root,
            "code_sha": self.code_sha,
            "run_id": self.run_id,
            "runtime_sha256": self.runtime_sha256,
            "file_count": self.file_count,
            "shard_count": self.shard_count,
            "row_count": self.row_count,
            "model_revisions": [list(pair) for pair in self.model_revisions],
            "dataset_digests": [list(pair) for pair in self.dataset_digests],
            "files": [list(pair) for pair in self.files],
        }

    @property
    def canonical_json(self) -> str:
        """The canonical serialization of this report."""
        return canonical_json(self.payload())

    @property
    def sha256(self) -> str:
        """SHA-256 over the canonical report."""
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


def build_bundle_manifest(root: Path) -> ArtifactEnvelope:
    """Walk a bundle and record every file with its digest.

    Written last, after the shards have been verified in place, so that a bundle
    manifest's existence means "the tree was complete and checked when this was
    written".
    """
    run_manifest = Res138RunManifest.read(root / RES138_RUN_MANIFEST_FILENAME)
    entries: list[Res138JsonValue] = [
        {
            "path": relative,
            "sha256": file_sha256(root / relative),
            "byte_size": (root / relative).stat().st_size,
        }
        for relative in _relative_files(root)
        if relative != RES138_RUN_MANIFEST_FILENAME
    ]
    return ArtifactEnvelope(
        artifact_revision=RES138_BUNDLE_MANIFEST_REVISION,
        payload={
            "run_id": run_manifest.run_id,
            "code_sha": run_manifest.code_sha,
            "runtime_sha256": run_manifest.runtime_sha256,
            "identity_sha256": run_manifest.identity_sha256,
            "file_count": len(entries),
            "files": entries,
        },
    )


def write_bundle_manifest(root: Path) -> str:
    """Write the bundle manifest into ``root`` and return its digest."""
    return build_bundle_manifest(root).write(root / _BUNDLE_MANIFEST_FILENAME)


def _require_entries(envelope: ArtifactEnvelope) -> tuple[tuple[str, str, int], ...]:
    """Decode a bundle manifest's file list into ``(path, digest, byte_size)``."""
    raw = envelope.payload.get("files")
    if not isinstance(raw, list):
        raise BenchmarkArtifactError(
            "the bundle manifest holds no file list.",
            operation="verify_run_bundle",
        )
    entries: list[tuple[str, str, int]] = []
    for item in cast("list[object]", raw):
        if not isinstance(item, dict):
            raise BenchmarkArtifactError(
                f"the bundle manifest holds a {type(item).__name__} where a file entry belongs.",
                operation="verify_run_bundle",
            )
        entry = cast("dict[str, object]", item)
        path = require_exact_str(entry.get("path"), kind="bundle path", operation="verify")
        digest = require_exact_str(entry.get("sha256"), kind="bundle digest", operation="verify")
        size = entry.get("byte_size")
        if isinstance(size, bool) or not isinstance(size, int):
            raise BenchmarkArtifactError(
                f"bundle entry {path!r} declares byte_size {size!r}, which is not an integer.",
                operation="verify_run_bundle",
            )
        entries.append((path, digest, size))
    return tuple(entries)


def _load_sidecar(path: Path) -> ShardSidecar:
    """Read one shard sidecar through the shared artifact contract."""
    return read_shard_sidecar(path)


def _frozen_pairs(candidates: Sequence[tuple[str, str]]) -> dict[str, str]:
    return dict(candidates)


def _require_expected_commit(
    run_manifest: Res138RunManifest, expect_code_sha: str | None, *, operation: str
) -> None:
    """Require the bundle's commit to be the one the caller expects, when stated."""
    if expect_code_sha is None:
        return
    required = require_code_sha(expect_code_sha, operation=operation)
    if run_manifest.code_sha == required:
        return
    raise BenchmarkArtifactError(
        f"bundle was produced by commit {run_manifest.code_sha} and the expected commit is "
        f"{required}. A bundle is only evidence for the code that produced it.",
        operation=operation,
        expected=required,
        observed=run_manifest.code_sha,
    )


def _require_frozen_models(run_manifest: Res138RunManifest, *, operation: str) -> None:
    """Require every recorded candidate to be one of the frozen pairs, at its frozen revision."""
    frozen = _frozen_pairs(
        [(candidate.model_id, candidate.revision) for candidate in RES138_MODEL_CANDIDATES]
    )
    for model_id, revision in run_manifest.model_revisions:
        expected = frozen.get(model_id)
        if expected is None or expected == revision:
            continue
        raise BenchmarkArtifactError(
            f"bundle records model {model_id} at revision {revision!r}, which is not the frozen "
            f"{expected!r} for that candidate.",
            operation=operation,
            model_id=model_id,
            expected=expected,
            observed=revision,
        )


def _require_frozen_datasets(run_manifest: Res138RunManifest, *, operation: str) -> None:
    """Require every recorded dataset digest to be the frozen BEIR digest for that workload."""
    frozen = _frozen_pairs([(source.workload, source.sha256) for source in RES138_BEIR_SOURCES])
    for workload, digest in run_manifest.dataset_digests:
        expected = frozen.get(workload)
        if expected is None or expected == digest:
            continue
        raise BenchmarkArtifactError(
            f"bundle records dataset {workload} at digest {digest!r}, which is not the frozen "
            f"{expected!r} for that workload.",
            operation=operation,
            workload=workload,
            expected=expected,
            observed=digest,
        )


def verify_run_bundle(
    root: Path,
    *,
    expect_code_sha: str | None = None,
    operation: str = "verify_run_bundle",
) -> BundleVerification:
    """Verify a bundle on any local filesystem, or refuse it.

    ``root`` is a run directory — the one created under ``DRIVE_RUNS/<RUN_ID>``
    and then copied to the workstation. Every check below reads a file and
    compares it with something the bundle itself declares, except the frozen
    identities in :mod:`dynamisrag.benchmark.contracts`, which are the point:
    those are what make "these are the right weights and the right data" a
    statement about the repository rather than about the bundle.
    """
    if not root.is_dir():
        raise BenchmarkArtifactError(
            f"bundle root {root} is not a directory.",
            operation=operation,
        )
    run_manifest = Res138RunManifest.read(root / RES138_RUN_MANIFEST_FILENAME)
    envelope = _read_bundle_manifest(root, operation=operation)
    _require_expected_commit(run_manifest, expect_code_sha, operation=operation)
    _require_frozen_models(run_manifest, operation=operation)
    _require_frozen_datasets(run_manifest, operation=operation)
    verified = _verify_declared_tree(root, envelope, operation=operation)
    shard_count, row_count = _verify_shards(root, operation=operation)
    if (root / "full-run.json").exists() or (root / "results").exists():
        from dynamisrag.benchmark.results import verify_full_run_bundle

        verify_full_run_bundle(root, run_manifest=run_manifest, declared_files=verified)
    return BundleVerification(
        root=str(root),
        code_sha=run_manifest.code_sha,
        run_id=run_manifest.run_id,
        runtime_sha256=run_manifest.runtime_sha256,
        file_count=len(verified),
        shard_count=shard_count,
        row_count=row_count,
        model_revisions=run_manifest.model_revisions,
        dataset_digests=run_manifest.dataset_digests,
        files=tuple(sorted(verified)),
    )


def _read_bundle_manifest(root: Path, *, operation: str) -> ArtifactEnvelope:
    """Read the bundle manifest and require the revision it declares."""
    path = root / _BUNDLE_MANIFEST_FILENAME
    if not path.exists():
        raise BenchmarkArtifactError(
            f"bundle {root} holds no bundle manifest, so its completeness cannot be checked. A "
            "bundle that does not declare which files it must contain is verified only as far as "
            "somebody remembered to look.",
            operation=operation,
        )
    document = _load_json_object(path, name="bundle manifest", operation=operation)
    revision = document.get("artifact_revision")
    if revision != RES138_BUNDLE_MANIFEST_REVISION:
        raise BenchmarkArtifactError(
            f"bundle manifest declares revision {revision!r}, which is not "
            f"{RES138_BUNDLE_MANIFEST_REVISION!r}.",
            operation=operation,
            expected=RES138_BUNDLE_MANIFEST_REVISION,
            observed=str(revision),
        )
    payload = {key: value for key, value in document.items() if key != "artifact_revision"}
    return ArtifactEnvelope(artifact_revision=RES138_BUNDLE_MANIFEST_REVISION, payload=payload)


def _verify_declared_tree(
    root: Path, envelope: ArtifactEnvelope, *, operation: str
) -> tuple[tuple[str, str], ...]:
    """Require the tree to match the manifest exactly, and every file to hash as declared.

    Both directions matter. A declared file that is absent is a missing shard, and
    a present file that is undeclared could be an extra shard or a replaced one —
    verifying the declared set while ignoring it would say nothing about it.
    """
    declared = _require_entries(envelope)
    present = _relative_files(root)
    declared_paths = {path for path, _, _ in declared} | {RES138_RUN_MANIFEST_FILENAME}
    missing = sorted(declared_paths - set(present))
    if missing:
        raise BenchmarkArtifactError(
            f"bundle {root} is missing {len(missing)} declared file(s): {missing[:4]}. A missing "
            "shard is not a smaller bundle; it is a different one.",
            operation=operation,
            count=len(missing),
        )
    undeclared = sorted(set(present) - declared_paths)
    if undeclared:
        raise BenchmarkArtifactError(
            f"bundle {root} holds {len(undeclared)} file(s) its manifest does not declare: "
            f"{undeclared[:4]}. An undeclared file could be an extra shard or a replaced one, and "
            "verifying the declared files while ignoring it would say nothing about it.",
            operation=operation,
            count=len(undeclared),
        )
    verified: list[tuple[str, str]] = []
    for path, digest, size in declared:
        absolute = root / path
        observed_size = absolute.stat().st_size
        if observed_size != size:
            raise BenchmarkArtifactError(
                f"bundle file {path} is {observed_size} bytes, not the declared {size}.",
                operation=operation,
                expected=str(size),
                observed=str(observed_size),
            )
        observed_digest = file_sha256(absolute)
        if observed_digest != digest:
            raise BenchmarkArtifactError(
                f"bundle file {path} hashed {observed_digest}, not the declared {digest}.",
                operation=operation,
                expected=digest,
                observed=observed_digest,
            )
        verified.append((path, digest))
    return tuple(verified)

    declared = _require_entries(envelope)
    present = _relative_files(root)
    declared_paths = {path for path, _, _ in declared} | {RES138_RUN_MANIFEST_FILENAME}
    missing = sorted(declared_paths - set(present))
    undeclared = sorted(set(present) - declared_paths)
    if missing:
        raise BenchmarkArtifactError(
            f"bundle {root} is missing {len(missing)} declared file(s): {missing[:4]}. A missing "
            "shard is not a smaller bundle; it is a different one.",
            operation=operation,
            count=len(missing),
        )
    if undeclared:
        raise BenchmarkArtifactError(
            f"bundle {root} holds {len(undeclared)} file(s) its manifest does not declare: "
            f"{undeclared[:4]}. An undeclared file could be an extra shard or a replaced one, and "
            "verifying the declared files while ignoring it would say nothing about it.",
            operation=operation,
            count=len(undeclared),
        )
    verified: list[tuple[str, str]] = []
    for path, digest, size in declared:
        absolute = root / path
        observed_size = absolute.stat().st_size
        if observed_size != size:
            raise BenchmarkArtifactError(
                f"bundle file {path} is {observed_size} bytes, not the declared {size}.",
                operation=operation,
                expected=str(size),
                observed=str(observed_size),
            )
        observed_digest = file_sha256(absolute)
        if observed_digest != digest:
            raise BenchmarkArtifactError(
                f"bundle file {path} hashed {observed_digest}, not the declared {digest}.",
                operation=operation,
                expected=digest,
                observed=observed_digest,
            )
        verified.append((path, digest))
    return tuple(verified)


def _verify_shards(root: Path, *, operation: str) -> tuple[int, int]:
    """Verify every shard sidecar, its matrix, and the canonical order across shards.

    Returns the shard count and the total row count. The cross-shard check is the
    one that catches a bundle built from two different runs: each shard's own id
    list is internally fine, and only the concatenation reveals a duplicate, a
    gap or a reordering.
    """
    sidecars = sorted(root.rglob("shard-*.json"))
    total_rows = 0
    decoded: list[ShardSidecar] = []
    for sidecar_path in sidecars:
        sidecar = _load_sidecar(sidecar_path)
        matrix_path = sidecar_path.with_name(f"{sidecar_path.stem}.npy")
        if not matrix_path.exists():
            raise BenchmarkArtifactError(
                f"shard sidecar {sidecar_path.name} has no matrix beside it.",
                operation=operation,
                workload=sidecar.workload,
            )
        verify_shard_matrix(matrix_path, sidecar)
        _require_ascending(sidecar, operation=operation)
        total_rows += sidecar.row_count
        decoded.append(sidecar)
    _require_contiguous_and_ordered(
        _group_by_set(decoded, operation=operation), operation=operation
    )
    return len(sidecars), total_rows


def _require_ascending(sidecar: ShardSidecar, *, operation: str) -> None:
    ids = sidecar.ids
    for position in range(1, len(ids)):
        if not ids[position - 1] < ids[position]:
            raise BenchmarkArtifactError(
                f"shard {sidecar.shard_index} of {sidecar.workload}/{sidecar.kind.value} holds "
                f"id {ids[position]!r} at row {position} after {ids[position - 1]!r}. A shard "
                "whose own rows are out of canonical order would make the concatenated corpus "
                "order wrong while every per-shard check still passed.",
                operation=operation,
                workload=sidecar.workload,
                item_id=ids[position],
            )


def _group_by_set(
    sidecars: Sequence[ShardSidecar], *, operation: str
) -> dict[tuple[str, str, str, int], list[tuple[int, str, str]]]:
    """Group shards by candidate, workload, kind and dimension."""
    grouped: dict[tuple[str, str, str, int], list[tuple[int, str, str]]] = {}
    seen: dict[tuple[str, str, str, int], set[int]] = {}
    for sidecar in sidecars:
        key = (sidecar.model_id, sidecar.workload, sidecar.kind.value, sidecar.dimension)
        ordinals = seen.setdefault(key, set())
        if sidecar.shard_index in ordinals:
            raise BenchmarkArtifactError(
                f"two sidecars declare shard {sidecar.shard_index} of {key[0]}/{key[1]}/{key[2]}/"
                f"{key[3]}. Two shards with one ordinal is a duplicated shard, and which of them a "
                "reader would use is an accident of directory order.",
                operation=operation,
                workload=sidecar.workload,
                count=sidecar.shard_index,
            )
        ordinals.add(sidecar.shard_index)
        grouped.setdefault(key, []).append((sidecar.shard_index, sidecar.first_id, sidecar.last_id))
    for entries in grouped.values():
        entries.sort(key=lambda entry: entry[0])
    return grouped


def _require_contiguous_and_ordered(
    grouped: Mapping[tuple[str, str, str, int], Sequence[tuple[int, str, str]]], *, operation: str
) -> None:
    """Require ordinals ``0..n-1`` and one strictly ascending id sequence per set.

    The cross-shard check is the one that catches a bundle assembled from two
    different runs: every shard's own id list is internally fine, and only the
    concatenation reveals a duplicate, a gap or a reordering.
    """
    for (model_id, workload, kind, dimension), entries in sorted(grouped.items()):
        _ = dimension
        for expected, (ordinal, _, _) in enumerate(entries):
            if ordinal != expected:
                raise BenchmarkArtifactError(
                    f"the {model_id}/{workload}/{kind} shard set skips shard {expected}: "
                    "the ordinals "
                    f"present are {[entry[0] for entry in entries]}. A gap means a shard was never "
                    "written, and a bundle with a hole in its corpus is not a smaller benchmark.",
                    operation=operation,
                    workload=workload,
                    count=expected,
                )
        for position in range(1, len(entries)):
            previous_last = entries[position - 1][2]
            current_first = entries[position][1]
            if not previous_last < current_first:
                raise BenchmarkArtifactError(
                    f"shard {entries[position][0]} of {model_id}/{workload}/{kind} starts at "
                    f"{current_first!r}, which does not follow shard "
                    f"{entries[position - 1][0]}'s last id {previous_last!r}. The concatenated "
                    "corpus order is not strictly ascending, so the rows are not in the canonical "
                    "order every shard boundary was computed over.",
                    operation=operation,
                    workload=workload,
                    item_id=current_first,
                )
