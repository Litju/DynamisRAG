"""Canonical bytes, streaming digests and verified archive extraction (RES-141).

Three rules, inherited from the RES-138 frozen-source discipline and restated
here because the dataset adapters depend on them:

1. **A digest is the only authority.** A file name, size or modification time is
   never compared to decide what a source is. Every accepted archive and every
   member read out of it is verified against a pinned SHA-256.
2. **Canonical JSON has one definition.** This module reuses
   :func:`~dynamisrag.ir.contracts.canonical_ir_json`, the RES-140
   canonicalisation, so a slice digest and an IR dataset digest are computed the
   same way on every platform: UTF-8, sorted keys, compact separators, one LF.
3. **Hashing is streamed.** The SciFact-Open corpus is 889 MB and the SciDocs
   corpus is 257 MB; nothing here materialises a corpus in memory to hash it.
"""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import IO, Final

from dynamisrag.datasets.errors import DatasetSourceError
from dynamisrag.ir.contracts import canonical_ir_json

__all__ = [
    "READ_CHUNK_BYTES",
    "canonical_bytes",
    "canonical_sequence_digest",
    "digest",
    "extract_members",
    "file_sha256",
    "ordered_ids_sha256",
    "read_member_bytes",
    "text_sha256",
    "verify_file_against",
    "verify_member_pin",
]

READ_CHUNK_BYTES: Final[int] = 1024 * 1024
"""Hash and copy chunk size. Large enough to keep syscalls cheap, small enough
that a 889 MB member is never resident in memory."""


def canonical_bytes(payload: object) -> bytes:
    """Exact UTF-8 canonical artifact bytes, one definition for every artifact."""
    return canonical_ir_json(payload)


def digest(payload: object) -> str:
    """SHA-256 of the canonical bytes of ``payload``."""
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def canonical_sequence_digest(items: Sequence[object]) -> str:
    """SHA-256 of the canonical JSON encoding of a list, computed in bounded memory.

    ``canonical_bytes`` on a list of 500,000 SciFact-Open identity entries would
    hold the whole encoding in memory; this framing emits exactly the same bytes
    incrementally: ``[``, comma-joined canonical items without their trailing LF,
    ``]`` and one final LF. A digest computed here is therefore identical to
    ``digest(list(items))``.
    """
    hasher = hashlib.sha256()
    hasher.update(b"[")
    for index, item in enumerate(items):
        if index:
            hasher.update(b",")
        encoded = canonical_bytes(item)
        hasher.update(encoded[:-1])
    hasher.update(b"]\n")
    return hasher.hexdigest()


def text_sha256(text: str) -> str:
    """SHA-256 over the exact UTF-8 bytes of one text value."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """SHA-256 over a file's bytes, streamed."""
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(READ_CHUNK_BYTES):
                hasher.update(chunk)
    except OSError as error:
        raise DatasetSourceError(
            f"could not read {path.name} to hash it ({type(error).__name__}). An unreadable "
            "source cannot be accepted as anything.",
            operation="file_sha256",
            item_id=path.name,
        ) from None
    return hasher.hexdigest()


def ordered_ids_sha256(ids: Iterable[str]) -> str | None:
    """Digest of a sorted, de-duplicated identifier list, or ``None`` when empty.

    Exclusions are declared by count *and* by identity digest so two runs that
    excluded different items with the same count are distinguishable without
    republishing the excluded content.
    """
    ordered = sorted(set(ids))
    if not ordered:
        return None
    return digest(ordered)


def verify_file_against(
    path: Path,
    *,
    size_bytes: int,
    sha256: str,
    operation: str,
    source_id: str | None = None,
    member: str | None = None,
) -> None:
    """Refuse ``path`` unless its size and digest are exactly the pinned pair."""
    try:
        observed_size = path.stat().st_size
    except OSError as error:
        raise DatasetSourceError(
            f"the source file {path.name!r} could not be inspected ({type(error).__name__}).",
            operation=operation,
            source_id=source_id,
            item_id=member,
        ) from None
    if observed_size != size_bytes:
        raise DatasetSourceError(
            f"the source artifact {member or path.name!r} is {observed_size} bytes, not the "
            f"pinned {size_bytes}.",
            operation=operation,
            source_id=source_id,
            item_id=member,
            expected=str(size_bytes),
            observed=str(observed_size),
        )
    observed_digest = file_sha256(path)
    if observed_digest != sha256:
        raise DatasetSourceError(
            f"the source artifact {member or path.name!r} hashed {observed_digest}, which is not "
            f"the pinned {sha256}. Either the transfer was truncated or the distribution changed; "
            "nothing was read.",
            operation=operation,
            source_id=source_id,
            item_id=member,
            expected=sha256,
            observed=observed_digest,
        )


def verify_member_pin(path: Path, *, member: str, size_bytes: int, sha256: str) -> None:
    """Refuse a member that is not its pinned bytes, naming the member."""
    verify_file_against(
        path, size_bytes=size_bytes, sha256=sha256, operation="verify_source_member", member=member
    )


def _member_parts(name: str, *, operation: str) -> tuple[str, ...]:
    """Split a declared member path, refusing traversal and absolute names."""
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"..", ""} for part in path.parts):
        raise DatasetSourceError(
            f"the declared member {name!r} is not a safe relative path; archives are extracted "
            "under the adapter's own scratch root only.",
            operation=operation,
            item_id=name,
        )
    return path.parts


def _copy_stream(source: IO[bytes], destination: Path) -> None:
    """Copy a binary stream to ``destination`` in bounded chunks."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as writer:
        while chunk := source.read(READ_CHUNK_BYTES):
            writer.write(chunk)


def read_member_bytes(archive: Path, *, member: str, archive_format: str) -> bytes:
    """Read one declared member into memory. Only for members pinned as small."""
    parts = _member_parts(member, operation="read_source_member")
    try:
        if archive_format == "zip":
            with zipfile.ZipFile(archive) as handle:
                return handle.read("/".join(parts))
        if archive_format == "tar.gz":
            with tarfile.open(archive, "r:gz") as handle:
                extracted = handle.extractfile("/".join(parts))
                if extracted is None:
                    raise KeyError(member)
                with extracted:
                    return extracted.read()
    except (OSError, KeyError, zipfile.BadZipFile, tarfile.TarError) as error:
        raise DatasetSourceError(
            f"the member {member!r} could not be read from the verified archive "
            f"({type(error).__name__}).",
            operation="read_source_member",
            item_id=member,
        ) from None
    raise DatasetSourceError(
        f"the archive format {archive_format!r} is not supported by this adapter.",
        operation="read_source_member",
        item_id=member,
    )


def extract_members(
    archive: Path,
    *,
    archive_format: str,
    members: Sequence[str],
    destination: Path,
) -> dict[str, Path]:
    """Extract exactly the declared members under ``destination``, and nothing else.

    Selective extraction is what lets the SciFact-Open ``candidates`` variant
    ignore the 889 MB full corpus: a member list is a contract, not a suggestion.
    No directory entries, no symlinks and no undeclared names are ever written.
    A member that is absent from the archive is refused before any write.
    """
    for member in members:
        _member_parts(member, operation="extract_source_members")
    destination.mkdir(parents=True, exist_ok=True)
    extracted: dict[str, Path] = {}
    if archive_format == "zip":
        try:
            with zipfile.ZipFile(archive) as handle:
                names = set(handle.namelist())
                missing = [member for member in members if member not in names]
                if missing:
                    raise DatasetSourceError(
                        "the verified archive does not contain every declared member.",
                        operation="extract_source_members",
                        count=len(missing),
                    )
                for member in members:
                    target = destination.joinpath(
                        *_member_parts(member, operation="extract_source_members")
                    )
                    with handle.open(member) as source:
                        _copy_stream(source, target)
                    extracted[member] = target
        except zipfile.BadZipFile:
            raise DatasetSourceError(
                "the archive is not a readable ZIP despite matching its pinned digest; that "
                "combination is impossible, so the working copy is wrong.",
                operation="extract_source_members",
            ) from None
        return extracted
    if archive_format == "tar.gz":
        try:
            with tarfile.open(archive, "r:gz") as handle:
                for member in members:
                    info = handle.getmember(member)
                    if not info.isfile() or info.issym() or info.islnk():
                        raise DatasetSourceError(
                            "a declared member is not a regular file in the archive.",
                            operation="extract_source_members",
                            item_id=member,
                        )
                    source = handle.extractfile(member)
                    if source is None:
                        raise DatasetSourceError(
                            "a declared member could not be extracted from the archive.",
                            operation="extract_source_members",
                            item_id=member,
                        )
                    target = destination.joinpath(
                        *_member_parts(member, operation="extract_source_members")
                    )
                    with source:
                        _copy_stream(source, target)
                    extracted[member] = target
        except (KeyError, tarfile.TarError):
            raise DatasetSourceError(
                "the verified archive could not be read as a gzip tar stream, or a declared "
                "member is absent.",
                operation="extract_source_members",
            ) from None
        return extracted
    raise DatasetSourceError(
        f"the archive format {archive_format!r} is not supported by this adapter.",
        operation="extract_source_members",
    )


def copy_file(source: Path, destination: Path) -> None:
    """Copy one local file in bounded chunks, creating parent directories."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("wb") as writer:
        shutil.copyfileobj(reader, writer, READ_CHUNK_BYTES)
