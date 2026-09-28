"""Local filesystem object store with content-addressed, atomic writes.

The store is the development/reference implementation of the object-store
boundary introduced by RES-132. Bytes are laid out under a deterministic
``sha256/<prefix>/<digest>.xml`` path derived only from their content digest,
so identical bytes always map to the same object and the storage location
never depends on random identifiers.

The store owns the content-addressing invariant itself: before any filesystem
mutation it proves ``sha256(data) == content_sha256``, so a caller can never
create an object whose bytes do not match its digest-addressed path.

Publication is truly put-if-absent: content is written to a temporary file in
the target directory, fsynced, and promoted with an atomic hard-link create
(:func:`os.link`). A successful link publishes the already-fsynced inode
without exposing partial bytes; a :class:`FileExistsError` means another
writer won the race, so the temporary file is removed and the winner's bytes
are verified against the digest — an existing target is never overwritten.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

__all__ = [
    "FileSystemObjectStore",
    "ObjectStore",
    "ObjectStoreError",
    "ObjectStoreIntegrityError",
    "StoredObject",
]

_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
"""The exact shape of a SHA-256 hex digest used as a storage key."""


class ObjectStoreError(Exception):
    """Base error for object-store failures."""


class ObjectStoreIntegrityError(ObjectStoreError):
    """A stored object's bytes do not match its content-addressed digest.

    This condition indicates storage corruption or a programming defect; the
    object is never overwritten.
    """


@dataclass(frozen=True)
class StoredObject:
    """The result of storing bytes: where the object lives and what it is."""

    storage_uri: str
    content_sha256: str
    byte_size: int


class ObjectStore(Protocol):
    """The smallest object-store boundary acquisition currently needs."""

    def put_if_absent(self, content_sha256: str, data: bytes) -> StoredObject:
        """Store ``data`` under its content digest unless already present."""
        raise NotImplementedError


class FileSystemObjectStore:
    """Content-addressed object store on the local filesystem.

    Layout under the store root::

        <root>/sha256/<first two digest chars>/<full digest>.xml

    so the same bytes always resolve to the same object and the path is a
    pure function of the content digest.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    def put_if_absent(self, content_sha256: str, data: bytes) -> StoredObject:
        if _DIGEST_PATTERN.fullmatch(content_sha256) is None:
            raise ObjectStoreError(f"invalid content digest: {content_sha256!r}")
        actual = hashlib.sha256(data).hexdigest()
        if actual != content_sha256:
            raise ObjectStoreIntegrityError(
                f"caller supplied digest {content_sha256} but the supplied bytes hash "
                f"to {actual}; refusing to create a content-addressed object whose "
                "path does not match its content"
            )
        target = self._path_for(content_sha256)
        if target.exists():
            return self._verify_existing(target, content_sha256)
        return self._publish(target, content_sha256, data)

    def _path_for(self, content_sha256: str) -> Path:
        return self._root / "sha256" / content_sha256[:2] / f"{content_sha256}.xml"

    def _verify_existing(self, target: Path, content_sha256: str) -> StoredObject:
        stored = target.read_bytes()
        actual = hashlib.sha256(stored).hexdigest()
        if actual != content_sha256:
            raise ObjectStoreIntegrityError(
                f"stored object {target} has digest {actual} but its content-addressed "
                f"path requires {content_sha256}; refusing to overwrite"
            )
        return StoredObject(
            storage_uri=target.as_uri(),
            content_sha256=content_sha256,
            byte_size=len(stored),
        )

    def _publish(self, target: Path, content_sha256: str, data: bytes) -> StoredObject:
        """Publish ``data`` at ``target`` without ever overwriting an existing
        object.

        The temporary file lives in the target directory (same filesystem), is
        fsynced, and is then promoted with an atomic hard-link create: a
        successful :func:`os.link` publishes the already-fsynced inode, while
        :class:`FileExistsError` means another writer created the target first
        — the temporary file is removed and the winner is verified instead.
        """
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                temporary.unlink()
                return self._verify_existing(target, content_sha256)
            temporary.unlink()
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        _sync_directory(target.parent)
        return StoredObject(
            storage_uri=target.as_uri(),
            content_sha256=content_sha256,
            byte_size=len(data),
        )


def _sync_directory(directory: Path) -> None:
    """Best-effort directory fsync so the rename itself is durable."""
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
