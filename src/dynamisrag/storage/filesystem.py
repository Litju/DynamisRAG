"""Local filesystem object store with content-addressed, atomic writes.

The store is the development/reference implementation of the object-store
boundary introduced by RES-132. Bytes are laid out under a deterministic
``sha256/<prefix>/<digest>.xml`` path derived only from their content digest,
so identical bytes always map to the same object and the storage location
never depends on random identifiers.

Writes are atomic: content is written to a temporary file in the target
directory and promoted with :func:`os.replace`, so a reader never observes a
partially-written artifact. An existing content-addressed object is never
rewritten blindly — its bytes are verified against the requested digest, and a
mismatch fails loudly with :class:`ObjectStoreIntegrityError` instead of
overwriting storage corruption or a programming defect.
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
        target = self._path_for(content_sha256)
        if target.exists():
            return self._verify_existing(target, content_sha256)
        self._write_atomically(target, data)
        return StoredObject(
            storage_uri=target.as_uri(),
            content_sha256=content_sha256,
            byte_size=len(data),
        )

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

    def _write_atomically(self, target: Path, data: bytes) -> None:
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
            temporary.replace(target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        _sync_directory(target.parent)


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
