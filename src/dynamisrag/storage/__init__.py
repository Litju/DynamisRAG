"""Content-addressed object storage for acquired source artifacts (RES-132)."""

from __future__ import annotations

from dynamisrag.storage.filesystem import (
    FileSystemObjectStore,
    ObjectStore,
    ObjectStoreError,
    ObjectStoreIntegrityError,
    StoredObject,
)

__all__ = [
    "FileSystemObjectStore",
    "ObjectStore",
    "ObjectStoreError",
    "ObjectStoreIntegrityError",
    "StoredObject",
]
