"""Idempotent Europe PMC acquisition: fetch, hash, store, persist (RES-132).

The acquisition flow is:

    fetch exact bytes
    -> SHA-256 over the exact bytes
    -> narrow metadata validation (inside the client)
    -> content-addressed object-store put/verify
    -> database get-or-insert on the deterministic artifact_key

The durable writes are ordered so a database record can never reference an
object that failed to persist: the object is stored and verified before any
row is written, and a harmless orphaned blob after a database failure is
preferable to a canonical artifact pointing at missing bytes.

Idempotency is the deterministic identity: same source system + same external
id + same content digest is the same artifact, so repeated acquisition of
identical bytes returns the existing row instead of inserting a duplicate. The
database race (two concurrent acquisitions of the same artifact) is handled
with a savepoint around the insert: a UNIQUE violation rolls back to the
savepoint and resolves through the deterministic lookup instead of escaping to
the caller.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from dynamisrag.db.canonical import get_source_artifact_by_key, insert_source_artifact
from dynamisrag.db.models import SourceArtifactRecord
from dynamisrag.domain.contracts import SourceArtifact
from dynamisrag.ingestion.europe_pmc import EuropePmcClient
from dynamisrag.storage import ObjectStore

__all__ = ["AcquisitionResult", "EuropePmcAcquisition"]

_UNIQUE_VIOLATION_SQLSTATE = "23505"
"""PostgreSQL SQLSTATE for a UNIQUE constraint violation."""


@dataclass(frozen=True)
class AcquisitionResult:
    """The outcome of one acquisition: the artifact and whether it is new."""

    artifact: SourceArtifact
    created: bool


class EuropePmcAcquisition:
    """Acquires Europe PMC full-text artifacts and persists them idempotently."""

    def __init__(self, client: EuropePmcClient, store: ObjectStore) -> None:
        self._client = client
        self._store = store

    def acquire(self, session: Session, pmcid: str) -> AcquisitionResult:
        """Acquire one article by PMCID, idempotently.

        Returns the persisted artifact and whether this call created it.
        Re-acquiring identical bytes returns the existing artifact with
        ``created=False``; the same PMCID with different bytes is a different
        artifact and creates a new row.
        """
        payload = self._client.fetch_fulltext(pmcid)
        content_sha256 = hashlib.sha256(payload.content).hexdigest()
        stored = self._store.put_if_absent(content_sha256, payload.content)
        artifact = SourceArtifact(
            source_system="europe_pmc",
            source_external_id=payload.pmcid,
            source_uri=payload.source_uri,
            media_type=payload.media_type,
            content_sha256=content_sha256,
            byte_size=len(payload.content),
            retrieved_at=datetime.now(UTC),
            storage_uri=stored.storage_uri,
            license_name=payload.license_name,
            license_uri=payload.license_uri,
        )
        return self._persist(session, artifact)

    def _persist(self, session: Session, artifact: SourceArtifact) -> AcquisitionResult:
        existing = get_source_artifact_by_key(session, artifact.artifact_key)
        if existing is not None:
            return AcquisitionResult(artifact=_artifact_from_record(existing), created=False)
        try:
            with session.begin_nested():
                insert_source_artifact(session, artifact)
        except IntegrityError as error:
            if not _is_unique_violation(error):
                raise
            winner = get_source_artifact_by_key(session, artifact.artifact_key)
            if winner is None:
                raise
            return AcquisitionResult(artifact=_artifact_from_record(winner), created=False)
        return AcquisitionResult(artifact=artifact, created=True)


def _is_unique_violation(error: IntegrityError) -> bool:
    return getattr(error.orig, "sqlstate", None) == _UNIQUE_VIOLATION_SQLSTATE


def _artifact_from_record(record: SourceArtifactRecord) -> SourceArtifact:
    return SourceArtifact(
        id=record.id,
        source_system=record.source_system,
        source_external_id=record.source_external_id,
        source_uri=record.source_uri,
        media_type=record.media_type,
        content_sha256=record.content_sha256,
        byte_size=record.byte_size,
        retrieved_at=record.retrieved_at,
        storage_uri=record.storage_uri,
        license_name=record.license_name,
        license_uri=record.license_uri,
    )
