"""Integration tests proving idempotent Europe PMC acquisition against live
PostgreSQL 18, a temporary filesystem object store and a mocked Europe PMC
HTTP response.

The mocked transport makes the suite deterministic — CI never depends on the
public Europe PMC network — while the real database proves the persistence
idempotency: repeated acquisition of identical bytes returns the existing
SourceArtifact, and changed bytes create a new artifact.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Final

import httpx2
import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from dynamisrag.config import Settings
from dynamisrag.db import create_database_engine, get_source_artifact_by_key
from dynamisrag.db.models import SourceArtifactRecord
from dynamisrag.domain.contracts import SourceArtifact
from dynamisrag.ingestion import EuropePmcAcquisition, EuropePmcClient
from dynamisrag.storage import FileSystemObjectStore

pytestmark = pytest.mark.integration

_PMCID: Final[str] = "PMC3253803"
_BASE_URL: Final[str] = "https://www.ebi.ac.uk/europepmc/webservices/rest"
_SOURCE_URI: Final[str] = f"{_BASE_URL}/{_PMCID}/fullTextXML"

_JATS_V1: Final[bytes] = b"""<?xml version="1.0" encoding="UTF-8"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink">
  <front>
    <article-meta>
      <permissions>
        <license license-type="CC BY" xlink:href="http://creativecommons.org/licenses/by/4.0/">
          <license-p>Distributed under the Creative Commons Attribution 4.0 License.</license-p>
        </license>
      </permissions>
    </article-meta>
  </front>
</article>
"""

_JATS_V2: Final[bytes] = b"""<?xml version="1.0" encoding="UTF-8"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink">
  <front>
    <article-meta>
      <permissions>
        <license license-type="CC BY" xlink:href="http://creativecommons.org/licenses/by/4.0/">
          <license-p>Revised distribution terms.</license-p>
        </license>
      </permissions>
    </article-meta>
  </front>
</article>
"""

_V1_SHA256: Final[str] = hashlib.sha256(_JATS_V1).hexdigest()
_V2_SHA256: Final[str] = hashlib.sha256(_JATS_V2).hexdigest()


class _UniqueViolationError(Exception):
    """A stand-in DBAPI exception carrying the PostgreSQL UNIQUE sqlstate."""

    sqlstate = "23505"


class _ScriptedTransport(httpx2.BaseTransport):
    """Serves a fixed sequence of bodies, repeating the last one, and records
    every request so a test can prove how many HTTP calls acquisition made."""

    def __init__(self, bodies: Sequence[bytes]) -> None:
        self._bodies = bodies
        self.requests: list[httpx2.Request] = []

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self._bodies) - 1)
        return httpx2.Response(
            200,
            content=self._bodies[index],
            headers={"content-type": "application/xml"},
            request=request,
        )


@pytest.fixture
def db_session(live_settings: Settings) -> Iterator[Session]:
    """A session whose transaction is always rolled back, keeping the
    canonical database pristine between tests."""
    engine = create_database_engine(live_settings)
    session = Session(bind=engine)
    transaction = session.begin()
    try:
        yield session
    finally:
        transaction.rollback()
        session.close()
        engine.dispose()


@pytest.fixture
def artifacts_root(tmp_path: Path) -> Path:
    return tmp_path / "artifacts"


def _stored_objects(root: Path) -> list[Path]:
    return sorted(root.rglob("*.xml"))


def _row_count(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(SourceArtifactRecord)) or 0


def test_first_acquisition_creates_exactly_one_artifact(
    db_session: Session, artifacts_root: Path
) -> None:
    transport = _ScriptedTransport([_JATS_V1])
    service = EuropePmcAcquisition(
        EuropePmcClient(transport=transport), FileSystemObjectStore(artifacts_root)
    )

    result = service.acquire(db_session, _PMCID)

    assert result.created is True
    assert len(transport.requests) == 1
    assert len(_stored_objects(artifacts_root)) == 1
    assert _row_count(db_session) == 1
    persisted = get_source_artifact_by_key(db_session, result.artifact.artifact_key)
    assert persisted is not None
    assert persisted.id == result.artifact.id


def test_repeated_acquisition_of_identical_bytes_is_idempotent(
    db_session: Session, artifacts_root: Path
) -> None:
    """The primary RES-132 acceptance test: a second complete acquisition run
    with identical source bytes returns the existing artifact — same key, same
    persisted id, one database row, one physical object, created=False."""
    transport = _ScriptedTransport([_JATS_V1])
    service = EuropePmcAcquisition(
        EuropePmcClient(transport=transport), FileSystemObjectStore(artifacts_root)
    )

    first = service.acquire(db_session, _PMCID)
    second = service.acquire(db_session, _PMCID)

    assert first.created is True
    assert second.created is False
    assert second.artifact.artifact_key == first.artifact.artifact_key
    assert second.artifact.id == first.artifact.id
    assert _row_count(db_session) == 1
    assert len(_stored_objects(artifacts_root)) == 1
    assert len(transport.requests) == 2


def test_changed_bytes_create_a_new_artifact(db_session: Session, artifacts_root: Path) -> None:
    """Same PMCID, different exact bytes: different SHA-256, different
    artifact_key, a second SourceArtifact row and a second content object."""
    transport = _ScriptedTransport([_JATS_V1, _JATS_V2])
    service = EuropePmcAcquisition(
        EuropePmcClient(transport=transport), FileSystemObjectStore(artifacts_root)
    )

    first = service.acquire(db_session, _PMCID)
    second = service.acquire(db_session, _PMCID)

    assert first.created is True
    assert second.created is True
    assert second.artifact.content_sha256 == _V2_SHA256
    assert second.artifact.artifact_key != first.artifact.artifact_key
    assert second.artifact.storage_uri != first.artifact.storage_uri
    assert _row_count(db_session) == 2
    assert len(_stored_objects(artifacts_root)) == 2


def test_acquisition_persists_complete_provenance(
    db_session: Session, artifacts_root: Path
) -> None:
    """Every provenance field on the persisted row is exactly what the source
    provided — nothing silently invented."""
    transport = _ScriptedTransport([_JATS_V1])
    service = EuropePmcAcquisition(
        EuropePmcClient(transport=transport), FileSystemObjectStore(artifacts_root)
    )

    result = service.acquire(db_session, _PMCID)

    row = get_source_artifact_by_key(db_session, result.artifact.artifact_key)
    assert row is not None
    assert row.source_system == "europe_pmc"
    assert row.source_external_id == _PMCID
    assert row.source_uri == _SOURCE_URI
    assert row.media_type == "application/xml"
    assert row.content_sha256 == _V1_SHA256
    assert row.byte_size == len(_JATS_V1)
    assert row.retrieved_at.tzinfo is not None
    assert row.storage_uri.endswith(f"{_V1_SHA256}.xml")
    assert row.license_name == "CC BY"
    assert row.license_uri == "http://creativecommons.org/licenses/by/4.0/"


def test_unique_violation_recovers_through_deterministic_lookup(
    db_session: Session, artifacts_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A concurrent acquisition that loses the insert race must not escape to
    the caller: the savepoint rolls back and the deterministic lookup returns
    the winner's row with created=False.

    The race window is simulated deterministically through the public
    ``acquire`` path: the row is already persisted (the concurrent winner's
    committed row), the first lookup is blind to it, and the insert raises a
    UNIQUE violation.
    """
    transport = _ScriptedTransport([_JATS_V1])
    service = EuropePmcAcquisition(
        EuropePmcClient(transport=transport), FileSystemObjectStore(artifacts_root)
    )

    first = service.acquire(db_session, _PMCID)
    assert first.created is True

    real_lookup = get_source_artifact_by_key
    looked_up: list[bool] = []

    def racing_lookup(session: Session, artifact_key: str) -> SourceArtifactRecord | None:
        looked_up.append(True)
        if len(looked_up) == 1:
            return None
        return real_lookup(session, artifact_key)

    def conflicting_insert(session: Session, candidate: SourceArtifact) -> SourceArtifactRecord:
        origin = _UniqueViolationError("duplicate key value violates unique constraint")
        raise IntegrityError("INSERT INTO source_artifact", {}, origin)

    monkeypatch.setattr(
        "dynamisrag.ingestion.acquisition.get_source_artifact_by_key", racing_lookup
    )
    monkeypatch.setattr(
        "dynamisrag.ingestion.acquisition.insert_source_artifact", conflicting_insert
    )

    second = service.acquire(db_session, _PMCID)

    assert second.created is False
    assert second.artifact.id == first.artifact.id
    assert _row_count(db_session) == 1
