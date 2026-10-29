"""Real PostgreSQL transactions, offline upstreams and interrupted corpus publication."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import httpx2
import psycopg
import pytest
from alembic import command
from alembic.config import Config
from psycopg import sql
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from dynamisrag.config import Settings
from dynamisrag.corpus import pipeline
from dynamisrag.corpus.discovery import BoundedHttp, DiscoveryPlan
from dynamisrag.corpus.pipeline import run_snapshot
from dynamisrag.corpus.snapshot import publish_json, read_json, verify_manifest
from dynamisrag.db import create_database_engine
from dynamisrag.db.models import DocumentVersionRecord, PassageRecord, SourceArtifactRecord
from dynamisrag.ingestion.europe_pmc import EuropePmcClient
from dynamisrag.storage import ObjectStoreIntegrityError
from tests._corpus import XML, CorpusTransport
from tests._support import ALEMBIC_INI, REPO_ROOT, build_settings

pytestmark = pytest.mark.integration


@pytest.fixture
def corpus_engine(live_settings: Settings, monkeypatch: pytest.MonkeyPatch) -> Iterator[Engine]:
    """Isolated database: test commits never enter the existing scientific baseline."""
    database = f"dynamisrag_res143_test_{uuid4().hex[:12]}"
    dsn = urlsplit(str(live_settings.database_url))
    maintenance = urlunsplit(dsn._replace(path="/postgres"))
    scratch = urlunsplit(dsn._replace(path=f"/{database}"))
    with psycopg.connect(maintenance, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    engine = create_database_engine(build_settings(database_url=scratch))
    try:
        monkeypatch.setenv("DYNAMISRAG_DATABASE_URL", scratch)
        config = Config(str(ALEMBIC_INI))
        config.set_main_option("script_location", str(REPO_ROOT / "alembic"))
        command.upgrade(config, "head")
        yield engine
    finally:
        engine.dispose()
        with psycopg.connect(maintenance, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(database)))


def _run(engine: Engine, root: Path, transport: CorpusTransport) -> dict[str, Any]:
    fulltext = EuropePmcClient(transport=transport)
    try:
        with httpx2.Client(transport=transport) as client:
            return run_snapshot(
                engine=engine,
                http=BoundedHttp(client),
                fulltext=fulltext,
                run_root=root / "run",
                store_root=root / "objects",
                plan=DiscoveryPlan(topics=("sprint",), page_size=4),
            )
    finally:
        fulltext.close()


def _counts(engine: Engine) -> tuple[int, int, int]:
    with Session(engine) as session:
        counts = [
            session.scalar(select(func.count()).select_from(model)) or 0
            for model in (SourceArtifactRecord, DocumentVersionRecord, PassageRecord)
        ]
        return counts[0], counts[1], counts[2]


def test_real_canonical_acquisition_and_network_free_replay(
    corpus_engine: Engine, tmp_path: Path
) -> None:
    transport = CorpusTransport()
    first = _run(corpus_engine, tmp_path, transport)
    assert first["acquired_articles"] == 1 and first["deduplicated_records"] == 2
    assert _counts(corpus_engine) == (1, 1, 0)
    assert transport.fulltext_requests == 1
    before = Path(first["manifest"]).read_bytes()
    # An upstream mutation cannot change the pinned source, even across a fresh transaction.
    transport.content = XML.replace(b"Synthetic human", b"Mutated human")
    transport.fail_network = True
    assert _run(corpus_engine, tmp_path, transport) == first
    assert Path(first["manifest"]).read_bytes() == before
    assert _counts(corpus_engine) == (1, 1, 0)
    verify_manifest(
        read_json(Path(first["manifest"])),
        read_json(tmp_path / "run/discovery.json"),
        tmp_path / "objects",
    )


@pytest.mark.parametrize("boundary", ["stage", "decision", "manifest"])
def test_recovery_after_interrupted_staging_commit_or_publication(
    corpus_engine: Engine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    transport = CorpusTransport()

    def interrupted(path: Path, value: object) -> None:
        if (
            (boundary == "stage" and path.parent.name == "staged")
            or (boundary == "decision" and path.parent.name == "decisions")
            or (boundary == "manifest" and path.parent.name == "snapshots")
        ):
            if boundary == "stage":
                publish_json(path, value)
            raise OSError(f"interrupted {boundary}")
        publish_json(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(pipeline, "publish_json", interrupted)
        with pytest.raises(OSError, match="interrupted"):
            _run(corpus_engine, tmp_path, transport)
    assert not list((tmp_path / "run/snapshots").glob("*.json"))
    assert _counts(corpus_engine) == ((0, 0, 0) if boundary == "stage" else (1, 1, 0))
    transport.fail_network = True
    result = _run(corpus_engine, tmp_path, transport)
    assert result["acquired_articles"] == 1 and _counts(corpus_engine) == (1, 1, 0)
    assert transport.fulltext_requests == 1


def test_rejected_original_license_never_enters_object_store_or_database(
    corpus_engine: Engine,
    tmp_path: Path,
) -> None:
    transport = CorpusTransport()
    transport.content = XML.replace(b"/licenses/by/", b"/licenses/by-nc/")
    result = _run(corpus_engine, tmp_path, transport)
    assert result["rejected_articles"] == 1 and result["acquired_articles"] == 0
    assert _counts(corpus_engine) == (0, 0, 0)
    assert not list((tmp_path / "objects").rglob("*.xml"))


def test_object_corruption_blocks_republication(corpus_engine: Engine, tmp_path: Path) -> None:
    transport = CorpusTransport()
    _run(corpus_engine, tmp_path, transport)
    target = next((tmp_path / "objects").rglob("*.xml"))
    target.write_bytes(b"corrupt")
    transport.fail_network = True
    with pytest.raises(ObjectStoreIntegrityError, match=r"hash|digest"):
        _run(corpus_engine, tmp_path, transport)
