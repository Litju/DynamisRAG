"""Bounded RES-145 real-source demonstration; reference access is read-only.

Copies one retained passage and its original lineage into a throwaway database,
verifies original XML bytes, publishes/reconstructs a provisional graph, and
round-trips the new migration. Only identifiers and digests enter the receipt.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit
from uuid import uuid4

import psycopg
from alembic import command
from alembic.config import Config
from psycopg import sql
from pydantic import PostgresDsn
from sqlalchemy import Engine, create_engine, insert, select, text
from sqlalchemy.orm import Session

from dynamisrag.config import load_settings, to_sqlalchemy_url
from dynamisrag.db import create_database_engine
from dynamisrag.db.canonical import list_passage_projection_records
from dynamisrag.db.evidence import canonical_member, get_graph, persist_graph, source_fingerprint
from dynamisrag.db.migration_safety import (
    DISPOSABLE_DATABASE_MARKER_PREFIX,
    MAINTENANCE_ROLE,
    TEST_DATABASE_ROLE,
)
from dynamisrag.db.models import (
    CitationRecord,
    CitationResolutionRecord,
    DocumentIdentifierRecord,
    DocumentRecord,
    DocumentTableRecord,
    DocumentVersionRecord,
    FigureRecord,
    ParagraphRecord,
    PassageRecord,
    PassageSourceSpanRecord,
    SectionRecord,
    SourceArtifactRecord,
)
from dynamisrag.domain.evidence import (
    Claim,
    EvidenceGraph,
    EvidenceLink,
    EvidenceMember,
    EvidenceSet,
    Relation,
)
from dynamisrag.domain.identity import evidence_identity
from dynamisrag.search.qdrant import QdrantClient
from dynamisrag.search.scientific import CHUNKER, COLLECTION

PASSAGE = "37ad148429ed0342fb2bc0e02fadc5194a4be1aa8f16bab1df8d433336a50144"
ROOT = Path(__file__).resolve().parents[1]
SOURCE_MODELS = (
    SourceArtifactRecord,
    DocumentRecord,
    DocumentIdentifierRecord,
    DocumentVersionRecord,
    SectionRecord,
    ParagraphRecord,
    PassageRecord,
    PassageSourceSpanRecord,
    CitationRecord,
    CitationResolutionRecord,
    DocumentTableRecord,
    FigureRecord,
)


def database_receipt(engine: Engine) -> dict[str, object]:
    result: dict[str, object] = {}
    with engine.connect() as connection:
        connection.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"))
        for model in SOURCE_MODELS:
            # Table names come exclusively from the fixed canonical ORM inventory.
            rows = connection.execute(select(model.__table__).order_by(model.id)).mappings().all()
            encoded = json.dumps([dict(row) for row in rows], default=str, sort_keys=True).encode()
            result[model.__tablename__] = {
                "count": len(rows),
                "sha256": sha256(encoded).hexdigest(),
            }
    return result


def protected_files() -> str:
    paths = sorted((ROOT / ".dynamisrag/real-science-v1").rglob("*"))
    return evidence_identity(
        "protected-files-v1",
        [
            (p.relative_to(ROOT).as_posix(), sha256(p.read_bytes()).hexdigest())
            for p in paths
            if p.is_file()
        ],
    )


def qdrant_receipt() -> str:
    client = QdrantClient(str(load_settings().scientific_qdrant_url).rstrip("/"))
    try:
        response = client.request(
            "POST",
            f"/collections/{COLLECTION}/points/scroll",
            body={"limit": 100, "with_payload": True, "with_vector": True},
        )
        result = response.get("result")
        if not isinstance(result, dict) or result.get("next_page_offset") is not None:
            raise ValueError("unexpected reference collection inventory")
        points = result.get("points")
        if not isinstance(points, list) or len(points) != 75:
            raise ValueError("expected exactly 75 retained vectors")
        return evidence_identity(
            "qdrant-reference-v1", {"aliases": client.aliases(), "points": points}
        )
    finally:
        client.close()


def example_graph(member: EvidenceMember, version_key: str, span_key: str) -> EvidenceGraph:
    evidence_set = EvidenceSet(
        question_id="res145-method-demo-v1",
        question=(
            "What instrument measured strength, power and work in the starting knee-angle study?"
        ),
        corpus_id=COLLECTION,
        retrieval_revision=CHUNKER,
        selection_context="Manual bounded source selection; no LLM or retrieval rerun",
        constraints=(
            "Original retained canonical source only",
            "Provisional methodological interpretation",
        ),
        members=(member,),
    )
    claim = Claim(
        proposition=(
            "The starting knee-angle study measured lower-limb strength, power and work "
            "with a functional electromechanical dynamometer."
        ),
        category="METHODOLOGICAL",
        origin="HUMAN_AUTHORED",
        origin_reference=f"{version_key}:{span_key}",
        authoring_process="res145-engineering-demo-v1",
        limitations=(
            "Unsigned provisional interpretation; no expert approval or entailment certification",
        ),
    )
    link = EvidenceLink(
        claim_key=claim.key,
        evidence_set_key=evidence_set.key,
        member=member,
        span_key=span_key,
        relation=Relation.CONTEXTUAL,
        rationale=(
            "The original methods paragraph describes the instrument; "
            "this is a provisional context assertion."
        ),
        authoring_process="res145-engineering-demo-v1",
    )
    return EvidenceGraph(evidence_set=evidence_set, claims=(claim,), links=(link,))


def _drop_scratch_database(
    test_dsn: str, database: str, database_oid: int, owner: str, marker: str
) -> None:
    with psycopg.connect(test_dsn, autocommit=True) as connection:
        identity_query = """
            SELECT d.oid, pg_get_userbyid(d.datdba),
                   shobj_description(d.oid, 'pg_database')
            FROM pg_database AS d WHERE d.datname=%s
            """
        identity = connection.execute(identity_query, (database,)).fetchone()
        if (
            identity is None
            or identity[0] != database_oid
            or identity[1] != owner
            or identity[2] != marker
        ):
            raise RuntimeError(
                f"Leaving disposable database {database!r} (OID {database_oid}) "
                "because its identity changed "
                f"(oid_match={identity is not None and identity[0] == database_oid}, "
                f"owner_match={identity is not None and identity[1] == owner}, "
                f"marker_match={identity is not None and identity[2] == marker})"
            )
        sessions = 1
        for _ in range(20):
            sessions = connection.execute(
                "SELECT count(*) FROM pg_stat_activity WHERE datname=%s AND pid<>pg_backend_pid()",
                (database,),
            ).fetchone()[0]
            if sessions == 0:
                break
            time.sleep(0.1)
        identity = connection.execute(identity_query, (database,)).fetchone()
        if (
            identity is None
            or identity[0] != database_oid
            or identity[1] != owner
            or identity[2] != marker
            or sessions
        ):
            raise RuntimeError(
                f"Leaving disposable database {database!r} (OID {database_oid}) "
                "because its identity or connection state changed "
                f"(oid_match={identity is not None and identity[0] == database_oid}, "
                f"owner_match={identity is not None and identity[1] == owner}, "
                f"marker_match={identity is not None and identity[2] == marker}, "
                f"active_sessions={sessions})"
            )
        connection.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(database)))


@contextmanager
def scratch_database(revision: str = "head") -> Generator[tuple[Engine, Config]]:
    settings = load_settings()
    test_database_url = settings.test_database_url
    if test_database_url is None:
        raise RuntimeError("DYNAMISRAG_TEST_DATABASE_URL must use the dedicated test role")

    test_dsn = str(test_database_url)
    test_url = urlsplit(test_dsn)
    app_url = urlsplit(str(settings.database_url))
    database = f"dynamisrag_test_{uuid4().hex}"
    if test_url.path.lstrip("/") == app_url.path.lstrip("/"):
        raise RuntimeError("The test database URL must not name the application database")
    scratch = urlunsplit(test_url._replace(path=f"/{database}", fragment=""))
    marker = f"{DISPOSABLE_DATABASE_MARKER_PREFIX}{uuid4().hex}"
    previous = os.environ.get("DYNAMISRAG_DATABASE_URL")
    engine: Engine | None = None
    database_oid: int | None = None
    created = False
    test_role: str | None = None
    try:
        with psycopg.connect(test_dsn, autocommit=True) as connection:
            role = connection.execute(
                """
                SELECT session_user, current_user, r.rolsuper, r.rolcreatedb,
                       r.rolcreaterole, r.rolreplication, r.rolbypassrls,
                       r.rolinherit,
                       EXISTS (
                           SELECT 1 FROM pg_auth_members AS m
                           JOIN pg_roles AS ops ON ops.oid=m.roleid
                           WHERE m.member=r.oid AND ops.rolname=%s
                       )
                FROM pg_roles AS r WHERE r.rolname=current_user
                """,
                (MAINTENANCE_ROLE,),
            ).fetchone()
            if (
                role is None
                or role[0] != role[1]
                or role[1] != TEST_DATABASE_ROLE
                or role[2]
                or not role[3]
                or role[4]
                or role[5]
                or role[6]
                or role[7]
                or role[8]
            ):
                raise RuntimeError(
                    "DYNAMISRAG_TEST_DATABASE_URL must use the unprivileged "
                    "dynamisrag_test role with CREATEDB and without CREATEROLE"
                )
            test_role = role[1]
            connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
            created = True
            identity = connection.execute(
                "SELECT oid, pg_get_userbyid(datdba) FROM pg_database WHERE datname=%s",
                (database,),
            ).fetchone()
            if identity is None or identity[1] != test_role:
                raise RuntimeError("The disposable database is not owned by dynamisrag_test")
            database_oid = identity[0]
            connection.execute(
                sql.SQL("COMMENT ON DATABASE {} IS {}").format(
                    sql.Identifier(database), sql.Literal(marker)
                )
            )

        engine = create_engine(to_sqlalchemy_url(PostgresDsn(scratch)))
        os.environ["DYNAMISRAG_DATABASE_URL"] = scratch
        config = Config(str(ROOT / "alembic.ini"))
        config.set_main_option("script_location", str(ROOT / "alembic"))
        command.upgrade(config, revision)
        yield engine, config
    finally:
        if previous is None:
            os.environ.pop("DYNAMISRAG_DATABASE_URL", None)
        else:
            os.environ["DYNAMISRAG_DATABASE_URL"] = previous
        if engine is not None:
            engine.dispose()
        if created and database_oid is not None and test_role is not None:
            _drop_scratch_database(test_dsn, database, database_oid, test_role, marker)


def demonstrate() -> dict[str, object]:
    settings = load_settings()
    reference = create_database_engine(settings)
    before = database_receipt(reference)
    files_before, vectors_before = protected_files(), qdrant_receipt()
    with Session(reference) as source:
        source.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"))
        records = list_passage_projection_records(
            source, chunker_revision=CHUNKER, passage_keys=(PASSAGE,)
        )
        if len(records) != 1:
            raise ValueError("required original canonical passage unavailable")
        record = records[0]
        fingerprint = source_fingerprint(record)
        uri = urlsplit(record.artifact.storage_uri)
        path = unquote(uri.path).lstrip("/") if uri.scheme == "file" else ""
        if (
            not path
            or sha256(Path(path).read_bytes()).hexdigest() != record.artifact.content_sha256
        ):
            raise ValueError("original XML unavailable or altered")
        span = next(
            x.span
            for x in record.source_spans
            if "functional electromechanical dynamometer" in x.paragraph.text
        )
        rows: list[
            SourceArtifactRecord
            | DocumentRecord
            | DocumentIdentifierRecord
            | DocumentVersionRecord
            | SectionRecord
            | ParagraphRecord
            | PassageRecord
            | PassageSourceSpanRecord
        ] = [record.artifact, record.document, *record.identifiers, record.version]
        # Parent-first sections retain exact original UUIDs and anchors.
        rows.extend(
            source.scalars(
                select(SectionRecord)
                .where(SectionRecord.document_version_id == record.version.id)
                .order_by(SectionRecord.depth)
            )
        )
        rows.extend(
            source.scalars(
                select(ParagraphRecord).where(
                    ParagraphRecord.document_version_id == record.version.id
                )
            )
        )
        rows.extend([record.passage, *(x.span for x in record.source_spans)])
        copies = [
            (type(row), {c.name: getattr(row, c.name) for c in row.__table__.columns})
            for row in rows
        ]
        version_key, span_key, artifact_hash = (
            record.version.version_key,
            span.span_key,
            record.artifact.content_sha256,
        )
    with scratch_database() as (engine, config):
        with Session(engine) as session, session.begin():
            for model, values in copies:
                session.execute(insert(model).values(**values))
            member = canonical_member(session, PASSAGE, version_key)
            if member.provenance_sha256 != evidence_identity("source-v1", fingerprint):
                raise ValueError("copied source provenance mismatch")
            graph = example_graph(member, version_key, span_key)
            evidence_set, claim, link = graph.evidence_set, graph.claims[0], graph.links[0]
            if persist_graph(session, graph) != graph or persist_graph(session, graph) != graph:
                raise ValueError("graph round-trip mismatch")
        with Session(engine) as session:
            if get_graph(session, evidence_set.key) != graph:
                raise ValueError("committed graph reconstruction mismatch")
        copied_before = database_receipt(engine)
        command.downgrade(config, "0004_passage_source_spans")
        if database_receipt(engine) != copied_before:
            raise ValueError("downgrade altered canonical sources")
        command.upgrade(config, "head")
        if database_receipt(engine) != copied_before:
            raise ValueError("upgrade altered canonical sources")
    after = database_receipt(reference)
    reference.dispose()
    if before != after or files_before != protected_files() or vectors_before != qdrant_receipt():
        raise ValueError("protected reference pre/post integrity mismatch")
    return {
        "schema_revision": "res145-demo-receipt-v1",
        "passage_key": PASSAGE,
        "version_key": version_key,
        "span_key": span_key,
        "artifact_sha256": artifact_hash,
        "evidence_set_key": evidence_set.key,
        "claim_key": claim.key,
        "link_key": link.key,
        "provenance_sha256": member.provenance_sha256,
        "review_state": "PROPOSED",
        "relation": "CONTEXTUAL",
        "reference_database": before,
        "reference_unchanged": True,
        "protected_files_sha256": files_before,
        "qdrant_sha256": vectors_before,
        "migration_roundtrip": "0005 -> 0004 -> 0005; canonical rows unchanged",
    }


if __name__ == "__main__":
    receipt = demonstrate()
    destination = ROOT / "docs/evidence/res145-demo-receipt.json"
    destination.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
