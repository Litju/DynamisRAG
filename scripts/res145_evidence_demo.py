"""Bounded RES-145 real-source demonstration; reference access is read-only.

Copies one retained passage and its original lineage into a throwaway database,
verifies original XML bytes, publishes/reconstructs a provisional graph, and
round-trips the new migration. Only identifiers and digests enter the receipt.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
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
from sqlalchemy import Engine, create_engine, select, text
from sqlalchemy.orm import Session

from dynamisrag.config import load_settings, to_sqlalchemy_url
from dynamisrag.db import create_database_engine
from dynamisrag.db.canonical import list_passage_projection_records
from dynamisrag.db.evidence import canonical_member, get_graph, persist_graph, source_fingerprint
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


@contextmanager
def scratch_database() -> Iterator[tuple[Engine, Config]]:
    dsn = urlsplit(str(load_settings().database_url))
    database = f"res145_demo_{uuid4().hex}"
    maintenance = urlunsplit(dsn._replace(path="/postgres"))
    scratch = urlunsplit(dsn._replace(path=f"/{database}"))
    with psycopg.connect(maintenance, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    previous = os.environ.get("DYNAMISRAG_DATABASE_URL")
    engine = create_engine(to_sqlalchemy_url(PostgresDsn(scratch)))
    try:
        os.environ["DYNAMISRAG_DATABASE_URL"] = scratch
        config = Config(str(ROOT / "alembic.ini"))
        config.set_main_option("script_location", str(ROOT / "alembic"))
        command.upgrade(config, "head")
        yield engine, config
    finally:
        if previous is None:
            os.environ.pop("DYNAMISRAG_DATABASE_URL", None)
        else:
            os.environ["DYNAMISRAG_DATABASE_URL"] = previous
        engine.dispose()
        with psycopg.connect(maintenance, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(database)))


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
        rows = [record.artifact, record.document, *record.identifiers, record.version]
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
            (type(row).__table__, {c.name: getattr(row, c.name) for c in row.__table__.columns})
            for row in rows
        ]
        version_key, span_key, artifact_hash = (
            record.version.version_key,
            span.span_key,
            record.artifact.content_sha256,
        )
    with scratch_database() as (engine, config):
        with Session(engine) as session, session.begin():
            for table, values in copies:
                session.execute(table.insert().values(**values))
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
