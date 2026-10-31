"""Adversarial evidence graph tests in an ephemeral PostgreSQL database only."""

from collections.abc import Iterator
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from psycopg import sql
from sqlalchemy import Engine, delete, event, func, insert, select, text, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from dynamisrag.chunking.config import ChunkerConfig
from dynamisrag.chunking.materializer import PassageMaterializer
from dynamisrag.chunking.planner import StructureAwareChunker
from dynamisrag.config import load_settings
from dynamisrag.db import create_database_engine
from dynamisrag.db.evidence import canonical_member, get_graph, persist_graph
from dynamisrag.db.evidence_models import (
    ClaimRecord,
    EvidenceLinkRecord,
    EvidenceMembershipRecord,
    EvidenceSetRecord,
)
from dynamisrag.db.models import PassageRecord, PassageSourceSpanRecord
from dynamisrag.domain.evidence import (
    EvidenceGraph,
    EvidenceMember,
    InvalidProvenanceError,
    ReviewState,
)
from dynamisrag.domain.identity import evidence_identity
from tests._support import ALEMBIC_INI, REPO_ROOT, build_settings
from tests.evidence_support import seed_source
from tests.unit.test_evidence_domain import graph

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def evidence_engine() -> Iterator[Engine]:
    """Never migrate or write the populated reference database."""
    dsn = urlsplit(str(load_settings().database_url))
    database = f"res145_test_{uuid4().hex}"
    maintenance = urlunsplit(dsn._replace(path="/postgres"))
    scratch = urlunsplit(dsn._replace(path=f"/{database}"))
    with psycopg.connect(maintenance, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    engine = create_database_engine(build_settings(database_url=scratch))
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("DYNAMISRAG_DATABASE_URL", scratch)
            config = Config(str(ALEMBIC_INI))
            config.set_main_option("script_location", str(REPO_ROOT / "alembic"))
            command.upgrade(config, "head")
            with Session(engine) as session, session.begin():
                version, sections, paragraphs = seed_source(session)
                chunker = StructureAwareChunker(
                    ChunkerConfig(target_tokens=5, max_tokens=10, min_tokens=1)
                )
                PassageMaterializer(session).materialize(
                    version, chunker.plan(version, sections, paragraphs)
                )
            yield engine
    finally:
        engine.dispose()
        with psycopg.connect(maintenance, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(database)))


@pytest.fixture
def session(evidence_engine: Engine) -> Iterator[Session]:
    with Session(evidence_engine) as session, session.begin():
        yield session
        session.rollback()


def candidate(session: Session) -> EvidenceGraph:
    passage = session.scalars(select(PassageRecord).order_by(PassageRecord.ordinal)).first()
    assert passage is not None
    version_key = session.scalar(
        text("SELECT version_key FROM document_version WHERE id=:id"),
        {"id": passage.document_version_id},
    )
    assert isinstance(version_key, str)
    return graph(canonical_member(session, passage.passage_key, version_key))


def counts(session: Session) -> tuple[int, ...]:
    return tuple(
        session.scalar(select(func.count()).select_from(model)) or 0
        for model in (EvidenceSetRecord, ClaimRecord, EvidenceMembershipRecord, EvidenceLinkRecord)
    )


def test_idempotent_reconstruction_before_and_after_commit(evidence_engine: Engine) -> None:
    with Session(evidence_engine) as session, session.begin():
        original = candidate(session)
        first = persist_graph(session, original)
        assert persist_graph(session, original) == first
        assert first == original
    with Session(evidence_engine) as session:
        assert get_graph(session, original.evidence_set.key) == original
        assert counts(session) == (1, 1, 1, 1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("passage_key", "f" * 64),
        ("version_key", "f" * 64),
        ("provenance_sha256", "f" * 64),
        ("span_keys", ("f" * 64,)),
    ],
)
def test_forged_member_rejected_atomically(session: Session, field: str, value: object) -> None:
    original = candidate(session)
    member = EvidenceMember.model_validate(
        original.evidence_set.members[0].model_dump() | {field: value}
    )
    forged = graph(member)
    before = counts(session)
    with pytest.raises(InvalidProvenanceError):
        persist_graph(session, forged)
    assert counts(session) == before


def test_interrupted_publication_and_repeated_recovery(session: Session) -> None:

    original = candidate(session)
    # Give this graph a distinct identity from the committed idempotency fixture.
    payload = original.model_dump(mode="json")
    payload["evidence_set"]["question_id"] = "interruption"
    updated_set = type(original.evidence_set).model_validate(payload["evidence_set"])
    payload["links"][0]["evidence_set_key"] = updated_set.key
    original = EvidenceGraph.model_validate(payload)
    before = counts(session)

    def fail_on_link(*args: Any) -> None:
        if str(args[2]).startswith("INSERT INTO evidence_link"):
            raise RuntimeError("interrupted before link")

    engine = session.get_bind()
    event.listen(engine, "before_cursor_execute", fail_on_link)
    try:
        with pytest.raises(RuntimeError, match="interrupted"):
            persist_graph(session, original)
        assert counts(session) == before
    finally:
        event.remove(engine, "before_cursor_execute", fail_on_link)
    assert persist_graph(session, original) == original
    assert persist_graph(session, original) == original


@pytest.mark.parametrize(
    "mutation", ["passage", "version", "span", "digest", "approval", "relation", "key"]
)
def test_direct_database_link_forgery_fails(session: Session, mutation: str) -> None:
    original = candidate(session)
    persist_graph(session, original)
    record = session.scalars(select(EvidenceLinkRecord)).first()
    assert record is not None
    payload = dict(record.payload)
    values = {
        "claim_key": record.claim_key,
        "set_key": record.set_key,
        "passage_key": record.passage_key,
        "span_id": record.span_id,
    }
    if mutation in ("passage", "version", "digest"):
        member = original.links[0].member.model_dump(mode="json")
        member[
            {"passage": "passage_key", "version": "version_key", "digest": "provenance_sha256"}[
                mutation
            ]
        ] = "f" * 64
        payload["member"] = member
    elif mutation == "approval":
        payload["review_state"] = "REVIEWED"
    elif mutation == "relation":
        payload["relation"] = "PROVES"
    elif mutation == "span":
        membership = session.get(EvidenceMembershipRecord, (record.set_key, record.passage_key))
        assert membership is not None
        other = session.scalars(
            select(PassageSourceSpanRecord).where(
                PassageSourceSpanRecord.passage_id != membership.passage_id
            )
        ).first()
        assert other is not None
        values["span_id"] = other.id
        payload["span_key"] = other.span_key
    key = evidence_identity("evidence-link-v1", payload) if mutation != "key" else "f" * 64
    with pytest.raises(DBAPIError), session.begin_nested():
        session.add(EvidenceLinkRecord(key=key, payload=payload, **values))
        session.flush()


@pytest.mark.parametrize("table", ["evidence_set", "claim", "evidence_membership", "evidence_link"])
@pytest.mark.parametrize("operation", ["UPDATE", "DELETE"])
def test_append_only_at_database_boundary(session: Session, table: str, operation: str) -> None:
    persist_graph(session, candidate(session))
    model = {
        "evidence_set": EvidenceSetRecord,
        "claim": ClaimRecord,
        "evidence_membership": EvidenceMembershipRecord,
        "evidence_link": EvidenceLinkRecord,
    }[table]
    statement = (
        update(model).values(payload=model.payload) if operation == "UPDATE" else delete(model)
    )
    with pytest.raises(DBAPIError), session.begin_nested():
        session.execute(statement)


def test_unsigned_claim_and_incomplete_set_cannot_commit(session: Session) -> None:
    original = candidate(session)
    payload = original.claims[0].model_dump(mode="json") | {"review_state": "REVIEWED"}
    with pytest.raises(DBAPIError), session.begin_nested():
        session.add(ClaimRecord(key=evidence_identity("claim-v1", payload), payload=payload))
        session.flush()
    payload = original.evidence_set.model_dump(mode="json") | {"question_id": "incomplete"}
    with pytest.raises(DBAPIError), session.begin_nested():
        session.add(
            EvidenceSetRecord(key=evidence_identity("evidence-set-v1", payload), payload=payload)
        )
        session.flush()
        session.execute(text("SET CONSTRAINTS trg_evidence_set_complete IMMEDIATE"))


@pytest.mark.parametrize("mutation", ["passage", "version", "digest", "span", "ordinal"])
def test_direct_membership_provenance_forgery(session: Session, mutation: str) -> None:
    original = candidate(session)
    member = original.evidence_set.members[0].model_dump(mode="json")
    if mutation in ("passage", "version", "digest"):
        field = {"passage": "passage_key", "version": "version_key", "digest": "provenance_sha256"}[
            mutation
        ]
        member[field] = "f" * 64
    elif mutation == "span":
        member["span_keys"] = ["f" * 64]
    payload = original.evidence_set.model_dump(mode="json") | {
        "question_id": f"forged-{mutation}",
        "members": [member],
    }
    key = evidence_identity("evidence-set-v1", payload)
    passage = session.scalars(select(PassageRecord).order_by(PassageRecord.ordinal)).first()
    assert passage is not None
    with pytest.raises(DBAPIError), session.begin_nested():
        session.add(EvidenceSetRecord(key=key, payload=payload))
        session.flush()
        session.add(
            EvidenceMembershipRecord(
                set_key=key,
                passage_key=str(member["passage_key"]),
                passage_id=passage.id,
                version_id=passage.document_version_id,
                ordinal=1 if mutation == "ordinal" else 0,
                payload=member,
            )
        )
        session.flush()


@pytest.mark.parametrize("mutation", ["end", "start", "hash", "span_key", "version"])
def test_python_source_mutations_fail_closed(session: Session, mutation: str) -> None:
    from dynamisrag.db.canonical import list_passage_projection_records
    from dynamisrag.db.evidence import source_fingerprint

    original = candidate(session)
    passage = session.scalars(
        select(PassageRecord).where(
            PassageRecord.passage_key == original.evidence_set.members[0].passage_key
        )
    ).one()
    record = list_passage_projection_records(
        session, chunker_revision=passage.chunker_revision, passage_keys=(passage.passage_key,)
    )[0]
    span = record.source_spans[0].span
    if mutation == "end":
        span.end_char = len(record.source_spans[0].paragraph.text) + 1
    elif mutation == "start":
        span.start_char += 1
    elif mutation == "hash":
        record.source_spans[0].paragraph.content_sha256 = "f" * 64
    elif mutation == "span_key":
        span.span_key = "f" * 64
    else:
        record.version.version_key = "f" * 64
    with pytest.raises(InvalidProvenanceError):
        source_fingerprint(record)
    session.expire_all()  # Discard in-memory mutation; never flush canonical changes.


def test_invalid_source_offsets_at_database_boundary(session: Session) -> None:
    span = session.scalars(select(PassageSourceSpanRecord)).first()
    assert span is not None
    values = {c.name: getattr(span, c.name) for c in span.__table__.columns}
    values.update(id=uuid4(), source_order=999, span_key="f" * 64, end_char=10**6)
    with pytest.raises(DBAPIError), session.begin_nested():
        session.execute(insert(PassageSourceSpanRecord).values(**values))


def test_constructor_bypass_cannot_promote_proposed_support(session: Session) -> None:
    original = candidate(session)
    forged_link = original.links[0].model_copy(update={"review_state": ReviewState.REVIEWED})
    forged = original.model_copy(update={"links": (forged_link,)})
    before = counts(session)
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="authenticated"):
        persist_graph(session, forged)
    assert counts(session) == before


def test_concurrent_identical_publication(evidence_engine: Engine) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    with Session(evidence_engine) as session:
        original = candidate(session)
    payload = original.model_dump(mode="json")
    payload["evidence_set"]["question_id"] = "concurrent-identical-publication"
    from dynamisrag.domain.evidence import EvidenceSet

    evidence_set = EvidenceSet.model_validate(payload["evidence_set"])
    payload["links"][0]["evidence_set_key"] = evidence_set.key
    original = EvidenceGraph.model_validate(payload)
    barrier = Barrier(2)

    def publish() -> EvidenceGraph:
        with Session(evidence_engine) as session, session.begin():
            barrier.wait(timeout=10)
            return persist_graph(session, original)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(publish) for _ in range(2)]
        assert [f.result(timeout=20) for f in futures] == [original, original]


def test_migration_downgrade_preserves_populated_sources(evidence_engine: Engine) -> None:
    from dynamisrag.db.models import DocumentVersionRecord, ParagraphRecord

    models = (DocumentVersionRecord, ParagraphRecord, PassageRecord, PassageSourceSpanRecord)

    def snapshot() -> list[object]:
        with evidence_engine.connect() as connection:
            return [connection.execute(select(m.__table__).order_by(m.id)).all() for m in models]

    before = snapshot()
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    command.downgrade(config, "0004_passage_source_spans")
    assert snapshot() == before
    command.upgrade(config, "head")
    assert snapshot() == before


def test_disagreement_and_existing_claim_are_preserved(session: Session) -> None:
    from dynamisrag.domain.evidence import EvidenceLink, Relation

    original = candidate(session)
    first = persist_graph(session, original)
    opposed = EvidenceLink.model_validate(
        original.links[0].model_dump()
        | {
            "relation": Relation.CONTRADICTS,
            "rationale": "Provisional opposing synthetic interpretation",
        }
    )
    extended = original.model_copy(update={"links": (original.links[0], opposed)})
    rebuilt = persist_graph(session, extended)
    assert {link.relation for link in rebuilt.links} == {Relation.CONTEXTUAL, Relation.CONTRADICTS}
    assert rebuilt.claims == first.claims
    assert counts(session)[1] == 1


def test_cross_version_source_binding_fails(session: Session) -> None:
    from dynamisrag.db.models import DocumentRecord, DocumentVersionRecord, SourceArtifactRecord
    from dynamisrag.domain.identity import document_version_key

    original = candidate(session)
    passage = session.scalars(select(PassageRecord).order_by(PassageRecord.ordinal)).first()
    assert passage is not None
    version = session.get(DocumentVersionRecord, passage.document_version_id)
    assert version is not None
    values = {c.name: getattr(version, c.name) for c in version.__table__.columns}
    # A legitimate second immutable version of this document cannot own the first version's passage.
    document = session.get(DocumentRecord, version.document_id)
    artifact = session.get(SourceArtifactRecord, version.source_artifact_id)
    assert document is not None and artifact is not None
    key = document_version_key(
        document.canonical_key,
        artifact.artifact_key,
        "test-other-version",
        version.normalizer_revision,
        version.content_fingerprint,
    )
    values.update(id=uuid4(), version_key=key, parser_revision="test-other-version")
    session.execute(insert(DocumentVersionRecord).values(**values))
    altered = original.evidence_set.members[0].model_copy(update={"version_key": key})
    with pytest.raises(InvalidProvenanceError):
        persist_graph(session, graph(altered))

    payload = original.evidence_set.model_dump(mode="json") | {
        "question_id": "wrong-version-binding"
    }
    set_key = evidence_identity("evidence-set-v1", payload)
    with pytest.raises(DBAPIError), session.begin_nested():
        session.add(EvidenceSetRecord(key=set_key, payload=payload))
        session.flush()
        session.add(
            EvidenceMembershipRecord(
                set_key=set_key,
                passage_key=passage.passage_key,
                ordinal=0,
                passage_id=passage.id,
                version_id=values["id"],
                payload=original.evidence_set.members[0].model_dump(mode="json"),
            )
        )
        session.flush()


def test_existing_forged_canonical_row_is_not_trusted(session: Session) -> None:
    original = candidate(session)
    passage = session.scalars(select(PassageRecord).order_by(PassageRecord.ordinal)).first()
    assert passage is not None
    values = {c.name: getattr(passage, c.name) for c in passage.__table__.columns}
    values.update(id=uuid4(), ordinal=1000, passage_key="f" * 64)
    session.execute(insert(PassageRecord).values(**values))
    # Existence and an internally consistent payload are not sufficient provenance.
    with pytest.raises(InvalidProvenanceError), session.begin_nested():
        canonical_member(session, "f" * 64, original.evidence_set.members[0].version_key)


@pytest.mark.parametrize("value", ["\u03b2", "\U0001f9ec", "e\u0301", 'a\n\t"b', "a\x1fb"])
def test_sql_python_identity_serialization_agree(session: Session, value: str) -> None:
    from sqlalchemy import bindparam
    from sqlalchemy.dialects.postgresql import JSONB

    payload = candidate(session).claims[0].model_dump(mode="json") | {"proposition": value}
    statement = text("SELECT evidence_key('claim-v1', :payload)").bindparams(
        bindparam("payload", type_=JSONB)
    )
    assert session.scalar(statement, {"payload": payload}) == evidence_identity("claim-v1", payload)
