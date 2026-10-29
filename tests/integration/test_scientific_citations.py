"""Real PostgreSQL citation verification using a rolled-back synthetic source fixture."""

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from dynamisrag.config import Settings
from dynamisrag.db import create_database_engine
from dynamisrag.db.canonical import PassageProjectionRecords, list_passage_projection_records
from dynamisrag.search.projection import build_projection_manifest
from dynamisrag.search.qdrant import Point
from dynamisrag.search.scientific import CHUNKER, ScientificUnavailable, verify_canonical
from tests.unit.test_scientific_retrieval import canonical_records

pytestmark = pytest.mark.integration


def test_scoped_sql_projection_and_citation_integrity(
    live_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = create_database_engine(live_settings)
    record = canonical_records()[0]
    try:
        with Session(engine) as session:
            transaction = session.begin()
            try:
                for row in (
                    record.artifact,
                    record.document,
                    record.version,
                    record.section,
                    record.source_spans[0].paragraph,
                    record.passage,
                    record.source_spans[0].span,
                    *record.identifiers,
                ):
                    if row is not None:
                        session.add(row)
                        session.flush()
                key = record.passage.passage_key
                scoped = tuple(
                    list_passage_projection_records(
                        session, chunker_revision=CHUNKER, passage_keys=(key,)
                    )
                )
                assert len(scoped) == 1
                assert key in {
                    r.passage.passage_key
                    for r in list_passage_projection_records(session, chunker_revision=CHUNKER)
                }
                assert not list_passage_projection_records(
                    session, chunker_revision=CHUNKER, passage_keys=()
                )
                assert not list_passage_projection_records(
                    session, chunker_revision=CHUNKER, passage_keys=("0" * 64,)
                )
                document = build_projection_manifest(scoped, chunker_revision=CHUNKER).documents[0]
                point = Point(id="synthetic-integration", payload=dict(document.payload()))
                verify_canonical(scoped, (point,))
                from dynamisrag.generation import scientific as answers
                from dynamisrag.generation.ollama import GenerationUnavailableError
                from dynamisrag.search.retrieval import PassageProvenance

                provenance = PassageProvenance.model_validate(
                    {
                        k: v
                        for k, v in document.payload().items()
                        if k in PassageProvenance.model_fields
                    }
                )

                def scoped_read(
                    _engine: Engine, keys: tuple[str, ...]
                ) -> tuple[PassageProjectionRecords, ...]:
                    return tuple(
                        list_passage_projection_records(
                            session, chunker_revision=CHUNKER, passage_keys=keys
                        )
                    )

                monkeypatch.setattr(answers, "read_canonical", scoped_read)
                answers.verify_evidence(engine, (provenance,))
                with pytest.raises(GenerationUnavailableError, match="provenance_invalid"):
                    answers.verify_evidence(
                        engine, (provenance.model_copy(update={"doi": "10.1/fabricated"}),)
                    )
                scoped[0].source_spans[0].span.end_char -= 1
                with pytest.raises(ScientificUnavailable, match="canonical_inconsistent"):
                    verify_canonical(scoped, (point,))
            finally:
                transaction.rollback()
    finally:
        engine.dispose()
