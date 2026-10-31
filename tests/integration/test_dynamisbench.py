"""Independent PostgreSQL provenance audit; rollback-only synthetic fixture, no inference."""

import pytest
from sqlalchemy.orm import Session

from dynamisrag.benchmark.dynamisbench_runner import provenance_digest
from dynamisrag.config import Settings
from dynamisrag.db import create_database_engine
from dynamisrag.db.canonical import list_passage_projection_records
from dynamisrag.search.projection import build_projection_manifest
from dynamisrag.search.qdrant import Point
from dynamisrag.search.retrieval import PassageProvenance
from dynamisrag.search.scientific import CHUNKER, ScientificUnavailable, verify_canonical
from tests.unit.test_scientific_retrieval import canonical_records

pytestmark = pytest.mark.integration


def test_benchmark_digest_bound_to_canonical_citation(live_settings: Settings) -> None:
    engine = create_database_engine(live_settings)
    record = canonical_records()[0]
    try:
        with Session(engine) as session, session.begin():
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
            records = tuple(
                list_passage_projection_records(
                    session,
                    chunker_revision=CHUNKER,
                    passage_keys=(record.passage.passage_key,),
                )
            )
            document = build_projection_manifest(records, chunker_revision=CHUNKER).documents[0]
            payload = document.payload()
            source = PassageProvenance.model_validate(
                {k: v for k, v in payload.items() if k in PassageProvenance.model_fields}
            )
            verify_canonical(records, (Point(id="synthetic", payload=dict(payload)),))
            original = provenance_digest(source)
            assert original == provenance_digest(
                PassageProvenance.model_validate_json(source.model_dump_json())
            )
            assert original != provenance_digest(source.model_copy(update={"doi": "10.1/wrong"}))
            records[0].source_spans[0].span.end_char -= 1
            with pytest.raises(ScientificUnavailable):
                verify_canonical(records, (Point(id="synthetic", payload=dict(payload)),))
            session.rollback()
    finally:
        engine.dispose()
