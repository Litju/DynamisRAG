"""Controlled context geometry and paragraph coordinates; source prose is inert data."""

from dataclasses import replace
from hashlib import sha256
from unittest.mock import Mock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from dynamisrag.db.canonical import PassageProjectionRecords
from dynamisrag.db.models import ParagraphRecord, SectionRecord
from dynamisrag.domain.evidence import InvalidProvenanceError
from dynamisrag.domain.identity import paragraph_key, section_key
from dynamisrag.evidence.context import expand_context
from tests._support import passage_projection_corpus, passage_projection_records
from tests.unit.test_evidence_selection import request_for, response_fixture


def source() -> PassageProjectionRecords:
    record = passage_projection_records(passage_projection_corpus())[0]
    assert record.section is not None
    record.version.version_key = "b" * 64
    record.passage.passage_key = "a" * 64
    record.section.section_key = section_key(
        record.version.version_key, record.section.structural_path
    )
    record.section.parent_section_id = None
    return record


def paragraph(record: PassageProjectionRecords) -> ParagraphRecord:
    value = "Synthetic numerical finding: 20 \u00b1 2 kg; conditions \u03b1\u03b2."
    assert record.section is not None
    return ParagraphRecord(
        id=uuid4(),
        document_version_id=record.version.id,
        section_id=record.section.id,
        section_document_version_id=record.version.id,
        ordinal=1,
        region="body",
        source_anchor="synthetic:/context[1]",
        text=value,
        content_sha256=sha256(value.encode()).hexdigest(),
        paragraph_key=paragraph_key(record.version.version_key, "synthetic:/context[1]"),
    )


def test_unicode_whole_context_and_no_invented_ranking() -> None:
    record = source()
    p = paragraph(record)
    session = Mock(spec=Session)
    session.scalars.return_value.all.return_value = [p]
    request = request_for(response_fixture(), parent_depth=0)
    context, _ = expand_context(session, request, (record,), 0, row_limit=10)
    assert len(context) == 1
    assert context[0].end == len(p.text)
    assert context[0].utf8_bytes == len(p.text.encode()) > len(p.text)
    assert context[0].coordinate_system == "canonical-paragraph-python-chars-half-open"
    assert p.text not in context[0].model_dump_json()
    assert "rank" not in context[0].model_dump()
    statement = session.scalars.call_args.args[0]
    assert "document_version_id" in str(statement)
    assert "section_id" in str(statement)
    assert "LIMIT" in str(statement)


@pytest.mark.parametrize("mutation", ["digest", "key"])
def test_corrupt_context_rejected(mutation: str) -> None:
    record = source()
    p = paragraph(record)
    setattr(p, "content_sha256" if mutation == "digest" else "paragraph_key", "f" * 64)
    session = Mock(spec=Session)
    session.scalars.return_value.all.return_value = [p]
    with pytest.raises(InvalidProvenanceError):
        expand_context(
            session, request_for(response_fixture(), parent_depth=0), (record,), 0, row_limit=10
        )


@pytest.mark.parametrize(
    "budget", [{"max_bytes": 1}, {"max_expansion_bytes": 1}, {"max_document_bytes": 1}]
)
def test_context_budget_excludes_complete_numeric_paragraph(budget: dict[str, int]) -> None:
    record = source()
    session = Mock(spec=Session)
    session.scalars.return_value.all.return_value = [paragraph(record)]
    context, reasons = expand_context(
        session,
        request_for(response_fixture(), parent_depth=0, **budget),
        (record,),
        0,
        row_limit=10,
    )
    assert context == ()
    assert "context_budget" in reasons


def test_overlap_and_front_matter_are_not_reopened() -> None:
    record = source()
    p = paragraph(record)
    p.region = "back"
    session = Mock(spec=Session)
    session.scalars.return_value.all.return_value = [record.source_spans[0].paragraph, p]
    context, reasons = expand_context(
        session, request_for(response_fixture(), parent_depth=0), (record,), 0, row_limit=10
    )
    assert context == ()
    assert "boundary" in reasons


def test_missing_section_and_sibling_and_row_bound() -> None:
    record = source()
    session = Mock(spec=Session)
    request = request_for(response_fixture(), parent_depth=0)
    assert expand_context(session, request, (replace(record, section=None),), 0, row_limit=10) == (
        (),
        ("missing_section",),
    )
    session.scalars.return_value.all.return_value = []
    assert "missing_sibling" in expand_context(session, request, (record,), 0, row_limit=10)[1]
    assert "row_budget" in expand_context(session, request, (record,), 0, row_limit=0)[1]


@pytest.mark.parametrize("mutation", ["missing", "version", "key"])
def test_parent_integrity(mutation: str) -> None:
    record = source()
    assert record.section is not None
    record.section.parent_section_id = uuid4()
    parent = SectionRecord(
        id=record.section.parent_section_id,
        document_version_id=uuid4() if mutation == "version" else record.version.id,
        structural_path="1",
        ordinal=0,
        depth=0,
        section_key="f" * 64,
    )
    session = Mock(spec=Session)
    session.scalar.return_value = None if mutation == "missing" else parent
    with pytest.raises(InvalidProvenanceError):
        expand_context(session, request_for(response_fixture()), (record,), 0, row_limit=10)
