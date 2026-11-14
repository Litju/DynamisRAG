"""Canonical citation-highlight mapping tests (RES-638)."""

from __future__ import annotations

import hashlib
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from dynamisrag.db.canonical import PassageProjectionRecords, list_passage_projection_records
from dynamisrag.db.models import PassageSourceSpanRecord
from dynamisrag.domain.identity import passage_source_span_key
from dynamisrag.evidence.inspection import HighlightMappingError, locate_verified_quote
from dynamisrag.search.scientific import CHUNKER
from tests.unit.test_agent_tools import (
    _create_sqlite_schema,  # pyright: ignore[reportPrivateUsage]
    _seed_study,  # pyright: ignore[reportPrivateUsage]
)

_FIRST = (
    "The pilot jump height increased by 14.2% in a small subsample, and the same "
    "jump height increased again in the second block."
)
_SECOND = "After training, the intervention jump height increased by 14.2% compared with control."


def _multi_span_engine() -> tuple[Engine, str]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    _create_sqlite_schema(engine)
    with Session(engine) as session:
        _seed_study(
            session,
            pmcid="PMC9000638",
            doi="10.0000/res638.highlight",
            pmid="90000638",
            title="Repeated Quotation Highlight Fixture",
            section_title="Results",
            paragraphs_text=(_FIRST, _SECOND),
        )
        session.commit()

    with Session(engine) as session:
        records = list_passage_projection_records(
            session, chunker_revision=CHUNKER, passage_keys=None
        )
        first_passage = next(r for r in records if r.passage.ordinal == 0)
        second_paragraph = (
            next(r for r in records if r.passage.ordinal == 1).source_spans[0].paragraph
        )
        span_key = passage_source_span_key(first_passage.passage.passage_key, 1)
        session.add(
            PassageSourceSpanRecord(
                id=uuid4(),
                document_version_id=first_passage.version.id,
                passage_id=first_passage.passage.id,
                passage_document_version_id=first_passage.version.id,
                paragraph_id=second_paragraph.id,
                paragraph_document_version_id=first_passage.version.id,
                source_order=1,
                start_char=0,
                end_char=len(second_paragraph.text),
                span_key=span_key,
            )
        )
        merged = f"{_FIRST}\n\n{_SECOND}"
        first_passage.passage.text = merged
        first_passage.passage.content_sha256 = hashlib.sha256(merged.encode()).hexdigest()
        session.commit()
    return engine, span_key


def _records(engine: Engine) -> tuple[PassageProjectionRecords, ...]:
    with Session(engine) as session:
        return tuple(
            list_passage_projection_records(session, chunker_revision=CHUNKER, passage_keys=None)
        )


def test_repeated_quote_maps_to_the_stored_paragraph_span() -> None:
    engine, second_span_key = _multi_span_engine()
    record = next(r for r in _records(engine) if r.passage.ordinal == 0)
    quote = "jump height increased by 14.2%"
    start = _SECOND.index(quote)
    mapping = locate_verified_quote(
        record,
        span_key=second_span_key,
        start_char=start,
        end_char=start + len(quote),
        quote=quote,
    )
    assert mapping.passage_start_char == len(_FIRST) + 2 + start
    assert record.passage.text[mapping.passage_start_char : mapping.passage_end_char] == quote
    # The first textual occurrence is in the first paragraph; the mapping must not use it.
    assert mapping.passage_start_char > record.passage.text.index(quote)


def test_wrong_offsets_never_fall_back_to_another_occurrence() -> None:
    engine, second_span_key = _multi_span_engine()
    record = next(r for r in _records(engine) if r.passage.ordinal == 0)
    quote = "jump height increased by 14.2%"
    start = _SECOND.index(quote)
    with pytest.raises(HighlightMappingError) as error:
        locate_verified_quote(
            record,
            span_key=second_span_key,
            start_char=start + 1,
            end_char=start + 1 + len(quote),
            quote=quote,
        )
    assert error.value.reason == "span_coordinate_mismatch"


def test_missing_span_and_incomplete_coordinates_fail_closed() -> None:
    engine, _ = _multi_span_engine()
    record = next(r for r in _records(engine) if r.passage.ordinal == 0)
    quote = "jump height increased by 14.2%"
    with pytest.raises(HighlightMappingError) as missing:
        locate_verified_quote(
            record,
            span_key="f" * 64,
            start_char=0,
            end_char=len(quote),
            quote=quote,
        )
    assert missing.value.reason == "span_not_found"
    with pytest.raises(HighlightMappingError) as incomplete:
        locate_verified_quote(record, span_key=None, start_char=0, end_char=1, quote="x")
    assert incomplete.value.reason == "incomplete_request"


def test_quote_outside_the_named_span_is_rejected() -> None:
    engine, _ = _multi_span_engine()
    record = next(r for r in _records(engine) if r.passage.ordinal == 0)
    first_span_key = record.source_spans[0].span.span_key
    quote = "jump height increased by 14.2%"
    start = _SECOND.index(quote)
    with pytest.raises(HighlightMappingError) as error:
        locate_verified_quote(
            record,
            span_key=first_span_key,
            start_char=len(_FIRST) + 2 + start,
            end_char=len(_FIRST) + 2 + start + len(quote),
            quote=quote,
        )
    assert error.value.reason in {"span_coordinate_mismatch", "quote_outside_span"}


def test_untrimmed_or_oversized_quotes_are_rejected() -> None:
    engine, second_span_key = _multi_span_engine()
    record = next(r for r in _records(engine) if r.passage.ordinal == 0)
    quote = "jump height increased by 14.2%"
    start = _SECOND.index(quote)
    with pytest.raises(HighlightMappingError) as error:
        locate_verified_quote(
            record,
            span_key=second_span_key,
            start_char=start,
            end_char=start + len(quote),
            quote=f" {quote}",
        )
    assert error.value.reason == "invalid_quote"
    with pytest.raises(HighlightMappingError) as oversized:
        locate_verified_quote(
            record,
            span_key=second_span_key,
            start_char=0,
            end_char=1000,
            quote="x" * 1000,
        )
    assert oversized.value.reason in {"invalid_coordinates", "invalid_quote"}
