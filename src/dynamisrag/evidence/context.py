"""Bounded whole-paragraph context; offsets always address persisted Paragraph.text."""

from collections import Counter
from hashlib import sha256

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from dynamisrag.db.canonical import PassageProjectionRecords, section_from_record
from dynamisrag.db.models import ParagraphRecord, SectionRecord
from dynamisrag.domain.evidence import InvalidProvenanceError
from dynamisrag.domain.identity import paragraph_key
from dynamisrag.evidence.contracts import ContextExclusion, ContextReference, SelectionRequest


def _ancestors(
    session: Session,
    record: PassageProjectionRecords,
    depth: int,
) -> list[SectionRecord]:
    if record.section is None:
        return []
    result = [record.section]
    for _ in range(depth):
        section = result[-1]
        if section.parent_section_id is None:
            break
        parent = session.scalar(
            select(SectionRecord).where(
                SectionRecord.id == section.parent_section_id,
                SectionRecord.document_version_id == record.version.id,
            )
        )
        if parent is None:
            raise InvalidProvenanceError("missing or cross-version parent section")
        result.append(parent)
    for section in result:
        if (
            section.document_version_id != record.version.id
            or section_from_record(section, version_key=record.version.version_key).section_key
            != section.section_key
        ):
            raise InvalidProvenanceError("context section identity mismatch")
    return result


def expand_context(
    session: Session,
    request: SelectionRequest,
    records: tuple[PassageProjectionRecords, ...],
    selected_bytes: int,
    *,
    row_limit: int,
) -> tuple[tuple[ContextReference, ...], tuple[ContextExclusion, ...]]:
    """Open siblings in the source section and bounded ancestor-section paragraphs."""
    policy = request.policy
    context: list[ContextReference] = []
    exclusions: list[ContextExclusion] = []
    opened = {s.paragraph.paragraph_key for r in records for s in r.source_spans}
    document_bytes: Counter[str] = Counter()
    for record in records:
        document_bytes[record.document.canonical_key] += len(record.passage.text.encode())
    expanded, rows = 0, 0
    if request.depth == "passages" or policy.max_expansions == 0:
        return (), ()
    for record in records:
        if record.section is None or any(s.paragraph.region != "body" for s in record.source_spans):
            exclusions.append("missing_section" if record.section is None else "boundary")
            continue
        if rows + policy.parent_depth > row_limit:
            exclusions.append("row_budget")
            break
        ancestors = _ancestors(session, record, policy.parent_depth)
        rows += len(ancestors) - 1
        sections = {s.id: s for s in ancestors}
        ordinals = [s.paragraph.ordinal for s in record.source_spans]
        low, high = min(ordinals), max(ordinals)
        if rows >= row_limit:
            exclusions.append("row_budget")
            break
        paragraphs = session.scalars(
            select(ParagraphRecord)
            .where(
                ParagraphRecord.document_version_id == record.version.id,
                ParagraphRecord.section_id.in_(tuple(sections)),
                func.octet_length(ParagraphRecord.text) <= policy.max_row_bytes,
                ParagraphRecord.ordinal
                >= low - min(policy.sibling_distance, policy.max_document_span),
                ParagraphRecord.ordinal
                <= min(high + policy.sibling_distance, low + policy.max_document_span),
            )
            .order_by(ParagraphRecord.ordinal, ParagraphRecord.paragraph_key)
            .limit(row_limit - rows)
        ).all()
        rows += len(paragraphs)
        exclusions.extend(("row_budget",) if rows == row_limit else ())
        available = {p.ordinal for p in paragraphs}
        if any(
            i not in available
            for i in (low - policy.sibling_distance, high + policy.sibling_distance)
        ):
            exclusions.append("missing_sibling")
        for p in paragraphs:
            if p.paragraph_key in opened:
                continue
            if p.region != "body":
                exclusions.append("boundary")
                continue
            size = len(p.text.encode("utf-8"))
            if (
                p.paragraph_key != paragraph_key(record.version.version_key, p.source_anchor)
                or sha256(p.text.encode()).hexdigest() != p.content_sha256
            ):
                raise InvalidProvenanceError("context paragraph identity mismatch")
            if (
                len(context) >= policy.max_expansions
                or size > policy.max_row_bytes
                or selected_bytes + expanded + size > policy.max_bytes
                or expanded + size > policy.max_expansion_bytes
                or document_bytes[record.document.canonical_key] + size > policy.max_document_bytes
            ):
                exclusions.append("context_budget")
                continue
            # SQL section membership is restricted to this verified ancestor map.
            owner = sections[p.section_id]  # type: ignore[index]
            context.append(
                ContextReference(
                    selected_passage_key=record.passage.passage_key,
                    version_key=record.version.version_key,
                    paragraph_key=p.paragraph_key,
                    paragraph_sha256=p.content_sha256,
                    section_key=owner.section_key,
                    end=len(p.text),
                    utf8_bytes=size,
                    kind="sibling" if p.section_id == record.section.id else "parent",
                )
            )
            opened.add(p.paragraph_key)
            expanded += size
            document_bytes[record.document.canonical_key] += size
    return tuple(context), tuple(exclusions)
