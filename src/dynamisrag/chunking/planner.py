"""Structure-aware deterministic passage planning (RES-134).

The pure chunker: canonical semantic inputs in, :class:`PassageManifest`
out. It never opens DB sessions, reads clocks, generates UUIDs for
semantics, makes network calls or reads JATS XML — passage semantics are a
pure function of the canonical source structure, the algorithm revision and
the frozen chunker configuration.

Chunking operates only on the persisted canonical layer (DocumentVersion,
Section, Paragraph). Paragraphs are immutable source structure; passages are
derived retrieval structure. One paragraph may contribute to one passage or
several (when a long paragraph is sentence-split); one passage may contain
one paragraph or several adjacent paragraphs of the same structural group.
There is no blind overlap and no duplicated source text.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from uuid import UUID

from dynamisrag.chunking.config import (
    MANIFEST_SCHEMA_REVISION,
    ChunkerConfig,
    chunker_revision,
    config_sha256,
)
from dynamisrag.chunking.errors import ChunkingError
from dynamisrag.chunking.manifest import (
    ManifestPassage,
    ManifestSourceSpan,
    PassageManifest,
    SectionMetadata,
)
from dynamisrag.chunking.sentences import split_oversized_sentence, split_sentences
from dynamisrag.chunking.tokens import count_lexical_tokens
from dynamisrag.domain.contracts import DocumentVersion, Paragraph, Section
from dynamisrag.domain.identity import passage_key

__all__ = ["PlannedSourceSpan", "StructureAwareChunker"]


@dataclass(frozen=True)
class PlannedSourceSpan:
    """One exact source span of a planned passage, addressed semantically."""

    paragraph_key: str
    paragraph_source_anchor: str
    start_char: int
    end_char: int


class StructureAwareChunker:
    """Plans deterministic structure-aware passages over canonical structure."""

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self._config = config if config is not None else ChunkerConfig()

    @property
    def config(self) -> ChunkerConfig:
        return self._config

    @property
    def chunker_revision(self) -> str:
        return chunker_revision(self._config)

    def plan(
        self,
        version: DocumentVersion,
        sections: Sequence[Section],
        paragraphs: Sequence[Paragraph],
    ) -> PassageManifest:
        """Plan the deterministic passage manifest for one document version.

        Paragraphs are processed in canonical global ordinal order and grouped
        into contiguous structural runs; passages never cross a section or
        region boundary, never exceed ``max_tokens`` and never duplicate
        source text. Ordinals are zero-based document-order output.
        """
        self._validate_inputs(version, sections, paragraphs)
        section_map = {section.id: section for section in sections}
        planned: list[ManifestPassage] = []
        for group, group_section_id in self._group_paragraphs(paragraphs):
            section = section_map.get(group_section_id) if group_section_id is not None else None
            planned.extend(self._plan_group(group, section))
        passages = tuple(
            replace(
                passage,
                ordinal=ordinal,
                passage_key=passage_key(version.version_key, self.chunker_revision, ordinal),
            )
            for ordinal, passage in enumerate(planned)
        )
        return PassageManifest(
            schema_revision=MANIFEST_SCHEMA_REVISION,
            document_version_key=version.version_key,
            chunker_revision=self.chunker_revision,
            algorithm_revision=self._config.algorithm_revision,
            config_sha256=config_sha256(self._config),
            config=self._config,
            passages=passages,
        )

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(
        version: DocumentVersion,
        sections: Sequence[Section],
        paragraphs: Sequence[Paragraph],
    ) -> None:
        """Every semantic input must belong to the version being chunked, and
        every section-owned paragraph must resolve to one of the supplied
        same-version sections — a missing section is never silently
        downgraded to sectionless passage metadata."""
        section_ids = {section.id for section in sections}
        for section in sections:
            if section.version_key != version.version_key:
                raise ChunkingError(
                    f"section {section.section_key!r} belongs to version "
                    f"{section.version_key!r}, not {version.version_key!r}"
                )
        for paragraph in paragraphs:
            if paragraph.version_key != version.version_key:
                raise ChunkingError(
                    f"paragraph {paragraph.paragraph_key!r} belongs to version "
                    f"{paragraph.version_key!r}, not {version.version_key!r}"
                )
            if paragraph.section_id is not None and paragraph.section_id not in section_ids:
                raise ChunkingError(
                    f"paragraph {paragraph.paragraph_key!r} references section "
                    f"{paragraph.section_id} which is not among the supplied "
                    f"sections of version {version.version_key!r}"
                )

    # ------------------------------------------------------------------
    # Structural grouping
    # ------------------------------------------------------------------

    def _group_paragraphs(
        self, paragraphs: Sequence[Paragraph]
    ) -> list[tuple[list[Paragraph], UUID | None]]:
        """Split paragraphs into contiguous structural groups.

        A group boundary is crossed whenever the section changes or — for
        sectionless paragraphs — the region changes. Paragraphs inside one
        section pack together; a nested child section starts a new group;
        returning to a parent section later is a new contiguous group;
        front/body/back never merge; two different sections never merge.
        Source document order is preserved.
        """
        groups: list[tuple[list[Paragraph], UUID | None]] = []
        current_region: str | None = None
        current_section_id: UUID | None = None
        current_group: list[Paragraph] = []
        for paragraph in sorted(paragraphs, key=lambda item: item.ordinal):
            if (
                not current_group
                or paragraph.region.value != current_region
                or paragraph.section_id != current_section_id
            ):
                if current_group:
                    groups.append((current_group, current_section_id))
                current_group = [paragraph]
                current_region = paragraph.region.value
                current_section_id = paragraph.section_id
            else:
                current_group.append(paragraph)
        if current_group:
            groups.append((current_group, current_section_id))
        return groups

    # ------------------------------------------------------------------
    # Packing
    # ------------------------------------------------------------------

    def _plan_group(self, group: list[Paragraph], section: Section | None) -> list[ManifestPassage]:
        """Plan the passages of one contiguous structural group."""
        paragraph_map = {paragraph.paragraph_key: paragraph for paragraph in group}
        units = self._build_units(group)
        packed = self._pack_units(units)
        merged = self._merge_small_tails(packed)
        passages: list[ManifestPassage] = []
        for passage_units in merged:
            spans = tuple(
                ManifestSourceSpan(
                    source_order=order,
                    paragraph_key=unit.paragraph_key,
                    paragraph_source_anchor=unit.paragraph_source_anchor,
                    start_char=unit.start_char,
                    end_char=unit.end_char,
                )
                for order, (unit, _) in enumerate(passage_units)
            )
            text = self._build_text(paragraph_map, passage_units)
            passages.append(
                ManifestPassage(
                    ordinal=0,
                    passage_key="",
                    text=text,
                    content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    token_count=sum(tokens for _, tokens in passage_units),
                    primary_source_anchor=passage_units[0][0].paragraph_source_anchor,
                    section=SectionMetadata(
                        section_key=section.section_key if section is not None else None,
                        structural_path=section.structural_path if section is not None else None,
                        source_anchor=section.source_anchor if section is not None else None,
                        title=section.title if section is not None else None,
                    ),
                    source_spans=spans,
                )
            )
        return passages

    def _build_units(self, group: list[Paragraph]) -> list[tuple[PlannedSourceSpan, int]]:
        """Split the group's paragraphs into atomic packing units.

        A paragraph within ``max_tokens`` is one whole-paragraph unit. A
        longer paragraph is sentence-split (only then — never eagerly), and
        any sentence still over the hard maximum is split at deterministic
        no-whitespace lexical cluster boundaries. Every unit is at most
        ``max_tokens``.
        """
        config = self._config
        units: list[tuple[PlannedSourceSpan, int]] = []
        for paragraph in group:
            token_count = count_lexical_tokens(paragraph.text)
            if token_count <= config.max_tokens:
                units.append(
                    (
                        PlannedSourceSpan(
                            paragraph.paragraph_key,
                            paragraph.source_anchor,
                            0,
                            len(paragraph.text),
                        ),
                        token_count,
                    )
                )
                continue
            if not config.split_long_paragraphs_by_sentence:
                raise ChunkingError(
                    f"paragraph {paragraph.paragraph_key!r} has {token_count} tokens, "
                    f"exceeding max_tokens {config.max_tokens}, and "
                    "split_long_paragraphs_by_sentence is disabled"
                )
            for span in split_sentences(paragraph.text):
                span_tokens = count_lexical_tokens(span.text)
                if span_tokens > config.max_tokens:
                    for sub in split_oversized_sentence(
                        paragraph.text, span.start, span.end, config.max_tokens
                    ):
                        units.append(
                            (
                                PlannedSourceSpan(
                                    paragraph.paragraph_key,
                                    paragraph.source_anchor,
                                    sub.start,
                                    sub.end,
                                ),
                                count_lexical_tokens(sub.text),
                            )
                        )
                else:
                    units.append(
                        (
                            PlannedSourceSpan(
                                paragraph.paragraph_key,
                                paragraph.source_anchor,
                                span.start,
                                span.end,
                            ),
                            span_tokens,
                        )
                    )
        return units

    def _pack_units(
        self, units: list[tuple[PlannedSourceSpan, int]]
    ) -> list[list[tuple[PlannedSourceSpan, int]]]:
        """Greedy deterministic packing in source order.

        Adding the next unit flushes when it would exceed ``max_tokens`` —
        the hard ceiling — or when the current passage already reached
        ``target_tokens`` and the next unit is independent. No overlap, no
        source-order changes, no structural-group crossings.
        """
        config = self._config
        passages: list[list[tuple[PlannedSourceSpan, int]]] = []
        current: list[tuple[PlannedSourceSpan, int]] = []
        current_tokens = 0
        for unit, tokens in units:
            if not current:
                current = [(unit, tokens)]
                current_tokens = tokens
            elif (
                current_tokens + tokens > config.max_tokens
                or current_tokens >= config.target_tokens
            ):
                passages.append(current)
                current = [(unit, tokens)]
                current_tokens = tokens
            else:
                current.append((unit, tokens))
                current_tokens += tokens
        if current:
            passages.append(current)
        return passages

    def _merge_small_tails(
        self, passages: list[list[tuple[PlannedSourceSpan, int]]]
    ) -> list[list[tuple[PlannedSourceSpan, int]]]:
        """Merge a short final passage backward into its predecessor.

        Only within the same structural group, only when the combined size
        stays within ``max_tokens``. ``min_tokens`` is soft; ``max_tokens``
        is hard. A short passage whose merge would exceed the maximum is
        preserved.
        """
        config = self._config
        while len(passages) >= 2:
            last_tokens = sum(tokens for _, tokens in passages[-1])
            previous_tokens = sum(tokens for _, tokens in passages[-2])
            if (
                last_tokens < config.min_tokens
                and previous_tokens + last_tokens <= config.max_tokens
            ):
                passages[-2] = passages[-2] + passages[-1]
                passages.pop()
            else:
                break
        return passages

    @staticmethod
    def _build_text(
        paragraph_map: dict[str, Paragraph],
        passage_units: list[tuple[PlannedSourceSpan, int]],
    ) -> str:
        """Construct the exact deterministic passage text.

        Multiple complete paragraphs are joined with ``"\\n\\n"``. The
        fragments one long paragraph contributes to one passage are taken
        from the exact canonical ``Paragraph.text`` substring between the
        first fragment's start and the last fragment's end — never
        synthesized with a fixed-space join — so the passage text stays
        byte-identical to the source text its spans address, whatever
        whitespace the source uses between fragments.
        """
        parts: list[str] = []
        current_key: str | None = None
        group_start = 0
        group_end = 0
        for unit, _ in passage_units:
            if unit.paragraph_key != current_key:
                if current_key is not None:
                    parts.append(paragraph_map[current_key].text[group_start:group_end])
                current_key = unit.paragraph_key
                group_start = unit.start_char
            group_end = unit.end_char
        if current_key is not None:
            parts.append(paragraph_map[current_key].text[group_start:group_end])
        return "\n\n".join(parts)
