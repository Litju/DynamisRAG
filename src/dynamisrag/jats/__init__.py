"""Deterministic JATS parsing and canonical materialization (RES-133).

    immutable SourceArtifact JATS XML
        -> JatsParser (pure: bytes -> deterministic ParsedJatsArticle)
        -> JatsCanonicalImporter (parsed article + artifact -> canonical
           domain records -> persistence)

Paragraphs are source structure; Passage generation (RES-134) and all
retrieval concerns are deliberately absent from this package.
"""

from __future__ import annotations

from dynamisrag.jats.anchors import AnchorIndex, build_anchor_index
from dynamisrag.jats.errors import (
    JatsDocumentIdentityConflict,
    JatsMissingRequiredMetadata,
    JatsParseError,
    JatsParseWarning,
    JatsSourceIntegrityError,
    JatsSourcePmcidConflict,
)
from dynamisrag.jats.importer import JatsCanonicalImporter, JatsImportCounts, JatsImportResult
from dynamisrag.jats.parser import (
    JATS_NORMALIZER_REVISION,
    JATS_PARSER_REVISION,
    JatsParser,
    ParsedCitation,
    ParsedFigure,
    ParsedJatsArticle,
    ParsedParagraph,
    ParsedSection,
    ParsedTable,
)
from dynamisrag.jats.text import (
    XLINK_HREF,
    XML_LANG,
    element_text,
    first_local,
    iter_local,
    local_name,
    normalize_language,
    normalize_text,
)

__all__ = [
    "JATS_NORMALIZER_REVISION",
    "JATS_PARSER_REVISION",
    "XLINK_HREF",
    "XML_LANG",
    "AnchorIndex",
    "JatsCanonicalImporter",
    "JatsDocumentIdentityConflict",
    "JatsImportCounts",
    "JatsImportResult",
    "JatsMissingRequiredMetadata",
    "JatsParseError",
    "JatsParseWarning",
    "JatsParser",
    "JatsSourceIntegrityError",
    "JatsSourcePmcidConflict",
    "ParsedCitation",
    "ParsedFigure",
    "ParsedJatsArticle",
    "ParsedParagraph",
    "ParsedSection",
    "ParsedTable",
    "build_anchor_index",
    "element_text",
    "first_local",
    "iter_local",
    "local_name",
    "normalize_language",
    "normalize_text",
]
