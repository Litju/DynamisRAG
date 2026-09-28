"""Cross-reconstruction identity proof (RES-131).

The contract under test: *the same semantic scientific corpus processed with
the same algorithms/configuration produces the same canonical identities
regardless of database instance, insertion order, random surrogate IDs, or
clean reconstruction.*

These tests are infrastructure-free. They build the same semantic graph
twice — every construction assigns fresh ``uuid4`` surrogate ids — and prove
every canonical identity is identical, and that each semantic change moves
exactly the derived keys it should. The database-level counterpart (two
independently created databases) lives in the integration suite.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from dynamisrag.domain.contracts import (
    Citation,
    Document,
    DocumentTable,
    DocumentVersion,
    Figure,
    Passage,
    Section,
    SourceArtifact,
)
from dynamisrag.domain.values import DocumentType

_NOW = datetime(2026, 9, 27, 12, 0, 0, tzinfo=UTC)
_CONTENT_SHA = "a" * 64
_ARTIFACT_SHA = "b" * 64


def _build_graph(
    *,
    parser_revision: str = "jats-1.2",
    child_path: str = "1.2",
    passage_ordinal: int = 0,
    citation_ordinal: int = 0,
    table_label: str | None = "Table 1",
    figure_label: str | None = "Figure 1",
) -> dict[str, Any]:
    """Build one full semantic graph.

    Every call assigns fresh surrogate ids; the keyword parameters are the
    semantic inputs each change test varies, one at a time.
    """
    artifact = SourceArtifact(
        source_system="europe_pmc",
        source_external_id="PMC123456",
        source_uri="https://www.ebi.ac.uk/europepmc/webservices/rest/PMC123456/fulltextXML",
        media_type="application/xml",
        content_sha256=_ARTIFACT_SHA,
        byte_size=2048,
        retrieved_at=_NOW,
        storage_uri="s3://dynamisrag-artifacts/europe_pmc/PMC123456.xml",
    )
    document = Document(
        document_type=DocumentType.JOURNAL_ARTICLE,
        doi="10.1038/nature12373",
        pmid="23656234",
        pmcid="PMC3656234",
        title="A foundational study",
    )
    version = DocumentVersion(
        document_id=document.id,
        document_canonical_key=document.canonical_key,
        source_artifact_id=artifact.id,
        source_artifact_key=artifact.artifact_key,
        parser_revision=parser_revision,
        normalizer_revision="norm-v3",
        content_fingerprint=_CONTENT_SHA,
        title="A foundational study",
        language="en",
        versioned_metadata={"journal": "Nature", "volume": "497"},
        created_at=_NOW,
    )
    root = Section(
        document_version_id=version.id,
        version_key=version.version_key,
        ordinal=1,
        depth=0,
        structural_path="1",
    )
    child = Section(
        document_version_id=version.id,
        version_key=version.version_key,
        ordinal=2,
        depth=child_path.count("."),
        structural_path=child_path,
        parent_section_id=root.id,
    )
    passage = Passage(
        document_version_id=version.id,
        version_key=version.version_key,
        section_id=root.id,
        chunker_revision="chunker-7",
        ordinal=passage_ordinal,
        text="A canonical passage of scientific content.",
        content_sha256=_CONTENT_SHA,
    )
    citation = Citation(
        document_version_id=version.id,
        version_key=version.version_key,
        ordinal=citation_ordinal,
        source_reference_id="ref-0",
        raw_reference_text="Doe et al. (2020). A cited study.",
    )
    table = DocumentTable(
        document_version_id=version.id,
        version_key=version.version_key,
        section_id=child.id,
        ordinal=1,
        label=table_label,
        caption="Baseline characteristics",
        source_anchor="table-wrap-1",
    )
    figure = Figure(
        document_version_id=version.id,
        version_key=version.version_key,
        section_id=child.id,
        ordinal=1,
        label=figure_label,
        caption="Study overview",
        source_anchor="fig-1",
    )
    return {
        "artifact": artifact,
        "document": document,
        "version": version,
        "root": root,
        "child": child,
        "passage": passage,
        "citation": citation,
        "table": table,
        "figure": figure,
    }


def _identities(graph: dict[str, Any]) -> dict[str, Any]:
    """Every canonical identity of a semantic graph, keyed by entity."""
    return {
        "artifact_key": graph["artifact"].artifact_key,
        "document_canonical_key": graph["document"].canonical_key,
        "version_key": graph["version"].version_key,
        "section_key_root": graph["root"].section_key,
        "section_key_child": graph["child"].section_key,
        "passage_key": graph["passage"].passage_key,
        "citation_key": graph["citation"].citation_key,
        "document_table_key": graph["table"].document_table_key,
        "figure_key": graph["figure"].figure_key,
    }


# ---------------------------------------------------------------------------
# Same semantic graph, different surrogate ids: identical canonical identities
# ---------------------------------------------------------------------------


def test_same_semantic_graph_rebuilt_with_fresh_surrogate_ids_has_identical_identities() -> None:
    """The core invariant: random surrogate ids never influence identity.

    Two independent constructions of the same semantic graph — different
    database-equivalent state, different insertion runs — must produce
    byte-for-byte identical canonical identities for DocumentVersion,
    Section, Passage, Citation, DocumentTable and Figure.
    """
    first = _build_graph()
    second = _build_graph()

    # Sanity: the surrogate ids really are random and different in every run.
    assert first["version"].id != second["version"].id
    assert first["artifact"].id != second["artifact"].id
    assert first["root"].id != second["root"].id
    assert first["child"].id != second["child"].id
    assert first["passage"].id != second["passage"].id
    assert first["citation"].id != second["citation"].id
    assert first["table"].id != second["table"].id
    assert first["figure"].id != second["figure"].id

    assert _identities(first) == _identities(second)


def test_identities_are_equal_when_the_whole_graph_is_recomputed_from_scratch() -> None:
    """A clean reconstruction (fresh contracts, fresh surrogates, same
    semantic inputs) reproduces the same canonical identities."""
    baseline = _build_graph()
    reconstructed = _build_graph()

    assert _identities(reconstructed) == _identities(baseline)


# ---------------------------------------------------------------------------
# Semantic changes move exactly the derived keys they should
# ---------------------------------------------------------------------------


def test_semantic_change_to_the_version_cascades_to_every_child_identity() -> None:
    """A parser revision change alters the version key and, because every
    child key digests the version key, every descendant identity too — while
    the upstream artifact and document identities stay fixed."""
    baseline = _build_graph()
    reparsed = _build_graph(parser_revision="jats-1.3")

    changed = _identities(reparsed)
    assert changed["version_key"] != _identities(baseline)["version_key"]
    assert changed["section_key_root"] != _identities(baseline)["section_key_root"]
    assert changed["section_key_child"] != _identities(baseline)["section_key_child"]
    assert changed["passage_key"] != _identities(baseline)["passage_key"]
    assert changed["citation_key"] != _identities(baseline)["citation_key"]
    assert changed["document_table_key"] != _identities(baseline)["document_table_key"]
    assert changed["figure_key"] != _identities(baseline)["figure_key"]

    assert changed["artifact_key"] == _identities(baseline)["artifact_key"]
    assert changed["document_canonical_key"] == _identities(baseline)["document_canonical_key"]


def test_section_key_changes_when_its_structural_path_changes() -> None:
    baseline = _build_graph()
    moved = _build_graph(child_path="1.3")

    assert moved["child"].section_key != baseline["child"].section_key
    assert moved["root"].section_key == baseline["root"].section_key
    assert moved["version"].version_key == baseline["version"].version_key
    assert moved["passage"].passage_key == baseline["passage"].passage_key


def test_passage_key_changes_when_its_ordinal_changes() -> None:
    baseline = _build_graph()
    rechunked = _build_graph(passage_ordinal=1)

    assert rechunked["passage"].passage_key != baseline["passage"].passage_key
    assert rechunked["citation"].citation_key == baseline["citation"].citation_key


def test_citation_key_changes_when_its_ordinal_changes() -> None:
    baseline = _build_graph()
    shifted = _build_graph(citation_ordinal=1)

    assert shifted["citation"].citation_key != baseline["citation"].citation_key
    assert shifted["passage"].passage_key == baseline["passage"].passage_key


def test_document_table_key_changes_when_its_label_changes() -> None:
    baseline = _build_graph()
    relabelled = _build_graph(table_label="Table 2")

    assert relabelled["table"].document_table_key != baseline["table"].document_table_key
    assert relabelled["figure"].figure_key == baseline["figure"].figure_key


def test_figure_key_changes_when_its_label_changes() -> None:
    baseline = _build_graph()
    relabelled = _build_graph(figure_label="Figure 2")

    assert relabelled["figure"].figure_key != baseline["figure"].figure_key
    assert relabelled["table"].document_table_key == baseline["table"].document_table_key
