"""Source-to-projection identity for the registered SciFact BEIR corpus."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from dynamisrag.datasets.scifact_projection import (
    SciFactCorpusDocument,
    scifact_projection_records,
)
from dynamisrag.domain.contracts import SourceArtifact
from dynamisrag.search.projection import build_projection_manifest


def _row(document_id: str, text: str) -> SciFactCorpusDocument:
    raw = (
        json.dumps(
            {"_id": document_id, "title": "Shared title", "text": text},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    return SciFactCorpusDocument(
        document_id=document_id,
        title="Shared title",
        text=text,
        source_artifact=SourceArtifact(
            source_system="beir_scifact",
            source_external_id=document_id,
            source_uri="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip#scifact/corpus.jsonl",
            media_type="application/json",
            content_sha256=hashlib.sha256(raw).hexdigest(),
            byte_size=len(raw),
            retrieved_at=datetime(2026, 10, 15, tzinfo=UTC),
            storage_uri=f"file:///verified/scifact/corpus.jsonl#{document_id}",
        ),
    )


def test_scifact_rows_reuse_the_production_chunker_and_projection_identity() -> None:
    rows = (_row("S2", "Evidence text two."), _row("S1", "Evidence text one."))

    forward = scifact_projection_records(rows)
    reverse = scifact_projection_records(tuple(reversed(rows)))
    revision = forward[0].passage.chunker_revision
    first = build_projection_manifest(forward, chunker_revision=revision)
    second = build_projection_manifest(reverse, chunker_revision=revision)

    assert first.projection_sha256 == second.projection_sha256
    assert first.document_count == 2
    assert {record.artifact.source_external_id for record in forward} == {"S1", "S2"}
    assert {record.document.canonical_key for record in forward} == {
        record.document.canonical_key for record in reverse
    }
    assert all(record.source_spans for record in forward)
    assert all(record.passage.text for record in forward)
