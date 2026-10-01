"""Helpers shared by the unit and integration suites."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final
from uuid import uuid4

import httpx2

from dynamisrag.config import Environment, Settings
from dynamisrag.db.canonical import PassageProjectionRecords, PassageSourceSpanLineage
from dynamisrag.db.models import (
    DocumentIdentifierRecord,
    DocumentRecord,
    DocumentVersionRecord,
    ParagraphRecord,
    PassageRecord,
    PassageSourceSpanRecord,
    SectionRecord,
    SourceArtifactRecord,
)

__all__ = [
    "ALEMBIC_INI",
    "FIXTURES_ROOT",
    "JATS_FULL_ARTICLE",
    "JATS_PMC2731074",
    "JATS_SPARSE_ARTICLE",
    "PMC2731074_ARTICLE_SHA256",
    "REPO_ROOT",
    "SECRET_ARTICLE_SENTINEL",
    "TEI_MODEL_ID",
    "TEI_MODEL_SHA",
    "TEI_VERSION",
    "UNIT_TEST_PASSWORD",
    "UNREACHABLE_DATABASE_URL",
    "UNREACHABLE_HOST",
    "UNREACHABLE_OPENSEARCH_URL",
    "PassageProjectionCorpus",
    "RecordedTeiRequest",
    "TeiMock",
    "TeiOutcome",
    "build_settings",
    "opensearch_root_document",
    "passage_projection_corpus",
    "passage_projection_records",
    "stub_transport",
    "tei_info_document",
]

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
"""Repository root, derived from the test package rather than the CWD."""

FIXTURES_ROOT: Final[Path] = Path(__file__).resolve().parent / "fixtures"
"""Third-party documents served by the integration suite from disk.

Each file is a byte-exact capture of what Europe PMC served, pinned by SHA-256
and attributed in ``tests/fixtures/README.md``. Serving them from a mocked
transport rather than fetching them is what keeps the suite deterministic and CI
independent of a public service.
"""

ALEMBIC_INI: Final[Path] = REPO_ROOT / "alembic.ini"

UNREACHABLE_HOST: Final[str] = "unreachable.invalid"
"""RFC 6761 reserves the ``.invalid`` TLD so that it must never resolve.

A connection attempt to it therefore fails during name resolution, which is both
faster and more portable than a refused loopback connect: on Windows a refused
``127.0.0.1`` connect blocks for roughly two seconds, and on some machines it
succeeds against an unexpected listener. The tests only require the probe to
report ``down``, so a network that hijacks NXDOMAIN still passes, just more
slowly.
"""

UNIT_TEST_PASSWORD: Final[str] = "unit-test-opensearch-password-1A"
"""Throwaway credential for tests. Never a real secret."""

SECRET_ARTICLE_SENTINEL: Final[str] = "SECRET_ARTICLE_SENTINEL"
"""Stands in for canonical article text inside a backend failure.

OpenSearch quotes the value it rejected in ``error.reason`` and
``caused_by.reason``, and for this projection the value it rejects is an indexed
passage. Tests plant this string in those fields and assert it appears in no
exception message, log record, readiness payload or terminal line.

An all-caps, unmistakably fake token is deliberate: it cannot collide with real
prose, so a failure to assert its absence can never be mistaken for a pass.
"""

UNREACHABLE_DATABASE_URL: Final[str] = (
    f"postgresql://dynamisrag:{UNIT_TEST_PASSWORD}@{UNREACHABLE_HOST}:5432/dynamisrag"
)

UNREACHABLE_OPENSEARCH_URL: Final[str] = f"http://{UNREACHABLE_HOST}:9200"

JATS_FULL_ARTICLE: Final[bytes] = b"""<?xml version="1.0" encoding="UTF-8"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink" article-type="research-article"
         xml:lang="en-US" dtd-version="1.3">
  <front>
    <article-meta>
      <article-id pub-id-type="doi">10.1371/journal.pone.03089012</article-id>
      <article-id pub-id-type="pmid">38888888</article-id>
      <article-id pub-id-type="pmcid">PMC123456</article-id>
      <title-group>
        <article-title>A <italic>synthetic</italic> study of <bold>things</bold>
          and <sup>numbers</sup></article-title>
      </title-group>
      <contrib-group>
        <contrib contrib-type="author">
          <name><surname>Smith</surname><given-names>Jane A.</given-names></name>
          <contrib-id contrib-id-type="orcid">0000-0002-1825-0097</contrib-id>
          <xref ref-type="aff" rid="aff1"/>
        </contrib>
        <contrib contrib-type="author">
          <collab>The Synthetic Consortium</collab>
        </contrib>
      </contrib-group>
      <aff id="aff1">
        <institution>Department of Synthetics, Example University</institution>
        <country>United States</country>
      </aff>
      <journal-meta>
        <journal-title>Journal of Synthetic Studies</journal-title>
        <issn pub-type="print">1234-5678</issn>
        <issn pub-type="electronic">8765-4321</issn>
        <publisher><publisher-name>Synthetic Press</publisher-name></publisher>
      </journal-meta>
      <pub-date pub-type="epub"><year>2024</year><month>5</month><day>15</day></pub-date>
      <volume>19</volume>
      <issue>5</issue>
      <elocation-id>e03089012</elocation-id>
      <abstract>
        <p>This is <italic>very</italic> important. It has two sentences.</p>
        <p>Second abstract paragraph with
          <alternatives><graphic xlink:href="eq1.png"/><tex-math>a^2 + b^2</tex-math></alternatives>
          math.</p>
      </abstract>
      <kwd-group kwd-group-type="author"><kwd>synthetic</kwd><kwd>testing</kwd></kwd-group>
    </article-meta>
  </front>
  <body>
    <p>Direct body paragraph without a section.</p>
    <sec id="sec1">
      <title>Introduction</title>
      <p>Intro paragraph one.</p>
      <sec id="sec1-1">
        <title>Background</title>
        <p>Background paragraph with a <xref ref-type="bibr" rid="R1">citation</xref>.</p>
        <list><list-item><p>List paragraph content.</p></list-item></list>
      </sec>
    </sec>
    <sec id="sec2">
      <title>Methods</title>
      <p>Methods paragraph.</p>
      <fig id="F1">
        <label>Figure 1</label>
        <caption><p>A synthetic figure caption.</p></caption>
        <graphic xlink:href="figure1.png"/>
      </fig>
      <table-wrap id="T1">
        <label>Table 1</label>
        <caption>A synthetic table.</caption>
        <table>
          <thead><tr><th rowspan="2">Group</th><th colspan="2">Values</th></tr></thead>
          <tbody>
            <tr><td>Control</td><td>1</td><td>2</td></tr>
            <tr><td>Treated</td><td>3</td><td>4</td></tr>
          </tbody>
        </table>
      </table-wrap>
    </sec>
  </body>
  <back>
    <ref-list>
      <ref id="R1">
        <mixed-citation>Smith J, 2020. A cited study. Journal of Citations.
          <pub-id pub-id-type="doi">10.1016/j.cell.2020.01.001</pub-id>
          <pub-id pub-id-type="pmid">31900000</pub-id></mixed-citation>
      </ref>
      <ref id="R2">
        <element-citation>
          <article-title>Another cited work</article-title>
          <year>2019</year>
          <pub-id pub-id-type="pmcid">PMC7000000</pub-id>
        </element-citation>
      </ref>
    </ref-list>
  </back>
</article>
"""
"""One representative synthetic JATS article: every canonical structure kind."""

JATS_SPARSE_ARTICLE: Final[bytes] = b"""<?xml version="1.0" encoding="UTF-8"?>
<article article-type="brief-report">
  <front>
    <article-meta>
      <article-id pub-id-type="pmcid">PMC22222222</article-id>
      <title-group><article-title>A sparse article</article-title></title-group>
    </article-meta>
  </front>
  <body>
    <p>Only paragraph.</p>
  </body>
</article>
"""
"""A minimal valid article: PMCID, title, one direct body paragraph."""


JATS_PMC2731074: Final[bytes] = (FIXTURES_ROOT / "PMC2731074.xml").read_bytes()
"""A real Europe PMC full-text article, captured byte for byte.

    Silva MF, Sivieri K, Rossi EA. J Int Soc Sports Nutr 2009;6:17.
    doi:10.1186/1550-2783-6-17, PMC2731074, CC BY 2.0.

Kept as a file rather than inlined so the bytes stay inspectable, diffable and
attributable; see ``tests/fixtures/README.md``. Its digest is pinned below because
the projection identity of this article derives from it, so an unexpected byte
change has to fail loudly rather than silently re-derive every passage key.
"""

PMC2731074_ARTICLE_SHA256: Final[str] = (
    "4a8ed3a3b7d3044697f1462eebe653b06086b267d1fae5300b5d4344d34f3ab4"
)
"""SHA-256 of :data:`JATS_PMC2731074`, exactly as Europe PMC served it."""


def build_settings(
    *,
    database_url: str = UNREACHABLE_DATABASE_URL,
    opensearch_url: str = UNREACHABLE_OPENSEARCH_URL,
    environment: Environment = Environment.TEST,
    tei_url: str | None = None,
) -> Settings:
    """Build fully explicit settings for a test.

    Every field is supplied and ``.env`` loading is bypassed, so neither the
    developer's ``.env`` nor an inherited ``DYNAMISRAG_*`` shell variable can
    change what a test exercises. The defaults point at an unresolvable host,
    which is what the "dependency is down" cases need.

    The embedding settings are supplied as unset explicitly rather than omitted.
    They all default to unset, but a key that is *omitted* still falls back to the
    process environment, so leaving them out would let a developer machine with
    ``DYNAMISRAG_TEI_URL`` set change what a test that constructs settings sees.
    """
    return Settings.from_mapping(
        {
            "environment": environment,
            "database_url": database_url,
            "opensearch_url": opensearch_url,
            "opensearch_username": "admin",
            "opensearch_password": UNIT_TEST_PASSWORD,
            "opensearch_verify_tls": False,
            "opensearch_bulk_batch_size": 500,
            "tei_url": tei_url,
            "tei_expected_model_id": None,
            "tei_expected_model_sha": None,
            "tei_api_key": None,
            "tei_verify_tls": False,
            "tei_timeout_seconds": 30.0,
            "tei_batch_size": 32,
            "tei_max_attempts": 3,
            "tei_retry_backoff_seconds": 0.5,
            "dependency_timeout_seconds": 2.0,
            "host": "127.0.0.1",
            "port": 8000,
            "log_level": "INFO",
        }
    )


def opensearch_root_document(version: str = "3.8.0") -> dict[str, object]:
    """Return a minimal but realistic OpenSearch ``GET /`` payload.

    Includes extra keys on purpose: the probe models must ignore unknown fields
    so that a future OpenSearch release cannot break readiness.
    """
    return {
        "name": "node-0",
        "cluster_name": "docker-cluster",
        "cluster_uuid": "0d2Qk3lFQf2R7QKz0d2Qk3lFQf2R7QKz",
        "version": {
            "distribution": "opensearch",
            "number": version,
            "build_type": "tar",
            "build_hash": "0000000000000000000000000000000000000000",
            "build_date": "2026-01-01T00:00:00.000000000Z",
            "build_snapshot": False,
            "lucene_version": "10.0.0",
            "minimum_wire_compatibility_version": "7.10.0",
            "minimum_index_compatibility_version": "7.0.0",
        },
        "tagline": "The OpenSearch Project: https://opensearch.org/",
    }


def stub_transport(
    *,
    status_code: int = 200,
    body: bytes = b"{}",
    raises: Exception | None = None,
) -> httpx2.MockTransport:
    """Return a transport that answers every request identically.

    ``raises`` models a transport-level failure such as
    :class:`httpx2.ConnectError`, which no HTTP server can reproduce.
    """
    if raises is not None:

        def failing(_: httpx2.Request) -> httpx2.Response:
            raise raises

        return httpx2.MockTransport(failing)

    def answering(request: httpx2.Request) -> httpx2.Response:
        assert request.method == "GET"
        return httpx2.Response(status_code, content=body, request=request)

    return httpx2.MockTransport(answering)


# ---------------------------------------------------------------------------
# A scripted Text Embeddings Inference server (RES-137)
# ---------------------------------------------------------------------------

TEI_MODEL_ID: Final[str] = "BAAI/bge-small-en-v1.5"
"""Repository used by the TEI adapter tests.

A real, small, TEI-compatible repository so the fixtures describe something that
actually exists. Only its *name* appears here; the adapter tests never download
it, and the value is not a statement about which model DynamisRAG should use --
that is RES-138's decision, made against retrieval quality rather than against a
test fixture.
"""

TEI_MODEL_SHA: Final[str] = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
"""An immutable Hub commit id for :data:`TEI_MODEL_ID`, 40 lowercase hex.

Shaped exactly like a real commit because that shape *is* the contract:
:class:`~dynamisrag.embedding.tei.ExpectedTeiModel` refuses anything that is not
40 lowercase hexadecimal characters precisely because a tag or a branch cannot be
told apart from a commit by shape.
"""

TEI_VERSION: Final[str] = "1.9.4"
"""The TEI release the adapter's protocol revision ``tei-http-v1`` targets."""


def tei_info_document(
    **overrides: object,
) -> dict[str, object]:
    """Return a ``GET /info`` payload shaped like a real TEI 1.9.4 response.

    Three fields exist purely to prove the parser's ``unknown keys are ignored
    deliberately`` claim: ``served_model_name``, ``tokenization_workers`` and
    ``auto_truncate`` all appeared in TEI between 1.7 and 1.9 and none of them
    decides an identity, so a closed key set would refuse to embed against a
    routine upstream addition.

    ``model_type`` is the externally tagged object TEI 1.9.x actually emits --
    ``{"embedding": {"pooling": "cls"}}`` -- not a bare string, because
    ``Embedding`` carries a pooling configuration. A test that asserted
    ``model_type == "embedding"`` against a string would have passed against a
    mock and failed against every live server.
    """
    document: dict[str, object] = {
        "model_id": TEI_MODEL_ID,
        "model_sha": TEI_MODEL_SHA,
        "model_dtype": "float32",
        "served_model_name": TEI_MODEL_ID,
        "model_type": {"embedding": {"pooling": "cls"}},
        "max_concurrent_requests": 512,
        "max_input_length": 512,
        "max_batch_tokens": 8192,
        "max_batch_requests": 8,
        "max_client_batch_size": 8,
        "auto_truncate": True,
        "tokenization_workers": 7,
        "version": TEI_VERSION,
        "sha": "e80ef225ed0e6cb1717ce632a6a84b6cf211bb67",
        "docker_label": "sha-e80ef22",
    }
    document.update(overrides)
    return document


def tei_error_envelope(
    *,
    error: str,
    error_type: str,
) -> dict[str, str]:
    """Return a TEI failure envelope shaped like the real thing.

    ``error`` carries :data:`SECRET_ARTICLE_SENTINEL` in every test that uses it,
    because that is what TEI genuinely does: the validation and backend failures
    of ``/embed`` quote the rejected input back. The adapter must therefore never
    read the field, and a test that only used a benign message would not notice if
    it did.
    """
    return {"error": error, "error_type": error_type}


@dataclass(frozen=True)
class RecordedTeiRequest:
    """One request the scripted server received, exactly as it was sent.

    ``body`` is the raw ``Content-Length`` payload rather than a re-parsed
    object, so a test can assert the *bytes* of a request -- which is what
    "every retry sends byte-identical content" actually claims -- and can compare
    two bodies with ``==`` instead of comparing parse trees that would look equal
    after any number of round trips.
    """

    method: str
    path: str
    body: bytes
    authorization: str | None


@dataclass(frozen=True)
class TeiOutcome:
    """What the scripted server does with one request.

    Exactly one of ``embeddings``, ``body`` or ``raises`` is used, in that order.
    ``raises`` models a transport-level failure such as
    :class:`httpx2.ReadTimeout`, which no HTTP server can produce and which the
    retry policy must still treat as transient.
    """

    embeddings: list[list[float]] | None = None
    status_code: int = 200
    body: bytes | None = None
    raises: Exception | None = None


@dataclass
class TeiMock:
    """A scripted TEI server over :class:`httpx2.MockTransport`, with no socket.

    ``info_documents`` is consumed one per ``GET /info``; ``embed_outcomes`` one
        per ``POST /embed``. Once a sequence is exhausted its **last** entry repeats,
        so a single-document or single-outcome mock serves any number of calls while
        a two-document mock is exactly the drift case: the identity is stable for the
        first read and different for the second.

        ``info_outcomes`` overrides ``info_documents`` for ``/info`` entirely, which
        is how a *transport or status* failure on the identity read is scripted. It is
        a separate list rather than a union type because "the document changed" and
        "the document could not be read" are different tests with different
        expectations, and mixing them into one list invites a test that silently
        exercises the wrong one.

        Recording every request is what lets a test assert the batch partition, the
        exact serialized body and the byte-identity of a retry without a network.
    """

    info_documents: list[dict[str, object]]
    embed_outcomes: list[TeiOutcome]
    info_outcomes: list[TeiOutcome] | None = None
    _info_calls: int = 0
    _embed_calls: int = 0

    def __post_init__(self) -> None:
        """Declare the recorder here rather than as a dataclass field.

        A ``field(default_factory=list)`` default makes the strict type checker
        infer the element type from the bare ``list`` builtin, which erases it;
        annotating the attribute where it is created keeps the recorded requests
        fully typed.
        """
        self.requests: list[RecordedTeiRequest] = []

    @property
    def embed_requests(self) -> list[RecordedTeiRequest]:
        """Every ``POST /embed`` received, in order."""
        return [request for request in self.requests if request.path == "/embed"]

    @property
    def info_requests(self) -> list[RecordedTeiRequest]:
        """Every ``GET /info`` received, in order."""
        return [request for request in self.requests if request.path == "/info"]

    def transport(self) -> httpx2.MockTransport:
        """The transport to hand to the adapter under test."""
        return httpx2.MockTransport(self._handle)

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(
            RecordedTeiRequest(
                method=request.method,
                path=request.url.path,
                body=request.content,
                authorization=request.headers.get("authorization"),
            )
        )
        if request.url.path == "/info":
            if self.info_outcomes is not None:
                index = min(self._info_calls, len(self.info_outcomes) - 1)
                self._info_calls += 1
                return self._embed_response(self.info_outcomes[index], request)
            index = min(self._info_calls, len(self.info_documents) - 1)
            self._info_calls += 1
            return httpx2.Response(200, json=self.info_documents[index], request=request)
        if request.url.path == "/embed":
            index = min(self._embed_calls, len(self.embed_outcomes) - 1)
            self._embed_calls += 1
            return self._embed_response(self.embed_outcomes[index], request)
        return httpx2.Response(404, json={"error": "no such route"}, request=request)

    def _embed_response(self, outcome: TeiOutcome, request: httpx2.Request) -> httpx2.Response:
        if outcome.raises is not None:
            raise outcome.raises
        if outcome.embeddings is not None:
            return httpx2.Response(outcome.status_code, json=outcome.embeddings, request=request)
        return httpx2.Response(
            outcome.status_code,
            content=outcome.body if outcome.body is not None else b"{}",
            headers={"content-type": "application/json"},
            request=request,
        )


# ---------------------------------------------------------------------------
# Canonical passage corpus for projection tests
# ---------------------------------------------------------------------------

_CORPUS_NOW: Final[datetime] = datetime(2026, 9, 29, 12, 0, 0, tzinfo=UTC)
_CORPUS_TITLE: Final[str] = "Probiotic soy and colon lesions in jumping rats"
_CORPUS_TEXT_A: Final[str] = "A probiotic soy diet reduced colon lesions in jumping rats."
_CORPUS_TEXT_B: Final[str] = "Exercise training improved the jump height of the rats."
_CORPUS_ANCHOR: Final[str] = "jats:/body[1]/sec[1]/p[1]"
_CORPUS_VERSION_KEY: Final[str] = "v" * 64
_CORPUS_CHUNKER_REVISION: Final[str] = "structure-v1.1.b19e0939b5de"
_CORPUS_PASSAGE_KEY_A: Final[str] = "a" * 64
_CORPUS_PASSAGE_KEY_B: Final[str] = "b" * 64


@dataclass(frozen=True)
class PassageProjectionCorpus:
    """One two-passage canonical graph, holding every surrogate id it uses."""

    artifact: SourceArtifactRecord
    document: DocumentRecord
    version: DocumentVersionRecord
    section: SectionRecord
    paragraph: ParagraphRecord
    passages: tuple[PassageRecord, ...]
    spans: tuple[PassageSourceSpanLineage, ...]
    identifiers: tuple[DocumentIdentifierRecord, ...]


def passage_projection_corpus(
    *, text_a: str = _CORPUS_TEXT_A, with_section: bool = True
) -> PassageProjectionCorpus:
    artifact_id, document_id, version_id = uuid4(), uuid4(), uuid4()
    section_id = uuid4()
    artifact = SourceArtifactRecord(
        id=artifact_id,
        source_system="europe_pmc",
        source_external_id="PMC2731074",
        source_uri="https://www.ebi.ac.uk/europepmc/webservices/rest/PMC2731074/fullTextXML",
        media_type="application/xml",
        content_sha256="d" * 64,
        byte_size=4096,
        retrieved_at=_CORPUS_NOW,
        storage_uri="file:///artifacts/PMC2731074.xml",
        license_name=None,
        license_uri=None,
        artifact_key="e" * 64,
    )
    document = DocumentRecord(
        id=document_id,
        canonical_key="doi:10.1371/journal.pone.03089012",
        document_type="journal_article",
        title=_CORPUS_TITLE,
    )
    version = DocumentVersionRecord(
        id=version_id,
        document_id=document_id,
        source_artifact_id=artifact_id,
        parser_revision="jats-1.0",
        normalizer_revision="norm-1.0",
        content_fingerprint="f" * 64,
        title=_CORPUS_TITLE,
        language="en",
        versioned_metadata={},
        created_at=_CORPUS_NOW,
        version_key=_CORPUS_VERSION_KEY,
    )
    section = SectionRecord(
        id=section_id,
        document_version_id=version_id,
        parent_section_id=None,
        parent_document_version_id=None,
        ordinal=0,
        depth=0,
        title="Results",
        semantic_type="sec",
        source_anchor="jats:#sec-results",
        structural_path="2",
        content_fingerprint="1" * 64,
        section_key="2" * 64,
    )
    paragraph = ParagraphRecord(
        id=uuid4(),
        document_version_id=version_id,
        section_id=section_id if with_section else None,
        section_document_version_id=version_id if with_section else None,
        ordinal=0,
        region="body",
        source_anchor=_CORPUS_ANCHOR,
        text=_CORPUS_TEXT_A,
        content_sha256="3" * 64,
        paragraph_key="4" * 64,
    )
    passages = tuple(
        PassageRecord(
            id=uuid4(),
            document_version_id=version_id,
            section_id=section_id if with_section else None,
            section_document_version_id=version_id if with_section else None,
            chunker_revision=_CORPUS_CHUNKER_REVISION,
            ordinal=ordinal,
            text=text,
            content_sha256="5" * 64,
            source_anchor=_CORPUS_ANCHOR,
            token_count=12,
            passage_key=key,
        )
        for ordinal, (key, text) in enumerate(
            ((_CORPUS_PASSAGE_KEY_A, text_a), (_CORPUS_PASSAGE_KEY_B, _CORPUS_TEXT_B))
        )
    )
    spans = tuple(
        PassageSourceSpanLineage(
            span=PassageSourceSpanRecord(
                id=uuid4(),
                document_version_id=version_id,
                passage_id=passage.id,
                passage_document_version_id=version_id,
                paragraph_id=paragraph.id,
                paragraph_document_version_id=version_id,
                source_order=0,
                start_char=0,
                end_char=len(_CORPUS_TEXT_A),
                span_key="6" * 64,
            ),
            paragraph=paragraph,
        )
        for passage in passages
    )
    identifiers = tuple(
        DocumentIdentifierRecord(
            id=uuid4(),
            document_id=document_id,
            namespace=namespace,
            normalized_value=value,
        )
        for namespace, value in (
            ("doi", "10.1371/journal.pone.03089012"),
            ("pmid", "38888888"),
            ("pmcid", "PMC2731074"),
        )
    )
    return PassageProjectionCorpus(
        artifact=artifact,
        document=document,
        version=version,
        section=section,
        paragraph=paragraph,
        passages=passages,
        spans=spans,
        identifiers=identifiers,
    )


def passage_projection_records(corpus: PassageProjectionCorpus) -> list[PassageProjectionRecords]:
    """Interpret the corpus the way the persistence read path does."""
    return [
        PassageProjectionRecords(
            passage=passage,
            version=corpus.version,
            document=corpus.document,
            artifact=corpus.artifact,
            section=corpus.section if passage.section_id is not None else None,
            source_spans=(span,),
            identifiers=corpus.identifiers,
        )
        for passage, span in zip(corpus.passages, corpus.spans, strict=True)
    ]
