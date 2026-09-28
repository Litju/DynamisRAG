"""Helpers shared by the unit and integration suites."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import httpx2

from dynamisrag.config import Environment, Settings

__all__ = [
    "ALEMBIC_INI",
    "JATS_FULL_ARTICLE",
    "JATS_SPARSE_ARTICLE",
    "REPO_ROOT",
    "UNREACHABLE_DATABASE_URL",
    "UNREACHABLE_HOST",
    "UNREACHABLE_OPENSEARCH_URL",
    "build_settings",
    "opensearch_root_document",
    "stub_transport",
]

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
"""Repository root, derived from the test package rather than the CWD."""

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


def build_settings(
    *,
    database_url: str = UNREACHABLE_DATABASE_URL,
    opensearch_url: str = UNREACHABLE_OPENSEARCH_URL,
    environment: Environment = Environment.TEST,
) -> Settings:
    """Build fully explicit settings for a test.

    Every field is supplied and ``.env`` loading is bypassed, so neither the
    developer's ``.env`` nor an inherited ``DYNAMISRAG_*`` shell variable can
    change what a test exercises. The defaults point at an unresolvable host,
    which is what the "dependency is down" cases need.
    """
    return Settings.from_mapping(
        {
            "environment": environment,
            "database_url": database_url,
            "opensearch_url": opensearch_url,
            "opensearch_username": "admin",
            "opensearch_password": UNIT_TEST_PASSWORD,
            "opensearch_verify_tls": False,
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
