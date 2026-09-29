"""The passage projector: rebuild ordering, idempotency and failure safety.

The projector is exercised against a scripted OpenSearch transport and a stubbed
canonical read, so the *order* of operations is asserted exactly: build, bulk,
verify, switch alias, then remove obsolete indexes. That ordering is the whole
safety argument, so it is pinned here rather than inferred from the code.

The two properties that must never regress:

* re-projecting an unchanged canonical state is a no-op (``created=False``) —
  no rebuild, no bulk traffic, no alias churn;
* any failure before the alias switch leaves the alias exactly where it was.
  An orphan physical index may be left behind; a partially indexed index the
  alias already points at never is.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Final

import httpx2
import pytest
from sqlalchemy.orm import Session

from dynamisrag.db.canonical import PassageProjectionRecords
from dynamisrag.search.client import OpenSearchClient
from dynamisrag.search.errors import OpenSearchBulkError, ProjectionError
from dynamisrag.search.projection import (
    PassageProjector,
    ProjectionResult,
    build_projection_manifest,
)
from tests._support import build_settings, passage_projection_corpus, passage_projection_records

_CHUNKER_REVISION: Final[str] = "structure-v1.1.b19e0939b5de"
_OTHER_REVISION: Final[str] = "structure-v0.9.000000000000"
_ALIAS: Final[str] = "dynamisrag-passages-test"
_CHANGED_TEXT: Final[str] = "A probiotic soy diet doubled the colon lesion score."

_SETTINGS = build_settings(opensearch_url="https://search.internal:9200")

_BULK_FAILURE: Final[dict[str, Any]] = {
    "errors": True,
    "items": [
        {
            "index": {
                "status": 400,
                "error": {"type": "mapper_parsing_exception", "reason": "failed to parse"},
            }
        }
    ],
}


# ---------------------------------------------------------------------------
# Stubbed canonical read
# ---------------------------------------------------------------------------


class _Canonical:
    """A stand-in for the canonical PostgreSQL read, with no database."""

    def __init__(
        self,
        *,
        records: Sequence[PassageProjectionRecords],
        available_revisions: Sequence[str] = (_CHUNKER_REVISION,),
    ) -> None:
        self.records = list(records)
        self.available_revisions = list(available_revisions)
        self.queries: list[str] = []

    def list_passage_projection_records(
        self, _session: Session, *, chunker_revision: str
    ) -> Sequence[PassageProjectionRecords]:
        self.queries.append(chunker_revision)
        if chunker_revision != _CHUNKER_REVISION:
            return []
        return list(self.records)

    def list_passage_chunker_revisions(self, _session: Session) -> Sequence[str]:
        return list(self.available_revisions)

    def manifest(self) -> Any:
        return build_projection_manifest(list(self.records), chunker_revision=_CHUNKER_REVISION)

    def index_name(self) -> str:
        return str(self.manifest().index_name(alias=_ALIAS))


# ---------------------------------------------------------------------------
# A minimal in-memory OpenSearch
# ---------------------------------------------------------------------------


class _Node:
    """Just enough OpenSearch to assert the call ordering and the alias state."""

    def __init__(self, *, indices: Sequence[str] = (), alias_targets: Sequence[str] = ()) -> None:
        self.indices: set[str] = set(indices)
        self.alias_targets: set[str] = set(alias_targets)
        self.documents: dict[str, list[dict[str, Any]]] = {}
        self.mapping_meta: dict[str, dict[str, str]] = {}
        self.settings: dict[str, Any] = {}
        self.bulk_failures: bool = False
        self.count_override: int | None = None
        self.calls: list[str] = []

    def transport(self) -> httpx2.MockTransport:
        def answer(request: httpx2.Request) -> httpx2.Response:
            self.calls.append(f"{request.method} {request.url.path}")
            return self.handle(request)

        return httpx2.MockTransport(answer)

    def handle(self, request: httpx2.Request) -> httpx2.Response:  # noqa: PLR0911
        method, path = request.method, request.url.path
        name = path.lstrip("/")

        if method == "GET" and path == f"/_alias/{_ALIAS}":
            if not self.alias_targets:
                return httpx2.Response(404, json={"error": {"type": "alias_not_found_exception"}})
            return httpx2.Response(200, json={n: {"aliases": {}} for n in self.alias_targets})
        if method == "HEAD":
            return httpx2.Response(200 if name in self.indices else 404)
        if method == "PUT":
            body = json.loads(request.content)
            meta = body["mappings"]["_meta"]
            assert isinstance(meta, dict)
            self.indices.add(name)
            self.mapping_meta[name] = meta
            self.settings[name] = body["settings"]
            return httpx2.Response(200, json={"acknowledged": True})
        if method == "DELETE":
            self.indices.discard(name)
            self.alias_targets.discard(name)
            self.documents.pop(name, None)
            self.mapping_meta.pop(name, None)
            return httpx2.Response(200, json={"acknowledged": True})
        if method == "GET" and path.endswith("/_count"):
            target = path.split("/")[1]
            count = self.count_override
            if count is None:
                count = len(self.documents.get(target, []))
            return httpx2.Response(200, json={"count": count})
        if method == "GET" and path.endswith("/_mapping"):
            target = path.split("/")[1]
            return httpx2.Response(
                200, json={target: {"mappings": {"_meta": self.mapping_meta.get(target, {})}}}
            )
        if method == "POST" and path == "/_bulk":
            return self._bulk(request)
        if method == "POST" and path == "/_aliases":
            return self._switch_alias(request)
        raise AssertionError(f"unexpected request: {method} {path}")

    def _bulk(self, request: httpx2.Request) -> httpx2.Response:
        lines = request.content.decode("utf-8").splitlines()
        target = json.loads(lines[0])["index"]["_index"]
        stored = self.documents.setdefault(target, [])
        for offset in range(1, len(lines), 2):
            document = json.loads(lines[offset])
            stored[:] = [d for d in stored if d["passage_key"] != document["passage_key"]]
            stored.append(document)
        if self.bulk_failures:
            return httpx2.Response(200, json=_BULK_FAILURE)
        return httpx2.Response(200, json={"errors": False, "items": []})

    def _switch_alias(self, request: httpx2.Request) -> httpx2.Response:
        for action in json.loads(request.content)["actions"]:
            if "remove" in action:
                self.alias_targets.discard(action["remove"]["index"])
            if "add" in action:
                self.alias_targets.add(action["add"]["index"])
        return httpx2.Response(200, json={"acknowledged": True})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def node() -> _Node:
    return _Node()


@pytest.fixture
def canonical() -> _Canonical:
    return _Canonical(records=passage_projection_records(passage_projection_corpus()))


@pytest.fixture
def changed() -> _Canonical:
    return _Canonical(
        records=passage_projection_records(passage_projection_corpus(text_a=_CHANGED_TEXT))
    )


def _projector(
    node: _Node, canonical: _Canonical, monkeypatch: pytest.MonkeyPatch
) -> PassageProjector:
    """Build a projector over a *deliberately unbound* session.

    The canonical reads are stubbed, so any other database access — a write in
    particular — would fail loudly instead of quietly mutating PostgreSQL.
    """
    monkeypatch.setattr(
        "dynamisrag.search.projection.list_passage_projection_records",
        canonical.list_passage_projection_records,
    )
    monkeypatch.setattr(
        "dynamisrag.search.projection.list_passage_chunker_revisions",
        canonical.list_passage_chunker_revisions,
    )
    return PassageProjector(
        Session(),
        OpenSearchClient(_SETTINGS, transport=node.transport()),
        alias=_ALIAS,
        batch_size=500,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_a_first_projection_builds_indexes_and_switches_the_alias(
    node: _Node, canonical: _Canonical, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _projector(node, canonical, monkeypatch).project(chunker_revision=_CHUNKER_REVISION)

    assert result.created is True
    assert result.index_name == canonical.index_name()
    assert result.alias == _ALIAS
    assert result.document_count == 2
    assert result.projection_schema_revision == "passage-index-v1"
    assert result.chunker_revision == _CHUNKER_REVISION
    assert result.removed_index_names == ()
    assert node.alias_targets == {canonical.index_name()}


def test_the_physical_index_is_created_with_the_declared_settings(
    node: _Node, canonical: _Canonical, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dynamisrag.search.schema import index_settings

    _projector(node, canonical, monkeypatch).project(chunker_revision=_CHUNKER_REVISION)

    assert node.settings[canonical.index_name()] == index_settings()


def test_the_operations_occur_in_the_documented_order(
    node: _Node, canonical: _Canonical, monkeypatch: pytest.MonkeyPatch
) -> None:
    _projector(node, canonical, monkeypatch).project(chunker_revision=_CHUNKER_REVISION)
    index_name = canonical.index_name()

    bulk_position = node.calls.index("POST /_bulk")
    aliases_position = node.calls.index("POST /_aliases")
    # Create, then bulk, then verify, and only then move the alias.
    assert f"PUT /{index_name}" in node.calls[:bulk_position]
    assert f"GET /{index_name}/_count" in node.calls[bulk_position:aliases_position]
    assert f"GET /{index_name}/_mapping" in node.calls[bulk_position:aliases_position]
    assert bulk_position < aliases_position


def test_documents_are_indexed_with_the_passage_key_as_the_opensearch_id(
    node: _Node, canonical: _Canonical, monkeypatch: pytest.MonkeyPatch
) -> None:
    _projector(node, canonical, monkeypatch).project(chunker_revision=_CHUNKER_REVISION)

    stored = node.documents[canonical.index_name()]
    assert [document["passage_key"] for document in stored] == ["a" * 64, "b" * 64]
    assert len({document["projection_sha256"] for document in stored}) == 1


def test_projection_reads_postgresql_and_never_writes_to_it(
    node: _Node, canonical: _Canonical, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The projector runs on an unbound session: any database write would fail
    immediately rather than mutate canonical state."""
    result = _projector(node, canonical, monkeypatch).project(chunker_revision=_CHUNKER_REVISION)

    assert result.created is True
    assert canonical.queries == [_CHUNKER_REVISION]


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_projecting_the_same_state_twice_is_a_no_op(
    node: _Node, canonical: _Canonical, monkeypatch: pytest.MonkeyPatch
) -> None:
    projector = _projector(node, canonical, monkeypatch)

    first = projector.project(chunker_revision=_CHUNKER_REVISION)
    node.calls.clear()
    second = projector.project(chunker_revision=_CHUNKER_REVISION)

    assert first.created is True
    assert second.created is False
    assert second.projection_sha256 == first.projection_sha256
    assert second.index_name == first.index_name
    assert "POST /_bulk" not in node.calls
    assert "POST /_aliases" not in node.calls


def test_a_changed_corpus_builds_a_new_index_and_cuts_the_alias_over(
    node: _Node,
    canonical: _Canonical,
    changed: _Canonical,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _projector(node, canonical, monkeypatch).project(chunker_revision=_CHUNKER_REVISION)

    second = _projector(node, changed, monkeypatch).project(chunker_revision=_CHUNKER_REVISION)

    assert second.created is True
    assert second.index_name != first.index_name
    assert second.projection_sha256 != first.projection_sha256
    # The old index was still present while the new one was being built, and is
    # removed only after the cutover.
    assert f"DELETE /{first.index_name}" in node.calls
    assert node.calls.index("POST /_aliases") < node.calls.index(f"DELETE /{first.index_name}")
    assert node.alias_targets == {second.index_name}
    assert first.index_name not in node.indices
    assert second.removed_index_names == (first.index_name,)


# ---------------------------------------------------------------------------
# Failure safety
# ---------------------------------------------------------------------------


def test_a_bulk_failure_leaves_the_alias_on_the_previous_verified_index(
    node: _Node,
    canonical: _Canonical,
    changed: _Canonical,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _projector(node, canonical, monkeypatch).project(chunker_revision=_CHUNKER_REVISION)
    assert node.alias_targets == {first.index_name}

    node.bulk_failures = True
    node.calls.clear()

    with pytest.raises(OpenSearchBulkError):
        _projector(node, changed, monkeypatch).project(chunker_revision=_CHUNKER_REVISION)

    # The reader keeps seeing the old, complete index. The orphan physical
    # index is a disposable leftover, strictly preferable to a partial one.
    assert node.alias_targets == {first.index_name}
    assert first.index_name in node.indices
    assert "POST /_aliases" not in node.calls


def test_a_verification_failure_leaves_the_alias_where_it_was(
    node: _Node,
    canonical: _Canonical,
    changed: _Canonical,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _projector(node, canonical, monkeypatch).project(chunker_revision=_CHUNKER_REVISION)

    node.calls.clear()
    node.count_override = 99  # the index does not hold the manifest's documents

    with pytest.raises(ProjectionError, match="the alias was not moved"):
        _projector(node, changed, monkeypatch).project(chunker_revision=_CHUNKER_REVISION)

    assert node.alias_targets == {first.index_name}
    assert "POST /_aliases" not in node.calls


def test_an_orphan_index_from_a_failed_build_is_deleted_and_rebuilt(
    node: _Node, canonical: _Canonical, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_name = canonical.index_name()
    node.indices.add(index_name)  # left behind by an earlier failed build
    node.documents[index_name] = [{"passage_key": "truncated"}]

    result = _projector(node, canonical, monkeypatch).project(chunker_revision=_CHUNKER_REVISION)

    assert result.created is True
    assert f"DELETE /{index_name}" in node.calls
    assert [document["passage_key"] for document in node.documents[index_name]] == [
        "a" * 64,
        "b" * 64,
    ]


def test_an_unverifiable_active_index_is_rebuilt_rather_than_trusted(
    node: _Node, canonical: _Canonical, monkeypatch: pytest.MonkeyPatch
) -> None:
    projector = _projector(node, canonical, monkeypatch)
    first = projector.project(chunker_revision=_CHUNKER_REVISION)

    # The alias still points at the right index and its `_meta` still matches,
    # but its content no longer does. Unknown state must not be served: the
    # index is deleted and rebuilt from canonical PostgreSQL.
    node.documents[first.index_name] = node.documents[first.index_name][:1]
    node.calls.clear()

    result = projector.project(chunker_revision=_CHUNKER_REVISION)

    assert result.created is True
    assert result.index_name == first.index_name
    assert f"DELETE /{first.index_name}" in node.calls
    assert len(node.documents[first.index_name]) == 2
    assert node.alias_targets == {first.index_name}


# ---------------------------------------------------------------------------
# Explicit chunker revision selection
# ---------------------------------------------------------------------------


def test_the_requested_revision_is_the_only_one_projected(
    node: _Node, canonical: _Canonical, monkeypatch: pytest.MonkeyPatch
) -> None:
    _projector(node, canonical, monkeypatch).project(chunker_revision=_CHUNKER_REVISION)

    assert canonical.queries == [_CHUNKER_REVISION]


def test_a_missing_chunker_revision_fails_and_names_the_revisions_that_exist(
    node: _Node, canonical: _Canonical, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ProjectionError) as caught:
        _projector(node, canonical, monkeypatch).project(chunker_revision=_OTHER_REVISION)

    message = str(caught.value)
    assert _OTHER_REVISION in message
    assert _CHUNKER_REVISION in message
    assert node.calls == []


def test_an_empty_corpus_fails_before_any_request(
    node: _Node, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = _Canonical(records=(), available_revisions=())

    with pytest.raises(ProjectionError, match="no passages at all"):
        _projector(node, empty, monkeypatch).project(chunker_revision=_CHUNKER_REVISION)

    assert node.calls == []


# ---------------------------------------------------------------------------
# Result contract
# ---------------------------------------------------------------------------


def test_the_result_payload_is_machine_readable(
    node: _Node, canonical: _Canonical, monkeypatch: pytest.MonkeyPatch
) -> None:
    result: ProjectionResult = _projector(node, canonical, monkeypatch).project(
        chunker_revision=_CHUNKER_REVISION
    )

    payload = result.to_payload()
    assert payload["created"] is True
    assert payload["document_count"] == 2
    assert payload["alias"] == _ALIAS
    assert payload["removed_index_names"] == []
    assert json.loads(json.dumps(payload)) == payload
