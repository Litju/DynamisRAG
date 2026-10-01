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
  alias already points at never is;
* an index the alias *already* targets is never deleted by pre-cutover
  recovery. When the desired deterministic index is active it is either
  verified — and then nothing happens — or the run fails closed, whether the
  verification found a mismatch or could not be performed at all.
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
from dynamisrag.search.errors import (
    OpenSearchBulkError,
    OpenSearchTransportError,
    OpenSearchUnexpectedResponse,
    ProjectionConflictError,
    ProjectionError,
)
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
        self.failing_reads: dict[str, str] = {}
        """Verification reads that must fail, keyed by ``count``/``mapping``.

        The value is the kind of failure: ``transport`` never produces a
        response, ``status`` produces a 503. Both leave every stored index,
        document and alias target untouched, which is what a real verification
        timeout looks like to the projector: no answer, and no change either.
        """

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
            if "count" in self.failing_reads:
                return self._fail_read("count")
            count = self.count_override
            if count is None:
                count = len(self.documents.get(target, []))
            return httpx2.Response(200, json={"count": count})
        if method == "GET" and path.endswith("/_mapping"):
            target = path.split("/")[1]
            if "mapping" in self.failing_reads:
                return self._fail_read("mapping")
            return httpx2.Response(
                200, json={target: {"mappings": {"_meta": self.mapping_meta.get(target, {})}}}
            )
        if method == "POST" and path == "/_bulk":
            return self._bulk(request)
        if method == "POST" and path == "/_aliases":
            return self._switch_alias(request)
        raise AssertionError(f"unexpected request: {method} {path}")

    def _fail_read(self, operation: str) -> httpx2.Response:
        """Fail a verification read without changing any stored state.

        Two shapes, because they fail for different reasons and both must reach
        the caller as an error rather than as a verdict about the index:
        a transport exception that never produces a response, and a node that
        answers 503. The 503 carries a backend ``reason`` so a test can prove it
        is never relayed.
        """
        fault = self.failing_reads[operation]
        if fault == "transport":
            raise httpx2.ReadTimeout(f"{operation} verification read timed out")
        return httpx2.Response(
            503,
            json={
                "error": {
                    "type": "no_shard_available_action_exception",
                    "reason": f"VERIFICATION_READ_SENTINEL_{operation}",
                }
            },
        )

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


def test_an_active_index_that_does_not_verify_is_reported_not_rebuilt(
    node: _Node, canonical: _Canonical, monkeypatch: pytest.MonkeyPatch
) -> None:
    projector = _projector(node, canonical, monkeypatch)
    first = projector.project(chunker_revision=_CHUNKER_REVISION)

    # The alias still points at the right index and its `_meta` still matches,
    # but its content no longer does. The index is the live read path, so it is
    # left exactly as it is and the conflict is reported: unknown live state is
    # never silently repaired, because deleting the served index risks trading a
    # detectable inconsistency for an absent search path.
    node.documents[first.index_name] = node.documents[first.index_name][:1]
    node.calls.clear()

    with pytest.raises(ProjectionConflictError):
        projector.project(chunker_revision=_CHUNKER_REVISION)

    assert node.alias_targets == {first.index_name}
    assert first.index_name in node.indices
    assert f"DELETE /{first.index_name}" not in node.calls


# ---------------------------------------------------------------------------
# Active-target failure safety
#
# The index the stable alias targets is the live read path. Every way of losing
# it is a write, so the property proven here is exhaustive in the vocabulary of
# the protocol: a failed run against the active target issues no DELETE, no PUT,
# no bulk request and no alias switch, and leaves the index and the alias
# exactly as they were.
# ---------------------------------------------------------------------------


def _assert_nothing_was_mutated(node: _Node) -> None:
    """Assert no mutating request left the process at all.

    Reads are the only thing a failed verification is allowed to have done.
    """
    mutating = [
        call
        for call in node.calls
        if call.startswith(("DELETE ", "PUT ")) or call in {"POST /_bulk", "POST /_aliases"}
    ]
    assert mutating == []


def test_an_active_target_with_the_wrong_document_count_is_reported_as_a_conflict(
    node: _Node, canonical: _Canonical, monkeypatch: pytest.MonkeyPatch
) -> None:
    projector = _projector(node, canonical, monkeypatch)
    first = projector.project(chunker_revision=_CHUNKER_REVISION)

    node.count_override = 1  # the live index no longer holds the manifest's documents
    node.calls.clear()

    with pytest.raises(ProjectionConflictError) as caught:
        projector.project(chunker_revision=_CHUNKER_REVISION)

    assert node.alias_targets == {first.index_name}
    assert first.index_name in node.indices
    assert node.documents[first.index_name] == node.documents[first.index_name]
    _assert_nothing_was_mutated(node)
    # The conflict is reportable: it names the index, the alias and the
    # projection it contradicts, all of them application-authored values.
    assert first.index_name in str(caught.value)
    assert _ALIAS in str(caught.value)
    assert first.projection_sha256 in str(caught.value)
    assert caught.value.target == first.index_name
    assert "ProjectionConflict" in caught.value.safe_summary()


def test_an_active_target_with_the_wrong_mapping_meta_is_reported_as_a_conflict(
    node: _Node, canonical: _Canonical, monkeypatch: pytest.MonkeyPatch
) -> None:
    projector = _projector(node, canonical, monkeypatch)
    first = projector.project(chunker_revision=_CHUNKER_REVISION)

    # The document count still agrees, so the contradiction can only be found by
    # reading the mapping provenance.
    node.mapping_meta[first.index_name] = {
        **node.mapping_meta[first.index_name],
        "chunker_revision": "structure-v0.0.000000000000",
    }
    corrupted_meta = dict(node.mapping_meta[first.index_name])
    node.calls.clear()

    with pytest.raises(ProjectionConflictError):
        projector.project(chunker_revision=_CHUNKER_REVISION)

    assert node.alias_targets == {first.index_name}
    assert first.index_name in node.indices
    # Not even the wrong provenance is corrected: this run reports, it does not
    # repair live state it does not own.
    assert node.mapping_meta[first.index_name] == corrupted_meta
    _assert_nothing_was_mutated(node)


@pytest.mark.parametrize("operation", ["count", "mapping"])
def test_a_timed_out_verification_of_the_active_target_propagates_and_mutates_nothing(
    node: _Node,
    canonical: _Canonical,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    """A transient read failure is not evidence that the index is wrong.

    The healthy index is left serving, the alias is untouched, and the caller
    learns the verification never happened — instead of the projector treating
    an unanswered question as a failed audit and deleting the live projection.
    """
    projector = _projector(node, canonical, monkeypatch)
    first = projector.project(chunker_revision=_CHUNKER_REVISION)
    documents_before = list(node.documents[first.index_name])
    meta_before = dict(node.mapping_meta[first.index_name])

    node.failing_reads[operation] = "transport"
    node.calls.clear()

    with pytest.raises(OpenSearchTransportError) as caught:
        projector.project(chunker_revision=_CHUNKER_REVISION)

    # The failure is the transport failure itself, not a claim about contents.
    assert caught.value.cause == "ReadTimeout"
    assert not isinstance(caught.value, ProjectionConflictError)
    assert f"GET /{first.index_name}/_{operation}" in node.calls
    assert node.alias_targets == {first.index_name}
    assert first.index_name in node.indices
    assert node.documents[first.index_name] == documents_before
    assert node.mapping_meta[first.index_name] == meta_before
    _assert_nothing_was_mutated(node)


@pytest.mark.parametrize("operation", ["count", "mapping"])
def test_a_rejected_verification_of_the_active_target_propagates_and_mutates_nothing(
    node: _Node,
    canonical: _Canonical,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    """A backend that answers "unavailable" is as uninformative as a timeout."""
    projector = _projector(node, canonical, monkeypatch)
    first = projector.project(chunker_revision=_CHUNKER_REVISION)

    node.failing_reads[operation] = "status"
    node.calls.clear()

    with pytest.raises(OpenSearchUnexpectedResponse) as caught:
        projector.project(chunker_revision=_CHUNKER_REVISION)

    assert caught.value.status_code == 503
    assert caught.value.error_type == "no_shard_available_action_exception"
    assert not isinstance(caught.value, ProjectionConflictError)
    # The node's prose about the failure is never relayed, here or by the safe
    # summary a log line is built from.
    assert "VERIFICATION_READ_SENTINEL" not in str(caught.value)
    assert "VERIFICATION_READ_SENTINEL" not in caught.value.safe_summary()
    assert node.alias_targets == {first.index_name}
    assert first.index_name in node.indices
    _assert_nothing_was_mutated(node)


def test_a_verified_active_target_is_read_and_never_written(
    node: _Node, canonical: _Canonical, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The successful path is as read-only as the failing ones."""
    projector = _projector(node, canonical, monkeypatch)
    first = projector.project(chunker_revision=_CHUNKER_REVISION)

    node.calls.clear()
    second = projector.project(chunker_revision=_CHUNKER_REVISION)

    assert second.created is False
    assert second.index_name == first.index_name
    assert node.calls == [
        f"GET /_alias/{_ALIAS}",
        f"GET /{first.index_name}/_count",
        f"GET /{first.index_name}/_mapping",
    ]


def test_a_deterministic_index_that_is_not_an_alias_target_is_rebuilt_and_cut_over(
    node: _Node,
    canonical: _Canonical,
    changed: _Canonical,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The orphan path is unchanged: delete, rebuild, verify, then cut over.

    The orphan contradicts the manifest exactly like a conflicting active target
    does — wrong count and wrong provenance — and is still rebuilt rather than
    reported. Nothing is served from it, so repairing it cannot cost anyone a
    search path, and the deterministic name makes the rebuild reproduce exactly
    the index that was removed.
    """
    first = _projector(node, canonical, monkeypatch).project(chunker_revision=_CHUNKER_REVISION)
    orphan = changed.index_name()
    node.indices.add(orphan)
    node.documents[orphan] = [{"passage_key": "truncated"}]
    node.mapping_meta[orphan] = {"schema_revision": "passage-index-v1"}
    assert orphan not in node.alias_targets
    node.calls.clear()

    result = _projector(node, changed, monkeypatch).project(chunker_revision=_CHUNKER_REVISION)

    assert result.created is True
    assert result.index_name == orphan
    assert result.removed_index_names == (first.index_name,)
    assert [document["passage_key"] for document in node.documents[orphan]] == ["a" * 64, "b" * 64]
    assert node.mapping_meta[orphan] == changed.manifest().expected_meta()
    assert node.alias_targets == {orphan}
    # Delete, create, bulk, verify, and only then move the alias.
    assert node.calls.index(f"DELETE /{orphan}") < node.calls.index(f"PUT /{orphan}")
    assert node.calls.index(f"PUT /{orphan}") < node.calls.index("POST /_bulk")
    assert node.calls.index("POST /_bulk") < node.calls.index(f"GET /{orphan}/_count")
    assert node.calls.index(f"GET /{orphan}/_count") < node.calls.index(f"GET /{orphan}/_mapping")
    assert node.calls.index(f"GET /{orphan}/_mapping") < node.calls.index("POST /_aliases")
    # The previously served index survived every step up to the cutover.
    assert node.calls.index("POST /_aliases") < node.calls.index(f"DELETE /{first.index_name}")


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
