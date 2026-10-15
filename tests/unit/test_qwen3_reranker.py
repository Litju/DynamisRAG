"""RES-142 adapter qualification against the frozen retrieval and RES-140 contracts."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from email.message import Message
from typing import cast

import pytest

from dynamisrag.ir.contracts import (
    IrContractError,
    IrDataset,
    IrExperimentConfig,
    IrPassageMapEntry,
    IrPassageMapping,
    IrQrel,
    IrQuery,
    document_run_from_passages,
)
from dynamisrag.ir.experiments import compare_ir_evaluations
from dynamisrag.ir.qwen3_reranker import (
    QWEN3_RERANKER_MODEL_ID,
    SCIFACT_PILOT_SESSION_LIMIT_SECONDS,
    Qwen3RerankerAdapter,
    QwenRerankedQuery,
    QwenScoringCandidate,
    SciFactCandidateSnapshot,
    build_scifact_candidate_snapshot,
    frozen_hybrid_passage_hits,
    qwen3_reranker_config,
    qwen3_reranker_run,
    qwen_scoring_candidates,
    ranked_passage_hits,
    select_scifact_pilot_dataset,
    select_scifact_pilot_query_ids,
)
from dynamisrag.ir.scoring import evaluate_ir_run
from dynamisrag.search.bm25 import SearchHit, SearchResponse
from dynamisrag.search.retrieval import (
    DENSE_MODEL_ID,
    DENSE_MODEL_REVISION,
    DenseCandidate,
    DenseSearchResponse,
    FusedHit,
    HybridRetrievalResponse,
    PassageProvenance,
    RrfLaneTrace,
    RrfTrace,
    reciprocal_rank_fusion,
)

_PROJECTION_SHA = "b" * 64
_CODE_SHA = "c" * 40
_INDEX = "scifact-passage-index-v2-test"
_CHUNKER = "structure-v1.test"


def _adapter(
    *, timeout_seconds: float = 120.0, deadline: float | None = None
) -> Qwen3RerankerAdapter:
    if deadline is None:
        deadline = time.monotonic() + SCIFACT_PILOT_SESSION_LIMIT_SECONDS
    return Qwen3RerankerAdapter(
        timeout_seconds=timeout_seconds,
        session_deadline_monotonic=deadline,
    )


def _dataset() -> IrDataset:
    queries = tuple(IrQuery(f"q{index:03d}", f"scientific claim {index}") for index in range(300))
    qrels = tuple(IrQrel(query.query_id, f"doc-{query.query_id}-00", 1) for query in queries)
    return IrDataset(
        source_id="beir.scifact",
        source_revision="beir-frozen-2021.test",
        corpus_sha256="a" * 64,
        queries=queries,
        qrels=qrels,
    )


def _response(query: IrQuery, count: int = 24) -> HybridRetrievalResponse:
    lexical_hits: list[SearchHit] = []
    dense_candidates: list[DenseCandidate] = []
    fused_hits: list[FusedHit] = []
    for index in range(count):
        rank = index + 1
        passage_id = f"{query.query_id}-p{index:03d}"
        document_id = f"doc-{query.query_id}-{index // 2:02d}"
        text = f"Evidence passage {passage_id} for {query.query_id}."
        provenance = PassageProvenance(
            passage_key=passage_id,
            text=text,
            document_canonical_key=document_id,
            document_version_key=f"{document_id}@v1",
            title=f"Title {document_id}",
            language="en",
            chunker_revision=_CHUNKER,
            passage_ordinal=index,
            token_count=7,
            document_type="abstract",
            content_sha256=hashlib.sha256(text.encode()).hexdigest(),
            source_system="beir.scifact",
            source_external_id=document_id,
        )
        bm25_score = 100.0 - rank
        dense_score = 1.0 - (rank / 100.0)
        lexical_hits.append(
            SearchHit(
                rank=rank,
                score=bm25_score,
                **provenance.model_dump(),
            )
        )
        dense_candidates.append(
            DenseCandidate(
                rank=rank, raw_score=dense_score, passage_key=passage_id, provenance=provenance
            )
        )
        contribution = 1.0 / (60 + rank)
        fused_hits.append(
            FusedHit(
                passage_key=passage_id,
                final_rank=rank,
                rrf_score=2.0 * contribution,
                lexical=RrfLaneTrace(
                    present=True,
                    rank=rank,
                    raw_score=bm25_score,
                    rrf_contribution=contribution,
                ),
                dense=RrfLaneTrace(
                    present=True,
                    rank=rank,
                    raw_score=dense_score,
                    rrf_contribution=contribution,
                ),
                provenance=provenance,
            )
        )

    lexical = SearchResponse(
        query=query.text,
        query_revision="bm25-v1",
        index_schema_revision="passage-index-v2",
        projection_sha256=_PROJECTION_SHA,
        chunker_revision=_CHUNKER,
        total=count,
        took_ms=1,
        hits=tuple(lexical_hits),
    )
    dense = DenseSearchResponse(
        query=query.text,
        query_revision="dense-knn-v1",
        physical_index=_INDEX,
        projection_schema_revision="passage-index-v2",
        projection_sha256=_PROJECTION_SHA,
        chunker_revision=_CHUNKER,
        vector_config_sha256="d" * 64,
        model_id=DENSE_MODEL_ID,
        model_revision=DENSE_MODEL_REVISION,
        dimension=512,
        vector_space="cosinesimil",
        query_embedding_config_sha256="e" * 64,
        document_embedding_config_sha256="f" * 64,
        embedding_profile="res138-stage-a-provisional-qwen3-embedding-0.6b-512-v1",
        embedding_profile_status="provisional",
        candidate_window=50,
        took_ms=1,
        candidates=tuple(dense_candidates),
    )
    return HybridRetrievalResponse(
        query=query.text,
        retrieval_revision="hybrid-rrf-v1",
        physical_index=_INDEX,
        embedding_profile="res138-stage-a-provisional-qwen3-embedding-0.6b-512-v1",
        embedding_profile_status="provisional",
        lexical=lexical,
        dense=dense,
        fusion=RrfTrace(revision="rrf-v1", k=60, candidate_window=50, hits=tuple(fused_hits)),
    )


def _response_with_dense_only_tail(query: IrQuery, *, passage_key: str) -> HybridRetrievalResponse:
    response = _response(query, 50)
    dense_tail = response.dense.candidates[-1]
    document_id = f"doc-{query.query_id}-{passage_key}"
    text = f"Nonfused dense source candidate {passage_key}."
    provenance = dense_tail.provenance.model_copy(
        update={
            "passage_key": passage_key,
            "text": text,
            "document_canonical_key": document_id,
            "document_version_key": f"{document_id}@v1",
            "title": f"Title {document_id}",
            "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "source_external_id": document_id,
        }
    )
    dense_only = dense_tail.model_copy(
        update={"passage_key": passage_key, "provenance": provenance}
    )
    dense = response.dense.model_copy(
        update={"candidates": (*response.dense.candidates[:-1], dense_only)}
    )
    fusion = reciprocal_rank_fusion(
        lexical=response.lexical.hits,
        dense=dense.candidates,
        limit=50,
    )
    return response.model_copy(update={"dense": dense, "fusion": fusion})


def _with_fused_hits(
    response: HybridRetrievalResponse, hits: Sequence[FusedHit]
) -> HybridRetrievalResponse:
    fusion = response.fusion.model_copy(update={"hits": tuple(hits)})
    return response.model_copy(update={"fusion": fusion})


def _mapping(
    responses: Mapping[str, HybridRetrievalResponse],
) -> IrPassageMapping:
    entries: dict[str, IrPassageMapEntry] = {}
    for response in responses.values():
        for hit in response.lexical.hits:
            entries[hit.passage_key] = IrPassageMapEntry(
                passage_id=hit.passage_key,
                document_id=hit.document_canonical_key,
                document_version_id=hit.document_version_key,
            )
        for candidate in response.dense.candidates:
            entries[candidate.passage_key] = IrPassageMapEntry(
                passage_id=candidate.passage_key,
                document_id=candidate.provenance.document_canonical_key,
                document_version_id=candidate.provenance.document_version_key,
            )
    return IrPassageMapping(tuple(entries[key] for key in sorted(entries)))


def _inputs(
    *, candidate_count: int = 24, zero_hit_query_ids: frozenset[str] = frozenset()
) -> tuple[
    IrDataset,
    IrDataset,
    tuple[str, ...],
    dict[str, HybridRetrievalResponse],
    IrPassageMapping,
    SciFactCandidateSnapshot,
]:
    dataset = _dataset()
    pilot, selected_ids = select_scifact_pilot_dataset(dataset)
    queries = {query.query_id: query for query in dataset.queries}
    responses = {
        query_id: _response(
            queries[query_id], 0 if query_id in zero_hit_query_ids else candidate_count
        )
        for query_id in selected_ids
    }
    mapping = _mapping(responses)
    snapshot = build_scifact_candidate_snapshot(dataset, responses, mapping)
    return dataset, pilot, selected_ids, responses, mapping, snapshot


def _patch_score(
    monkeypatch: pytest.MonkeyPatch,
    score_function: Callable[[Sequence[str]], Sequence[float]],
    *,
    reverse_rows: bool = False,
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def fake_urlopen(request: urllib.request.Request, *, timeout: float) -> io.BytesIO:
        assert request.full_url == "http://127.0.0.1:8000/score"
        raw_data = request.data
        assert raw_data is not None
        payload = cast("dict[str, object]", json.loads(cast("bytes", raw_data)))
        documents = cast("list[str]", payload["documents"])
        calls.append({"timeout": timeout, **payload})
        scores = tuple(score_function(documents))
        rows = [{"index": index, "score": score} for index, score in enumerate(scores)]
        if reverse_rows:
            rows.reverse()
        body = json.dumps({"model": QWEN3_RERANKER_MODEL_ID, "data": rows}).encode()
        return io.BytesIO(body)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return calls


def _responses_from_scores(rows: object) -> io.BytesIO:
    return io.BytesIO(json.dumps({"model": QWEN3_RERANKER_MODEL_ID, "data": rows}).encode())


def _baseline_config(dataset: IrDataset, mapping: IrPassageMapping) -> IrExperimentConfig:
    return IrExperimentConfig(
        dataset_sha256=dataset.sha256,
        code_sha=_CODE_SHA,
        retrieval_revision="hybrid-rrf-v1",
        projection_sha256=_PROJECTION_SHA,
        parameters_json='{"candidate_window":50,"passage_mapping_sha256":"'
        + mapping.sha256
        + '","rrf_k":60}',
    )


def test_selection_is_deterministic_from_the_complete_query_id_universe() -> None:
    dataset = _dataset()
    query_ids = tuple(query.query_id for query in dataset.queries)
    selected = select_scifact_pilot_query_ids(query_ids)

    assert len(selected) == 20
    assert selected == select_scifact_pilot_query_ids(tuple(reversed(query_ids)))
    assert selected == select_scifact_pilot_dataset(dataset)[1]


def test_adapter_scores_all_frozen_top_50_candidates_and_preserves_retrieval_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = _dataset().queries[0]
    response = _response(query, 50)
    calls = _patch_score(monkeypatch, lambda documents: tuple(range(len(documents))))

    result = _adapter().rerank(query, response)

    assert result.candidate_count == 50
    assert result.request_count == 7
    assert [len(cast("list[str]", call["documents"])) for call in calls] == [8, 8, 8, 8, 8, 8, 2]
    assert [call["model"] for call in calls] == [QWEN3_RERANKER_MODEL_ID] * 7
    assert [call["timeout"] for call in calls] == [120.0] * 7
    by_passage = {candidate.passage_id: candidate for candidate in result.candidates}
    first = by_passage[f"{query.query_id}-p000"].payload()
    assert first["original_rrf_rank"] == 1
    retrieval = cast("dict[str, object]", first["original_retrieval"])
    assert retrieval["passage_key"] == f"{query.query_id}-p000"
    assert retrieval["original_rrf_score"] == pytest.approx(2 / 61)
    lexical = cast("dict[str, object]", retrieval["lexical"])
    dense = cast("dict[str, object]", retrieval["dense"])
    assert (lexical["rank"], lexical["raw_score"]) == (1, 99.0)
    assert (dense["rank"], dense["raw_score"]) == (1, 0.99)


@pytest.mark.parametrize("corruption", ("limit-10", "missing-best", "reordered", "empty"))
def test_frozen_fusion_must_equal_the_complete_saved_lane_rrf_top_50(
    monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    query = _dataset().queries[0]
    response = _response(query, 50)
    if corruption == "limit-10":
        invalid = _with_fused_hits(response, response.fusion.hits[:10])
    elif corruption == "missing-best":
        invalid = _with_fused_hits(
            response,
            tuple(
                hit.model_copy(update={"final_rank": rank})
                for rank, hit in enumerate(response.fusion.hits[1:], start=1)
            ),
        )
    elif corruption == "reordered":
        invalid = _with_fused_hits(
            response,
            tuple(
                hit.model_copy(update={"final_rank": rank})
                for rank, hit in enumerate(reversed(response.fusion.hits), start=1)
            ),
        )
    else:
        invalid = _with_fused_hits(response, ())

    def unexpected_score(*_args: object, **_kwargs: object) -> io.BytesIO:
        raise AssertionError("invalid frozen fusion must fail before a model request")

    monkeypatch.setattr(urllib.request, "urlopen", unexpected_score)
    with pytest.raises(IrContractError, match="complete deterministic RRF top-50"):
        _adapter().rerank(query, invalid)


def test_production_rrf_tie_order_is_checked_before_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = _dataset().queries[0]
    response = _response(query, 2)
    dense_candidates = (
        response.dense.candidates[1].model_copy(update={"rank": 1}),
        response.dense.candidates[0].model_copy(update={"rank": 2}),
    )
    dense = response.dense.model_copy(update={"candidates": dense_candidates})
    expected_fusion = reciprocal_rank_fusion(
        lexical=response.lexical.hits, dense=dense.candidates, limit=50
    )
    tied = response.model_copy(update={"dense": dense, "fusion": expected_fusion})
    assert tied.fusion.hits[0].rrf_score == tied.fusion.hits[1].rrf_score
    calls = _patch_score(monkeypatch, lambda documents: (0.5,) * len(documents))
    result = _adapter().rerank(query, tied)
    assert tuple(candidate.passage_id for candidate in result.candidates) == (
        f"{query.query_id}-p000",
        f"{query.query_id}-p001",
    )
    assert len(calls) == 1

    wrong_tie_order = _with_fused_hits(
        tied,
        tuple(
            hit.model_copy(update={"final_rank": rank})
            for rank, hit in enumerate(reversed(tied.fusion.hits), start=1)
        ),
    )
    with pytest.raises(IrContractError, match="complete deterministic RRF top-50"):
        _adapter().rerank(query, wrong_tie_order)
    assert len(calls) == 1


def test_response_indices_map_scores_back_to_their_documents_even_when_reordered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = _dataset().queries[0]
    _patch_score(monkeypatch, lambda _documents: (0.1, 0.2, 0.3), reverse_rows=True)

    result = _adapter().rerank(query, _response(query, 3))

    assert {candidate.passage_id: candidate.reranker_score for candidate in result.candidates} == {
        f"{query.query_id}-p000": 0.1,
        f"{query.query_id}-p001": 0.2,
        f"{query.query_id}-p002": 0.3,
    }


def test_minimal_b2_candidates_retain_the_frozen_mapping_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = _dataset().queries[0]
    response = _response(query, 9)
    candidates = qwen_scoring_candidates(query, response)
    assert tuple(candidate.original_rrf_rank for candidate in candidates) == tuple(range(1, 10))
    assert all(
        candidate.candidate_text_sha256 == hashlib.sha256(candidate.text.encode()).hexdigest()
        for candidate in candidates
    )

    _patch_score(
        monkeypatch,
        lambda documents: tuple(float(index) for index in range(len(documents))),
    )
    result = _adapter().rerank_frozen_candidates(
        query,
        retrieval_response_sha256="d" * 64,
        projection_sha256=response.dense.projection_sha256,
        candidates=candidates,
    )
    assert result.candidate_count == len(candidates)
    assert result.request_count == 2
    assert {candidate.passage_id for candidate in result.candidates} == {
        candidate.passage_id for candidate in candidates
    }

    with pytest.raises(IrContractError, match="rank or text digest"):
        QwenScoringCandidate(
            query_id=candidates[0].query_id,
            passage_id=candidates[0].passage_id,
            original_rrf_rank=candidates[0].original_rrf_rank,
            candidate_text_sha256="f" * 64,
            text=candidates[0].text,
            original_retrieval_json=candidates[0].original_retrieval_json,
        )
    with pytest.raises(IrContractError, match="unique contiguous query prefix"):
        _adapter().rerank_frozen_candidates(
            query,
            retrieval_response_sha256="d" * 64,
            projection_sha256=response.dense.projection_sha256,
            candidates=(candidates[0], dataclasses.replace(candidates[1], query_id="foreign")),
        )


def test_equal_scores_use_original_rrf_rank_as_a_stable_tie_breaker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = _dataset().queries[0]
    _patch_score(monkeypatch, lambda documents: (0.5,) * len(documents))
    adapter = _adapter()
    response = _response(query, 8)

    first = adapter.rerank(query, response)
    second = adapter.rerank(query, response)

    expected = tuple(f"{query.query_id}-p{index:03d}" for index in range(8))
    assert tuple(candidate.passage_id for candidate in first.candidates) == expected
    assert tuple(candidate.passage_id for candidate in second.candidates) == expected


@pytest.mark.parametrize(
    ("rows", "message"),
    (
        ([{"index": 0, "score": 0.1}, {"index": 0, "score": 0.2}], "duplicate or foreign index"),
        ([{"index": 0, "score": 0.1}, {"index": 2, "score": 0.2}], "duplicate or foreign index"),
        ([{"index": 0, "score": 0.1}], "cardinality differs"),
        ([{"index": 0, "score": float("nan")}, {"index": 1, "score": 0.2}], "finite numeric"),
        ([{"index": 0, "score": float("inf")}, {"index": 1, "score": 0.2}], "finite numeric"),
    ),
    ids=("duplicate", "foreign", "missing", "nan", "infinity"),
)
def test_invalid_score_rows_fail_closed(
    monkeypatch: pytest.MonkeyPatch, rows: object, message: str
) -> None:
    def fake_urlopen(*_args: object, **_kwargs: object) -> io.BytesIO:
        return _responses_from_scores(rows)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(IrContractError, match=message):
        _adapter().rerank(_dataset().queries[0], _response(_dataset().queries[0], 2))


def test_wrong_model_identity_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _responses_from_scores([{"index": 0, "score": 0.1}, {"index": 1, "score": 0.2}])
    response_data = cast("dict[str, object]", json.loads(response.getvalue()))
    body = json.dumps({"model": "Qwen/Qwen3-Reranker-4B", "data": response_data["data"]}).encode()

    def fake_urlopen(*_args: object, **_kwargs: object) -> io.BytesIO:
        return io.BytesIO(body)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(IrContractError, match="response model differs"):
        _adapter().rerank(_dataset().queries[0], _response(_dataset().queries[0], 2))


def test_http_failures_and_timeouts_are_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    query = _dataset().queries[0]
    response = _response(query, 2)
    calls = 0

    def http_failure(*_args: object, **_kwargs: object) -> io.BytesIO:
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError("http://127.0.0.1:8000/score", 503, "busy", Message(), None)

    monkeypatch.setattr(urllib.request, "urlopen", http_failure)
    with pytest.raises(urllib.error.HTTPError):
        _adapter().rerank(query, response)
    assert calls == 1

    def timeout(*_args: object, **_kwargs: object) -> io.BytesIO:
        nonlocal calls
        calls += 1
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", timeout)
    with pytest.raises(TimeoutError):
        _adapter(timeout_seconds=3.0).rerank(query, response)
    assert calls == 2


def test_global_session_deadline_caps_requests_and_stops_later_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = _dataset().queries[0]
    request_calls = _patch_score(monkeypatch, lambda documents: (0.5,) * len(documents))
    monotonic_values = iter((100.0, 100.1, 101.0))

    def fake_monotonic() -> float:
        return next(monotonic_values, 101.0)

    monkeypatch.setattr("dynamisrag.ir.qwen3_reranker.time.monotonic", fake_monotonic)
    adapter = Qwen3RerankerAdapter(
        timeout_seconds=120.0,
        session_deadline_monotonic=101.0,
    )

    with pytest.raises(TimeoutError, match="global SciFact pilot session deadline"):
        adapter.rerank(query, _response(query, 9))

    assert len(request_calls) == 1
    assert cast("float", request_calls[0]["timeout"]) == pytest.approx(0.9)


def test_expired_global_session_deadline_fails_before_the_first_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = _dataset().queries[0]
    request_calls = _patch_score(monkeypatch, lambda documents: (0.5,) * len(documents))
    monkeypatch.setattr("dynamisrag.ir.qwen3_reranker.time.monotonic", lambda: 200.0)
    adapter = Qwen3RerankerAdapter(session_deadline_monotonic=199.0)

    with pytest.raises(TimeoutError, match="global SciFact pilot session deadline"):
        adapter.rerank(query, _response(query, 1))

    assert not request_calls


def test_zero_and_short_responses_are_reported_without_hidden_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset()
    pilot, selected_ids = select_scifact_pilot_dataset(dataset)
    queries = {query.query_id: query for query in dataset.queries}
    responses = {
        query_id: _response(queries[query_id], 3 if query_id == selected_ids[0] else 0)
        for query_id in selected_ids
    }
    mapping = _mapping(responses)
    snapshot = build_scifact_candidate_snapshot(dataset, responses, mapping)
    calls = _patch_score(monkeypatch, lambda documents: tuple(0.5 for _ in documents))
    adapter = _adapter()
    results = tuple(
        adapter.rerank(queries[query_id], responses[query_id]) for query_id in selected_ids
    )

    short = next(result for result in results if result.query_id == selected_ids[0])
    assert short.candidate_count == 3
    assert short.request_count == 1
    assert sum(result.candidate_count == 0 for result in results) == 19
    assert len(calls) == 1
    config = qwen3_reranker_config(
        pilot,
        code_sha=_CODE_SHA,
        projection_sha256=snapshot.projection_sha256,
        candidate_snapshot=snapshot,
        passage_mapping=mapping,
    )
    with pytest.raises(IrContractError, match="source exhaustion"):
        qwen3_reranker_run(pilot, config, mapping, snapshot, results)

    run = qwen3_reranker_run(
        pilot,
        config,
        mapping,
        snapshot,
        results,
        source_exhausted_query_ids=(selected_ids[0],),
    )
    assert run.query_ids == tuple(query.query_id for query in pilot.queries)
    assert len(run.hits) == 2
    assert run.source_exhausted_query_ids == (selected_ids[0],)


def test_frozen_snapshot_binds_projection_mapping_and_all_original_candidate_traces() -> None:
    dataset, _pilot, selected_ids, responses, mapping, snapshot = _inputs(candidate_count=2)
    first_response = responses[selected_ids[0]]
    changed_embedding_config = first_response.dense.model_copy(
        update={"query_embedding_config_sha256": "9" * 64}
    )
    changed_embedding = {
        **responses,
        selected_ids[0]: first_response.model_copy(update={"dense": changed_embedding_config}),
    }
    with pytest.raises(IrContractError, match="mix retrieval/index/embedding identities"):
        build_scifact_candidate_snapshot(dataset, changed_embedding, mapping)

    changed_dense = first_response.dense.model_copy(update={"projection_sha256": "f" * 64})
    changed = {
        **responses,
        selected_ids[0]: first_response.model_copy(update={"dense": changed_dense}),
    }
    with pytest.raises(IrContractError, match="lane identities"):
        build_scifact_candidate_snapshot(dataset, changed, mapping)

    missing_mapping = IrPassageMapping(mapping.entries[1:])
    with pytest.raises(IrContractError, match="full passage map"):
        build_scifact_candidate_snapshot(dataset, responses, missing_mapping)

    first_entry = mapping.entries[0]
    wrong_mapping = IrPassageMapping(
        tuple(
            dataclasses.replace(entry, document_id="foreign-document")
            if entry.passage_id == first_entry.passage_id
            else entry
            for entry in mapping.entries
        )
    )
    with pytest.raises(IrContractError, match="disagrees with the full passage map"):
        build_scifact_candidate_snapshot(dataset, responses, wrong_mapping)

    with pytest.raises(IrContractError, match="fields disagree"):
        dataclasses.replace(snapshot, projection_sha256="f" * 64)


@pytest.mark.parametrize(
    ("identity_field", "value"),
    (
        ("vector_config_sha256", "9" * 64),
        ("query_embedding_config_sha256", "8" * 64),
        ("document_embedding_config_sha256", "7" * 64),
    ),
)
def test_snapshot_rejects_mixed_embedding_configuration(identity_field: str, value: str) -> None:
    dataset, _pilot, selected_ids, responses, mapping, _snapshot = _inputs(candidate_count=2)
    query_id = selected_ids[0]
    response = responses[query_id]
    changed_dense = response.dense.model_copy(update={identity_field: value})
    changed = {
        **responses,
        query_id: response.model_copy(update={"dense": changed_dense}),
    }

    with pytest.raises(IrContractError, match="mix retrieval/index/embedding identities"):
        build_scifact_candidate_snapshot(dataset, changed, mapping)


def test_snapshot_rejects_a_different_physical_index_even_with_same_projection() -> None:
    dataset, _pilot, selected_ids, responses, mapping, _snapshot = _inputs(candidate_count=2)
    query_id = selected_ids[0]
    response = responses[query_id]
    dense = response.dense.model_copy(update={"physical_index": "foreign-physical-index"})
    changed = {
        **responses,
        query_id: response.model_copy(
            update={"physical_index": "foreign-physical-index", "dense": dense}
        ),
    }

    with pytest.raises(IrContractError, match="mix retrieval/index/embedding identities"):
        build_scifact_candidate_snapshot(dataset, changed, mapping)


def test_response_rejects_an_unpinned_bm25_query_revision() -> None:
    query = _dataset().queries[0]
    response = _response(query, 2)
    lexical = response.lexical.model_copy(update={"query_revision": "bm25-v2"})
    changed = response.model_copy(update={"lexical": lexical})

    with pytest.raises(IrContractError, match="pinned BM25 query revision"):
        _adapter().rerank(query, changed)


@pytest.mark.parametrize(
    ("lane", "dense_only_key"),
    (("bm25", "000-dense-only"), ("dense", "zzz-dense-only")),
)
def test_snapshot_digest_seals_nonfused_bm25_and_dense_lane_rows(
    lane: str, dense_only_key: str
) -> None:
    dataset = _dataset()
    _pilot, selected_ids = select_scifact_pilot_dataset(dataset)
    queries = {query.query_id: query for query in dataset.queries}
    responses = {query_id: _response(queries[query_id], 50) for query_id in selected_ids}
    query_id = selected_ids[0]
    response = _response_with_dense_only_tail(queries[query_id], passage_key=dense_only_key)
    fused_ids = {hit.passage_key for hit in response.fusion.hits}
    if lane == "bm25":
        assert f"{query_id}-p049" not in {hit.passage_key for hit in response.fusion.hits}
        assert dense_only_key in fused_ids
    else:
        assert dense_only_key not in fused_ids
        assert f"{query_id}-p049" in fused_ids
    responses[query_id] = response
    mapping = _mapping(responses)
    snapshot = build_scifact_candidate_snapshot(dataset, responses, mapping)

    if lane == "bm25":
        changed_hits = tuple(
            hit.model_copy(update={"score": 42.0}) if hit.passage_key == f"{query_id}-p049" else hit
            for hit in response.lexical.hits
        )
        changed_lexical = response.lexical.model_copy(update={"hits": changed_hits})
        changed_response = response.model_copy(update={"lexical": changed_lexical})
    else:
        changed_candidates = tuple(
            candidate.model_copy(update={"raw_score": 0.123})
            if candidate.passage_key == dense_only_key
            else candidate
            for candidate in response.dense.candidates
        )
        changed_dense = response.dense.model_copy(update={"candidates": changed_candidates})
        changed_response = response.model_copy(update={"dense": changed_dense})
    changed_response = changed_response.model_copy(
        update={
            "fusion": reciprocal_rank_fusion(
                lexical=changed_response.lexical.hits,
                dense=changed_response.dense.candidates,
                limit=50,
            )
        }
    )
    changed_responses = {**responses, query_id: changed_response}

    changed_snapshot = build_scifact_candidate_snapshot(dataset, changed_responses, mapping)
    assert changed_snapshot.sha256 != snapshot.sha256
    assert len(mapping.entries) == 20 * 50 + 1


def test_snapshot_digest_binds_bm25_total_hit_count() -> None:
    dataset = _dataset()
    _pilot, selected_ids = select_scifact_pilot_dataset(dataset)
    queries = {query.query_id: query for query in dataset.queries}
    responses = {query_id: _response(queries[query_id], 50) for query_id in selected_ids}
    mapping = _mapping(responses)
    first = build_scifact_candidate_snapshot(dataset, responses, mapping)

    query_id = selected_ids[0]
    response = responses[query_id]
    changed_lexical = response.lexical.model_copy(update={"total": response.lexical.total + 1})
    changed_response = response.model_copy(update={"lexical": changed_lexical})
    changed = build_scifact_candidate_snapshot(
        dataset,
        {**responses, query_id: changed_response},
        mapping,
    )

    assert changed.sha256 != first.sha256


def test_snapshot_hash_and_ir_config_reject_dataset_projection_and_mapping_mismatches() -> None:
    _dataset_full, pilot, _selected_ids, _responses, mapping, snapshot = _inputs(candidate_count=1)
    config = qwen3_reranker_config(
        pilot,
        code_sha=_CODE_SHA,
        projection_sha256=snapshot.projection_sha256,
        candidate_snapshot=snapshot,
        passage_mapping=mapping,
    )
    assert config.projection_sha256 == snapshot.projection_sha256
    assert json.loads(config.parameters_json)["candidate_snapshot_sha256"] == snapshot.sha256

    with pytest.raises(IrContractError, match="projection differs"):
        qwen3_reranker_config(
            pilot,
            code_sha=_CODE_SHA,
            projection_sha256="f" * 64,
            candidate_snapshot=snapshot,
            passage_mapping=mapping,
        )
    with pytest.raises(IrContractError, match="passage mapping identities differ"):
        qwen3_reranker_config(
            pilot,
            code_sha=_CODE_SHA,
            projection_sha256=snapshot.projection_sha256,
            candidate_snapshot=snapshot,
            passage_mapping=IrPassageMapping(()),
        )


def test_reranking_validates_saved_fusion_without_backend_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = _dataset().queries[0]
    response = _response(query, 1)
    calls = _patch_score(monkeypatch, lambda _documents: (0.5,))

    def unexpected_backend_retrieval(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("reranking must not call the retrieval service")

    fusion_calls = 0

    def observed_fusion(
        *, lexical: Sequence[SearchHit], dense: Sequence[DenseCandidate], limit: int
    ) -> RrfTrace:
        nonlocal fusion_calls
        fusion_calls += 1
        return reciprocal_rank_fusion(lexical=lexical, dense=dense, limit=limit)

    monkeypatch.setattr(
        "dynamisrag.search.retrieval.HybridRetrievalService.retrieve",
        unexpected_backend_retrieval,
    )
    monkeypatch.setattr(
        "dynamisrag.ir.qwen3_reranker.reciprocal_rank_fusion",
        observed_fusion,
    )

    _adapter().rerank(query, response)

    assert len(calls) == 1
    assert fusion_calls == 1


def test_frozen_baseline_and_reranked_results_use_res140_and_pair_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, pilot, selected_ids, responses, mapping, snapshot = _inputs(
        candidate_count=24,
        zero_hit_query_ids=frozenset({select_scifact_pilot_dataset(_dataset())[1][0]}),
    )
    baseline_hits = frozen_hybrid_passage_hits(dataset, responses, mapping, snapshot)
    assert len(baseline_hits) == 19 * 24
    assert baseline_hits[0].raw_score == pytest.approx(2 / 61)

    baseline_config = _baseline_config(pilot, mapping)
    baseline_run = document_run_from_passages(
        dataset=pilot,
        config=baseline_config,
        passage_mapping=mapping,
        hits=baseline_hits,
        evaluation_depth=50,
    )
    assert baseline_run.query_ids == tuple(query.query_id for query in pilot.queries)
    assert len(baseline_run.hits) == 19 * 12

    reranker_config = qwen3_reranker_config(
        pilot,
        code_sha=_CODE_SHA,
        projection_sha256=snapshot.projection_sha256,
        candidate_snapshot=snapshot,
        passage_mapping=mapping,
    )
    _patch_score(monkeypatch, lambda documents: tuple(1.0 for _ in documents))
    queries = {query.query_id: query for query in dataset.queries}
    adapter = _adapter()
    results = tuple(
        adapter.rerank(queries[query_id], responses[query_id]) for query_id in selected_ids
    )
    assert len(ranked_passage_hits(pilot, results, snapshot)) == 19 * 24

    candidate_run = qwen3_reranker_run(pilot, reranker_config, mapping, snapshot, results)
    assert candidate_run.query_ids == baseline_run.query_ids
    assert len(candidate_run.hits) == 19 * 12
    assert len({(hit.query_id, hit.document_id) for hit in candidate_run.hits}) == len(
        candidate_run.hits
    )
    assert all(hit.source_passage_id is not None for hit in candidate_run.hits)

    baseline_evaluation = evaluate_ir_run(
        pilot, baseline_config, baseline_run, passage_mapping=mapping
    )
    candidate_evaluation = evaluate_ir_run(
        pilot, reranker_config, candidate_run, passage_mapping=mapping
    )
    comparison = compare_ir_evaluations(baseline_evaluation, candidate_evaluation)
    assert len(comparison.per_query) == 20 * 4
    assert tuple(row.measure for row in comparison.aggregate) == (
        "nDCG@10",
        "Recall@10",
        "MAP",
        "MRR",
    )


def test_ranked_hits_refuse_incomplete_duplicate_and_foreign_query_results() -> None:
    dataset, pilot, _selected_ids, _responses, mapping, snapshot = _inputs(candidate_count=0)
    results = tuple(
        QwenRerankedQuery(
            query_id=query.query_id,
            query_text_sha256=hashlib.sha256(query.text.encode()).hexdigest(),
            retrieval_response_sha256=dict(snapshot.response_sha256)[query.query_id],
            projection_sha256=snapshot.projection_sha256,
            candidate_count=0,
            request_count=0,
            elapsed_ms=0.0,
            candidates=(),
        )
        for query in pilot.queries
    )
    assert len(ranked_passage_hits(pilot, results, snapshot)) == 0
    with pytest.raises(IrContractError, match="cover each pilot query"):
        ranked_passage_hits(pilot, results[:-1], snapshot)
    with pytest.raises(IrContractError, match="cover each pilot query"):
        ranked_passage_hits(pilot, (*results[:-1], results[0]), snapshot)
    foreign = dataclasses.replace(results[-1], query_id="foreign-query")
    with pytest.raises(IrContractError, match="cover each pilot query"):
        ranked_passage_hits(pilot, (*results[:-1], foreign), snapshot)
    assert not mapping.entries
    assert dataset.source_id == "beir.scifact"
