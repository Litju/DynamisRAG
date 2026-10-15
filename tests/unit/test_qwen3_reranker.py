"""RES-142 adapter qualification against the frozen retrieval and RES-140 contracts."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
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
    Qwen3RerankerAdapter,
    QwenRerankedQuery,
    SciFactCandidateSnapshot,
    build_scifact_candidate_snapshot,
    frozen_hybrid_passage_hits,
    qwen3_reranker_config,
    qwen3_reranker_run,
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
)

_PROJECTION_SHA = "b" * 64
_CODE_SHA = "c" * 40
_INDEX = "scifact-passage-index-v2-test"
_CHUNKER = "structure-v1.test"


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


def _mapping(
    responses: Mapping[str, HybridRetrievalResponse],
) -> IrPassageMapping:
    entries = {
        hit.passage_key: IrPassageMapEntry(
            passage_id=hit.passage_key,
            document_id=hit.provenance.document_canonical_key,
            document_version_id=hit.provenance.document_version_key,
        )
        for response in responses.values()
        for hit in response.fusion.hits
    }
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

    result = Qwen3RerankerAdapter().rerank(query, response)

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


def test_response_indices_map_scores_back_to_their_documents_even_when_reordered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = _dataset().queries[0]
    _patch_score(monkeypatch, lambda _documents: (0.1, 0.2, 0.3), reverse_rows=True)

    result = Qwen3RerankerAdapter().rerank(query, _response(query, 3))

    assert {candidate.passage_id: candidate.reranker_score for candidate in result.candidates} == {
        f"{query.query_id}-p000": 0.1,
        f"{query.query_id}-p001": 0.2,
        f"{query.query_id}-p002": 0.3,
    }


def test_equal_scores_use_original_rrf_rank_as_a_stable_tie_breaker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = _dataset().queries[0]
    _patch_score(monkeypatch, lambda documents: (0.5,) * len(documents))
    adapter = Qwen3RerankerAdapter()
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
        Qwen3RerankerAdapter().rerank(_dataset().queries[0], _response(_dataset().queries[0], 2))


def test_wrong_model_identity_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _responses_from_scores([{"index": 0, "score": 0.1}, {"index": 1, "score": 0.2}])
    response_data = cast("dict[str, object]", json.loads(response.getvalue()))
    body = json.dumps({"model": "Qwen/Qwen3-Reranker-4B", "data": response_data["data"]}).encode()

    def fake_urlopen(*_args: object, **_kwargs: object) -> io.BytesIO:
        return io.BytesIO(body)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(IrContractError, match="response model differs"):
        Qwen3RerankerAdapter().rerank(_dataset().queries[0], _response(_dataset().queries[0], 2))


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
        Qwen3RerankerAdapter().rerank(query, response)
    assert calls == 1

    def timeout(*_args: object, **_kwargs: object) -> io.BytesIO:
        nonlocal calls
        calls += 1
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", timeout)
    with pytest.raises(TimeoutError):
        Qwen3RerankerAdapter(timeout_seconds=3.0).rerank(query, response)
    assert calls == 2


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
    adapter = Qwen3RerankerAdapter()
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
    assert build_scifact_candidate_snapshot(dataset, changed_embedding, mapping).sha256 != (
        snapshot.sha256
    )

    changed_dense = first_response.dense.model_copy(update={"projection_sha256": "f" * 64})
    changed = {
        **responses,
        selected_ids[0]: first_response.model_copy(update={"dense": changed_dense}),
    }
    with pytest.raises(IrContractError, match="share one physical projection"):
        build_scifact_candidate_snapshot(dataset, changed, mapping)

    missing_mapping = IrPassageMapping(mapping.entries[1:])
    with pytest.raises(IrContractError, match="missing from the snapshot-wide passage map"):
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
    with pytest.raises(IrContractError, match="disagrees with the passage map"):
        build_scifact_candidate_snapshot(dataset, responses, wrong_mapping)

    with pytest.raises(IrContractError, match="fields disagree"):
        dataclasses.replace(snapshot, projection_sha256="f" * 64)


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


def test_no_retrieval_lane_or_fusion_is_called_during_reranking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = _dataset().queries[0]
    response = _response(query, 1)
    calls = _patch_score(monkeypatch, lambda _documents: (0.5,))

    def unexpected_retrieval(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("reranking must consume the saved response only")

    monkeypatch.setattr(
        "dynamisrag.search.retrieval.HybridRetrievalService.retrieve", unexpected_retrieval
    )
    monkeypatch.setattr("dynamisrag.search.retrieval.reciprocal_rank_fusion", unexpected_retrieval)

    Qwen3RerankerAdapter().rerank(query, response)

    assert len(calls) == 1


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
    results = tuple(
        Qwen3RerankerAdapter().rerank(queries[query_id], responses[query_id])
        for query_id in selected_ids
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
