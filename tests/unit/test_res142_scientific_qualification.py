from __future__ import annotations

import copy
import hashlib
from typing import cast

import pytest

from dynamisrag.ir.contracts import IrContractError
from dynamisrag.ir.qwen3_reranker import (
    QWEN3_RERANKER_INSTRUCTION,
    QWEN3_RERANKER_INSTRUCTION_SHA256,
    QWEN3_RERANKER_MODEL_ID,
    QWEN3_RERANKER_REVISION,
    QWEN3_RERANKER_TEMPLATE_SHA256,
    QWEN3_RERANKER_TOKENIZER_REVISION,
    QWEN3_RERANKER_VLLM_VERSION,
)
from dynamisrag.ir.res142_qualification import (
    exact_sign_flip_test,
    exact_sign_test,
    holm_adjust,
    paired_bootstrap_intervals,
    restore_qwen_results,
)

_PROJECTION = "a" * 64


def _frozen_and_scored(
    query_id: str, *, tied_top_scores: bool = False
) -> tuple[dict[str, object], dict[str, object]]:
    query_text = f"frozen claim for {query_id}"
    query_text_sha256 = hashlib.sha256(query_text.encode()).hexdigest()
    response_sha256 = hashlib.sha256(f"retrieval-{query_id}".encode()).hexdigest()
    inputs: list[dict[str, object]] = []
    outputs: list[dict[str, object]] = []
    for rank in range(1, 51):
        passage_id = f"{query_id}-p{rank:03d}"
        text = f"passage {rank} for {query_id}"
        text_sha256 = hashlib.sha256(text.encode()).hexdigest()
        original = {
            "passage_key": passage_id,
            "original_rrf_rank": rank,
            "original_rrf_score": 1.0 / (rank + 60),
            "candidate_text_sha256": text_sha256,
            "provenance": {"source_external_id": f"{query_id}-doc-{rank}"},
        }
        inputs.append(
            {
                "query_id": query_id,
                "passage_id": passage_id,
                "original_rrf_rank": rank,
                "candidate_text_sha256": text_sha256,
                "text": text,
                "original_retrieval": original,
            }
        )
        score = 1.0 if tied_top_scores and rank <= 2 else float(51 - rank)
        outputs.append(
            {
                "query_id": query_id,
                "passage_id": passage_id,
                "original_rrf_rank": rank,
                "reranked_rank": rank,
                "reranker_score": score,
                "original_retrieval": original,
            }
        )
    return (
        {
            "query_id": query_id,
            "query_text": query_text,
            "query_text_sha256": query_text_sha256,
            "retrieval_response_sha256": response_sha256,
            "candidate_count": 50,
            "candidates": inputs,
        },
        {
            "schema_revision": "res142-qwen-reranked-query-v1",
            "query_id": query_id,
            "query_text_sha256": query_text_sha256,
            "retrieval_response_sha256": response_sha256,
            "projection_sha256": _PROJECTION,
            "candidate_count": 50,
            "request_count": 7,
            "elapsed_ms": 10.0,
            "reranker": {
                "adapter_revision": "qwen3-reranker-adapter-v1",
                "model_id": QWEN3_RERANKER_MODEL_ID,
                "model_revision": QWEN3_RERANKER_REVISION,
                "tokenizer_revision": QWEN3_RERANKER_TOKENIZER_REVISION,
                "template_sha256": QWEN3_RERANKER_TEMPLATE_SHA256,
                "vllm_version": QWEN3_RERANKER_VLLM_VERSION,
                "dtype": "bfloat16",
                "instruction": QWEN3_RERANKER_INSTRUCTION,
                "instruction_sha256": QWEN3_RERANKER_INSTRUCTION_SHA256,
                "use_activation": True,
                "tie_break": "score_desc,original_rrf_rank_asc",
            },
            "ranked_candidates": outputs,
        },
    )


def _population() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = [_frozen_and_scored(f"q{index:02d}") for index in range(20)]
    return [row[0] for row in rows], [row[1] for row in rows]


def test_restore_qwen_results_accepts_exact_frozen_population_and_scores() -> None:
    candidates, results = _population()

    restored = restore_qwen_results(
        candidates,
        results,
        expected_query_ids=tuple(f"q{index:02d}" for index in range(20)),
        expected_projection_sha256=_PROJECTION,
    )

    assert len(restored) == 20
    assert sum(query.candidate_count for query in restored) == 1000
    assert sum(query.request_count for query in restored) == 140
    assert [candidate.reranked_rank for candidate in restored[0].candidates] == list(range(1, 51))


def test_restore_qwen_results_rejects_missing_fabricated_nonfinite_and_bad_tie_order() -> None:
    candidates, results = _population()
    missing = copy.deepcopy(results)
    missing[0]["ranked_candidates"] = cast(list[object], missing[0]["ranked_candidates"])[:-1]
    with pytest.raises(IrContractError, match="exactly 50"):
        restore_qwen_results(
            candidates,
            missing,
            expected_query_ids=tuple(f"q{index:02d}" for index in range(20)),
            expected_projection_sha256=_PROJECTION,
        )

    duplicated = copy.deepcopy(results)
    duplicated_rows = cast(list[dict[str, object]], duplicated[0]["ranked_candidates"])
    duplicated_rows[1]["passage_id"] = duplicated_rows[0]["passage_id"]
    with pytest.raises(IrContractError, match="duplicated passage"):
        restore_qwen_results(
            candidates,
            duplicated,
            expected_query_ids=tuple(f"q{index:02d}" for index in range(20)),
            expected_projection_sha256=_PROJECTION,
        )

    fabricated = copy.deepcopy(results)
    fabricated_rows = cast(list[dict[str, object]], fabricated[0]["ranked_candidates"])
    fabricated_rows[0]["passage_id"] = "not-in-the-frozen-pool"
    with pytest.raises(IrContractError, match="absent from the frozen candidate pool"):
        restore_qwen_results(
            candidates,
            fabricated,
            expected_query_ids=tuple(f"q{index:02d}" for index in range(20)),
            expected_projection_sha256=_PROJECTION,
        )

    nonfinite = copy.deepcopy(results)
    nonfinite_rows = cast(list[dict[str, object]], nonfinite[0]["ranked_candidates"])
    nonfinite_rows[0]["reranker_score"] = float("nan")
    with pytest.raises(IrContractError, match="finite number"):
        restore_qwen_results(
            candidates,
            nonfinite,
            expected_query_ids=tuple(f"q{index:02d}" for index in range(20)),
            expected_projection_sha256=_PROJECTION,
        )

    tied_input, tied_result = _frozen_and_scored("q00", tied_top_scores=True)
    tied_rows = cast(list[dict[str, object]], tied_result["ranked_candidates"])
    first, second = tied_rows[:2]
    tied_rows[:2] = [
        {**second, "reranked_rank": 1},
        {**first, "reranked_rank": 2},
    ]
    candidates[0] = tied_input
    results[0] = tied_result
    with pytest.raises(IrContractError, match="tie-break"):
        restore_qwen_results(
            candidates,
            results,
            expected_query_ids=tuple(f"q{index:02d}" for index in range(20)),
            expected_projection_sha256=_PROJECTION,
        )


def test_exact_paired_tests_and_holm_adjustment_match_small_sample_cases() -> None:
    assert exact_sign_flip_test((1.0, 1.0, 1.0, 1.0)) == (0.125, 16)
    assert exact_sign_test((1.0, 1.0, 1.0, 1.0)) == (0.125, 4)
    assert exact_sign_test((1.0, 0.0, -1.0)) == (1.0, 2)
    assert holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.20}) == {
        "a": 0.04,
        "c": 0.09,
        "b": 0.09,
        "d": 0.20,
    }


def test_paired_query_bootstrap_is_reproducible_for_a_fixed_seed() -> None:
    differences = {"metric": (-1.0, 0.0, 1.0, 2.0)}

    first = paired_bootstrap_intervals(differences, seed=1422026, samples=1000)
    second = paired_bootstrap_intervals(differences, seed=1422026, samples=1000)

    assert first == second
    assert first["metric"][0] <= 0.5 <= first["metric"][1]
