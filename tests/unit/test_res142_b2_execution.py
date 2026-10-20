"""RES-142 B2 transfer execution, import-smoke, and failure-boundary tests."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import cast

import pytest

from dynamisrag.ir.b2_execution import (
    b2_queries_from_payload,
    score_b2_queries,
)
from dynamisrag.ir.contracts import IrContractError
from dynamisrag.ir.qwen3_reranker import (
    QWEN3_RERANKER_MODEL_ID,
    Qwen3RerankerAdapter,
    QwenRerankedQuery,
)


def _payload() -> tuple[dict[str, object], list[str]]:
    query_ids = [f"q{index:02d}" for index in range(20)]
    rows: list[dict[str, object]] = []
    for query_id in query_ids:
        text = f"claim {query_id}"
        candidates: list[dict[str, object]] = []
        for rank in range(1, 51):
            candidate_text = f"abstract {query_id} {rank}"
            text_sha256 = hashlib.sha256(candidate_text.encode()).hexdigest()
            passage_id = f"{query_id}-p{rank:02d}"
            identity = {
                "candidate_text_sha256": text_sha256,
                "original_rrf_rank": rank,
                "original_rrf_score": 1 / rank,
                "passage_key": passage_id,
            }
            candidates.append(
                {
                    "candidate_text_sha256": text_sha256,
                    "original_retrieval": identity,
                    "original_rrf_rank": rank,
                    "passage_id": passage_id,
                    "query_id": query_id,
                    "text": candidate_text,
                }
            )
        rows.append(
            {
                "candidate_count": 50,
                "candidates": candidates,
                "query_id": query_id,
                "query_text": text,
                "query_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "retrieval_response_sha256": hashlib.sha256(query_id.encode()).hexdigest(),
            }
        )
    return (
        {
            "candidate_count": 1000,
            "candidate_limit": 50,
            "projection_sha256": "a" * 64,
            "query_count": 20,
            "queries_in_selection_order": rows,
            "schema_revision": "res142-scifact-b2-candidate-input-v1",
        },
        query_ids,
    )


def _write_smoke_zip(path: Path, payload: dict[str, object], query_ids: list[str]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("candidate-input.json", json.dumps(payload, sort_keys=True))
        archive.writestr(
            "pilot-query-selection.json",
            json.dumps({"selected_query_ids": query_ids}, sort_keys=True),
        )


def _run_import_smoke(input_zip: Path) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).parents[2]
    runner = root / "notebooks" / "res142_b2_runner.py"
    return subprocess.run(  # noqa: S603 - fixed local script and arguments; no untrusted command input.
        [
            sys.executable,
            str(runner),
            "--input-zip",
            str(input_zip),
            "--support-dir",
            str(root / "notebooks"),
            "--smoke-only",
            "--source-root",
            str(root),
            "--source-commit",
            "85024771215947e0315f37a6dd67d1e8d0575eff",
        ],
        capture_output=True,
        check=False,
        cwd=root,
        text=True,
        timeout=30,
    )


def test_import_smoke_subprocess_imports_real_scorers_and_reconstructs_all_pairs(
    tmp_path: Path,
) -> None:
    payload, query_ids = _payload()
    archive = tmp_path / "b2-import-smoke.zip"
    _write_smoke_zip(archive, payload, query_ids)

    result = _run_import_smoke(archive)

    assert result.returncode == 0, result.stderr
    smoke = cast("dict[str, object]", json.loads(result.stdout))
    assert smoke["status"] == "verified"
    assert smoke["source_commit"] == "85024771215947e0315f37a6dd67d1e8d0575eff"
    assert smoke["query_count"] == 20
    assert smoke["candidate_count"] == 1000
    assert smoke["selected_query_ids"] == query_ids
    assert smoke["adapter_class"] == "dynamisrag.ir.qwen3_reranker.Qwen3RerankerAdapter"
    assert smoke["sqlalchemy_version"] == "2.1.1"
    assert "sqlalchemy" in {
        str(name).casefold().replace("_", "-")
        for name in cast("list[object]", smoke["imported_distributions"])
    }
    mapping_sha256 = cast("str", smoke["candidate_mapping_sha256"])
    assert len(mapping_sha256) == 64
    assert all(character in "0123456789abcdef" for character in mapping_sha256)


@pytest.mark.parametrize("invalid_population", ("short-query", "wrong-passage-map"))
def test_import_smoke_subprocess_rejects_invalid_candidate_populations(
    tmp_path: Path, invalid_population: str
) -> None:
    payload, query_ids = _payload()
    rows = cast("list[dict[str, object]]", payload["queries_in_selection_order"])
    candidates = cast("list[dict[str, object]]", rows[0]["candidates"])
    if invalid_population == "short-query":
        candidates.pop()
    else:
        candidates[0]["passage_id"] = "foreign-passage"
    archive = tmp_path / f"invalid-{invalid_population}.zip"
    _write_smoke_zip(archive, payload, query_ids)

    result = _run_import_smoke(archive)

    assert result.returncode != 0
    assert "IrContractError" in result.stderr


def _mock_adapter(monkeypatch: pytest.MonkeyPatch) -> tuple[Qwen3RerankerAdapter, list[object]]:
    requests: list[object] = []

    def fake_urlopen(request: urllib.request.Request, *, timeout: float) -> io.BytesIO:
        assert request.full_url == "http://127.0.0.1:8000/score"
        assert timeout > 0
        body = request.data
        assert body is not None
        request_payload = cast("dict[str, object]", json.loads(cast("bytes", body)))
        documents = cast("list[str]", request_payload["documents"])
        requests.append((request_payload["queries"], documents))
        rows = [{"index": index, "score": index / 10} for index in range(len(documents))]
        return io.BytesIO(
            json.dumps({"model": QWEN3_RERANKER_MODEL_ID, "data": list(reversed(rows))}).encode()
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    adapter = Qwen3RerankerAdapter(session_deadline_monotonic=time.monotonic() + 600)
    return adapter, requests


def test_mock_b2_execution_scores_1000_pairs_and_checkpoints_each_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, query_ids = _payload()
    queries = b2_queries_from_payload(payload, query_ids)
    adapter, requests = _mock_adapter(monkeypatch)
    checkpoints: list[tuple[str, int, int]] = []

    results = score_b2_queries(
        queries,
        adapter,
        on_query_complete=lambda item, result: checkpoints.append(
            (item.query.query_id, result.candidate_count, result.request_count)
        ),
        work_deadline_monotonic=time.monotonic() + 600,
    )

    assert len(results) == 20
    assert len(requests) == 140
    assert checkpoints == [(query_id, 50, 7) for query_id in query_ids]
    result = results[0]
    by_passage = {candidate.passage_id: candidate for candidate in result.candidates}
    assert by_passage["q00-p08"].reranker_score == pytest.approx(0.7)
    assert by_passage["q00-p01"].reranker_score == pytest.approx(0.0)
    assert by_passage["q00-p50"].reranker_score == pytest.approx(0.1)


def test_b2_payload_rejects_candidate_text_digest_tampering() -> None:
    payload, query_ids = _payload()
    rows = cast("list[dict[str, object]]", payload["queries_in_selection_order"])
    candidates = cast("list[dict[str, object]]", rows[0]["candidates"])
    candidates[0]["candidate_text_sha256"] = "b" * 64

    with pytest.raises(IrContractError, match="text digest"):
        b2_queries_from_payload(payload, query_ids)


def test_b2_payload_rejects_foreign_query_order() -> None:
    payload, query_ids = _payload()
    with pytest.raises(IrContractError, match="frozen selection"):
        b2_queries_from_payload(payload, list(reversed(query_ids)))


def test_b2_stops_before_scoring_after_work_deadline() -> None:
    payload, query_ids = _payload()
    queries = b2_queries_from_payload(payload, query_ids)
    adapter = Qwen3RerankerAdapter(session_deadline_monotonic=time.monotonic() + 60)
    completed: list[QwenRerankedQuery] = []

    with pytest.raises(TimeoutError, match="work deadline"):
        score_b2_queries(
            queries,
            adapter,
            on_query_complete=lambda _item, result: completed.append(result),
            work_deadline_monotonic=0,
        )
    assert completed == []
