"""Run one query at a time through the frozen RES-142 B2 reranker adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast


def _write_json_create_only(path: Path, payload: object) -> bytes:
    data = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
    return data


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--support-dir", type=Path, required=True)
    parser.add_argument("--drive-state", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--work-deadline-monotonic", type=float, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.support_dir))
    from res142_b1_support import persist_drive_bytes, persist_drive_checkpoint, sanitize_output

    sys.path.insert(0, str(args.support_dir / "dynamisrag-src" / "src"))
    from dynamisrag.ir.b2_execution import b2_queries_from_payload, score_b2_queries
    from dynamisrag.ir.qwen3_reranker import Qwen3RerankerAdapter

    state_raw = json.loads(args.drive_state.read_text(encoding="utf-8"))
    if not isinstance(state_raw, dict):
        raise ValueError("Drive run state must be a JSON object")
    state = cast("dict[str, object]", state_raw)
    with zipfile.ZipFile(args.input_zip) as archive:
        candidate_payload = json.loads(archive.read("candidate-input.json"))
        selection = json.loads(archive.read("pilot-query-selection.json"))
    if not isinstance(candidate_payload, dict) or not isinstance(selection, dict):
        raise ValueError("B2 candidate ZIP payload or selection is invalid")
    selected_ids = selection.get("selected_query_ids")
    if not isinstance(selected_ids, list) or any(
        not isinstance(value, str) for value in selected_ids
    ):
        raise ValueError("B2 selected query IDs are invalid")
    queries = b2_queries_from_payload(cast("Mapping[str, object]", candidate_payload), selected_ids)
    adapter = Qwen3RerankerAdapter(
        session_deadline_monotonic=args.work_deadline_monotonic,
        timeout_seconds=120.0,
    )
    completed: list[dict[str, Any]] = []
    checkpoint_number = 0

    def checkpoint_query(item: object, result: object) -> None:
        nonlocal checkpoint_number
        checkpoint_number += 1
        query_input = cast("Any", item)
        query_result = cast("Any", result)
        payload = query_result.payload()
        result_bytes = _write_json_create_only(
            args.output_dir
            / "queries"
            / f"{checkpoint_number:02d}-{query_input.query.query_id}.json",
            payload,
        )
        persist_drive_bytes(
            state,
            args.staging_dir,
            f"queries/{checkpoint_number:02d}-{query_input.query.query_id}.json",
            result_bytes,
            deadline=args.work_deadline_monotonic,
        )
        checkpoint = {
            "query_id": query_input.query.query_id,
            "candidate_count": query_result.candidate_count,
            "request_count": query_result.request_count,
            "elapsed_ms": query_result.elapsed_ms,
            "scores_sha256": hashlib.sha256(result_bytes).hexdigest(),
        }
        persist_drive_checkpoint(
            state,
            args.staging_dir,
            f"query-{checkpoint_number:02d}",
            checkpoint,
            deadline=args.work_deadline_monotonic,
        )
        completed.append(payload)
        state_data = (json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n").encode()
        temporary = args.drive_state.with_suffix(".tmp")
        temporary.write_bytes(state_data)
        temporary.replace(args.drive_state)
        print(json.dumps(checkpoint, sort_keys=True), flush=True)

    try:
        score_b2_queries(
            queries,
            adapter,
            on_query_complete=checkpoint_query,
            work_deadline_monotonic=args.work_deadline_monotonic,
        )
        results = {"schema_revision": "res142-qwen3-reranked-results-v1", "queries": completed}
        results_bytes = _write_json_create_only(args.output_dir / "results.json", results)
        persist_drive_bytes(
            state,
            args.staging_dir,
            "scores/results.json",
            results_bytes,
            deadline=args.work_deadline_monotonic,
        )
        persist_drive_checkpoint(
            state,
            args.staging_dir,
            "all-queries-complete",
            {
                "query_count": len(completed),
                "candidate_count": sum(row["candidate_count"] for row in completed),
                "results_sha256": hashlib.sha256(results_bytes).hexdigest(),
            },
            deadline=args.work_deadline_monotonic,
        )
        args.drive_state.write_text(
            json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
        )
        print(json.dumps({"status": "complete", "query_count": len(completed)}, sort_keys=True))
        return 0
    except BaseException as error:
        failure = {
            "stage": "b2_query_scoring",
            "error_type": type(error).__name__,
            "message": sanitize_output(error, limit=2_000),
            "traceback": sanitize_output(traceback.format_exc(), limit=12_000),
            "completed_query_count": len(completed),
            "drive_state": state,
        }
        _write_json_create_only(args.output_dir / "failure.json", failure)
        print(json.dumps({"status": "failed", **failure}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
