"""Run one query at a time through the frozen RES-142 B2 reranker adapter."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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


def _import_smoke(
    *,
    input_zip: Path,
    support_dir: Path,
    source_root: Path,
    source_commit: str,
) -> dict[str, object]:
    """Import the real scorer and reconstruct every frozen candidate mapping."""
    modules_before = set(sys.modules)
    sys.path.insert(0, str(support_dir))
    import res142_b1_support  # noqa: F401

    source_path = source_root / "src"
    if not source_path.is_dir():
        raise FileNotFoundError(f"B2 source checkout is missing {source_path}.")
    sys.path.insert(0, str(source_path))
    from dynamisrag.ir.b2_execution import b2_queries_from_payload
    from dynamisrag.ir.qwen3_reranker import Qwen3RerankerAdapter

    archive_bytes = input_zip.read_bytes()
    with zipfile.ZipFile(input_zip) as archive:
        payload = json.loads(archive.read("candidate-input.json"))
        selection = json.loads(archive.read("pilot-query-selection.json"))
    if not isinstance(payload, dict) or not isinstance(selection, dict):
        raise ValueError("B2 import smoke requires object candidate and selection payloads.")
    selected_ids = selection.get("selected_query_ids")
    if not isinstance(selected_ids, list) or any(not isinstance(value, str) for value in selected_ids):
        raise ValueError("B2 import smoke requires the frozen query-ID selection.")
    source_rows = payload.get("queries_in_selection_order")
    if not isinstance(source_rows, list):
        raise ValueError("B2 import smoke requires selection-ordered query rows.")

    queries = b2_queries_from_payload(payload, selected_ids)
    if len(queries) != 20 or len(source_rows) != 20:
        raise ValueError("B2 import smoke reconstructed a non-20-query population.")
    if [item.query.query_id for item in queries] != selected_ids:
        raise ValueError("B2 import smoke changed the frozen query order.")

    mapping: list[dict[str, object]] = []
    candidate_count = 0
    for item, row_value in zip(queries, source_rows, strict=True):
        if not isinstance(row_value, dict):
            raise ValueError("B2 import smoke encountered a non-object query row.")
        row = cast("dict[str, object]", row_value)
        if item.query.query_id != row.get("query_id") or item.query.text != row.get("query_text"):
            raise ValueError("B2 import smoke changed a frozen query identity.")
        source_candidates = row.get("candidates")
        if not isinstance(source_candidates, list) or len(item.candidates) != 50:
            raise ValueError("B2 import smoke reconstructed an invalid candidate population.")
        candidate_mapping: list[dict[str, object]] = []
        for candidate, candidate_value in zip(item.candidates, source_candidates, strict=True):
            if not isinstance(candidate_value, dict):
                raise ValueError("B2 import smoke encountered a non-object candidate row.")
            source = cast("dict[str, object]", candidate_value)
            original_retrieval = json.loads(candidate.original_retrieval_json)
            if (
                candidate.query_id != source.get("query_id")
                or candidate.passage_id != source.get("passage_id")
                or candidate.original_rrf_rank != source.get("original_rrf_rank")
                or candidate.candidate_text_sha256 != source.get("candidate_text_sha256")
                or candidate.text != source.get("text")
                or original_retrieval != source.get("original_retrieval")
            ):
                raise ValueError("B2 import smoke changed a frozen candidate mapping.")
            candidate_mapping.append(
                {
                    "passage_id": candidate.passage_id,
                    "original_rrf_rank": candidate.original_rrf_rank,
                    "candidate_text_sha256": candidate.candidate_text_sha256,
                    "original_retrieval": original_retrieval,
                }
            )
        candidate_count += len(candidate_mapping)
        mapping.append(
            {
                "query_id": item.query.query_id,
                "query_text_sha256": row.get("query_text_sha256"),
                "retrieval_response_sha256": item.retrieval_response_sha256,
                "projection_sha256": item.projection_sha256,
                "candidates": candidate_mapping,
            }
        )
    if candidate_count != 1_000:
        raise ValueError("B2 import smoke reconstructed a non-1,000-candidate population.")

    mapping_bytes = json.dumps(
        mapping, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    modules_after = set(sys.modules)
    external_module_roots = sorted(
        {
            module.partition(".")[0]
            for module in modules_after - modules_before
            if module.partition(".")[0] not in sys.stdlib_module_names
            and module.partition(".")[0] not in {"dynamisrag", "res142_b1_support"}
        }
    )
    distributions_by_module = importlib.metadata.packages_distributions()
    imported_distributions = sorted(
        {
            distribution
            for module in external_module_roots
            for distribution in distributions_by_module.get(module, [])
        },
        key=str.casefold,
    )
    imported_distribution_versions = {
        distribution: importlib.metadata.version(distribution)
        for distribution in imported_distributions
    }
    imported_modules = sorted(modules_after - modules_before)
    third_party_modules = sorted(
        module
        for module in imported_modules
        if module.partition(".")[0] not in sys.stdlib_module_names
        and module.partition(".")[0] not in {"dynamisrag", "res142_b1_support"}
    )
    return {
        "schema_revision": "res142-b2-import-smoke-v1",
        "status": "verified",
        "source_commit": source_commit,
        "b2_execution_sha256": hashlib.sha256(
            (source_path / "dynamisrag" / "ir" / "b2_execution.py").read_bytes()
        ).hexdigest(),
        "input_zip_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "query_count": len(queries),
        "candidate_count": candidate_count,
        "selected_query_ids": selected_ids,
        "candidate_mapping_sha256": hashlib.sha256(mapping_bytes).hexdigest(),
        "adapter_class": f"{Qwen3RerankerAdapter.__module__}.{Qwen3RerankerAdapter.__qualname__}",
        "sqlalchemy_version": importlib.metadata.version("sqlalchemy"),
        "imported_modules": imported_modules,
        "third_party_modules": third_party_modules,
        "imported_module_roots": external_module_roots,
        "imported_distributions": imported_distributions,
        "imported_distribution_versions": imported_distribution_versions,
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--support-dir", type=Path, required=True)
    parser.add_argument("--drive-state", type=Path)
    parser.add_argument("--staging-dir", type=Path)
    parser.add_argument("--work-deadline-monotonic", type=float)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    if args.smoke_only:
        if args.source_root is None or args.source_commit is None:
            parser.error("--smoke-only requires --source-root and --source-commit")
        smoke = _import_smoke(
            input_zip=args.input_zip,
            support_dir=args.support_dir,
            source_root=args.source_root,
            source_commit=args.source_commit,
        )
        print(json.dumps(smoke, sort_keys=True))
        return 0
    if (
        args.output_dir is None
        or args.drive_state is None
        or args.staging_dir is None
        or args.work_deadline_monotonic is None
    ):
        parser.error(
            "scoring requires --output-dir, --drive-state, --staging-dir, "
            "and --work-deadline-monotonic"
        )
    output_dir = cast("Path", args.output_dir)
    drive_state = cast("Path", args.drive_state)
    staging_dir = cast("Path", args.staging_dir)
    work_deadline = cast("float", args.work_deadline_monotonic)
    output_dir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.support_dir))
    from res142_b1_support import persist_drive_bytes, persist_drive_checkpoint, sanitize_output

    sys.path.insert(0, str(args.support_dir / "dynamisrag-src" / "src"))
    from dynamisrag.ir.b2_execution import b2_queries_from_payload, score_b2_queries
    from dynamisrag.ir.qwen3_reranker import Qwen3RerankerAdapter

    state_raw = json.loads(drive_state.read_text(encoding="utf-8"))
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
        session_deadline_monotonic=work_deadline,
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
            output_dir
            / "queries"
            / f"{checkpoint_number:02d}-{query_input.query.query_id}.json",
            payload,
        )
        persist_drive_bytes(
            state,
            staging_dir,
            f"queries/{checkpoint_number:02d}-{query_input.query.query_id}.json",
            result_bytes,
            deadline=work_deadline,
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
            staging_dir,
            f"query-{checkpoint_number:02d}",
            checkpoint,
            deadline=work_deadline,
        )
        completed.append(payload)
        state_data = (json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n").encode()
        temporary = drive_state.with_suffix(".tmp")
        temporary.write_bytes(state_data)
        temporary.replace(drive_state)
        print(json.dumps(checkpoint, sort_keys=True), flush=True)

    try:
        score_b2_queries(
            queries,
            adapter,
            on_query_complete=checkpoint_query,
            work_deadline_monotonic=work_deadline,
        )
        results = {"schema_revision": "res142-qwen3-reranked-results-v1", "queries": completed}
        results_bytes = _write_json_create_only(output_dir / "results.json", results)
        persist_drive_bytes(
            state,
            staging_dir,
            "scores/results.json",
            results_bytes,
            deadline=work_deadline,
        )
        persist_drive_checkpoint(
            state,
            staging_dir,
            "all-queries-complete",
            {
                "query_count": len(completed),
                "candidate_count": sum(row["candidate_count"] for row in completed),
                "results_sha256": hashlib.sha256(results_bytes).hexdigest(),
            },
            deadline=work_deadline,
        )
        drive_state.write_text(
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
        _write_json_create_only(output_dir / "failure.json", failure)
        print(json.dumps({"status": "failed", **failure}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
