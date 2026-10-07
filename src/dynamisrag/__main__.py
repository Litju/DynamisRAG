"""Console entry point: ``python -m dynamisrag`` / ``dynamisrag``.

Two behaviours, selected by the first argument:

    dynamisrag                              start the development server
    dynamisrag search "probiotic exercise"   query the BM25 passage projection
    dynamisrag project-passages --chunker-revision structure-v1.1.b19e0939b5de
    dynamisrag benchmark verify-res138-bundle <path>
    dynamisrag benchmark verify-stage-a <bundle-root>
    dynamisrag benchmark run-opensearch --bundle <root> --work-dir <dir>
    dynamisrag benchmark verify-gpu-evidence --bundle <root> --evidence <file>
    dynamisrag benchmark assemble-qualification --bundle <root> --work-dir <dir>
    dynamisrag benchmark select --bundle <root> --work-dir <dir>

``dynamisrag`` with no arguments still starts the server, unchanged. The
subcommands are ``argparse`` over the same services the API uses — the CLI has
no search implementation of its own, so a result obtained from the terminal and
one obtained from ``GET /search`` are produced by the same request against the
same projection.

The ``benchmark`` group is **not** a search implementation. Its Stage A
subcommands write the frozen RES-138 plan from a code commit and verify a benchmark
bundle downloaded from Drive on this workstation, with no trust on first use. Its
Stage B subcommands are the executable production-qualification workflow: load the
sealed Stage A result, build the deterministic Stage B plan, run the local
OpenSearch Lucene HNSW lane against the workstation's own node, verify a GPU TEI
evidence artifact imported from the remote A100, assemble
``res138-production-qualification-v1`` and run the frozen selection rule. Only
``run-opensearch`` and ``cleanup-stage-b-indexes`` open a service; the rest read a
bundle and write an artifact.

The process exit status is the contract: ``0`` on success, non-zero with one
safe line on stderr when configuration, the database, the search backend or a
benchmark artifact cannot be used.

The stderr line is built by :func:`_safe_error_line`: an application-authored
failure such as a rejected request, a missing chunker revision or a refused
artifact is shown in full, while a low-level OpenSearch failure is rendered from
its safe summary — exception class, operation, HTTP status, ``error.type``, target —
and never from the exception's detail, which may quote an OpenSearch ``error.reason``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final

import uvicorn
from sqlalchemy.orm import Session

from dynamisrag.application import create_app
from dynamisrag.config import load_settings
from dynamisrag.db.engine import create_database_engine
from dynamisrag.logging_config import configure_logging
from dynamisrag.search.bm25 import DEFAULT_LIMIT, MAX_LIMIT, Bm25SearchService
from dynamisrag.search.client import OpenSearchClient
from dynamisrag.search.errors import OpenSearchError, ProjectionError
from dynamisrag.search.projection import PassageProjector

if TYPE_CHECKING:
    from dynamisrag.benchmark.gpu_evidence import GpuEvidenceVerdict
    from dynamisrag.benchmark.opensearch_lane import OpenSearchLaneResult
    from dynamisrag.benchmark.stage_a import SealedStageA
    from dynamisrag.benchmark.stage_b import StageBPlan

__all__ = [
    "PRODUCTION_QUALIFICATION_ARTIFACT_REVISION",
    "STAGE_B_INDEX_ARTIFACT_REVISION",
    "STAGE_B_PLAN_ARTIFACT_REVISION",
    "build_parser",
    "main",
]

STAGE_B_PLAN_ARTIFACT_REVISION: Final[str] = "res138-stage-b-plan-v1"
"""The plan revision this command prints, restated so the CLI owns no benchmark import.

The benchmark modules are imported inside the Stage B commands rather than at module
scope, so starting the served application never loads the harness or the numeric stack
behind it. The two revision strings below are the only Stage B names the command surface
carries, and they are literals exactly because they are part of what a reviewer reads on
stdout.
"""

STAGE_B_INDEX_ARTIFACT_REVISION: Final[str] = "res138-stage-b-opensearch-v1"
"""The lane result revision this command prints; see :data:`STAGE_B_PLAN_ARTIFACT_REVISION`."""

PRODUCTION_QUALIFICATION_ARTIFACT_REVISION: Final[str] = "res138-production-qualification-v1"
"""The qualification revision this command prints; see :data:`STAGE_B_PLAN_ARTIFACT_REVISION`."""

_PROGRAM: Final[str] = "dynamisrag"
_SEARCH: Final[str] = "search"
_PROJECT: Final[str] = "project-passages"
_BENCHMARK: Final[str] = "benchmark"
_RES138_PLAN: Final[str] = "res138-plan"
_VERIFY_RES138_BUNDLE: Final[str] = "verify-res138-bundle"
_VERIFY_STAGE_A: Final[str] = "verify-stage-a"
_STAGE_B_PLAN: Final[str] = "stage-b-plan"
_RUN_OPENSEARCH: Final[str] = "run-opensearch"
_CLEANUP_STAGE_B_INDEXES: Final[str] = "cleanup-stage-b-indexes"
_VERIFY_GPU_EVIDENCE: Final[str] = "verify-gpu-evidence"
_ASSEMBLE_QUALIFICATION: Final[str] = "assemble-qualification"
_SELECT: Final[str] = "select"

_EXIT_SUCCESS: Final[int] = 0
_EXIT_FAILURE: Final[int] = 1
"""A non-zero status is all a script needs to detect a failure."""


def build_parser() -> argparse.ArgumentParser:
    """The command surface. Deliberately stdlib ``argparse``, no CLI framework."""
    parser = argparse.ArgumentParser(
        prog=_PROGRAM,
        description=(
            "DynamisRAG. With no arguments, serves the HTTP API. "
            "'search' queries the versioned BM25 passage projection; "
            "'project-passages' rebuilds that projection from canonical PostgreSQL; "
            "'benchmark' writes the frozen RES-138 plan and verifies benchmark bundles."
        ),
    )
    commands = parser.add_subparsers(dest="command")

    search = commands.add_parser(_SEARCH, help="query the BM25 passage projection and print JSON")
    search.add_argument("query", help="non-whitespace search text")
    search.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"maximum number of hits, 1-{MAX_LIMIT} (default: {DEFAULT_LIMIT})",
    )

    project = commands.add_parser(
        _PROJECT, help="rebuild the OpenSearch passage projection from PostgreSQL"
    )
    project.add_argument(
        "--chunker-revision",
        required=True,
        help="the exact chunker revision to project; never inferred",
    )

    benchmark = commands.add_parser(
        _BENCHMARK,
        help="RES-138 benchmark tooling: the frozen plan and bundle verification",
    )
    benchmark_commands = benchmark.add_subparsers(dest="benchmark_command", required=True)

    plan = benchmark_commands.add_parser(
        _RES138_PLAN,
        help="write the frozen benchmark plan for an exact code commit and print its SHA-256",
    )
    plan.add_argument(
        "--code-sha",
        required=True,
        help="the exact 40-character commit the plan is for; never a branch, tag or main",
    )
    plan.add_argument(
        "--out",
        help=f"write {RES138_PLAN_FILENAME} here as well as to stdout (optional)",
    )

    verify = benchmark_commands.add_parser(
        _VERIFY_RES138_BUNDLE,
        help="verify a downloaded run bundle: every digest, every shard, the canonical order",
    )
    verify.add_argument("path", help="the bundle root, i.e. a downloaded run directory")
    verify.add_argument(
        "--code-sha",
        help="require the bundle to have been produced by this exact commit",
    )

    stage_a = benchmark_commands.add_parser(
        _VERIFY_STAGE_A,
        help="load the sealed Stage A result and print its identity, or refuse it",
    )
    stage_a.add_argument("bundle", help="the sealed Stage A run directory")
    stage_a.add_argument(
        "--code-sha",
        help=(
            "the commit the Stage B plan will bind; required by the Stage B subcommands and "
            "accepted here so one value can be shared across the whole workflow"
        ),
    )

    plan_b = benchmark_commands.add_parser(
        _STAGE_B_PLAN,
        help="build the deterministic Stage B plan and print its SHA-256",
    )
    _add_stage_b_arguments(plan_b)
    plan_b.add_argument("--out", help="write the Stage B plan here as well as to stdout (optional)")

    run = benchmark_commands.add_parser(
        _RUN_OPENSEARCH,
        help="build and measure the local OpenSearch HNSW lane from the sealed matrices",
    )
    _add_stage_b_arguments(run)
    run.add_argument("--work-dir", required=True, help="where lane results are written")
    run.add_argument(
        "--dimension",
        type=int,
        action="append",
        help="measure only this dimension; repeatable. Defaults to every planned dimension",
    )

    cleanup = benchmark_commands.add_parser(
        _CLEANUP_STAGE_B_INDEXES,
        help="delete only the indexes this Stage B plan created, verified from their _meta",
    )
    _add_stage_b_arguments(cleanup)

    gpu = benchmark_commands.add_parser(
        _VERIFY_GPU_EVIDENCE,
        help="re-verify an imported A100 TEI artifact against the sealed Stage A reference",
    )
    _add_stage_b_arguments(gpu)
    gpu.add_argument("--evidence", required=True, help="the GPU evidence artifact to import")
    gpu.add_argument(
        "--vectors-dir",
        help="where the artifact's calibration .npy lives; defaults to the artifact's directory",
    )

    assemble = benchmark_commands.add_parser(
        _ASSEMBLE_QUALIFICATION,
        help="assemble and verify res138-production-qualification-v1 from the recorded evidence",
    )
    _add_stage_b_arguments(assemble)
    assemble.add_argument("--work-dir", required=True, help="where the qualification is written")

    select = benchmark_commands.add_parser(
        _SELECT,
        help="run the frozen selection rule over Stage A quality plus Stage B evidence",
    )
    _add_stage_b_arguments(select)
    select.add_argument("--work-dir", required=True, help="where the qualification is read from")
    return parser


def _add_stage_b_arguments(parser: argparse.ArgumentParser) -> None:
    """The two arguments every Stage B subcommand takes: the sealed bundle and the commit."""
    parser.add_argument("--bundle", required=True, help="the sealed Stage A run directory")
    parser.add_argument(
        "--code-sha",
        required=True,
        help="the exact 40-character commit the Stage B plan binds; never a branch or a tag",
    )


RES138_PLAN_FILENAME: Final[str] = "res138-plan.json"


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one command and return the process exit status."""
    arguments: list[str] = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not arguments:
        return _serve()

    parsed = parser.parse_args(arguments)
    if parsed.command == _SEARCH:
        return _search(parsed.query, parsed.limit)
    if parsed.command == _PROJECT:
        return _project(parsed.chunker_revision)
    if parsed.command == _BENCHMARK:
        return _benchmark(parsed.benchmark_command, parsed)
    return _serve()


def _serve() -> int:
    """Load configuration, install logging and serve the application."""
    settings = load_settings()
    configure_logging(settings.log_level)
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )
    return _EXIT_SUCCESS


def _search(query: str, limit: int) -> int:
    """Run one BM25 query and print the same response the API would return."""
    settings = load_settings()
    client = OpenSearchClient(settings)
    try:
        response = Bm25SearchService(client, alias=settings.opensearch_index_alias).search(
            query, limit=limit
        )
    except (ValueError, OpenSearchError) as error:
        return _fail(_safe_error_line(_SEARCH, error, allow_application_detail=False))
    finally:
        client.close()
    _emit(response.model_dump(mode="json"))
    return _EXIT_SUCCESS


def _project(chunker_revision: str) -> int:
    """Rebuild the projection from canonical PostgreSQL, inside one transaction."""
    settings = load_settings()
    engine = create_database_engine(settings)
    client = OpenSearchClient(settings)
    try:
        with Session(bind=engine) as session, session.begin():
            projector = PassageProjector(
                session,
                client,
                alias=settings.opensearch_index_alias,
                batch_size=settings.opensearch_bulk_batch_size,
            )
            result = projector.project(chunker_revision=chunker_revision)
    except (ValueError, OpenSearchError) as error:
        return _fail(_safe_error_line(_PROJECT, error, allow_application_detail=True))
    finally:
        client.close()
        engine.dispose()
    _emit(result.to_payload())
    return _EXIT_SUCCESS


def _benchmark(  # noqa: PLR0911 - a flat command table is the shape a dispatch wants
    benchmark_command: str | None, arguments: argparse.Namespace
) -> int:
    """Dispatch the ``benchmark`` group.

    One visible line per command, deliberately: a dispatch table is exactly the place where
    a new subcommand should be an addition rather than a new nesting level.
    """
    if benchmark_command == _RES138_PLAN:
        return _res138_plan(arguments.code_sha, arguments.out)
    if benchmark_command == _VERIFY_RES138_BUNDLE:
        return _verify_res138_bundle(arguments.path, arguments.code_sha)
    if benchmark_command == _VERIFY_STAGE_A:
        return _verify_stage_a(arguments.bundle, arguments.code_sha)
    if benchmark_command == _STAGE_B_PLAN:
        return _stage_b(arguments)
    if benchmark_command == _RUN_OPENSEARCH:
        return _run_opensearch(arguments)
    if benchmark_command == _CLEANUP_STAGE_B_INDEXES:
        return _cleanup_stage_b_indexes(arguments)
    if benchmark_command == _VERIFY_GPU_EVIDENCE:
        return _verify_gpu_evidence(arguments)
    if benchmark_command == _ASSEMBLE_QUALIFICATION:
        return _assemble_qualification(arguments)
    if benchmark_command == _SELECT:
        return _select(arguments)
    return _fail(f"{_PROGRAM} {_BENCHMARK}: unknown command {benchmark_command!r}")


def _res138_plan(code_sha: str, out: str | None) -> int:
    """Write the frozen plan and print its digest.

    The digest is the point of the command: a reviewer can compute the plan's
    identity on their own machine, with no GPU, no Drive mount and no model, and
    compare it with the one a Colab session recorded.

    The harness is imported here rather than at module scope so that starting the
    served application — the other thing this program does — never loads benchmark
    code or the numeric stack behind it.
    """
    from dynamisrag.benchmark.errors import BenchmarkError
    from dynamisrag.benchmark.res138 import benchmark_plan

    try:
        envelope = benchmark_plan(code_sha)
    except BenchmarkError as error:
        return _fail(
            _safe_error_line(f"{_BENCHMARK} {_RES138_PLAN}", error, allow_application_detail=True)
        )
    if out is not None:
        envelope.write(Path(out))
    _emit({"artifact_revision": envelope.artifact_revision, "sha256": envelope.sha256})
    return _EXIT_SUCCESS


def _verify_res138_bundle(path: str, expect_code_sha: str | None) -> int:
    """Verify a bundle and print the report, or exit non-zero naming the failure."""
    from dynamisrag.benchmark.bundle import verify_run_bundle
    from dynamisrag.benchmark.errors import BenchmarkError

    try:
        report = verify_run_bundle(Path(path), expect_code_sha=expect_code_sha)
    except BenchmarkError as error:
        return _fail(
            _safe_error_line(
                f"{_BENCHMARK} {_VERIFY_RES138_BUNDLE}", error, allow_application_detail=True
            )
        )
    _emit(report.payload() | {"verification_sha256": report.sha256})
    return _EXIT_SUCCESS


def _sealed_and_plan(arguments: argparse.Namespace) -> tuple[SealedStageA, StageBPlan]:
    """Load the sealed Stage A result and build the Stage B plan, or raise.

    The benchmark modules are imported here rather than at module scope for the same
    reason the Stage A commands import lazily: starting the served application must never
    load benchmark code or the numeric stack behind it. The names themselves are
    annotations only, so the runtime imports below are the only ones that happen.
    """
    from dynamisrag.benchmark.stage_a import load_sealed_stage_a
    from dynamisrag.benchmark.stage_b import build_stage_b_plan

    sealed = load_sealed_stage_a(Path(arguments.bundle))
    return sealed, build_stage_b_plan(reference=sealed.reference, code_sha=arguments.code_sha)


def _verify_stage_a(bundle: str, code_sha: str | None) -> int:
    """Load and verify the sealed Stage A result, or exit non-zero naming the failure."""
    from dynamisrag.benchmark.errors import BenchmarkError
    from dynamisrag.benchmark.stage_a import load_sealed_stage_a
    from dynamisrag.benchmark.stage_b import build_stage_b_plan

    try:
        sealed = load_sealed_stage_a(Path(bundle))
        summary: dict[str, object] = {
            "stage_a": dict(sealed.payload()),
            "labels": list(sealed.labels),
        }
        if code_sha:
            summary["stage_b_plan_sha256"] = build_stage_b_plan(
                reference=sealed.reference, code_sha=code_sha
            ).sha256
    except BenchmarkError as error:
        return _fail(
            _safe_error_line(
                f"{_BENCHMARK} {_VERIFY_STAGE_A}", error, allow_application_detail=True
            )
        )
    _emit(summary)
    return _EXIT_SUCCESS


def _stage_b(arguments: argparse.Namespace) -> int:
    """Build the deterministic Stage B plan and print its digest, writing it if asked."""
    from dynamisrag.benchmark.errors import BenchmarkError

    try:
        sealed, plan = _sealed_and_plan(arguments)
    except BenchmarkError as error:
        return _fail(
            _safe_error_line(f"{_BENCHMARK} {_STAGE_B_PLAN}", error, allow_application_detail=True)
        )
    out: str | None = arguments.out
    if out is not None:
        plan.write(Path(out))
    _emit(
        {
            "artifact_revision": STAGE_B_PLAN_ARTIFACT_REVISION,
            "stage": "production-qualification",
            "plan_sha256": plan.sha256,
            "shortlist": [list(pair) for pair in plan.candidates],
            "dimensions": list(plan.dimensions),
            "model_revisions": dict.fromkeys(plan.model_ids, plan.model_revision),
            "reference_bundle_sha256": sealed.reference.bundle_sha256,
        }
    )
    return _EXIT_SUCCESS


def _run_opensearch(arguments: argparse.Namespace) -> int:
    """Run the local OpenSearch lane and write one result per measured configuration."""
    from dynamisrag.benchmark.errors import BenchmarkError
    from dynamisrag.benchmark.opensearch_lane import lane_result_path, measure_configuration

    try:
        sealed, plan = _sealed_and_plan(arguments)
        settings = load_settings()
        client = OpenSearchClient(settings)
        results: list[dict[str, object]] = []
        try:
            for dimension in _requested_dimensions(plan.dimensions, arguments.dimension):
                lane = measure_configuration(
                    client=client, sealed=sealed, plan=plan, dimension=dimension
                )
                path = lane_result_path(
                    Path(arguments.work_dir), model_id=plan.model_ids[0], dimension=dimension
                )
                results.append(
                    {
                        "dimension": dimension,
                        "sha256": lane.write(path),
                        "measurements": lane.payload(),
                    }
                )
        finally:
            client.close()
    except BenchmarkError as error:
        return _fail(
            _safe_error_line(
                f"{_BENCHMARK} {_RUN_OPENSEARCH}", error, allow_application_detail=True
            )
        )
    except (ValueError, OpenSearchError) as error:
        return _fail(
            _safe_error_line(
                f"{_BENCHMARK} {_RUN_OPENSEARCH}", error, allow_application_detail=False
            )
        )
    _emit({"artifact_revision": STAGE_B_INDEX_ARTIFACT_REVISION, "results": results})
    return _EXIT_SUCCESS


def _cleanup_stage_b_indexes(arguments: argparse.Namespace) -> int:
    """Remove only the indexes this plan created, each verified from its own recorded _meta."""
    from dynamisrag.benchmark.errors import BenchmarkError
    from dynamisrag.benchmark.opensearch_lane import cleanup_stage_b_indexes

    try:
        _sealed, plan = _sealed_and_plan(arguments)
        settings = load_settings()
        client = OpenSearchClient(settings)
        try:
            removed = cleanup_stage_b_indexes(client=client, plan=plan)
        finally:
            client.close()
    except BenchmarkError as error:
        return _fail(
            _safe_error_line(
                f"{_BENCHMARK} {_CLEANUP_STAGE_B_INDEXES}", error, allow_application_detail=True
            )
        )
    except (ValueError, OpenSearchError) as error:
        return _fail(
            _safe_error_line(
                f"{_BENCHMARK} {_CLEANUP_STAGE_B_INDEXES}", error, allow_application_detail=False
            )
        )
    _emit({"removed": list(removed), "count": len(removed)})
    return _EXIT_SUCCESS


def _verify_gpu_evidence(arguments: argparse.Namespace) -> int:
    """Re-verify an imported GPU artifact, or exit non-zero naming the gate that refused it."""
    from dynamisrag.benchmark.errors import BenchmarkError
    from dynamisrag.benchmark.gpu_evidence import verify_gpu_evidence

    try:
        sealed, plan = _sealed_and_plan(arguments)
        vectors = Path(arguments.vectors_dir) if arguments.vectors_dir else None
        verdict = verify_gpu_evidence(
            Path(arguments.evidence),
            sealed=sealed,
            plan=plan,
            vectors_directory=vectors,
        )
    except BenchmarkError as error:
        return _fail(
            _safe_error_line(
                f"{_BENCHMARK} {_VERIFY_GPU_EVIDENCE}", error, allow_application_detail=True
            )
        )
    _emit(dict(verdict.payload()))
    return _EXIT_SUCCESS


def _load_evidence(
    arguments: argparse.Namespace,
) -> tuple[SealedStageA, StageBPlan, list[OpenSearchLaneResult], list[GpuEvidenceVerdict]]:
    """Re-verify every recorded GPU artifact and re-read every lane result from disk.

    Assembly and selection both start here, so neither can consume an artifact this process
    has not just re-verified. Lane results come back through
    :func:`~dynamisrag.benchmark.opensearch_lane.load_lane_result`, which refuses a result
    produced under another plan.
    """
    from dynamisrag.benchmark.gpu_evidence import verify_gpu_evidence
    from dynamisrag.benchmark.opensearch_lane import (
        lane_result_from_payload,
        lane_result_path,
        load_lane_result,
    )

    sealed, plan = _sealed_and_plan(arguments)
    work_dir = Path(arguments.work_dir)
    verdicts = [
        verify_gpu_evidence(path, sealed=sealed, plan=plan, vectors_directory=path.parent)
        for path in sorted(work_dir.glob("gpu-evidence/*.json"))
    ]
    lanes: list[OpenSearchLaneResult] = []
    for dimension in plan.dimensions:
        result_path = lane_result_path(work_dir, model_id=plan.model_ids[0], dimension=dimension)
        if not result_path.is_file():
            continue
        lanes.append(lane_result_from_payload(load_lane_result(result_path, plan=plan), plan=plan))
    return sealed, plan, lanes, verdicts


def _assemble_qualification(arguments: argparse.Namespace) -> int:
    """Assemble, verify and write ``res138-production-qualification-v1``."""
    from dynamisrag.benchmark.errors import BenchmarkError
    from dynamisrag.benchmark.qualification import (
        assemble_production_qualification,
        qualification_path,
        write_qualification,
    )

    try:
        sealed, plan, lanes, verdicts = _load_evidence(arguments)
        qualification = assemble_production_qualification(
            sealed=sealed, plan=plan, lanes=lanes, verdicts=verdicts
        )
        digest = write_qualification(qualification, qualification_path(Path(arguments.work_dir)))
    except BenchmarkError as error:
        return _fail(
            _safe_error_line(
                f"{_BENCHMARK} {_ASSEMBLE_QUALIFICATION}", error, allow_application_detail=True
            )
        )
    _emit(
        {
            "artifact_revision": PRODUCTION_QUALIFICATION_ARTIFACT_REVISION,
            "sha256": digest,
            "qualification": qualification.payload(),
        }
    )
    return _EXIT_SUCCESS


def _select(arguments: argparse.Namespace) -> int:
    """Run the frozen selection rule, write its outcome, and exit non-zero if it halted.

    A halted selection is a result a reviewer needs to see, so it is written and printed
    and the process reports failure — not because the command errored, but because the
    workflow did not reach a decision and an operator must look at why.
    """
    from dynamisrag.benchmark.errors import BenchmarkError
    from dynamisrag.benchmark.qualification import (
        qualification_path,
        read_qualification,
        run_stage_b_selection,
        write_selection_artifact,
    )
    from dynamisrag.benchmark.selection import SelectionStatus

    try:
        sealed, _plan, _lanes, _verdicts = _load_evidence(arguments)
        work_dir = Path(arguments.work_dir)
        qualification = read_qualification(qualification_path(work_dir))
        outcome = run_stage_b_selection(sealed=sealed, qualification=qualification)
        digest = write_selection_artifact(
            path=work_dir / "res138-selection.json", outcome=outcome, qualification=qualification
        )
    except BenchmarkError as error:
        return _fail(
            _safe_error_line(f"{_BENCHMARK} {_SELECT}", error, allow_application_detail=True)
        )
    payload = dict(outcome.payload())
    payload["sha256"] = digest
    _emit(payload)
    return _EXIT_FAILURE if outcome.status is SelectionStatus.HALTED else _EXIT_SUCCESS


def _requested_dimensions(planned: Sequence[int], requested: Sequence[int] | None) -> list[int]:
    """The dimensions to measure: the requested ones in canonical order, all of them by default."""
    if not requested:
        return list(planned)
    wanted = set(requested)
    unknown = sorted(wanted - set(planned))
    if unknown:
        raise ValueError(f"these dimensions are not in the Stage B plan: {unknown}")
    return [dimension for dimension in planned if dimension in wanted]


def _safe_error_line(command: str, error: Exception, *, allow_application_detail: bool) -> str:
    """Render one safe stderr line for ``command``.

    Two classes of failure are treated differently, and the difference is
    deliberate:

    *Application-authored* failures — a rejected query, an out-of-range limit,
    a requested chunker revision that does not exist — are written by this
    codebase out of configuration and canonical revision names. They are the
    actionable part of the message, so they are echoed verbatim where the
    command can produce them.

    *Everything else at the OpenSearch boundary* is rendered from
    :meth:`~dynamisrag.search.errors.OpenSearchError.safe_summary` alone.
    ``OpenSearchError.detail`` is never printed: it is free-form prose, and
    OpenSearch's own ``error.reason`` — which can quote a credential, a
    rejected value, the query or the rejected document — is not something this
    process controls. The summary keeps the exception class, the operation, the
    HTTP status, ``error.type`` and the index addressed, which is enough to act
    on, and cannot carry backend content.
    """
    prefix = f"{_PROGRAM} {command}: {type(error).__name__}"
    if isinstance(error, OpenSearchError):
        if allow_application_detail and isinstance(error, ProjectionError):
            return f"{prefix}: {error}"
        return f"{prefix}: {error.safe_summary()}"
    return f"{prefix}: {error}"


def _emit(payload: object) -> None:
    """Write machine-readable JSON to stdout."""
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def _fail(message: str) -> int:
    """Write one safe line to stderr and fail the process."""
    print(message, file=sys.stderr)
    return _EXIT_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
