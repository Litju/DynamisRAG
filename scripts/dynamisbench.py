"""Explicit DynamisBench operator. Captures only; never ingests, embeds documents or publishes."""
# ruff: noqa: T201 -- operator output

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from shutil import which

from dynamisrag.benchmark.dynamisbench import Contract, Run, Specification
from dynamisrag.benchmark.dynamisbench_answers import evaluate_answers
from dynamisrag.benchmark.dynamisbench_metrics import compare, evaluate, verify_report
from dynamisrag.benchmark.dynamisbench_runner import (
    capture,
    digest_json,
    verify_raw_responses,
    verify_spec_evidence,
)
from dynamisrag.config import load_settings
from dynamisrag.db import create_database_engine
from dynamisrag.generation.scientific import ScientificAnswer
from dynamisrag.search.scientific import ScientificResponse, ScientificRetrievalService


def write_artifact(directory: Path, value: Contract) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{value.digest}.json"
    if target.exists():
        if target.read_bytes() != value.canonical_bytes():
            raise ValueError("artifact collision")
    else:
        with target.open("xb") as handle:
            handle.write(value.canonical_bytes())
    return target


def require_private_capture_output(out: Path, repo_root: Path, git: str) -> Path:
    """Refuse public or tracked output before any full-text response can be captured."""
    root = repo_root.resolve()
    private_root = root / ".dynamisrag" / "dynamisbench-v0"
    destination = out.resolve()
    if private_root.resolve() != private_root or not destination.is_relative_to(private_root):
        raise ValueError("raw capture requires the private ignored .dynamisrag/dynamisbench-v0")
    tracked = subprocess.check_output(
        [git, "-C", str(root), "ls-files", "--cached", "--", ".dynamisrag/dynamisbench-v0"],
        text=True,
    )
    if tracked.strip():
        raise ValueError("private capture directory contains Git-tracked files")
    probe = destination / "private-responses" / ".dynamisbench-private-probe.json"
    ignored = subprocess.run(
        [git, "-C", str(root), "check-ignore", "-q", "--no-index", "--", str(probe)],
        check=False,
    )
    if ignored.returncode != 0:
        raise ValueError("raw capture destination is not confirmed Git-ignored")
    return destination


def capture_live(spec: Specification, out: Path) -> None:
    git = which("git")
    if git is None:
        raise ValueError("git is unavailable")
    root = Path(
        subprocess.check_output([git, "rev-parse", "--show-toplevel"], text=True).strip()
    ).resolve()
    out = require_private_capture_output(out, root, git)
    head = subprocess.check_output([git, "rev-parse", "HEAD"], text=True).strip()  # noqa: S603
    dirty = subprocess.check_output([git, "diff", "HEAD", "--name-only"], text=True)  # noqa: S603
    untracked = subprocess.check_output(  # noqa: S603 -- fixed argv and resolved git executable
        [git, "ls-files", "--others", "--exclude-standard", "src", "scripts"],
        text=True,
    )
    if dirty or untracked:
        raise ValueError("capture requires committed implementation")
    settings = load_settings()
    engine = create_database_engine(settings)
    service = ScientificRetrievalService(engine, settings)
    try:
        corpus = verify_spec_evidence(spec, engine, settings.scientific_evidence_directory)
        service.initialize()

        def retrieve(query: str, limit: int, filters: dict[str, str]) -> ScientificResponse:
            response = service.retrieve(query, limit, filters)
            # Full service evidence stays private; public runs contain only keys/hashes.
            raw = response.model_dump(mode="json")
            path = out / "private-responses" / f"{digest_json(raw)}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as handle:
                json.dump(raw, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            return response

        run = capture(spec, retrieve, git_commit=head, verified_corpus=corpus)
        print(write_artifact(out, run))
        print(write_artifact(out, evaluate(spec, run)))
    finally:
        service.close()
        engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("validate", "capture", "score", "compare", "answers"))
    parser.add_argument(
        "--spec", type=Path, default=Path("docs/evidence/dynamisbench-v0/candidate-bank.json")
    )
    parser.add_argument("--expected-spec", required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--expected-run")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--expected-candidate")
    parser.add_argument("--responses", type=Path, help="Private raw-response directory for replay")
    parser.add_argument("--out", type=Path, default=Path(".dynamisrag/dynamisbench-v0"))
    args = parser.parse_args()
    spec = Specification.model_validate_json(args.spec.read_bytes())
    if spec.digest != args.expected_spec:
        raise ValueError("specification external pin mismatch")
    if args.operation == "validate":
        print(
            json.dumps(
                {
                    "spec_sha256": spec.digest,
                    "questions": len(spec.questions),
                    "status": spec.status,
                    "performance_gate_enabled": False,
                }
            )
        )
    elif args.operation == "capture":
        capture_live(spec, args.out)
    elif args.operation == "answers":
        score_answers(spec, args.run, args.expected_run, args.out)
    else:
        if args.run is None:
            raise ValueError("--run is required")
        run = Run.model_validate_json(args.run.read_bytes())
        report = evaluate(spec, run)
        verify_report(
            spec,
            run,
            report,
            expected_spec_sha256=args.expected_spec,
            expected_run_sha256=args.expected_run,
        )
        if args.responses is not None:
            verify_raw_responses(
                spec,
                run,
                tuple(
                    ScientificResponse.model_validate_json(
                        (args.responses / f"{r.response_sha256}.json").read_bytes()
                    )
                    for r in run.results
                ),
            )
        if args.operation == "compare":
            if args.candidate is None:
                raise ValueError("--candidate is required")
            candidate = Run.model_validate_json(args.candidate.read_bytes())
            if candidate.digest != args.expected_candidate:
                raise ValueError("candidate external pin mismatch")
            print(write_artifact(args.out, compare(spec, run, candidate)))
        else:
            print(write_artifact(args.out, report))


def score_answers(spec: Specification, path: Path | None, expected: str, out: Path) -> None:
    if path is None:
        raise ValueError("--run must name a private answer inventory")
    raw = path.read_bytes()
    if digest_json(json.loads(raw)) != expected:
        raise ValueError("answer inventory external pin mismatch")
    inventory = json.loads(raw)
    answers = {key: ScientificAnswer.model_validate(value) for key, value in inventory.items()}
    print(write_artifact(out, evaluate_answers(spec, answers)))


if __name__ == "__main__":
    main()
