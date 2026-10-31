"""Bounded RES-146 demonstration and exploratory comparison; never writes reference data."""

from __future__ import annotations

import json
from hashlib import sha256
from time import perf_counter

from scripts.res145_evidence_demo import (
    ROOT,
    SOURCE_MODELS,
    database_receipt,
    protected_files,
    qdrant_receipt,
    scratch_database,
)
from sqlalchemy import Engine, insert, select, text
from sqlalchemy.orm import Session

from dynamisrag.benchmark.dynamisbench import Run, Specification
from dynamisrag.benchmark.dynamisbench_runner import verify_raw_responses, verify_spec_evidence
from dynamisrag.config import load_settings
from dynamisrag.db import create_database_engine
from dynamisrag.db.models import SectionRecord
from dynamisrag.domain.identity import evidence_identity
from dynamisrag.evidence.contracts import RetrievalIdentity, SelectionPolicy, SelectionRequest
from dynamisrag.evidence.selection import redundant
from dynamisrag.evidence.service import select_live, select_offline
from dynamisrag.search.scientific import (
    ScientificResponse,
    ScientificRetrievalService,
    ScientificUnavailable,
)


def protected_benchmark_files() -> str:
    paths = sorted(
        p
        for directory in (
            ROOT / ".dynamisrag/res-143",
            ROOT / ".dynamisrag/dynamisbench-v0",
            ROOT / "docs/evidence/dynamisbench-v0",
        )
        for p in directory.rglob("*")
        if p.is_file()
    )
    return evidence_identity(
        "res146-protected-benchmarks-v1",
        [(p.relative_to(ROOT).as_posix(), sha256(p.read_bytes()).hexdigest()) for p in paths],
    )


def copy_reference(reference: Engine, target: Engine) -> None:
    """Copy the bounded original canonical inventory into an isolated migrated database."""
    with Session(reference) as source, Session(target) as destination, destination.begin():
        source.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"))
        for model in SOURCE_MODELS:
            query = select(model.__table__)
            if model is SectionRecord:
                query = query.order_by(SectionRecord.depth)
            rows = source.execute(query.limit(10001)).mappings().all()
            if len(rows) > 10000:
                raise ValueError("reference copy row budget exceeded")
            for row in rows:
                destination.execute(insert(model).values(**dict(row)))


def _compare(
    engine: Engine,
    spec: Specification,
    run: Run,
    responses: tuple[ScientificResponse, ...],
    corpus: RetrievalIdentity,
    *,
    demo: dict[str, object],
) -> list[dict[str, object]]:
    observations: list[dict[str, object]] = []
    for question, response in zip(spec.questions, responses, strict=True):
        for mode in ("rank_order", "deterministic", "diversity", "context"):
            policy = SelectionPolicy(
                prefer_document_diversity=mode in ("diversity", "context"),
                prefer_section_diversity=mode in ("diversity", "context"),
                max_expansions=6 if mode == "context" else 0,
            )
            request = SelectionRequest(
                question=question.query,
                corpus=corpus,
                policy=policy,
                minimum_documents=2 if question.category == "multi_study" else 1,
                response_requirement=(
                    "ABSTAIN"
                    if question.category == "unanswerable"
                    else "CORRECT_PREMISE"
                    if question.category == "false_premise"
                    else "UNASSESSED"
                ),
            )
            start = perf_counter()
            result = select_offline(engine, request, response)
            elapsed = (perf_counter() - start) * 1000
            replay = select_offline(engine, request, response)
            if result != replay or result.key != replay.key:
                raise ValueError("selection replay unstable")
            selected: set[str] = (
                {m.passage_key for m in result.evidence_set.members}
                if result.evidence_set
                else set()
            )
            selected_sources = [
                h.provenance for h in response.fusion.hits if h.passage_key in selected
            ]
            if mode == "rank_order":
                # Legacy rank-order-only baseline: first five whole ranked passages.
                baseline = response.fusion.hits[: policy.max_passages]
                selected = {h.passage_key for h in baseline}
                documents = len({h.provenance.document_canonical_key for h in baseline})
                sections = len(
                    {
                        (h.provenance.document_version_key, h.provenance.section_key)
                        for h in baseline
                    }
                )
                used = sum(len(h.provenance.text.encode()) for h in baseline)
                selected_sources = [h.provenance for h in baseline]
            else:
                documents, sections, used = (
                    result.documents,
                    result.sections,
                    result.selected_bytes + result.expanded_bytes,
                )
            proposed = {j.passage_key for j in question.judgments if j.grade > 0}
            duplicate_pairs = sum(
                redundant(a, b)
                for i, a in enumerate(selected_sources)
                for b in selected_sources[i + 1 :]
            )
            observations.append(
                {
                    "question_id": question.id,
                    "mode": mode,
                    "selected": len(selected),
                    "retrieved_fused": len(response.fusion.hits),
                    "documents": documents,
                    "sections": sections,
                    "proposed_relevant_retained": len(selected & proposed),
                    "proposed_relevant_total": len(proposed),
                    "duplicates_suppressed": sum(
                        d.disposition == "DUPLICATE" for d in result.decisions
                    )
                    if mode != "rank_order"
                    else 0,
                    "duplicate_pairs": duplicate_pairs,
                    "duplication_rate": duplicate_pairs
                    / max(1, len(selected_sources) * (len(selected_sources) - 1) / 2),
                    "candidate_retention": len(selected) / len(response.fusion.hits),
                    "byte_utilization": used / policy.max_bytes,
                    "bytes": used,
                    "byte_budget": policy.max_bytes,
                    "expansions": len(result.context) if mode == "context" else 0,
                    "expanded_bytes": result.expanded_bytes if mode == "context" else 0,
                    "provenance_failures": sum(
                        d.disposition == "SOURCE_INVALID" for d in result.decisions
                    ),
                    "selection_ms": elapsed if mode != "rank_order" else None,
                    "replay_stable": result == replay,
                    "selection_key": result.key if mode != "rank_order" else None,
                    "evidence_set_key": result.evidence_set.key
                    if result.evidence_set and mode != "rank_order"
                    else None,
                }
            )
            if question.id == "db00-02" and mode == "context":
                demo["offline_replay"] = {
                    "question_id": question.id,
                    "query_sha256": run.results[spec.questions.index(question)].query_sha256,
                    "selection_key": result.key,
                    "evidence_set_key": result.evidence_set.key if result.evidence_set else None,
                    "authenticity": result.authenticity,
                    "context_items": len(result.context),
                    "trace": result.model_dump(mode="json"),
                }
    return observations


def demonstrate() -> dict[str, object]:
    settings = load_settings()
    reference = create_database_engine(settings)
    before = database_receipt(reference)
    files_before, vectors_before = protected_files(), qdrant_receipt()
    benchmarks_before = protected_benchmark_files()
    benchmark_dir = ROOT / "docs/evidence/dynamisbench-v0"
    spec = Specification.model_validate_json((benchmark_dir / "candidate-bank.json").read_bytes())
    run = Run.model_validate_json((benchmark_dir / "baseline-run.json").read_bytes())
    verify_spec_evidence(spec, reference, settings.scientific_evidence_directory)
    responses = tuple(
        ScientificResponse.model_validate_json(
            (
                ROOT
                / ".dynamisrag/dynamisbench-v0/private-responses"
                / f"{row.response_sha256}.json"
            ).read_bytes()
        )
        for row in run.results
    )
    verify_raw_responses(spec, run, responses)
    corpus = RetrievalIdentity(
        embedding_manifest_sha256=spec.corpus.embedding_manifest_sha256,
        projection_sha256=spec.corpus.projection_sha256,
    )
    observations: list[dict[str, object]] = []
    demo: dict[str, object] = {}
    with scratch_database() as (engine, _):
        copy_reference(reference, engine)
        copied = database_receipt(engine)
        if copied != before:
            raise ValueError("reference copy differs")
        # Methodological case from the existing inventory, not a newly embedded query.
        question = next(q for q in spec.questions if q.category == "methodological")
        request = SelectionRequest(question=question.query, corpus=corpus)
        service = ScientificRetrievalService(engine, settings)
        try:
            service.initialize()
            result = select_live(service, request)
            demo = {
                "live_status": "completed",
                "selection_key": result.key,
                "evidence_set_key": result.evidence_set.key if result.evidence_set else None,
                "authenticity": result.authenticity,
            }
        except ScientificUnavailable as error:
            demo = {"live_status": "blocked", "blocker": error.code}
        finally:
            service.close()
        observations.extend(_compare(engine, spec, run, responses, corpus, demo=demo))
        if database_receipt(engine) != copied:
            raise ValueError("selection wrote copied canonical data")
    after = database_receipt(reference)
    protected = (
        before == after
        and protected_files() == files_before
        and qdrant_receipt() == vectors_before
        and protected_benchmark_files() == benchmarks_before
    )
    reference.dispose()
    if not protected:
        raise ValueError("protected reference inventory changed")
    return {
        "revision": "res146-qualification-v1",
        "base_commit": "ef5ed691b82af93489badbbb43dea439d10d904a",
        "demonstration": demo,
        "protected_integrity": protected,
        "canonical_before": before,
        "canonical_after": after,
        "protected_files_sha256": files_before,
        "protected_benchmark_files_sha256": benchmarks_before,
        "qdrant_sha256": vectors_before,
        "benchmark_status": (
            "exploratory unsigned provisional judgments; no scientific sufficiency claim"
        ),
        "observations": observations,
        "receipt_identity": evidence_identity("res146-diagnostics-v1", observations),
    }


if __name__ == "__main__":
    output = ROOT / "docs/evidence/res146-selection-receipt.json"
    receipt = demonstrate()
    output.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
