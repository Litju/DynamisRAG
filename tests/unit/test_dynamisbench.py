"""Mutation-sensitive mechanics. Generated rankings and answers are synthetic, never evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from math import log2
from pathlib import Path
from typing import Any

import pytest

from dynamisrag.benchmark.dynamisbench import Passage, Run, Specification, validate_run
from dynamisrag.benchmark.dynamisbench_answers import SignedReview, evaluate_answer
from dynamisrag.benchmark.dynamisbench_metrics import (
    compare,
    evaluate,
    score_ranking,
    verify_report,
)
from dynamisrag.benchmark.dynamisbench_runner import (
    EXECUTION,
    capture,
    capture_response,
    digest_json,
    provenance_digest,
)
from dynamisrag.generation.scientific import ScientificAnswer
from dynamisrag.search.bm25 import SearchHit
from dynamisrag.search.retrieval import DenseCandidate, PassageProvenance, reciprocal_rank_fusion
from dynamisrag.search.scientific import ScientificCandidate, ScientificLane, ScientificResponse
from tests.unit.test_scientific_answers import identity, retrieval_fixture

BANK = Path(__file__).resolve().parents[2] / "docs/evidence/dynamisbench-v0/candidate-bank.json"
SPEC_PIN = "1df5bacb6cfc786f29b8f5055c33c8e6e3a7be59399744fd55e96ce9d7a755f8"


@pytest.fixture
def spec() -> Specification:
    return Specification.model_validate_json(BANK.read_bytes())


def synthetic(spec: Specification) -> tuple[Specification, ScientificResponse]:
    base = retrieval_fixture()
    proto = base.fusion.hits[0].provenance
    passages: list[PassageProvenance] = []
    inventory: list[Passage] = []
    for i, p in enumerate(spec.corpus.passages):
        text = f"Synthetic finding {i}: impulse was 168 plus or minus 21 N s. Synthetic test only."
        source = proto.model_copy(
            update={
                "passage_key": p.key,
                "document_version_key": p.version_key,
                "source_external_id": p.pmcid,
                "pmcid": p.pmcid,
                "text": text,
                "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "source_spans": tuple(
                    s.model_copy(update={"end_char": len(text)}) for s in proto.source_spans
                ),
            }
        )
        passages.append(source)
        inventory.append(
            p.model_copy(
                update={
                    "provenance_sha256": provenance_digest(source),
                    "characters": 10000,
                }
            )
        )
    corpus = spec.corpus.model_copy(update={"passages": tuple(inventory)})
    spec = spec.model_copy(update={"corpus": corpus})
    # Place the first question's two proposed positives at ranks one and two.
    positive = [j.passage_key for j in spec.questions[0].judgments if j.grade > 0]
    ordered = sorted(passages, key=lambda p: (p.passage_key not in positive, p.passage_key))[:50]
    candidates = tuple(
        ScientificCandidate(
            rank=i,
            point_id=p.passage_key,
            raw_score=float(51 - i),
            provenance=p,
        )
        for i, p in enumerate(ordered, 1)
    )
    lexical = tuple(
        SearchHit.model_validate(
            {**c.provenance.model_dump(), "rank": c.rank, "score": c.raw_score}
        )
        for c in candidates
    )
    dense = tuple(
        DenseCandidate(
            rank=c.rank,
            raw_score=c.raw_score,
            passage_key=c.provenance.passage_key,
            provenance=c.provenance,
        )
        for c in candidates
    )
    response = base.model_copy(
        update={
            "embedding_manifest_sha256": corpus.embedding_manifest_sha256,
            "projection_sha256": corpus.projection_sha256,
            "bm25": ScientificLane(lane="text", candidates=candidates),
            "dense": ScientificLane(lane="dense", candidates=candidates),
            "fusion": reciprocal_rank_fusion(lexical=lexical, dense=dense, limit=20),
        }
    )
    return spec, response


def make_run(spec: Specification, response: ScientificResponse) -> Run:
    return capture(
        spec, lambda _q, _n, _f: response, git_commit="a" * 40, verified_corpus=spec.corpus
    )


def test_candidate_inventory_and_pins(spec: Specification) -> None:
    assert spec.digest == SPEC_PIN
    assert len(spec.questions) == 13
    assert len({q.category for q in spec.questions}) == 9
    assert spec.status == "AWAITING_EXPERT_REVIEW"
    assert len(spec.corpus.passages) == 75
    assert Specification.model_validate_json(spec.canonical_bytes()) == spec
    assert spec.digest == digest_json(spec.model_dump(mode="json"))


def test_captured_baseline_reconstructs_without_inference(spec: Specification) -> None:
    from dynamisrag.benchmark.dynamisbench_metrics import Report

    run = Run.model_validate_json((BANK.parent / "baseline-run.json").read_bytes())
    report = Report.model_validate_json((BANK.parent / "baseline-metrics.json").read_bytes())
    verify_report(
        spec,
        run,
        report,
        expected_spec_sha256=SPEC_PIN,
        expected_run_sha256="7527818b76f65b98d5707c0301a4923369552b3bf9931cfb45d8bcc6f5a3bf23",
    )
    assert report.digest == "9b43f5ff02389d43e5b4ebfd17c8a20e36be8d46d0b67c35239d1f2209ffc067"
    assert run.git_commit == "e9f29b87a8ae7c608e3fe74e30734986112fe121"
    assert len(report.rows) == 39
    assert all(r.ndcg10 is None for r in report.rows if r.question_id in ("db00-08", "db00-13"))
    fused = next(r for r in report.rows if r.question_id == "db00-09" and r.lane == "fused")
    assert fused.failure == "no_relevant_hit_at_10"


@pytest.mark.parametrize(
    "field,value",
    [
        ("scientific_status", "EXPERT_APPROVED"),
        ("performance_gate_enabled", True),
        ("canonical_passages_verified", 74),
        ("run_sha256", "0" * 64),
        ("sources", []),
        ("refusal_reasons", []),
        ("original_artifact_sha256", {}),
    ],
)
def test_qualification_receipt_refuses_mutations(
    spec: Specification, field: str, value: Any
) -> None:
    from dynamisrag.benchmark.dynamisbench import QualificationReceipt
    from dynamisrag.benchmark.dynamisbench_runner import verify_qualification_receipt

    run = Run.model_validate_json((BANK.parent / "baseline-run.json").read_bytes())
    receipt = QualificationReceipt.model_validate_json(
        (BANK.parent / "qualification-receipt.json").read_bytes()
    )
    verify_qualification_receipt(
        spec, run, "9b43f5ff02389d43e5b4ebfd17c8a20e36be8d46d0b67c35239d1f2209ffc067", receipt
    )
    raw = receipt.model_dump(mode="json")
    raw[field] = value
    with pytest.raises(ValueError):
        altered = QualificationReceipt.model_validate_json(json.dumps(raw))
        verify_qualification_receipt(spec, run, receipt.metrics_sha256, altered)


@pytest.mark.parametrize(
    "mutation",
    [
        "extra",
        "duplicate_question",
        "unknown_passage",
        "source_version",
        "unknown_article",
        "grade",
        "bool_grade",
        "span",
        "anchor",
        "false_premise",
        "unanswerable",
        "forged_approval",
        "missing_answerability",
        "missing_status",
        "duplicate_judgment",
    ],
)
def test_spec_corruption(spec: Specification, mutation: str) -> None:  # noqa: PLR0912 -- independent mutations
    data: Any = spec.model_dump(mode="json")
    q = data["questions"][0]
    if mutation == "extra":
        data["extra"] = True
    elif mutation == "duplicate_question":
        data["questions"].append(q)
    elif mutation == "unknown_passage":
        q["judgments"][0]["passage_key"] = "f" * 64
    elif mutation == "source_version":
        data["corpus"]["sources"][0]["version_key"] = "f" * 64
    elif mutation == "unknown_article":
        q["articles"] = ["PMC0"]
    elif mutation == "grade":
        q["judgments"][0]["grade"] = 4
    elif mutation == "bool_grade":
        q["judgments"][0]["grade"] = True
    elif mutation == "span":
        q["gold_spans"][0]["end"] = 1000000
    elif mutation == "anchor":
        q["gold_spans"][0]["source_anchor"] = "jats:#wrong"
    elif mutation == "false_premise":
        data["questions"][8]["answerability"] = "answerable"
    elif mutation == "unanswerable":
        data["questions"][7]["judgments"] = q["judgments"]
    elif mutation == "forged_approval":
        data["status"] = "EXPERT_APPROVED"
    elif mutation == "missing_answerability":
        del q["answerability"]
    elif mutation == "missing_status":
        del data["status"]
    else:
        q["judgments"].append(q["judgments"][0])
    with pytest.raises(ValueError):
        Specification.model_validate_json(json.dumps(data))


def test_known_metrics_and_unanswerable_exclusion(spec: Specification) -> None:
    spec, response = synthetic(spec)
    run = make_run(spec, response)
    row = evaluate(spec, run).rows[0]
    assert (row.recall5, row.recall10, row.ndcg10, row.mrr10, row.source_coverage10) == (
        1,
        1,
        1,
        1,
        1,
    )
    absent = next(q for q in spec.questions if q.category == "unanswerable")
    excluded = score_ranking(spec, absent, run.results[0].rankings[0])
    assert excluded.ndcg10 is None and excluded.recall10 is None and excluded.unjudged10 == 10
    assert all(a.denominator == 11 for a in evaluate(spec, run).aggregates if a.category is None)
    q = spec.questions[0]
    ranking = run.results[0].rankings[0]
    swapped = ranking.model_copy(
        update={
            "hits": tuple(
                h.model_copy(update={"rank": i + 1}) for i, h in enumerate(reversed(ranking.hits))
            )
        }
    )
    assert score_ranking(spec, q, swapped).mrr10 == 0
    shortened = ranking.model_copy(update={"hits": ranking.hits[1:2]})
    assert score_ranking(spec, q, shortened).recall5 == 0.5
    assert score_ranking(spec, q, shortened).ndcg10 == pytest.approx(1 / (1 + 1 / log2(3)))


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_question",
        "duplicate_question",
        "query",
        "spec",
        "corpus",
        "unknown_hit",
        "wrong_provenance",
        "reorder",
        "duplicate_hit",
        "missing_lane",
        "missing_fused",
        "fusion_score",
    ],
)
def test_run_mutations(spec: Specification, mutation: str) -> None:
    spec, response = synthetic(spec)
    run = make_run(spec, response)
    data: Any = run.model_dump(mode="json")
    r = data["results"][0]
    ranking = r["rankings"][0]
    if mutation == "missing_question":
        data["results"].pop()
    elif mutation == "duplicate_question":
        data["results"][1] = r
    elif mutation == "query":
        r["query_sha256"] = "0" * 64
    elif mutation == "spec":
        data["spec_sha256"] = "0" * 64
    elif mutation == "corpus":
        data["corpus_sha256"] = "0" * 64
    elif mutation == "unknown_hit":
        ranking["hits"][0]["passage_key"] = "0" * 64
    elif mutation == "wrong_provenance":
        ranking["hits"][0]["provenance_sha256"] = "0" * 64
    elif mutation == "reorder":
        ranking["hits"] = list(reversed(ranking["hits"]))
    elif mutation == "duplicate_hit":
        ranking["hits"][1]["passage_key"] = ranking["hits"][0]["passage_key"]
    elif mutation == "missing_lane":
        r["rankings"].pop()
    elif mutation == "missing_fused":
        r["rankings"][2]["hits"].pop()
    else:
        r["rankings"][2]["hits"][0]["score"] += 0.01
    with pytest.raises(ValueError):
        validate_run(spec, Run.model_validate_json(json.dumps(data)))


@pytest.mark.parametrize(
    "field",
    [
        "projection_sha256",
        "embedding_manifest_sha256",
        "model_revision",
        "query_embedding_config_sha256",
        "physical_collection",
        "document_embedding_config_sha256",
    ],
)
def test_adapter_identity_refusal(spec: Specification, field: str) -> None:
    spec, response = synthetic(spec)
    with pytest.raises(ValueError, match="identity"):
        capture_response(spec, spec.questions[0].id, response.model_copy(update={field: "0" * 64}))


def test_paired_and_coordinated_manipulation(spec: Specification) -> None:
    spec, response = synthetic(spec)
    baseline = make_run(spec, response)
    comparison = compare(spec, baseline, baseline)
    assert not comparison.regressions
    assert all(d.delta in (0, None) for d in comparison.differences)
    assert len(comparison.excluded_questions) == 2
    report = evaluate(spec, baseline)
    verify_report(
        spec,
        baseline,
        report,
        expected_spec_sha256=spec.digest,
        expected_run_sha256=baseline.digest,
    )
    with pytest.raises(ValueError, match="reconstruct"):
        verify_report(
            spec,
            baseline,
            report.model_copy(update={"rows": report.rows[:-1]}),
            expected_spec_sha256=spec.digest,
            expected_run_sha256=baseline.digest,
        )
    changed = baseline.model_copy(update={"git_commit": "b" * 40})
    with pytest.raises(ValueError, match="pin"):
        verify_report(
            spec,
            changed,
            evaluate(spec, changed),
            expected_spec_sha256=spec.digest,
            expected_run_sha256=baseline.digest,
        )
    q = spec.questions[0]
    changed_q = q.model_copy(
        update={
            "judgments": tuple(
                j.model_copy(update={"grade": 1}) if j.grade > 0 else j for j in q.judgments
            )
        }
    )
    new_spec = spec.model_copy(update={"questions": (changed_q, *spec.questions[1:])})
    with pytest.raises(ValueError):
        compare(new_spec, baseline, baseline)


def answer_fixture(
    spec: Specification, response: ScientificResponse, question: int = 0
) -> ScientificAnswer:
    source = response.fusion.hits[0].provenance
    quote = source.text
    from dynamisrag.generation.scientific import AnswerTimings, EvidenceExcerpt, VerifiedClaim

    return ScientificAnswer(
        status="supported",
        answer=quote + " [1]",
        claims=(
            VerifiedClaim(
                text=quote,
                source=source,
                supporting_excerpt=quote,
                excerpt_start=0,
                excerpt_end=len(quote),
            ),
        ),
        reason="supported",
        evidence=(source,),
        context_excerpts=(
            EvidenceExcerpt(passage_key=source.passage_key, text=quote, end=len(quote)),
        ),
        excluded_evidence=(),
        generation_model=identity(),
        generation_invoked=True,
        generation_metrics=None,
        retrieval=response,
        timings=AnswerTimings(
            retrieval_ms=1, generation_ms=1, citation_validation_ms=1, total_ms=3
        ),
    )


def test_answers_structure_never_implies_scientific_approval(spec: Specification) -> None:
    spec, response = synthetic(spec)
    answer = answer_fixture(spec, response)
    result = evaluate_answer(spec, spec.questions[0].id, answer)
    assert result.structural_citations_valid and result.scientific_pass is None
    assert result.review_status == "AWAITING_EXPERT_REVIEW"
    claim = answer.claims[0]
    wrong = claim.model_copy(update={"text": claim.text.replace("168", "999")})
    assert not evaluate_answer(
        spec, spec.questions[0].id, answer.model_copy(update={"claims": (wrong,)})
    ).structural_citations_valid
    wrong = claim.model_copy(update={"source": response.fusion.hits[1].provenance})
    assert not evaluate_answer(
        spec, spec.questions[0].id, answer.model_copy(update={"claims": (wrong,)})
    ).structural_citations_valid
    data: Any = answer.model_dump(mode="json")
    del data["status"]
    with pytest.raises(ValueError):
        ScientificAnswer.model_validate_json(json.dumps(data))


def test_expert_signature_and_numerical_label_mutations(spec: Specification) -> None:
    spec, response = synthetic(spec)
    answer = answer_fixture(spec, response)
    q = spec.questions[2]
    labels = {
        "spec_sha256": spec.digest,
        "question_id": q.id,
        "answer_sha256": digest_json(answer.model_dump(mode="json")),
        "reviewer": "SYNTHETIC TEST REVIEWER",
        "reviewed_at": datetime.now(UTC).isoformat(),
        "protocol_reference": "synthetic-protocol",
        "rationale": "Synthetic test only",
        "citation_support": "pass",
        "scientific_entailment": "pass",
        "numerical_fidelity": "pass",
        "population_applicability": "pass",
        "contradiction_handling": "not_applicable",
        "abstention_quality": "pass",
        "premise_correction": "not_applicable",
    }
    review = SignedReview.model_validate_json(
        json.dumps(
            {"labels": labels, "key_id": "synthetic-key", "signature": "not-a-real-signature"}
        )
    )
    with pytest.raises(ValueError, match="unauthenticated"):
        evaluate_answer(spec, q.id, answer, review=review)
    assert evaluate_answer(
        spec, q.id, answer, review=review, verify_signature=lambda _k, _p, _s: True
    ).scientific_pass
    changed = review.model_copy(
        update={"labels": review.labels.model_copy(update={"numerical_fidelity": "fail"})}
    )
    assert (
        evaluate_answer(
            spec, q.id, answer, review=changed, verify_signature=lambda _k, _p, _s: True
        ).scientific_pass
        is False
    )

    # The original signature must not verify changed labels, even if their new digest is supplied.
    def verifier(_k: str, p: bytes, _s: str) -> bool:
        return p == review.labels.canonical_bytes()

    with pytest.raises(ValueError):
        evaluate_answer(spec, q.id, answer, review=changed, verify_signature=verifier)
    false_q = spec.questions[8]
    unreviewed = evaluate_answer(spec, false_q.id, answer)
    assert unreviewed.scientific_pass is None


def test_execution_receipt_records_contract(spec: Specification) -> None:
    spec, response = synthetic(spec)
    run = make_run(spec, response)
    assert run.execution_config_sha256 == digest_json(EXECUTION)
    assert len(run.results) == 13 and all(len(r.rankings) == 3 for r in run.results)
    assert not evaluate(spec, run).performance_gate_enabled
    from dynamisrag.benchmark.dynamisbench_runner import verify_raw_responses

    verify_raw_responses(spec, run, (response,) * len(spec.questions))
    with pytest.raises(ValueError, match="incomplete raw"):
        verify_raw_responses(spec, run, (response,))
    changed = response.model_copy(
        update={
            "timings": response.timings.model_copy(update={"total_request_ms": 123456.0}),
        }
    )
    with pytest.raises(ValueError, match="reconstruct"):
        verify_raw_responses(spec, run, (changed,) * len(spec.questions))
    with pytest.raises(ValueError):
        compare(spec, run, run.model_copy(update={"results": run.results[:-1]}))


def test_grade_ranking_and_candidate_regressions(spec: Specification) -> None:
    spec, response = synthetic(spec)
    run = make_run(spec, response)
    first = spec.questions[0]
    ranking = run.results[0].rankings[0]
    top = ranking.hits[0].passage_key
    revised = first.model_copy(
        update={
            "judgments": tuple(
                j.model_copy(update={"grade": 1}) if j.passage_key == top else j
                for j in first.judgments
            )
        }
    )
    ndcg = score_ranking(spec, revised, ranking).ndcg10
    assert ndcg is not None and ndcg < 1
    # A genuine candidate ranking change must change paired rows and wins/losses.
    candidates = tuple(
        c.model_copy(update={"rank": i, "raw_score": float(51 - i)})
        for i, c in enumerate(reversed(response.bm25.candidates), 1)
    )
    lexical = tuple(
        SearchHit.model_validate(
            {**c.provenance.model_dump(), "rank": c.rank, "score": c.raw_score}
        )
        for c in candidates
    )
    dense = tuple(
        DenseCandidate(
            rank=c.rank,
            raw_score=c.raw_score,
            passage_key=c.provenance.passage_key,
            provenance=c.provenance,
        )
        for c in candidates
    )
    changed = response.model_copy(
        update={
            "bm25": ScientificLane(lane="text", candidates=candidates),
            "dense": ScientificLane(lane="dense", candidates=candidates),
            "fusion": reciprocal_rank_fusion(lexical=lexical, dense=dense, limit=20),
        }
    )
    comparison = compare(spec, run, make_run(spec, changed))
    assert comparison.regressions and any(a.losses for a in comparison.aggregates)
    with pytest.raises(ValueError, match="execution configuration"):
        evaluate(spec, run.model_copy(update={"execution_config_sha256": "0" * 64}))


def test_answer_inventory_and_abstention(spec: Specification) -> None:
    from dynamisrag.benchmark.dynamisbench_answers import compare_answers, evaluate_answers

    spec, response = synthetic(spec)
    answer = answer_fixture(spec, response)
    answers = {q.id: answer for q in spec.questions}
    baseline = evaluate_answers(spec, answers)
    assert len(baseline.results) == 13
    assert all(r.scientific_pass is None for r in baseline.results)
    with pytest.raises(ValueError, match="inventory"):
        evaluate_answers(spec, {q.id: answer for q in spec.questions[:-1]})
    abstention = answer.model_copy(
        update={"status": "insufficient_evidence", "answer": None, "claims": ()}
    )
    differences = compare_answers(spec, answers, {q.id: abstention for q in spec.questions})
    assert differences[0].unnecessary_refusal
    unanswerable = next(
        d for d in compare_answers(spec, answers, answers) if d.question_id == "db00-08"
    )
    assert unanswerable.unsupported_acceptance
