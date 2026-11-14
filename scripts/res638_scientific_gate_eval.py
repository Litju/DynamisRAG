"""RES-638 scientific relevance/answerability qualification against the licensed corpus.

This is an operator script, not a production entry point. It runs the real LangGraph
orchestrator, the real canonical PostgreSQL tools, the deterministic zero-inference model and
the structured evidence gate. It never calls a paid model or a hosted service.

    uv run python -m scripts.res638_scientific_gate_eval --output <path>


A failed expectation exits non-zero and records the failure; it is never rewritten as a pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from dynamisrag.agent.graph import ConversationalResearchOrchestrator
from dynamisrag.agent.persistence import InMemoryAgentConversationStore
from dynamisrag.agent.providers import (
    AgentModelConfig,
    AgentProviderKind,
    build_agent_model_runtime,
)
from dynamisrag.agent.tools import ScientificToolRegistry
from dynamisrag.config import Settings, to_sqlalchemy_url
from dynamisrag.db.models import DocumentVersionRecord, PassageRecord
from dynamisrag.search.scientific import CHUNKER


def _settings_engine() -> tuple[Settings, Engine]:
    load_dotenv(".env.workspace")
    load_dotenv()
    settings = Settings()
    if settings.database_url is None:
        raise SystemExit("DYNAMISRAG_DATABASE_URL is required")
    url = to_sqlalchemy_url(settings.database_url)
    if url.startswith("postgresql") and "workspace_reader" not in url:
        raise SystemExit("Refusing a non-read-only scientific corpus URL")
    return settings, create_engine(url)


def _corpus_receipt(engine: Engine) -> dict[str, object]:
    with Session(engine) as session:
        versions = tuple(
            session.execute(
                select(
                    DocumentVersionRecord.version_key,
                    DocumentVersionRecord.title,
                ).order_by(DocumentVersionRecord.version_key)
            ).all()
        )
        passages = session.scalar(select(func.count()).select_from(PassageRecord)) or 0
    return {
        "document_versions": [
            {"version_key": key, "title": title[:300]} for key, title in versions
        ],
        "passage_count": passages,
        "chunker_revision": CHUNKER,
        "document_count": len(versions),
    }


def _run_case(
    orchestrator: ConversationalResearchOrchestrator,
    store: InMemoryAgentConversationStore,
    *,
    case_id: str,
    turns: tuple[str, ...],
) -> dict[str, object]:
    thread = store.create_thread(owner_id="res638-eval", title=f"RES-638 {case_id}")
    runs: list[dict[str, object]] = []
    for index, question in enumerate(turns):
        run = orchestrator.execute_run(
            thread_id=thread.thread_id,
            user_id="res638-eval",
            question=question,
            idempotency_key=f"{case_id}-{index}",
        )
        events = [
            event.model_dump(mode="json")
            for event in store.list_events(run.run_id, user_id="res638-eval")
            if event.event_type.value == "evidence.relevance_assessed"
        ]
        runs.append(
            {
                "question": question,
                "run_id": run.run_id,
                "status": run.status.value,
                "abstention_reason": run.abstention_reason,
                "verified_citation_count": len(run.verified_citations),
                "verified_quotes": [
                    {
                        "document_title": citation.get("document_title"),
                        "pmcid": citation.get("pmcid"),
                        "passage_key": citation.get("passage_key"),
                        "verbatim_quote": citation.get("verbatim_quote"),
                    }
                    for citation in run.verified_citations
                ],
                "answerability_events": [
                    {
                        "stage": event["stage"],
                        "verdict": event["payload"].get("verdict"),
                        "code": event["payload"].get("code"),
                        "question_kind": event["payload"].get("question_kind"),
                        "accepted": len(event["payload"].get("accepted_passage_keys", [])),
                        "rejected": len(event["payload"].get("rejected_passage_keys", [])),
                    }
                    for event in events
                ],
            }
        )
    return {
        "case_id": case_id,
        "turns": runs,
        "final_status": runs[-1]["status"],
        "final_abstention_reason": runs[-1]["abstention_reason"],
    }


def _case_specs() -> tuple[dict[str, object], ...]:
    return (
        {
            "case_id": "positive_achilles",
            "expect": {"status": "completed", "min_citations": 1},
            "turns": ("What happened to Achilles tendon stiffness during depth jumping?",),
        },
        {
            "case_id": "positive_paraphrase_knee_angle",
            "expect": {"status": "completed", "min_citations": 1},
            "turns": ("How did knee flexion angle affect muscle activity?",),
        },
        {
            "case_id": "positive_comparison",
            "expect": {"status": "completed", "min_citations": 1, "comparison": True},
            "turns": (
                "Compare the study designs and populations of the plyometric studies "
                "in this corpus.",
            ),
        },
        {
            "case_id": "positive_multi_turn",
            "expect": {"status": "completed", "min_citations": 1},
            "turns": (
                "What did the depth jumping study find about Achilles tendon stiffness?",
                "How many weeks did those participants complete?",
            ),
        },
        {
            "case_id": "negative_off_domain_kepler",
            "expect": {"status": "abstained", "abstention_contains": "unsupported_entity"},
            "turns": ("What is the orbital period of the exoplanet Kepler-22b?",),
        },
        {
            "case_id": "negative_off_domain_proxima",
            "expect": {"status": "abstained", "abstention_contains": "unsupported_entity"},
            "turns": ("What is the orbital radius of Proxima Centauri b?",),
        },
        {
            "case_id": "negative_close_domain_sprint",
            "expect": {"status": "abstained", "abstention_contains": "insufficient_query_coverage"},
            "turns": ("By what percentage did plyometric training improve 100 m sprint times?",),
        },
        {
            "case_id": "negative_off_domain_sourdough",
            "expect": {"status": "abstained", "abstention_contains": "insufficient_query_coverage"},
            "turns": ("What hydration ratio do the studies recommend for sourdough bread?",),
        },
        {
            "case_id": "negative_unknown_entity",
            "expect": {"status": "abstained", "abstention_contains": "unsupported_entity"},
            "turns": ("What did the Proxima Centauri study report about jump height?",),
        },
    )


def _verify(case: dict[str, object], result: dict[str, object]) -> list[str]:
    expected = case["expect"]
    if not isinstance(expected, dict):
        return ["invalid_expectation"]
    failures: list[str] = []
    status = result["final_status"]
    if status != expected["status"]:
        failures.append(f"status={status}; expected {expected['status']}")
    min_citations = expected.get("min_citations")
    if min_citations is not None:
        turns = result["turns"]
        last = turns[-1] if isinstance(turns, list) and turns else {}
        citations = last.get("verified_citation_count") if isinstance(last, dict) else 0
        if not isinstance(citations, int) or citations < min_citations:
            failures.append(f"verified_citation_count={citations}; expected >= {min_citations}")
    needle = expected.get("abstention_contains")
    if needle is not None and needle not in str(result["final_abstention_reason"] or ""):
        failures.append(
            f"abstention_reason={result['final_abstention_reason']!r}; expected {needle!r}"
        )
    if expected.get("comparison"):
        turns = result["turns"]
        last = turns[-1] if isinstance(turns, list) and turns else {}
        events = last.get("answerability_events") if isinstance(last, dict) else []
        if not any(
            isinstance(event, dict) and "comparison" in str(event.get("code"))
            for event in (events or [])
        ):
            failures.append("no comparison answerability decision recorded")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/evidence/res638-scientific-gate/receipt.json"),
    )
    args = parser.parse_args()

    _, engine = _settings_engine()
    corpus = _corpus_receipt(engine)
    if corpus["document_count"] != 3 or corpus["passage_count"] != 75:
        raise SystemExit(
            f"Expected the licensed three-paper / 75-passage corpus, got "
            f"{corpus['document_count']} papers / {corpus['passage_count']} passages"
        )

    store = InMemoryAgentConversationStore()
    registry = ScientificToolRegistry(engine)
    runtime = build_agent_model_runtime(AgentModelConfig(provider=AgentProviderKind.DETERMINISTIC))
    orchestrator = ConversationalResearchOrchestrator(
        tool_registry=registry,
        store=store,
        model_runtime=runtime,
    )
    results: list[dict[str, object]] = []
    failures: list[str] = []
    for case in _case_specs():
        turns = case["turns"]
        if not isinstance(turns, tuple):
            raise SystemExit("invalid case turns")
        result = _run_case(
            orchestrator,
            store,
            case_id=str(case["case_id"]),
            turns=turns,
        )
        case_failures = _verify(case, result)
        result["expect"] = case["expect"]
        result["passed"] = not case_failures
        result["failures"] = case_failures
        results.append(result)
        failures.extend(f"{case['case_id']}: {failure}" for failure in case_failures)

    receipt = {
        "schema": "res638-scientific-gate-evaluation-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "code_revision": subprocess.check_output(  # noqa: S603 - fixed local command
            [shutil.which("git") or "git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "engine": {
            "provider": "deterministic",
            "paid_inference": False,
            "gate_version": "structured-evidence-gate-v1",
            "claim_support_label": "SUPPORTED_BY_STRUCTURED_EVIDENCE_GATE",
        },
        "corpus": corpus,
        "results": results,
        "summary": {
            "total": len(results),
            "passed": sum(1 for result in results if result["passed"]),
            "failed": len(failures),
        },
        "limitations": [
            "The deterministic model is a fixed test adapter, not a scientific language model.",
            "The gate is lexical/structural and does not establish semantic entailment "
            "or validity.",
            "Passing positive cases show relevance plumbing, not expert-reviewed science.",
            "The original Kepler-22b failure is preserved separately in "
            "original-kepler-failure.json.",
        ],
    }
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    sys.stdout.write(
        json.dumps({"output": str(output), "sha256": digest, "summary": receipt["summary"]}) + "\n"
    )
    if failures:
        for failure in failures:
            sys.stdout.write(f"FAILED: {failure}\n")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
