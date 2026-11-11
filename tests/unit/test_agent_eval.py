"""Unit tests for LangSmith privacy-safe redaction and offline/CI evaluators."""

from __future__ import annotations

from typing import cast

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from dynamisrag.agent.graph import ConversationalResearchOrchestrator
from dynamisrag.agent.langsmith_eval import (
    configure_privacy_safe_langsmith,
    redact_trace_payload,
    run_offline_evaluation_suite,
)
from dynamisrag.agent.persistence import InMemoryAgentConversationStore
from dynamisrag.agent.providers import (
    AgentModelConfig,
    AgentProviderKind,
    build_agent_model_runtime,
)
from dynamisrag.agent.tools import ScientificToolRegistry
from tests.unit.test_agent_tools import (
    _create_sqlite_schema,  # pyright: ignore[reportPrivateUsage]
    _seed_study,  # pyright: ignore[reportPrivateUsage]
)


def test_redact_trace_payload_scrubs_secrets_dsns_and_emails() -> None:
    raw_payload = {
        "api_key": "sk-secret-key-12345",
        "database_url": "postgresql://dynamisrag:supersecret@127.0.0.1:55433/dynamisrag",
        "nested": {
            "auth_header": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
            "note": "Contact researcher@university.edu or postgresql://u:pw@host/db",
            "text": "A" * 500,
        },
    }
    redacted = cast("dict[str, object]", redact_trace_payload(raw_payload))
    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["database_url"] == "[REDACTED]"
    nested = cast("dict[str, object]", redacted["nested"])
    assert "eyJhbG" not in str(nested["auth_header"])
    assert "researcher@university.edu" not in str(nested["note"])
    assert "pw@" not in str(nested["note"])
    assert str(nested["text"]).endswith("...[TRUNCATED]")

    env_flags = configure_privacy_safe_langsmith(tracing_enabled=False)
    assert env_flags["LANGSMITH_TRACING"] == "false"
    assert env_flags["LANGCHAIN_HIDE_INPUTS"] == "true"


def test_offline_evaluation_suite_scores_all_six_scenarios() -> None:
    engine: Engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    _create_sqlite_schema(engine)
    with Session(engine) as session:
        _seed_study(
            session,
            pmcid="PMC5263151",
            doi="10.3389/fphys.2017.00016",
            pmid="28174540",
            title=(
                "Plyometric Training Favors Optimizing Muscle-Tendon Behavior during Depth Jumping"
            ),
            section_title="Muscle-Tendon Behavior Results",
            paragraphs_text=(
                (
                    "Twenty-one young adults completed 12 weeks of plyometric training "
                    "with ultrasound and motion capture measurements during depth jumping."
                ),
                (
                    "During depth jumping, Achilles tendon stiffness increased by 24.5% "
                    "(p < 0.01) while gastrocnemius medialis fascicle length behavior "
                    "shifted toward isometric contraction."
                ),
            ),
        )
        _seed_study(
            session,
            pmcid="PMC10780924",
            doi="10.3390/sports12010015",
            pmid="38251201",
            title=(
                "Effects of Plyometric Training on Muscle-Tendon Stiffness and Jump Performance"
            ),
            section_title="Results and Discussion",
            paragraphs_text=(
                (
                    "Countermovement jump height increased by 14.2% (p < 0.001) in "
                    "the plyometric group, measured via force plate at 48.5 ± 4.1 cm."
                ),
            ),
        )
        session.commit()

    store = InMemoryAgentConversationStore()
    registry = ScientificToolRegistry(engine)
    runtime = build_agent_model_runtime(AgentModelConfig(provider=AgentProviderKind.DETERMINISTIC))
    orchestrator = ConversationalResearchOrchestrator(
        tool_registry=registry,
        store=store,
        model_runtime=runtime,
    )

    report = run_offline_evaluation_suite(orchestrator=orchestrator, store=store)
    assert report.total_cases == 6
    assert report.passed_cases == 6
    assert report.pass_rate == 1.0
    assert report.mean_citation_fidelity_score == 1.0
    assert report.mean_abstention_accuracy_score == 1.0
    assert report.mean_context_resolution_score == 1.0
    assert report.mean_injection_resistance_score == 1.0
