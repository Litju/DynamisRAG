"""Conversational scientific research-agent package (RES-637).

Provides:
- Typed Pydantic contracts for scientific tools, study findings, cross-paper
  comparisons, statistical calculations, citation verification, and agent state.
- Read-only scientific tool adapters backed by canonical PostgreSQL, Qdrant
  hybrid retrieval, and OpenSearch projections.
- LangChain model provider abstraction and tool bindings.
- Explicit LangGraph stateful orchestration with durable PostgreSQL checkpoints.
- Versioned FastAPI `/api/v2` thread, run, event-streaming, and evidence routes.
- LangSmith privacy-redacted tracing and deterministic evaluation harnesses.
"""

from __future__ import annotations

from dynamisrag.agent.contracts import (
    AGENT_GRAPH_VERSION,
    AGENT_TOOL_POLICY_VERSION,
    PROPOSED_LIMITATION_NOTICE,
    CalculateStatisticsInput,
    CalculateStatisticsOutput,
    CompareStudiesInput,
    CompareStudiesOutput,
    DataScope,
    ExpandedParagraphItem,
    ExpandPassageContextInput,
    ExpandPassageContextOutput,
    ExtractStudyFindingsInput,
    ExtractStudyFindingsOutput,
    GetPassageInput,
    GetPassageOutput,
    GetStudyMetadataInput,
    GetStudyMetadataOutput,
    HybridSearchInput,
    HybridSearchOutput,
    ListPapersInput,
    ListPapersOutput,
    PaperSummary,
    PassageDetail,
    PassageSpanItem,
    ProposedStudyFinding,
    SearchHitItem,
    SearchIndexInput,
    SearchIndexOutput,
    StatisticalCalculationKind,
    StudyComparisonDimension,
    StudyComparisonMatrix,
    StudyMetadataDetail,
    ToolErrorCode,
    ToolErrorDetail,
    ToolExecutionEnvelope,
    ToolName,
    ToolProvenance,
    VerifiedCitationItem,
    VerifyClaimsInput,
    VerifyClaimsOutput,
)
from dynamisrag.agent.graph import ConversationalResearchOrchestrator
from dynamisrag.agent.langsmith_eval import (
    EvaluationCaseResult,
    EvaluationCaseSpec,
    EvaluationSuiteReport,
    configure_privacy_safe_langsmith,
    default_res637_evaluation_specs,
    evaluate_agent_run,
    redact_trace_payload,
    redact_trace_string,
    run_offline_evaluation_suite,
)
from dynamisrag.agent.persistence import (
    AgentConversationStore,
    InMemoryAgentConversationStore,
    PostgresAgentConversationStore,
)
from dynamisrag.agent.providers import (
    AgentModelConfig,
    AgentModelRuntime,
    AgentProviderKind,
    DeterministicResearchChatModel,
    build_agent_model_runtime,
)
from dynamisrag.agent.router import build_agent_v2_router
from dynamisrag.agent.tools import ScientificToolRegistry

__all__ = [
    "AGENT_GRAPH_VERSION",
    "AGENT_TOOL_POLICY_VERSION",
    "PROPOSED_LIMITATION_NOTICE",
    "AgentConversationStore",
    "AgentModelConfig",
    "AgentModelRuntime",
    "AgentProviderKind",
    "CalculateStatisticsInput",
    "CalculateStatisticsOutput",
    "CompareStudiesInput",
    "CompareStudiesOutput",
    "ConversationalResearchOrchestrator",
    "DataScope",
    "DeterministicResearchChatModel",
    "EvaluationCaseResult",
    "EvaluationCaseSpec",
    "EvaluationSuiteReport",
    "ExpandPassageContextInput",
    "ExpandPassageContextOutput",
    "ExpandedParagraphItem",
    "ExtractStudyFindingsInput",
    "ExtractStudyFindingsOutput",
    "GetPassageInput",
    "GetPassageOutput",
    "GetStudyMetadataInput",
    "GetStudyMetadataOutput",
    "HybridSearchInput",
    "HybridSearchOutput",
    "InMemoryAgentConversationStore",
    "ListPapersInput",
    "ListPapersOutput",
    "PaperSummary",
    "PassageDetail",
    "PassageSpanItem",
    "PostgresAgentConversationStore",
    "ProposedStudyFinding",
    "ScientificToolRegistry",
    "SearchHitItem",
    "SearchIndexInput",
    "SearchIndexOutput",
    "StatisticalCalculationKind",
    "StudyComparisonDimension",
    "StudyComparisonMatrix",
    "StudyMetadataDetail",
    "ToolErrorCode",
    "ToolErrorDetail",
    "ToolExecutionEnvelope",
    "ToolName",
    "ToolProvenance",
    "VerifiedCitationItem",
    "VerifyClaimsInput",
    "VerifyClaimsOutput",
    "build_agent_model_runtime",
    "build_agent_v2_router",
    "configure_privacy_safe_langsmith",
    "default_res637_evaluation_specs",
    "evaluate_agent_run",
    "redact_trace_payload",
    "redact_trace_string",
    "run_offline_evaluation_suite",
]
