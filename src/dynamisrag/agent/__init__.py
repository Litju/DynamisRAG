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

__all__ = [
    "AGENT_GRAPH_VERSION",
    "AGENT_TOOL_POLICY_VERSION",
    "PROPOSED_LIMITATION_NOTICE",
    "CalculateStatisticsInput",
    "CalculateStatisticsOutput",
    "CompareStudiesInput",
    "CompareStudiesOutput",
    "DataScope",
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
    "ListPapersInput",
    "ListPapersOutput",
    "PaperSummary",
    "PassageDetail",
    "PassageSpanItem",
    "ProposedStudyFinding",
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
]
