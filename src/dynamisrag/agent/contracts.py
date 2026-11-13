"""Typed Pydantic contracts for the RES-637 conversational scientific research agent.

Every tool input, tool output, study finding, cross-paper comparison,
statistical calculation, claim verification result, and provenance record is
defined here with strict validation (`extra="forbid"`, `frozen=True`).

Key invariants:
- All generated or extracted study findings carry `origin="MODEL_PROPOSED"` and
  `review_state="PROPOSED"`. No finding or claim may claim `REVIEWED` or
  expert-adjudicated scientific gold.
- Structural quotation and span verification establish that a quoted sentence
  exists at exact canonical character coordinates in PostgreSQL; they do not
  establish scientific entailment, study validity, or truth.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from dynamisrag.domain.evidence import Relation, ReviewState
from dynamisrag.domain.identity import evidence_identity
from dynamisrag.domain.values import Sha256Hex

AGENT_GRAPH_VERSION: Literal["dynamisrag-agent-graph-v2.0"] = "dynamisrag-agent-graph-v2.0"
AGENT_TOOL_POLICY_VERSION: Literal["scientific-tools-readonly-v2.0"] = (
    "scientific-tools-readonly-v2.0"
)
PROPOSED_LIMITATION_NOTICE: str = (
    "All study findings, comparisons, and synthesized claims are MODEL_PROPOSED / PROPOSED. "
    "Exact quotation and canonical source-span checks verify text provenance only and do not "
    "establish scientific entailment, study validity, clinical applicability, or expert review."
)

SafeShortText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=500,
        pattern=r"^[^\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+$",
    ),
]

SafeQueryText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=512,
        pattern=r"^[^\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+$",
    ),
]

SafeLongText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=4000,
        pattern=r"^[^\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+$",
    ),
]

UserId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}$",
    ),
]


class ToolName(StrEnum):
    """Permitted scientific tools in the RES-637 registry."""

    LIST_PAPERS = "list_papers"
    HYBRID_SEARCH = "hybrid_search"
    SEARCH_INDEX = "search_index"
    GET_PASSAGE = "get_passage"
    EXPAND_PASSAGE_CONTEXT = "expand_passage_context"
    GET_STUDY_METADATA = "get_study_metadata"
    EXTRACT_STUDY_FINDINGS = "extract_study_findings"
    COMPARE_STUDIES = "compare_studies"
    CALCULATE_STATISTICS = "calculate_statistics"
    VERIFY_CLAIMS = "verify_claims"
    DISCOVER_EXTERNAL_LITERATURE = "discover_external_literature"


class ToolErrorCode(StrEnum):
    """Deterministic, machine-readable tool failure codes."""

    INVALID_INPUT = "invalid_input"
    UNAUTHORIZED_SCOPE = "unauthorized_scope"
    RESOURCE_NOT_FOUND = "resource_not_found"
    PROVENANCE_INVALID = "provenance_invalid"
    INJECTION_DETECTED = "injection_detected"
    INCOMPATIBLE_UNITS_OR_OUTCOMES = "incompatible_units_or_outcomes"
    BUDGET_EXCEEDED = "budget_exceeded"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    EXTERNAL_DISCOVERY_DISABLED = "external_discovery_disabled"
    TIMEOUT = "timeout"
    INTERNAL_ERROR = "internal_error"


class AgentContract(BaseModel):
    """Frozen, closed base model for all agent contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DataScope(AgentContract):
    """Explicit authorization and corpus boundary enforced by every tool call."""

    user_id: UserId
    thread_id: SafeShortText
    run_id: SafeShortText
    allowed_version_keys: tuple[Sha256Hex, ...] = ()
    allow_opensearch_projection: bool = True
    allow_external_discovery: bool = False
    max_passages_per_call: int = Field(default=10, ge=1, le=25)
    max_output_bytes: int = Field(default=32768, ge=1024, le=131072)
    timeout_seconds: float = Field(default=15.0, gt=0.0, le=120.0)

    def allows_version(self, version_key: str) -> bool:
        if not self.allowed_version_keys:
            return True
        return version_key in self.allowed_version_keys


class ToolProvenance(AgentContract):
    """Audit provenance attached to every tool execution result."""

    tool_name: ToolName
    trace_id: SafeShortText
    policy_version: Literal["scientific-tools-readonly-v2.0"] = AGENT_TOOL_POLICY_VERSION
    corpus_identity: str | None = None
    canonical_verified: bool = False
    duration_ms: float = Field(ge=0.0)
    output_sha256: Sha256Hex


class ToolErrorDetail(AgentContract):
    """Structured error returned when a tool fails closed."""

    code: ToolErrorCode
    message: SafeShortText
    tool_name: ToolName
    trace_id: SafeShortText
    retryable: bool = False


# ---------------------------------------------------------------------------
# 1. list_papers
# ---------------------------------------------------------------------------


class ListPapersInput(AgentContract):
    """Input contract for `list_papers`."""

    document_version_key: Sha256Hex | None = None
    query_filter: SafeShortText | None = None
    limit: int = Field(default=20, ge=1, le=50)


class PaperSummary(AgentContract):
    """Authorized scientific paper summary."""

    document_version_key: Sha256Hex
    document_canonical_key: SafeShortText
    title: SafeShortText
    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    language: str = "en"
    passage_count: int = Field(ge=0)
    section_titles: tuple[str, ...] = ()


class ListPapersOutput(AgentContract):
    """Output contract for `list_papers`."""

    papers: tuple[PaperSummary, ...]
    total_authorized: int = Field(ge=0)
    corpus_identity: str


# ---------------------------------------------------------------------------
# 2. hybrid_search
# ---------------------------------------------------------------------------


class HybridSearchInput(AgentContract):
    """Input contract for `hybrid_search` over canonical scientific evidence."""

    query: SafeQueryText
    limit: int = Field(default=5, ge=1, le=20)
    document_version_keys: tuple[Sha256Hex, ...] = Field(default=(), max_length=10)
    section_filter: SafeShortText | None = None


class PassageSpanItem(AgentContract):
    """Canonical paragraph character span for a passage."""

    span_key: Sha256Hex
    paragraph_key: Sha256Hex
    paragraph_source_anchor: SafeShortText
    source_order: int = Field(ge=0)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)

    @model_validator(mode="after")
    def valid_range(self) -> Self:
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")
        return self


class SearchHitItem(AgentContract):
    """One ranked passage hit with verified canonical provenance."""

    passage_key: Sha256Hex
    document_version_key: Sha256Hex
    document_canonical_key: SafeShortText
    title: SafeShortText
    section_key: Sha256Hex | None = None
    section_title: str | None = None
    section_path: str | None = None
    text: str = Field(min_length=1, max_length=12000)
    content_sha256: Sha256Hex
    rank: int = Field(ge=1)
    score: float
    bm25_rank: int | None = Field(default=None, ge=1)
    dense_rank: int | None = Field(default=None, ge=1)
    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    source_spans: tuple[PassageSpanItem, ...] = ()
    canonical_verified: bool = True


class HybridSearchOutput(AgentContract):
    """Output contract for `hybrid_search`."""

    query: SafeQueryText
    hits: tuple[SearchHitItem, ...]
    corpus_identity: str
    retrieval_mode: Literal["qdrant_hybrid_rrf", "canonical_lexical_fallback"]


# ---------------------------------------------------------------------------
# 3. search_index
# ---------------------------------------------------------------------------


class SearchIndexInput(AgentContract):
    """Input contract for `search_index` over OpenSearch projections."""

    query: SafeQueryText
    limit: int = Field(default=5, ge=1, le=20)
    mode: Literal["bm25", "hybrid"] = "bm25"
    document_version_key: Sha256Hex | None = None


class SearchIndexOutput(AgentContract):
    """Output contract for `search_index`."""

    query: SafeQueryText
    mode: Literal["bm25", "hybrid"]
    physical_index: SafeShortText
    projection_sha256: Sha256Hex
    hits: tuple[SearchHitItem, ...]


# ---------------------------------------------------------------------------
# 4. get_passage
# ---------------------------------------------------------------------------


class GetPassageInput(AgentContract):
    """Input contract for `get_passage`."""

    passage_key: Sha256Hex
    document_version_key: Sha256Hex | None = None


class PassageDetail(AgentContract):
    """Complete canonical passage with reconstructed source spans."""

    passage_key: Sha256Hex
    document_version_key: Sha256Hex
    document_canonical_key: SafeShortText
    title: SafeShortText
    language: str
    chunker_revision: SafeShortText
    ordinal: int = Field(ge=0)
    token_count: int = Field(ge=0)
    section_key: Sha256Hex | None = None
    section_title: str | None = None
    section_path: str | None = None
    text: str = Field(min_length=1, max_length=12000)
    content_sha256: Sha256Hex
    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    source_spans: tuple[PassageSpanItem, ...] = Field(min_length=1)
    provenance_sha256: Sha256Hex
    canonical_verified: Literal[True] = True


class GetPassageOutput(AgentContract):
    """Output contract for `get_passage`."""

    passage: PassageDetail


# ---------------------------------------------------------------------------
# 5. expand_passage_context
# ---------------------------------------------------------------------------


class ExpandPassageContextInput(AgentContract):
    """Input contract for `expand_passage_context`."""

    passage_key: Sha256Hex
    sibling_distance: int = Field(default=1, ge=1, le=3)
    include_parent_section: bool = True
    max_paragraphs: int = Field(default=4, ge=1, le=8)


class ExpandedParagraphItem(AgentContract):
    """Neighboring canonical paragraph in the same body section or parent section."""

    paragraph_key: Sha256Hex
    document_version_key: Sha256Hex
    section_key: Sha256Hex | None = None
    section_title: str | None = None
    ordinal: int = Field(ge=0)
    region: Literal["body"] = "body"
    source_anchor: SafeShortText
    kind: Literal["sibling", "parent"]
    text: str = Field(min_length=1, max_length=12000)
    content_sha256: Sha256Hex
    start_char: Literal[0] = 0
    end_char: int = Field(gt=0)


class ExpandPassageContextOutput(AgentContract):
    """Output contract for `expand_passage_context`."""

    anchor_passage_key: Sha256Hex
    document_version_key: Sha256Hex
    paragraphs: tuple[ExpandedParagraphItem, ...]
    exclusions: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# 6. get_study_metadata
# ---------------------------------------------------------------------------


class GetStudyMetadataInput(AgentContract):
    """Input contract for `get_study_metadata`."""

    document_version_key: Sha256Hex


class StudyMetadataDetail(AgentContract):
    """Canonical study identity and structural metadata."""

    document_version_key: Sha256Hex
    document_canonical_key: SafeShortText
    title: SafeShortText
    document_type: SafeShortText
    language: str
    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    source_system: SafeShortText
    source_external_id: SafeShortText
    source_artifact_sha256: Sha256Hex
    parser_revision: SafeShortText
    normalizer_revision: SafeShortText
    content_fingerprint: Sha256Hex
    license_uri: str | None = None
    versioned_metadata: dict[str, object] = Field(default_factory=dict)
    sections: tuple[str, ...] = ()
    passage_count: int = Field(ge=0)
    paragraph_count: int = Field(ge=0)
    table_count: int = Field(ge=0)
    figure_count: int = Field(ge=0)
    citation_count: int = Field(ge=0)


class GetStudyMetadataOutput(AgentContract):
    """Output contract for `get_study_metadata`."""

    study: StudyMetadataDetail


# ---------------------------------------------------------------------------
# 7. extract_study_findings
# ---------------------------------------------------------------------------


class ExtractStudyFindingsInput(AgentContract):
    """Input contract for `extract_study_findings`."""

    passage_keys: tuple[Sha256Hex, ...] = Field(min_length=1, max_length=8)
    research_focus: SafeQueryText | None = None
    max_findings: int = Field(default=5, ge=1, le=10)


class ProposedStudyFinding(AgentContract):
    """Structured study finding extracted from a canonical passage span.

    Always carries `origin="MODEL_PROPOSED"` and `review_state="PROPOSED"`.
    """

    finding_id: Sha256Hex
    document_version_key: Sha256Hex
    document_title: SafeShortText
    doi: str | None = None
    pmcid: str | None = None
    passage_key: Sha256Hex
    span_key: Sha256Hex
    paragraph_key: Sha256Hex
    paragraph_source_anchor: SafeShortText
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    verbatim_quote: str = Field(min_length=15, max_length=650)
    category: Literal["DESCRIPTIVE", "METHODOLOGICAL", "QUANTITATIVE", "ASSOCIATIONAL", "CAUSAL"]
    study_design: str | None = None
    population: str | None = None
    intervention: str | None = None
    comparator: str | None = None
    measurement_protocol: str | None = None
    outcome_measures: tuple[str, ...] = ()
    units: str | None = None
    reported_statistics: tuple[str, ...] = ()
    missing_information: tuple[str, ...] = ()
    uncertainty_notes: tuple[str, ...] = ()
    origin: Literal["MODEL_PROPOSED"] = "MODEL_PROPOSED"
    review_state: Literal[ReviewState.PROPOSED] = ReviewState.PROPOSED
    structurally_verified: Literal[True] = True

    @model_validator(mode="after")
    def valid_coordinates(self) -> Self:
        if self.end_char <= self.start_char:
            raise ValueError("finding end_char must exceed start_char")
        return self


class ExtractStudyFindingsOutput(AgentContract):
    """Output contract for `extract_study_findings`."""

    findings: tuple[ProposedStudyFinding, ...]
    excluded_passages: tuple[str, ...] = ()
    limitation_notice: str = PROPOSED_LIMITATION_NOTICE


# ---------------------------------------------------------------------------
# 8. compare_studies
# ---------------------------------------------------------------------------


class CompareStudiesInput(AgentContract):
    """Input contract for `compare_studies`."""

    document_version_keys: tuple[Sha256Hex, ...] = Field(min_length=2, max_length=6)
    passage_keys: tuple[Sha256Hex, ...] = Field(default=(), max_length=15)
    comparison_aspect: SafeQueryText | None = None


class StudyComparisonDimension(AgentContract):
    """Per-study structured attributes for cross-paper comparison."""

    document_version_key: Sha256Hex
    title: SafeShortText
    doi: str | None = None
    pmcid: str | None = None
    study_design: str
    population: str
    intervention: str
    control_or_comparator: str
    measurement_protocol: str
    outcomes: tuple[str, ...]
    units: tuple[str, ...]
    reported_results: tuple[str, ...]
    supporting_passage_keys: tuple[Sha256Hex, ...] = Field(min_length=1)
    supporting_span_keys: tuple[Sha256Hex, ...] = Field(min_length=1)
    missing_information: tuple[str, ...] = ()


class StudyComparisonMatrix(AgentContract):
    """Cross-paper comparison synthesis distinguishing methodology, outcomes, and compatibility."""

    comparison_id: Sha256Hex
    studies: tuple[StudyComparisonDimension, ...] = Field(min_length=2)
    evidence_compatibility: Literal[
        "DIRECTLY_COMPARABLE",
        "METHODOLOGICALLY_HETEROGENEOUS",
        "INCOMPATIBLE_OUTCOMES_OR_UNITS",
        "INSUFFICIENT_OVERLAP",
    ]
    compatibility_rationale: SafeLongText
    shared_constructs: tuple[str, ...] = ()
    methodology_differences: tuple[str, ...] = ()
    outcome_and_unit_differences: tuple[str, ...] = ()
    unfounded_pooling_prohibited: Literal[True] = True
    uncertainty_and_gaps: tuple[str, ...] = ()
    origin: Literal["MODEL_PROPOSED"] = "MODEL_PROPOSED"
    review_state: Literal[ReviewState.PROPOSED] = ReviewState.PROPOSED
    limitation_notice: str = PROPOSED_LIMITATION_NOTICE


class CompareStudiesOutput(AgentContract):
    """Output contract for `compare_studies`."""

    comparison: StudyComparisonMatrix


# ---------------------------------------------------------------------------
# 9. calculate_statistics
# ---------------------------------------------------------------------------


class StatisticalCalculationKind(StrEnum):
    """Permitted bounded deterministic statistical calculations."""

    SUMMARY_STATS = "summary_stats"
    PERCENTAGE_CHANGE = "percentage_change"
    MEAN_DIFFERENCE = "mean_difference"
    HEDGES_G_SAME_UNIT = "hedges_g_same_unit"


class StatisticalGroupInput(AgentContract):
    """Validated numerical group input with explicit unit and outcome provenance."""

    label: SafeShortText
    outcome_name: SafeShortText
    unit: SafeShortText
    measurement_protocol: SafeShortText | None = None
    values: tuple[float, ...] = Field(default=(), max_length=200)
    mean: float | None = Field(default=None, allow_inf_nan=False)
    sd: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    sample_size: int | None = Field(default=None, ge=1, le=100000)
    source_passage_key: Sha256Hex | None = None


class CalculateStatisticsInput(AgentContract):
    """Input contract for `calculate_statistics`."""

    calculation: StatisticalCalculationKind
    groups: tuple[StatisticalGroupInput, ...] = Field(min_length=1, max_length=5)


class CalculateStatisticsOutput(AgentContract):
    """Output contract for `calculate_statistics`."""

    calculation: StatisticalCalculationKind
    outcome_name: SafeShortText
    unit: SafeShortText
    compatible: bool
    result_metrics: dict[str, float]
    formula_used: SafeShortText
    warnings: tuple[str, ...] = ()
    limitation_notice: str = (
        "Deterministic arithmetic on reported values only; does not establish statistical "
        "significance or cross-study meta-analytic validity."
    )


# ---------------------------------------------------------------------------
# 10. verify_claims
# ---------------------------------------------------------------------------


class CandidateClaimInput(AgentContract):
    """A candidate scientific claim and its cited canonical evidence."""

    claim_text: str = Field(min_length=1, max_length=650)
    passage_key: Sha256Hex
    span_key: Sha256Hex | None = None
    verbatim_quote: str = Field(min_length=10, max_length=650)
    category: Literal[
        "DESCRIPTIVE", "METHODOLOGICAL", "QUANTITATIVE", "ASSOCIATIONAL", "CAUSAL"
    ] = "DESCRIPTIVE"
    relation: Relation = Relation.SUPPORTS


class VerifyClaimsInput(AgentContract):
    """Input contract for `verify_claims`."""

    question: SafeQueryText
    claims: tuple[CandidateClaimInput, ...] = Field(min_length=1, max_length=10)


class VerifiedCitationItem(AgentContract):
    """Independent canonical verification outcome for one candidate claim."""

    claim_index: int = Field(ge=0)
    claim_text: str
    verbatim_quote: str
    passage_key: Sha256Hex
    span_key: Sha256Hex | None = None
    document_version_key: Sha256Hex | None = None
    document_title: str | None = None
    doi: str | None = None
    pmcid: str | None = None
    paragraph_key: Sha256Hex | None = None
    paragraph_source_anchor: str | None = None
    start_char: int | None = None
    end_char: int | None = None
    verified: bool
    """Accepted for display: layers 1-3 below did not fail. NEVER means expert-approved."""
    rejection_reason: str | None = None
    # --- independent verification layers (never collapsed into one flag) -------------------
    quote_verified: bool = False
    """Layer 1: the quote occurs exactly in the canonical span and source identity is valid."""
    numeric_provenance: Literal["VERIFIED", "FAILED", "NOT_APPLICABLE"] = "NOT_APPLICABLE"
    """Layer 2: every number in the claim occurs in the quoted span."""
    semantic_support: Literal[
        "IDENTICAL_TO_QUOTE", "SUPPORTED_BY_LEXICAL_HEURISTIC", "UNSUPPORTED", "NOT_ASSESSED"
    ] = "NOT_ASSESSED"
    """Layer 3: does the quote support the claim's wording (heuristic, not entailment)."""
    semantic_support_score: float | None = None
    semantic_support_reasons: tuple[str, ...] = ()
    semantic_assessor: Literal["lexical-consistency-v1"] = "lexical-consistency-v1"
    expert_review: Literal["NOT_REVIEWED"] = "NOT_REVIEWED"
    """Layer 4: independent expert approval. Nothing in RES-637 ever sets this."""
    origin: Literal["MODEL_PROPOSED"] = "MODEL_PROPOSED"
    review_state: Literal[ReviewState.PROPOSED] = ReviewState.PROPOSED


class VerifyClaimsOutput(AgentContract):
    """Output contract for `verify_claims`."""

    all_verified: bool
    verified_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    items: tuple[VerifiedCitationItem, ...]
    evidence_set_key: Sha256Hex | None = None
    evidence_graph_key: Sha256Hex | None = None
    limitation_notice: str = PROPOSED_LIMITATION_NOTICE


# ---------------------------------------------------------------------------
# External literature discovery (explicitly restricted / disabled by default)
# ---------------------------------------------------------------------------


class DiscoverExternalLiteratureInput(AgentContract):
    """Input contract for restricted external literature discovery."""

    pmcid: Annotated[str, StringConstraints(pattern=r"^PMC[0-9]{1,12}$")]


class DiscoverExternalLiteratureOutput(AgentContract):
    """Output contract for restricted external literature discovery.

    Never ingests external material into canonical state.
    """

    enabled: bool
    pmcid: str
    status: Literal["disabled_by_policy", "preview_only"]
    message: SafeShortText
    canonical_ingested: Literal[False] = False


# ---------------------------------------------------------------------------
# Tool execution envelope
# ---------------------------------------------------------------------------


class ToolExecutionEnvelope(AgentContract):
    """Standardized wrapper for tool execution results or structured errors."""

    ok: bool
    tool_name: ToolName
    provenance: ToolProvenance
    payload: dict[str, Any] | None = None
    error: ToolErrorDetail | None = None

    @model_validator(mode="after")
    def valid_state(self) -> Self:
        if self.ok and (self.payload is None or self.error is not None):
            raise ValueError("successful tool envelope requires payload and no error")
        if not self.ok and (self.error is None or self.payload is not None):
            raise ValueError("failed tool envelope requires error and no payload")
        return self

    @property
    def key(self) -> str:
        return evidence_identity("tool-envelope-v2", self.model_dump(mode="json"))
