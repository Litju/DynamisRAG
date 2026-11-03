# RES-146 selection-v1 decision

Base: `ef5ed691b82af93489badbbb43dea439d10d904a`. Implementation is opt-in through
`dynamisrag.evidence.service.select_live(service, request)`. No answer route,
generation policy, retrieval model, embedding, historical migration, or benchmark
artifact changes. PostgreSQL must have migration `0005_evidence_graph`; the
demonstration installs it only in an ephemeral copy of the reference database.

## Reuse and boundaries

Reuse ScientificRetrievalService, its qualified projection and pinned query
runtime, the existing reciprocal_rank_fusion implementation, projection metadata,
and RES-145 canonical_member/validate_evidence_set/source_fingerprint verification.
Revalidate serialized Pydantic objects at entry, including nested copied objects.
Reconstruct every fused hit, lane contribution, score and rank. Qdrant `text` is
explicitly the lexical lane; `bm25` is its diagnostic label. Point UUIDs must derive
from passage keys; repeated keys within a lane and contradictory cross-lane
provenance are errors. Non-finite or increasing lane scores, missing ranks, wrong
runtime/configuration/collection/alias and forged fusion fail closed.

Leave generation/scientific.py's extractive-findings-v1 unchanged. Its question
overlap, sentence clipping and exclusion of Methods are unsuitable as a general
scientific evidence selector. selection-v1 neither bans Methods nor gives Results
an implicit priority. Table passages retain their complete original text. Source
prose, including instruction-like prose, is inert untrusted data: there is no
model, interpreter, claim authoring, or scientific approval operation here.

## Trust and execution identity

The application owns the initialized ScientificRetrievalService. select_live calls
retrieve with the exact request.question, policy retrieval limit and empty filters;
it accepts no caller-provided response. It checks requested projection/manifest
against that service's qualified projection before retrieval and independently
checks the resulting metadata and rankings. Existing retrieval verifies query
provider semantics and resolves the physical collection before/after execution.

select_offline accepts captured responses for structural qualification only. Its
`offline-operator-replay` label cannot certify which question generated them.
Matching caller-supplied question/corpus hashes never creates live authenticity.
Serialized authenticity labels are declarations, not signed execution receipts;
do not authorize a live run by loading a SelectionResult from an untrusted caller.
The demonstration additionally reconstructs every captured DynamisBench row from
the original private responses and verifies the pinned canonical specification.

Request identity binds exact question bytes, declared task requirements, depth,
corpus identity and every policy field. Result identity binds the request, exact
response (including original timings), decisions, context references and canonical
EvidenceSet. Identical frozen inputs replay identically without inference. A new
live request with different timing evidence has a distinct execution identity.

## Deterministic selection

The fused window is the sole ranking authority. Retain original fused and lane
ranks/scores in every decision; lane-only candidates receive outside_fused_window.
Every unique lane-union candidate has one disposition. Within-lane duplicates
are invalid retrieval, while normal cross-lane appearances represent one candidate.

First pass follows fused order with configurable soft document/section quotas.
Second pass revisits only deferred candidates in that same order when relaxation
is enabled. Record soft_quota_relaxed; never relax hard limits. There is no minimum
diversity backfill from outside retrieval, new relevance score, embedding or model.
allowed_versions provides explicit single-study scope; required_versions and
minimum_documents diagnose unmet coverage without fabricating evidence. A caller
must declare a narrow study scope when the question requires one. Unjudged returned
sources remain candidates; ranking and diversity do not establish task relevance.

Suppress exact verified text hashes and intersecting paragraph intervals only
within the same document version. Independent studies with identical wording
remain distinct. This conservative overlap rule can omit partly novel material;
the exclusion is recorded and available for independent review. Sections are exact
version/section identities, with a sectionless bucket per version.

## Hard limits and context

Frozen defaults: 20 fused candidates, at most 50 per lane, 5 selected passages,
3 documents, 5 passages per document/section, 24,000 total source UTF-8 bytes,
16,000 bytes per document, 6 context paragraphs and 8,000 expansion bytes.
Soft preferences are one passage per document and section, with explicit fallback.
Full passages and whole paragraphs fit or are omitted; no sentence/numeric result
is clipped. These are exact text-byte budgets, excluding JSON framing, with no
token-count or complete-numerical-interpretation claim.

Context opens body paragraphs in the anchor's exact section or up to one ancestor
section, within one ordinal sibling distance and a ten-ordinal local document
span. All controls are bounded typed policy fields. Never traverse front/back
regions, versions, artifacts, articles, or unrelated sections. Missing structure,
boundaries and exhausted context budgets are explicit diagnostics. Already selected
paragraphs and previously opened paragraphs are not reopened. Ancestor section keys,
version membership, paragraph keys, text hashes and whole-paragraph ranges are
independently checked. A context reference uses half-open Python character offsets
into persisted Paragraph.text; it has no retrieval rank/score or EvidenceMember.

Use one REPEATABLE READ READ ONLY snapshot, a 5-second statement timeout and a
30-second selection completion deadline, checked between canonical verifications
and before returning. A running SQL statement may finish up to its statement
timeout after the deadline; network retrieval has the existing bounded provider
timeout. This is a practical fail-closed completion bound, not hard cancellation.
Before complete-inventory reads, scalar SQL preflight reserves a conservative
fetched-row upper bound for projection construction, both canonical verification
passes and scalar results. The remaining part of the default 2,000-row budget is
shared by ancestor and paragraph context reads, each with explicit SQL limits.
Individual source text rows are capped at 100,000 UTF-8 bytes. Oversized context
paragraphs are not fetched and appear as missing context; source inventories over
budget reject the operation. SQL aggregate work is bounded by statement timeout.

## Evidence and authority

Canonical members come only from canonical_member, including SQL evidence_source,
independent Python source_fingerprint and complete ordered span inventory. Rebuild
and validate the entire EvidenceSet before returning. Canonical tables already
reject concurrent UPDATE/DELETE; the snapshot preserves a coherent inventory.
No persistence extension, fabricated Claims/Links, or new migration is needed.

Empty selection is evidence_set=null with no_admissible_evidence. Nonempty sets
retain explicit unmet document/version/context requirements and conservative
INSUFFICIENT or UNASSESSED sufficiency. Declared ABSTAIN/CORRECT_PREMISE requirements
remain unmet scientific-response obligations, never erased by retrieval. All
scientific sufficiency stays UNASSESSED. Public traces contain bounded reason
codes, IDs, hashes, coordinates and counts, with no source text. EvidenceSet does
retain the research question as required by RES-145; deployment-specific question
privacy remains the caller's responsibility. RES-147 owns synthesis; broader
citation rendering and scientific adjudication remain outside this change.
