# DynamisBench v0 — implementation and qualification

RES-144 adds a scientific QA evaluation layer beside the frozen RES-138 embedding
benchmark. It changes no serving policy, corpus ingestion, corpus vectors, model
selection or RES-143 overlay. The candidate bank is **AWAITING_EXPERT_REVIEW**.
Mechanical CI is active through the existing Ruff, strict Pyright, unit and
integration jobs. Performance acceptance is unconditionally disabled in v0.

## Evidence path and measurement boundaries

| Stage | Existing authority | Benchmark evidence / evaluation |
|---|---|---|
| Scientific question | candidate-bank.json, ordered question ID/query | Canonical specification digest, category, answerability, constraints |
| Query embedding | embedding/scientific_cpu.py; scientific service | Pinned model revision, CPU deployment, query/document config fingerprints; fresh query inference only |
| BM25 / dense | ScientificRetrievalService, QdrantRetrieval | Complete returned lane ranks, scores, passage keys and provenance hashes; Recall@5/@10, nDCG@10, MRR@10 |
| RRF fusion | retrieval.reciprocal_rank_fusion | k=60, window=50, limit=20; fused keys/scores independently reconstructed from both lanes |
| Canonical provenance | PostgreSQL and original JATS; load_scientific_projection | All 75 mappings, source hashes, passage content hashes, anchors; relevant-source coverage@10 |
| Evidence selection | generation.scientific.select_evidence | ScientificAnswer evidence, excerpts and exclusions; retrieval relevance does not imply selector success |
| Generation | ScientificAnswerService /answer/scientific | Separate complete answer inventory; status, model/policy identity and raw answer digest |
| Citation validation | verify_evidence, validate_claims | Independently compare provenance, quote offsets, context membership, evidence membership and rendered answer |
| Scientific evaluation | Independent expert labels | Citation support, entailment, numbers/units/uncertainty, population, contradictions, abstention and premise correction |

The existing structural verifier explicitly cannot establish scientific validity.
No LLM judge is used. Expert labels bind the exact specification, question and full
answer digest. A digest is not a signature. The Python review API requires an
independently trusted signature-verification callback and rejects unsigned,
unbound or unverifiable reviews. The CLI cannot approve reviews. An always-true
callback is a test fixture, never a permissible reviewer trust policy.

`CANDIDATE` and `AWAITING_EXPERT_REVIEW` are dataset states supported here.
Authenticated answer reviews can be `EXPERT_APPROVED`. The vocabulary also reserves
`QUALIFIED_FOR_REGRESSION`, but v0 has no automatic promotion or threshold-enabling
path. Answer review is not approval of the question bank or qualification of a gate.

## Immutable baseline

Only PMC10780924 (25 passages), PMC10680285 (18), PMC5263151 (32) are admitted.
Original manifest SHA-256:
`46f96db3a02cf61374003475e3f006258ecbd790a4a78bd858b305c25f5104ff`.
Original receipt SHA-256:
`8ede85766f656acccf7fd44b9c3fde627bd18289b74fe375dbf895d02ff4d240`.
Paper receipt SHA-256:
`1503e43db080e20db050a4f804597dedda0136c0d7f055b0fd21662d6683eadf`.
Projection SHA-256:
`c805183fa694986ea127fb62c9541b7acdcaa1c4af2fa8cfae5ee099fa9e0eb4`.
The bank contains the complete 75-key inventory and source-to-version mappings.
Its corpus identity is
`833117eaa0ae5cf591d838b989a3475c3e079defa07a43c1b4db3cafe251f3ed`.

`verify_baseline` invokes the existing immutable-manifest loader, reconstructs
canonical passage citations, reads each retained original XML file, verifies its
original byte hash, and reparses all paragraphs against PostgreSQL and its content
fingerprint. `verify_spec_evidence` additionally checks candidate span text hashes.
These operations are read-only. They require the private retained source store;
fresh upstream downloads cannot substitute for missing original bytes.

The alias is `dynamisrag-real-science-v1`; collection/model/config identities are
recorded in the bank. The ten RES-143 acquisitions are excluded. The original
three-paper/75-vector artifacts are untouched. No corpus embedding inference is
performed by this framework. Starting the existing pinned TEI container permits
only the existing approved query embedding configuration.

## Scoring and comparisons

Grades are proposed 0 (judged irrelevant), 1 (context), 2 (useful partial support),
3 (direct evidence). Binary recall and MRR use grades >0; nDCG uses gain 2^grade−1
and log2(rank+1) discounts. IDCG includes all positive judgments, truncated at ten.
Source coverage counts sources with a **relevant retrieved passage**, not any
passage from that article. Null metrics are excluded from means, with denominators
retained. Questions without positive judgments have null retrieval metrics, not
zero. False-premise corrective evidence can have positive relevance.

Unjudged passages contribute no gain to this pooled diagnostic calculation but
are explicitly counted as unjudged; they are not confirmed negatives. The pool is
incomplete and may favor the system used to discover evidence. Full-pool judging,
hard-negative review and independent pooling are prerequisites for qualification.

Reports preserve all question × lane rows and category/overall aggregates.
Comparisons recompute both reports from runs against the same complete
specification/corpus; produce each paired metric delta, wins/losses/ties,
regressions and category means. No significance test or confidence interval is
claimed for this 13-item convenience sample. `compare_answers` separately records
abstention transitions, unsupported acceptance and unnecessary refusal relative
to proposed answerability; semantic outcomes remain pending expert review.

Every input boundary revalidates even Pydantic copies. External specification/run
digest pins prevent coordinated run/summary edits from passing reconstruction.
Hashes provide integrity relative to a trusted pin, not proof that an operator
really executed a query. Preserve private raw responses, independent operator
receipts and Git/CI evidence for audit. A public compact run's response hash links
to a full private API response; it cannot replace access to that evidence.

## Operator commands

Run from the repository root with locked dependencies and existing `.env`.

```powershell
$specPin = '1df5bacb6cfc786f29b8f5055c33c8e6e3a7be59399744fd55e96ce9d7a755f8'
uv run python -m scripts.dynamisbench validate --expected-spec $specPin
# Requires original JATS + PostgreSQL + existing Qdrant collection + pinned TEI.
# Requires committed code; performs no ingestion, corpus embedding, or publication.
uv run python -m scripts.dynamisbench capture --expected-spec $specPin
uv run python -m scripts.dynamisbench score --expected-spec $specPin --run <run.json> --expected-run <run-content-digest>
# Add --responses .dynamisrag/dynamisbench-v0/private-responses to reconstruct
# every compact row from its original private service response.
uv run python -m scripts.dynamisbench compare --expected-spec $specPin --run <baseline.json> --expected-run <baseline-content-digest> --candidate <candidate.json> --expected-candidate <candidate-content-digest>
# Private ordered JSON object: question ID -> actual /answer/scientific response.
uv run python -m scripts.dynamisbench answers --expected-spec $specPin --run <private-answers.json> --expected-run <canonical-inventory-digest>
```

Canonical digests use UTF-8, sorted object keys, compact separators, no NaN and
preserved array order. Artifact filenames are content digests. Missing, duplicate,
unknown or changed query inventories fail closed. A failed live capture writes no
completed run; partial private responses are not evaluation results. Timing fields
come from the real service; they are single-request observations, not percentiles.

Output defaults to ignored `.dynamisrag/dynamisbench-v0`. Full raw API responses
contain source text and stay in `private-responses/`. New public artifacts contain
only identifiers, hashes, span offsets, metadata and authored candidate questions
and interpretations. Do not publish private answer inventories or database rows.

## Validation and qualification gaps

Run Ruff, formatter, strict Pyright and the infrastructure-free suite normally.
Run `tests/integration/test_dynamisbench.py` and
`tests/integration/test_scientific_citations.py` with `-m integration` against the
existing PostgreSQL database; these use rolled-back synthetic records. Full
integration/migration CI uses an isolated database, never the preserved corpus.

Mutation tests cover malformed specifications, grades and spans; duplicate/missing
questions; wrong sources and passages; reordered/duplicated rankings; missing
lanes and fused hits; altered RRF scores; query/model/projection changes; citation
and numerical substitutions; forged reviews; incomplete candidate runs; altered
relevance inventories and coordinated report edits.

Pending scientific work: two independent domain reviewers, adjudication and signed
gold-set approval; narrowed claim spans; validated answer labels; full-pool bias
assessment; additional independent studies for contradiction and efficacy synthesis;
preregistered sampling/statistical protocol; and frozen acceptance thresholds
chosen before candidate outcomes. Unit-tested evaluator, executable benchmark,
approved gold set and qualified regression gate are four different milestones.
Technical validation alone completes only the first two.

## Executed reference run

The committed compact [baseline run](evidence/dynamisbench-v0/baseline-run.json)
was captured at implementation commit
`e9f29b87a8ae7c608e3fe74e30734986112fe121`, at
2026-10-31T02:50:09Z (2026-10-30 in Buenos Aires). All 13 questions executed through
the actual ScientificRetrievalService with fresh approved CPU query embeddings.
All 39 question/lane rows reconstructed from retained private raw responses.
The [qualification receipt](evidence/dynamisbench-v0/qualification-receipt.json)
records original file hashes and verification of all source reparses, canonical
passages and original Qdrant vectors after execution. Corpus embeddings were reused.

Run digest:
`7527818b76f65b98d5707c0301a4923369552b3bf9931cfb45d8bcc6f5a3bf23`.
[Metric report](evidence/dynamisbench-v0/baseline-metrics.json) digest:
`9b43f5ff02389d43e5b4ebfd17c8a20e36be8d46d0b67c35239d1f2209ffc067`.
Tests independently reconstruct the public report without inference. Eleven
questions contribute retrieval metrics; the two unanswerable items are excluded.
Fused retrieval missed the proposed corrective evidence for db00-09 at ten.
This visible failure is retained. There is no measured candidate system or
statistically reliable winner. All scores depend on unapproved proposed judgments.

Live answer generation was checked separately using the existing Ollama identity
client and returned `generation_unavailable`. No live answer scores are invented.
The answer evaluator is exercised with synthetic contracts; evaluating real
answers requires the pinned generation runtime and independent expert labels.
This optional runtime gap does not change the captured retrieval evidence or
promote the question bank to gold.

Local validation: 2,428 infrastructure-free tests passed before adding the public
baseline replay test; 168 focused scientific tests and two PostgreSQL integration
tests passed. The expanded benchmark suite includes the captured-run replay test.
Use a fresh `--basetemp .tmp/<run-name>` on this machine because its default pytest
temporary directory is inaccessible. Ruff/format pass with the pre-existing
untracked `reports/` excluded; strict Pyright passes. Unrelated operator notebooks
and reports were preserved. Exact PR-head CI evidence is recorded in draft
[PR #24](https://github.com/Litju/DynamisRAG/pull/24) and RES-144 after CI completes;
the preserved populated database is never used for migration/reset CI.
