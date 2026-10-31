# RES-146 qualification

Disposition: **BLOCKED_BY_EVIDENCE** for live query execution. The deterministic
implementation and offline source qualification are complete; the pinned query
embedding runtime returned `embedding_unavailable`. No synthetic vector was used
to claim a live scientific retrieval execution. Hosted exact-head status is
recorded on the draft PR and RES-146 after publication.

Starting main was independently fetched and matched
`ef5ed691b82af93489badbbb43dea439d10d904a`. Work uses
`feat/res-146-evidence-selection`. Implementation commits are `834a6b3` and
`2b835f0`; the qualification commit adds the reproducible script, design decision
and receipt. RES-146 was moved from Backlog to In Progress. It is not Done.

## Reproducible checks

```powershell
uv lock --check
uv run ruff check --exclude reports .
uv run ruff format --check --exclude reports .
uv run pyright
uv run pyright scripts/res146_evidence_selection.py
uv run pytest -q --basetemp .tmp/res146-unit-fresh
uv run pytest -m integration tests/integration/test_evidence_selection.py `
    tests/integration/test_evidence_graph.py -q --basetemp .tmp/res146-focused-fresh
uv run python -m scripts.res146_evidence_selection
```

The root Ruff exclusion preserves the pre-existing untracked reports archive,
as in RES-145. No repository lint or type rule was weakened. Dependencies and
lockfile are unchanged. The pre-existing operator notebook is untouched.

Full integration safely uses an ephemeral database, including old migration
tests that intentionally downgrade their target:

```powershell
@'
import subprocess, sys
from scripts.res145_evidence_demo import scratch_database
with scratch_database():
    subprocess.run([sys.executable, '-m', 'pytest', '-m', 'integration', '-q',
                    '--basetemp', '.tmp/res146-integration-fresh'], check=True)
'@ | uv run python -
```

Local lint, format, strict Pyright and lock validation pass. Full unit suite:
**2,503 passed**, including 45 focused selection/context cases. Full isolated
integration suite after the final savepoint repair: **274 passed**, including
the real SQL-rejection/transaction-recovery regression. RES-145, canonical provenance, scientific retrieval, scientific answer and
DynamisBench tests were retained without weakening assertions. The final complete
suite result and exact-head hosted URL are recorded in the PR/Linear receipt.

Tests reject identity/configuration/projection drift, lane-name confusion, duplicate
or reordered hits, altered raw-score order, forged RRF, model-copy validator bypass,
unknown sources, changed versions/text/source metadata/inventories and forged
verified flags. Controlled selection tests cover same-version overlap/exact-text
redundancy, identical words from independent studies, section/document quotas,
single-study scope, soft fallback, exact Unicode byte boundaries and deterministic
replay. Context tests cover missing sections/siblings/parents, cross-version parent
identity, text/key corruption, full numerical paragraphs, parent/sibling origins,
front/back boundaries, overlap and row/byte limits. Existing append-only triggers
reject canonical mutation; pre/post inventories are checked after rejected writes.
Source prose never creates Claims or reviewed authority. Empty selection and
declared abstention/correction obligations remain explicit.

## Actual three-paper observations

The script verifies the original DynamisBench specification and raw captured
responses, copies all twelve canonical tables into a new migrated database, then
performs read-only selection and independently reconstructs every selected member
using the RES-145 Python and SQL verifiers. The original reference database was
not migrated. All source passage bytes, UUIDs, document/version/paragraph/span
identities, artifact metadata, all 75 vectors/payloads and aliases, original
reference files, RES-143 files and RES-144 bank/capture artifacts match pre/post.
The copied database also remains unchanged throughout selection. Only identifiers,
counts, hashes and public question text enter the committed receipt.

See [machine-readable receipt](evidence/res146-selection-receipt.json). The Methods
demonstration is existing question `db00-02`: “What population, load and instruments
were used in the knee-angle experiment?” It produces five canonical members and
five context references. The captured response retains its exact BM25, dense and
fused rankings, linked through its verified original response digest. Every
candidate disposition is in the receipt; context paragraphs have no invented
retrieval scores. Repeating each frozen execution produces exactly the same
EvidenceSet, trace and selection key without additional model calls.

The 52 diagnostic observations compare four variants across 13 provisional
questions: a first-five fused-rank slice, verified deterministic selection without
diversity preferences, selection with diversity preferences, and selection with
context. The first variant is a diagnostic rank-only comparator, not a replacement
of the existing extractive answer selector. The verified variants enforce budgets
and suppress redundancy; context opens whole canonical paragraphs separately.

All modes retain 65 passages across 13 questions. Mean document counts are
1.462 for rank-order/deterministic and 2.769 for diversity/context; mean section
counts are 4.385 and 4.923 respectively. No duplicate pair or provenance failure
was observed in this sample. The context variant opens 33 paragraphs. Proposed
positive-judgment retention is 14 passages for rank-order/deterministic and 13
for diversity/context: increased diversity does **not** demonstrate increased
relevance. These unsigned judgments are not independent gold, and selection was
not tuned against them. No statistical improvement or scientific adequacy claim
is made. All 52 complete frozen selections replay identically.

The receipt contains per-execution bytes, utilization, duplicate pairs/rate,
candidate retention, proposed coverage, source/section diversity, context bytes
and measured local selection time. Times cover canonical selection/verification
and expansion on the scratch PostgreSQL instance, under the actual local load;
they are not live retrieval timings or a controlled performance experiment.
Rank slicing is not latency-qualified as a provenance-verifying selector and its
selection_ms is null. Context overhead can be inspected against the same question's
diversity run, without claiming causal timing improvements. Original retrieval
timings remain captured evidence and are never presented as a new live run.

Scientific authority stays provisional. A valid canonical member does not establish
causality, significance, effect generalization, entailment, relevance, sufficiency
or expert approval. Selection is an opt-in evidence-availability boundary; existing
generation behavior and its policy digest are unchanged. No Claim/EvidenceLink,
standalone EvidenceSet persistence, new migration or broader synthesis milestone
was fabricated to complete this work.
