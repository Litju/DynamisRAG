# RES-638 final scientific and engineering qualification

**Code under test:** `502d4b5` for the scientific gate, citation mapping and signed sessions;
`57b2f2e` for the dense projection. Documentation-only commits after those revisions do not
change runtime code. **Generated:** 2026-11-14. **Spend:** $0. No hosted deployment, paid
inference, Oracle/Railway resource or additional local-model campaign was created.

This record separates what was **executed** from what remains **unqualified**. The deterministic
model is a fixed test adapter, not a scientific language model. The answerability gate is a
calibrated lexical/structural heuristic, never semantic entailment or expert review.

## 1. Scientific relevance and abstention

The original licensed-corpus failure is preserved unedited in
[original-kepler-failure.json](res638-scientific-gate/original-kepler-failure.json): the question
“What is the orbital period of the exoplanet Kepler-22b?” returned four source-verified but
unrelated quotations with status `completed`.

The repair adds `structured-evidence-gate-v1` before synthesis:
[`answerability.py`](../../src/dynamisrag/agent/answerability.py). It separates lexical
similarity, canonical authenticity, query relevance (entities and weighted facets) and claim
support. It rejects retrieval candidates that do not contain the question's named entities or
enough query facets, then applies a second gate to verified quotations. Contradictory directions
on a shared outcome abstain; a comparison needs at least two structurally compared studies.

The gate is calibrated against a disjoint author-constructed calibration/held-out fixture pair
(`tests/fixtures/answerability/`). Thresholds are asserted from the calibration interval, not
from question strings. There is no question-specific blocklist; the former
`_OUT_OF_CORPUS_TOPICS` table was removed.

### Licensed three-paper corpus results

Command:
`python -m scripts.res638_scientific_gate_eval --output docs/evidence/res638-scientific-gate/receipt.json`

Receipt: [receipt.json](res638-scientific-gate/receipt.json) — 9/9 cases passed, 3 papers,
75 passages, deterministic inference, zero inference spend.

| Case | Result | Gate code |
| --- | --- | --- |
| Achilles tendon stiffness (supported) | completed, 4 citations | `evidence_relevant_and_answerable` |
| Knee flexion / muscle activity (paraphrase) | completed, 4 citations | `evidence_relevant_and_answerable` |
| Study designs and populations (comparison) | completed, 4 citations | `comparison_structurally_answerable` |
| Multi-turn “How many weeks did those participants complete?” | completed, 1 citation from the same study as turn 1 | `evidence_relevant_and_answerable` |
| Kepler-22b orbital period | abstained, 0 citations | `unsupported_entity` |
| Proxima Centauri b orbital radius | abstained, 0 citations | `unsupported_entity` |
| 100 m sprint improvement (close-domain) | abstained, 0 citations | `insufficient_query_coverage` |
| Sourdough hydration ratio (off-domain) | abstained, 0 citations | `insufficient_query_coverage` |
| Unknown-entity jump-height study | abstained, 0 citations | `unsupported_entity` |

The browser scenario “document selection is immutable and unsupported questions abstain” also
passes against this stack, including no `Inspect source` button after abstention.

**Limits.** A pass means the asked-about entities and facets are present and the quotation
survives the existing exact/numeric/lexical checks. It does not prove entailment, study validity
or clinical applicability. The deterministic adapter cannot qualify a live scientific model.

## 2. Citation highlighting

`text.indexOf(quote)` was removed. The backend now maps the verified `span_key` and
paragraph-relative `start_char`/`end_char` to the reconstructed passage by replaying the same
source reconstruction used for canonical verification
([`inspection.py`](../../src/dynamisrag/evidence/inspection.py)). It validates paragraph/span
identity, span membership, exact text equality and the final passage offset. A missing span,
wrong offset or mismatched text returns `highlight: null` with a reason; no other occurrence is
silently chosen.

Regression tests cover repeated quotations, shifted offsets, missing spans, out-of-span quotes,
untrimmed quotes and multi-span passages (`tests/unit/test_evidence_inspection.py`,
`apps/workspace/tests/highlight.test.ts`). The synthetic and licensed browser runs click a real
citation and assert `<mark>` is visible; an invalid coordinate was verified live to return
`span_coordinate_mismatch` and no highlight.

## 3. Frontend authentication

The synchronous Basic path blocked the Node request thread: 32 concurrent authentications took
**1.23 s** and delayed the event loop by **1.245 s** on the benchmark host. A short-lived
HttpOnly HMAC session (15 minutes) replaces scrypt after the first exchange: 32 concurrent
session verifications took **1.3 ms** and delayed the loop by **13 ms** including timer overhead.

Benchmark receipt: [benchmark.json](res638-auth/benchmark.json). Scope: one local Node 24
process; hosted instances are separate bounds.

Controls preserved or added:

- Sessions are signed with a domain-separated key, bound to the account's current hash, so
  changing the password hash invalidates existing sessions immediately.
- Failed Basic attempts are bounded at 5 per user and 20 per client address per 60 s per
  instance; the sixth attempt returns 429 with `Retry-After` before another scrypt evaluation.
- Unsafe cookie-authenticated requests require a matching Origin or
  `Sec-Fetch-Site: same-origin`; missing cross-site evidence is 403.
- Logout clears the cookie. User ownership remains server-derived from the verified session or
  signed Python assertion; no browser-supplied owner ID is trusted.
- The previously repaired Eve session takeover, bodyless cancellation, session streaming and
  worker-recovery paths pass again through Next/Eve/Workerd/Python.

## 4. Dense retrieval qualification

**Status: `QUALIFIED` for the pinned local CPU projection.** Receipt:
[receipt.json](res638-dense-qualification/receipt.json).

A **new isolated Qdrant 1.20.0 instance** was started on `127.0.0.1:16335` with a new volume. The
target was refused unless empty; no retained Qdrant collection, alias, snapshot or receipt was
read. The licensed 75-passage corpus was read from the branch's fresh canonical database.

| Check | Observed |
| --- | --- |
| Model | `Qwen/Qwen3-Embedding-0.6B` |
| Revision | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |
| Runtime | TEI 1.9.4, CPU/candle-MKL, float32, last-token, image `sha256:2538ea1c…` |
| Document fingerprint | `8438d37d8e8c8f975f62cbc6f6c644abc27304b18b7a330d9157a70f21c9c1f7` |
| Query fingerprint | `94245fa64ff582ab1428c7aea38387e9091a23e538c96d7fbf7b48e1755da8fd` |
| Embedding manifest | `46f96db3a02cf61374003475e3f006258ecbd790a4a78bd858b305c25f5104ff` |
| Projection | `c805183fa694986ea127fb62c9541b7acdcaa1c4af2fa8cfae5ee099fa9e0eb4` |
| Points | 75 exact canonical keys; 512 dimensions; norm 0.99999996–1.00000004 |
| Positive recall | Achilles, knee-flexion and comparison cases recalled the expected document |

The physical collection name is deterministic from the canonical corpus and embedding manifest;
it is the same name a rebuilt projection of these artifacts would have. The instance was empty
before creation, so this is not reused retained data. Off-domain queries still return nearest
neighbours by construction; the answerability gate, not Qdrant, performs abstention.

**Not qualified:** hosted Qdrant/TEI, multi-tenant load, TLS/access control, backup/restore and
paid inference. The local CPU projection is evidence of compatibility, not capacity.

## 5. Verification at this revision

| Check | Result |
| --- | --- |
| Python unit suite | **2,837 passed**, 309 deselected, 121 s |
| PostgreSQL/OpenSearch integration | **306 passed**, 2,836 deselected, 349 s |
| Ruff lint/format | Passed on `src`, `tests`, `scripts` |
| Strict Pyright | 0 errors, 0 warnings |
| TypeScript tests | **16 passed** (auth, sessions, SSE, dispatch, highlighting) |
| Next route generation + TypeScript | Passed |
| Synthetic browser E2E | **6 passed**, 52.8 s through Next/Eve/Workerd/Python |
| Licensed-corpus browser E2E | **4 passed**, 27.4 s (question/isolation, mobile, comparison, Kepler abstention) |
| Exact-head GitHub Actions | Tracked on draft PR #35; local results are not a substitute |

The integration rerun used the documented 1 s lease / 3 s expiry harness for the formerly flaky
attempt-budget contract. One earlier run failed that single assertion under full-suite load; the
contract passed alone and the adjusted harness passed all 306.

## 6. Independent final-head review

Claude Opus 5.5 independently reviewed `00ea816` without editing or deploying. It found two
blocking scientific-gate defects and no blocking citation, authentication or worker-recovery
defects:

| Finding | Repair at `502d4b5` | Regression evidence |
| --- | --- | --- |
| P1: a multi-turn follow-up was gated on the raw question while retrieval used the resolved question; the receipt accepted a wrong-study citation | Gate follow-ups on the current proposition, use prior context only when the current turn has no content, scope follow-up evidence to the referenced study, and require the follow-up citation to remain in turn 1's study | New graph regression; the licensed receipt now cites the 12-week depth-jumping passage and asserts the same document version as turn 1 |
| P2: an irrelevant document could satisfy the comparison "two studies" requirement | Accept only admissible comparison candidates when salient facets exist; require two distinct documents among admissible claims; filter verified citations by accepted claim index | Mixed-relevance comparison fixture and claim-index test |

The reviewer's residual risks (first-occurrence verifier selection, Basic credentials surviving
sign-out in the browser, thin calibration, corpus-fitted lexicon, per-instance login lockout)
are recorded below as limits; they are not counted as passes or as scientific validation.

## 7. Remaining blockers (not scientific-gate failures)

- Hosting/provider authorization: OpenCode Go still requires written provider approval;
  `DYNAMISRAG_AGENT_OPENCODE_GO_USAGE_APPROVED=false`.
- No authorized live-model E2E or expert scientific review; deterministic passes are plumbing.
- No dedicated persistent external Python host at $0; no hosted preview, public domain or launch.
- Hosted dense retrieval capacity, TLS/Access and backup restore remain unqualified.
- PR #35 stays draft and unmerged pending owner review.
