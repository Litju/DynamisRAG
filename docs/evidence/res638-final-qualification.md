# RES-638 final scientific and engineering qualification

**Code under test:** `b3a46f0` for the scientific gate, citation mapping, signed sessions and
pronoun/ambiguity repairs. `57b2f2e` introduced the original dense-qualification operator; the
projection was re-created from scratch at `b3a46f0` with identical fingerprints. Documentation-only
commits after `b3a46f0` do not change runtime code. **Generated:** 2026-11-15. **Spend:** $0. No
hosted deployment, paid inference, Oracle/Railway resource or additional local-model campaign was
created.

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
on a shared outcome abstain; a comparison needs at least two structurally compared studies. The
inventory/metadata route no longer bypasses relevance: a named entity or topical qualifier must
still be present in canonical evidence. A pronoun/demonstrative follow-up is scoped to the prior
study, and an unresolved reference abstains rather than switching studies. A generic measurement
question ("What is the sample size?") abstains when it spans several studies and is answerable
only when a thread scope identifies one document. A value question must have at least one
admissible quotation containing a number; a qualitative sentence that merely says "small sample
size" is not accepted as the requested value.

The gate is calibrated against a disjoint author-constructed calibration/held-out fixture pair
(`tests/fixtures/answerability/`). Thresholds are asserted from the calibration interval, not
from question strings. There is no question-specific blocklist; the former
`_OUT_OF_CORPUS_TOPICS` table was removed.

### Licensed three-paper corpus results

Command:
`python -m scripts.res638_scientific_gate_eval --output docs/evidence/res638-scientific-gate/receipt.json`

Receipt: [receipt.json](res638-scientific-gate/receipt.json) — 14/14 cases passed, 3 papers,
75 passages, deterministic inference, zero inference spend.

| Case | Result | Gate code |
| --- | --- | --- |
| Achilles tendon stiffness (supported) | completed, 3 citations | `evidence_relevant_and_answerable` |
| Knee flexion / muscle activity (paraphrase) | completed, 4 citations | `evidence_relevant_and_answerable` |
| Study designs and populations (comparison) | completed, 4 citations | `comparison_structurally_answerable` |
| Multi-turn “How many weeks did those participants complete?” | completed, 1 citation from the same study as turn 1 | `evidence_relevant_and_answerable` |
| Multi-turn “How many participants were in its training group?” | completed, 4 citations from the same study as turn 1 | `evidence_relevant_and_answerable` |
| Follow-up “What was its sample size?” (only qualitative quotes) | abstained, 0 citations | `insufficient_numeric_evidence` |
| Follow-up pronoun after an abstained turn | abstained, 0 citations | `unresolved_coreference_reference` |
| Unscoped “What is the sample size?” | abstained, 0 citations | `ambiguous_question` |
| Kepler-22b orbital period | abstained, 0 citations | `unsupported_entity` |
| Proxima Centauri b orbital radius | abstained, 0 citations | `insufficient_canonical_evidence_for_query` (no hits) |
| 100 m sprint improvement (close-domain) | abstained, 0 citations | `insufficient_query_coverage` |
| Sourdough hydration ratio (off-domain) | abstained, 0 citations | `insufficient_query_coverage` |
| Unknown-entity jump-height study | abstained, 0 citations | `unsupported_entity` |
| Metadata “DOI of the Kepler-22b paper” | abstained, 0 citations | `unsupported_entity` |

The browser scenario “document selection is immutable and unsupported questions abstain” also
passes against this stack, including no `Inspect source` button after abstention.

**Limits.** A pass means the asked-about entities and facets are present and the quotation
survives the existing exact/numeric/lexical checks. It does not prove entailment, study validity
or clinical applicability. The deterministic adapter cannot qualify a live scientific model.
Value detection is digit-based, so a spelled-out number is treated as unsupported rather than
guessed.

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

The synchronous Basic path blocked the Node request thread: in the final benchmark, 32 concurrent
authentications took **1.51 s** and delayed the event loop by **1.52 s** on the benchmark host
(earlier runs ranged from 1.20 s to 1.51 s). A short-lived HttpOnly HMAC session (15 minutes)
replaces scrypt after the first exchange: 32 concurrent session verifications took **1.9 ms**
(earlier runs: 1.3–1.9 ms) and delayed the loop by less than 1 ms excluding timer overhead.

Benchmark receipt: [benchmark.json](res638-auth/benchmark.json). Scope: one local Node 24
process; hosted instances are separate bounds. The exact numbers vary by host load; the order of
magnitude and the event-loop blocking are stable.

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

**Limits.** The browser's HTTP Basic credential cache is outside server control: clearing the
session cookie does not necessarily clear credentials the browser already cached for the origin.
A shared browser must use a private window and close it; a form-based login would be required to
remove this residual. Failed-attempt counters are per process, so a multi-instance deployment
multiplies the bound. Neither limit changes the server-side session, ownership or Origin checks.

## 4. Dense retrieval qualification

**Status: `QUALIFIED` for the pinned local CPU projection.** Receipt:
[receipt.json](res638-dense-qualification/receipt.json).

A **new isolated Qdrant 1.20.0 instance** was started on `127.0.0.1:16336` with an ephemeral
volume at final head `b3a46f0`. The target was refused unless empty; no retained Qdrant
collection, alias, snapshot or receipt was read. The licensed 75-passage corpus was read from the
branch's fresh canonical database. A second independent run on a separate fresh instance produced
the same collection identity, fingerprints, 75 points and 512-dimensional normalized vectors as
the first run, so the projection is reproducible from the pinned artifacts.

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
| Python unit suite | **2,856 passed**, 309 deselected, 120 s |
| PostgreSQL/OpenSearch integration | **306 passed**, 2,848 deselected, 465 s |
| Ruff lint/format | Passed on `src`, `tests`, `scripts` |
| Strict Pyright | 0 errors, 0 warnings |
| TypeScript tests | **16 passed** (auth, sessions, SSE, dispatch, highlighting) |
| Next route generation + TypeScript + production build | Passed |
| Edge-gateway tests + Wrangler dry-run bundle | **6 passed**; bundle passed |
| Synthetic browser E2E | **6 passed**, 35.9 s through Next/Eve/Workerd/Python |
| Licensed-corpus browser E2E | **5 passed**, 24.0 s (question/isolation, mobile, comparison, Kepler abstention, cancellation) |
| Exact-head GitHub Actions | The preceding code head passed all 6 checks; the final-head run status is tracked on PR #35 |

The licensed browser suite intentionally excludes the synthetic-only worker-failure fault-marker
scenario; that path is covered by the six synthetic scenarios. One earlier licensed run hit a
transient Eve "session is no longer active" after a dev-server restart and a cross-paper test
timeout; a clean rerun passed 5/5. The exact-head CI run is the authoritative result.

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

### Final-head adversarial review at `b3a46f0`

A further independent-context review of the repaired gate and graph (separate review session, same
repository, no edits) found ten reachable relevance or streaming defects. They are repaired in
`b3a46f0` with new unit and licensed-corpus regressions:

| Finding | Repair | Regression evidence |
| --- | --- | --- |
| P1: the inventory/metadata shortcut returned `ANSWERABLE` for an unsupported named entity, e.g. “What is the DOI of the Kepler-22b paper?” | The structural route now requires every named entity and every topical qualifier in canonical evidence; metadata-only facets (`doi`, `section title`, …) are excluded from passage coverage | New unit positives/negatives; the 14-case licensed receipt abstains with `unsupported_entity` |
| P1: a possessive follow-up (“What was its sample size?”) ignored the prior study and answered from a different paper | Word-boundary reference detection for possessives/demonstratives and definite study references; the follow-up is scoped to the latest cited document. If a prior turn abstained, the reference has no antecedent and abstains `unresolved_coreference_reference` | New graph regressions; the licensed receipt cites only the turn-1 document and shows the unresolved case abstaining |
| P1: active version keys were cumulative, so a later pronoun could still resolve to an older study | The active scope is replaced by the latest answer's cited versions; an abstained turn retains the last successful scope | New three-turn graph regression |
| P1: a claim passed because its document title contained the question facets while the quotation was a methods sentence | The title may satisfy study context; the quotation itself must supply a direction/effect, a value, or a non-title query term. A missing salient qualifier is a hard abstention | New title-leak regression with the real knee-flexion methods sentence |
| P1: a title-supplied entity short-circuited the finding-signal check, so a quote with no outcome passed | Entity support is necessary but not sufficient; the signal checks the quotation independently of the title | New title-only entity regression |
| P1: a possessive comparison follow-up (“How does its jump height compare…?”) took the comparison branch before coreference and could compare unrelated studies | A possessive/demonstrative comparison is scoped to the latest active study; the comparison gate then abstains unless two admissible documents are in that scope | New three-turn possessive-comparison regression |
| P2: an unscoped generic question (“What is the sample size?”) chose the top-ranked study | A factual question whose only salient terms are measurement nouns abstains `ambiguous_question` when candidates span several documents; a thread-scoped follow-up with one document remains answerable | Unit tests for both branches; licensed 14-case receipt |
| P2: a value question accepted a qualitative sentence (“a relatively small sample size”) as its answer | A factual value question now requires at least one admissible quotation containing a digit; otherwise it abstains `insufficient_numeric_evidence`. Contracted and “how large” forms are included | New unit tests and licensed numeric negative |
| P2: a contracted value question (“What’s its sample size?”) bypassed the numeric check | The value pattern covers contractions, “how large/small”, and size/count/proportion forms | New contracted-value regression |
| P2: the Python `stream.end` sentinel has no sequence number; the browser parser threw and retried a completed stream | The SSE adapter consumes the named terminal sentinel and does not treat it as a malformed research event | New TypeScript stream-termination test |

The review also re-ran the preserved Kepler failure and confirmed the original receipt is not
edited. Residual risks not counted as passes: the numeric check is digit-presence rather than
unit alignment (a year or p-value can satisfy it); a comparison containing a possessive may still
use all authorized documents and is labelled as a comparison; ABSTAIN audit payloads contain
accepted keys but no returned citations; and the Basic-credential cache and per-process login
counter limits are unchanged.

## 7. Remaining blockers (not scientific-gate failures)

- Hosting/provider authorization: OpenCode Go still requires written provider approval;
  `DYNAMISRAG_AGENT_OPENCODE_GO_USAGE_APPROVED=false`.
- No authorized live-model E2E or expert scientific review; deterministic passes are plumbing.
- No dedicated persistent external Python host at $0; no hosted preview, public domain or launch.
- Hosted dense retrieval capacity, TLS/Access and backup restore remain unqualified.
- PR #35 stays draft and unmerged pending owner review.
