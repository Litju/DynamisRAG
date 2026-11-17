# RES-638 real research workflow

Status: **REAL_RESEARCH_WORKFLOW_PARTIAL_PROVIDER_BLOCKED** (2026-11-17,
[live-v2 receipt](evidence/res638-research-workflow/live-v2/receipt.json)). Full acceptance did **not** pass.

## Live v2 (2026-11-17): the real model drives the research

On fresh, empty stores (`dynamisrag-res638-accept2`) and the exact acceptance question,
`gpt-6-luna` itself chose the tools: `list_papers`, a `discover_external_literature` request with
three concept-group searches (force–velocity / load–velocity profiling × linear position
transducer × resistance training × validity / reliability / applications), and four
aspect-focused `hybrid_search` calls. Europe PMC returned 41 records (38 works); 4 CC BY 4.0
JATS full texts were ingested (107 passages, collection
`qdrant-passages-v1-56c6358c…7fd8`, pinned TEI). 5 model requests, nominal USD 0.0024421.

**It abstained** (`canonical_citation_verification_failed`): both model claims added
interpretation that the unchanged verifier rejected, and 2 of the 4 ingested works were
off-topic (query-order ingestion). Fixes since: one bounded model repair round with the
verifier's reasons, a stricter paraphrase rule, topical ranking of discovered works, and
serialized agent-schema setup (attempt 2 crashed at startup with a deadlock, 0 requests).
None of these fixes is re-verified live.

**Budget.** The owner requires USD 0 new spending. Whether the 5 requests were inside the
existing Go allowance is **unverified** (console-only). Inference was stopped, the local approval
flags were reset to `false`, and no further live run is made until a zero incremental charge is
verified. **Authorization** was re-evaluated against the published terms: no written permission is
required and OpenCode never rejected anything; the owner approved local, owner-only use
([record](evidence/res638-research-workflow/opencode-go-authorization.json)).

What changed in code: the model may request a literature search (validated concept groups,
once per turn, rights gate unchanged); works already in the corpus are not recounted, so a
follow-up can add new studies; undefined acronyms ("LPT") are resolved from the conversation;
enumerated multi-aspect questions are judged collectively (core topic plus any one aspect per
passage, uncovered aspects reported) while a missing core topic still abstains; synthesis labels
each claim with its aspect and a proposed support label (study finding / author interpretation /
background citation) and lists gaps and tensions, all `MODEL_PROPOSED`.

The sections below describe the 2026-11-16 provider-blocked run (v1).

## 0. Accepted run (2026-11-16, code `a0e147d`)

| | Initial question | Follow-up |
| --- | --- | --- |
| Corpus before | 0 document versions, 0 Qdrant collections | 4 papers / 111 passages |
| Search requests | 3 (2 Europe PMC + OpenAlex 429) | 3 (2 Europe PMC + OpenAlex 429) |
| Records → unique works | 21 → 21 | 26 → 26 |
| Ingested (CC BY 4.0) | 4 papers, 111 passages | same 4 (idempotent; 111 vectors reused) |
| Rejected | 3 (scope 2, licence URI 1) | 5 (scope 2, repository rights 2, licence URI 1) |
| Retrieval | `qdrant_hybrid_rrf`, 6 candidates, 0 admitted | 6 candidates (PMC12608286), 0 admitted |
| Model calls / cost | 0 / USD 0 | 0 / USD 0 |
| Outcome | `abstained` `evidence_gate:insufficient_query_coverage` | same |

Ingested: DOI 10.3390/jfmk6020032 (PMC8103256, PMID 33808291); DOI 10.7717/peerj.21357
(PMC13380235, PMID 42472305); DOI 10.3390/s25216542 (PMC12608286, PMID 41228765); DOI
10.3390/sports14070263 (PMC13416992, PMID 42506808). Collection
`qdrant-passages-v1-444fe4cf90e2679dd09d465e2d92ef8371315712dffadea2e2091302d09abb90` (111 points,
green, alias `dynamisrag-research-workspace`). The browser test inspected a retrieved canonical
passage and its source spans.

## 1. Model and authorization

* Sole model: `gpt-6-luna` for every role, native Responses endpoint
  `https://opencode.ai/inference/go/openai/v1/responses`, `store=false`, a stable
  `x-opencode-session` per run, SDK retries off, per-role and per-run budgets unchanged
  (USD 0.20 / 250,000 tokens / one escalation). See [model routing](res637-model-routing.md).
* Verified without the key on 2026-11-16: the documented catalogue
  `https://opencode.ai/inference/go/v1/models` lists `gpt-6-luna` (`openai/gpt-6-luna`, chat,
  `tools.supported: true`, `api.responses`).
* Authorization (v1, corrected 2026-11-17): the v1 record treated written permission from
  OpenCode as required; the Terms require none. The block was the local flag, not OpenCode
  ([record v2](evidence/res638-research-workflow/opencode-go-authorization.json)).
* With approval false the service runs `ProviderBlockedRuntime`: no client, no key, every call
  refused before the network. Retrieval and verification still run; synthesis abstains with
  `provider_authorization_blocked`. No local or substitute model is used.

## 2. Flow

1. A thread **without a fixed document scope** asks a question. The existing LangGraph graph
   retrieves; the structured evidence gate judges relevance.
2. If the gate abstains (an empty or irrelevant corpus), `acquire_external_literature` runs
   **once**: a fixed-rule query ladder over Europe PMC (`OPEN_ACCESS:Y`) and OpenAlex. Only
   `www.ebi.ac.uk` and `api.openalex.org` over HTTPS are reachable; 1 s spacing per host, at most
   6 search and 8 full-text requests, bounded `Retry-After`, every request recorded with its
   SHA-256.
3. DOI / PMID / PMCID are linked across providers; a component that names two different values
   of one identifier is quarantined.
4. Rights: Europe PMC metadata must say `cc by`/`cc0`, open access and in Europe PMC before the
   full text is requested; the original JATS article-level licence (Creative Commons BY 2.0–4.0
   or CC0 URI) and human-research scope are re-checked on the exact bytes. `by-nd`, `by-nc` and
   ambiguous licences are rejected and listed.
5. Exact bytes are stored content-addressed and imported by the existing idempotent JATS
   importer, chunker (`structure-v1.1.b19e0939b5de`) and materializer (owner role; runtime
   retrieval keeps the read-only role).
6. All canonical passages are embedded by the pinned TEI (`cpu-1.9.4`,
   `Qwen/Qwen3-Embedding-0.6B@97b0c614…`, 512-d, document fingerprint checked); a new
   content-addressed collection `qdrant-passages-v1-<sha>` is created, populated, verified and the
   alias `dynamisrag-research-workspace` moves from exactly its previous target. Vectors are
   reused only for byte-identical passages.
7. The graph re-enters retrieval (`qdrant_hybrid_rrf`), the gate and canonical verification.
   Citations stay `MODEL_PROPOSED / PROPOSED`.

The browser shows the literature search (records, unique works, ingested DOIs/PMCIDs/PMIDs,
licences, collection) and, when no claim exists, the gate-accepted passages with **Inspect**
buttons that open the canonical passage and its source spans.

## 3. Running it

Write fresh hex secrets (`RESEARCH_*_PASSWORD`, `RESEARCH_QDRANT_KEY`, identity secret) and the
`DYNAMISRAG_*` URLs into the git-ignored `.env.research`, then:

```
docker compose --env-file .env.research -f compose.research-workflow.yaml up -d --wait
uv run python -m scripts.res638_research_bootstrap       # migrates; imports NO papers
set -a; . ./.env.research; set +a
uv run uvicorn dynamisrag.application:create_app --factory --host 127.0.0.1 --port 8029
uv run python -m dynamisrag.agent.worker                  # the only canonical writer
cd apps/workspace && npx next dev --hostname 127.0.0.1 --port 3028
WORKSPACE_RESEARCH_WORKFLOW=1 WORKSPACE_PROVIDER_BLOCKED=1 WORKSPACE_EXPECT_EMPTY_CORPUS=1 \
  WORKSPACE_TEST_URL=http://127.0.0.1:3028 WORKSPACE_TEST_PASSWORD=... \
  npx playwright test e2e/research-workflow.spec.ts
```

Required settings: `DYNAMISRAG_RESEARCH_ACQUISITION_ENABLED=true`, a separate canonical owner
`DYNAMISRAG_RESEARCH_INGEST_DATABASE_URL`, `DYNAMISRAG_RESEARCH_QDRANT_URL/_API_KEY`,
`DYNAMISRAG_RESEARCH_TEI_URL`, `DYNAMISRAG_RESEARCH_MAX_PAPERS` (≤10) and a run timeout long
enough for CPU embedding (`DYNAMISRAG_AGENT_RUN_TIMEOUT_SECONDS=3600` locally).

## 4. Limits

* CPU embedding is ~10 s per 512-token passage on this host (TEI saturates ~4 cores). The first
  attempt (5 papers, 136 passages) could not finish inside the 1,500 s run timeout and was
  stopped; the accepted run uses at most 4 papers and a 3,600 s timeout (first projection: 445 s
  of embedding). Hosted capacity is not qualified.
* The second attempt exposed no retrieval because the gate rejected every candidate; candidates
  and gate verdicts are now recorded and shown as "not admitted". The gate was not loosened.
* A follow-up acquisition counts already-ingested works against `max_papers`, so it re-selected the
  same 4 papers instead of adding reference-system studies.
* OpenAlex anonymous search returned HTTP 429 in the accepted run; it is recorded and discovery
  continued on Europe PMC.
* The query ladder is lexical and deterministic; it is not a systematic search and can miss
  relevant work. The relevance gate is lexical/structural, not semantic entailment.
* Acquisition is per worker process (one writer lock); a multi-worker deployment needs a
  database-level writer lease. Retrieval in every process follows the alias and re-verifies.
* No expert review. No deploy, hosting or spend.
