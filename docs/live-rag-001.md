# LIVE-RAG-001: live scientific retrieval API

The application accepts new questions at **POST
`http://127.0.0.1:8011/retrieve/qdrant`**. The running API is opt-in; the existing
OpenSearch `/retrieve`, its truncation policy, and frozen RES-142 results are
unchanged. There is no answer generator, reranker, training or GPU allocation.

The original PostgreSQL corpus and Qdrant projection were reused without
re-ingestion, document inference, collection construction or publication:

```text
documents=3 document_versions=3 passages=75 vectors=75
alias=dynamisrag-real-science-v1
collection=qdrant-passages-v1-5a20d2780fe6cd0ce0155ed7535c6507ec452f9ad4ae14cbd5990e13d7e8a709
document fingerprint=8438d37d8e8c8f975f62cbc6f6c644abc27304b18b7a330d9157a70f21c9c1f7
query fingerprint=94245fa64ff582ab1428c7aea38387e9091a23e538c96d7fbf7b48e1755da8fd
```

The existing exact numerical manifests are read from
`docs/evidence/real-science-v1/`. The original receipt must hash to
`8ede85766f656acccf7fd44b9c3fde627bd18289b74fe375dbf895d02ff4d240`.
The retained replay vector is verified at startup but never used for a user
question. Every valid request invokes TEI afresh, including repeated questions
and filters yielding no candidates.

The shared `embedding/scientific_cpu.py` definition preserves Qwen revision
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, float32, last-token pooling, 512
dimensions, normalization, the pinned `query` prompt and **no truncation**.
Its CPU/image/thread deployment attestation is identical to the earlier demo.
Both document and query generation fingerprints are checked against the observed
TEI runtime before inference, and the embedding run is bracketed by identity
observations. A different dtype, pooling, model, build, input length, generation
policy or CPU deployment fingerprint fails closed. These CPU timings do not
establish equivalence to the earlier GPU benchmark.

Application construction performs no network or artifact I/O. When enabled,
lifespan startup loads the trusted manifests, reads the exact canonical passage
keys in a repeatable-read transaction, verifies source reconstruction, and
qualifies Qdrant's schema, payloads, membership and all vectors once. Startup also
checks TEI and the alias. A failed initialization leaves this endpoint at 503;
restart the application after correcting the startup dependency or artifact.
Other application routes remain available.

Each request uses the existing adapter's independent BM25 and dense windows of
50 candidates and application RRF(k=60). The alias must still target the qualified
physical collection; a second alias check after fusion also rejects publication
during a request. Returned payloads must match their qualified hashes. Fresh
PostgreSQL reads are restricted to the returned candidate keys, and independently
compare document/version identifiers, citations, text, paragraph hashes, span
ownership, anchors and exact source slices. Only verified candidates are returned.
The request does not scroll or requalify the whole Qdrant collection.

The application reuses its SQLAlchemy connection pool and dedicated persistent
TEI/Qdrant HTTP clients. SQL verification has a five-second statement timeout;
TEI uses a single attempt and a 120-second timeout. Clients and the engine close
at lifespan shutdown. Query code has no insertion, publication, deletion or
reindexing path. The before/after receipt confirms all stored collection/vector
hashes, aliases and canonical row counts stayed unchanged during the live probe.

## Send a previously unseen question now

PowerShell, from `E:\Data\Projects\DynamisRAG`:

```powershell
$body = @{
    question = 'How does starting knee flexion angle influence muscle activation during plyometric exercises?'
    limit = 3
} | ConvertTo-Json -Depth 5
$timer = [System.Diagnostics.Stopwatch]::StartNew()
$result = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8011/retrieve/qdrant -ContentType 'application/json' -Body $body
$timer.Stop()
$result | ConvertTo-Json -Depth 25
$result.timings
$timer.Elapsed.TotalMilliseconds
```

Request bounds: 8,192 bytes including chunked bodies; question 1–512 characters
after trimming; fused limit 1–20 (default 5). Optional `filters` is an object with
at most seven exact-match keys: `passage_key`, `document_canonical_key`,
`document_version_key`, `language`, `source_system`, `source_external_id`,
`chunker_revision`. Values must be nonempty strings of at most 200 characters
without control characters. Unknown fields/filters and coercions such as boolean
limits are rejected. For example, add `filters = @{ language = 'en' }` to the body.

Response revision: `qdrant-retrieval-v1`. It contains physical collection and
projection identity, numerical-manifest hash, model/revision, both generation
fingerprints, complete independently ranked BM25/dense candidate traces, raw
scores, source text/identifiers/spans, citation verification flags, RRF ranks and
contributions, and separate pipeline timings. Vectors are not returned.

Responses explicitly declare `application/json; charset=utf-8` so Windows
PowerShell preserves scientific symbols and exact citation character spans.
All responses disable caching. Invalid JSON/fields return a sanitized 422
`invalid_request`; an oversized body returns 413 `request_too_large`. A TEI
capacity/validation rejection returns 422 `query_rejected`. Dependency, identity,
alias or provenance failures return 503 with an application-authored code such as
`embedding_unavailable`, `embedding_identity_mismatch`, `canonical_unavailable`,
`canonical_inconsistent`, `artifacts_invalid` or `qdrant_unavailable`. Disabled
configuration returns 503 `qdrant_disabled`. Unexpected failures return static
`internal_error`. Error logging includes only fixed codes, without query text,
article content, backend prose or tracebacks. The original readiness endpoint
continues to check its original PostgreSQL/OpenSearch dependencies.

## Observed live question and response

The exact command used for the five-request live probe was:

```powershell
uv run python scripts/probe_live_scientific_api.py
```

The first request used the question above and returned HTTP 200. Its observed
leader in **both** lanes was passage
`a2a28740ccd4e71ab85adc2970a962e5ab7ead39563ff9d217f9b8dfed84c877`:

```text
PMCID: PMC10780924
PMID: 38202906
DOI: 10.3390/s24010044
Title: Impact of Starting Knee Flexion Angle on Muscle Activity and Performance during Plyometrics without Jumping
Section: 5. Conclusions
BM25 rank=1 score=16.201311
dense rank=1 cosine=0.79328495
RRF rank=1 score=0.03278688524590164 = 1/61 + 1/61
version_key=958583513329768db12d9cd6c4b6730ed16d8b83ba5bc5203d5856983dd22643
paragraph_key=7e7f41c5eb15d7ccac13d9eabec93967988d78537551b5bbc5763d864a798f61
anchor=jats:/body[1]/sec[5]/p[1]
character span=[0,838)
```

The returned paragraph begins: “Our findings suggest that initiating resisted
plyometric exercises without jumping at a 90° knee flexion position results in
significantly increased muscle activity”. This is extracted source text, not an
LLM-generated answer. The complete paragraph and exact source spans are retained.

Observed first-request timings, milliseconds:

```json
{
  "query_embedding_ms": 725.6499000013719,
  "database_retrieval_ms": 76.43420000022161,
  "citation_verification_ms": 33.50749999844993,
  "total_request_ms": 835.5916000000434,
  "client_wall_ms": 912.9345999990619
}
```

`query_embedding_ms` includes runtime identity checks and fresh CPU inference.
`database_retrieval_ms` includes Qdrant/alias calls and RRF; PostgreSQL verification
is reported separately. `total_request_ms` is the server's retrieval pipeline,
excluding HTTP body validation and response serialization. `client_wall_ms` is
the measured total HTTP request including those costs and transfer. These are
observed individual requests, not benchmark percentiles.

All **57** unique candidates from the first question's two lanes were checked
again by the independent demonstration verifier against fresh PostgreSQL reads.
The second question asked about complex versus plyometric training in male
cricketers; its observed fused leader cited PMC10680285. A repeated knee-flexion
request performed another embedding and returned the same leading identities.
A PMCID filter returned 25 independently verified candidates; a nonmatching
language filter returned empty traces and fusion. The restarted CPU server log
contains exactly **five successful /embed calls** for those five valid requests.

Full evidence:

* [Retrieval log: actual candidate text, scores and citation spans](evidence/live-rag-001/retrieval-log.txt)
* [Complete first HTTP response](evidence/live-rag-001/response.json)
* [Five request receipts, independent verification counts and unchanged-state hashes](evidence/live-rag-001/execution-receipt.json)
* [Observed TEI runtime, image arguments, CPU/memory limits and prompt configuration](evidence/live-rag-001/tei-runtime.json)
* [Five fresh TEI request records](evidence/live-rag-001/tei-query-requests.txt)
* [Unavailable TEI: actual 503 from a separate application using a closed port](evidence/live-rag-001/tei-unavailable.json)

## Start the API after a future application restart

The existing CPU container and API are already running. These are the startup
commands used; do not launch another process on an occupied port:

```powershell
docker start dynamisrag-real-qwen-cpu
docker logs --tail 10 dynamisrag-real-qwen-cpu
# Continue once TEI logs Ready.
Invoke-RestMethod http://127.0.0.1:8081/info | ConvertTo-Json -Depth 8
$env:DYNAMISRAG_SCIENTIFIC_QDRANT_ENABLED = 'true'
$uvPath = (Get-Command uv).Source
$apiProcess = Start-Process -FilePath $uvPath -ArgumentList @('run','uvicorn','--factory','dynamisrag.application:create_app','--host','127.0.0.1','--port','8011','--no-access-log') -WorkingDirectory 'E:/Data/Projects/DynamisRAG' -WindowStyle Hidden -RedirectStandardOutput '.tmp/live-rag-server.out.log' -RedirectStandardError '.tmp/live-rag-server.err.log' -PassThru
$apiProcess.Id | Set-Content .tmp/live-rag-server.pid
```

No model download or document embedding command is part of application startup.
The existing container reuses its cached pinned weights.

## Validation

```powershell
uv run pytest tests/unit/test_scientific_retrieval.py -q --basetemp .tmp/live-rag-focused-final
uv run pytest tests/integration/test_scientific_citations.py tests/integration/test_qdrant_live.py -m integration -q --basetemp .tmp/live-rag-integration-focused
uv run pytest --basetemp .tmp/live-rag-full-unit-final -q
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

The tests cover fresh/repeat questions, exact generation semantics, runtime/model
drift, corrupt retained manifests, closed TEI/database/Qdrant dependencies,
missing/moved aliases, corrupted payloads, source text/spans/citations, bounded
and chunked requests, sanitized errors, opt-in startup and client lifecycle.
Local live integration includes a rolled-back synthetic PostgreSQL source and
the existing Qdrant correctness tests. Full integration runs in GitHub CI's
isolated empty database; it does not use or modify the scientific corpus.

The benchmark boundary guard now explicitly permits the extracted CPU definition
and asserts its model/revision equal the existing provisional Qwen identity.
Benchmark imports, GPU requirements, old OpenSearch generation semantics, frozen
results and the production backend default remain unchanged.

The focused endpoint suite passed **41 tests**; local live integration passed
**16 tests**. The final full local suite passed **2,253 tests**, with 182
integration tests deselected; Ruff lint/format and strict Pyright passed.
Exact-final-HEAD CI is established by the stacked PR checks, whose
head SHA must match `git rev-parse HEAD`; ancestor CI is not claimed for this change.
