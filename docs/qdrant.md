# Local Qdrant retrieval layer

PostgreSQL 18 owns canonical documents, versions, passages and source spans.
Qdrant **1.20.0** is a rebuildable retrieval projection. OpenSearch remains the
reference backend. Existing HTTP routes, OpenSearch queries and RES-142 are unchanged.
There is no inference, model download, training, reranking or generation in this layer.

## Run and verify from PowerShell

Run from `E:\Data\Projects\DynamisRAG` with Docker Desktop and the existing `.env`:

```powershell
docker compose up -d --wait qdrant
docker compose ps qdrant
Invoke-RestMethod http://127.0.0.1:6333/
$testDirectory = '.tmp/qdrant-tests-' + [guid]::NewGuid().ToString('N')
uv run pytest tests/unit/test_qdrant.py tests/integration/test_qdrant_live.py -m 'integration or not integration' --basetemp $testDirectory -q
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/Test-QdrantPersistence.ps1
Invoke-RestMethod http://127.0.0.1:6333/collections
Invoke-RestMethod http://127.0.0.1:6333/aliases
```

The live tests retain explicitly synthetic collections for inspection. The persistence
script snapshots all stored point payloads and vectors, restarts only Qdrant, then
rechecks hashes, aliases, dense/BM25 results, RRF and source-span reconstruction.
Its post-restart probe performs **no inserts, updates or publication**. CI repeats
the live test and the same read-only persistence comparison on Linux.

REST is bound only to `127.0.0.1:6333`; gRPC is not published. The container is
digest-pinned and uses `dynamisrag_qdrant-data` and `dynamisrag_qdrant-snapshots`
named volumes, with explicit storage paths, WAL capacity and search thread limits.
The HTTP readiness health check uses Bash's TCP support because the pinned image
does not need a separately installed curl. Telemetry is disabled. Other containers
on the existing Docker network can reach the service: local Docker access is a trust
boundary. This single-node deployment has no HA. Back up volume data and take
Qdrant snapshots before host maintenance; a named volume alone is not a backup.
Never use `docker compose down -v` against retained data.

## Versioned schema

Schema revision: `qdrant-passages-v1`.
Fingerprint: `b83e0682ea5a90589e4411c2964703bcf3fd57203176729275818db13f0aaca1`.
The code computes SHA-256 over canonical JSON covering collection configuration,
payload indexes, point-ID rule and query parameters. Physical names are
`qdrant-passages-v1-<full SHA-256 of projection identity>`.

* Named vector `dense`: 512 dimensions, cosine distance, float32, on disk.
* HNSW: `m=16`, `ef_construct=100`, `full_scan_threshold=10000` KiB,
  `on_disk=true`; queries use `hnsw_ef=128`. Optimizer indexing threshold is
  10000 KiB. These are conservative single-node defaults, not tuned benchmark
  results. The two-point fixture uses a full scan and reports zero indexed HNSW
  vectors; larger segments build HNSW automatically when the threshold is crossed.
* One shard, one replica, write consistency 1, on-disk payload; 32 MiB WAL.
* Full-text index on `text`: word tokenizer, lowercase, token lengths 1–64,
  `scoring={"type":"bm25"}`. Qdrant 1.20 defaults `k=1.2`, `b=0.75`.
  No sparse embeddings or extra embedding model. Text processing is deliberately
  separate from OpenSearch's existing boosted BM25 semantics; rankings need not agree.
* Keyword indexes: passage key, canonical document key, version key, language,
  source system, source external ID, chunker revision, schema fingerprint,
  embedding config fingerprint and canonical projection fingerprint.
* Every point carries the canonical document payload, full source spans, content
  hash, model ID/revision, embedding configuration and manifest hashes, schema
  revision/fingerprint, expected point count and original vector hash.
* Point IDs are UUIDv5 of `dynamisrag:passage:<exact canonical passage_key>` under
  `uuid.NAMESPACE_URL`. The full canonical passage key remains in the payload.
* No quantization and no additional vectors. Qualification checks actual server
  schema, readiness, indexes, exact point membership, complete payload equality,
  and every stored named vector. Cosine storage normalizes to float32: comparison
  against original normalized vectors allows absolute component error `2e-6`.
  Original exact numerical values remain bound by the retained embedding manifest;
  Qdrant's normalized values do not replace that artifact.

## Canonical projection and publication

`QdrantProjection` consumes the existing `PassageProjectionManifest`, containing
`PassageProjectionDocument` values, and the existing `PassageEmbeddingManifest`.
It rejects empty snapshots, wrong dimensions, nonfinite/zero/float32-unsafe vectors,
stale content or chunker identity, missing source spans, configuration mismatches
and any missing or extra passage key. It does not infer vectors or match by row order.

The operator reads canonical passage records from PostgreSQL in a repeatable-read
transaction and makes no PostgreSQL writes. The retained artifact loader requires
an independently recorded SHA-256 receipt, validates the original manifest contracts
and reproduces its exact canonical bytes before any collection write.

```powershell
# Supply an existing retained artifact and its ORIGINAL trusted receipt/config.
# These prompts ask only for local paths and nonsecret identities.
$manifestPath = Read-Host 'Retained embedding-manifest.json path'
$artifactSha = Read-Host 'Expected artifact SHA-256 from original receipt'
$configSha = Read-Host 'Expected document embedding configuration SHA-256'
$chunker = Read-Host 'Exact canonical chunker revision'
$projectionArgs = @('--embedding-manifest', $manifestPath, '--artifact-sha256', $artifactSha,
    '--embedding-config-sha256', $configSha, '--chunker-revision', $chunker)
uv run python -m dynamisrag.search.qdrant_operator populate @projectionArgs
uv run python -m dynamisrag.search.qdrant_operator verify @projectionArgs
uv run python -m dynamisrag.search.qdrant_operator publish @projectionArgs

# Replacing an existing publication requires the explicitly reviewed physical target.
$previous = Read-Host 'Previously published physical collection to replace'
uv run python -m dynamisrag.search.qdrant_operator publish @projectionArgs --expected-previous $previous
```

Create, populate and verify never move an alias. Publish reruns qualification and
uses one atomic Qdrant alias transaction. A wrong or omitted previous target prevents
replacement. Publishing an already qualified current target is a no-op. Previous
physical collections are never deleted. Rollback is another explicit, qualified
publication of the retained old manifest with the current target as expected previous.

Populating a nonempty collection verifies its exact contents without overwriting
them. A repeated build is idempotent. A partial interrupted build or externally
corrupted collection fails closed for operator investigation; this implementation
does not silently repair or delete it. An OS file lock serializes alias publications
from this checkout and is released on process exit. Use one checkout as the local
publication authority; direct REST writers or other machines bypass that lock.
This is not a distributed compare-and-swap guarantee.

## Read-only queries and backend selection

`QdrantRetrieval.dense` takes a previously generated verified query vector and its
expected generation fingerprint. `bm25` takes text. Both return an independent
`QdrantLane` with physical collection, raw backend order, raw scores and full point
provenance. Filters are exact matches on the declared keyword fields. Neither method
generates vectors or writes data. Constructor identities pin the same model ID and
revision for documents and queries, while allowing distinct document/query prompt
configuration fingerprints.

`hybrid` resolves the alias once, queries both lanes against that physical target,
and passes typed candidates to the existing `reciprocal_rank_fusion`: candidate
window 50, `k=60`, deterministic final ties by passage key. Raw lanes are returned
alongside the RRF trace. It never invokes Qdrant's built-in fusion.

`search.backend.select_retrieval_backend` is the explicit opt-in construction
boundary (`RetrievalBackend.OPENSEARCH` or `.QDRANT`); only the selected factory
executes. There is no fallback. Existing `/search` and hybrid routes continue using
OpenSearch, so callers adopting Qdrant must select the adapter explicitly. There is
no change to the frozen RES-142 experiment or its baselines.

## Observed local execution, 2026-10-27 UTC

Qdrant reports version `1.20.0`, commit
`222b2e44ab1c3f17e8246396ead6558e45041df7`. Docker reports healthy and loopback-only
REST binding. Three retained physical collections contain **4 total points**:

* Published synthetic collection:
  `qdrant-passages-v1-22ce4f03b2280fd63c06a993b79fbbceece3180f08d3050b6fd55d6f6187e21d`
  — 2 points, alias `dynamisrag-synthetic-database-tests`.
* Alternate qualified synthetic collection:
  `qdrant-passages-v1-cffdc46774f6fa5b6b06c02d8fcb52b89c5764bfe2d8c00848d4fc2ea1ad9dfb`
  — 2 points; alias switch and rollback were exercised.
* Deliberately unpopulated qualification-negative collection:
  `qdrant-passages-v1-635b92e9da271b6655319aca948be86639bb497a872b9df4f5f3482ab313cae8`
  — 0 points; publication refused and the prior alias stayed intact.

All vectors are labeled `synthetic/database-correctness-only`, not real embeddings.
Query `[1,0,...,0]` ranks passage `a` × 64 first with cosine `1.0`, passage `b` × 64
second with `0.0`. BM25 `probiotic` returns A with `0.6785375`; an unmatched language
filter returns no hits. RRF ranks A first with `2/61 = 0.03278688524590164`.
Canonical source slices reconstruct both returned passage texts. The restart
comparison passed without re-inserting data; see [execution receipt](evidence/qdrant-local.json).
Validation passed: 2183 unit tests (167 integration tests deselected), 11 focused
unit/live tests, Ruff lint/format and strict Pyright. The existing full OpenSearch
integration suite was not run locally; CI retains it and adds the Qdrant restart check.

**Real retrieval is blocked by missing retained inputs.** The running PostgreSQL
database has zero passages. The previous local runner at
`.tmp/res142_scifact_local_runner.py:265` intended to write
`.tmp/res142-scifact-local-2af726c-20261015/embedding-manifest.json`; that artifact
and output directory are absent. Inspection of the repository, `.tmp`, and relevant
`E:\Data` artifact/download/dataset/run/staging/export/backup locations found no
verified real 512-dimensional manifest associated with canonical passage IDs.
The three downloaded RES-142 evidence ZIPs each contain 33 members but no `.npy`,
`.npz` or embedding manifest; the final audit ZIP has 29 members and none either.
There are synthetic matrix fixtures, plans, metadata and ranking evidence, which
cannot reconstruct exact numerical embeddings. Restore the canonical passage
snapshot plus its exact numerical `PassageEmbeddingManifest` and trusted receipt
(and a compatible retained query vector for a real dense retrieval test).
No new inference was attempted. No scientific retrieval deployment is claimed.

Qdrant API references: [full-text BM25](https://qdrant.tech/documentation/search/text-search/full-text-search/)
and [payload indexes](https://qdrant.tech/documentation/manage-data/indexing/).
