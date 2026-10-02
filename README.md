# DynamisRAG

Evaluation-first RAG platform for auditable retrieval and evidence-grounded AI.

This repository implements the retrieval path end to end, from a scientific
document to a ranked, auditable hit:

- **canonical scientific ingestion** — JATS import and structure-aware chunking,
  producing immutable passages keyed by a document version and a chunker revision;
- **deterministic passages** — the same document version and chunker revision
  always yield byte-identical passages and the same `passage_key`s, so retrieval
  is reproducible rather than merely repeatable;
- **PostgreSQL as the authority** — canonical state lives in PostgreSQL and is
  read from there; every search hit addresses the exact canonical rows and
  character ranges it came from;
- **a disposable projection** — OpenSearch `passage-index-v1` serves BM25 over
  that state, and `passage-index-v2` is a vector-capable projection (Lucene HNSW,
  explicit dimension, space and pinned build parameters) that keeps v1's text
  mapping byte for byte. Both are rebuildable caches that can be deleted at any
  time;
- **deterministic embedding generation** — an `EmbeddingProvider` port with a
  Text Embeddings Inference adapter turns canonical passages into exact vectors
  and binds them in a `passage-embeddings-v1` manifest whose SHA-256 is a durable,
  byte-reproducible record of which model, at which weights, under which
  generation semantics, produced which vector for which passage.

Deliberately **not** implemented here, by design: the choice of a **default
embedding model and dimension** (a retrieval-quality benchmark, taken in a later
issue), a production ANN retrieval API, BM25+dense fusion, reranking, generation,
and agents. Those are scoped to later Linear issues. No model is selected, no
dimension is chosen and nothing is published to a vector index: a
`passage-index-v2` index therefore still holds vectors **supplied by the caller**,
and this repository does not index the vectors it generates.

---

Copyright © 2026 Julio Rodriguez. All rights reserved.

The source code is publicly viewable. No license is granted to use, copy,
modify, redistribute, sublicense, or create derivative works except where
applicable law or GitHub's Terms of Service provide otherwise.

The all-rights-reserved notice applies to original DynamisRAG code and
documentation only. Third-party materials retain their own copyright and
license terms. See `tests/fixtures/README.md` for notices applicable to test
fixtures.

---

## Platform

| Component      | Version    | Notes                                             |
| -------------- | ---------- | ------------------------------------------------- |
| Python         | 3.12.13    | Exact patch pinned in `.python-version` and CI |
| uv             | 0.12+      | Locked dependency resolution via `uv.lock`         |
| FastAPI        | 0.141.x    | Application factory, sync endpoints                |
| PostgreSQL     | 18.6       | `postgres:18.6-alpine`, container port 5432        |
| OpenSearch     | 3.8.0      | `opensearchproject/opensearch:3.8.0`               |
| TEI (optional) | 1.9.4      | `text-embeddings-inference:cpu-1.9.4`, separate compose file, not required to run anything |
| SQLAlchemy     | 2.1.x      | 2.x API, `postgresql+psycopg` dialect              |
| psycopg        | 3.3.x      | psycopg 3 only                                     |
| Alembic        | 1.20.x     | Four revisions, head `0004_passage_source_spans` |
| Ruff           | 0.16.x     | Lint **and** format                                |
| Pyright        | 1.1.414    | `typeCheckingMode = "strict"`                      |
| pytest         | 9.x        | Unit suite needs no infrastructure                  |

## Requirements

- Windows 11 with **PowerShell** — the documented commands are PowerShell and
  are what CI mirrors.
- [Docker Desktop](https://docs.docker.com/desktop/install/windows-install/)
  with the Linux container engine. DynamisRAG development is native
  Windows/PowerShell and does not depend on a user WSL distribution, WSL paths,
  WSL shells, or a WSL-hosted checkout. Docker Desktop may use its own configured
  virtualization backend internally.
- [uv](https://docs.astral.sh/uv/) on `PATH`.

There is exactly one checkout of this repository, at
`E:\Data\Projects\DynamisRAG`, and no second worktree.

## Quick start

```powershell
Set-Location "E:\Data\Projects\DynamisRAG"

# 1. Dependencies (creates .venv and installs the locked set)
uv python install 3.12.13
uv sync

# 2. Local configuration. .env is git-ignored; never commit it.
Copy-Item .env.example .env

# 3. Infrastructure
docker compose up -d
docker compose ps

# 4. Migrations
uv run alembic upgrade head

# 5. Serve
uv run python -m dynamisrag
```

Then, from another PowerShell window:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/healthz
Invoke-RestMethod http://127.0.0.1:8000/readyz
```

## Repository layout

```
.
├── .github/workflows/ci.yml   # lint, types, unit tests, migrations, integration
├── alembic/                   # migration environment and revisions
│   ├── env.py                 # reads its DSN from dynamisrag.config, never alembic.ini
│   ├── script.py.mako
│   └── versions/                 # 0001_foundation_baseline → 0004_passage_source_spans
├── src/dynamisrag/
│   ├── application.py         # create_app() factory; no I/O at construction
│   ├── config.py              # strict, frozen, validated settings
│   ├── embedding/             # contracts.py (generation semantics, provider port,
│   │                          # deployment-semantics protocol), identity.py (model
│   │                          # identity), errors.py, manifest.py
│   │                          # (passage-embeddings-v1), tei.py (tei-http-v1 adapter
│   │                          # + TeiDeploymentSemantics)
│   ├── benchmark/             # RES-138 retrieval benchmark; no result, no default.
│   │                          # contracts.py (frozen workloads, model revisions,
│   │                          # prompts, dimensions, Drive layout, artifact
│   │                          # revisions), artifacts.py (canonical envelopes,
│   │                          # shards, run manifests, verified Drive copies),
│   │                          # runtime.py (runtime fingerprint, CUDA/torch guards),
│   │                          # beir.py (verified acquisition + loading),
│   │                          # retrieval.py + quality.py (exact scoring, metrics,
│   │                          # paired bootstrap), mrl.py (Matryoshka gate),
│   │                          # calibration.py (deterministic item selection),
│   │                          # selection.py (predeclared rule, never applied),
│   │                          # res138.py (notebook-facing facade), runner.py
│   │                          # (Colab-only GPU encoder; the only torch importer),
│   │                          # bundle.py (local no-trust-on-first-use verifier)
│   ├── logging_config.py      # stdlib-only deterministic logging
│   ├── db/                    # engine.py (SQLAlchemy/psycopg), probe.py
│   ├── health/                # models.py, router.py (/healthz, /readyz)
│   └── search/                # client.py (shared HTTP boundary), errors.py,
│                              # schema.py (versioned index), projection.py,
│                              # bm25.py (versioned query + service),
│                              # router.py (/search), opensearch.py (probe)
├── notebooks/
│   └── res138_colab.ipynb     # orchestration only; algorithms live in the harness
├── requirements/
│   └── res138-colab.txt       # Colab-only pins; no torch, no CUDA wheel
├── tests/
│   ├── unit/                  # no infrastructure required
│   └── integration/           # requires the live stack
├── .env.example
├── alembic.ini
├── compose.yaml               # the default stack: postgres + opensearch
├── compose.embedding.yaml     # optional, TEI; never a default-stack dependency
├── pyproject.toml
├── uv.lock
└── README.md
```

## Endpoints

### `GET /healthz` — liveness

Returns `200` whenever the process is running. It touches **no** infrastructure,
so a dependency outage can never cause an orchestrator to restart a healthy
process.

```json
{
  "status": "up",
  "service": "dynamisrag",
  "version": "0.1.0",
  "environment": "local"
}
```

### `GET /readyz` — readiness

Probes every required dependency explicitly and in a fixed order. Returns `200`
when all are healthy, `503` otherwise. **The body shape is identical in both
cases**, so a failure is machine-readable rather than a stack trace.

```json
{
  "status": "up",
  "service": "dynamisrag",
  "version": "0.1.0",
  "environment": "local",
  "dependencies": {
    "postgres": {
      "name": "postgres", "status": "up", "latency_ms": 11,
      "version": "18.6", "detail": null
    },
    "opensearch": {
      "name": "opensearch", "status": "up", "latency_ms": 90,
      "version": "3.8.0", "detail": null
    }
  }
}
```

When a dependency is down, its `status` is `down`, `version` is `null`, and
`detail` explains why — for example
`"TransportError cause=ConnectError operation=node_root"` or
`"AuthenticationFailed operation=node_root HTTP 401"`.

The OpenSearch detail is an **application-authored summary**, assembled from
the exception class, the operation, the HTTP status and OpenSearch
`error.type`. The node's own `error.reason` is untrusted: it routinely quotes
the value that was rejected, the query, the document that failed to index or an
internal detail, and here that document is article text. It is therefore never
read, so it cannot reach the readiness payload. Details are also length-bounded
so a long identifier cannot dominate the payload.

Both endpoints send `Cache-Control: no-store`.

Verify the failure path without touching the code:

```powershell
docker compose stop opensearch
curl.exe -s -o NUL -w "healthz=%{http_code}`n" http://127.0.0.1:8000/healthz
curl.exe -s -w "`nreadyz=%{http_code}`n"   http://127.0.0.1:8000/readyz
docker compose start opensearch      # readiness returns to 200 once healthy
```

### `GET /search` — BM25 passage search

Ranks the versioned OpenSearch passage projection with an explicit, versioned
BM25 query. The projection is a **disposable, rebuildable cache** of canonical
PostgreSQL state: nothing canonical is ever read from it, and it can be deleted
and rebuilt at any time.

| Parameter | Required | Default | Range     |
| --------- | -------- | ------- | --------- |
| `q`       | **yes**  | —       | non-whitespace, ≤ 512 characters |
| `limit`   | no       | `10`    | `1`–`50`  |

```powershell
curl.exe -s "http://127.0.0.1:8000/search?q=probiotic+soy+exercise&limit=5"
curl.exe -s "http://127.0.0.1:8000/search?q=colon+lesions"
```

```json
{
  "query": "probiotic soy exercise",
  "query_revision": "bm25-v1",
  "index_schema_revision": "passage-index-v1",
  "projection_sha256": "b62af58acf996733a67dd5955384eb8c6a43b029626d6df8461fed653734048d",
  "chunker_revision": "structure-v1.1.b19e0939b5de",
  "total": 19,
  "took_ms": 11,
  "hits": [
    {
      "rank": 1,
      "score": 1.777401,
      "passage_key": "1b272624bb39fe23cca4a9db124f0be85add56cf66d0f136f0003f53e5ca382c",
      "section_path": "2.5",
      "section_title": "Physical exercise",
      "primary_source_anchor": "jats:/body[1]/sec[2]/sec[5]/p[1]",
      "source_spans": [
        {
          "source_order": 0,
          "paragraph_key": "3aa502f0046f3aad94bd0002307a9c14ef6b1a281d9f97906e48ff2ab8823870",
          "paragraph_source_anchor": "jats:/body[1]/sec[2]/sec[5]/p[1]",
          "start_char": 0,
          "end_char": 264
        }
      ]
    }
  ]
}
```

Every hit is auditable: `passage_key`, `document_version_key`,
`document_canonical_key` and `primary_source_anchor` address the canonical
PostgreSQL rows, and each `source_spans` entry's character range addresses the
canonical `Paragraph.text` exactly.

- Results are ordered by descending BM25 score with `passage_key` ascending as
  a stable tie-break, so tied rankings are reproducible.
- `422` for a blank query or an out-of-range `limit`; `503` with the fixed
  message `search is temporarily unavailable` when the backend cannot answer.
  The failure is logged as structured safe values only — exception class,
  operation, HTTP status, `error.type`, target — never the exception text, so
  an OpenSearch `error.reason` cannot be relayed into log aggregation.
- `Cache-Control: no-store` on every search response, success or failure.
- No PostgreSQL is read: search reads the projection only.

### Projecting passages

The projector is a service, not an endpoint. Rebuild the projection from
canonical PostgreSQL with the exact chunker revision to project:

```powershell
uv run dynamisrag project-passages --chunker-revision structure-v1.1.b19e0939b5de
```

The revision is mandatory and never inferred: PostgreSQL may hold several
immutable passage sets for one document version, and indexing more than one
would duplicate retrieval content under different identities.

## Command line

```powershell
dynamisrag                                     # serve the HTTP API (unchanged)
dynamisrag search "probiotic soy exercise"     # BM25 over the projection, JSON on stdout
dynamisrag search "colon lesions" --limit 5

dynamisrag benchmark res138-plan --code-sha <40-hex>
dynamisrag benchmark verify-res138-bundle <path>
```

The CLI uses the same search service and the same `SearchResponse` as
`GET /search`; there is no second search implementation. A backend failure
exits non-zero with one safe line on stderr: an application-authored failure
such as a blank query or a chunker revision that does not exist is shown in
full, while a low-level OpenSearch failure is rendered from its safe summary
(exception class, operation, HTTP status, `error.type`, target) and never from
the exception text.

## Configuration

All variables are prefixed `DYNAMISRAG_` and are read from the process
environment and from `.env` at the repository root. The `.env` location is
derived from `__file__` with `pathlib`, so it does **not** depend on the current
working directory and behaves identically on Windows and Linux.

| Variable                                  | Required | Default     | Purpose                                  |
| ----------------------------------------- | -------- | ----------- | ---------------------------------------- |
| `DYNAMISRAG_ENVIRONMENT`                  | no       | `local`     | `local`, `test`, `ci`, `staging`, `production` |
| `DYNAMISRAG_DATABASE_URL`                 | **yes**  | —           | PostgreSQL DSN (`postgresql://`)          |
| `DYNAMISRAG_OPENSEARCH_URL`               | **yes**  | —           | OpenSearch base URL                       |
| `DYNAMISRAG_OPENSEARCH_USERNAME`          | no       | `admin`     |                                          |
| `DYNAMISRAG_OPENSEARCH_PASSWORD`          | **yes**  | —           | Wrapped in `SecretStr`; never logged      |
| `DYNAMISRAG_OPENSEARCH_VERIFY_TLS`        | no       | `true`      | `false` for the self-signed demo cert     |
| `DYNAMISRAG_OPENSEARCH_INDEX_ALIAS`       | no       | `dynamisrag-passages` | Stable query target of the passage projection |
| `DYNAMISRAG_OPENSEARCH_BULK_BATCH_SIZE`   | no       | `500`       | Documents per bulk request                 |
| `DYNAMISRAG_DEPENDENCY_TIMEOUT_SECONDS`   | no       | `5`         | Bound on every readiness probe            |
| `DYNAMISRAG_TEI_URL`                      | no       | unset       | TEI base URL; `http://127.0.0.1:8080` for the reference container |
| `DYNAMISRAG_TEI_MODEL_ID`                 | no       | unset       | Repository the deployment insists is served |
| `DYNAMISRAG_TEI_MODEL_SHA`                | no       | unset       | Immutable Hub commit id (40 lowercase hex) |
| `DYNAMISRAG_TEI_API_KEY`                  | no       | unset       | Bearer token; `SecretStr`, never logged   |
| `DYNAMISRAG_TEI_VERIFY_TLS`               | no       | `true`      | Only consulted for an `https://` URL      |
| `DYNAMISRAG_TEI_TIMEOUT_SECONDS`          | no       | `30`        | Bound on each `/info` and `/embed`        |
| `DYNAMISRAG_TEI_BATCH_SIZE`               | no       | `32`        | Inputs per `/embed` request               |
| `DYNAMISRAG_TEI_MAX_ATTEMPTS`             | no       | `3`         | Total attempts per request, jitter-free    |
| `DYNAMISRAG_TEI_RETRY_BACKOFF_SECONDS`    | no       | `0.5`       | Base of the linear backoff schedule        |
| `DYNAMISRAG_HOST` / `DYNAMISRAG_PORT`     | no       | `127.0.0.1` / `8000` | Bind address                     |
| `DYNAMISRAG_LOG_LEVEL`                    | no       | `INFO`      |                                          |

The `DYNAMISRAG_TEI_*` variables are all optional and default to unset. Nothing in
the liveness, readiness or BM25 paths reads them, and the readiness probe does not
treat a missing model server as a degraded dependency. `tei_url`,
`tei_expected_model_id` and `tei_expected_model_sha` must be set **together**:
a URL with no pinned revision is the configuration most likely to be wrong and
looks configured, so it is refused rather than accepted and left to record
whatever the server happened to be serving. `DYNAMISRAG_TEI_VERSION` and
`DYNAMISRAG_TEI_MAX_CLIENT_BATCH_SIZE` are read by `compose.embedding.yaml`, never
by the application.

**No variable configures the generation semantics or the deployment attestation.**
Which model, and under which normalization, truncation and dimensions, is a caller
argument, and the attested startup policy (`TeiDeploymentSemantics`) is passed
alongside it — both are hashed into the embedding fingerprint, and neither belongs
in configuration next to a timeout.

Design rules, all covered by tests:

- Required values have **no defaults**. A missing DSN fails at startup with a
  validation error instead of guessing.
- `Settings` is `frozen=True`; the model cannot be mutated after construction.
- `database_url` and `opensearch_password` use `Field(repr=False)`, so a
  password embedded in a DSN can never reach a log line through `repr()`.
  `Settings.redacted_sqlalchemy_url()` gives a loggable DSN.
- The DSN is written with the plain `postgresql://` scheme.
  `dynamisrag.config.to_sqlalchemy_url` attaches `+psycopg` in exactly one
  place, so the driver is never repeated across code and `.env` templates.
- `alembic/env.py` takes its DSN from `dynamisrag.config`, so migrations and the
  application can never disagree about the target database. `alembic.ini`
  contains no `sqlalchemy.url` at all.

### About the local credentials

`.env.example` contains **published, local-only development passwords**. They
exist so `docker compose up -d` works on a fresh checkout with nothing secret in
version control. The services bind to `127.0.0.1` only, and the passwords are
documented as non-reusable. Provision real secrets for anything shared.

### Why PostgreSQL is published on host port 55432

A workstation that already runs its own PostgreSQL on `5432` cannot also bind
`5432`. DynamisRAG publishes the container's `5432` on host port `55432` so the
stack starts regardless of what else is running. The DSN points at the host
port; nothing inside the network uses `55432`.

## Local runtime

```powershell
docker compose up -d          # start
docker compose ps             # both services report (healthy)
docker compose logs           # tail logs
docker compose stop opensearch    # stop one dependency
docker compose down           # stop and remove containers (volumes persist)
docker compose down -v        # also delete the data volumes
```

Only two services exist here. Temporal, Valkey, Keycloak, OpenFGA, vLLM,
Kafka, Neon and Vercel integrations are explicitly **out of scope** and are
deliberately absent. TEI is the one optional service, and it lives in a separate
`compose.embedding.yaml` precisely so that the default stack cannot depend on a
model server — see [Serving an embedding model](#serving-an-embedding-model).

### OpenSearch notes

The official image enables the security plugin, which serves **HTTPS on 9200
with a self-signed demo certificate**. Plain HTTP is refused with a TLS
protocol error, and anonymous or wrong credentials return `401`. That is why
`.env.example` sets `DYNAMISRAG_OPENSEARCH_URL=https://127.0.0.1:9200` and
`DYNAMISRAG_OPENSEARCH_VERIFY_TLS=false`. Set the latter to `true` once a
trusted certificate is provisioned.

## Embeddings

```
canonical passages
    -> EmbeddingInput                 frozen, content-addressed, text bound to
                                      its own content digest
    -> canonical order                passage_key ascending, duplicates refused
    -> EmbeddingProvider              a port that knows no vendor
    -> TEI adapter                    protocol revision tei-http-v1
    -> exact vectors                  validated on arrival, never repaired
    -> PassageEmbeddingManifest       passage-embeddings-v1, canonical JSON + SHA-256
    -> PassageVector + EmbeddingModelIdentity   the RES-136 boundary
```

There is **no embedding table, no vector table and no model-run table**, and no
migration is added: the deterministic manifest *is* the artifact. A database row
would have to be trusted to reproduce, and an index whose identity is a trusted
row is not an index whose identity is a digest.

Every entry's `content_sha256` is the SHA-256 of the **exact UTF-8 bytes** of that
entry's passage text, and the constructor verifies it against the text rather than
merely checking its shape. A shape-valid digest belonging to other content would
let a run record an identity for a passage it never embedded, so the check happens
at construction — before a batch is built, before a socket is touched, and long
before a manifest exists to disagree with. The failure names the passage key and
both digests, never the passage.

### What is decided here, and what is not

Decided here: the canonical input contract, the semantic generation config, the
observed model and runtime identity, the attested deployment semantics,
deterministic client batching, the bounded retry policy, response validation, and
the manifest.

Not decided here, deliberately: **which embedding model is best** and **which
dimension to index**. Those are a retrieval-quality question. The harness that will
answer them lives in [`src/dynamisrag/benchmark/`](#the-res-138-retrieval-benchmark)
and has produced no result: nothing has been ranked, no candidate has been
selected, and no default model or dimension is configured anywhere in this tree.

### Identity is observed, never asserted

Before any vector is generated the adapter reads `GET /info` and refuses to
proceed unless the server can prove it is serving the expected embedding model at
the expected **immutable Hub commit id**, with a non-empty named `pooling`. A
configured model name is a claim; a branch, a tag or `latest` is a claim about a
moving target, and both are refused — the expectation at construction and the
observation at request time.

```
EmbeddingModelIdentity.model_id       = observed /info model_id
EmbeddingModelIdentity.model_revision = observed /info model_sha
EmbeddingModelIdentity.embedding_config_sha256 = digest over
    observed:  provider, protocol revision, TEI version, TEI sha, dtype,
               pooling, max_input_length
    attested:  default-prompt policy, dense-path policy
    requested: normalize, truncate, truncation_direction, prompt_name, dimensions
```

Three halves, hashed together, because each changes the returned floats
independently. Two TEI builds can honour identical bytes and return different
vectors; so can two servers differing only in a startup flag nobody mentioned; so
can the same weights under different normalization. Pooling and
`max_input_length` are in it too: pooling is not a request parameter, and
`max_input_length` is the tokenizer truncation boundary, so with truncation on,
512 and 1024 embed different tokens. The model id and revision are **not** — they
stay readable first-class fields, because folding them into a digest would only
make them unreadable without decompressing it.

**Batch size, timeout, attempt count, backoff and the server's capacity limits are
execution policy and are never hashed.** Two runs that needed different amounts of
luck produce the same manifest, because a transient overload must not invalidate
every vector index built from a model.

### Some startup semantics are attested, not observed

`/info` reports the build, the model, the dtype, the pooling and the batching
limits. It reports **nothing** about `--default-prompt`, `--default-prompt-name` or
`--dense-path`, and no request can set or clear them. Upstream tokenization
resolves a null `prompt_name` to the deployment's default, so:

> `prompt_name = null` means **"use the attested server default"** — not
> "no prompt".

Two containers with an identical `/info` and different command lines therefore
return different vectors for byte-identical requests, which is why
`TeiDeploymentSemantics` states the policy explicitly, hashes it into the
fingerprint, and records it in every manifest. It is *attested* because it has to
be: no status document can report it. The reference deployment in
`compose.embedding.yaml` is attested to have **no default prompt** and **no
dense-path override**, and a test fails if that file ever passes a flag
contradicting the attestation.

A literal `--default-prompt` is supported by binding the SHA-256 of the prompt
rather than the prompt, so an operator's template never reaches a manifest, an
exception or a log line.

### Drift is refused

A model server can restart, or be replaced behind the same URL, while batches are
in flight. One run therefore reads `/info` before its first batch and again after
its last, and abandons the whole run unless the **run identity** held across the
generation. That identity is the semantic runtime payload *plus the model id and
its immutable revision* — checked on the vendor-blind abstraction, with no TEI
involved, because a port that compared only the runtime would accept a provider
that swapped models mid-run and then record the first identity over vectors from
both. Capacity limits are excluded, so a restart that came back with the same
weights is still one run.

A manifest of vectors produced under two identities is not reproducible and would
name an index nothing could rebuild.

### Canonical order comes before batching

Inputs are sorted by `passage_key` **before** the provider is touched, and
duplicate keys are refused. That sort decides the sequence of `/embed` request
bodies as well as the manifest, so a shuffled caller produces byte-identical
requests and a byte-identical artifact. Batches are then fixed-size contiguous
slices: five inputs at `batch_size = 2` are exactly 2 / 2 / 1. A configured batch
larger than the server's advertised `max_client_batch_size` is a configuration
error and is **not** silently shrunk, because the partition is part of what makes
a run reproducible.

### Serving an embedding model

Text Embeddings Inference lives in its own compose file, so the default stack
cannot depend on it:

```powershell
docker compose up -d --wait                                    # postgres + opensearch only
docker compose -f compose.yaml -f compose.embedding.yaml up -d --wait tei
docker compose -f compose.yaml -f compose.embedding.yaml logs -f tei
docker compose -f compose.yaml -f compose.embedding.yaml down
```

Set `DYNAMISRAG_TEI_MODEL_ID` and `DYNAMISRAG_TEI_MODEL_SHA` in `.env` first;
`compose.embedding.yaml` fails loudly without them. It is a separate file rather
than a profile inside `compose.yaml` because Compose interpolates every service's
variables when it loads a file — a required TEI variable behind a profile would
still be demanded by a plain `docker compose up`, and would make CI fail over a
model server CI never starts.

The reference container serves **plaintext HTTP on `127.0.0.1:8080`**, which is
what `DYNAMISRAG_TEI_URL=http://127.0.0.1:8080` in `.env.example` says.
`DYNAMISRAG_TEI_VERIFY_TLS=false` disables certificate *verification*; it does not
make an HTTPS URL speak plaintext, so it is not a downgrade. HTTPS remains fully
supported for an external deployment — point the URL at `https://…` and set
`VERIFY_TLS=true` once a trusted certificate is provisioned.

The container is started to match the attestation DynamisRAG fingerprints: no
`--default-prompt`, no `--dense-path`, an immutable `--revision`, and an explicit
`--max-client-batch-size`. The model cache is mounted at `/data`, where the
official image keeps its Hugging Face cache, so the named volume actually
persists the download.

A TEI container and model are **not** required for `dynamisrag`, `/healthz`,
`/readyz` or BM25. No health probe, readiness check, migration or integration
test references it, and the CI suite never downloads a Hub model.

### Failure safety

Passage text is canonical scientific content and a dense vector is derived from
it, so neither is ever admitted into an exception, a log line, a retry diagnostic
or a provider summary. TEI's own failure prose is not read either - for this
endpoint the rejected value *is* a passage, and TEI genuinely quotes it back into
the `error` field. Only the machine-generated `error_type`, the HTTP status, the
batch and input ordinals, the attempt number and the content-addressed
`passage_key` are carried, and `safe_summary()` is assembled from those structured
fields alone. A bearer token, an `Authorization` header and a URL with userinfo
never appear either.

A content-digest mismatch is the one case that would otherwise tempt a boundary to
echo content, and it does not: the failure names the `passage_key` and both
SHA-256 digests, which locate the offending caller without disclosing the passage.
A literal TEI default prompt is reduced to its digest at construction for the same
reason.

The configuration contracts are strict at runtime rather than merely annotated.
`normalize=1`, `truncate="false"`, `dimensions=1.5`, `max_attempts=True`,
`timeout_seconds=NaN` and a `retry` that merely looks like a policy are all refused
as `EmbeddingContractError` at construction — before a request, a sleep or a
digest. A boolean reaching the hashed bytes would give one semantic setting two
fingerprints; a NaN backoff satisfies every bound in Python and then fails inside
`time.sleep`, at a point chosen by when the server happened to be overloaded.

## The RES-138 retrieval benchmark

The embedding model and the dimension to index at are **not decided in this
repository**. `src/dynamisrag/benchmark/` is the harness that will produce that
evidence on a GPU, and it currently holds no result: no ranking has been run, no
candidate has been selected, and no default exists anywhere in the tree. That last
property is asserted rather than promised — `tests/unit/test_benchmark_boundaries.py`
walks every source file and fails if any module binds `DEFAULT_EMBEDDING_MODEL`,
`DEFAULT_DIMENSION` or a selected candidate, if any production package imports
`dynamisrag.benchmark` at all, or if a production module names either candidate.

What the harness *is*: frozen workloads, frozen model revisions, exact retrieval,
two recall metrics and nDCG@10, a paired bootstrap, exact Matryoshka derivation, a
deterministic calibration set, content-addressed artifacts, a resumable run
manifest, and a local verifier that re-checks a finished bundle from its bytes
alone.

### Two commands run locally, with no GPU and no model

```powershell
uv run dynamisrag benchmark res138-plan --code-sha <40-hex> --out plan.json
uv run dynamisrag benchmark verify-res138-bundle <path> [--code-sha <40-hex>]
```

`res138-plan` writes the frozen plan for an exact commit and prints its SHA-256.
That digest is reproducible on any machine with no GPU, no Drive and no Hub
account, which is what makes it useful: a reviewer computes the plan identity
independently and compares it with the one a Colab session recorded.

`verify-res138-bundle` takes a run directory or a zip of one and refuses it unless
every shard, sidecar, run manifest and bundle manifest agrees — declared digests,
shard ordinals with no gap and no duplicate, canonical `passage_key` order within
and across shards, matrix dtype and normalisation, the runtime revision, and the
model and dataset identities the run claims. It trusts nothing on first use: it
recomputes every digest it verifies.

Both commands import the benchmark inside the handler, so starting the served
application never loads the harness or the numeric stack behind it.

### The GPU preflight, in Colab

`notebooks/res138_colab.ipynb` is orchestration only. It computes no metric, parses
no corpus, derives no Matryoshka shortcut, ranks nothing and writes no artifact of
its own — every one of those is an import from `dynamisrag.benchmark`, and
`tests/unit/test_benchmark_colab.py` fails if the notebook ever grows its own.

A preflight run, top to bottom. The ordering is part of the contract, not a matter
of taste: a clean Colab runtime has no DynamisRAG installed, so nothing may import any part
of it until the exact commit has been cloned, verified and put on `sys.path`.

1. **Parameters.** Set `CODE_SHA` to the exact 40-character commit of the harness
   branch. `RUN_MODE` is `"preflight"` by default; `APPROVED_PREFLIGHT_SHA256` is
   empty. Nothing is imported here.
2. **GPU.** Raw `torch` only. A CPU runtime fails here with the actionable message,
   because a CPU MRL comparison would not answer the question the gate asks, and
   because the refusal has to arrive before twenty minutes of model load rather
   than after it.
3. **Drive.** Mount, then require the literal paths from the parameter cell to
   exist. A missing folder is named. The repository's storage contract cannot be read
   yet, so it is compared against these literals in step 6.
4. **`CODE_SHA` shape.** 40 lowercase hexadecimal characters, checked with the
   standard library before a clone is attempted. Blank and `main` both fail.
5. **Code.** Clone `--no-checkout`, `fetch --depth 1 origin <CODE_SHA>`,
   `checkout --detach`, compare `rev-parse HEAD` with `CODE_SHA`, require a clean
   tree, and only then insert `REPO_DIR/src` into `sys.path` and import DynamisRAG.
   GitHub is the only code transport: no bundle, no tarball, and Colab never authors or
   pushes anything.
6. **Literals against contracts.** The parameter cell repeats the frozen Drive root,
   archive digests, model revisions, shard size, candidate dimensions and bootstrap
   triple so it can state what it is about to check. This step makes that repetition
   load-bearing: it compares them with the contracts at this commit and stops on any
   disagreement, then constructs `Res138ColabConfig`.
7. **Install.** `pip install -r requirements/res138-colab.txt`, then
   `require_torch_unchanged(before, after)`. That file pins
   `sentence-transformers==5.0.0` and `transformers==4.51.3` — the versions both
   pinned model repositories declare in their own `config_sentence_transformers.json` —
   and pins **no torch and no CUDA wheel**, because Colab owns the CUDA runtime and a
   pin would either fail to install or silently replace a working build. The
   before/after comparison is what makes that safe to rely on.
8. **Plan, runtime, sources, prompts.** The plan is written from the cloned tree and
   its digest printed. The runtime fingerprint records the torch, CUDA and Colab
   versions. `verify_and_cache_beir_sources` checks each archive against the frozen
   SHA-256 in `contracts.py` and refuses on mismatch; `verify_pinned_model_metadata`
   confirms the served commit ids, pooling, prompt strings, positional limits and the
   frozen loading semantics.
9. **Calibration.** `select_calibration_set` draws 2 items per
   (workload × kind × length band) — 36 items over the three workloads — **once**, and
   `calibrate_frozen_candidates` loads each candidate **once** and decides the
   Matryoshka shortcut separately for every model, path and workload: 2 models × 2
   paths × 3 workloads is 12 decisions from 2 model loads. Between candidates the
   encoder is deleted, collected and the CUDA caching allocator emptied, so the second
   model does not share a card with the first model's dead blocks.
10. **Preflight bundle and hard stop.** `write_preflight_bundle` writes the artifact
    and prints its SHA-256. The cell then stops.

### Frozen model loading semantics

Loading is part of the candidate identity, because it changes the vectors and nothing
else would record the change:

| Candidate                   | `trust_remote_code` | compute dtype | output dtype |
| --------------------------- | ------------------- | ------------- | ------------ |
| `voyageai/voyage-4-nano`    | `true`              | `float32`     | `float32`    |
| `Qwen/Qwen3-Embedding-0.6B` | `false`             | `float32`     | `float32`    |

`trust_remote_code` is `true` for Voyage because the pinned repository ships custom
modelling code and cannot be constructed without it, and `false` for Qwen because it
does not. That difference is the reason the field exists: a runner that branched on the
model id would give the same two answers today and would be one candidate away from
handing the wrong policy to a third, invisibly.
`tests/unit/test_benchmark_boundaries.py` refuses any string literal naming a candidate
in `runner.py`.

The compute dtype is frozen to `float32` for both. The local sealed TEI 1.9.4 reference
reported `model_dtype float32`, TEI equivalence is the gate that decides whether native
Colab vectors may be used in production at all, and one explicit compute dtype keeps the
four candidate runs comparable. Qwen's pinned config declares `bfloat16` and Voyage's
recommended GPU path is BF16; neither is inherited. The dtype is passed explicitly as
`model_kwargs={"torch_dtype": torch.float32}` and then **read back** off
`next(model.parameters()).dtype` and compared with the frozen value — a model that
ignored the request is a failed preflight, not a quietly mislabelled artifact. If float32
turns out not to fit the assigned Colab GPU, that is a feasibility finding to report, not
a licence to switch: amending the closed `RES138_SUPPORTED_DTYPES` set is a visible
contract edit, and it changes the plan digest.

Provenance records `requested_compute_dtype`, `observed_compute_dtype` and
`output_dtype` as three separate fields and no bare `dtype`. The output matrix itself is
unchanged: `numpy.float32`, C-contiguous, finite, normalised. No CUDA allocator telemetry
is recorded anywhere, because free and reserved VRAM differ between two runs of identical
code.


The last cell can only run with `RUN_MODE == "full"` **and** an
`APPROVED_PREFLIGHT_SHA256` that matches the digest a reviewer computed locally.
It stops rather than embedding anything until both hold.

### Storage

| Purpose                                | Path                                                    |
| -------------------------------------- | ------------------------------------------------------- |
| Mounted Drive root                     | `/content/drive/MyDrive/DynamisRAG/RES-138`             |
| All active work (ephemeral disk)       | `/content/res138`                                       |
| Finished checkpoints and evidence      | `…/RES-138/runs/<run-id>/`                              |

Drive holds evidence and finished checkpoints. It is never used for large random
I/O: a mounted Drive is a network filesystem, and writing a shard matrix into it
byte by byte turns a compute run into an I/O benchmark.

The folder ids are recorded in `RES138_DRIVE_LOCATIONS` for provenance and
operator confirmation only. No computation requires one — a run that needed a
folder id to start would break the moment the tree is reorganised.

### What the preflight proves, and what it does not

It proves that on one GPU, at one pinned torch build, the derived 512-dimension
prefix is a valid substitute for the full 1024-dimension vector **on 36 short
scientific items** — cosine ≥ 0.999999, max component difference ≤ 1e-5, identical
top-10 — and that every frozen identity in the plan is the one on disk.

It proves nothing about retrieval quality, which needs the corpus pass, and it
records no TEI equivalence result: that gate is evaluated locally against
TEI 1.9.4 because Colab cannot run Docker.

### Interrupting and resuming

`open_drive_run` is idempotent. The run manifest is keyed by a derived run id
computed from the plan, source, runtime and model identities — not from a
timestamp — so a reconnect finds its own run, re-verifies every artifact already
present and resumes at the first shard that is missing. A manifest whose identity
disagrees with the run refuses to be reused.

## Migrations

```powershell
uv run alembic upgrade head      # apply
uv run alembic current           # what is applied
uv run alembic history           # revision graph
uv run alembic downgrade base    # revert to an empty database
uv run alembic upgrade head --sql   # render SQL, no database needed
```

The revision graph has four revisions, and the head is `0004_passage_source_spans`:

| Revision                          | Effect                                          |
| --------------------------------- | ----------------------------------------------- |
| `0001_foundation_baseline`        | Creates `alembic_version`; intentionally empty so the baseline cannot pre-empt the scientific schema |
| `0002_canonical_document_model`   | Canonical documents, versions and acquisition provenance |
| `0003_jats_source_structure`      | Paragraphs and their JATS source anchors          |
| `0004_passage_source_spans`       | Passages and their exact paragraph character ranges |

`0001_foundation_baseline` still asserts what it was built to assert: applying
*only* that revision to an empty database succeeds and leaves `public` containing
no table other than `alembic_version`. An integration test proves it directly and
also proves it is reachable from the head. The scientific schema is owned by the
later revisions, and every one of them is exercised against a real database.

## Tests

```powershell
uv run pytest                 # unit suite, no infrastructure, ~0.5s
uv run pytest -m integration  # requires the live stack
```

`-m 'not integration'` is set in `addopts`, so the default run cannot be broken
by an absent database. The integration suite reads the same `.env` as the
application. An explicit `uv run pytest -m integration` run fails loudly with
an actionable message when configuration is missing or invalid — it can never
degrade into a silent skip. Unit-level regression tests
(`tests/unit/test_integration_contract.py`) prove both halves of that contract
in a subprocess, with no live services required.

`pytest-asyncio` is **not** included: this slice is fully synchronous (readiness
runs in FastAPI's threadpool), so the plugin would be an unused dependency.

Neither suite touches a Hugging Face Hub endpoint and neither needs a GPU. The
embedding adapter's protocol tests run against `httpx2.MockTransport` over a
scripted TEI 1.9.x server, so `POST /embed` bodies, retry behaviour and
mid-run identity drift are all asserted with no socket open and no model
downloaded. The benchmark suite is the same in kind: the retrieval metrics are
checked against exact brute-force results on small corpora, the Matryoshka gate
against constructed vectors, the bundle verifier against bundles a test assembles
and then corrupts one field at a time, and the Colab notebook against its
committed JSON. The PostgreSQL and OpenSearch integration suite is unchanged and
remains deterministic.

## Quality gates

```powershell
uv lock --check             # uv.lock is current for pyproject.toml
uv run ruff format --check .
uv run ruff check .
uv run pyright              # typeCheckingMode = "strict"
uv run pytest
```

Ruff owns formatting *and* linting, and Pyright runs in strict mode with
`reportUnnecessaryTypeIgnoreComment` and `reportPrivateUsage` promoted to
errors. The whole suite — `src`, `tests` and `alembic` — is covered; only
`tests/**` relaxes `assert`-style rules and `alembic/versions/*.py` tolerates the
unused `op` / `sa` imports that Alembic's own revision template emits.

Two exclusions are deliberate and both are narrow. `notebooks/` is outside ruff,
because a notebook cell is meant to print; its invariants are asserted by parsing
the committed JSON in `tests/unit/test_benchmark_colab.py` instead.
`src/dynamisrag/benchmark/runner.py` is outside Pyright, because it is the only
module that imports `torch` and `sentence_transformers`, and neither is installed
in CI by design — the type checker would otherwise report the absence of a
dependency the repository refuses to depend on. Nothing else is excluded.

## Running the server

```powershell
uv run python -m dynamisrag                                    # local development
uv run uvicorn --factory dynamisrag.application:create_app \
  --host 0.0.0.0 --port 8000                                   # production-shaped
```

`create_app()` performs no I/O — the SQLAlchemy engine connects lazily and the
OpenSearch client opens no socket until the first probe — so the application
can be constructed before any infrastructure exists. `create_app` is a factory
rather than a module-level singleton, and holds no mutable global state, which
is what lets the test suite run several differently-configured apps in one
process.

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request:

- **quality** — matrix over `ubuntu-latest` and `windows-latest`. The Windows
  leg is not decoration: this project is developed natively on Windows, and
  path handling, `pathlib` resolution and subprocess-free configuration are
  exactly the things that break on one platform only.
- **integration** — starts this repository's own `compose.yaml` via Docker
  Compose, applies migrations to an empty database, exercises
  `downgrade base` / `upgrade head` again to prove reversibility, runs the
  integration suite, and boots the real uvicorn server to exercise
  `/healthz` and `/readyz` over HTTP.

## Scope

Implemented here: repository layout, dependency lock, strict configuration,
FastAPI shell, liveness and readiness, PostgreSQL 18 connectivity, OpenSearch 3
connectivity, Alembic baseline, JATS canonical import, structure-aware chunking,
the versioned deterministic passage projection (`passage-index-v1`), versioned BM25
retrieval, the vector-capable projection (`passage-index-v2`, Lucene HNSW), the
model-agnostic embedding boundary with its TEI adapter
(`EmbeddingProvider`, `passage-embeddings-v1`), the RES-138 retrieval benchmark
harness (frozen contracts, exact scoring, Matryoshka gate, content-addressed
artifacts, resumable run manifest, local bundle verifier, Colab notebook — with no
result and no default), and the lint/type/test/CI gates.

Not implemented here, by design: which embedding model and dimension are **best**
(the harness for that question exists but has not been run through a corpus pass),
a production ANN retrieval API, BM25+dense fusion, reranking, generation, and
agents. Those belong to later issues in this milestone.

What that means concretely. Embedding *generation* is implemented and
reproducible: the adapter calls a TEI server you point it at, proves which model
and which immutable weights answered, verifies each passage against its own
content digest, validates every returned vector without repairing it, and hands
you a deterministic passage-to-embedding manifest whose SHA-256 names the observed
runtime, the attested startup policy and the requested generation semantics
together. What does **not** exist is a *choice* — no model is selected, no
dimension is chosen, no default is configured, and nothing is published to
OpenSearch. A `passage-index-v2` index therefore still holds vectors **supplied by
the caller**: this repository does not index the vectors it generates, and the
vectors used in its tests are labelled synthetic test values. BM25 search serves
both revisions and never selects a vector.

## Copyright

This repository is public-source: the code is publicly viewable, but no
software licence is granted. Copyright © 2026 Julio Rodriguez. All rights
reserved. See the notice at the top of this README.

The all-rights-reserved notice applies to original DynamisRAG code and
documentation only. Third-party materials retain their own copyright and
license terms. See `tests/fixtures/README.md` for notices applicable to test
fixtures.
