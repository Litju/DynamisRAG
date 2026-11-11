# Architecture

DynamisRAG separates canonical scientific state from rebuildable retrieval
projections. PostgreSQL owns the canonical record; OpenSearch and Qdrant are
caches that can be rebuilt from it. The browser talks only to the FastAPI
application, and the scientific answer path stays local.

## System context

```mermaid
flowchart LR
    subgraph Sources
        JATS["Licensed JATS articles"]
    end

    subgraph Ingestion
        Parser["JATS parser"]
        Chunker["Structure-aware chunker"]
    end

    PG[("PostgreSQL\ncanonical authority")]

    subgraph Projections
        OS[("OpenSearch\nBM25 + vector indexes")]
        QD[("Qdrant\ndense / hybrid index")]
    end

    subgraph Serving
        API["FastAPI"]
        UI["Research UI\n/research"]
        TEI["TEI\nQwen3 Embedding 0.6B"]
        OLL["Ollama\nQwen3 4B"]
    end

    JATS --> Parser --> Chunker --> PG
    PG -->|"projection"| OS
    PG -->|"projection + vectors"| QD

    UI --> API
    API -->|"/search, /retrieve"| OS
    API -->|"scientific retrieval"| QD
    API -->|"query embedding"| TEI
    API -->|"extractive answer"| OLL
    API -->|"canonical evidence"| PG
```

The generic search and retrieval endpoints use the OpenSearch projection. The
scientific research path uses the Qdrant projection for BM25, dense candidates
and reciprocal-rank fusion, then reads and verifies the cited rows from
PostgreSQL. TEI embeds the query; Ollama produces only an extractive claim or an
explicit abstention.

## Ingestion and projection

```mermaid
flowchart LR
    A["Licensed JATS source bytes"] --> B["JATS parser"]
    B --> C["Structure-aware chunker"]
    C --> D[("PostgreSQL\nsource artifacts, documents,\nversions, paragraphs, passages")]
    D --> E["Projection builder"]
    E --> F[("OpenSearch\npassage-index-v1\npassage-index-v2")]
    D --> G["Verified embedding manifests"]
    G --> H[("Qdrant\nimmutable passage collection")]
```

Ingestion is deterministic: the same document version and chunker revision
produce the same passages and `passage_key`s. Source bytes, canonical rows and
projection identities are content-addressed. OpenSearch and Qdrant can be
deleted and rebuilt without changing the canonical record.

## Scientific answer request

```mermaid
sequenceDiagram
    autonumber
    participant B as Browser
    participant A as FastAPI
    participant S as Scientific answer service
    participant T as TEI
    participant Q as Qdrant
    participant P as PostgreSQL
    participant G as Ollama

    B->>A: POST /answer/scientific
    A->>S: question + optional document_version_key
    S->>T: embed query
    T-->>S: query vector
    S->>Q: BM25 + dense candidates
    Q-->>S: ranked passages
    S->>S: reciprocal-rank fusion + evidence selection
    S->>P: read canonical rows
    S->>S: verify passages and source spans
    S->>G: extractive answer request
    G-->>S: draft claim or abstention
    S->>P: re-read and verify evidence
    S->>S: validate exact quote and citation
    S-->>A: supported claim or insufficient_evidence
    A-->>B: answer + DOI + canonical coordinates
```

A structurally invalid claim is rejected, and an unsupported question returns
`insufficient_evidence` instead of an answer. Citation checks prove that a
claim is copied from the cited span; they do not prove scientific entailment or
truth. See [testing and evidence boundaries](testing-and-evidence.md) for what
the recorded acceptance does and does not establish.

## Conversational scientific research agent (`/api/v2`)

For multi-turn scientific investigation, DynamisRAG exposes a stateful
10-node LangGraph orchestrator under `/api/v2` (see
[RES-637 conversational agent architecture](res637-conversational-agent-architecture.md)
and [RES-637 model qualification](res637-agent-qualification.md)):

- **Canonical read-only boundary**: all 10 scientific tools execute inside
  `REPEATABLE READ READ ONLY` transactions on canonical PostgreSQL tables while
  reading hybrid retrieval candidates from Qdrant and OpenSearch projections.
- **Isolated operational state**: conversation threads, run records, SSE event
  logs, and LangGraph checkpoints live in an isolated PostgreSQL schema
  (`agent_v2`) and never mutate canonical scientific tables.
- **Independent canonical verification**: provisional model claims are verified
  against `passage_source_span` character coordinates (`MODEL_PROPOSED`
  provenance) before any citation is emitted.

