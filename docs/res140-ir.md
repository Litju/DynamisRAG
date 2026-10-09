# RES-140 canonical IR evaluation

RES-140 scores ranked document runs independently of the production retrieval
service and the frozen RES-138 model-selection harness. It does not query a
database, OpenSearch, TEI, a hosted model or a GPU. Runs are supplied as sealed
canonical JSON, so evaluation can be repeated on an offline workstation.

## Windows PowerShell quickstart

Install the locked CPU-only dependencies from the repository root:

```powershell
uv sync --locked
```

The checked-in input fixture is synthetic. Score and verify it with its pinned
run identity:

```powershell
$runSha256 = "b85b7c334d630ad5b9f7489679a56a257c0038d451d99ea76ecfa464718c714e"
uv run dynamisrag ir score `
  --inputs tests/fixtures/ir-res140 `
  --run-sha256 $runSha256 `
  --out .tmp/ir-res140-scored
uv run dynamisrag ir verify .tmp/ir-res140-scored --run-sha256 $runSha256
```

The output directory must not already exist. The command writes JSON, TREC and
Parquet result files together and refuses to replace an earlier result. To
compare two verified bundles, supply each sealed run identity:

```powershell
$diff = uv run dynamisrag ir compare .tmp/baseline .tmp/candidate `
  --baseline-run-sha256 $baselineRunSha256 `
  --candidate-run-sha256 $candidateRunSha256 `
  --out .tmp/ir-comparison | ConvertFrom-Json
uv run dynamisrag ir verify-diff .tmp/ir-comparison `
  --baseline .tmp/baseline `
  --baseline-run-sha256 $baselineRunSha256 `
  --candidate .tmp/candidate `
  --candidate-run-sha256 $candidateRunSha256 `
  --comparison-sha256 $diff.comparison_sha256
```

## Sealed run inputs

`ir score` accepts a directory containing exactly these canonical JSON files:

| File | Meaning |
| --- | --- |
| `dataset.json` | Stable source and corpus identity, the complete query universe and original qrels |
| `config.json` | Exact code SHA, retrieval revision, projection SHA, semantic parameters and metric policy |
| `run.json` | Complete query universe, document-ranked hits, raw lane scores and declared retrieval depth |
| `passage-mapping.json` | Audited passage-to-document/version mapping; empty for an already document-level run |

The caller must provide the expected run SHA-256 out of band. The scorer checks
that the run binds to the selected dataset and configuration, that every query
belongs to the dataset, and that the mapping identity agrees with the run.
Canonical JSON bytes use UTF-8, sorted keys and one final LF. TREC identifiers
are stable, whitespace-free tokens.

The bundle verifier checks a closed file inventory, every file digest, all typed
identities and cross-file references. It regenerates the scores and JSON, TREC
and Parquet outputs, reads the Parquet tables back, checks row equivalence, and
refuses tampered or swapped artifacts. Parquet schemas carry their revision and
PyArrow version. `pyarrow==25.0.1` and `ir-measures==0.4.3` are pinned in the
non-production benchmark dependency group and `uv.lock`.

The scored bundle contains `manifest.json`, `config.json`, `dataset.json`,
`passage-mapping.json`, `qrels.trec`, `run.json`, `run.trec`, and
`evaluation.json`, plus each table in JSON and Parquet:
`per-query.json`/`per-query.parquet`,
`aggregate.json`/`aggregate.parquet`, and
`ranked-run.json`/`ranked-run.parquet`. Rows use fixed schemas and stable
query/rank/metric ordering.

## Frozen metric policy

`ir-metric-policy-v1` is part of the configuration and evaluation identities.
The locked `ir_measures` provider computes these measures:

| Output | Definition |
| --- | --- |
| `nDCG@10` | Graded nDCG with linear relevance gain, logarithmic rank discount and a cutoff of 10 |
| `Recall@10` | Relevant when the original judgment is at least 1; numerator is relevant documents retrieved by rank 10 |
| `MAP` | `AP(rel=1)` over the complete declared run depth |
| `MRR` | `RR(rel=1)` over the complete declared run depth |

Negative source judgments are preserved in `dataset.json` and `qrels.trec`.
For metric computation only, negative relevance is scored as zero; positive
graded relevance is unchanged. A missing qrel remains unjudged, is counted in
the per-query row, and contributes zero gain. A query with no qrels or no
positive qrels receives zero for each metric and remains in the macro-average
denominator. The aggregate table reports each metric's summed numerator,
query denominator, count of queries without qrels and count of zero-positive
queries. Source judgments are never rewritten in the sealed dataset.

The retrieval depth is an explicit part of each run identity. `Recall@10` is
refused when a run declares a depth below 10. The existing hybrid retrieval
window is 50 candidates, so this policy reports `Recall@10`; it does not claim
`Recall@100` from that window. To change a cutoff or gain rule, define a new
metric-policy revision and rerun scoring.

Raw BM25, dense and RRF lane scores are retained for inspection but do not
control metric ordering. `ir_measures` receives `-rank`, which preserves the
explicit one-based rank even when raw scores tie or use different scales.

## Passage and document identity

Qrels name documents. Production retrieval returns passages, so a passage key
is never used as a qrel document ID. `IrPassageMapping` records each returned
passage key, the evaluation document ID and its document version ID. The full
mapping for the submitted run is included in the bundle and its SHA is tied to
mapped runs. Repeated passages
for one document collapse to the first passage within the declared window;
rank ties break by passage key, independent of lane score. A single evaluation
document mapping to conflicting versions is rejected, as is any returned
passage without a mapping.

To adapt an existing `GET /retrieve` response, map its `FusedHit` passage and
provenance fields to the IR input types. `qrel_id_by_canonical_key` below is a
caller-owned mapping to the document IDs used in that dataset's qrels:

```python
mapping = IrPassageMapping(tuple(sorted(
    (
        IrPassageMapEntry(
            hit.passage_key,
            qrel_id_by_canonical_key[hit.provenance.document_canonical_key],
            hit.provenance.document_version_key,
        )
        for hit in response.fusion.hits
    ),
    key=lambda entry: entry.passage_id,
)))
run = document_run_from_passages(
    dataset=dataset,
    config=config,
    passage_mapping=mapping,
    hits=tuple(
        IrPassageHit(query_id, hit.passage_key, hit.final_rank, hit.rrf_score)
        for hit in response.fusion.hits
    ),
    evaluation_depth=50,
)
```

BM25 `SearchHit` and dense candidates can be converted from their explicit
rank and raw score fields the same way. The mapping must cover every hit in
the submitted window.

Comparisons require the same corpus/query/qrel dataset SHA, projection
snapshot, passage mapping, evaluation depth, metric-policy SHA and scoring
engine. Candidate code and retrieval parameters may differ; both candidate
identities are retained in the comparison artifact.

## Dataset identity and rights

The dataset author supplies `source_id`, immutable `source_revision`, exact
`corpus_sha256`, query IDs/text and qrel document IDs/relevance values. The
scorer checks their shape and binds them into the dataset SHA; it does not
download a dataset, infer source IDs, re-hash a corpus it was not given, or
determine reuse rights. The dataset author is responsible for permission,
license terms, attribution and for ensuring qrel document IDs refer to the
document-level corpus identity. Corpora and source documents are not copied
into the result bundle; only the corpus hash, queries, judgments and ranked
document IDs are included. The checked-in fixture contains synthetic IDs,
queries and judgments and provides no retrieval-quality or model-qualification
evidence.

`config.json` is limited to stable scientific settings. Host names, URLs,
paths, credential fields, timestamps, timings and backend error/response fields
are rejected when supplied under their standard parameter names; do not place
environment details or service response prose in semantic parameters.
