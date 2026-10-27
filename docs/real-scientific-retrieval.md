# First real scientific retrieval

Executed locally on 2026-10-27 from reviewed Qdrant commit
`54c2c4c2078621b8f0106fee00bbb54017496b05`, on the separate
`feat/real-scientific-retrieval` branch. PR #20's branch is unchanged.
This is a three-paper functional demonstration, not an effectiveness benchmark.
No OpenSearch route, production configuration, or frozen RES-142 artifact changed.

Three documents, three document versions, 117 canonical paragraphs, **75 passages,
75 real 512-dimensional vectors**. All 75 passages passed independent PostgreSQL
identity, content-hash, citation-identifier and source-span reconstruction checks.
Reparsing each retained original JATS file reproduced its PostgreSQL paragraph
anchors/text and document content fingerprint. Repeated ingestion created no new
versions or passages; see `import_created`/`passages_created` in the paper receipt.

Sources and rights:

* [Impact of Starting Knee Flexion Angle on Muscle Activity and Performance during
  Plyometrics without Jumping](https://pmc.ncbi.nlm.nih.gov/articles/PMC10780924/):
  25 passages; JATS explicitly links CC BY 4.0.
* [Effect of plyometric versus complex training on core strength, lower limb, and
  upper limb power in male cricketers: a randomized controlled trial](https://pmc.ncbi.nlm.nih.gov/articles/PMC10680285/):
  18 passages; JATS explicitly links CC BY 4.0. Its third-party-material caveat is
  retained verbatim. No image assets were acquired or embedded.
* [Plyometric Training Favors Optimizing Muscle–Tendon Behavior during Depth
  Jumping](https://pmc.ncbi.nlm.nih.gov/articles/PMC5263151/): 32 passages; JATS
  explicitly grants CC BY, without stating a license version or URI. No license
  version was invented. The [publisher notice](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2017.00016/full)
  also identifies CC BY.

[Paper receipts](evidence/real-science-v1/papers.json) retain contributors,
copyright/license notices, source URLs, retrieval times, original byte hashes,
canonical version keys and chunking-manifest hashes. Exact original XML bytes
remain in `.dynamisrag/real-science-v1/sources/`, outside Git. Canonical text is
normalized by the existing JATS parser and split by the existing deterministic
chunker; original source bytes are unchanged.

The earlier documented artifact search was repeated in this checkout, `.tmp`,
reports and relevant `E:\Data` Models/Artifacts/Runs/Backups/Exports/Staging/
Downloads/Datasets locations. No compatible numerical embedding manifest was
found; unrelated biomechanical NPZ data and synthetic fixtures were not used.

The existing TEI adapter worked without a new inference provider. The optional
ONNX backend logged missing `onnx/model.onnx`; TEI then loaded the original
safetensors successfully through **Candle CPU/MKL**. This was backend fallback
inside TEI, not an embedding-model or retrieval-backend substitution.

Observed model revision: `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`;
TEI 1.9.4 build `e80ef225ed0e6cb1717ce632a6a84b6cf211bb67`;
float32; last-token pooling; 512 dimensions; normalization enabled; truncation
disabled on every request. The pinned model's prompt table was inspected:
`document` is empty; `query` is exactly
`Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:`.
There is no extra space after the colon. No default prompt or dense-path override.

The CPU deployment fingerprint separately binds device, image digest, platform
and MKL/OMP thread setting. Docker enforces four CPU cores and 8 GiB memory;
MKL/OMP use four threads, the image's Rayon default is eight, and tokenization
uses one worker. Batch size is one; served maximum input length is 2,048 tokens.
The operator limits the corpus to 150 passages and checks its 30-minute deadline
before every batch, reserving the per-request 120-second timeout. Inference for
75 documents plus one query took **217.86 seconds**. Sampled container memory
was 2.463 GiB; this is not a measured peak. CPU results do not qualify equivalence
to the earlier GPU benchmark. No training, alternative model, GPU, reranker or
LLM answer generation was used. An isolated native CPU Python environment was
prepared during compatibility investigation but was not used for inference.

Retained numerical artifacts (exact bytes, including all float components):

* [Document manifest](evidence/real-science-v1/embedding-manifest.json), SHA-256
  `46f96db3a02cf61374003475e3f006258ecbd790a4a78bd858b305c25f5104ff`.
* [Query manifest](evidence/real-science-v1/query-manifest.json), SHA-256
  `4ac70bedcc23ffc2d3acfcba46ffb42ad1b68de55e0cbad782678f1f4b83fb29`.
* [Original receipt](evidence/real-science-v1/receipt.json), independently recorded
  SHA-256 **`8ede85766f656acccf7fd44b9c3fde627bd18289b74fe375dbf895d02ff4d240`**.
  Retrieval requires this original digest, then checks each manifest against the
  receipt before writing any Qdrant data. The digest is also fixed in an offline
  artifact-integrity test. Git EOL conversion is disabled for these JSON files.

Document generation fingerprint:
`8438d37d8e8c8f975f62cbc6f6c644abc27304b18b7a330d9157a70f21c9c1f7`.
Query generation fingerprint:
`94245fa64ff582ab1428c7aea38387e9091a23e538c96d7fbf7b48e1755da8fd`.

Separate qualified collection:
`qdrant-passages-v1-5a20d2780fe6cd0ce0155ed7535c6507ec452f9ad4ae14cbd5990e13d7e8a709`.
Its alias is `dynamisrag-real-science-v1`. Existing five synthetic collections
were preserved; complete payload/vector hashes before and after appear in the
[observed retrieval receipt](evidence/real-science-v1/retrieval.json).
The small real collection uses exact scans under the existing Qdrant threshold;
this does not qualify large-scale HNSW performance.

## PowerShell commands and observed output

Run from `E:\Data\Projects\DynamisRAG`, using the existing `.env`.
The executed Docker launch used `:cpu-1.9.4`; inspection verified the digest below.
The equivalent command here pins both tag and observed digest. The initial
unsupported `--max-input-length` flag was removed; `--max-batch-tokens 2048`
produced the observed `/info` maximum input length of 2048.

```powershell
docker compose up -d --wait postgres qdrant
uv run alembic upgrade head
docker run -d --name dynamisrag-real-qwen-cpu --cpus 4 --memory 8g `
  -p 127.0.0.1:8081:80 -v dynamisrag-real-qwen-cache:/data `
  -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 `
  ghcr.io/huggingface/text-embeddings-inference:cpu-1.9.4@sha256:2538ea1c9640d3763b15af668039d24172d063b42337b0c27796fc2be180c78d `
  --model-id Qwen/Qwen3-Embedding-0.6B `
  --revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 --dtype float32 `
  --max-batch-tokens 2048 --max-client-batch-size 1 --tokenization-workers 1
docker logs --tail 15 dynamisrag-real-qwen-cpu
# Continue after the log says Ready. Startup/download used about three minutes.
Invoke-RestMethod http://127.0.0.1:8081/info | ConvertTo-Json -Depth 8
uv run python scripts/real_scientific_retrieval.py ingest
uv run python scripts/real_scientific_retrieval.py embed
# embed refuses to overwrite an existing receipted artifact.
uv run python scripts/real_scientific_retrieval.py retrieve --receipt-sha256 8ede85766f656acccf7fd44b9c3fde627bd18289b74fe375dbf895d02ff4d240
```

`ingest` and `retrieve` are idempotent. `retrieve` uses the retained query vector
and requires no running inference server. After a fresh ingestion with identical
source versions, existing numerical artifacts can be restored without inference:

```powershell
Copy-Item docs/evidence/real-science-v1/embedding-manifest.json,docs/evidence/real-science-v1/query-manifest.json,docs/evidence/real-science-v1/receipt.json .dynamisrag/real-science-v1/
uv run python scripts/real_scientific_retrieval.py retrieve --receipt-sha256 8ede85766f656acccf7fd44b9c3fde627bd18289b74fe375dbf895d02ff4d240
```

Upstream XML changes produce different canonical versions; old vectors then fail
qualification rather than being attached to new content. Keep the original local
source store for exact replay.

Observed embedding stdout:

```text
document 75 512 46f96db3a02cf61374003475e3f006258ecbd790a4a78bd858b305c25f5104ff
query 1 512 4ac70bedcc23ffc2d3acfcba46ffb42ad1b68de55e0cbad782678f1f4b83fb29
```

Observed retrieval: question **How does plyometric training change muscle and
tendon behavior during depth jumping?** Raw top results below use exact passage
keys; all cite PMC5263151, DOI `10.3389/fphys.2017.00016`.

```text
papers=3 passages=75 vectors=75 independently_verified=75
bm25_ms=15.04300000033254 dense_ms=12.130899999647227 hybrid_ms=23.51379999981873
bm25 1 04d4565cd5a7cd106944580d4fe621057f1b7c32c870f638209e976303725dff 13.638429
bm25 2 9cbd015305b222b8ed8c37eb3b0ed0bb18b9847e13aaec41ae1f7410e9be7abe 12.920453
bm25 3 52dd7ddbf821c16d5ee704ec0e934bb87a0e5824bb6d7b69d08083f91c9b053d 11.753768
dense 1 9cbd015305b222b8ed8c37eb3b0ed0bb18b9847e13aaec41ae1f7410e9be7abe 0.794664
dense 2 04d4565cd5a7cd106944580d4fe621057f1b7c32c870f638209e976303725dff 0.76765484
dense 3 561ab9ffb8dd9064d43e03881794dbaedea6a3dc909dcc95a06f3b74ae141baf 0.75050867
rrf 1 04d4565cd5a7cd106944580d4fe621057f1b7c32c870f638209e976303725dff 0.03252247488101534
rrf 2 9cbd015305b222b8ed8c37eb3b0ed0bb18b9847e13aaec41ae1f7410e9be7abe 0.03252247488101534
rrf 3 561ab9ffb8dd9064d43e03881794dbaedea6a3dc909dcc95a06f3b74ae141baf 0.03149801587301587
```

These are observed single-request client latencies, excluding query inference,
qualification, source verification and startup. They are not percentiles or a
comparison with GPU/OpenSearch. RRF uses application code, k=60, lane window 50.
The first two scores tie (ranks 1/2 exchanged); passage-key tie-breaking selects
`04d456…` first. It is an introduction/hypothesis passage, not an experimental
conclusion. The dense leader includes the abstract plus adjacent front matter:
the existing parser/chunker behavior is retained and is a visible corpus-quality
limitation. No answer was generated from these results.

The first fused passage maps to PostgreSQL paragraph key
`eff1b9923bce97683391c6f7e7344cd40d37ba67cee485287da9cb453d98302c`,
JATS anchor `jats:/body[1]/sec[1]/p[3]`, characters **[0, 883)**. Its exact text,
all returned citation/spans, raw top-five lanes, RRF contributions, all-source
reparse checks and existing-collection hashes are in `retrieval.json`.

```powershell
uv run pytest tests/unit/test_real_science_evidence.py tests/unit/test_real_science_operator.py -q --basetemp .tmp/real-evidence-tests-01
uv run pytest tests/unit/test_real_science_operator.py tests/unit/test_qdrant.py tests/unit/test_qdrant_review.py tests/unit/test_passage_embedding_manifest.py tests/unit/test_jats_parser.py -q --basetemp .tmp/real-science-focused-final
uv run pytest tests/integration/test_qdrant_live.py -m integration -q --basetemp .tmp/real-qdrant-live-01
uv run pytest --basetemp .tmp/real-science-full-unit-01 -q
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

Observed: 10 artifact/operator tests passed; 261 focused tests passed; 15 live
Qdrant tests passed. The full suite passed 2,209 tests with 181 integration tests
deselected before the final two fragment cases and artifact-integrity test were
added; those three subsequently passed in the focused runs. Ruff and strict
Pyright passed. Full live PostgreSQL/OpenSearch integration runs in isolated CI;
its empty-database assumptions make the populated canonical database unsuitable
as a test fixture.

GitHub CI was dispatched on the separate branch with:

```powershell
git push -u origin feat/real-scientific-retrieval
gh workflow run ci.yml --ref feat/real-scientific-retrieval
```

Run: [40602086964](https://github.com/Litju/DynamisRAG/actions/runs/40602086964).
After stopping the CPU server, the final local replay again verified all 75
passages and all three sources; observed BM25/dense/hybrid latencies were
12.0017 / 10.4669 / 21.4377 ms. PostgreSQL and Qdrant remain running.
