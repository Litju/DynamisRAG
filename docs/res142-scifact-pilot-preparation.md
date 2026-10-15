# RES-142 SciFact pilot: local preparation

Status on 2026-10-15: B1 is `PASS_WITH_EVIDENCE`. The adapter is qualified by deterministic local tests against `HybridRetrievalResponse` and RES-140 contracts. A real SciFact input slice and the frozen 20-query selection are materialized locally. No BM25/dense/RRF retrieval or reranking was run, and no scientific text was uploaded.

## Frozen pilot contract

- Source is the registered `beir.scifact.test` split: 5,183 abstracts, 300 queries, and 339 binary link judgments. BEIR test corresponds to SciFact `claims_dev`; it is not the unlabeled `claims_test` or claim-veracity task.
- Select 20 IDs by sorting `(SHA256(b"res142-sf-pilot-v1\0" + query_id_utf8), query_id)` over all 300 IDs and taking the first 20. This selection uses no qrel values or retrieval output. The result stays frozen in selection order; RES-140 dataset query rows stay query-ID ascending.
- The treatment consumes saved BM25 + dense + `rrf-v1` responses requested with `limit=50`. It does not recompute retrieval. For every candidate, preserve the passage key, document/version identity, text hash, source provenance, original BM25 and dense ranks/raw scores, RRF rank/score, projection, and embedding profile.
- Reuse one complete projection-wide `IrPassageMapping` for the frozen baseline and Qwen run. Build document runs with RES-140 `document_run_from_passages`, which deduplicates passages by document and enforces query coverage and short-result exhaustion rules.
- Evaluate both runs at depth 50 with RES-140 `evaluate_ir_run`, then use `compare_ir_evaluations` for paired per-query and aggregate nDCG@10, Recall@10, MAP, and MRR deltas. Both sides must share the exact pilot dataset/qrels digest, projection, mapping, depth, metric policy, and scoring engine. Keep qrels and the full corpus local.

## Local source and query selection

RES-141 materialized the official registered BEIR archive at `.tmp/res142-scifact-beir-test-20261015/`. The archive size is 2,816,079 bytes and its SHA-256 matches the registry pin `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165`. The slice verified against the registered source and its manifest digest; verification reported `artifact-provenance-qualified` with both source identity and derived artifacts authenticated.

| Local artifact | SHA-256 |
|---|---|
| Source archive | `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165` |
| RES-141 materialized manifest | `13346b186c531fa7dfc6e4fbb9ed3606de4e1ad10ae24564ac742ab7ee37aa15` |
| RES-140 full test dataset | `4e7123751dc926ebd35c4f2f681565b7dfa0fed405bdf7f762ca85f1f46240d4` |
| Frozen pilot selection file | `9448b2cccf9d5755c8029b70f73c83da924a6b16f68779a5dddaf0b6ddd67aa6` |
| Ordered selected-query ID list | `6f84f7bb46976a5baafbc5d9be958f300dde35ddc74a23c51940f46045f5db3c` |
| Pilot dataset with original qrels | `70c07ebaaac9d7244540c8a92e8605df9d2dbaaa7002b4e1021a7962800784d7` |

The selection file is `.tmp/res142-scifact-beir-test-20261015/pilot-query-selection.json`. It stores the 20 IDs and hashes, not query text. No candidate snapshot, baseline run, or mapping hash exists yet; those inputs cannot be fabricated from the selection artifact.

## Adapter qualification

[`src/dynamisrag/ir/qwen3_reranker.py`](../src/dynamisrag/ir/qwen3_reranker.py) consumes the concrete frozen `HybridRetrievalResponse` type. Before scoring, it checks query and lane identity, the shared projection/index metadata, contiguous lane ranks, candidate membership, raw lane scores, and RRF contributions against the saved response. It submits only the fused candidates in sequential batches of 8 to loopback `127.0.0.1:8000/score`; HTTP errors and timeouts propagate without retry or fallback.

`SciFactCandidateSnapshot` seals the pilot dataset, projection and full mapping hashes, selected query IDs, candidate text hashes, full candidate provenance, original lane ranks/scores, and each frozen response digest. `frozen_hybrid_passage_hits` preserves original RRF ranks/scores for the baseline. The Qwen run path requires all 20 query results and verifies that every reranked passage and original identity belongs to the sealed candidate snapshot before calling RES-140 document collapse.

The Qwen treatment is pinned to `Qwen/Qwen3-Reranker-0.6B` revision `e61197ed45024b0ed8a2d74b80b4d909f1255473`, the same tokenizer revision, B1-verified template SHA-256 `bf7b104801a4e4ad6a147d732e2d5a837a36f708cdbd1b871b864979d3e10acd`, BF16, and the fixed claim-retrieval instruction. The model revision, template, instruction, batch size, and snapshot/map digests are bound into `IrExperimentConfig`.

Qualification record: the 17 focused adapter tests pass; full `ruff check` passes; repository-wide strict Pyright reports zero errors; the non-integration suite passes 2,286 tests; the live PostgreSQL/OpenSearch integration suite passes 166 tests. These checks qualify the adapter and contract integration only; they do not claim real SciFact retrieval or model inference.

## Rights and transfer status

The RES-141 source registry accepts the SciFact source for local evaluation and records redistribution as `UNVERIFIED`. SciFact's [official license notice](https://github.com/allenai/scifact/blob/master/LICENSE.md) says claims and evidence annotations use [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and corpus abstracts are from S2ORC under [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/). The pinned [BEIR SciFact dataset card](https://huggingface.co/datasets/BeIR/scifact/blob/b3b5335604bf5ee3c4447671af975ea25143d4f5/README.md) separately labels the packaged dataset CC BY-SA 4.0, while BEIR's own [dataset notice](https://github.com/beir-cellar/beir/wiki/Datasets-available) says BEIR does not warrant users' rights and leaves license checks to users.

Attribution must identify SciFact (Wadden et al., 2020), BEIR (Thakur et al., 2021), and Semantic Scholar/S2ORC; link the applicable licenses; retain supplied notices; and state any changes. CC BY 4.0 requires appropriate credit, a license link, and an indication of changes. ODC-By 1.0 requires a copy or URI of the license and intact notices when publicly conveying a database or substantial part, and explicitly does not clear independent rights in individual content. The package-level CC BY-SA declaration and component-level notices have not been reconciled for this exact archive and derived candidate texts.

**The proposed Colab transfer is not cleared.** The registry still says redistribution is unverified, the package declarations conflict, and the BEIR notice disclaims rights clearance. Keep all real scientific text local until the component and package scope has been reconciled and a separate authorization names the exact input SHA and destination. B1's Drive authorization covers synthetic-only evidence and does not authorize SciFact upload.

## Exact input blockers

The archive and deterministic pilot selection are ready. The local OpenSearch alias `dynamisrag-passages` currently has **zero physical targets**. The local configuration has no TEI endpoint, expected embedding model ID, or expected immutable model revision. The RES-141 slice does not create the application's SciFact corpus/projection or run the BM25+dense retrieval service. No real top-50 snapshots, complete physical-projection mapping, or baseline run are present in the checkout or local execution area.

Therefore the input package is `BLOCKED_INPUT`: the exact missing runtime dependencies are the indexed SciFact passage projection under the configured alias and the pinned Qwen3-Embedding query service; the exact missing evidence artifacts are the 20 saved hybrid responses, projection-wide passage mapping, and matching baseline `IrRun`. Until those exist, no candidate identities, baseline hashes, or scientific metrics can be sealed.

## B2 output and GPU plan

The review package must contain the RES-141 source/rights receipt, pilot dataset and original qrels, the frozen 20-query top-50 response snapshot, the complete `IrPassageMapping`, immutable baseline `IrRun`, Qwen model/config identity, and an input manifest whose SHA is independently verifiable. B2 produces per-query and aggregate nDCG@10, Recall@10, MAP, and MRR for both runs, paired deltas, and source/candidate/model provenance. Text-bearing inputs and qrels remain local until transfer approval; outputs contain IDs, ranks, scores, metrics, and hashes.

If the inputs and transfer rights are later cleared, the proposed single supervised run is G4 / Colab runtime 2026.04, the B1-verified vLLM environment and immutable model revision, at most 1,000 pairs (20 × 50), batches of 8, one request at a time, one attempt, no fallback, and a 25-minute connected-runtime ceiling. At the last observed 8.9 CU/hour, 25 minutes projects to **3.71 CU**. Inspect live rate and balance before connecting and set a separately approved CU cap; the projection is not a platform hard cap. Do not allocate Colab or infer until the rights receipt, exact candidate-bundle SHA, and separate pilot authorization are approved.
