# RES-142 SciFact pilot: retrieval readiness

Status on 2026-10-16: B1 remains `PASS_WITH_EVIDENCE`. SciFact retrieval integrity fixes are in `79c440f55216b2d32334aa2b4568f25129483544`; canonical qrel identity support is in `b748cfcf3c3f6aff9c8aa167864d80fa34f4be1a`. The latest RES-142 execution direction is followed: embedding, indexing, and retrieval ran on Colab 2026.04 G4. The 20-query retrieval evidence is saved in Drive run `20261016-10bce56f6ea04d949355af34fba2698c`; deterministic validation and RES-140 baseline construction ran locally from those saved records, without retrieval calls. The Colab runtime is disconnected. Qwen reranking has not run.

## Frozen pilot contract

- Source is the registered `beir.scifact.test` split: 5,183 abstracts, 300 queries, and 339 binary link judgments. BEIR test corresponds to SciFact `claims_dev`; it is not the unlabeled `claims_test` or claim-veracity task.
- Select 20 IDs by sorting `(SHA256(b"res142-sf-pilot-v1\0" + query_id_utf8), query_id)` over all 300 IDs and taking the first 20. This selection uses no qrel values or retrieval output. The result stays frozen in selection order; RES-140 dataset query rows stay query-ID ascending.
- The treatment consumes saved BM25 + dense + `rrf-v1` responses requested with `limit=50`. It does not rerun either retrieval lane; validation recomputes only pure fusion. For every candidate, preserve the passage key, document/version identity, text hash, source provenance, original BM25 and dense ranks/raw scores, RRF rank/score, projection, and embedding profile.
- The integrity validator does pure deterministic RRF reconstruction from those saved lanes with the production `reciprocal_rank_fusion`; it makes no OpenSearch, TEI, or ANN call. It requires exact full-prefix identity, rank, score, lane trace, and tie order.
- Reuse one complete projection-wide `IrPassageMapping` for the frozen baseline and Qwen run. Build document runs with RES-140 `document_run_from_passages`, which deduplicates passages by document and enforces query coverage and short-result exhaustion rules.
- Preserve the source BEIR pilot and qrels unchanged. The source qrels use BEIR IDs while the production passage map uses canonical document IDs, so derive a separate evaluation dataset with `canonical_scifact_evaluation_dataset` and seal the complete source-ID to canonical-ID map. Build both RES-140 runs against that same evaluation dataset, projection, passage map, depth, metric policy, and scoring engine. The candidate snapshot continues to bind the original pilot dataset SHA. Qrels were not transferred to Colab.

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

The selection file is `.tmp/res142-scifact-beir-test-20261015/pilot-query-selection.json`. It stores the 20 IDs and hashes, not query text. The exact Colab result archive is `.tmp/res142-colab-res140-evidence.zip` (SHA-256 `4be0b44075836ff68580ec6a6de19b74283b9afa74c152179c3b91bb5d2a258a`). Its `validation/saved-rrf-validation.json` SHA-256 is `76f8ce9daba2171859ea988bef9df843c9cf865979b4876966daff4f58fa8c81`; it validates all 20 responses from saved lane results with `retrieval_calls=0`.

The complete local snapshot and baseline artifacts are in `.tmp/res142-scifact-colab-baseline-b748cfc/`. The original source dataset and selected pilot qrels remain byte-for-byte available beside the canonical-evaluation bundle. The complete source-document identity map has 5,183 rows and was independently regenerated from the input corpus with the production SciFact projection helper; its passage mapping exactly matches the Colab mapping SHA.

| Real run artifact | SHA-256 |
|---|---|
| Colab retrieval summary | `4e2474efd2833f22841b304be110ec172ee439405066b90ef337f3caddcef83c` |
| RRF validation report | `76f8ce9daba2171859ea988bef9df843c9cf865979b4876966daff4f58fa8c81` |
| Complete passage mapping | `f2489a8dd194849921c81fec8fbd2959567913ef1b58bb56e6d27c9af37854ad` |
| Passage-index-v2 projection | `f8483d00a654a1a2c3d5a8d5f712676ba79edc006ae513bf0a3398ce963a0473` |
| Candidate snapshot, 20 queries | `c4320cebfca3350a8c0d2070c6ed3f20a68692cfdc9b3fc4f583b0a92dd5310e` |
| Original full RES-141 dataset | `4e7123751dc926ebd35c4f2f681565b7dfa0fed405bdf7f762ca85f1f46240d4` |
| Original selected pilot with source qrels | `70c07ebaaac9d7244540c8a92e8605df9d2dbaaa7002b4e1021a7962800784d7` |
| Canonical qrel identity map | `c43b1020ddf783edc9759e0b6abd5a250beeb6d5a37ffca8e47e82eb347193ab` |
| Evaluation dataset used by RES-140 | `d73f7463c9338ce8134ddb5ff0bbebaeae8e390edf768e092e803fb961d59f12` |
| Baseline RES-140 run | `e2831be83856c98d752c570aad93d0f4734e20598be7e563b33ba21866dc3453` |
| Baseline bundle manifest | `13661d791a9a49c3313c98c247f78814c1aea074d21d84b1169a0b33ff6ab1fc` |
| Baseline evaluation | `0ffe6d4811ae42b754e9b04ccda8c079e0efbbbc3f60d0d75d11698e5c3a2e66` |
| Prepared Qwen run config | `9d3565ea1fb7ae863cc38b3119a9384a4a0327120b6c5272428450113c65f756` |

RES-140 baseline metrics over all 20 selected queries (24 original binary qrels) are nDCG@10 **0.725411**, Recall@10 **0.862500**, MAP **0.676548**, and MRR **0.688333**. These are first-stage reference values only; no Qwen reranker scores or paired deltas exist yet.

The source-dataset SHA remains `4e712...`; the source pilot SHA remains `70c07...`. The evaluation dataset SHA is different by design because it changes only qrel document IDs through the sealed source-to-canonical map; query IDs, text, judgments, relevance values, and qrel count are preserved. This identity projection is required to join source BEIR qrels to RES-140 document runs correctly.

The local Colab indexing input is `.tmp/res142-scifact-colab-input-20261015/res142-scifact-index-input-v1.zip`, 2,765,915 bytes, SHA-256 `f358111ef72e4c19000f5b27956cd8249980736f2d244884c8cdecfadf41733e`. Its closed inventory contains the 5,183-row abstract corpus, the selected 20 query texts, the selection, source-provenance and notice records; it excludes qrels and the other 280 query texts. The source manifest retains its original `transfer_status=not-cleared`; the adjacent requester-confirmed sidecar authorizes this exact input for this private run.

## Adapter qualification

[`src/dynamisrag/ir/qwen3_reranker.py`](../src/dynamisrag/ir/qwen3_reranker.py) consumes the concrete frozen `HybridRetrievalResponse` type. Before scoring, it rebuilds pure deterministic RRF from the saved BM25 and dense lanes with `limit=50` and requires exact equality with the complete fused trace. It rejects a `limit=10` response, a dropped candidate, wrong order/ties, invalid BM25 lane totals, or mixed physical-index, vector, embedding, BM25, projection, and fusion identity. No OpenSearch or TEI retrieval call is repeated.

Each query digest binds every BM25 and dense lane row, including non-fused rows, with its passage key, rank, raw score, text SHA-256, and complete provenance. `SciFactCandidateSnapshot` binds those response hashes to one homogeneous retrieval identity, the projection, dataset, and complete passage map. `frozen_hybrid_passage_hits` preserves original RRF ranks/scores for the baseline. `QwenScoringCandidate` forms the minimal B2 transfer: query text and the exact 50 fused candidate texts/identities, without qrels, full corpus, or vectors.

The Qwen treatment is pinned to `Qwen/Qwen3-Reranker-0.6B` revision `e61197ed45024b0ed8a2d74b80b4d909f1255473`, the same tokenizer revision, B1-verified template SHA-256 `bf7b104801a4e4ad6a147d732e2d5a837a36f708cdbd1b871b864979d3e10acd`, BF16, and the fixed claim-retrieval instruction. The model revision, template, instruction, batch size, and snapshot/map digests are bound into `IrExperimentConfig`.

Qualification record for code head `b748cfcf3c3f6aff9c8aa167864d80fa34f4be1a`: 35 focused adapter/qrel-identity tests pass; full Ruff check and format pass; strict Pyright reports zero errors; the full unit suite passes 2,313 tests; live PostgreSQL/OpenSearch integration passes 166 tests. Exact-head GitHub Actions run [38993320241](https://github.com/Litju/DynamisRAG/actions/runs/38993320241) completed successfully, 3/3 jobs. These checks qualify source contracts; they do not claim reranker inference or paired reranking gains.

## Rights and transfer status

The RES-141 source registry accepts the SciFact source for local evaluation and records redistribution as `UNVERIFIED`. SciFact's [official license notice](https://github.com/allenai/scifact/blob/master/LICENSE.md) says claims and evidence annotations use [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and corpus abstracts are from S2ORC under [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/). The pinned [BEIR SciFact dataset card](https://huggingface.co/datasets/BeIR/scifact/blob/b3b5335604bf5ee3c4447671af975ea25143d4f5/README.md) separately labels the packaged dataset CC BY-SA 4.0, while BEIR's own [dataset notice](https://github.com/beir-cellar/beir/wiki/Datasets-available) says BEIR does not warrant users' rights and leaves license checks to users.

The approved Colab retrieval transfer was the exact indexing ZIP above: 5,183 abstracts plus the selected 20 claim texts. It excluded qrels and the other 280 query texts. The B2 scoring input is a separate minimal artifact containing those 20 claims and the exact fused top-50 texts, ranks, scores, hashes, and source identities. It excludes qrels, the full corpus, vectors, and the OpenSearch index. B2 scoring was not run during this 14-CU retrieval authorization.

For this private run, attribution names SciFact (Wadden et al., 2020), BEIR (Thakur et al., 2021), and Semantic Scholar/S2ORC; original notices and license links are retained; the archive SHA and deterministic 20-query selection are recorded; and the processing records that query selection and passage projection were performed. The [SciFact source notice](https://github.com/allenai/scifact/blob/master/LICENSE.md) assigns CC BY 4.0 to claims/evidence annotations and ODC-By 1.0 to the S2ORC abstract database. The [ODC-By 1.0 license](https://opendatacommons.org/licenses/by/1-0/) covers database rights, not independent rights in individual contents. The pinned [BEIR package card](https://huggingface.co/datasets/BeIR/scifact/blob/b3b5335604bf5ee3c4447671af975ea25143d4f5/README.md) separately declares CC BY-SA 4.0, and BEIR's [dataset notice](https://github.com/beir-cellar/beir/wiki/Datasets-available) leaves rights clearance to users. Google's [Terms of Service](https://policies.google.com/terms/embedded?hl=en-US) license user-uploaded protected content for hosting and service operation, including automated analysis; the [Colab Additional Terms](https://research.google.com/colaboratory/tos_v6.html) make users responsible for rights to third-party offerings and content.

**Rights decision update (2026-10-16): requester confirms rights for the exact indexing ZIP SHA `f358111ef72e4c19000f5b27956cd8249980736f2d244884c8cdecfadf41733e` and its derived B2 candidate payload, and asks to proceed with this Colab/Drive workflow.** Record this as requester-provided clearance for this run. Preserve the original component notices, package declaration, license links, and underlying-rights caveat in every artifact; this confirmation does not change those source statements or authorize public redistribution. A local `rights-confirmation.json` sidecar binds the confirmation to the exact input SHA. B1's earlier Drive authorization remains synthetic-only and is not reused.

## Retrieval execution record

The dedicated SciFact index was created only in the Colab run: alias `dynamisrag-res142-scifact` points to `dynamisrag-res142-scifact-passage-index-v2-f8483d00a654`, with OpenSearch 3.8.0 and 5,364 passages. The reserved production alias remains untouched. The earlier local CPU TEI warmup and Docker/WSL failure are historical evidence; no local model service was restarted.

The production embedding contract used the pinned `Qwen/Qwen3-Embedding-0.6B` revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, 512 dimensions, normalized vectors, right truncation, and distinct `query` and `document` prompt names. TEI ran on Colab loopback `http://127.0.0.1:8081`; its `/info` response matched the required model/runtime identity. The exact run summary and `tei-info.json` are included in the Drive evidence archive.

Retrieval readiness is **PASS_WITH_EVIDENCE**. The exact pinned Qwen3-Embedding-0.6B profile, passage-index-v2 projection, 20 BM25/dense/RRF responses, complete projection-wide `IrPassageMapping`, candidate snapshot, and RES-140 original-RRF baseline now exist. The baseline's source dataset and original qrels are preserved locally; its evaluation qrels are canonicalized through the sealed 5,183-row identity map. B2 reranker scoring remains unrun because the 14-CU Colab authorization was consumed by indexing/retrieval and the runtime has been disconnected. The measured balance delta was approximately 14.84 CU from the last reading, about 0.84 CU above the approved 14-CU cap.

## B2 output and GPU plan

The B2 input is prepared locally at `.tmp/res142-b2-transfer-20261016/res142-b2-candidate-input-v3.zip` (SHA-256 `c98d26a7aaf38f35b66203f0d27a73f618800218b1a90c4e322a02ab2c4100ac`). It contains exactly 20 queries and 1,000 fused candidate texts, with no qrels or full corpus. Its manifest binds the retrieval validator, baseline run/bundle, canonical qrel map, evaluation dataset, and code SHA. Its transfer decision notice preserves attribution and component-rights caveats; the adjacent requester-confirmation sidecar is SHA-256 `65980a1e19e6a01444193df55e98fd90a941b4c14b4c83efbd32db3564449d92`. It remains local and was not uploaded. B2 should emit one response per selected query with all 50 neural scores, rank mapping, model/template identity, timing, and resource observations. The post-run path uses `canonical_scifact_evaluation_dataset` with the same complete source-ID map for both original-RRF and reranked runs, then uses RES-140 to compute per-query and aggregate nDCG@10, Recall@10, MAP, and MRR plus paired deltas. The comparison binds the same canonical-evaluation dataset, projection, passage mapping, evaluation depth, metric policy, and scoring engine; the original source dataset and qrels remain separately sealed.

Harness ceiling: one supervised G4 run on Colab `2026.04`, 1,000 pairs, sequential batches of 8 (140 `/score` requests), one attempt, no fallback, and a 25-minute connection-to-deletion maximum. The monotonic session deadline reserves 2 minutes for process-group termination, Drive read-back/flush, and cleanup; the adapter receives the 23-minute work deadline. The evidence destination is a `b2/` subdirectory under the approved retrieval run folder `/content/drive/MyDrive/DynamisRAG/RES-142/retrieval/runs/20261016-10bce56f6ea04d949355af34fba2698c/`. It stores the exact candidate input ZIP, initial/final manifests, per-query score checkpoints, package freeze, GPU/software observations, logs and hashes. The harness will use the existing B1 support functions for hash read-back, failure capture, process termination, and Drive flush, plus the tested global session-budget contract. B2 inference has not run and needs additional CU authorization because the retrieval run exceeded the approved total.

At the last observed G4 rate of 8.9 CU/hour, 25 minutes projects to **3.71 CU**. Before any future connection, inspect the live rate and available balance and authorize a fresh numeric CU maximum; this projection is not a platform-enforced cap. The retrieval run's Colab allocation and data transfer are complete; its runtime was deleted. No further allocation, candidate upload, B2 inference, or merge is authorized under the spent 14-CU cap.
