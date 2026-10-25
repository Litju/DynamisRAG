# RES-142 B2 scientific qualification

## Result

The 20-query, 1,000-score Qwen3 result was evaluated offline against the frozen document-level SciFact RRF run and canonicalized original qrels. All four macro metrics increased. This supports a bounded pilot result, not a general effect.

| Metric | Frozen RRF | Qwen3 | Absolute delta | Relative delta | Nominal unadjusted paired 95% CI | Improved/regressed/tied | Sign-flip p (Holm) | Sign-test p (Holm) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nDCG@10 | 0.725411 | 0.832896 | 0.107485 | 0.148171 | [0.032456, 0.192385] | 6/1/13 | 0.031250 (0.125000) | 0.125000 (0.500000) |
| Recall@10 | 0.862500 | 0.925000 | 0.062500 | 0.072464 | [0.000000, 0.175000] | 2/0/18 | 0.500000 (0.500000) | 0.500000 (0.500000) |
| MAP | 0.676548 | 0.798889 | 0.122341 | 0.180832 | [0.030119, 0.230556] | 6/1/13 | 0.031250 (0.125000) | 0.125000 (0.500000) |
| MRR | 0.688333 | 0.813889 | 0.125556 | 0.182405 | [0.035000, 0.230000] | 6/1/13 | 0.031250 (0.125000) | 0.125000 (0.500000) |

## Query regressions

Per-query metrics, positive relevance ranks and qrel-derived pool counts are retained in the private audit evaluation. Public redistribution of those annotations is not cleared by the existing private-run rights decision.

## Ranking and candidate ceiling

Mean top-10 overlap was 5.35 of 10 (53.5%). Mean absolute document-rank displacement was 12.439 positions across the pool and 9.395 from the RRF top ten.
The 50-passage pool supplied a macro recall ceiling of 0.925000. Maximum Recall@10 from those candidates was 0.925000. A reranker cannot recover relevant documents absent from this pool.

## Uncertainty and limits

Paired percentile intervals resample 20 query-level deltas 50,000 times with seed 1422026 using SplitMix64. These are nominal, unadjusted 95% intervals, not simultaneous or multiplicity-adjusted intervals. Exact two-sided sign-flip tests assess the mean delta; exact binomial sign tests assess direction. Holm correction is applied per four-metric family. Power is limited at n=20; sign-flip tests assume symmetric deltas and bootstrap intervals treat selected queries as exchangeable. The result is not causal and does not validate other query populations. No metric has a Holm-adjusted sign-flip p-value below 0.05.

## Evidence and reproducibility

Evidence verification: **pass**. Drive readbacks reconcile 59 persisted artifacts, 28 checkpoints, the final manifest, and evidence.zip. Results SHA-256: `731afdcbff1db68e4093bf262a7803b734bd45b9094734223be1400d14e8815e`. The baseline audit reproduced the frozen metrics before Qwen evaluation. No inference or GPU work ran during review.
Provenance records Qwen/Qwen3-Reranker-0.6B revision `e61197ed45024b0ed8a2d74b80b4d909f1255473`, tokenizer `e61197ed45024b0ed8a2d74b80b4d909f1255473`, BF16, vLLM `0.31.0+cu129`, and template SHA-256 `bf7b104801a4e4ad6a147d732e2d5a837a36f708cdbd1b871b864979d3e10acd`. B2 execution helper, runner, template, and lock match PR #17 head `45697089fbf55b5e71d72b44c80ffd1086daadd0`. The A100 support and orchestration files are separate SHA-pinned Drive overlays. The runtime source commit is `85024771215947e0315f37a6dd67d1e8d0575eff`; Qwen reranker and evaluation modules are byte-identical at that commit and the verified PR head.
Public reporting excludes query text and all per-query outcomes, including qrel-derived counts and positive rank arrays. Full evidence is retained privately.

## Operations

Recorded run duration: 725.55 seconds against a 1,380-second work deadline and 1,500-second session ceiling. The recorded work duration is under both limits. Connection-to-deletion, current zero active sessions, actual CU balance, and billed consumption remain unverified; this review used no Colab connection and no billing records were available. The manifest's 2.208 CU is a projection, not an observed or enforced charge.

## External review and operational follow-up

CodeRabbit skipped the original draft PR; that status was not approval. Manual review of the original HEAD was COMMENTED with a spacing finding, now repaired; this was not approval. Verify review status at the repaired HEAD. See the [repair record](../../docs/res142-independent-review-repairs.md) for evidence retention and historical disclosure limits. Runtime shutdown and actual CU billing remain unresolved in [operational issue #19](https://github.com/Litju/DynamisRAG/issues/19), separately from offline scientific validity.

## Public release status

Public-release clearance is unresolved. Aggregate metrics, outcome counts and ceilings still derive from qrels; sanitization is not distribution authorization. Direct judgment cardinalities are withheld. See the [closure audit](../../docs/res142-publication-closure.md) for owner decisions.

## Decision

**GO for review of this bounded 20-query scientific pilot. NO-GO for a general performance claim or production-default change.** The aggregate gains are positive, but the fixed sample, candidate ceiling, and unresolved operations evidence limit any broader release decision.

Reproduce with the private authorized audit package and read-only Drive downloads: `uv run python scripts/res142_scientific_qualification.py --evidence-root <private-readback> --audit-package <authorized-private-package> --out <new-public-directory> --private-out <new-private-directory>`. Both outputs are create-only. Use a pre-protected parent (POSIX 0700 or restricted Windows DACL); repository destinations must also be untracked and Git-ignored.
