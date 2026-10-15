# RES-142 B2 execution harness

Status: prepared for review, **not executed**. No Colab runtime, reranker inference, or scientific-text transfer is authorized by this document. B2 remains blocked until a real, locally generated candidate ZIP exists and its exact transfer rights and SHA receive separate approval.

## Frozen input

The single B2 input is a closed ZIP with `candidate-input.json`, `NOTICES.txt`, and `manifest.json`. It carries the selected 20 SciFact queries and exactly the 50 fused passage texts for each query, with original passage IDs, RRF ranks/scores, source identities, and text hashes. It excludes qrels, the remaining corpus, projection vectors, and the OpenSearch index. The ZIP must bind the pilot dataset, selected-query list, snapshot, projection, and complete passage-map SHA-256 values. The notebook must refuse any inventory, identity, candidate count, or digest mismatch before mounting Drive or starting vLLM.

## Runtime and scoring contract

- Colab runtime: `2026.04`; one manually selected G4, NVIDIA RTX PRO 6000 Blackwell Server Edition, compute capability 12.0. No A100 or CPU fallback.
- One session: **25 minutes from connection to deletion**. The monotonic work deadline is 23 minutes, with the last 2 minutes reserved for process termination, Drive read-back/flush, and cleanup. Start an external stopwatch at connection because the notebook clock starts on its first cell.
- Before work, inspect the live CU/hour rate and available balance. At the last observed 8.9 CU/hour, the 25-minute ceiling projects to 3.71 CU. Require a fresh numeric approval cap and refuse a projected or available balance above it; the estimate is not a platform-enforced cap.
- Bootstrap the exact pinned uv `0.12.4`, managed Python 3.12, and `requirements/res142-colab.lock.txt` with the existing hash-required `cu129` sync. Lock SHA-256: `d8a87ad2f1f2a7faccfb96fee9fa859995f61caa7b8a7b845e268888c4c111dc`.
- Verify PyTorch `2.13.0+cu129`, CUDA 12.9, BF16, G4 name/compute capability/VRAM, and pinned package versions before model startup.
- Model: `Qwen/Qwen3-Reranker-0.6B`, revision and tokenizer revision `e61197ed45024b0ed8a2d74b80b4d909f1255473`; vLLM `0.31.0+cu129`; BF16; loopback `127.0.0.1:8000/score`; one request at a time, batches of 8. Template SHA-256 is `bf7b104801a4e4ad6a147d732e2d5a837a36f708cdbd1b871b864979d3e10acd` (`notebooks/res142_qwen3_reranker.jinja`).
- Pass the 23-minute work deadline to subprocesses and `Qwen3RerankerAdapter`. Score 20 queries in frozen selection order; each has seven sequential requests, for at most 140 requests. The adapter clamps request timeouts to remaining work time and does not retry, substitute dependencies, or fall back.

## Drive evidence and cleanup

Use the separately approved Google account and destination `/content/drive/MyDrive/DynamisRAG/RES-142/b2/runs/<run-id>`. Persist the exact input ZIP and initial manifest before installs or model downloads. After each query, persist its score JSON and a checkpoint containing query ID, candidate count, request count, elapsed time, and SHA-256 read-back. Persist the final model/GPU/software identity, model snapshot file hashes, environment freeze, vLLM log, score outputs, artifact inventory, session timing, and spend observations. Keep downloaded model weights in ephemeral `/content`, never in Drive.

On the first failure, stop scoring and preserve the stage, exception and traceback, failed command, captured stdout/stderr, and available logs to Drive. Terminate the vLLM process group, perform independent Drive read-back, flush/unmount Drive, and require the operator to disconnect and delete the runtime inside the 25-minute ceiling. Do not launch a replacement run automatically.

Use the existing [B1 Drive/process support helper](../notebooks/res142_b1_support.py) for verified Drive writes, checkpoints, failure capture, subprocess deadlines, process-group termination, and flush. The B2 adapter entry point is `Qwen3RerankerAdapter.rerank_frozen_candidates`, supplied only the hash-verified fused candidate records; it receives the 23-minute work deadline. Local RES-140 scoring then joins the returned scores with the preserved qrels and original baseline to compute per-query and aggregate nDCG@10, Recall@10, MAP, and MRR and their paired deltas.

## Rights and launch gates

The proposed payload includes up to 1,000 abstract passages and 20 claims. SciFact labels the abstracts as S2ORC under ODC-By 1.0; the pinned BEIR package card declares CC BY-SA 4.0; and BEIR states that it does not clear downstream user rights. The source registry records redistribution as `UNVERIFIED`. The exact Colab/Drive transfer remains **not cleared** until this package-level/component-level discrepancy and the cloud processing/storage terms are resolved. Attribution must name SciFact/Wadden et al., BEIR/Thakur et al., and Semantic Scholar/S2ORC; preserve license/source notices and identify the deterministic query selection and unchanged candidate text. See the [SciFact license](https://github.com/allenai/scifact/blob/master/LICENSE.md), [pinned BEIR card](https://huggingface.co/datasets/BeIR/scifact/blob/b3b5335604bf5ee3c4447671af975ea25143d4f5/README.md), and [BEIR disclaimer](https://github.com/beir-cellar/beir/wiki/Datasets-available).

The run bundle must carry separate approvals for the exact input ZIP SHA, rights-clearance reference, one B2 attempt, live rate/balance, numerical CU ceiling, and Drive storage destination. These fields remain blank until approval. B1 authorization applies only to its synthetic evidence.
