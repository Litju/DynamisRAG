# RES-142 Colab pre-run preparation

**Local repair only; GPU execution is not approved.** The companion notebook remains unexecuted. No runtime was allocated, package installed, model downloaded, dataset transferred, inference, smoke test, or benchmark run. Dependency metadata resolution was performed locally without installing packages; the resolver fetched the vLLM wheel and small wheel metadata artifacts needed to read package metadata. A browser-only Colab inspection was completed separately; it did not connect a runtime.

## Finding status

| Finding | Status | Evidence / remaining gate |
|---|---|---|
| Local Git baseline | **RESOLVED** | Preparation branch `res-142-colab-preflight` is based on merged `main` `67edcc785af2084e2cc4d8c27b08bea43a5f4bd3`; audited preparation commit `c6b30ff73694042cca910096dab0005db3a43b1f` is repaired by artifact commit `725aac310667922028889c769ce22983407c74b9`. The pre-existing untracked RES-138 notebook is preserved. |
| Python 3.12 / Linux x86_64 / manylinux 2.31 dependency resolution | **RESOLVED — candidate only** | Hash-locked candidate at `requirements/res142-colab.lock.txt`; metadata resolution succeeded with pinned direct constraints. No packages were installed. Runtime installability remains unverified. |
| Non-3.12 notebook kernel | **RESOLVED IN DESIGN** | The notebook kernel is an orchestrator only. A separate managed Python 3.12 interpreter and venv run uv, vLLM, probes, and scoring. Colab-side behavior remains unverified. |
| Prepared artifacts | **REPAIRED LOCALLY** | Notebook, support helper, direct requirements, transitive candidate lock, Jinja template, and this execution document are separate files. Notebook cells remain unexecuted. |
| Colab availability, CU rate, and budget | **PARTIALLY OBSERVED** | The runtime selector showed G4/A100 as selectable, not allocated. The disconnected Resources panel showed an approximate 0 CU/hour current rate; no GPU-specific rate or VRAM variant was displayed. A separately approved, short manual rate probe is required before B1 setup. |
| B2 pilot | **REDESIGNED; NOT RUN** | A deterministic 20-query SciFact test-split pilot precedes the separately gated 300-query evaluation. Frozen candidates, mapping, rights, and budget remain required. |

## Authority and local state

- **[VERIFIED]** The latest independent source audit in Linear RES-142 found P0 notebook syntax/template blockers and P1 authorization, evidence, cleanup, deadline, and provenance defects. This repair addresses those findings locally. GPU execution remains unapproved.
- **[PROPOSED GATES]** The browser inspection is complete. Because the disconnected UI did not show a GPU-specific rate, the next remote step is separately gated: `APPROVE COLAB RATE INSPECTION` must name one GPU and a numeric maximum CU charge. After that approval only, the operator manually connects the named GPU, reads the positive rate in Resources without running cells or installing packages, then disconnects/deletes the runtime. B1 setup requires a later `APPROVE COLAB SETUP` naming the same GPU and a numeric maximum CU charge. Neither approval is granted here. `APPROVE RES142 PILOT` and `APPROVE RES142 FULL BENCHMARK` remain separate later gates.
- **[VERIFIED]** The 2026-10-12 GPU amendment supersedes T4-first guidance: G4 (RTX PRO 6000 Blackwell, 96 GB) first if offered and compatible; otherwise A100 (record 40 GB vs 80 GB). Do not auto-select another accelerator. If neither G4 nor A100 is available, stop.
- **[VERIFIED]** The old local branch was `res-141-scientific-retrieval-evidence-adapters` at `cb8e05cb178121ab445ef3fb19086206c753ef99`; that commit is the first-parent baseline of merged RES-141 `main`. After read-only remote inspection and `git fetch origin main`, `origin/main` resolved to the requested commit `67edcc785af2084e2cc4d8c27b08bea43a5f4bd3`.
- **[VERIFIED]** The audited preparation branch was at `c6b30ff73694042cca910096dab0005db3a43b1f`; local repair commits preserve the existing branch and untracked `notebooks/res138_colab.operator.ipynb`. No reset or force operation was used.
- **[VERIFIED]** B1 execution artifacts and regression tests are in preparation artifact commit `725aac310667922028889c769ce22983407c74b9`. The documentation update is a separate atomic commit and does not change the notebook, support helper, lock, or template bytes.
- **[VERIFIED]** `preparation_base_git_sha` is `67edcc785af2084e2cc4d8c27b08bea43a5f4bd3`. No RES-142 implementation exists, so `res142_implementation_git_sha` is **null / not applicable**. The B1 execution manifest now records the exact preparation artifact commit and notebook, template, and lock SHA-256 values supplied from the review packet or verified uploaded bytes.

## Colab MCP connection and capability boundary

- **[VERIFIED]** Codex's built-in browser and browser UI control were available. The local `google-colab` MCP tool inventory exposed only `open_colab_browser_connection`; no notebook tools were listed. The tool was not called, and no unofficial fork or MCP implementation change was used.
- **[VERIFIED]** The signed-in Colab selector showed G4 and A100 as selectable. H100 was disabled; the dialog showed a Google AI Ultra upgrade prompt. No GPU was allocated. Resources showed zero active sessions and a disconnected runtime. The current disconnected rate was approximately 0 CU/hour; no GPU-specific rate or GPU VRAM variant was visible. High-RAM was available as an off switch. The visible CU balance is account state and should be checked again at the separately approved rate probe, not treated as a fixed repository value.
- **[LIMITATION]** A selectable G4/A100 is not an allocation guarantee. The rate and actual GPU memory configuration require a connected runtime, so the browser-only inspection did not establish either. Do not execute cells or install packages during the separately approved rate probe.
- **[PROPOSED]** After separate B1 setup approval, use the approved browser path for the selected runtime. Keep vLLM bound to `127.0.0.1:8000` inside the Colab VM. Never forward or publish the endpoint; do not rely on dynamic notebook tools that are absent from the exposed MCP inventory.

## Proposed GPU and software profile

| Item | G4: RTX PRO 6000 Blackwell | A100 | Status |
|---|---|---|---|
| Device identity | Blackwell; CUDA compute capability 12.0; 96 GB GDDR7 | Ampere; compute capability 8.0; record actual 40 GB or 80 GB | Hardware specs **VERIFIED**; Colab availability **UNVERIFIED** |
| vLLM floor | NVIDIA compute capability >=7.5; CUDA 12.9 precompiled binaries; Blackwell-class minimum CUDA 12.8 | Same cu129 software profile; A100 is within the documented GPU support floor | vLLM docs **VERIFIED**; exact Colab runtime compatibility **UNVERIFIED** |
| Host driver gate | Linux driver >=575.57.08 (conservative CUDA 12.9 Update 1 floor) | Same | NVIDIA CUDA release table **VERIFIED**; actual driver **UNVERIFIED** |
| CUDA userspace | CUDA 12.9 (cu129 PyTorch/vLLM wheels); no local `nvcc` build | CUDA 12.9 (cu129 PyTorch/vLLM wheels); no local `nvcc` build | **PROPOSED**, upstream documented |
| OS/Python | Linux x86_64, glibc >=2.31, Python 3.12.x; exact Python patch captured in B1 | Same | Python 3.12 compatibility **VERIFIED**; glibc 2.31 comes from the selected llguidance wheel floor; actual Colab image **UNVERIFIED** |
| PyTorch | `torch==2.13.0`, CUDA 12.9 wheel | Same | vLLM 0.31.0 CUDA requirements **VERIFIED**; live wheel/runtime **UNVERIFIED** |
| vLLM | pinned `vllm==0.31.0+cu129` x86_64 wheel | Same | Upstream release/tag/asset digest **VERIFIED**; runtime smoke **UNVERIFIED** |
| Transformers/tokenizer stack | `transformers==5.10.4`, `huggingface-hub==1.31.0`, `tokenizers==0.22.1`, `safetensors==0.6.2` | Same | Version metadata ranges intersect **VERIFIED**; integrated runtime **UNVERIFIED** |
| Precision | BF16 weights/compute | BF16 weights/compute | Model weights and both GPU product specifications support BF16 (**VERIFIED**); runtime check remains **UNVERIFIED** |

**Recommendation — [PROPOSED]: G4 first, A100 second.** G4 matches the explicit user priority and supplies 96 GB memory. The proposed vLLM/PyTorch cu129 BF16 profile is shared across both choices, so batch, input, and server settings can remain identical. A100 40 GB is sufficient for this 0.6B model under the conservative memory budget; 80 GB is also acceptable. No Colab SKU is assumed available. Select hardware manually in Colab after approval; notebook checks fail closed on a mismatch. Never substitute T4, L4, H100, CPU, or another GPU.

**Compatibility limitation — [UNVERIFIED]:** vLLM's current docs give a general compute-capability floor and CUDA 12.8 Blackwell minimum, and list B200/GB200 as Blackwell examples. They do not explicitly certify this exact `vllm==0.31.0` wheel on a Colab G4 RTX PRO 6000. The matching compute capability and CUDA level support the proposal, but only the approved post-connection synthetic smoke can establish the account's exact runtime works. If G4 fails compatibility, stop and ask before switching profiles; do not retry on different hardware automatically.

### Direct inputs and resolved candidate lock

**[RESOLVED — candidate only]** `requirements/res142-colab.txt` preserves the proposed direct pins: vLLM 0.31.0 cu129 x86_64 release wheel, `torch==2.13.0`, `torchvision==0.28.0`, `torchaudio==2.11.0`, `numba==0.65.0`, `transformers==5.10.4`, `huggingface-hub==1.31.0`, `tokenizers==0.22.1`, and `safetensors==0.6.2`. The vLLM wheel SHA-256 is `5176ab746f4e736c4a354a11d0620abf0a941de287cad937ac34039293ca6f5f`.

**[RESOLVED — candidate only]** A transitive, hash-bearing requirements lock was generated without installing packages for Python 3.12, Linux x86_64, `x86_64-manylinux_2_31`, binary wheels only. Resolver: uv `0.12.4`. Source policy: PyPI `https://pypi.org/simple`, plus uv's explicit `--torch-backend cu129` mapping to `https://download.pytorch.org/whl/cu129` for the PyTorch packages; vLLM itself is the pinned official GitHub release wheel. The resulting candidate resolves torch `2.13.0+cu129`, torchvision `0.28.0+cu129`, and torchaudio `2.11.0+cu129`, with wheel hashes recorded in the lock. The candidate lock is 407,840 bytes / 5,459 lines; SHA-256 `d8a87ad2f1f2a7faccfb96fee9fa859995f61caa7b8a7b845e268888c4c111dc`.

Exact local resolution command (metadata resolution only; **not** an installation). The candidate lock was first written under the pre-approved temporary directory and then copied byte-for-byte into the workspace:

```powershell
uv pip compile requirements/res142-colab.txt `
  --python-version 3.12 `
  --python-platform x86_64-manylinux_2_31 `
  --no-python-downloads `
  --only-binary :all: `
  --generate-hashes `
  --emit-index-url `
  --emit-index-annotation `
  --default-index https://pypi.org/simple `
  --torch-backend cu129 `
  --no-cache `
  --output-file D:\Dev\Temp\User\opencode\res142-colab-candidate.lock.txt
```

An initial resolver-only attempt gave PyTorch's index priority over PyPI and stopped at a `packaging` index-selection conflict; no versions were changed. The final command used PyPI as the default and uv's dedicated cu129 backend, resolving the same pinned direct versions successfully. The resolver fetched the 527.7 MiB vLLM release wheel and small wheel metadata artifacts to read package metadata; it did not install them, and no model weights were downloaded. Runtime dependency installation remains **NOT RUN / UNVERIFIED**. The original glibc-floor failure at manylinux 2.28 is superseded by the successful manylinux 2.31-targeted metadata resolution. The actual Colab image must still report glibc >=2.31.

After later setup authorization only, the planned sync commands are:

```bash
/content/res142/uv-tools/bin/uv pip sync \
  --python /content/res142/venv/bin/python \
  --torch-backend cu129 \
  --only-binary :all: \
  --require-hashes \
  --strict \
  /content/res142/res142-colab.lock.txt
/content/res142/uv-tools/bin/uv pip check --python /content/res142/venv/bin/python
```

To handle a Colab Jupyter kernel that is not Python 3.12, the setup cell uses the kernel interpreter only to place the pinned uv tool in a separate tools directory, then runs every workload command through a separately managed Python 3.12 venv. The equivalent startup sequence is:

```python
import subprocess
import sys

subprocess.run([
    sys.executable, "-m", "pip", "install",
    "--disable-pip-version-check", "--no-deps", "--only-binary=:all:",
    "--target", "/content/res142/uv-tools", "uv==0.12.4",
], check=True)
```

```bash
/content/res142/uv-tools/bin/uv --version  # must print: uv 0.12.4
export UV_PYTHON_INSTALL_DIR=/content/res142/managed-python
export UV_PYTHON_PREFERENCE=only-managed
/content/res142/uv-tools/bin/uv python install 3.12
/content/res142/uv-tools/bin/uv venv --python 3.12 /content/res142/venv
/content/res142/uv-tools/bin/uv pip sync \
  --python /content/res142/venv/bin/python \
  --torch-backend cu129 --only-binary :all: --require-hashes --strict \
  /content/res142/res142-colab.lock.txt
/content/res142/uv-tools/bin/uv pip check --python /content/res142/venv/bin/python
```

The kernel interpreter is not used for model/runtime imports or scoring. uv-managed Python is selected explicitly, and the notebook records its exact patch version. Bootstrap package source is PyPI (`uv==0.12.4`). uv-managed CPython 3.12 is fetched by uv's Python distribution manager (python-build-standalone artifacts) and kept under `/content/res142/managed-python`; the inference lock's package sources are documented above.

The notebook captures the exact runtime Python patch, `uv pip freeze`, `uv pip check`, torch/CUDA, and vLLM versions into the B1 manifest. Neither the candidate lock nor its successful resolution qualifies a remote runtime or authorizes package installation.

## Model identity, scoring template, and serving contract

- **[VERIFIED] Model:** `Qwen/Qwen3-Reranker-0.6B`, Hugging Face immutable commit `e61197ed45024b0ed8a2d74b80b4d909f1255473` (current API `sha`; model card reports Apache-2.0, BF16, 0.6B parameters, 32K model context). The tokenizer revision is the same repository commit, explicitly pinned. No `main`/`latest` references are used.
- **[VERIFIED] Official original-model architecture overrides:** `architectures=["Qwen3ForSequenceClassification"]`, `classifier_from_token=["no","yes"]`, `is_original_qwen3_reranker=true`. Do not replace with a converted third-party checkpoint.
- **[VERIFIED] Template source:** vLLM's `examples/pooling/score/template/qwen3_reranker.jinja` at vLLM commit `db9527a46873454610df6dbedf79a36d6bf1a7f6` (release tag `v0.31.0`); vendored as `notebooks/res142_qwen3_reranker.jinja` and embedded into the standalone notebook. SHA-256: `bf7b104801a4e4ad6a147d732e2d5a837a36f708cdbd1b871b864979d3e10acd`. It uses query/document semantic roles and the yes/no system instruction.
- **[PROPOSED] Instruction:** explicitly pass the official Qwen default, `Given a web search query, retrieve relevant passages that answer the query`, for every dataset in the initial profile. Do not silently introduce a custom task instruction. Any later instruction change creates a new serving/config identity and requires a new comparison.
- **[PROPOSED] API:** vLLM `/score`, request batch of up to 8 document pairs for one query, `use_activation=true`, per-query token cap 1,024, per-document cap 6,144, and no request-side combined truncation. Before B2 upload, tokenize with the pinned tokenizer and score template; require each query/document under its cap and the complete rendered pair (including template tokens) <=8,192. Refuse over-limit pairs rather than letting vLLM's per-field caps clip them silently. Validate every response's cardinality, finite score, and zero-based index before restoring candidate IDs.
- **[PROPOSED] Serving:** `runner=pooling`, `dtype=bfloat16`, `max-model-len=8192`, `tensor-parallel-size=1`, `max-num-seqs=8`, `max-num-batched-tokens=16384`, `gpu-memory-utilization=0.50`, seed 0, host `127.0.0.1`, port 8000. The template digest is generated from the exact UTF-8 bytes before launch and included in evidence; the notebook refuses a digest mismatch.

```bash
export HF_HOME=/content/res142/hf-home
export HF_HUB_DISABLE_TELEMETRY=1
/content/res142/venv/bin/vllm serve Qwen/Qwen3-Reranker-0.6B \
  --revision e61197ed45024b0ed8a2d74b80b4d909f1255473 \
  --tokenizer-revision e61197ed45024b0ed8a2d74b80b4d909f1255473 \
  --runner pooling \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.50 \
  --tensor-parallel-size 1 \
  --seed 0 \
  --hf_overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' \
  --chat-template /content/res142/qwen3_reranker.jinja \
  --host 127.0.0.1 \
  --port 8000
```

The score call is `POST http://127.0.0.1:8000/score` with `model`, one `queries` string, a `documents` array of at most 8 texts, the pinned `instruction`, `use_activation: true`, `max_tokens_per_query: 1024`, and `max_tokens_per_doc: 6144`. It is localhost-only in the runtime. The original Qwen model card and vLLM v0.31.0 online example document this architecture override, scoring path, and Qwen3 template. Full command execution remains **NOT RUN**.

## Prepared files for inspection

These are the actual local artifacts, not summaries or generated placeholders:

1. **Notebook cells:** [`notebooks/res142_qwen3_reranker_colab.ipynb`](../notebooks/res142_qwen3_reranker_colab.ipynb) — two approval gates, exact artifact provenance prompts, bounded subprocesses, GPU/runtime checks, hash-locked sync, uploaded-template verification, two-pair synthetic `/score` smoke, failure evidence, and process/runtime cleanup instructions. Every cell remains unexecuted with empty outputs.
2. **Requirements:** [`requirements/res142-colab.txt`](../requirements/res142-colab.txt) contains direct constraints; [`requirements/res142-colab.lock.txt`](../requirements/res142-colab.lock.txt) contains the successful Python 3.12 / x86_64 / manylinux 2.31 transitive candidate with hashes.
3. **Jinja template:** [`notebooks/res142_qwen3_reranker.jinja`](../notebooks/res142_qwen3_reranker.jinja) — exact vLLM v0.31.0 vendored Qwen3 scoring template.
4. **B1 support helper:** [`notebooks/res142_b1_support.py`](../notebooks/res142_b1_support.py) contains the stdlib-only authorization, process-group, deadline, provenance, and failure-manifest helpers used by the notebook.
5. **Execution documentation:** this file, [`docs/res142-colab-preflight.md`](res142-colab-preflight.md), contains the gates, exact commands, dependency/model sources, pilot contract, transfer boundary, and cleanup behavior.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `notebooks/res142_qwen3_reranker_colab.ipynb` | 41,972 | `eeda7e4a8c7155fb57b06973e506893302741a560b1759c5aed963a1dee18317` |
| `requirements/res142-colab.txt` | 891 | `3df83a1da185da638bc96bc7b78bf3469df5a771448e549c844b139cd8de5193` |
| `requirements/res142-colab.lock.txt` | 407,840 | `d8a87ad2f1f2a7faccfb96fee9fa859995f61caa7b8a7b845e268888c4c111dc` |
| `notebooks/res142_qwen3_reranker.jinja` | 684 | `bf7b104801a4e4ad6a147d732e2d5a837a36f708cdbd1b871b864979d3e10acd` |
| `notebooks/res142_b1_support.py` | 10,712 | `ca3c631a03edca02b6d7576e0d14e5af2a8101b413dc866efbe3189a1158426e` |

The packet's own SHA-256 is reported outside this file to avoid self-reference. Model revision SHA: `e61197ed45024b0ed8a2d74b80b4d909f1255473`. vLLM release wheel SHA-256: `5176ab746f4e736c4a354a11d0620abf0a941de287cad937ac34039293ca6f5f`.

## Notebook sequence and gates

1. **Rate inspection is separately gated:** the disconnected Resources panel only showed an approximate zero CU/hour. A separate `APPROVE COLAB RATE INSPECTION` must name G4 or A100 and a numeric CU maximum before the operator manually connects that SKU to read its displayed rate. Run no notebook cells, install nothing, then disconnect/delete the rate-probe runtime. If the rate is absent or not positive, stop.
2. **Setup parameters (default closed):** the first notebook cell prompts for both approval phrases, the rate-probe GPU and CU maximum, the observed positive rate, the B1 GPU and CU maximum, and the preparation commit/notebook SHA-256 from the review packet. Blank approvals stop before artifact upload. The rate-probe GPU and B1 GPU must match. No dataset path, `ALLOW_DATA_TRANSFER=False`, and no Drive mount.
3. **Projection is not a cap:** the notebook checks the projected 45-minute CU use using the positive rate observed in the account UI. This is a planning check only; Colab exposes no hard per-run CU cap. The approved CU maximum is a human spending authorization, not a guaranteed platform enforcement mechanism. Never substitute an estimate for an observed rate.
4. **Manual hardware check:** after setup approval, the operator connects the already-selected GPU and records model name, compute capability, VRAM, and driver. Require the manually approved target to match. No fallback or reallocation code.
5. **Isolated Python 3.12:** the notebook kernel is only an orchestrator. It uses its current Python solely to place pinned uv `0.12.4` in `/content/res142/uv-tools`; uv then installs a separate managed Python 3.12 interpreter, creates `/content/res142/venv`, and runs all package sync, probes, vLLM, and scoring through that venv's absolute interpreter. Kernel Python is not required to be 3.12.
6. **Supporting artifacts:** upload the review-approved support helper, lock, and Jinja file after setup approval; verify their SHA-256 values before dependency setup. The notebook consumes only the uploaded Jinja bytes and rechecks their digest immediately before server startup. Sync the hash-locked requirements with binary wheels only and `--torch-backend cu129`; do not resolve or upgrade during B1.
7. **Runtime check:** verify the venv is Python 3.12, torch 2.13.0/cu129, BF16 support, vLLM 0.31.0, selected GPU identity, and the full package freeze.
8. **Model initialization:** reverify the uploaded template digest, launch the exact command above in a dedicated process group, poll local `/health` for at most 15 minutes, and check the vLLM process remains alive. `/health` proves liveness; it is not model identity.
9. **Synthetic scoring smoke:** exactly two synthetic pairs, one request, no dataset text. Check result count/index/finite-score shape only; do not assert quality improvement from two examples.
10. **Evidence and cleanup:** every CUDA, bootstrap, runtime-verification, template, startup, model-verification, scoring, cleanup, or export failure writes an atomic sanitized failure manifest with available runtime observations, artifact hashes, and a stderr/log tail, then attempts to export a synthetic-only evidence archive. Subprocesses use a dedicated process group; timeout handling sends TERM, then KILL, and verifies child exit. Each blocking subprocess timeout is bounded by the remaining notebook deadline. After success or failure, verify/export evidence and manually disconnect/delete the Colab runtime in the UI. The runtime is not considered closed until the UI reports disconnected. B1 ends here and requires review before any data or pilot.

All code cells have null execution counts and empty outputs. The Colab runtime, model weights, and install are not initialized by opening or reading the local notebook. The notebook's 45-minute timer starts at its first code cell; the operator must start a separate timer when connecting because the pre-cell interval is not measurable by notebook code.

## Scientific comparison contract

- **[VERIFIED] Unchanged baseline:** `HybridRetrievalService` uses BM25 top 50, dense top 50, `rrf-v1` with `RRF_K=60`, stable passage-key ties, and a versioned physical projection. The initial reranker receives the frozen fused top 50 *before* any requested top-10 response truncation. It never reruns either retrieval lane or edits the baseline.
- **[PROPOSED] Candidate input:** one sealed, canonical per-query top-50 snapshot with `dataset/source/revision/split`, corpus SHA, query ID/text SHA, physical index/projection SHA, retrieval/code identity, `passage_key`, document ID/version, source-span IDs, text SHA, BM25 rank/raw score, dense rank/raw score, RRF rank/raw score, and exact candidate position. Preserve short-result and zero-hit queries; never pad, duplicate, relabel or rebuild candidate IDs. Upload only approved candidate/query text, not the full corpus or qrels.
- **[VERIFIED] Document boundary:** use RES-140's same full snapshot-wide `IrPassageMapping` for baseline and reranked runs, exact mapping SHA, version checks, stable passage-to-document deduplication, and the same complete query universe. Preserve raw lane scores and original RRF rank separately from the reranker score. Store exactly 50 reranked candidate positions (or the true shorter source prefix); tie-break equal reranker scores by original RRF rank, then passage key. Fail closed on missing score/index, non-finite score, duplicate/foreign passage, incomplete rank prefix, mismatched mapping/projection or unsupported document top-10 after deduplication.
- **[VERIFIED] Document metrics:** RES-140 `ir-metric-policy-v1`: nDCG@10 (linear graded gain), Recall@10 (`rel>=1`), MAP/AP(rel=1), and MRR/RR(rel=1), MAP/MRR through the declared depth 50. Preserve source qrels, negative/zero judgments and unjudged counts. Missing-qrel and zero-positive queries score zero and remain in the macro denominator. Compare paired per-query deltas and aggregate metrics with identical dataset SHA, candidate snapshot, projection SHA, mapping SHA, depth, policy and scoring engine.
- **[PROPOSED] B2 pilot:** use 20 queries selected by sorting `(SHA256(b"res142-sf-pilot-v1\0" + query_id_utf8), query_id)` over the complete 300-query SciFact BEIR test query-ID universe and taking the first 20. This uniform deterministic sample is fixed before reranker output or qrels values are inspected; it is not a tuned slice. Freeze the 20 IDs, full first-stage top-50 candidates for each, passage/document mapping, rights receipt, and the corresponding baseline output before transfer. Maximum 1,000 query-document pairs; true shorter/zero-hit prefixes remain shorter. Report selected-query first-stage return counts and compare the same 20 paired candidates/queries. Treat pilot metrics as operational/exploratory evidence only; do not claim full-test quality improvement or use the pilot to tune prompt/model settings. Full SciFact 300-query evaluation requires fresh approval.
- BEIR shortlist tests can be separate per dataset (NFCorpus, SciDocs, ArguAna, FiQA) only after full-benchmark approval and only when their exact RES-141 slices and protocol receipts are supplied. Train/dev splits are not mixed with test. ArguAna's identical query/document exclusion is applied before evaluation-depth truncation and must retain RES-141 candidate evidence.
- **[VERIFIED] Scientifically distinct tasks:** SciFact-Open is `pooled-partial`; use a declared corpus variant and report pooled metrics/unjudged retrieval, never full-corpus recall. QASPER is a separate within-paper paragraph-anchor task under `qasper-paragraph-f1-v3`, with its own complete/partial/unavailable annotation rules and denominator; do not merge it into document-level nDCG/Recall/MAP/MRR or compare its F1 to BEIR.
- **[UNVERIFIED] Inputs:** no materialized RES-141 slice, approved rights manifest, snapshot-wide passage map, sealed BM25/dense/RRF top-50 candidates, or baseline run is present in this checkout. The checked-in fixtures are synthetic and cannot qualify retrieval quality. A future pilot is blocked until the candidate bundle and its out-of-band SHA are prepared and verified locally.

## Data transfer, rights, and persistence

- **B1:** synthetic text only; no Drive mount, dataset, qrels, corpus, private text, or source download. Return the small evidence archive through a user-visible Colab download and verify it locally.
- **B2/Pilot:** before any upload, require a separate written authorization naming the exact input bundle SHA, source/split, rights receipt and destination. RES-141's accepted local-read decision is not permission for a third-party upload. SciFact-Open is license-unstated/evaluation-only with redistribution prohibited; QASPER paper text inherits publisher rights; SciFact/BEIR source licenses and attribution must travel with the decision. If rights do not explicitly permit the intended Colab transfer, do not transfer that text. Never include credentials, personal, confidential, or non-public user content.
- **Minimal transfer:** only frozen query/top-50 candidate texts plus stable IDs/provenance and hashes; keep qrels and full corpus local. Do not place query/candidate text in notebook outputs or server logs. Responses contain IDs/positions/scores only.
- **Persistence:** a Colab VM is ephemeral. For a future approved pilot, require a user-selected Drive destination or an explicit local download/export, write a unique create-only run folder, store the input/output bytes plus manifest and SHA-256 inventory, verify read-back, then transfer a copy to the Windows workspace. Never rely on `/content` surviving. Drive mounting and persistent storage themselves require authorization.

## Limits, stop conditions, and resource risk

| Stage | Hard proposed limit | Gate |
|---|---|---|
| CU-rate inspection | Manually connect one named G4/A100 runtime only after approval; inspect Resources rate, run no cells, install nothing, then disconnect/delete. This may consume CUs. | `APPROVE COLAB RATE INSPECTION` with GPU and numeric maximum CU |
| B1 GPU allocation | One manually selected GPU, one runtime, G4 or A100 only; the setup approval names the same SKU used for rate inspection and a numeric maximum CU charge | Separate `APPROVE COLAB SETUP` after the observed rate is reported |
| B1 wall clock | 45-minute notebook deadline from first code cell; startup 15 min; one 2-pair synthetic smoke; one request batch; no retries. Operator timer starts at runtime connect. | `APPROVE COLAB SETUP` with GPU and numeric CU maximum |
| CU enforcement | Observed rate × allowed wall-clock time is only a projection. Colab exposes no hard per-run CU cap in the inspected UI; cleanup/export can also consume time. | Numeric maximum is a user spending authorization, not a technical guarantee |
| B1 batching | server `max_num_seqs=8`; `max_num_batched_tokens=16384`; smoke batch 2 | Fixed in command |
| GPU memory | vLLM utilization budget 0.50; stop if observed used VRAM >80% or free <20%; never lower batch/precision automatically | Stop and ask |
| B2 pilot (future) | One designated dataset only: deterministic 20-query SciFact test sample; at most 1,000 pairs; one client request at a time, batch 8; 25 min maximum GPU time; report a projection from an observed rate against the approved maximum, without claiming hard enforcement | `APPROVE RES142 PILOT` plus exact input SHA, rights, and CU authorization |
| Full benchmark (future) | Dataset/query/pair totals explicitly reviewed after pilot; one GPU at a time; no unattended multi-run | `APPROVE RES142 FULL BENCHMARK` |
| Retries/fallback | Zero automatic GPU, environment, model, or dataset retries; no hardware fallback | Ask user |

**[OBSERVED; RATE UNRESOLVED] Colab compute-unit cost and availability:** G4 and A100 appeared as selectable runtime types, but neither was allocated and actual capacity is not guaranteed. The account Resources panel showed an approximate zero CU/hour while disconnected; no GPU-specific rate or VRAM variant was visible. The visible balance is transient account data and must be rechecked during the separately approved rate probe. If a positive rate is still unavailable, do not start B1. Never substitute an estimated rate, and never claim the notebook enforces the numeric CU authorization ceiling.

Immediate stop on target mismatch, driver/CUDA mismatch, non-BF16 GPU, dependency conflict, model/tokenizer SHA mismatch, template mismatch, wrong served architecture, non-loopback listener, server startup over 15 minutes, GPU memory threshold, malformed scoring response, timeout, or any rights/input identity failure. No silent degraded mode, substitute model, precision, CUDA build, or GPU.

## Evidence manifest

The success or failure JSON manifest is canonical UTF-8, sorted keys, stable row order, and written atomically. The failure path preserves the stage, exception type, sanitized message, sanitized stderr/log tail, observations available at that point, and hashes available for uploaded artifacts; it then attempts to download a synthetic-only archive without masking the original failure. The manifest records at minimum:

`manifest_revision`, exact preparation artifact commit SHA, notebook SHA-256, template SHA-256, lock SHA-256, null/not-applicable RES-142 implementation SHA, future evaluation code SHA, model repository and immutable model SHA, tokenizer revision, uv version, full package freeze and dependency-check result, Python patch/platform/glibc, vLLM and torch/CUDA versions, GPU model/compute capability/VRAM/driver, observed CU rate, separate rate-inspection approval/target/maximum, B1 approved CU maximum and a clearly labeled projection, all serving flags, score endpoint and response schema revision, exact task instruction and SHA, phase/gate, failure stage/type/message/log tail when applicable, dataset/source/split/slice and rights receipt IDs, dataset/corpus/query/qrel/candidate-run/mapping/projection SHA-256, evaluation depth, pair/query/batch counts, ordered output score/rank digest, timing boundaries and p50/p95/p99 (null for B1), server startup and smoke latency, GPU memory/utilization observations and high-water marks, host-memory snapshot and child-process max RSS, process-group cleanup status, manual runtime-disconnect requirement, stop/exit status, and the SHA-256/size inventory of all artifacts. The preparation commit and notebook SHA are supplied from the local review packet so the notebook does not hash itself.

B1 sets dataset, candidate, baseline, mapping and evaluation SHA fields to `null`, identifies the two synthetic inputs, and makes no scientific claim. For B2, the input manifest is verified before upload and the output manifest is write-once. Metrics remain local until output hashes and all RES-140/RES-141 gates have passed.

## Official references consulted

- [RES-142 latest Linear authorization gate and GPU amendment](https://linear.app/alignerr-cmj/issue/RES-142/add-qwen3-neural-reranking-through-vllm)
- [RES-140 canonical IR contract in this repository](res140-ir.md)
- [RES-141 dataset/rights contract in this repository](res141-datasets.md)
- [vLLM v0.31.0 GPU installation guidance](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
- [vLLM v0.31.0 scoring API and model/template source](https://github.com/vllm-project/vllm/tree/v0.31.0/examples/pooling/score)
- [vLLM v0.31.0 CUDA package requirements](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/requirements/cuda.txt)
- [vLLM v0.31.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.31.0)
- [llguidance 1.7.6 wheel metadata](https://pypi.org/project/llguidance/1.7.6/)
- [Qwen/Qwen3-Reranker-0.6B card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
- [Qwen model metadata API (immutable current SHA)](https://huggingface.co/api/models/Qwen/Qwen3-Reranker-0.6B)
- [NVIDIA CUDA GPU compute capability table](https://developer.nvidia.com/cuda-gpus)
- [NVIDIA RTX PRO 6000 Blackwell Server Edition specifications](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/)
- [NVIDIA A100 specifications](https://www.nvidia.com/en-us/data-center/a100/)
- [NVIDIA CUDA Toolkit driver compatibility release table](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
- [Google Colab resource limits FAQ](https://research.google.com/colaboratory/faq.html)
- [Official Google Colab MCP source](https://github.com/googlecolab/colab-mcp)
- [uv managed Python installation](https://docs.astral.sh/uv/guides/install-python/)
