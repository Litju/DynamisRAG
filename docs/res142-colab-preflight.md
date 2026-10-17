# RES-142 Colab preflight and execution record

**Historical repair pass:** it did not allocate or reconnect a Colab runtime, mount Drive, or authorize Google access. The checked-in notebook cells remain unexecuted. That pass did not install packages, download model weights, transfer a dataset, run inference, or run a smoke test or benchmark. The later B1 execution is recorded below.

## First B1 attempt and preserved evidence status

The first B1 attempt used branch head `f14b19a1e54fe26c366ab1fa53dc17e8d32aaa81`, G4 / RTX PRO 6000 Blackwell, and Colab runtime **2026.04**. Hardware preflight passed. Bootstrap then stopped at the uv version-banner check, before managed Python 3.12 provisioning, hash-locked inference dependency installation, vLLM, model weights, or scoring; **0/2 pairs** were scored. The runtime was disconnected and deleted, and the prior approval does not authorize a retry.

Colab's retained output reports `/content/res142/b1-synthetic-smoke/manifest.json` and `/content/res142/res142-b1-failure-7bbad04601dd43c4a7a1995ed3b354f3.zip`. Neither file was available in the local Downloads or evidence directory when this repair began. The actual `uv --version` stdout was not captured; this record does not reconstruct it. The lost archive motivates the new mandatory Drive policy. The old `f14b19a1...` local ZIP is an execution bundle, not the failure archive, and is obsolete. Schema v3 rejects it. Do not upload or reuse it.

## Repaired B1 execution — 2026-10-15

**B1 PASS_WITH_EVIDENCE.** One supervised attempt ran from reviewed commit `d0e16f6aee906084e003d25dfc668de272261b0b` on G4 / Colab runtime **2026.04**. The actual model `Qwen/Qwen3-Reranker-0.6B` loaded at immutable revision `e61197ed45024b0ed8a2d74b80b4d909f1255473`; `/score` returned exactly two finite scores: candidate 0 `0.9965360164642334`, candidate 1 `0.07977034151554108`.

The runtime reported NVIDIA RTX PRO 6000 Blackwell Server Edition, compute capability 12.0, 97,887 MiB VRAM, driver 580.82.07, CUDA 12.9, Python 3.12.13, PyTorch 2.13.0+cu129, and vLLM 0.31.0+cu129. The pinned uv validator accepted both `uv 0.12.4 (x86_64-unknown-linux-gnu)` banners. Package installation took 55.377 seconds, vLLM startup 110.856 seconds, and the two-pair request 19.433 ms. The notebook runtime timer recorded 408.967 seconds.

The observed rate remained 8.9 CU/hour. Available balance moved from 301.03 CU immediately before connection to 299.45 CU before session termination (about 1.58 CU observed). Colab's session manager then showed **No active sessions**; the notebook toolbar showed **Reconnect G4 High-RAM**. The Drive flush status was `flushed`, and the final manifest marked all persisted artifacts `verified`.

Drive run directory: `/content/drive/MyDrive/DynamisRAG/RES-142/b1/runs/1792098943664755104-6de0362b7add4586b1ff1f2645845ca1/`.

| Artifact | SHA-256 |
|---|---|
| Fresh schema-v3 execution bundle | `1e7be9dd76e429cbe0c5088b9b8d376322ae7f56193dfbd9d620ca10a94e016c` |
| Initial manifest | `0de4cea4872e31cd1e19b39edc5b2de84d1e5ac14c07a3cfd0529e4d361d01fb` |
| Final manifest | `7673f9fcc5124d49f13712e07352055f921850a80a7002f0667d46e9ad4272aa` |
| Evidence ZIP | `54515b655cb668c15baf80306580ad08b3ffc6a135018e5e07adcde72ff2ae28` |
| `synthetic-scores.json` | `00c1c9cc047d66fa2563adc5aab88c0acb2adff281ab63845f5a47ece2871703` |
| `environment/pip-freeze.txt` | `fa8c371244881603f407b0885986a2fcd4f97bae525ee1651feb62f1c7efe29d` |
| `vllm.log` | `3275fa983d614f712cd80bdf5ee635589f1808f50cd7c94b87af0430285efa82` |

The final manifest and evidence archive are visible in the Drive UI. The complete model snapshot hashes, GPU observations, logs, checkpoints and artifact hashes are in that run directory.

## Finding status

| Finding | Status | Evidence / remaining gate |
|---|---|---|
| Local Git baseline | **RESOLVED** | Preparation branch `res-142-colab-preflight` is based on merged `main` `67edcc785af2084e2cc4d8c27b08bea43a5f4bd3`. Earlier audit `c6b30ff73694042cca910096dab0005db3a43b1f` was repaired in `725aac310667922028889c769ce22983407c74b9`; this release-gate PR adds the final P1 hardening. The pre-existing untracked RES-138 notebook is preserved. |
| Python 3.12 / Linux x86_64 / manylinux 2.31 dependency resolution | **RESOLVED — candidate only** | Hash-locked candidate at `requirements/res142-colab.lock.txt`; metadata resolution succeeded with pinned direct constraints. No packages were installed. Runtime installability remains unverified. |
| Non-3.12 notebook kernel | **RESOLVED IN DESIGN** | The notebook kernel is an orchestrator only. A separate managed Python 3.12 interpreter and venv run uv, vLLM, probes, and scoring. Colab-side behavior remains unverified. |
| Prepared artifacts | **REPAIRED LOCALLY** | Notebook, support helper, direct requirements, transitive candidate lock, Jinja template, and local bundle builder are separate reviewable files. Notebook cells remain unexecuted. |
| Colab availability, CU rate, and budget | **B1 PASS_WITH_EVIDENCE** | The repaired 2026-10-15 G4 / 2026.04 run scored both synthetic pairs. It used about 1.58 CU by the displayed balance delta, within the fresh 9.2 CU cap; the timer and software details are in the execution record above. |
| Durable B1 artifacts | **VERIFIED; FLUSHED** | The fresh bundle, final manifest, evidence ZIP, environment, checkpoints and scores were SHA-256 read-back verified under `/content/drive/MyDrive/DynamisRAG/RES-142/b1/runs/1792098943664755104-6de0362b7add4586b1ff1f2645845ca1/`. Drive UI visibility was independently confirmed and flush returned `flushed`. |
| SciFact retrieval and original-RRF baseline | **PASS_WITH_EVIDENCE** | The exact 20-query run completed in Colab G4 with 50 saved BM25, dense, and fused results per query. Offline production validation made zero retrieval calls and passed deterministic top-50 fusion, cross-query identity homogeneity, and full source-lane seals. Candidate snapshot SHA `c4320cebfca3350a8c0d2070c6ed3f20a68692cfdc9b3fc4f583b0a92dd5310e`; passage map SHA `f2489a8dd194849921c81fec8fbd2959567913ef1b58bb56e6d27c9af37854ad`; projection SHA `f8483d00a654a1a2c3d5a8d5f712676ba79edc006ae513bf0a3398ce963a0473`. The baseline uses RES-140 and a sealed canonical-qrel evaluation dataset while preserving original source dataset/pilot qrel SHAs `4e712...` / `70c07...`. Full hashes and metrics are in `docs/res142-scifact-pilot-preparation.md`.
| B2 reranker inference | **PREPARED; NOT RUN** | The exact local 20-query / 1,000-candidate ZIP is `.tmp/res142-b2-transfer-20261016/res142-b2-candidate-input-v3.zip` (SHA `c98d26a7aaf38f35b66203f0d27a73f618800218b1a90c4e322a02ab2c4100ac`); its adjacent rights sidecar binds this hash, and the ZIP carries the attribution/decision notice. The reviewed 25-minute global session harness specifies pinned model/tokenizer/template checks, Drive checkpoints, and process cleanup. The Colab runtime is disconnected. The 14-CU total was exceeded by about 0.84 CU during retrieval/indexing, so another GPU run requires a new numeric CU cap. |

## Authority and local state

- **[VERIFIED]** The latest independent source re-audit in Linear RES-142 confirmed the uv banner-check repair requirements. A later user storage amendment makes Google Drive the canonical durable B1 evidence store and supersedes the previous no-Drive policy. This local repair implements the design; it does not mount Drive, authorize an account, or grant another GPU run.
- **[REQUIRED FOR ANY FUTURE B1 RUN]** Fresh setup authorization must name G4, the numeric CU maximum, runtime 2026.04, exactly two synthetic pairs, the Google Drive mount, and destination `/content/drive/MyDrive/DynamisRAG/RES-142/`. The local bundle builder additionally requires `APPROVE COLAB DRIVE STORAGE` and the exact destination confirmation. `APPROVE RES142 PILOT` and `APPROVE RES142 FULL BENCHMARK` remain separate gates.
- **[REQUIRED FOR THIS B1 REVISION]** GPU is G4 (RTX PRO 6000 Blackwell, 96 GB) only. The A100 column below is retained as historical hardware context; the bundle builder and notebook reject A100, CPU, and all other devices. Stop on any SKU mismatch.
- **[REQUIRED FOR ANY FUTURE B1 RUN]** Select and visually verify Colab runtime version **2026.04** in the Runtime settings UI before running any notebook cell. `Latest` is not acceptable. The notebook does not attempt to detect the runtime version; the operator performs this UI check. Execution provenance records the required version and the manual verification procedure, not an automatically detected runtime claim.
- **[VERIFIED]** The old local branch was `res-141-scientific-retrieval-evidence-adapters` at `cb8e05cb178121ab445ef3fb19086206c753ef99`; that commit is the first-parent baseline of merged RES-141 `main`. After read-only remote inspection and `git fetch origin main`, `origin/main` resolved to the requested commit `67edcc785af2084e2cc4d8c27b08bea43a5f4bd3`.
- **[VERIFIED]** The audited preparation branch was at `c6b30ff73694042cca910096dab0005db3a43b1f`; local repair commits preserve the existing branch and untracked `notebooks/res138_colab.operator.ipynb`. No reset or force operation was used.
- **[VERIFIED]** Earlier B1 artifact and regression repairs are in `725aac310667922028889c769ce22983407c74b9`. Final hardening code and tests are in atomic commit `92fff2b6c663a43ffa4283585089666077622555`; the documentation update is a separate commit. The hardening adds separately gated rate inspection, pre-connection verified bundling, early failure evidence, exact execution provenance, bounded process/archive handling, and regression coverage.
- **[VERIFIED]** `preparation_base_git_sha` is `67edcc785af2084e2cc4d8c27b08bea43a5f4bd3`. No RES-142 implementation exists, so `res142_implementation_git_sha` is **null / not applicable**. The local bundle builder records the actual full Git HEAD and computes notebook, support-helper, Jinja, and lock SHA-256 values from canonical Git LF bytes; it accepts no operator-entered SHA values and requires the branch tip to match `origin/res-142-colab-preflight`.

## Colab MCP connection and capability boundary

- **[VERIFIED]** Codex's built-in browser and browser UI control were available. The local `google-colab` MCP tool inventory exposed only `open_colab_browser_connection`; no notebook tools were listed. The tool was not called, and no unofficial fork or MCP implementation change was used.
- **[OBSERVED 2026-10-12; NOT RECHECKED IN THIS LOCAL RELEASE GATE]** The signed-in Colab selector showed G4 and A100 as selectable. H100 was disabled; the dialog showed a Google AI Ultra upgrade prompt. No GPU was allocated. Resources showed zero active sessions and a disconnected runtime. The disconnected rate was approximately 0 CU/hour; no GPU-specific rate or GPU VRAM variant was visible. High-RAM was available as an off switch. Current account balance/runtime state must be checked again only during the separately authorized rate inspection.
- **[LIMITATION]** A selectable G4/A100 is not an allocation guarantee. The rate and actual GPU memory configuration require a connected runtime, so the browser-only inspection did not establish either. Do not execute cells or install packages during the separately approved rate probe.
- **[PROPOSED]** After separate B1 setup approval, use the approved browser path for the selected runtime. Keep vLLM bound to `127.0.0.1:8000` inside the Colab VM. Never forward or publish the endpoint; do not rely on dynamic notebook tools that are absent from the exposed MCP inventory.
- **[REQUIRED]** Drive access uses Colab's official `google.colab.drive.mount` and `flush_and_unmount` APIs. Mounting requests explicit account authorization. The notebook does not depend on Colab MCP notebook tools or a custom OAuth bypass.

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

**B1 hardware requirement: G4 only.** The proposed vLLM/PyTorch cu129 BF16 profile is pinned for this G4 run. No Colab SKU is assumed available. Select G4 manually in Colab after approval; notebook checks fail closed on a mismatch. Do not switch to A100 or another GPU without a revised approval and independent review.

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

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-deps",
        "--only-binary=:all:",
        "--target",
        "/content/res142/uv-tools",
        "uv==0.12.4",
    ],
    check=True,
)
```

```bash
/content/res142/uv-tools/bin/uv -V         # must print exactly: uv 0.12.4
/content/res142/uv-tools/bin/uv --version  # full banner with build metadata; version must be 0.12.4
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

The notebook captures the exact runtime Python patch, both uv version banners (`uv -V` and `uv --version`), `uv pip freeze`, `uv pip check`, torch/CUDA, and vLLM versions into the B1 manifest. Both banners are parsed and checked against the exact semantic version; a platform suffix such as `(x86_64-unknown-linux-gnu)` is accepted. Neither the candidate lock nor its successful resolution qualifies a remote runtime or authorizes package installation.

## Model identity, scoring template, and serving contract

- **[VERIFIED] Model:** `Qwen/Qwen3-Reranker-0.6B`, Hugging Face immutable commit `e61197ed45024b0ed8a2d74b80b4d909f1255473` (current API `sha`; model card reports Apache-2.0, BF16, 0.6B parameters, 32K model context). The tokenizer revision is the same repository commit, explicitly pinned. No `main`/`latest` references are used.
- **[VERIFIED] Official original-model architecture overrides:** `architectures=["Qwen3ForSequenceClassification"]`, `classifier_from_token=["no","yes"]`, `is_original_qwen3_reranker=true`. Do not replace with a converted third-party checkpoint.
- **[VERIFIED] Template source:** vLLM's `examples/pooling/score/template/qwen3_reranker.jinja` at vLLM commit `db9527a46873454610df6dbedf79a36d6bf1a7f6` (release tag `v0.31.0`); vendored as `notebooks/res142_qwen3_reranker.jinja` and included as a separately SHA-verified bundle member. The notebook never reconstructs or overwrites it. SHA-256: `bf7b104801a4e4ad6a147d732e2d5a837a36f708cdbd1b871b864979d3e10acd`. It uses query/document semantic roles and the yes/no system instruction.
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

The score call is `POST http://127.0.0.1:8000/score` with `model`, one `queries` string, a `documents` array of at most 8 texts, the pinned `instruction`, `use_activation: true`, `max_tokens_per_query: 1024`, and `max_tokens_per_doc: 6144`. It is localhost-only in the runtime. The original Qwen model card and vLLM v0.31.0 online example document this architecture override, scoring path, and Qwen3 template. The repaired B1 execution passed; see the execution record above.

## Prepared files for inspection

These are the actual local artifacts, not summaries or generated placeholders:

1. **Notebook cells:** [`notebooks/res142_qwen3_reranker_colab.ipynb`](../notebooks/res142_qwen3_reranker_colab.ipynb) — a single locally verified bundle upload, bounded subprocesses, GPU/runtime checks, hash-locked sync, uploaded-template verification, two-pair synthetic `/score` smoke, failure evidence, and process/runtime cleanup instructions. The checked-in notebook remains unexecuted with empty outputs; the separate Colab copy used for B1 completed successfully.
2. **Requirements:** [`requirements/res142-colab.txt`](../requirements/res142-colab.txt) contains direct constraints; [`requirements/res142-colab.lock.txt`](../requirements/res142-colab.lock.txt) contains the successful Python 3.12 / x86_64 / manylinux 2.31 transitive candidate with hashes.
3. **Jinja template:** [`notebooks/res142_qwen3_reranker.jinja`](../notebooks/res142_qwen3_reranker.jinja) — exact vLLM v0.31.0 vendored Qwen3 scoring template.
4. **B1 support helper:** [`notebooks/res142_b1_support.py`](../notebooks/res142_b1_support.py) contains the stdlib-only authorization, process-group, deadline, provenance, and failure-manifest helpers used by the notebook.
5. **Local bundle builder:** [notebooks/res142_b1_prepare.py](../notebooks/res142_b1_prepare.py) requires a clean checkout on `res-142-colab-preflight`, derives the exact commit from local Git, requires it to equal the branch's upstream tip, canonicalizes text files to Git's LF bytes before hashing the four reviewed source artifacts, validates GPU/rate/CU and explicit Drive authorization/destination, and writes a create-only schema-v3 ZIP before B1 allocation. After fresh approvals, run it locally with `--setup-authorization "APPROVE COLAB SETUP" --rate-inspection-authorization "APPROVE COLAB RATE INSPECTION" --drive-authorization "APPROVE COLAB DRIVE STORAGE" --drive-destination-confirmation "/content/drive/MyDrive/DynamisRAG/RES-142" --gpu G4 --rate-inspection-max-cu <N> --observed-cu-per-hour <observed-value> --approved-max-cu <N> --output <path-outside-repository>`. It accepts no notebook/preparation/artifact hash arguments. Schema v3 invalidates all earlier bundles.
6. **Execution documentation:** this file, [`docs/res142-colab-preflight.md`](res142-colab-preflight.md), contains the gates, exact commands, dependency/model sources, pilot contract, transfer boundary, and cleanup behavior.

| Artifact (canonical Git LF bytes) | Bytes | SHA-256 |
|---|---:|---|
| `notebooks/res142_qwen3_reranker_colab.ipynb` | 72,274 | `992aeda7a5b94d38da6eb5017bb3427fd2b92993c3c0ba46f44a8c43e78f021c` |
| `requirements/res142-colab.txt` | 891 | `3df83a1da185da638bc96bc7b78bf3469df5a771448e549c844b139cd8de5193` |
| `requirements/res142-colab.lock.txt` | 407,840 | `d8a87ad2f1f2a7faccfb96fee9fa859995f61caa7b8a7b845e268888c4c111dc` |
| `notebooks/res142_qwen3_reranker.jinja` | 684 | `bf7b104801a4e4ad6a147d732e2d5a837a36f708cdbd1b871b864979d3e10acd` |
| `notebooks/res142_b1_support.py` | 34,185 | `d98f7ea12d1df559f11c8be132cc033befb66bebfa13af56e630c870130feedc` |
| `notebooks/res142_b1_prepare.py` | 11,146 | `c503e533784ddb32630f8486eba7ee175fc3b728de7c056709b68f1229e80819` |

The local bundle configuration records the exact preparation commit and four execution-artifact hashes. Model revision SHA: `e61197ed45024b0ed8a2d74b80b4d909f1255473`. vLLM release wheel SHA-256: `5176ab746f4e736c4a354a11d0620abf0a941de287cad937ac34039293ca6f5f`.

## Separate CPU-only Drive storage qualification

[`notebooks/res142_drive_storage_qualification_colab.ipynb`](../notebooks/res142_drive_storage_qualification_colab.ipynb) is a separate, unexecuted Colab probe; it is not part of the G4-locked B1 notebook or its bundle. Before running any cell, the operator must visually select runtime **2026.04** and **CPU / no accelerator**. The probe requires both `APPROVE RES-142 CPU-ONLY DRIVE QUALIFICATION` and the existing `APPROVE COLAB DRIVE STORAGE`, plus confirmation of the exact `/content/drive/MyDrive/DynamisRAG/RES-142` root, before it calls `drive.mount()`. Confirm the intended Google account before accepting Colab's mount prompt, then use the same account for the independent Drive UI check.

Before creating the unique run directory, the probe calls the same B1 `support._drive_available_bytes` callback on the approved Drive root and fails closed below the documented 16 MiB minimum. It records observed bytes and labels the check `shutil.disk_usage` filesystem capacity, **not** an authoritative Google account quota check.

The probe creates only `drive-qualifications/<unique_run_id>/sentinel.json` and `qualification-manifest.json` beneath that root, using the reviewed helper's create-only write/read-back functions. It then calls Colab's official Drive flush/unmount and requires the operator to download exactly those two files from the independent Drive UI; the notebook compares their uploaded SHA-256 values to the helper's read-back hashes. Any write, flush, or UI verification failure is reported as `incomplete` and stops qualification. The code does not list or read unrelated Drive files, install packages, access a GPU, or load a model. This notebook documents a future qualification workflow and does not authorize connecting Colab or mounting Drive.

After the Drive UI check, the probe creates a separate, create-only JSON receipt under local `/content` scratch, flushes it to disk, and verifies its SHA-256 by read-back. It contains the run ID, reviewed PR head and helper digest, observed free bytes, sentinel and immutable pre-flush manifest hashes, flush outcome, independent UI file hashes, and final status. The notebook downloads the receipt and prints its SHA-256. Before disconnecting the runtime, the operator must complete the selected handoff: attach that exact JSON file to a new Linear RES-142 comment and include its SHA-256 in the comment, or save it create-only at the exact immutable audit-workspace path selected before mounting. The notebook then requires the RES-142 comment URL or exact audit path and a re-entered receipt SHA-256; it rejects a mismatched target or digest. This is an operator attestation of the external handoff, not an API read-back. It never rewrites the pre-flush Drive manifest or remounts Drive to store the receipt.

## Notebook sequence and gates

1. **Runtime version gate:** before any code cell, use Runtime > Change runtime type to select **2026.04**, confirm the selected value in the UI, and do not choose `Latest`. The kernel has no trusted runtime-version detector; provenance records this as a required operator check.
2. **Prepare before B1 allocation:** after fresh setup, rate-inspection, and Drive-storage approvals, run the local bundle builder with the approved G4, observed rate, numeric limits, `APPROVE COLAB DRIVE STORAGE`, and exact destination confirmation. It requires a clean artifact tree, verifies local Git HEAD equals the pushed upstream branch tip, hashes the four reviewed artifacts from canonical Git LF bytes, and writes a create-only schema-v3 ZIP outside the repository.
3. **Explicit Drive authorization and validation:** the first cell accepts only the local v3 bundle. Before approving the UI prompt, the operator verifies Colab is signed into the intended Google account. `google.colab.drive.mount('/content/drive')` uses Colab's official account-authorization flow. The notebook verifies the FUSE mount, `MyDrive`, exact confirmed root, reported free-space floor, and an exclusive sentinel write/read-back SHA-256 before installing uv or starting vLLM. It creates `/content/drive/MyDrive/DynamisRAG/RES-142/b1/runs/<unique_run_id>/`; failures stop before dependency/model setup.
4. **Initial durable record:** before dependency installation, write the reviewed bundle, config, source identities, and initial run manifest to that unique run directory. Copy from local `/content` staging to create-only Drive files and verify each by SHA-256 read-back. If Drive persistence cannot be established, stop before installing packages or loading weights.
5. **GPU and time gates:** verify the manually selected G4 hardware, driver, capability, and VRAM. CU projections are not hard caps; actively supervise the approved CU and 45-minute limits. No automatic retry or fallback.
6. **Execution locality:** use Drive only for durable records and outputs. uv, managed Python 3.12, venv, package/model caches, vLLM, and inference stay under local `/content`. Model-cache mirroring is not implemented and requires separate explicit authorization.
7. **Hash-locked environment:** verify pinned uv `0.12.4` with exact `uv -V` and parsed `uv --version` banners, then install only the current hash-locked requirements with binary wheels and `--torch-backend cu129`. Persist the package freeze and dependency-check result to Drive.
8. **Runtime check and checkpoint:** verify Python 3.12, torch 2.13.0/cu129, BF16, vLLM 0.31.0, GPU identity, and the full package freeze. Persist runtime/environment checkpoints before model initialization.
9. **Model initialization:** verify the pinned model/tokenizer revision and template; checkpoint the model/revision/GPU identities before vLLM can download weights to local `/content`. Do not copy model weights or caches to Drive.
10. **Synthetic scoring:** score exactly two synthetic query-document pairs in one request. Persist score positions/values and hashes only; do not add scientific inputs.
11. **Success persistence:** persist the final manifest and bounded evidence archive, read back their SHA-256 values, write the final persistence checkpoint, flush Drive best-effort, and report status.
12. **Failure persistence and cleanup:** persist a sanitized failure manifest, available environment/GPU observations, artifact hashes, and sanitized logs on every available failure path. Mark partial or failed writes `incomplete`; keep browser downloads secondary. After success or failure, verify file visibility and hashes in the Drive UI, then manually disconnect/delete the Colab runtime. Disconnect at the authorized time/CU limit even if a flush is incomplete.

All code cells have null execution counts and empty outputs. The Colab runtime, model weights, and install are not initialized by opening or reading the local notebook. Extract and open the notebook from the locally verified bundle. The notebook's 45-minute timer starts at its first code cell; the operator timer starts at Connect and remains active until the UI confirms disconnection.

## Scientific comparison contract

- **[VERIFIED] Unchanged baseline:** `HybridRetrievalService` uses BM25 top 50, dense top 50, `rrf-v1` with `RRF_K=60`, stable passage-key ties, and a versioned physical projection. The initial reranker receives the frozen fused top 50 *before* any requested top-10 response truncation. It never reruns either retrieval lane or edits the baseline.
- **[PROPOSED] Candidate input:** one sealed, canonical per-query top-50 snapshot with `dataset/source/revision/split`, corpus SHA, query ID/text SHA, physical index/projection SHA, retrieval/code identity, `passage_key`, document ID/version, source-span IDs, text SHA, BM25 rank/raw score, dense rank/raw score, RRF rank/raw score, and exact candidate position. Preserve short-result and zero-hit queries; never pad, duplicate, relabel or rebuild candidate IDs. Upload only approved candidate/query text, not the full corpus or qrels.
- **[VERIFIED] Document boundary:** use RES-140's same full snapshot-wide `IrPassageMapping` for baseline and reranked runs, exact mapping SHA, version checks, stable passage-to-document deduplication, and the same complete query universe. Preserve raw lane scores and original RRF rank separately from the reranker score. Store exactly 50 reranked candidate positions (or the true shorter source prefix); tie-break equal reranker scores by original RRF rank, then passage key. Fail closed on missing score/index, non-finite score, duplicate/foreign passage, incomplete rank prefix, mismatched mapping/projection or unsupported document top-10 after deduplication.
- **[VERIFIED] Document metrics:** RES-140 `ir-metric-policy-v1`: nDCG@10 (linear graded gain), Recall@10 (`rel>=1`), MAP/AP(rel=1), and MRR/RR(rel=1), MAP/MRR through the declared depth 50. Preserve source qrels, negative/zero judgments and unjudged counts. Missing-qrel and zero-positive queries score zero and remain in the macro denominator. Compare paired per-query deltas and aggregate metrics with identical dataset SHA, candidate snapshot, projection SHA, mapping SHA, depth, policy and scoring engine.
- **[VERIFIED] B2 input:** the frozen 20-query selection, complete 50-per-query RRF snapshot, projection-wide passage map, and RES-140 baseline are built. Source dataset/pilot qrel SHAs remain `4e712...` / `70c07...`; the baseline uses a separately sealed canonical-qrel dataset and source-ID map so passage-run canonical document IDs join the original BEIR judgments without changing their relevance values. The exact local B2 candidate ZIP is recorded in `docs/res142-b2-execution-harness.md` and excludes qrels/full corpus.
- **[NOT RUN] B2 reranking:** the candidate input has not been uploaded and Qwen3-Reranker inference has not run. The Colab retrieval session consumed about 14.84 CU from the last observed balance, exceeding the 14-CU cap by about 0.84 CU. Any B2 GPU session needs a new numeric CU cap; the 25-minute global ceiling, exact model/template checks, Drive evidence, and cleanup remain the execution contract. Treat pilot metrics as operational/exploratory evidence only; do not claim full-test quality improvement or use the pilot to tune prompt/model settings. Full SciFact 300-query evaluation requires fresh approval.
- BEIR shortlist tests can be separate per dataset (NFCorpus, SciDocs, ArguAna, FiQA) only after full-benchmark approval and only when their exact RES-141 slices and protocol receipts are supplied. Train/dev splits are not mixed with test. ArguAna's identical query/document exclusion is applied before evaluation-depth truncation and must retain RES-141 candidate evidence.
- **[VERIFIED] Scientifically distinct tasks:** SciFact-Open is `pooled-partial`; use a declared corpus variant and report pooled metrics/unjudged retrieval, never full-corpus recall. QASPER is a separate within-paper paragraph-anchor task under `qasper-paragraph-f1-v3`, with its own complete/partial/unavailable annotation rules and denominator; do not merge it into document-level nDCG/Recall/MAP/MRR or compare its F1 to BEIR.
- **[VERIFIED] Retrieval inputs:** actual saved BM25/dense/RRF responses, the complete passage map, candidate snapshot, and baseline are present in the local evidence workspace and Drive run. The offline validator reconstructed the full deterministic RRF top-50 from saved lanes and made zero retrieval calls. Their hashes are recorded in `docs/res142-scifact-pilot-preparation.md`. The checked-in unit fixtures remain synthetic; they qualify code behavior only, not SciFact metrics.

## Data transfer, rights, and persistence

- **B1:** Drive is mandatory for the reviewed bundle/config, initial and stage manifests, environment freeze, sanitized logs, GPU/resource observations, checksums, two synthetic scores, and bounded evidence archive. Do not upload scientific datasets, qrels, corpora, private text, or source files. `files.download` is a secondary backup.
- **B2/Pilot:** the user confirmed rights for the exact indexing ZIP and derived candidate payload. The local B2 ZIP is SHA-bound by its adjacent rights sidecar; it has not been uploaded. Preserve original notices and license links, and do not imply public redistribution clearance. Never include credentials, personal, confidential, or unrelated non-public content.
- **Minimal transfer:** B2 uses only frozen query/top-50 candidate texts plus stable IDs/provenance and hashes; qrels and the full corpus remain local for scoring. Do not place query/candidate text in notebook outputs or server logs. Responses contain IDs/positions/scores only.
- **Persistence:** the retrieval evidence is under `/content/drive/MyDrive/DynamisRAG/RES-142/retrieval/runs/20261016-10bce56f6ea04d949355af34fba2698c/`. A future B2 session writes under that approved run folder's `b2/` subdirectory. `/content` is execution and staging scratch only. Model-cache mirroring remains disabled.

## Limits, stop conditions, and resource risk

| Stage | Hard proposed limit | Gate |
|---|---|---|
| CU-rate inspection | Manually connect the named G4 runtime only after approval; inspect Resources rate, run no cells, install nothing, then disconnect/delete. This may consume CUs. | `APPROVE COLAB RATE INSPECTION` with G4 and numeric maximum CU |
| B1 GPU allocation | One manually selected G4 runtime; the setup approval names G4 and a numeric maximum CU charge | Separate `APPROVE COLAB SETUP` after the observed rate is reported |
| B1 Drive storage | Reported free-space floor of 16 MiB; unique create-only run directory; initial bundle/config/manifest and each checkpoint read-back SHA-256 verified before installs | `APPROVE COLAB DRIVE STORAGE` plus exact `/content/drive/MyDrive/DynamisRAG/RES-142` confirmation |
| B1 wall clock | 45-minute notebook deadline from first code cell; startup 15 min; one 2-pair synthetic smoke; one request batch; no retries. Operator timer starts at runtime connect. | `APPROVE COLAB SETUP` with GPU and numeric CU maximum |
| CU enforcement | Observed rate × allowed wall-clock time is only a projection. Colab exposes no hard per-run CU cap in the inspected UI; cleanup/export can also consume time. | Numeric maximum is a user spending authorization, not a technical guarantee |
| B1 batching | server `max_num_seqs=8`; `max_num_batched_tokens=16384`; smoke batch 2 | Fixed in command |
| GPU memory | vLLM utilization budget 0.50; stop if observed used VRAM >80% or free <20%; never lower batch/precision automatically | Stop and ask |
| B2 pilot | One designated dataset only: the prepared deterministic 20-query SciFact input; at most 1,000 pairs; one client request at a time, batch 8; 25-minute connection-to-deletion maximum with 23-minute work deadline; report a projection from the live rate against a fresh numeric CU maximum, without claiming hard enforcement | B2 code/input are prepared; obtain a new numeric CU cap before any Colab runtime because the previous 14-CU total is spent |
| Full benchmark (future) | Dataset/query/pair totals explicitly reviewed after pilot; one GPU at a time; no unattended multi-run | `APPROVE RES142 FULL BENCHMARK` |
| Retries/fallback | Zero automatic GPU, environment, model, or dataset retries; no hardware fallback | Ask user |

**[OBSERVED] Colab compute-unit cost:** the G4 rate was about **8.9 CU/hour**. The balance moved from 298.15 CU at 20:59 local to 283.31 CU when the runtime was disconnected, approximately **14.84 CU used**. The approved 14-CU ceiling was exceeded by about 0.84 CU because the runtime remained connected after retrieval finished. Colab now reports zero active sessions and approximately 0 CU/hour. Any B2 run requires a fresh cap; Colab does not enforce the user's per-run ceiling.

Immediate stop on target mismatch, driver/CUDA mismatch, non-BF16 GPU, dependency conflict, model/tokenizer SHA mismatch, template mismatch, wrong served architecture, non-loopback listener, server startup over 15 minutes, GPU memory threshold, malformed scoring response, timeout, or any rights/input identity failure. No silent degraded mode, substitute model, precision, CUDA build, or GPU.

## Evidence manifest

The success or failure JSON manifest is canonical UTF-8, sorted keys, stable row order, and written atomically. The failure path preserves the stage, exception type, sanitized message, sanitized stderr/log tail, observations available at that point, and hashes available for uploaded artifacts; it then attempts to download a synthetic-only archive without masking the original failure. The manifest records at minimum:

`manifest_revision`, exact preparation artifact commit SHA, notebook SHA-256, support-helper SHA-256, template SHA-256, lock SHA-256, required Colab runtime version (`2026.04`) and operator-UI verification procedure, null/not-applicable RES-142 implementation SHA, future evaluation code SHA, model repository and immutable model SHA, tokenizer revision, exact uv version plus full and short uv banners, full package freeze and dependency-check result, Python patch/platform/glibc, vLLM and torch/CUDA versions, GPU model/compute capability/VRAM/driver, observed CU rate, separate rate-inspection approval/target/maximum, B1 approved CU maximum and a clearly labeled projection, all serving flags, score endpoint and response schema revision, exact task instruction and SHA, phase/gate, failure stage/type/message/log tail when applicable, dataset/source/split/slice and rights receipt IDs, dataset/corpus/query/qrel/candidate-run/mapping/projection SHA-256, evaluation depth, pair/query/batch counts, ordered output score/rank digest, timing boundaries and p50/p95/p99 (null for B1), server startup and smoke latency, GPU memory/utilization observations and high-water marks, host-memory snapshot and child-process max RSS, process-group cleanup status, manual runtime-disconnect requirement, stop/exit status, and the SHA-256/size inventory of all artifacts. Preparation provenance is derived from the reviewed Git checkout and verified artifact bytes; RES-142 implementation and evaluation-code identities remain separate fields.

Drive persistence metadata records the unique run ID, required root, mount verification, observed free bytes/floor, per-file expected/read-back hashes, checkpoint sequence, flush result, and an explicit `complete` or `incomplete` persistence status. The initial manifest and execution bundle are write-once before dependency installation. Later checkpoints and failure records use unique names and create-only writes.

The free-space probe reports the capacity exposed by the mounted DriveFS path; it is not a guaranteed Google account quota API. Google Drive can still reject later operations with quota or I/O errors. Each write is therefore create-only and SHA-256 read-back verified, with incomplete persistence recorded and surfaced. The operator must verify the final run folder in the Drive UI before disconnecting when time/CU limits permit.

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
- [Official Colab Drive mount and flush implementation](https://github.com/googlecolab/colabtools/blob/main/google/colab/drive.py)
- [Google Colab Drive mount, I/O, and storage-quota FAQ](https://research.google.com/colaboratory/faq.html)
- [uv managed Python installation](https://docs.astral.sh/uv/guides/install-python/)
- [uv version banner formats](https://docs.astral.sh/uv/getting-started/help/#viewing-the-version)
