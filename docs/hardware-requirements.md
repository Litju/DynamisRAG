# Hardware and runtime requirements

Status: evidence-limited. The figures on this page are observations from the
published Research Preview receipts, not validated minimum requirements.
Unmeasured values are marked **NOT MEASURED**.

## Supported local platform

| Item | Requirement | Evidence |
| --- | --- | --- |
| Operating system | Windows 11 with PowerShell for the documented local installation | Clean-install receipt records a Windows 11 host with Docker Desktop `linux/amd64` |
| Python | 3.12.13, exact patch | `.python-version` and the CI workflow |
| Package manager | `uv` 0.12+, with locked dependencies from `uv.lock` | CI pins `uv` 0.12.4; the receipt does not record the local `uv` version |
| Containers | Docker Desktop with the Linux container engine and Docker Compose 2.24.4 or newer | The Compose files use `!reset` and `!override`, which require a current Compose implementation |
| Inference | CPU-only | The Research Edition Compose file starts TEI with `--dtype float32` on CPU and Ollama with `CUDA_VISIBLE_DEVICES=-1`, Vulkan disabled and `num_gpu=0` |
| GPU | Not required for the Preview | The separate RES-138 benchmark uses an A100-80GB host; it is not part of the Preview runtime |

Ubuntu is exercised by CI for the quality, browser, and integration jobs. This
guide does not document or qualify Ubuntu as an end-user installation target.

## Observed service memory

The clean-install receipt records one warmed `docker stats`-style snapshot on a
single Windows 11 workstation on 2026-11-07. These are observations, not peaks,
reservations, or minimums.

| Service | Observed memory | Context |
| --- | --- | --- |
| OpenSearch | 1.587 GiB | Running Research Edition container |
| TEI | 2.964 GiB | Warm CPU embedding service |
| Qdrant | 68.94 MiB | Running container |
| PostgreSQL | 51.88 MiB | Running container |
| Ollama | 47.73 MiB | Idle container, model not loaded |
| DynamisRAG application | 104,837,120 bytes (about 100 MiB) | Host process working set |

Docker Desktop/VM overhead is outside those container figures and is
**NOT MEASURED**. Total host RAM is **NOT MEASURED**. A minimum RAM requirement
cannot be derived from this snapshot.

Evidence:
[`clean-install-receipt.json`](evidence/research-edition-sample-v1/clean-install-receipt.json).

## Available Ollama model-memory evidence

The strongest available generator measurement comes from the historical
LIVE-RAG-002 CPU run:

| Observation | Value |
| --- | --- |
| Installed model file | 2,497,293,931 bytes (2.33 GiB) |
| Ollama-reported loaded model size | 4,105,380,864 bytes (3.82 GiB) |
| GPU memory used | 0 bytes |
| Container memory limit | 6 GiB |
| Sampled container memory | 5.539 GiB, explicitly not a measured peak |
| Runtime profile | 4 CPU cores, 4 threads, 8,192-token context, `qwen3:4b` Q4_K_M, one loaded model, one parallel request |

This is a selector-v1 historical observation, not a fresh selector-v2
qualification. It shows that the generator fit and ran under a 6 GiB container
limit on the observed host. It does not establish a minimum host RAM requirement.

Evidence:
[`model-provenance.json`](evidence/live-rag-002/model-provenance.json) and
[`live-rag-002.md`](live-rag-002.md).

## Storage considerations

| Item | Evidence-backed size | Notes |
| --- | --- | --- |
| Bundled three-article JATS sample | 250,718 bytes (0.24 MiB) | Source bytes only |
| `qwen3:4b` Q4_K_M model download | 2,497,293,931 bytes (2.33 GiB) | Installed Ollama model file |
| Qwen3 Embedding 0.6B TEI weights | **NOT MEASURED** | The repository does not record the downloaded size |
| Docker images and named volumes | **NOT MEASURED** | Includes PostgreSQL, Qdrant, OpenSearch, TEI and Ollama state |
| Python environment and locked dependencies | **NOT MEASURED** | Depends on the local `uv` cache and platform |
| Database and vector growth for other corpora | **NOT MEASURED** | The sample is not a capacity model |

Reserve free disk for the Docker images, both model downloads, and the named
volumes. No minimum free-disk figure is claimed.

## CPU-only inference profile

The documented Research Edition profile is deliberately serial and CPU-bound:

- TEI serves `Qwen/Qwen3-Embedding-0.6B` at the pinned revision with `float32`,
  `--max-batch-tokens 2048`, `--max-client-batch-size 1`,
  `--tokenization-workers 1`, and `OMP_NUM_THREADS=4` / `MKL_NUM_THREADS=4`.
- Ollama serves `qwen3:4b` Q4_K_M with `num_gpu=0`, four threads, an 8,192-token
  context, at most 1,024 generated tokens, one loaded model, and one parallel
  request.
- The application permits one answer pipeline at a time; a competing request
  fails with `generation_busy`.

This is the profile that the recorded acceptance used. It is not a tuned
throughput or concurrency configuration.

## Observed performance, not a benchmark

| Operation | Observation | Scope |
| --- | --- | --- |
| BM25 retrieval | 19.9 ms | One run, clean-install receipt |
| Dense retrieval | 12.8 ms | One run, clean-install receipt |
| Hybrid retrieval | 23.0 ms | One run, clean-install receipt |
| Browser answer, total | 44.588 s | One extractive answer; retrieval 1.225 s, generation 43.294 s, citation validation 0.069 s |
| Historical cold generation | 134.173 s | Selector-v1 run, not the current acceptance |
| Historical warm generation | 71.336 s | Selector-v1 run, not the current acceptance |

These are single local observations. They are not percentiles, a latency
benchmark, a GPU comparison, or a capacity claim.

Known limitations:

- TEI cold start and model-download time: **NOT MEASURED**.
- Concurrent request throughput and multi-user behaviour: **NOT MEASURED**.
- Long-running stability and recovery under load: **NOT MEASURED**.
- Cross-host reproducibility of these timings: **NOT MEASURED**.
- Production reliability, availability, or capacity: **not qualified**.

## What this page does not claim

- No minimum RAM requirement is established.
- No minimum free-disk requirement is established.
- No performance benchmark or percentile is established.
- No production reliability or capacity qualification is established.
- No claim is made that a different host, model cache state, or Docker Desktop
  configuration will reproduce the observed memory figures.
