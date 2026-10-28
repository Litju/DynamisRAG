# LIVE-RAG-002: locally generated scientific answers

The running application accepts **POST
`http://127.0.0.1:8021/answer/scientific`**. It retrieves from the original
75-passage scientific collection, selects up to five suitable verified evidence passages, calls
the local pretrained `qwen3:4b` model through Ollama, and returns generated
extractive claims with independently verified canonical citations.

This is an **extractive answer contract**, not free-form synthesis. The model
selects complete finding sentences verbatim and supplies their passage keys.
The server validates them and joins the claims into the answer with citation
indices. It does not hardcode the knee-flexion answer or select a required winning
document. Original BM25/dense ranks, scores and RRF traces remain in the response.

## Run the real question

PowerShell from the repository (the API is already running):

```powershell
$body = @{
    question = 'How does starting knee flexion angle influence vastus medialis and vastus lateralis activation during resisted plyometric exercise?'
} | ConvertTo-Json
$clock = [System.Diagnostics.Stopwatch]::StartNew()
$answer = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8021/answer/scientific -ContentType 'application/json' -Body $body -TimeoutSec 430
$clock.Stop()
$answer.answer
$answer.claims | ConvertTo-Json -Depth 20
$answer.timings
$clock.Elapsed.TotalMilliseconds
```

The historical cold/warm/abstention execution and independent SQL verification used:

```powershell
uv run python scripts/probe_scientific_answer.py
```

These retained receipts were generated under the prior selector policy
`body-findings-sentence-excerpts-v1`. They remain unchanged historical evidence;
they do not qualify real generation under selector v2. No fresh real CPU
generation run has been executed under selector v2. The repaired code is validated
by deterministic tests with mocked Ollama output, including in CI.

The historical cold and warm requests both returned HTTP 200, `status: supported`,
five selected evidence passages and three generated claims. The generated answer
was:

> Study results demonstrated notable variations in muscle activity between
> 60°and 90° positions, particularly in favor of the 90° knee flexion position. [1]
>
> Our findings suggest that initiating resisted plyometric exercises without
> jumping at a 90° knee flexion position results in significantly increased muscle
> activity in both the VM and VL muscles, regardless of limb dominance. [2]
>
> The results revealed a significantly higher level of muscle activity in the VM
> and VL muscles when the exercises were performed at a 90-degree knee flexion
> position in both the D and ND legs. [3]

These are exact source quotations selected by the local model, including source
typography. All three cite [PMC10780924](https://pmc.ncbi.nlm.nih.gov/articles/PMC10780924/),
DOI **10.3390/s24010044**, *Impact of Starting Knee Flexion Angle on Muscle
Activity and Performance during Plyometrics without Jumping*. Its CC BY 4.0
notice and contributor attribution remain in the existing real-science paper
receipt. Canonical document version:
`958583513329768db12d9cd6c4b6730ed16d8b83ba5bc5203d5856983dd22643`.

The generated passage keys, in claim order, are:

```text
c15877f3938f3a25bc734c99fe15cfb29a25805bdfc2ec7b124ea41e41295cf0
a2a28740ccd4e71ab85adc2970a962e5ab7ead39563ff9d217f9b8dfed84c877
a73f69fb8c9eb8d3a4c266f8af55febd81c892052838fe2577fd8a2f7e79ca86
```

Each returned claim includes its full canonical source passage, DOI/PMCID/PMID,
document version, paragraph/JATS source spans, exact supporting excerpt and its
character offsets within the passage. The separate operator verifier re-read
PostgreSQL and checked all of these after each actual HTTP request.

## Actual generation runtime and resources

No Ollama installation/server was present initially. The implementation installed
one local Docker runtime and downloaded only the requested pretrained generation
model. Installed facts were verified using `/api/version`, `/api/tags`,
`/api/show`, the installed manifest's actual SHA-256, and `/api/ps`:

```text
model: qwen3:4b
digest: 359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7
format: gguf
family: qwen3
parameter_size: 4.0B
quantization: Q4_K_M
runtime: Ollama 0.13.5
loaded context: 8192 tokens
GPU memory: 0 bytes
```

The container has four CPU cores, a 6 GiB memory limit, no GPU device requests,
`CUDA_VISIBLE_DEVICES=-1`, and Vulkan disabled. Sampled memory was 5.539 GiB,
not a measured peak. Ollama allows one loaded model, one parallel generation and
one queued request. The application additionally permits only one answer pipeline
at a time; competing requests fail with `generation_busy`.

Each generation uses the supported native [`POST /api/chat`](https://docs.ollama.com/api/chat)
API with a JSON schema, `stream=false`, `think=false`, `num_gpu=0`, four threads,
8,192 context tokens and at most 1,024 generated tokens. Temperature is zero and
seed is 42. The client is loopback-only, disables proxies/redirects, permits one
attempt, bounds provider responses to 64 KiB and uses a 300-second read timeout.
Models remain resident for up to ten idle minutes. The installed model, digest,
quantization and runtime are checked before and after generation. A loaded-model
check using [`GET /api/ps`](https://docs.ollama.com/api/ps) requires the same digest,
the pinned context length, and zero GPU memory. Model substitutions fail closed.

The [official model entry](https://ollama.com/library/qwen3:4b) identifies the
pretrained quantized model; its installed manifest, layer digests, Apache 2.0
license, runtime/deployment facts and exact system policy are retained in
`model-provenance.json`. The API also returns a generation-policy SHA-256.

## Cold, warm and abstention latency

Cold means **model not resident**: `/api/ps` returned `models: []` immediately
before the final cold request. It does not mean an empty download/disk cache.
Ollama had expired the idle model naturally; no model unload command executed.
The paired requests used the same model, policy, question and corpus.

Observed cold request, milliseconds:

```json
{
  "retrieval_ms": 4879.112700000405,
  "generation_ms": 134173.10029999862,
  "citation_validation_ms": 43.88679999829037,
  "server_pipeline_total_ms": 139096.09979999732,
  "complete_http_ms": 139148.3376000033
}
```

Observed warm request, milliseconds:

```json
{
  "retrieval_ms": 849.1026999981841,
  "generation_ms": 71336.39969999786,
  "citation_validation_ms": 27.21400000154972,
  "server_pipeline_total_ms": 72212.71639999759,
  "complete_http_ms": 72217.19460000168
}
```

Both generated 347 output tokens from 1,277 prompt tokens. Backend-reported model
load time was 6.238 seconds cold and 0.094 seconds warm. Generation timing includes
client identity checks/transport; backend load/prefill/decode timings are retained
separately in nanoseconds. HTTP time includes serialization and transfer. These
are observed CPU smoke requests, not a latency benchmark or a GPU comparison.
Earlier retrieval-only timings are not answer-generation timings.

The out-of-corpus question **“What determines neutrino mass ordering and leptonic
CP violation?”** returned HTTP 200 with `insufficient_evidence`, `answer: null`,
zero claims and `generation_invoked: false`, in **538.4441 ms** over HTTP. The
conservative relevance gate rejected the available sports-science passages, so
Ollama was not asked to answer from unrelated evidence.

An initial development request safely abstained because the prompt omitted an
explicit positive-status instruction. That receipt is retained separately and
is not counted as the successful cold answer above. The corrected prompt was
validated with the real cold and warm requests. A combined model-unload/API-restart
command was rejected by automatic approval review and did not execute; the model
was left to its normal idle expiry.

## Grounding and failure contract

The question schema is strict and bounded to 512 characters; body size is capped
at 8 KiB, including chunked requests. Model parameters cannot be supplied by the
caller. The server invokes the existing live scientific retrieval service and
retains its unmodified top-20 RRF trace and independent 50-candidate lanes.

Selection takes up to five suitable passages in original RRF order. Front/back
matter, editorial/author information, methods/background/hypothesis sections,
instruction-like content and low question overlap are excluded with passage keys,
original RRF ranks and explicit reasons. Selector v2 normalizes Unicode with NFKC
and case folding, preserving scientific abbreviations such as VM, VL, RSI and CMJ,
numerical values, decimal measurements and angles such as 60° and 90°. It matches
complete tokens, so VM does not match SVM and 90° does not match 190°. Common
question stopwords are filtered without imposing a four-character minimum.
The overlap threshold is `min(2, number of informative question tokens)`;
questions with no informative tokens are rejected. Thus one informative token
requires one match, while longer questions still require at least two matches.
Lexical overlap is a conservative eligibility gate, not a semantic relevance
model; it can abstain on suitable paraphrases and does not establish sufficiency.

Complete sentence prefixes of at most 800 UTF-8 bytes are used as generation
excerpts. Full canonical text and all JATS spans remain available in the response.
The combined system/question/evidence prompt is capped at 6,000 bytes, reserving
context headroom. One suitable, independently verified passage is sufficient to
attempt extractive generation; zero suitable passages cause `insufficient_evidence`
without model inference. Passage count alone does not establish scientific
validity or answer sufficiency. The prompt requires Ollama to abstain if the
available evidence does not directly answer the question. Only the question, source keys and bounded
excerpts go to the model, serialized as untrusted JSON data. No tools, credentials,
private paths, external search or document instructions are given authority.

The model output schema allows only a sufficiency status and up to three claims,
each with a passage key and exact quotation. Canonical identifiers never come
from model output: the server attaches them from verified retrieval evidence.
Fresh PostgreSQL verification runs before and after generation. Every accepted
quote must occur in the actual supplied excerpt and full source passage and must
start/end at sentence boundaries. Unknown citations, fabricated numbers/wording,
duplicated quotations and unsupported extra identifier fields are rejected.
Invalid generated citations, quotations, identifiers or source membership cause a
sanitized HTTP 503 `generation_invalid_output` failure. All claims are validated
atomically: one invalid claim rejects the entire generated answer, with no partial
acceptance. Invalid or truncated provider JSON uses the same failure path.
These contract failures are distinct from legitimate `insufficient_evidence`
responses caused by zero usable evidence or a valid model abstention
(`reason: model_abstained`). They are not scientific abstentions.

**Structural validation does not prove semantic entailment, study validity or
applicability.** An authentic quotation can still be selected for the wrong
question or omit important surrounding qualifications. The model's sufficiency
judgment and instruction filtering are not general proofs of relevance or prompt
injection immunity. This endpoint is intentionally extractive; it does not claim
to provide unrestricted scientific synthesis or clinical recommendations.

Missing/unavailable/timed-out generation, identity drift, corrupted provenance,
or unavailable retrieval fail closed with fixed codes. Error responses/logs do
not echo model-server bodies, query text, article content or private paths.
Responses use UTF-8 JSON and `Cache-Control: no-store`; vector arrays are absent.
Original OpenSearch endpoints and default health behavior remain unchanged.

## Startup and verification commands

The existing container/API are already running. For a clean future setup, the
executed image tag was `0.13.5`; this equivalent command also pins its observed
image digest. Do not run it against an already existing container name:

```powershell
docker run -d --name dynamisrag-ollama-cpu --cpus 4 --memory 6g -p 127.0.0.1:11434:11434 -v dynamisrag-ollama-models:/root/.ollama -e OLLAMA_NUM_PARALLEL=1 -e OLLAMA_MAX_LOADED_MODELS=1 -e OLLAMA_MAX_QUEUE=1 -e OLLAMA_CONTEXT_LENGTH=8192 -e CUDA_VISIBLE_DEVICES=-1 -e OLLAMA_VULKAN=false ollama/ollama:0.13.5@sha256:2c9595c555fd70a28363489ac03bd5bf9e7c5bdf2890373c3a830ffd7252ce6d
docker exec dynamisrag-ollama-cpu ollama pull qwen3:4b
Invoke-RestMethod http://127.0.0.1:11434/api/version
Invoke-RestMethod http://127.0.0.1:11434/api/tags | ConvertTo-Json -Depth 8
Invoke-RestMethod -Method Post http://127.0.0.1:11434/api/show -ContentType 'application/json' -Body '{"model":"qwen3:4b"}'
Invoke-RestMethod http://127.0.0.1:11434/api/ps | ConvertTo-Json -Depth 8

$env:DYNAMISRAG_SCIENTIFIC_QDRANT_ENABLED = 'true'
$env:DYNAMISRAG_SCIENTIFIC_ANSWERS_ENABLED = 'true'
$env:DYNAMISRAG_SCIENTIFIC_EVIDENCE_DIRECTORY = 'docs/evidence/real-science-v1'
$apiProcess = Start-Process -FilePath (Get-Command uv).Source -ArgumentList @('run','--project','E:/Data/Projects/DynamisRAG','uvicorn','--factory','dynamisrag.application:create_app','--host','127.0.0.1','--port','8021','--no-access-log') -WorkingDirectory 'E:/Data' -WindowStyle Hidden -RedirectStandardOutput 'E:/Data/Projects/DynamisRAG/.tmp/live-rag-002-final-server.out.log' -RedirectStandardError 'E:/Data/Projects/DynamisRAG/.tmp/live-rag-002-final-server.err.log' -PassThru
```

Both feature flags default to false. Construction is free of network/database
I/O; lifespan startup qualifies retrieval and verifies installed generator
identity without loading the model. The app reuses the retrieval service, SQL
pool and dedicated HTTP clients and closes all resources through `ExitStack`.
It does not pull models during serving. A missing model requires operator setup
and an application restart; it does not trigger an automatic substitute/download.

```powershell
uv run pytest tests/unit/test_scientific_answers.py tests/unit/test_scientific_retrieval.py tests/unit/test_health_endpoints.py tests/unit/test_config.py tests/unit/test_benchmark_boundaries.py -q --basetemp .tmp/live-rag-002-focused-final
uv run pytest tests/integration/test_scientific_citations.py tests/integration/test_qdrant_live.py -m integration -q --basetemp .tmp/live-rag-002-integration
uv run pytest --basetemp .tmp/live-rag-002-full-unit -q
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

The original local execution qualification passed **2,312 unit tests**, **147
focused tests** and **16 live integration tests**. After the three independently
reviewed repairs, commit `0ec5cb1f8454833e848105906f4980281043d96e` passed
**2,340 unit tests on Linux and Windows**, **182 integration tests**, Ruff
lint/format and strict Pyright in [exact-repaired-head CI](https://github.com/Litju/DynamisRAG/actions/runs/40675734211).
Locally, **177 focused tests** and the rolled-back PostgreSQL citation fixture
passed. The regressions first reproduced **17 failures** before production fixes.
They cover abbreviations/Unicode/numerical token matching, unrelated questions,
zero/one/two/three/five eligible passages and atomic invalid-generation failures.
Tests use explicit mock generation in CI; no model download/inference is part of CI.
This deterministic validation does not replace a real selector-v2 CPU generation
qualification; that execution remains unperformed.
Coverage includes missing/fabricated citations, unsupported wording/numbers,
corrupt canonical metadata, model/quantization/runtime/GPU mismatches, unavailable
and timed-out models, invalid/truncated JSON, injected document instructions,
empty/out-of-corpus evidence, concurrency, opt-in startup and cleanup.

## Retained execution evidence

The cold/warm and abstention receipts below retain their original bytes and prior
selector-v1 policy hashes. Their model identity, provenance and latency facts are
historical observations, not fresh selector-v2 results. The three-paper,
75-passage corpus limits coverage; historical CPU requests took about 139 seconds
cold and 72 seconds warm. Neither latency nor semantic entailment is guaranteed.

* [Readable generated answer and canonical citations](evidence/live-rag-002/answer-log.txt)
* [Complete cold HTTP request/response](evidence/live-rag-002/cold.json)
* [Complete warm HTTP request/response](evidence/live-rag-002/warm.json)
* [Out-of-corpus abstention](evidence/live-rag-002/out-of-corpus.json)
* [Installed model/runtime/manifest/deployment provenance](evidence/live-rag-002/model-provenance.json)
* [Empty model residency before cold request](evidence/live-rag-002/model-before-first-request.json)
* [All original collection/vector hashes and canonical counts unchanged](evidence/live-rag-002/state-unchanged.json)
* [Initial safely abstaining development request](evidence/live-rag-002/cold-initial-model-abstention.json)

No article ingestion, document embedding, collection write, training, reranker,
hosted inference or RES-142 run was performed. The original trusted manifests
and embedding fingerprints are unchanged. This branch is stacked on PR #21;
neither PR #20 nor #21 is modified or merged.

## Owner integration preparation (no merges performed)

Verified stack before the documentation-only commit:

| PR | GitHub base | Head branch | Reviewed head SHA |
| --- | --- | --- | --- |
| #20 | `main` (`67edcc785af2084e2cc4d8c27b08bea43a5f4bd3`) | `feat/qdrant-vector-backend` | `54c2c4c2078621b8f0106fee00bbb54017496b05` |
| #21 | `feat/qdrant-vector-backend` | `feat/real-scientific-retrieval` | `ae1c5e1591c48df543363bfb420d98157f284df2` |
| #22 | `feat/real-scientific-retrieval` | `feat/grounded-scientific-answers` | repair: `0ec5cb1f8454833e848105906f4980281043d96e`; documentation commit follows |

Each parent head is an ancestor of its child. The layer diffs contain no changes
to frozen benchmark implementations or RES-142 artifacts. The benchmark-boundary
test changes only permit the pinned scientific CPU profile (requiring its model
identity to equal the existing profile) and extend boundary checks to generation.
PR #21 intentionally adds the licensed scientific receipts and numerical
manifests; PR #22 preserves them. Do not interpret that intentional addition as
a corpus rewrite.

Read-only integration preparation rechecked all 75 PostgreSQL passages and source
spans against the retained projection, reparsed all three original JATS files
against canonical paragraphs, and verified the real-science alias and all 75
Qdrant payloads/vectors. Counts remain 3 documents, 3 versions and 75 passages.
All five `real-science-v1` JSON files are byte-identical to PR #21's reviewed head.
Key SHA-256 values remain:

| Artifact | SHA-256 |
| --- | --- |
| Document embedding manifest | `46f96db3a02cf61374003475e3f006258ecbd790a4a78bd858b305c25f5104ff` |
| Query embedding manifest | `4ac70bedcc23ffc2d3acfcba46ffb42ad1b68de55e0cbad782678f1f4b83fb29` |
| Original numerical receipt | `8ede85766f656acccf7fd44b9c3fde627bd18289b74fe375dbf895d02ff4d240` |
| Real-science collection payload/vector snapshot | `2640a00053e5c8231e7db28873a42e7b9465830693f6c4c56f722c7ba2f78d2a` |

This check performed no inference, embedding or service writes. The retained
cold/warm receipts and frozen benchmark evidence were not modified.

Safe owner-authorized order, preserving commits and branches:

1. Merge **#20 into `main`** using a merge commit. Preserve its existing commits;
   do not squash/rebase-merge or delete its branch.
2. Reinspect **#21**. This is a branch-based chain (GitHub REST `stack: null`),
   not a registered native stack. With branches retained, merging #20 alone
   leaves #21 based on `feat/qdrant-vector-backend` and its layer diff unchanged.
   After owner authorization, retarget #21 to `main`; verify that the refreshed
   diff contains only its reviewed layer. Require fresh CI against the new base
   and exact head before merging **#21 into `main`** with a merge commit.
3. Reinspect **#22**. Merging #21 similarly leaves its retained branch and #22's
   current base/diff unchanged until retargeting. After owner authorization,
   retarget #22 to `main`, inspect the refreshed diff and require fresh exact-head
   CI against that base before merging **#22 into `main`** with a merge commit.

GitHub documents automatic base retargeting when a merged head branch is deleted
in [branch management](https://docs.github.com/en/pull-requests/how-tos/commit-changes/managing-branches-within-your-repository).
Branch deletion is excluded here; verify actual GitHub base/head fields after
each integration rather than relying on automatic retargeting. Do not merge
children into their old parent branches and assume they reach `main`.
Any changed head, base, conflict resolution or unexpected diff requires renewed
review and CI for that combination. Current green checks are prerequisites, not
qualification of future retargeted merge results. No merge, base change,
force-push, history rewrite or branch deletion is authorized by this preparation.
