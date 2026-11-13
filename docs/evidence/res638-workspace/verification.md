# RES-638 verification record — 2026-11-13

Code revision: `26b37a3` (four atomic RES-638 commits). Evidence-only changes and the test-mode label follow this revision. Final exact-head CI is tracked on the draft PR; local results below are not a substitute for that CI.

## Passed

| Check | Result |
| --- | --- |
| Complete Python unit suite | 2,792 passed, 309 integration/browser tests deselected; 126.22 s |
| Focused Python API/store tests | 32 passed, including authenticated exact-session lookup |
| Ruff lint / format | Passed; 344 files formatted |
| Full strict Pyright | 0 errors, 0 warnings |
| TypeScript boundary tests | 4 passed: account auth, signed identity, deterministic dispatch/compaction, split UTF-8 SSE |
| Next route generation + TypeScript | Passed |
| Eve 0.67.2 production build | Passed |
| Next.js production build | Passed |
| Synthetic-corpus browser E2E | 6 passed, 34.6 s; real Eve/HTTP/Python/LangGraph, deterministic test model |
| New PostgreSQL persistence contracts | 13 passed, including contention, ownership, crash recovery, fencing and ordering |
| Licensed corpus ingestion | 3 CC BY articles / 75 passages; second import idempotent; ~10 MB canonical database |
| Dedicated API/worker image build | Passed; non-root runtime, locked Python dependencies, pinned base digests |
| Licensed corpus container smoke | Completed, 4 verified citations, no error; ~0.62 s observed after enqueue |
| Licensed corpus browser E2E | 3 passed, 16.9 s: question/follow-up/source/refresh/isolation, mobile, comparison |

The synthetic fixture includes explicit slow/failure markers to prove cooperative cancellation, bounded worker failure and successful subsequent turns. It is loopback-only and never part of the production image. The licensed run uses the actual production app and a **separate worker container**, new persistent databases and the unchanged deterministic adapter. Neither run used paid model inference.

Windows' default shared pytest temporary directory was inaccessible. A long nested replacement also failed. The final full-suite run used the fresh shorter path `E:/Data/Projects/DynamisRAG/.tmp/u638-final` and passed. No ACLs or historical data were modified.

## Failed / not qualified

- **Licensed-corpus off-domain abstention failed.** The question “What is the orbital period of the exoplanet Kepler-22b?” returned unrelated, source-verified quotations with status `completed` under deterministic inference. This is a failed scientific negative case. It is not suppressed or counted as passing; the same browser assertion remains available with `WORKSPACE_LICENSED_CORPUS=1`. The smaller synthetic corpus abstains successfully. Source provenance checks alone do not establish relevance or entailment.
- The first rapid licensed browser batch hit the default 60 requests/minute user limit. Comparison passed after the loopback test stack was explicitly configured for 600 requests/minute. No production rate configuration or model budget was relaxed.
- Shared-preview automation twice failed with `Preview automation snapshot timed out after 15000ms.` The tab was closed. The browser evidence comes from local Chromium/Playwright instead.
- Qdrant/OpenSearch are private infrastructure definitions, not exercised dense retrieval. Current successful container runs use canonical lexical fallback. TEI/projection qualification and Qdrant's keyed/container deployment integration remain open.
- No authorized real-model E2E, adversarial scientific qualification, expert review, load test, hosted preview, backup restore drill or external TLS/access verification is claimed.

## Deployment gate

OpenCode Go external scientific-inference usage remains unapproved. The template leaves `DYNAMISRAG_AGENT_OPENCODE_GO_USAGE_APPROVED=false`; staging/production still reject deterministic inference. No new model provider or local Ollama service was added. A dedicated external Python host target/access is still missing. No Vercel deployment or cloud spend was created, and no public promotion/merge is authorized.

The screenshots below were captured against the new licensed PostgreSQL corpus with deterministic inference. They demonstrate UI/source plumbing, not scientific model qualification. Retain the packaged [CC BY attributions](../../../data/research-edition-sample-v1/LICENSES.md).

- [Licensed source inspection](licensed-source-inspection.png)
- [Licensed mobile layout](licensed-mobile.png)
