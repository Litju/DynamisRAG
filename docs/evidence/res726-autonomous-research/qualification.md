# RES-726 autonomous research qualification

The owner-operated acceptance run on 2026-11-20 exercised Next.js and Eve, FastAPI and LangGraph, Europe PMC and OpenAlex, canonical PostgreSQL, Qwen3 TEI, Qdrant, and DeepSeek V4.1 Flash through OpenCode Go. Automatic Research was selected by default with zero checked papers. The captured create request used automatic scope. [Acceptance receipt](acceptance-receipt.json).

## Observed research

| Turn | Outcome | Recorded evidence |
| --- | --- | --- |
| Force–velocity profiling with linear position transducers | Completed with six canonical quotations across four studies. The model requested a library listing and twelve hybrid searches, then proposed seven claims. | [Run events](turn-1-events.json), [answer screenshot](01-initial-answer.png) |
| Force plates and other reference systems | Completed with four canonical quotations. The structural comparison reported incompatible outcomes or units and the answer identified gaps in direct agreement evidence. | [Run events](turn-2-events.json), [comparison screenshot](03-follow-up-answer.png) |
| Competitive weightlifting literature | Model-directed discovery found 44 records and deduplicated them to 40 works. Two permitted full texts were ingested. The scientific answer abstained with zero verified citations. | [Run events](turn-3-events.json), [discovery screenshot](05-discovery-answer.png) |
| Restricted force-plate question | The explicit allowlist restricted the thread to one selected paper. The run abstained with zero verified citations. | [Acceptance receipt](acceptance-receipt.json), [restricted screenshot](04-restricted-abstention.png) |
| Narrow follow-up on snatch barbell velocity | Completed with four canonical quotations from the newly ingested PMC7281229. Four citation controls were present and the canonical source drawer opened. | [Run receipt](turn-4-receipt.json), [answer screenshot](06-enriched-corpus-answer.png), [source drawer screenshot](02-source-inspection.png) |

The source drawer screenshot captures its loading state. The original receipts retain quotation text, passage identifiers, JATS anchors and character ranges. The initial capture recorded zero citation controls at its inspection step. The later run recorded four controls and an opened drawer. [Acceptance receipt](acceptance-receipt.json), [enriched-corpus receipt](turn-4-receipt.json).

## Source and rights records

- The initial run cites PMC7850492 (force–velocity profiling methods), PMC12608286 (force plates and linear position transducer), PMC6316018 (commercial velocity and power device), and PMC12526633 (back-squat angular velocity). The reference-system follow-up cites PMC7850492, PMC12608286 and PMC6316018. [Acceptance receipt](acceptance-receipt.json).
- The two CC BY 4.0 additions were PMC7281229, DOI `10.3390/sports8050059`, PMID `32397110`, and PMC13108039, DOI `10.3390/jfmk11020147`, PMID `42029515`. The new projection contained six documents and 175 passages. It embedded 50 new passages and reused 125 vectors with `Qwen/Qwen3-Embedding-0.6B` at revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` and 512 dimensions. [Discovery and projection events](turn-3-events.json).
- The repository rights gate rejected PMC11728576, DOI `10.70252/mfjk8861`, PMID `39807279`, *Validity and Reliability of a Linear Position Transducer to Measure Barbell Velocity, Duration, and Displacement During the Bench Press*. The captured run telemetry recorded Europe PMC's `cc by-nd` licence and the rejection at `repository_rights`. [Discovery run events](turn-3-events.json).

## Inference usage

These five run records were read from the owner agent store in read-only mode using the saved run IDs. All inference roles used `deepseek-v4.1-flash`. Costs are the application's estimates in USD.

| Run | Requests | Prompt tokens | Completion tokens | Total tokens | Estimated USD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial `run-0efac6b4ce824e77` | 4 | 28,597 | 7,037 | 35,634 | 0.00851175 |
| Reference comparison `run-b3d2a7a2be4e4693` | 5 | 27,839 | 5,984 | 33,823 | 0.00776625 |
| Discovery `run-cf436bcbfb62459c` | 6 | 45,882 | 5,768 | 51,650 | 0.01034310 |
| Restricted `run-21b7a5c4f7404907` | 4 | 20,895 | 6,103 | 26,998 | 0.00679605 |
| Enriched `run-55fb1177f85b4ccb` | 4 | 19,613 | 5,013 | 24,626 | 0.00594975 |
| **Total** | **23** | **142,826** | **29,905** | **172,731** | **0.03936690** |

The discovery tool recorded zero acquisition-internal model requests. Its parent research run recorded six model calls and used the model to direct the tool. The captured projection took 313.3 seconds, including 222.0 seconds of embeddings. The pre-ingestion browser capture shows the worker-connection state during an earlier attempt. [Discovery events](turn-3-events.json), [connection screenshot](error.png).

## Research scope and limitations

- Explicit `allowed_version_keys` are the immutable user authorization boundary. Automatic evidence selection can search the corpus and request permitted discovery. Conversational study references guide only the relevant turn. Durable study exclusions remove evidence from the thread's analysis within that authorization boundary.
- The owner acceptance started with four seeded papers. It measured a warm-corpus expansion and a follow-up after projection, with limited evidence on unseeded cold starts and the time to a first answer on the protected Preview.
- Comparisons require two canonical sides and retain required references. A missing or ambiguous reference, absent named comparator, or unsupported numeric cutoff can cause abstention. Unit-aware evaluation of numeric inequalities remains unavailable.
- Findings, comparisons, and paraphrases retain `MODEL_PROPOSED / PROPOSED` status. Quote matches and source coordinates establish provenance. The acceptance questions cover a small body of sports-science literature and leave direct measurement-agreement, statistical pooling, study validity and wider scientific applicability unresolved.
- The Preview uses separate state and owner-only access under the documented operating conditions. The recorded live run used the owner-operated loopback workspace. [Protected Preview operations](../../res726-workspace-operations.md).

## Engineering verification

The independent Codex review passed on code revision `f3239bd679aa8ecc534e3c39345b8fc59dec9126` after corrections to study exclusions, comparison admission, reference resolution, comparison matrices and numeric cutoffs. The final review reproduced the prior count and local-exclusion cases and passed the targeted regressions.

Local checks passed for Ruff, strict Pyright, 66 focused regressions, 27 workspace tests, the workspace typecheck and build, eight gateway tests, the gateway dry run and three Python Chromium tests. The full local unit run passed 2,905 tests on the preceding code revision. All nine applicable workspace browser cases passed across an isolated run and a retry of the sign-in test with a longer cold-start wait. Two environment-specific browser cases were skipped. The original owner acceptance scenarios were preserved as captured.

The local integration suite covered 306 cases. The first run passed 275. All 31 cases affected by the loopback OpenSearch certificate setting passed in a 52-case retry that also reran the PostgreSQL store contracts. The retry used the repository test profile on the isolated cluster.

The original twelve evidence files retain their byte identities at this path.

GitHub Actions passed all six jobs on the exact reviewed code revision, including Linux and Windows lint, types and unit tests, Chromium, migrations and live integration, the workspace and the edge gateway. [CI](https://github.com/Litju/DynamisRAG/actions/runs/44763824893), [workspace and gateway](https://github.com/Litju/DynamisRAG/actions/runs/44763824820).
