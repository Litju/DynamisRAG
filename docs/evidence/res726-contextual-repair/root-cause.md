# RES-726 contextual acceptance failure diagnosis

## Evidence inventory

No `release-blocker.json` is present in the checkout or retained RES-726 evidence directory. The diagnosis below uses the original event captures and run records already committed; those artifacts are left unchanged.

| Artifact | SHA-256 | Finding |
| --- | --- | --- |
| `res638-research-workflow/browser/turn1-events.sse` | `68aa58910b49563b6b687f00bf5380255d57beae8c7c1a0666ec94c8100b8a9d` | Four licensed works were projected to 111 passages. The provider authorization gate refused model calls, so graph fallback searched once before and once after discovery. Retrieval returned canonical passages, then the evidence gate abstained with `insufficient_query_coverage`. |
| `res638-research-workflow/browser/turn2-events.sse` | `8c9489e2e385f495a981743bfb78e05178a2759c03182a3a69c7b81723a560b6` | The comparison follow-up had canonical candidates marked admissible, but a global missing-qualifier check rejected the whole run for `3d` (and, in this older run, the generic word `such`). No model calls occurred; acquisition reselected the same four papers. |
| `res638-research-workflow/live-v2/receipt.json` | `0b409dc7c2cc8fca8105a3125ef4302e369e7c3db65e9f328972537ceaac21ff` | A separate real-model run was rejected by canonical claim checks for `content_not_in_quote`, `negation_mismatch`, and `hedge_dropped`. This is unsafe model paraphrasing, not a verifier defect. |
| `res638-research-workflow/live-v2/attempt1/turn1-run.json` | `260fc7d806a8fed501a96270e2e84f6c401b355ea33cbd3ea1a438bc91d266f5` | Preserves the rejected answer and canonical rejection outcome. |
| `res638-research-workflow/flash-v3/turn1-run.json` | `5b19a96146a27baa7f4d0b84584988b73885a9b8821cfcee8a03190aa0da9a51` | Later DeepSeek V4.1 Flash run produced seven independently verified citations from canonical passages. |
| `res638-research-workflow/flash-v3/turn2-run.json` | `b5a4234140ab8513fac4f40f602d4840879f90cc8115343ab0514c3fb4e4b489` | The focused contextual comparison produced four verified citations, including the primary study's stated comparison against force plates and motion capture. It correctly described the motion-capture result as an aim rather than a result. |
| `res726-autonomous-research/turn-2-events.json` | `f867d43d9a89f8be4e1af88bfbf9029614b44def23cf794163771f99cd122a57` | A later authorized model-directed follow-up retrieved the canonical PMC7850492 passage explicitly mentioning force plates and 3D motion capture; the run completed with four verified citations. |

## Established causes

1. **Evidence-gate false abstention on broad follow-ups.** `_enumerated_aspects` required an enumerated list to end its sentence. In the failed question, `validity, reliability, practical applications and limitations` was followed by the scientific topic (`... force-velocity profiling ... in resistance training`). The parser therefore left all listed aspects in the per-passage core requirement. Distinct canonical passages supporting different aspects could not jointly clear the per-passage threshold. The phrase `the evidence say` also left the framing word `say` at full topical weight.
2. **Retrieval fallback could not recover a missed passage without model authorization.** The failed comparison's candidates supported force-plate/reference-system language, but the top results did not include the corpus passage mentioning 3D motion capture. The recorded run had zero model calls due to `provider_authorization_unverified`; its graph fallback repeated the broad search, and discovery added no new work. The gate accurately recorded that the retrieved set lacked `3d`, but the run did not establish that the corpus lacked the evidence. The later authorized RES-726 model-directed run searched the same subject, retrieved the canonical PMC7850492 passage, and cited it.
3. **Citation rejection was correct.** The rejected response in `live-v2` used `gpt-6-luna` (not the current DeepSeek V4.1 Flash model), added interpretation, and changed the source's certainty or polarity. Canonical verification rejected those claims for concrete mismatches. The later accepted Flash run shows model behavior can vary; exact source verification must remain in place, with abstention when a repair cannot produce a faithful citation.

## Execution-path checks

The failed run events begin at `resolve_conversation_context`; the comparison turn is marked `context_resolution: none_needed` because it names LPT and its comparator explicitly. The thread loader still serializes the durable summary and last four messages into the research state; the model tool loop treats them as untrusted context and uses the current turn as the search question when it has its own scientific content. Thus this captured follow-up was not caused by dropped conversation history or an incorrect pronoun rewrite.

The citation-failure record contains the model name, generated claim text, submitted passage keys, quote text and verifier rejection reasons. Submitted keys resolve to canonical passages, but the generated claims add or change meaning. In the accepted DeepSeek Flash v3 record, the four comparison citations map to canonical passage/span keys; the answer distinguishes a study aim from a reported result. No change to citation ID handling or canonical verification is warranted.

The historical failures had multiple independent causes. The repair changes question-shape recognition only. It does not lower relevance thresholds, relax canonical citation checks, or convert the failed fallback run into evidence of supported scientific claims.

## Limits of this reconstruction

The cited captures contain no LangGraph checkpoint dump or standalone `release-blocker.json`. Retrieval trace IDs and canonical passage keys are retained in events, but the original live database snapshot needed to replay those exact IDs is not included. The raw model failure receipt and successful Flash citation records are preserved. Fresh live-model and hosted checks are required for release acceptance.
