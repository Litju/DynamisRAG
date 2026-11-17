# RES-638 DeepSeek Flash acceptance

On 2026-11-17 the owner authorized paid API use and requested DeepSeek V4.1 Flash
instead of Luna, and asked to stop using Claude. This supersedes the original Luna-only
and zero-spend requirements. The implementation and validation in this continuation are
performed directly by Codex without Claude delegation.

All research roles now default to `deepseek-v4.1-flash` on the existing OpenCode Go
Chat Completions adapter, with `store=false`, a stable conversation session header,
native tools, low reasoning effort, and the existing bounded retries and cost budgets.
Approval remains false in committed configuration; the owner-approved local acceptance
environment enables the existing approval gate explicitly. Keys stay in ignored local files.

The unauthenticated official catalogue was checked on 2026-11-17:
<https://opencode.ai/inference/go/v1/models>. It advertises native tool calling,
`deepseek/deepseek-v4.1-flash-1017`, Chat Completions at
`https://opencode.ai/inference/go/openai/v1`, and low/high/max reasoning efforts.
The sole-model escalation must preserve low reasoning effort: Flash does not support medium.
Nominal prices are USD 0.15/M input and USD 0.60/M output, with published peak multipliers.

The browser acceptance uses the exact initial and follow-up questions in
`apps/workspace/e2e/research-workflow.spec.ts`. Fresh project
`dynamisrag-res638-accept4` starts with zero canonical documents and zero Qdrant collections.
Previous acceptance stores and the main checkout's uncommitted changes are preserved.

## Final result

**REAL_RESEARCH_WORKFLOW_OPERATIONAL**. The exact two-question Chromium acceptance
passed in 14.3 minutes against fresh project `dynamisrag-res638-accept6`.
The corpus started with zero documents and zero collections. The research model
generated its own searches, and both Europe PMC and OpenAlex returned live results.

| Turn | Discovery | Newly ingested | Projection | Model calls | Verified citations | Estimated USD |
| --- | --- | --- | --- | --- | --- | --- |
| Initial | 47 records / 44 unique works | 2 CC BY papers | 77 new vectors | 6 | 7 | 0.00835230 |
| Follow-up | 44 records / 38 unique works | 2 additional CC BY papers | 48 new / 77 reused vectors | 5 | 4 | 0.01050165 |

Initial ingestions: `10.3390/s25196047` (PMC12526633),
`10.1371/journal.pone.0245791` (PMC7850492). Follow-up ingestions:
`10.3390/s25216542` (PMC12608286), `10.3390/sports6040170` (PMC6316018).
The final collection is
`qdrant-passages-v1-76a2a8ee2a2a7943c00b9fb698d8e4e5f4c68756fc32bee3942c93827da45f5c`,
125 points, under alias `dynamisrag-research-workspace`. Qwen3 TEI retains the
pinned model/revision and 512-dimensional document embedding identity.

Both turns completed with model synthesis, native model-selected tools and inspectable
source spans. The second question resolved LPT using the earlier conversation and
requested new reference-system literature rather than repeating the first sources.
The [receipt and raw browser evidence](evidence/res638-research-workflow/flash-v3/receipt.json)
contain tool executions, all discovered identifiers/access decisions and projection identity.

## Scientific support review

The initial answer's seven accepted quotations are principally background statements,
explicitly labelled as such. It reports gaps instead of presenting these as primary
force-velocity validity or reliability measurements. This is a limited evidence readout,
not a systematic review or proof of all requested applications.

The follow-up includes two primary findings from PMC12608286: daily reliability of
countermovement depth/velocity (ICCs above 0.70) and LPT ICCs of 0.900 for depth and
0.863 for velocity. The quoted canonical Results text contains those exact values.
These findings concern countermovement jumps, not all barbell resistance exercises or
extrapolated F0/V0. The answer identifies the motion-capture comparison as a study aim
and explicitly says the retrieved quotes do not establish criterion-validity results.
It distinguishes reproducibility from accuracy and refuses numerical pooling across
incompatible outcomes. This source check is not expert review: every conclusion remains
`MODEL_PROPOSED / PROPOSED`, and quotation verification is separate from scientific support.

## Observed failures and repairs

The first Flash run ingested 135 passages but synthesis exhausted the 2,048-token cap,
then timed out during recovery. Only sole-Flash synthesis now has an 8,192-token cap
and 120-second timeout; tool calls retain 2,048 tokens/60 seconds. Failed calls without
reported usage conservatively reserve the output allowance. Existing price/token/request
budgets still apply.

Active research previously lost its workspace cookie after 15 minutes. Authenticated
activity now renews the 15-minute idle window; expired/revoked tokens still fail.
A second attempt reached TEI before warmup completed. Compose now waits for TEI's
health endpoint before starting acceptance. These failures are retained in the receipt.

Successful-run nominal usage totals USD 0.01885395. An earlier aborted acquisition
reported USD 0.00091425; the earlier synthesis failures incurred additional usage that
cannot be determined exactly from the client. No claim is made about actual billed totals.

Validation: 2,874 unit tests passed before the final narrow timeout/health changes;
subsequent targeted routing tests, workspace tests (20), Ruff and full Pyright/type checks
passed. Exact-head GitHub CI is recorded on PR #35. No public deployment or merge.
