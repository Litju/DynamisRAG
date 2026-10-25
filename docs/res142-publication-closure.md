# RES-142 publication and private-evidence closure audit

Audit baseline: PR #18 HEAD `dfc01c1f06b3d8925c6e7bd287c60cb075933346`, 2026-10-25. This audit reads existing evidence and public GitHub metadata only. No scientific calculation, inference, retrieval, or Colab session was repeated. The scientific analysis and its frozen input identities remain unchanged.

## Historical public exposure

The following are confirmed content events, not merely occurrences of field names in source code. The originals remain available in public commit views and commit-specific review diffs. Descendant commits inherit earlier content until it is changed.

| Commit | Affected path | Verified exposure |
| --- | --- | --- |
| `4001da4fc1c399e05bd26d2e55776f93842be50f` | `reports/res142-b2-scientific-review/evaluation.json` | Actual per-query metrics, judgment/pool cardinalities, positive-at-10 counts and positive relevance rank arrays; also aggregate judgment counts. |
| `4001da4fc1c399e05bd26d2e55776f93842be50f` | `reports/res142-b2-scientific-review/scientific-review.md` | Query regression table containing an actual query identity, positive rank movement and pool/judgment counts. |
| `4001da4fc1c399e05bd26d2e55776f93842be50f` | `reports/res142-b2-scientific-review/verification-receipt.json` | Aggregate baseline judgment cardinality, not per-query labels. |
| `dfc01c1f06b3d8925c6e7bd287c60cb075933346` | `reports/res142-b2-scientific-review/evaluation.json`, `scientific-review.md` | Tip contents removed actual per-query rows, but this commit's deletion diff still contains the earlier rows. The evaluation retained aggregate positive-qrel-query cardinality. |
| `dfc01c1f06b3d8925c6e7bd287c60cb075933346` | `reports/res142-b2-scientific-review/verification-receipt.json` | Aggregate baseline judgment cardinality remained. |
| `a4c984c8498d1633056a9f08336998295e529ee5` | `docs/res142-scifact-pilot-preparation.md` | Earlier aggregate pilot judgment cardinality in the baseline description. |
| `f86670b5d98f795e4e752572112b085d35d05b8e` | `docs/res142-b2-execution-harness.md` | Earlier full-source and selected-pilot judgment cardinalities. |

This closure commit also removes direct aggregate-cardinality fields; its deletion patch necessarily retains those earlier public values. This is another reason current-source redaction alone cannot close the historical distribution question.

The current combined PR #18 diff against base `45697089fbf55b5e71d72b44c80ffd1086daadd0` no longer includes actual per-query study rows. It does include field names in implementation and synthetic tests; those are not actual study annotations. Selecting the original commit or the repair commit still exposes the historical annotation-bearing addition/deletion patches.

GitHub's accessible PR #18 `userContentEdits` returned annotation-bearing description revisions timestamped `2026-10-25T06:08:03Z`, `2026-10-25T06:12:28Z` and `2026-10-25T06:47:25Z`. These revisions describe an actual query's positive-document rank movement. Subsequent returned revisions and the current body omit that per-query statement. Editing the current description does not delete its revision history.

The audit inspected all returned description revisions for PRs #16 (five), #17 (two), and #18 (six), plus current issue comments, submitted reviews, inline review comments and file diffs for all three PRs. No additional actual per-query study annotation disclosure was identified in #16/#17 descriptions or in the inspected review/comment bodies. This is bounded to accessible remote refs/API records; deleted refs, external forks, clones, caches, emails, downloads and inaccessible GitHub storage are not verified. A100 test XML matches were test identifiers, not study judgment counts.

Raw GitHub snapshots and the content-event inventory are retained locally under `.tmp/res142-review-repairs-private/history-audit/`, within the restricted evidence directory. This document deliberately identifies paths and timestamps without reproducing the disclosed relevance rows.

## Aggregate policy audit and repair

The recorded rights policy accepts the exact private run and records public redistribution as UNVERIFIED. The requester confirmation explicitly does not authorize public redistribution. No separate public-distribution authorization was found for aggregate annotation counts or for the aggregate report as a whole. Removing per-query rows was therefore insufficient to establish clearance; the earlier repair record must not be interpreted as public-release permission.

The current qualifier and checked-in report now withhold direct annotation-cardinality fields `candidate_pool_ceiling.positive_qrel_queries` and `baseline.qrel_count`. Before that reporting-only change, the complete public artifacts were retained in `.tmp/res142-review-repairs-private/pre-closure-public/`. The full private evaluation continues to retain all original study evidence. Earlier immutable documentation and historical snapshots remain identified above; no evidence was deleted.

Remaining public count classes were audited: `improved_queries`, `regressed_queries`, `tied_queries`, sign-test `non_tied_query_count`, and sign-flip `permutation_count` are statistical outcome/sample-support counts derived from qrels. Recall ceilings, aggregate metrics, confidence intervals and p-values also derive from judgments. They remain unchanged to preserve the reported scientific result, with explicit `public_release_clearance=unresolved`; none is declared distribution-cleared by this audit. Query/score/candidate counts and bootstrap sample/seed settings describe experiment execution or method, rather than judgment cardinality. Hashes identify restricted artifacts without publishing their contents. The earlier documentation's aggregate counts likewise remain uncleared pending the owner decision; a new commit cannot remove their historical copies.

Scientific validity and legal/public distribution authority are separate questions. The repository policy supplies no basis here for converting a successful numeric check into public rights clearance.

## Private output guard

`--private-out` requires an existing protected parent. Repository destinations must be untracked and actually Git-ignored, as checked against the repository containing that destination, including nested repositories. Tracked descendants or ignore-rule exceptions fail closed. Outside a repository, the protected-parent requirement still applies. Symlink and Windows junction ancestors are rejected. The guard runs before evidence reads and again immediately before private publication.

Accepted protection is owner-owned POSIX mode 0700, or an inheritance-protected Windows DACL allowing only the current user/OWNER RIGHTS, SYSTEM and Administrators, with inheritable current-user full control. Missing Git, failed boundary discovery or unverified protection fails closed. Windows ACL inspection is read-only. Provision the parent before invoking the qualifier; the command does not silently create a private root in publishable content. Git ignore rules protect against ordinary staging, not an owner's deliberate force-add; preserve the owner-controlled access and publication boundary.

## Retained evidence: verified facts and durability limits

Primary file: `E:\Data\Projects\DynamisRAG\.tmp\res142-review-repairs-private\evaluation.json`.

SHA-256: `bf1d320620fd1908122560577a5573bbf705fdb0fd0efeb11fd7255c553909bd`; size: 25,412 bytes. A second identical retained copy exists at `E:\Data\Projects\DynamisRAG\.tmp\res142-review-repairs-private-final\evaluation.json`. The primary and second copy were read back and compared; the public receipt binds the same private SHA. Reporting repair did not rewrite either file.

The original file DACL was already protected (OWNER RIGHTS, SYSTEM, Administrators), but its parent directory inherited broader access. This repair restricted both named evidence directory DACLs to current user, SYSTEM and Administrators with inheritable full control and disabled inheritance. The existing file's protected ACL remains in place. The private-output guard accepts the repaired primary parent. Both files are untracked and ignored by `.tmp/`. No private evidence was uploaded or moved to another service.

Both retained copies reside on the same local E: NTFS volume; the volume reported Healthy. This proves current local retention and access protection, not disaster recovery or archival durability. `.tmp/` is explicitly ephemeral in `.gitignore`. No independently protected backup on a separate failure domain, backup schedule, restore test, retention obligation or external custody has been verified. A same-volume second copy is not an independent backup. Long-term retention remains an owner decision; the audit does not label private-evidence durability closed.

## Owner decisions: two possible release paths

1. Rights clearance: obtain and record source/rightsholder and owner authorization covering the actual historical per-query annotations, direct aggregate cardinalities, statistical outcome counts and aggregate reports, with any required attribution/notices. Bind the decision to exact disclosed artifacts and permitted audiences. Private-run confirmation alone does not supply this authorization. Preserve scientific evidence and record the decision before declaring a public release cleared.
2. History remediation: if public annotation distribution remains uncleared, the owner can authorize a reviewed remediation plan covering the listed commits/paths, all affected refs/PRs, historical description edits and review diffs. First secure the complete evidence in an owner-approved durable private store. A Git rewrite/force-push would change commit identities and invalidate prior exact-head checks and reviews; it would require retesting and coordination. GitHub-managed PR refs, cached patches and edited-body history may require GitHub support. Existing clones/forks/downloads cannot be recalled or certified erased by a branch rewrite. Any visibility change is a separate owner decision and would not revoke prior copies.

No history rewrite, force-push, evidence deletion, repository visibility change, or new-service evidence upload was performed. Neither option above is authorized merely by this document. Current-source repairs can proceed while those decisions remain unresolved.

## Validation

Focused local regressions: 46 passed, 1 POSIX-only test skipped on Windows. The tests cover tracked/unignored destinations, forced-tracked content under an ignore rule (including literal bracket pathnames), ignore exceptions, missing parents, protected versus broadly readable Windows DACLs, Windows junctions, symlink aliases, and a private gate before evidence reads. POSIX ownership/mode checks run in Linux CI. Existing adversarial scientific-evidence and create-only publication regressions also pass. Full Ruff lint/format, notebook helper lint and configured strict Pyright pass with zero diagnostics. The optional direct-script typing limitation is recorded below. No scientific qualification/evaluation was rerun.

## Verdict

Scientific: PASS for the previously verified bounded pilot; all frozen scores, inputs, provenance, evaluation definitions, exact tests, nominal unadjusted bootstrap intervals and conclusions are unchanged. No new science was performed.

Engineering: private-output and reporting boundary repairs are subject to focused regressions, full Ruff, strict Pyright, and exact-head Linux/Windows/integration CI. Their exact final result is recorded in the PR and final response. The configured strict Pyright gate covers `src`, `tests` and `alembic`. An additional direct check of the qualification script (outside that configured scope) found 17 pre-existing typing diagnostics; a same-location comparison against the original HEAD confirmed identical messages and no new diagnostics. No unrelated type refactor was introduced. Private evidence is locally verified and access-restricted, but long-term backup/retention is unresolved.

Public release: NO-GO pending owner rights-clearance versus history-remediation decision, aggregate distribution clearance and durable evidence custody. Current sanitization is not clearance or historical erasure. PRs #16–18 remain draft and unmerged. [Issue #19](https://github.com/Litju/DynamisRAG/issues/19) remains open for operational shutdown and actual CU accounting; it was inspected read-only and not changed.
