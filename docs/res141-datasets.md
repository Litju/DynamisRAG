# RES-141 dataset qualification and frozen slices

RES-141 turns the official distributions of SciFact (BEIR), SciFact-Open,
QASPER and a fixed BEIR shortlist into **sealed, reproducible evaluation
slices**. Adapters qualify sources offline: they verify pinned digests, derive
canonical RES-140 dataset inputs, preserve native identifiers and evidence
semantics, and refuse to read a source whose rights decision is not accepted.
They never contact a network, open a database, load a model or touch the served
search stack.

## Windows PowerShell quickstart

Install the locked CPU-only dependencies from the repository root:

```powershell
uv sync --locked
```

Inspect the frozen registry — source identities, split names, archive digests
and the recorded rights decision — without downloading anything:

```powershell
uv run dynamisrag datasets list
```

Download one official archive and materialize a slice. Downloads stay outside
the tool: the adapter only accepts files whose bytes match the pins.

```powershell
# SciFact BEIR test split (2.7 MB official archive)
Invoke-WebRequest `
  -Uri 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip' `
  -OutFile .tmp/scifact.zip
uv run dynamisrag datasets materialize `
  --source beir.scifact `
  --split test `
  --archive .tmp/scifact.zip `
  --out .tmp/scifact-test
uv run dynamisrag datasets verify .tmp/scifact-test --registered-source
```

Already extracted a distribution? `--source-dir` verifies every member in place
and writes nothing:

```powershell
uv run dynamisrag datasets materialize `
  --source beir.scifact `
  --split train `
  --source-dir .tmp/extracted/scifact `
  --out .tmp/scifact-train
```

SciFact-Open ships one tarball; the `candidates` variant reads the 12,236
pooled abstracts, `full` reads the 500,000-abstract corpus:

```powershell
uv run dynamisrag datasets materialize `
  --source scifact-open `
  --split test `
  --corpus-variant candidates `
  --archive .tmp/scifact_open.tar.gz `
  --out .tmp/scifact-open-candidates
```

QASPER needs two tarballs (train+dev and test):

```powershell
uv run dynamisrag datasets materialize `
  --source qasper `
  --split test `
  --archive .tmp/qasper-train-dev-v0.3.tgz `
  --archive .tmp/qasper-test-and-evaluator-v0.3.tgz `
  --out .tmp/qasper-test
```

Score paragraph selections against the frozen QASPER task. The rankings file is
one JSON object mapping question IDs to ordered paragraph anchors:

```powershell
uv run dynamisrag datasets score-evidence `
  --slice .tmp/qasper-test `
  --rankings .tmp/rankings.json `
  --out .tmp/qasper-evidence.json
```

## What a slice contains

A document-retrieval slice (`dataset.json` plus a manifest, a rights notice and
any task sidecar) is written atomically and never overwrites an existing
directory. `dataset.json` is exactly the canonical RES-140 `IrDataset` payload,
so it feeds `ir score` directly; the checked-in integration test does this
without any translation layer.

| File | Meaning |
| --- | --- |
| `dataset.json` | Canonical RES-140 dataset: source identity, corpus digest, query universe and original judgments |
| `manifest.json` | Slice revision, full source and rights description, corpus identity, counts, per-file sizes and digests |
| `rights.txt` | Human-readable license, attribution and decision notice, generated from the registry |
| `evidence-provenance.json` | SciFact-Open only: per-link provenance, labels, sentence indexes, model ranks and pool membership |

A QASPER slice is deliberately **not** a document-retrieval dataset. It contains
`task.json` (`qasper-evidence-selection-v1`) and no `dataset.json`: a paragraph
annotation is not a document qrel, and inventing one would be the exact
conflation the mission forbids.

## Source qualification

Every source in `dynamisrag.datasets.sources` pins the official distribution
URL, the archive SHA-256 and the SHA-256 and size of every member this
repository reads. The SciFact-Open URL ends in `latest` because that is how the
authors publish it; the *digest* is the authority, not the URL. No Hub
conversion, no moving branch, no filename trust.

A slice can only be materialized through an **accepted** rights decision. The
registry records, per source, the dataset license, the license scope, where the
license was read, what the underlying content is, redistribution status,
attribution and the basis for the decision. One fail-closed rule is structural:
an accepted source whose license is *unstated* must prohibit redistribution.
SciFact-Open is exactly that case — publicly released by its authors for
research, with no LICENSE file — so it is evaluation-only and this repository
never republishes its bytes.

## Task semantics

**SciFact (BEIR).** BEIR's `train.tsv` is the original SciFact `claims_train`
(809 claims, 919 judgments); BEIR's `test.tsv` is the original `claims_dev`
(300 claims, 339 judgments). These are binary link judgments, not the SciFact
claim-veracity labels, and neither split distributes negative or zero judgments.
Queries are exactly the split's judged claims; the excluded query count is
recorded.

**SciFact-Open.** `scifact-open-retrieval-projection-v1` makes a claim a query
and an abstract relevant when it contains annotated evidence; both SUPPORT and
CONTRADICT evidence mark relevance, because the retrieval task is finding
evidence, not deciding the claim. Evidence provenance (`citation` versus
`pooling`), sentence highlights and model ranks are preserved verbatim in the
sidecar. Judgments are `pooled-partial`: documents outside the released pool are
unjudged, not non-relevant, so any metric here is pooled recall. The corpus
variant (`candidates` or `full`) is part of the dataset identity.

**QASPER.** `qasper-evidence-selection-v1` is a within-document task: each
question belongs to one known paper, and the selection unit is one paragraph
with a stable anchor (`paper/section/paragraph`). Every annotation is preserved
(answer forms, annotation and worker IDs, both evidence fields). Anchors are
resolved by exact text match and every resolution is recorded as `unique`,
`ambiguous` or `unmatched`; unresolvable evidence stays in the task. The
reference metric, `qasper-paragraph-f1-v2`, follows the official evaluator's
evidence-F1 shape (per-question maximum over annotation references;
empty-versus-empty is 1.0; a missing prediction scores 0.0 and is counted) over
anchors instead of strings, and each annotation is classified as `complete`
(every reference resolved, or genuinely no evidence), `partial` (resolved and
unresolved references mixed) or `unavailable` (nonempty evidence with no
resolved anchor). A question is scored only when **every** annotation is
`complete`; otherwise it is excluded from the metric denominator with a recorded
reason, and the evaluation reports per-question status, resolution coverage and
every denominator count. Positive but unresolved evidence — unmatched,
ambiguous or figure/table (`FLOAT SELECTED`) — is therefore never scored as
absent gold: an unresolved annotator can no longer yield a false perfect F1.
`qasper-paragraph-f1-v1` treated an all-unresolved annotation as an empty
reference and is superseded by this revision.

**BEIR shortlist.** A fixed set of comparators — NFCorpus and SciDocs
(scientific), ArguAna and FiQA-2018 (out-of-domain) — each with per-split
cardinality pins. NFCorpus and FiQA query splits are disjoint; SciDocs ships a
single test split with explicit zero judgments. ArguAna's queries are themselves
corpus arguments, so its dataset identity pins the standard BEIR
`ignore-identical-query-document-ids` protocol
(`beir-ignore-identical-query-document-ids-v1`) in the dataset revision and in
the manifest diagnostics, together with the counted, hashed set of query ids
that occur in the corpus. The reader preserves the self-documents in the corpus
(removing them would change the corpus identity); instead,
`validate_run_protocol` refuses a sealed run whose evaluated prefix still
contains a query's own document as non-comparable to standard BEIR, and
`exclude_identical_document_hits` applies the reference rule to an untruncated
candidate list before evaluation depth is chosen. ArguAna's five qrels whose
documents are absent from the distributed corpus are declared, hashed and
excluded — the slice explicitly records that its qrel set is **not** the
original source qrel set (`qrels_are_source_complete: false`).

## Verification and refusal

`datasets verify` runs in one of two explicit modes, and reports which one it
ran:

- **self-consistency** (default): recomputes everything the sealed bytes can
  prove — the closed directory inventory, every file digest, the typed dataset
  or QASPER task identity, the manifest's query, qrel and QASPER counts, the
  source/split revision binding, the generated `rights.txt` and every
  SciFact-Open provenance-sidecar statistic. A re-signed manifest with a false
  count, a changed rights notice, a changed source description or a mismatched
  split/source refuses.
- **qualified**: self-consistency plus a caller-supplied trust anchor. Pass
  `--registered-source` to require the manifest source to equal the registered
  frozen source exactly, and/or `--expect-manifest-sha256 <hex>` to require the
  canonical manifest digest to match an out-of-band value.

Corpus identity counts (`document_count`, `documents_without_text`) and the
original archive/member digests cannot be recomputed without the source bytes.
The receipt lists them under `attested_claims` — never as verified — unless a
trust anchor authenticates the manifest they live in. There is no
trust-on-first-use: a manifest that matches nothing trusted stays
self-consistency-only, and the CLI output says so. A synthetic fixture is not
the registered distribution, so `--registered-source` refuses it by design.

Materialization refuses an existing output directory, a member that drifted
after pinning, an archive that matches no pin, a source whose rights decision is
rejected, and any request that cannot name exactly one input mode. Tests pin all
of this on synthetic fixtures under `tests/fixtures/datasets/`, and the real
registry digests are asserted in `tests/unit/test_datasets_sources.py`.
