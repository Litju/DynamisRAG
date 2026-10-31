# RES-145 qualification

Implementation commit: `04ef019bf5b08e6eb9e8b0e2bcf6c50d68cae53a`.
The subsequent qualification commit adds this receipt and strict typing for the
standalone demonstration script. The exact publication SHA and hosted run are
recorded on draft PR #25 and RES-145; this document does not claim its own commit hash.

Starting state was verified before edits: existing remote branch
`feat/res-145-evidence-domain` and `origin/main` both pointed to
`36cacfee1864080b4f1382ac74ede8e9431f559e`. No intervening main changes were present.
Migration head was `0004_passage_source_spans`. Pre-existing untracked
`notebooks/res138_colab.operator.ipynb` and `reports/` were preserved.

Implemented contracts and boundaries are described in
[the architecture decision](res145-evidence-domain-contract.md).
The reversible new migration is `0005_evidence_graph`, based on the actual 0004
head. No historical migration was edited. The populated reference database was
never migrated; all migration tests and graph commits used ephemeral databases.

Local qualification:

- Ruff lint and format: pass for tracked source, tests, scripts and migrations.
  Root checks use `--exclude reports` solely to preserve the pre-existing untracked
  audit archive, whose Python file fails lint/format. No repository rule was relaxed.
- Strict Pyright: zero errors, warnings or informational diagnostics. The standalone
  `scripts/res145_evidence_demo.py` was additionally checked explicitly in strict mode.
- `uv lock --check`: pass; dependencies and lockfile unchanged.
- Full unit suite: **2,458 passed**, with 243 integration cases deselected at that run.
  Command: `uv run pytest --basetemp .tmp/res145-unit-02 -q`.
  A fresh workspace temporary directory avoids an inaccessible Windows system temp
  directory. The stale migration inventory assertion was repaired by explicitly
  admitting only 0005; its checks prohibiting embedding/vector migration state remain.
- Domain evidence tests: **19 passed**, included in the full unit run; canonical
  identity, ingestion, generation, citation and frozen benchmark unit regressions pass.
- Evidence PostgreSQL suite: **44 passed**, including direct database bypass attacks,
  Unicode serialization parity, existing forged canonical rows, concurrent retries,
  contradictory interpretations, invalid offsets, immutable UPDATE/DELETE rejection,
  unsigned approval refusal, deferred completeness, interruption recovery and migration
  reversal on populated source tables.
- Complete isolated PostgreSQL/OpenSearch/Qdrant integration suite: **251 passed**,
  with 2,458 unit cases deselected, on PostgreSQL 18.6. All legacy migration,
  ingestion and citation regressions passed. Hosted exact-head status is recorded
  on the PR/Linear receipt after completion.
- Real-source demonstration: pass, including commit, fresh-session reconstruction,
  repeat insertion and migration `0005 -> 0004 -> 0005` with exact canonical-row
  equality. See [identifier-only receipt](evidence/res145-demo-receipt.json).

The complete integration suite can be reproduced safely from PowerShell with a new
base-temp path (the helper creates, migrates and removes an ephemeral database):

```powershell
@'
import subprocess
import sys
from scripts.res145_evidence_demo import scratch_database
with scratch_database():
    subprocess.run([sys.executable, '-m', 'pytest', '-m', 'integration', '-q',
                    '--basetemp', '.tmp/res145-integration-fresh'], check=True)
'@ | uv run python -
```

The bounded real-source demonstration is reproducible with
`uv run python -m scripts.res145_evidence_demo` when the authorized original baseline
and XML are available. It selects passage
`37ad148429ed0342fb2bc0e02fadc5194a4be1aa8f16bab1df8d433336a50144`
from the starting knee-angle study. It verifies original XML SHA-256, independently
reconstructs paragraph spans in Python and SQL, retains original UUIDs and semantic
keys in the scratch copy, and records one provisional methodological context link.
Only identifiers, counts and hashes are written to the public receipt. The private
source text, original XML, raw service responses and database contents are not exported.

Pre/post protection: all twelve canonical tables (including the 75 passages), every
file under the original real-science artifact directory, all 75 original Qdrant
vectors and payloads, and aliases matched. Existing documents, embeddings, benchmark
runs, question bank and RES-143 adjudication artifacts were not edited. Legacy
integration tests use their own synthetic search projections. No inference,
embedding generation or reference collection writes occur in the demonstration.

Authority remains bounded: `SUPPORTS` and `CONTRADICTS` are declared interpretations.
A source-valid link never proves entailment, causality, applicability or expert
approval. `REVIEWED` is refused in both Python and PostgreSQL. All demonstration
interpretations are `PROPOSED`; no reviewer, signature or key was fabricated.
RES-143/144 unsigned decisions retain their original restrictions. Database owners
able to disable triggers are outside the enforcement boundary; closed scientific
payload schemas are enforced by application contracts, while SQL independently
checks identity, provenance, relationship, minimum claim meaning, review states,
relational membership and append-only behavior.

No merge, release, issue completion, RES-146 or RES-147 work is included.
