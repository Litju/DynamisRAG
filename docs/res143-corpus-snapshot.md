# RES-143 bounded corpus demonstration

Run locally with `uv run python -m scripts.corpus_snapshot acquire --run-root .dynamisrag/res-143/demo-v2`; verify with the same command using `verify`. Source XML remains in the existing local object store and is not committed. The run directory retains exact discovery pages, decisions, permissions bytes and the immutable manifest.

Snapshot SHA-256: `1b4548b23eb1284ab4a659e5071b77563b0220791918939d9b723cc28e4255d7`. Discovery records: 58; malformed: 2; unique article groups: 52; duplicates: 4; conflicts: 0; rejected: 42; eligible and acquired: 10; deferred: 0. Crossref returned HTTP 429 for one topic after retries, recorded in discovery. Counts refer to the bounded run, not the global literature.

The repository metadata preflight and original JATS explicit CC BY grant were checked for each member. Third-party assets were not acquired. Scientific scope is a conservative automated screen and requires independent review. The preliminary development run persisted 13 articles before human-scope filtering, including an animal paper and a commentary; these remain in the append-only local canonical store but are excluded from this snapshot and all vector publication.

| PMCID | Original license | Source SHA-256 |
| --- | --- | --- |
| [PMC10117840](https://europepmc.org/articles/PMC10117840) | https://creativecommons.org/licenses/by/4.0/ | `a37bd4d8d25002ffa25c451bee01d34029d965a8eea5fdaf536737ef1879e7f7` |
| [PMC12640931](https://europepmc.org/articles/PMC12640931) | https://creativecommons.org/licenses/by/4.0/ | `848fe68b22e84d96ea77d031f9bdf77f3526b14cfe49f65baf58ed64b027823c` |
| [PMC12791081](https://europepmc.org/articles/PMC12791081) | https://creativecommons.org/licenses/by/4.0/ | `c843b995551447f6e46e7df8956f15e7b731a54bcbbff33fbfd991319ba134d1` |
| [PMC12819620](https://europepmc.org/articles/PMC12819620) | https://creativecommons.org/licenses/by/4.0/ | `98a1a919e48b8da0622dfd0455f44bfe5b7fcb6daeb853ed21d11cbeb2fab0c3` |
| [PMC13042950](https://europepmc.org/articles/PMC13042950) | https://creativecommons.org/licenses/by/4.0/ | `c1024fe79beff0751061d5bc914ec757c3aad17dc5aff153ec017edb58458ffc` |
| [PMC13364619](https://europepmc.org/articles/PMC13364619) | https://creativecommons.org/licenses/by/4.0/ | `1696b15858e8e097b6571e7936ab18cbf8958abdca1ab6b1b89c6737ca6c06f7` |
| [PMC13375495](https://europepmc.org/articles/PMC13375495) | https://creativecommons.org/licenses/by/4.0/ | `1dc893e38ec11e1b5912bd1a3477d280abef244d677446b5b84d44c035add695` |
| [PMC13416682](https://europepmc.org/articles/PMC13416682) | https://creativecommons.org/licenses/by/4.0/ | `33a90452b9e2428cbe5f4b3db2857899393eefe4930557401f3e14c00c730787` |
| [PMC13707224](https://europepmc.org/articles/PMC13707224) | https://creativecommons.org/licenses/by/4.0/ | `1a814d50a7e31bcf5400385c93055c15d993a3ca2627b684b528fd3acc4ee310` |
| [PMC9691012](https://europepmc.org/articles/PMC9691012) | https://creativecommons.org/licenses/by/4.0/ | `3e486477d100fd623a02014af676950a4bd0f69a947ef724c909175508f5ea80` |


## Independent verifier repair scope (PR #23 follow-up)

Offline verification now reconstructs all normalized and rejected records from the
retained source pages as multisets, including repeated records. It independently
checks member title, contributors, identifiers, canonical-key plausibility,
source URI and XML type against original JATS and discovery sources. The
run-aware verifier also checks the staged acquisition and decision receipts,
the reported snapshot ID, and the run/qualification summary.

The documented snapshot identity, 10 original source hashes, JATS bytes, and
canonical database records are not rewritten by these deterministic code and
test repairs. An offline replay verifies the evidence available locally; it
cannot independently authenticate an original HTTP response or prove an older
document's historical canonical-database alias without database evidence.
A human review of each member's scientific scope remains necessary.

**Discovery coverage: PARTIAL.** One Crossref topic exhausted its bounded
HTTP-429 retries. The operator's verification output now derives an explicit
partial/complete label from the retained failure receipts; a partial snapshot
must not be reported as an exhaustive search result. No source acquisition,
embedding, indexing, or model run is required by this verifier repair.

## P1 recovery and failure receipts

Resume validates retained discovery and existing stage and decision receipts before
acquisition, object publication, or database transactions. Missing discovery with
retained receipts fails closed. Newly recorded `discovery-2` failure receipts bind
the planned provider, topic, page and request URL; this is structural consistency,
not independent proof that an HTTP failure occurred. The original `discovery-1`
failure receipt lacks those structured fields. It remains available for offline
verification, reported as `legacy-unbound`, but cannot be resumed for acquisition.
No historical failure details are inferred or backfilled. Fresh discovery creates a
separate snapshot; it cannot repair the historical search coverage.
