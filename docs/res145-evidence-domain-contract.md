# RES-145 architecture decision

Baseline: 36cacfee1864080b4f1382ac74ede8e9431f559e; migration head 0004_passage_source_spans.

SourceArtifact identifies acquired bytes; Document fixes creation-time logical identity
and later aliases do not replace it. DocumentVersion binds artifact, parser,
normalizer and canonical content fingerprint. Passage identity is version/chunker/
ordinal; Paragraph identity is version/source anchor. PassageSourceSpan identity is
passage/source order. UUIDs are local foreign keys, never semantic identity inputs.
Existing composite foreign keys bind spans and paragraphs to the same version;
offset triggers bound half-open Python character offsets into Paragraph.text.
Passage reconstruction joins paragraphs with two newlines and preserves intervening
whitespace for adjacent spans of the same paragraph. Canonical insert functions
flush inside the caller transaction; they do not commit.

PassageProvenance is the retrieval projection, ScientificResponse records lanes and
RRF trace, ScientificAnswer records extractive claims and structural citation checks.
Neither answer support labels nor unsigned RES-143/144 artifacts confer expert authority.

Decision: add frozen evidence-v1 contracts. EvidenceSet stores exact ordered passage
membership (duplicates rejected), question text/task identity, corpus identity,
selection constraints, optional retrieval revision, and declared sufficiency.
Each member binds the version, complete ordered span inventory and a reproducible
source digest. Claim models a proposition/category with optional population,
condition, comparator, outcome, relationship, quantitative qualifiers, time,
applicability and limitations. Provenance identifies the authoring process and
origin reference; these are declarations, not authenticated identities.
EvidenceLink binds a claim, set, member and one complete canonical source span.
No new excerpt coordinates are introduced. SUPPORTS is an interpretation, not entailment.

Only PROPOSED, AWAITING_REVIEW and REJECTED can be persisted. REVIEWED is a known
but unsupported state, rejected in Python and SQL. Source validity and adjudication
remain separate. Changes to meaning, relationship, rationale or status produce new
semantic keys; no record can be updated or deleted.

Identity uses existing SHA-256 digest primitives over a versioned namespace and
canonical UTF-8 JSON (sorted keys, no whitespace, no Unicode normalization).
No timestamps, random UUIDs or caller-supplied canonical keys participate.

Persistence adds four tables: evidence_set, claim, evidence_membership, evidence_link.
Typed payloads are stored as JSONB, with first-class keys and relational membership;
JSON is a serialization of the closed domain contracts, not extensible metadata.
SQL triggers recompute keys, verify relational/payload agreement and exact source
lineage, and reject approval forgery. Deferred completeness validation prevents
committing a set with missing members. Application publication revalidates contracts,
uses a savepoint, reconstructs and checks the graph before returning, and never commits.
The caller owns the outer transaction. SQL constraints cover forged source bindings,
identity, relationship, membership, review states and immutability; the closed Python
contracts additionally enforce the complete scientific field schema. Database owners
who disable triggers are outside this trust boundary. No source-byte authenticity or
scientific adjudication is claimed beyond retained canonical records.
