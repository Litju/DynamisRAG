# Test fixtures

Third-party documents used by the integration suite, and how to obtain them.

Every file here is a byte-exact capture of what Europe PMC served for the PMCID
named in its filename, pinned by SHA-256. The suite serves them from a mocked
transport rather than fetching them, so no test depends on a public service being
reachable and CI stays deterministic.

## `PMC2731074.xml`

    Effects of a probiotic soy product and physical exercise on formation of
    pre-neoplastic lesions in rat colons in a short-term model of carcinogenic

    Silva MF, Sivieri K, Rossi EA.
    Journal of the International Society of Sports Nutrition, 2009; 6:17.
    doi:10.1186/1550-2783-6-17   pmid:19660118   pmcid:PMC2731074

    SHA-256  4a8ed3a3b7d3044697f1462eebe653b06086b267d1fae5300b5d4344d34f3ab4
    Source  https://www.ebi.ac.uk/europepmc/webservices/rest/PMC2731074/fullTextXML
    Licence CC BY 2.0 — https://creativecommons.org/licenses/by/2.0

Licensed for reuse under the Creative Commons Attribution 2.0 International
licence, which permits redistribution provided the author, title, source and
licence are attributed as above. The attribution is also carried in the
`permissions` block of the document itself, so it survives any copy of the file.

Used by `tests/integration/test_opensearch_vector_projection_live.py` to prove the
vectorized projection over a real full-text article: this is the document the
synthetic-vector mechanics proof is measured against, because it chunks into 19
passages with the canonical chunker configuration.

No embedding is ever generated from any document in this directory. The vectors
used against them are synthetic test values, labelled as such at every use.

## `datasets/`

Synthetic fixtures for the RES-141 dataset adapters. Nothing in that directory
is third-party content; see `datasets/README.md` for what each miniature mirrors
and which adapter it exercises.