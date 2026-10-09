# RES-141 dataset adapter fixtures

Every file under this directory is **synthetic content written for this
repository**. No third-party dataset text is checked in here, and no archive
pinned by the RES-141 source registry is stored here.

* ``beir-*-mini/`` mirror the layout of an official BEIR archive member
  (``<dataset>/corpus.jsonl``, ``queries.jsonl``, ``qrels/<split>.tsv``) with
  three to four invented documents. They exercise the SciFact adapter and the
  heterogeneous shortlist: graded judgments (NFCorpus), explicit zero judgments
  (SciDocs), a dangling qrel (ArguAna) and a blank document (FiQA).
* ``scifact-open-mini/`` mirrors the official SciFact-Open tarball layout
  (``data/``, ``prediction/``) with invented claims, abstracts and retrievals,
  including one evidence link outside the released pool.
* ``qasper-mini/`` mirrors the official QASPER split files, with an ambiguous
  repeated paragraph, an unmatched evidence string, a float caption, a null
  section name, two annotators on one question and an unanswerable question.

Tests compute each member's SHA-256 at run time and build a synthetic source
whose pins are those computed digests, so the fixture tests exercise the same
verification code paths as an official download.
