"""SciFact via the frozen BEIR conversion (RES-141).

SciFact is released twice: as the original AI2 annotations
(``claims_train``/``claims_dev``/``claims_test``, with claim-veracity labels and
rationales) and as the BEIR benchmark's frozen retrieval conversion, which the
mission names as the source. This adapter reads the BEIR conversion and records
the semantics that matter for not mistaking one task for the other:

* BEIR ``qrels/train.tsv`` is the original SciFact **claims_train** (809 claims,
  919 link judgments); BEIR ``qrels/test.tsv`` is the original **claims_dev**
  (300 claims, 339 link judgments). SciFact's own test split ships unlabeled and
  contributes nothing here.
* Every BEIR qrel is a *binary link judgment* (score 1) over an S2ORC abstract.
  It is not a claim-veracity label: no SUPPORT/CONTRADICT annotation is present
  in, or recoverable from, the BEIR conversion.
* Neither split distributes a negative or zero-relevance qrel. That is a
  property of this distribution, not a loader assumption: the reader preserves
  any integer judgment, and the pinned expectations would notice a changed
  score range.
"""

from __future__ import annotations

from typing import Final

from dynamisrag.datasets.beir import BeirSliceSpec, BeirSplitExpectation

__all__ = ["SCIFACT_BEIR_SPEC"]

SCIFACT_BEIR_SPEC: Final[BeirSliceSpec] = BeirSliceSpec(
    source_id="beir.scifact",
    role="scientific",
    domain="biomedical claim retrieval",
    projection_note=(
        "binary BEIR link relevance; queries are exactly the split's judged claims; BEIR train "
        "is the original SciFact claims_train and BEIR test is the original claims_dev, whose "
        "claim-veracity labels are not part of this retrieval projection"
    ),
    splits=(
        (
            "train",
            BeirSplitExpectation(
                documents=5183,
                documents_without_text=0,
                queries_in_archive=1109,
                queries=809,
                qrels=919,
                min_relevance=1,
                max_relevance=1,
            ),
        ),
        (
            "test",
            BeirSplitExpectation(
                documents=5183,
                documents_without_text=0,
                queries_in_archive=1109,
                queries=300,
                qrels=339,
                min_relevance=1,
                max_relevance=1,
            ),
        ),
    ),
)
