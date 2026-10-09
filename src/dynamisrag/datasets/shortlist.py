"""The frozen heterogeneous BEIR shortlist (RES-141).

The shortlist exists so "we evaluated on BEIR" can never mean "whichever
datasets happened to be cached". It names a fixed set of comparators, pins each
official archive and every read member, and states what each one is for:

===========================  ==========  ==========================================
source                       category    role
===========================  ==========  ==========================================
SciFact (main adapter)       scientific  the primary task; see ``scifact.py``
NFCorpus                     scientific  medical IR with graded 1-2 judgments
SciDocs                      scientific  citation-prediction IR with explicit zeros
ArguAna                      out-domain  counter-argument retrieval
FiQA-2018                    out-domain  financial question retrieval
===========================  ==========  ==========================================

Each spec's split pins are exact cardinalities counted from the official
archives at registry time. "No leakage across splits" is a property of the
distribution that the registry states and the tests verify: NFCorpus and FiQA
query sets are disjoint across train/dev/test, and SciDocs and ArguAna ship a
single test split. Corpus identity is computed per split from the full archive
corpus, so two datasets can never be confused for one another through a
normalized alias: every dataset keeps its original IDs and its own digest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from dynamisrag.datasets.beir import (
    BeirSliceSpec,
    BeirSplitExpectation,
    DanglingQrelPolicy,
)

__all__ = [
    "R141_BEIR_SHORTLIST",
    "ShortlistEntry",
    "shortlist_spec",
]

_NFCORPUS: Final[BeirSliceSpec] = BeirSliceSpec(
    source_id="beir.nfcorpus",
    role="scientific",
    domain="medical information retrieval",
    projection_note=(
        "original graded judgments 1-2 preserved; queries are exactly the split's judged "
        "NutritionFacts.org queries; splits are query-disjoint"
    ),
    splits=(
        (
            "train",
            BeirSplitExpectation(
                documents=3633,
                documents_without_text=0,
                queries_in_archive=3237,
                queries=2590,
                qrels=110575,
                min_relevance=1,
                max_relevance=1,
            ),
        ),
        (
            "dev",
            BeirSplitExpectation(
                documents=3633,
                documents_without_text=0,
                queries_in_archive=3237,
                queries=324,
                qrels=11385,
                min_relevance=1,
                max_relevance=2,
            ),
        ),
        (
            "test",
            BeirSplitExpectation(
                documents=3633,
                documents_without_text=0,
                queries_in_archive=3237,
                queries=323,
                qrels=12334,
                min_relevance=1,
                max_relevance=2,
            ),
        ),
    ),
)

_SCIDOCS: Final[BeirSliceSpec] = BeirSliceSpec(
    source_id="beir.scidocs",
    role="scientific",
    domain="citation-prediction retrieval",
    projection_note=(
        "original binary judgments including 25,000 explicit zero-relevance pairs, so "
        "known-nonrelevant documents are not silently unjudged"
    ),
    splits=(
        (
            "test",
            BeirSplitExpectation(
                documents=25657,
                documents_without_text=0,
                queries_in_archive=1000,
                queries=1000,
                qrels=29928,
                min_relevance=0,
                max_relevance=1,
            ),
        ),
    ),
)

_ARGUANA: Final[BeirSliceSpec] = BeirSliceSpec(
    source_id="beir.arguana",
    role="out-of-domain",
    domain="counter-argument retrieval",
    projection_note=(
        "one relevant counter-argument per query; five qrels whose documents are absent from "
        "the distributed corpus are declared and excluded rather than invented"
    ),
    splits=(
        (
            "test",
            BeirSplitExpectation(
                documents=8674,
                documents_without_text=0,
                queries_in_archive=1406,
                queries=1406,
                qrels=1401,
                min_relevance=1,
                max_relevance=1,
                dangling_qrels=5,
                dangling_policy=DanglingQrelPolicy.EXCLUDE_AND_DECLARE,
            ),
        ),
    ),
)

_FIQA: Final[BeirSliceSpec] = BeirSliceSpec(
    source_id="beir.fiqa",
    role="out-of-domain",
    domain="financial question retrieval",
    projection_note=(
        "binary link relevance over Stack Exchange posts; queries are exactly the split's "
        "judged questions; splits are query-disjoint"
    ),
    splits=(
        (
            "train",
            BeirSplitExpectation(
                documents=57638,
                documents_without_text=38,
                queries_in_archive=6648,
                queries=5500,
                qrels=14166,
                min_relevance=1,
                max_relevance=1,
            ),
        ),
        (
            "dev",
            BeirSplitExpectation(
                documents=57638,
                documents_without_text=38,
                queries_in_archive=6648,
                queries=500,
                qrels=1238,
                min_relevance=1,
                max_relevance=1,
            ),
        ),
        (
            "test",
            BeirSplitExpectation(
                documents=57638,
                documents_without_text=38,
                queries_in_archive=6648,
                queries=648,
                qrels=1706,
                min_relevance=1,
                max_relevance=1,
            ),
        ),
    ),
)


@dataclass(frozen=True)
class ShortlistEntry:
    """One shortlisted comparator with its frozen spec and its role."""

    spec: BeirSliceSpec
    category: str
    rationale: str


R141_BEIR_SHORTLIST: Final[tuple[ShortlistEntry, ...]] = (
    ShortlistEntry(
        spec=_NFCORPUS,
        category="scientific",
        rationale="medical full-text IR with graded judgments; small and fully judged",
    ),
    ShortlistEntry(
        spec=_SCIDOCS,
        category="scientific",
        rationale="citation-prediction IR with explicit known-nonrelevant documents",
    ),
    ShortlistEntry(
        spec=_ARGUANA,
        category="out-of-domain",
        rationale="argument retrieval; no lexical overlap with the scientific corpora",
    ),
    ShortlistEntry(
        spec=_FIQA,
        category="out-of-domain",
        rationale="financial community QA; a zero-shot out-of-domain comparator",
    ),
)


def shortlist_spec(source_id: str) -> BeirSliceSpec:
    """The shortlisted spec for ``source_id``."""
    from dynamisrag.datasets.errors import DatasetContractError

    for entry in R141_BEIR_SHORTLIST:
        if entry.spec.source_id == source_id:
            return entry.spec
    raise DatasetContractError(
        f"{source_id!r} is not on the frozen shortlist.",
        operation="shortlist_spec",
        item_id=source_id,
        expected=str([entry.spec.source_id for entry in R141_BEIR_SHORTLIST]),
    )
