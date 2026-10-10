"""The frozen RES-141 source registry: distributions, members, rights.

Every entry here names a **public distribution of a dataset exactly as its
authors or the BEIR benchmark released it**, pins the SHA-256 of the archive and
of every member this repository reads, and carries an explicit rights decision.
The registry is deliberately not a downloader: adapters accept a verified local
archive or an already-extracted source directory, per the mission's offline
constraint.

The distinction that makes the digests meaningful:

* an **archive pin** is what makes a downloaded file the distribution;
* a **member pin** is what makes the extracted ``corpus.jsonl`` the one inside
  that archive, checked again after extraction because a copy is exactly where
  bytes drift;
* the **rights decision** is what makes reading it permitted at all.

Nothing here is derived from a Hub `load_dataset` conversion, a ``latest``
redirect or a moving branch. The SciFact-Open S3 object is fetched from a URL
whose final path segment is ``latest``; that is the authors' publication, and
its identity is the *digest recorded below*, not the URL. The QASPER digests
begin as the official ``dataset_infos.json`` checksums and were re-verified
against the official S3 tarballs when this registry was written.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from dynamisrag.datasets.errors import DatasetContractError
from dynamisrag.datasets.rights import (
    LicenseScope,
    Redistribution,
    RightsDecision,
    RightsOutcome,
)

__all__ = [
    "FAMILY_BEIR",
    "FAMILY_QASPER",
    "FAMILY_SCIFACT_OPEN",
    "RES141_SOURCES",
    "FrozenDatasetSource",
    "SourceArtifact",
    "SourceFormat",
    "SourceMember",
    "source_by_id",
]

_SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}\Z")

FAMILY_BEIR: Final[str] = "beir"
FAMILY_SCIFACT_OPEN: Final[str] = "scifact-open"
FAMILY_QASPER: Final[str] = "qasper"


class SourceFormat(StrEnum):
    """The two archive formats the official distributions use."""

    ZIP = "zip"
    TAR_GZ = "tar.gz"


def _require_token(value: str, *, field: str) -> None:
    if not value or value != value.strip() or any(character.isspace() for character in value):
        raise DatasetContractError(
            f"{field} must be a non-empty whitespace-free token.",
            operation="validate_source_registry",
            item_id=field,
        )


def _require_sha256(value: str, *, field: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise DatasetContractError(
            f"{field} must be a lowercase 64-character SHA-256.",
            operation="validate_source_registry",
            item_id=field,
        )


@dataclass(frozen=True)
class SourceMember:
    """One file inside a frozen archive, pinned by exact size and digest."""

    name: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.name or self.name.startswith("/") or ".." in self.name.split("/"):
            raise DatasetContractError(
                "a source member name must be a safe relative archive path.",
                operation="validate_source_registry",
                item_id=self.name,
            )
        if self.size_bytes < 0:
            raise DatasetContractError(
                "a source member size cannot be negative.",
                operation="validate_source_registry",
                item_id=self.name,
                observed=str(self.size_bytes),
            )
        _require_sha256(self.sha256, field=f"member sha256 for {self.name}")

    def payload(self) -> dict[str, object]:
        """The hashed description of this member pin."""
        return {"name": self.name, "size_bytes": self.size_bytes, "sha256": self.sha256}


@dataclass(frozen=True)
class SourceArtifact:
    """One frozen archive: URL, format, archive pin and its read-member pins."""

    url: str
    archive_name: str
    format: SourceFormat
    size_bytes: int
    sha256: str
    members: tuple[SourceMember, ...]

    def __post_init__(self) -> None:
        if not self.url.startswith("https://"):
            raise DatasetContractError(
                f"the frozen source URL {self.url!r} is not https. The digest is what verifies "
                "the bytes, but a plaintext transport adds nothing and can mutate them.",
                operation="validate_source_registry",
            )
        if not self.archive_name or "/" in self.archive_name or "\\" in self.archive_name:
            raise DatasetContractError(
                "an archive name is a bare file name, never a path.",
                operation="validate_source_registry",
                item_id=self.archive_name,
            )
        if self.size_bytes <= 0:
            raise DatasetContractError(
                "an archive pin needs a positive size.",
                operation="validate_source_registry",
                item_id=self.archive_name,
            )
        _require_sha256(self.sha256, field=f"archive sha256 for {self.archive_name}")
        if not self.members:
            raise DatasetContractError(
                "an artifact with no member pins would accept anything the digest-matching "
                "archive happens to contain.",
                operation="validate_source_registry",
                item_id=self.archive_name,
            )
        names = [member.name for member in self.members]
        if names != sorted(set(names)):
            raise DatasetContractError(
                "source members must be unique and ascending so a manifest is canonical.",
                operation="validate_source_registry",
                item_id=self.archive_name,
            )

    def payload(self) -> dict[str, object]:
        """The hashed description of this archive and its read members."""
        return {
            "url": self.url,
            "archive_name": self.archive_name,
            "format": self.format.value,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "members": [member.payload() for member in self.members],
        }


@dataclass(frozen=True)
class FrozenDatasetSource:
    """One dataset distribution: identity, artifacts, rights and content notes."""

    source_id: str
    family: str
    revision: str
    documentation: str
    content_note: str
    artifacts: tuple[SourceArtifact, ...]
    rights: RightsDecision

    def __post_init__(self) -> None:
        _require_token(self.source_id, field="source_id")
        if self.family not in {FAMILY_BEIR, FAMILY_SCIFACT_OPEN, FAMILY_QASPER}:
            raise DatasetContractError(
                f"unknown source family {self.family!r}.",
                operation="validate_source_registry",
                source_id=self.source_id,
            )
        _require_token(self.revision, field="source_revision")
        if not self.documentation.startswith("https://"):
            raise DatasetContractError(
                "a source's documentation pointer must be an https URL.",
                operation="validate_source_registry",
                source_id=self.source_id,
            )
        if not self.content_note.strip():
            raise DatasetContractError(
                "a source must state what its corpus is made of.",
                operation="validate_source_registry",
                source_id=self.source_id,
            )
        if not self.artifacts:
            raise DatasetContractError(
                "a source must declare at least one artifact.",
                operation="validate_source_registry",
                source_id=self.source_id,
            )

    def artifact_for_digest(self, digest: str) -> SourceArtifact | None:
        """The declared artifact whose archive digest is ``digest``, if any."""
        for artifact in self.artifacts:
            if artifact.sha256 == digest:
                return artifact
        return None

    def member(self, name: str) -> SourceMember:
        """The pinned member called ``name`` across every artifact of this source."""
        for artifact in self.artifacts:
            for member in artifact.members:
                if member.name == name:
                    return member
        raise DatasetContractError(
            "the requested member is not declared by this source.",
            operation="source_member",
            source_id=self.source_id,
            item_id=name,
        )

    def payload(self) -> dict[str, object]:
        """The hashed description of this distribution."""
        return {
            "source_id": self.source_id,
            "family": self.family,
            "revision": self.revision,
            "documentation": self.documentation,
            "content_note": self.content_note,
            "artifacts": [artifact.payload() for artifact in self.artifacts],
            "rights": self.rights.payload(),
        }

    @property
    def sha256(self) -> str:
        """SHA-256 of the canonical source description."""
        from dynamisrag.datasets.primitives import digest

        return digest(self.payload())


def _scifact_rights() -> RightsDecision:
    return RightsDecision(
        dataset_license="CC-BY-4.0 (claims and evidence annotations); ODC-By-1.0 (S2ORC abstracts)",
        license_scope=LicenseScope.OPEN,
        license_source="https://github.com/allenai/scifact/blob/master/LICENSE.md",
        underlying_content=(
            "the BEIR corpus abstracts are S2ORC content; original articles retain their "
            "publishers' rights"
        ),
        redistribution=Redistribution.UNVERIFIED,
        attribution=(
            "Wadden et al. 2020 (SciFact); Thakur et al. 2021 (BEIR); Semantic Scholar (S2ORC)"
        ),
        outcome=RightsOutcome.ACCEPTED,
        basis=(
            "the SciFact repository license declares the claim annotations CC BY 4.0 and the "
            "abstracts ODC-By 1.0; BEIR only repackages the frozen files; evaluation use with "
            "attribution"
        ),
    )


def _scifact_open_rights() -> RightsDecision:
    return RightsDecision(
        dataset_license="unstated",
        license_scope=LicenseScope.UNSTATED,
        license_source=(
            "https://github.com/dwadden/scifact-open (no LICENSE file at the pinned revision)"
        ),
        underlying_content=(
            "S2ORC abstracts (ODC-By 1.0 under the SciFact license notes); shared claims and "
            "citation evidence from SciFact (CC BY 4.0); new pooling annotations carry no "
            "separate license"
        ),
        redistribution=Redistribution.PROHIBITED,
        attribution="Wadden et al. 2022 (SciFact-Open); Semantic Scholar (S2ORC)",
        outcome=RightsOutcome.ACCEPTED,
        basis=(
            "the authors publicly released the dataset for research via arXiv:2210.13777 and the "
            "repository README; no license file exists, so this repository accepts local "
            "evaluation use only, prohibits redistribution, and never republishes corpus bytes"
        ),
    )


def _qasper_rights() -> RightsDecision:
    return RightsDecision(
        dataset_license="CC-BY-4.0",
        license_scope=LicenseScope.OPEN,
        license_source="https://huggingface.co/datasets/allenai/qasper (dataset card)",
        underlying_content=(
            "paper full texts were extracted from S2ORC; original articles retain their "
            "publishers' rights"
        ),
        redistribution=Redistribution.UNVERIFIED,
        attribution="Dasigi et al. 2021 (QASPER); Semantic Scholar (S2ORC)",
        outcome=RightsOutcome.ACCEPTED,
        basis=(
            "the official loader and dataset card declare CC BY 4.0; evaluation use with "
            "attribution"
        ),
    )


def _nfcorpus_rights() -> RightsDecision:
    return RightsDecision(
        dataset_license="academic-use-only",
        license_scope=LicenseScope.RESEARCH_ONLY,
        license_source="https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/ (Terms of Use)",
        underlying_content=(
            "medical documents mostly from PubMed and queries derived from NutritionFacts.org "
            "content; non-academic use of that content requires the author's permission"
        ),
        redistribution=Redistribution.UNVERIFIED,
        attribution="Boteva et al. 2016 (NFCorpus); NutritionFacts.org (Dr. Michael Greger)",
        outcome=RightsOutcome.ACCEPTED,
        basis=(
            "the official terms declare the corpus free for academic purposes; this repository "
            "uses it offline for non-commercial retrieval evaluation with attribution"
        ),
    )


def _scidocs_rights() -> RightsDecision:
    return RightsDecision(
        dataset_license="CC-BY-4.0",
        license_scope=LicenseScope.OPEN,
        license_source="https://github.com/allenai/scidocs/blob/master/LICENSE",
        underlying_content=(
            "citation graphs and paper metadata derived from Semantic Scholar; the abstract and "
            "title text carries the source corpus rights"
        ),
        redistribution=Redistribution.UNVERIFIED,
        attribution="Cohan et al. 2020 (SciDocs/SPECTER); Semantic Scholar",
        outcome=RightsOutcome.ACCEPTED,
        basis=(
            "the SciDocs repository ships the CC BY 4.0 license text; evaluation use with "
            "attribution"
        ),
    )


def _arguana_rights() -> RightsDecision:
    return RightsDecision(
        dataset_license="CC-BY-SA-4.0",
        license_scope=LicenseScope.OPEN,
        license_source=(
            "https://huggingface.co/datasets/mteb/arguana (license metadata for the offline "
            "original at http://argumentation.bplaced.net/arguana/data)"
        ),
        underlying_content=(
            "user-generated web arguments; texts remain attributable to their authors and the "
            "share-alike term travels with reuse"
        ),
        redistribution=Redistribution.UNVERIFIED,
        attribution="Wachsmuth et al. 2018 (ArguAna); Thakur et al. 2021 (BEIR)",
        outcome=RightsOutcome.ACCEPTED,
        basis=(
            "CC BY-SA 4.0 is recorded by the maintained MTEB mirror of the original "
            "distribution, whose site is now offline; evaluation use with share-alike "
            "attribution preserved in the rights notice"
        ),
    )


def _fiqa_rights() -> RightsDecision:
    return RightsDecision(
        dataset_license="non-commercial-use-only",
        license_scope=LicenseScope.RESEARCH_ONLY,
        license_source=(
            'https://sites.google.com/view/fiqa/home ("available only for non-commercial use")'
        ),
        underlying_content=(
            "Stack Exchange investment Q&A posts and financial news; Stack Exchange content "
            "carries CC BY-SA terms"
        ),
        redistribution=Redistribution.UNVERIFIED,
        attribution="Maia et al. 2018 (FiQA); Stack Exchange contributors",
        outcome=RightsOutcome.ACCEPTED,
        basis=(
            "the official challenge site declares the data available for non-commercial use "
            "only; this repository uses it offline for non-commercial evaluation with "
            "attribution"
        ),
    )


RES141_SOURCES: Final[tuple[FrozenDatasetSource, ...]] = (
    FrozenDatasetSource(
        source_id="beir.scifact",
        family=FAMILY_BEIR,
        revision="beir-frozen-2021",
        documentation="https://github.com/beir-cellar/beir/wiki/Datasets-available",
        content_note=(
            "BEIR conversion of SciFact: 5,183 S2ORC abstracts, 1,109 claims, and BEIR's train "
            "and test qrels. BEIR train is SciFact claims_train; BEIR test is the original "
            "claims_dev. Every BEIR qrel is a binary link judgment, not a claim-veracity label."
        ),
        artifacts=(
            SourceArtifact(
                url=(
                    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
                ),
                archive_name="scifact.zip",
                format=SourceFormat.ZIP,
                size_bytes=2816079,
                sha256="536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165",
                members=(
                    SourceMember(
                        name="scifact/corpus.jsonl",
                        size_bytes=8106566,
                        sha256=("dec31c8182f3d744c7d2c09423756fd1d17cbef75808db13ba01cc0aab4d1ac6"),
                    ),
                    SourceMember(
                        name="scifact/qrels/test.tsv",
                        size_bytes=5389,
                        sha256=("0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c"),
                    ),
                    SourceMember(
                        name="scifact/qrels/train.tsv",
                        size_bytes=14502,
                        sha256=("a53f2114831916c096b6c37d9e54da68cef4efdcdbd5ed46533601af972acf1d"),
                    ),
                    SourceMember(
                        name="scifact/queries.jsonl",
                        size_bytes=209731,
                        sha256=("8ff84a7c903f722981cd8d595c022660140c51867b27608a6d4910db86080313"),
                    ),
                ),
            ),
        ),
        rights=_scifact_rights(),
    ),
    FrozenDatasetSource(
        source_id="scifact-open",
        family=FAMILY_SCIFACT_OPEN,
        revision="release-2022-12-05",
        documentation="https://github.com/dwadden/scifact-open/blob/main/doc/data.md",
        content_note=(
            "SciFact-Open: 279 claims (SciFact test less 21 removed for missing source "
            "metadata), 460 evidence links over 406 S2ORC abstracts, a 500K-abstract corpus "
            "and the 12,236-abstract pooled candidate subset. Evidence provenance is citation "
            "(SciFact, hand-annotated) or pooling (top-250 model predictions, machine "
            "highlights)."
        ),
        artifacts=(
            SourceArtifact(
                url=(
                    "https://scifact.s3.us-west-2.amazonaws.com/scifact-open/latest/"
                    "scifact_open.tar.gz"
                ),
                archive_name="scifact_open.tar.gz",
                format=SourceFormat.TAR_GZ,
                size_bytes=287369773,
                sha256="8f4dc238b2eff422bab5c7918fcbddb6996b29d37cb887942ef6ee1a16fb3415",
                members=(
                    SourceMember(
                        name="data/claims.jsonl",
                        size_bytes=101404,
                        sha256=("ecd7a1bec2fc1b483f9127f6d0927ff17ebeb08e120088c6e037dfc85ce2b63b"),
                    ),
                    SourceMember(
                        name="data/claims_metadata.jsonl",
                        size_bytes=105777,
                        sha256=("8151419a85d7dabe076c0b38d9cf4cb96ad59dec3c3b1be1781b2b75c1d93195"),
                    ),
                    SourceMember(
                        name="data/corpus.jsonl",
                        size_bytes=889005055,
                        sha256=("ae01ebc375974110b7e672ce1a1866ba56ec15e450f4aededfd2f263aa1b640a"),
                    ),
                    SourceMember(
                        name="data/corpus_candidates.jsonl",
                        size_bytes=24850554,
                        sha256=("664801e5088c177cd09c01bb410b61e73312bf393f88726f2b9ffb8416f44e0f"),
                    ),
                    SourceMember(
                        name="prediction/retrievals.jsonl",
                        size_bytes=143679,
                        sha256=("2e83af109e69e850e7a0ed9413441f5eba0e550a0a03edcd66aafe2a7eb77e2e"),
                    ),
                ),
            ),
        ),
        rights=_scifact_open_rights(),
    ),
    FrozenDatasetSource(
        source_id="qasper",
        family=FAMILY_QASPER,
        revision="v0.3.0",
        documentation="https://allenai.org/data/qasper",
        content_note=(
            "QASPER v0.3.0: 1,585 NLP papers, 5,049 information-seeking questions and their "
            "multi-annotator answers with paragraph-level evidence. Train and dev ship in one "
            "tarball, test in another with the official evaluator. Splits are train (888), "
            "validation (281) and test (416) papers."
        ),
        artifacts=(
            SourceArtifact(
                url=("https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"),
                archive_name="qasper-train-dev-v0.3.tgz",
                format=SourceFormat.TAR_GZ,
                size_bytes=10835856,
                sha256="a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a",
                members=(
                    SourceMember(
                        name="qasper-dev-v0.3.json",
                        size_bytes=11398686,
                        sha256=("2ae7ee62a65b1c4225791c70de80c2aad4e8998cf1fd4f09a53103db4f21af93"),
                    ),
                    SourceMember(
                        name="qasper-train-v0.3.json",
                        size_bytes=31969387,
                        sha256=("9458bfe76074a8fa8d1685af02bcc73537aa6d338ad20591dfaff1946bc88bf4"),
                    ),
                ),
            ),
            SourceArtifact(
                url=(
                    "https://qasper-dataset.s3.us-west-2.amazonaws.com/"
                    "qasper-test-and-evaluator-v0.3.tgz"
                ),
                archive_name="qasper-test-and-evaluator-v0.3.tgz",
                format=SourceFormat.TAR_GZ,
                size_bytes=3865061,
                sha256="72a52a41193e2838b8074f80ac074b94f956b84886c36a61c58a7df4171bdd72",
                members=(
                    SourceMember(
                        name="qasper-test-v0.3.json",
                        size_bytes=18078957,
                        sha256=("6e29ad410e6e39aa1936017fb965b30a20eb2e7751997f55b97c9d281aa884e5"),
                    ),
                ),
            ),
        ),
        rights=_qasper_rights(),
    ),
    FrozenDatasetSource(
        source_id="beir.nfcorpus",
        family=FAMILY_BEIR,
        revision="beir-frozen-2021",
        documentation="https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/",
        content_note=(
            "NFCorpus medical IR: 3,633 PubMed documents, 3,237 NutritionFacts.org queries and "
            "query-disjoint train (2,590), dev (324) and test (323) splits with graded 1-2 "
            "judgments."
        ),
        artifacts=(
            SourceArtifact(
                url=(
                    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/"
                    "nfcorpus.zip"
                ),
                archive_name="nfcorpus.zip",
                format=SourceFormat.ZIP,
                size_bytes=2448432,
                sha256="efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b",
                members=(
                    SourceMember(
                        name="nfcorpus/corpus.jsonl",
                        size_bytes=6219364,
                        sha256=("10cc83ef1826b1425e6a87090b5140b39b27755d5a27e48215a88611c899991f"),
                    ),
                    SourceMember(
                        name="nfcorpus/qrels/dev.tsv",
                        size_bytes=258218,
                        sha256=("b1d38b5e8f78c4a5820bce2b7ec2db54911d7690dc601e76811846b211180bd8"),
                    ),
                    SourceMember(
                        name="nfcorpus/qrels/test.tsv",
                        size_bytes=279572,
                        sha256=("f8fba6ef3d4dd9c3a242a8ba4ae38276fc3622fce7dcbae764766d564542fd2a"),
                    ),
                    SourceMember(
                        name="nfcorpus/qrels/train.tsv",
                        size_bytes=2504643,
                        sha256=("6336b80f9bffc4f063f3aa450047ad35c0b7c534efe4a6ba35e16dbace047f6a"),
                    ),
                    SourceMember(
                        name="nfcorpus/queries.jsonl",
                        size_bytes=441466,
                        sha256=("d024e6621b84925d485ae473d316a0c3af31c62c8068a59fb29d22f7613aef2a"),
                    ),
                ),
            ),
        ),
        rights=_nfcorpus_rights(),
    ),
    FrozenDatasetSource(
        source_id="beir.scidocs",
        family=FAMILY_BEIR,
        revision="beir-frozen-2021",
        documentation="https://allenai.org/data/scidocs",
        content_note=(
            "SciDocs citation-prediction retrieval: 25,657 papers and 1,000 test queries with "
            "explicitly judged non-relevant documents (25,000 zero-score qrels) and 4,928 "
            "positive judgments."
        ),
        artifacts=(
            SourceArtifact(
                url=(
                    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip"
                ),
                archive_name="scidocs.zip",
                format=SourceFormat.ZIP,
                size_bytes=142471588,
                sha256="96640201687767c9b1fcc5af7a80b90fb325b37fa25329c2586c25edcfa17ef1",
                members=(
                    SourceMember(
                        name="scidocs/corpus.jsonl",
                        size_bytes=257334350,
                        sha256=("328bb38854179c83ee40d4f49fae2ef7209ede7163785193cbdcd5906b28eecc"),
                    ),
                    SourceMember(
                        name="scidocs/qrels/test.tsv",
                        size_bytes=2543906,
                        sha256=("dcd3d7f77417294bb6f338537f3c28d8a5ea72b30fe6633f407fa85528767e35"),
                    ),
                    SourceMember(
                        name="scidocs/queries.jsonl",
                        size_bytes=3166530,
                        sha256=("3c85b68419bd579ff52edb58f9e7b32eb3d18bc83435edb71dbb68aaf79531d6"),
                    ),
                ),
            ),
        ),
        rights=_scidocs_rights(),
    ),
    FrozenDatasetSource(
        source_id="beir.arguana",
        family=FAMILY_BEIR,
        revision="beir-frozen-2021",
        documentation="https://huggingface.co/datasets/mteb/arguana",
        content_note=(
            "ArguAna counter-argument retrieval: 8,674 arguments and 1,406 test queries with "
            "one relevant counter-argument each. Five qrel documents are absent from the "
            "distributed corpus; the slice declares and excludes them rather than inventing "
            "documents."
        ),
        artifacts=(
            SourceArtifact(
                url=(
                    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip"
                ),
                archive_name="arguana.zip",
                format=SourceFormat.ZIP,
                size_bytes=3773617,
                sha256="cfdf79adce27a401b3cd3ea267903134dbfab2c6afeb95d7fe5724a00bf7557b",
                members=(
                    SourceMember(
                        name="arguana/corpus.jsonl",
                        size_bytes=9829628,
                        sha256=("0627982cec4ee896df6dc98604ef6b499c83fd87a19eaaeb3f23b88bbef6cbce"),
                    ),
                    SourceMember(
                        name="arguana/qrels/test.tsv",
                        size_bytes=96296,
                        sha256=("0c47b481fa8b47fb9ca5e74bc8a017bdea14518c0e4a5d21345291180b3aca76"),
                    ),
                    SourceMember(
                        name="arguana/queries.jsonl",
                        size_bytes=1805776,
                        sha256=("f8b7c903f95d9a2d291de3915a8fac25085e9db679f19f59fea4145bfd1df618"),
                    ),
                ),
            ),
        ),
        rights=_arguana_rights(),
    ),
    FrozenDatasetSource(
        source_id="beir.fiqa",
        family=FAMILY_BEIR,
        revision="beir-frozen-2021",
        documentation="https://sites.google.com/view/fiqa/home",
        content_note=(
            "FiQA-2018 financial QA retrieval: 57,638 Stack Exchange posts, 6,648 queries and "
            "query-disjoint train (5,500), dev (500) and test (648) splits with binary link "
            "judgments."
        ),
        artifacts=(
            SourceArtifact(
                url=("https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"),
                archive_name="fiqa.zip",
                format=SourceFormat.ZIP,
                size_bytes=17948027,
                sha256="32c7df99ed21252fdfb2cf3f5673502a8d245ee0c44c4a133570d92ce2b3ad02",
                members=(
                    SourceMember(
                        name="fiqa/corpus.jsonl",
                        size_bytes=47949497,
                        sha256=("ff593e4df9933955dc3af83be0c3fa28ac7465f627e08c2e53593e734d506517"),
                    ),
                    SourceMember(
                        name="fiqa/qrels/dev.tsv",
                        size_bytes=18327,
                        sha256=("03b27e547cd29dc721c3b93ce8ddd70df6b2d0ce954d8ef3659d386e8cd72c11"),
                    ),
                    SourceMember(
                        name="fiqa/qrels/test.tsv",
                        size_bytes=25256,
                        sha256=("6adc2a640dcdd22bb8b3858f89107adef2a7c3db20a63550dfa7a0f71e379e44"),
                    ),
                    SourceMember(
                        name="fiqa/qrels/train.tsv",
                        size_bytes=209842,
                        sha256=("e46a99529aa61b12e086a5c057e4e0ecaeda9502e53d5fe855efce778f1ec59a"),
                    ),
                    SourceMember(
                        name="fiqa/queries.jsonl",
                        size_bytes=706247,
                        sha256=("eede1e61d4a0188940239b53ebc2da91f577a6a34679c812d1eb9090c29877bc"),
                    ),
                ),
            ),
        ),
        rights=_fiqa_rights(),
    ),
)


def source_by_id(source_id: str) -> FrozenDatasetSource:
    """The registered source called ``source_id``, or a contract error."""
    for source in RES141_SOURCES:
        if source.source_id == source_id:
            return source
    raise DatasetContractError(
        f"no frozen source is registered as {source_id!r}.",
        operation="source_by_id",
        item_id=source_id,
        expected=str([source.source_id for source in RES141_SOURCES]),
    )
