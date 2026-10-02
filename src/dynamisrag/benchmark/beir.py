"""Frozen BEIR acquisition, verification and workload loading.

Three workloads, three original ZIP distributions, three SHA-256 digests. The
rule that makes this reproducible is short: **the digest is the only authority**,
and it is checked before extraction, again after any copy, and never inferred
from a filename, a size or a folder.

    first run   download to local scratch -> verify -> extract -> cache the ZIP
    later run   copy the cached ZIP to local scratch -> verify -> extract

A cached file is re-verified on every use even though it was verified once before
it was written. That is not paranoia about the code: Drive is a synchronising
network filesystem shared with a browser, and the cost of the check is one
streaming hash of a file that is about to be read anyway.

**No parquet, no ``pyarrow``, no Hub dataset.** BEIR ships JSONL and TSV inside the
ZIP and the standard library reads both. The alternative — ``datasets.load_dataset``
over the Hub's parquet conversion — would substitute a repackaging of the corpus
for the corpus, and its row order, its null handling and its id types would become
undocumented inputs to a benchmark whose whole point is that its inputs are
frozen.

**Two declared loader policies**, both of which change the corpus and so are
recorded in the source manifest:

* :data:`~dynamisrag.benchmark.contracts.RES138_DOCUMENT_TEXT_POLICY` — how a
  ``title``/``text`` pair becomes the one string that is embedded. Implemented
  once, in
  :func:`~dynamisrag.benchmark.contracts.beir_document_embedding_text`, and called
  here rather than interpolated by a notebook.
* :data:`~dynamisrag.benchmark.contracts.RES138_QUERY_SELECTION_POLICY` — the
  workload's queries are exactly those with a qrel row in the frozen split.
  NFCorpus ships 3,237 query records for 323 judged ones; embedding the rest
  would spend GPU time on vectors no metric can use, and dropping them silently
  would make the workload unaccountable for. The excluded count is recorded.

**One document in TREC-COVID has neither a title nor a body** and is excluded,
because RES-137 refuses empty passage text and a well-formed meaningless vector
would occupy a corpus slot and perturb ``Recall@100``. The exclusion is counted
and its id list hashed, not hidden: 42,140 TREC-COVID documents have an empty
*body* but a title, and they are embedded as their title rather than dropped.
"""

from __future__ import annotations

import json
import shutil
import urllib.request
import zipfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from dynamisrag.benchmark.artifacts import Res138JsonValue, file_sha256
from dynamisrag.benchmark.contracts import (
    BEIR_QREL_SPLIT,
    RES138_BEIR_SOURCES,
    RES138_DOCUMENT_TEXT_POLICY,
    RES138_QUERY_SELECTION_POLICY,
    BeirSourceSpec,
    RetrievalDocument,
    RetrievalQrel,
    RetrievalQuery,
    RetrievalWorkload,
    beir_document_embedding_text,
    ordered_ids_sha256,
    text_sha256,
)
from dynamisrag.benchmark.errors import BenchmarkSourceError

__all__ = [
    "BeirWorkloadReport",
    "VerifiedSource",
    "extract_verified_archive",
    "load_beir_workload",
    "sha256_of_file",
    "verify_and_cache_beir_sources",
]

_QREL_HEADER: Final[tuple[str, ...]] = ("query-id", "corpus-id", "score")
"""The BEIR qrels header, verified rather than skipped.

A TSV whose header is not this one is not a BEIR qrels file, and parsing it by
position would silently read judgments out of the wrong columns.
"""

_DOWNLOAD_TIMEOUT_SECONDS: Final[int] = 600


def sha256_of_file(path: Path, *, spec: BeirSourceSpec) -> str:
    """SHA-256 of a candidate archive, or refuse it against the frozen digest.

    The mismatch message names the workload and both digests and never the
    contents: an archive that does not match is either a corrupt transfer or a
    different distribution, and printing its text would republish third-party
    scientific content through a log line.
    """
    observed = file_sha256(path)
    if observed == spec.sha256:
        return observed
    raise BenchmarkSourceError(
        f"the archive for workload {spec.workload!r} hashed {observed}, which is not "
        f"the frozen {spec.sha256}. Either the transfer was truncated or the distribution changed. "
        "Nothing was extracted and nothing was cached: an unverified archive must never reach the "
        "corpus.",
        operation="verify_beir_source",
        workload=spec.workload,
        expected=spec.sha256,
        observed=observed,
    )


@dataclass(frozen=True)
class VerifiedSource:
    """One archive, verified against its frozen digest, at a known local path."""

    spec: BeirSourceSpec
    path: Path
    sha256: str

    def payload(self) -> dict[str, Res138JsonValue]:
        """The hashed description of this verified source."""
        return {
            **cast("dict[str, Res138JsonValue]", dict(self.spec.payload())),
            "verified_sha256": self.sha256,
        }


def _download(spec: BeirSourceSpec, destination: Path) -> None:
    """Fetch one archive over https, streamed to ``destination``.

    Streamed rather than downloaded to memory and written, because TREC-COVID is
    73 MB and a hosted session has no reason to hold it twice. The partial file is
    removed on failure so a retry does not verify a truncated file.
    """
    partial = destination.with_name(f"{destination.name}.partial")
    try:
        with (
            urllib.request.urlopen(  # noqa: S310
                spec.url, timeout=_DOWNLOAD_TIMEOUT_SECONDS
            ) as response,
            partial.open("wb") as handle,
        ):
            shutil.copyfileobj(response, handle)
        partial.replace(destination)
    except (OSError, ValueError) as error:
        partial.unlink(missing_ok=True)
        raise BenchmarkSourceError(
            f"the archive for workload {spec.workload!r} could not be fetched from its frozen URL "
            f"({type(error).__name__}). Nothing was extracted and nothing was cached.",
            operation="download_beir_source",
            workload=spec.workload,
        ) from None


def verify_and_cache_beir_sources(
    *,
    scratch_dir: Path,
    cache_dir: Path,
    specs: Sequence[BeirSourceSpec] = RES138_BEIR_SOURCES,
) -> tuple[VerifiedSource, ...]:
    """Make every frozen archive available locally, verified, caching what is new.

    Three cases, in this order for each workload: a verified copy already in local
    scratch; otherwise a cached copy copied out of Drive and verified **again**;
    otherwise a download, verified, and only then copied into the cache with its
    digest re-checked on the way in.

    The order matters. Verifying after the copy into the cache is what makes the
    cache trustworthy at all: a cache entry that was never hashed is a file of
    unknown provenance wearing a name from the frozen spec.
    """
    scratch_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    verified: list[VerifiedSource] = []
    for spec in specs:
        local = scratch_dir / spec.archive_name
        cached = cache_dir / spec.archive_name
        if local.exists():
            digest = sha256_of_file(local, spec=spec)
        elif cached.exists():
            shutil.copyfile(cached, local)
            digest = sha256_of_file(local, spec=spec)
        else:
            _download(spec, local)
            digest = sha256_of_file(local, spec=spec)
            shutil.copyfile(local, cached)
            cached_digest = file_sha256(cached)
            if cached_digest != spec.sha256:
                cached.unlink(missing_ok=True)
                raise BenchmarkSourceError(
                    f"the cached copy of the {spec.workload!r} archive hashed {cached_digest} on "
                    f"arrival, not the frozen {spec.sha256}, and has been removed. A cache entry "
                    "that cannot be re-verified is not evidence of anything.",
                    operation="verify_and_cache_beir_sources",
                    workload=spec.workload,
                    expected=spec.sha256,
                    observed=cached_digest,
                )
        verified.append(VerifiedSource(spec=spec, path=local, sha256=digest))
    return tuple(verified)


def extract_verified_archive(source: VerifiedSource, destination: Path) -> Path:
    """Extract one verified archive under ``destination`` and return the root.

    Re-verifies first, so an archive that was verified on acquisition and has
    since been replaced cannot be extracted. The destination is required to be
    empty of the archive's own top-level directory, so a second run overwrites
    rather than nesting.
    """
    sha256_of_file(source.path, spec=source.spec)
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(source.path) as archive:
            names = archive.namelist()
            required = (
                f"{source.spec.workload}/corpus.jsonl",
                f"{source.spec.workload}/queries.jsonl",
                f"{source.spec.workload}/qrels/{BEIR_QREL_SPLIT}.tsv",
            )
            missing = [member for member in required if member not in names]
            if missing:
                raise BenchmarkSourceError(
                    f"the {source.spec.workload!r} archive does not have the expected BEIR layout; "
                    f"missing {missing}. Nothing was extracted.",
                    operation="extract_verified_archive",
                    workload=source.spec.workload,
                )
            archive.extractall(destination)
    except zipfile.BadZipFile:
        raise BenchmarkSourceError(
            f"the {source.spec.workload!r} archive is not a readable ZIP, despite matching its "
            "frozen digest. That combination is impossible, so the working copy or the extraction "
            "destination is wrong.",
            operation="extract_verified_archive",
            workload=source.spec.workload,
        ) from None
    return destination / source.spec.workload


@dataclass(frozen=True)
class BeirWorkloadReport:
    """What a loaded workload contains, for the ``res138-source-manifest-v1`` artifact.

    Counts and digests only. A source manifest states what was measured without
    republishing 180,147 documents of third-party scientific text, and two
    manifests can be compared for equality without either corpus in hand.
    """

    spec: BeirSourceSpec
    workload_summary: dict[str, object]
    queries_in_archive: int
    documents_in_archive: int
    documents_without_embedding_text: int
    excluded_document_ids_sha256: str | None
    queries_without_judgement: int
    queries_without_embedding_text: int
    qrel_rows: int
    max_relevance: int
    min_relevance: int

    def payload(self) -> dict[str, Res138JsonValue]:
        """The hashed description of the loaded workload."""
        summary: dict[str, Res138JsonValue] = cast(
            "dict[str, Res138JsonValue]", dict(self.workload_summary)
        )
        return {
            "artifact_revision": "res138-source-manifest-v1",
            **cast("dict[str, Res138JsonValue]", dict(self.spec.payload())),
            "document_text_policy": RES138_DOCUMENT_TEXT_POLICY,
            "query_selection_policy": RES138_QUERY_SELECTION_POLICY,
            "qrel_split": BEIR_QREL_SPLIT,
            "workload": summary,
            "documents_in_archive": self.documents_in_archive,
            "documents_without_embedding_text": self.documents_without_embedding_text,
            "excluded_document_ids_sha256": self.excluded_document_ids_sha256,
            "queries_in_archive": self.queries_in_archive,
            "queries_without_judgement": self.queries_without_judgement,
            "queries_without_embedding_text": self.queries_without_embedding_text,
            "qrel_rows": self.qrel_rows,
            "max_relevance": self.max_relevance,
            "min_relevance": self.min_relevance,
        }


def _jsonl_records(path: Path, *, workload: str) -> Iterator[dict[str, object]]:
    """Stream one JSONL member as decoded objects, refusing anything else.

    Streamed because TREC-COVID's ``corpus.jsonl`` is 221 MB: materialising it as
    a list of dicts would hold the whole corpus twice, and a corpus this size is
    exactly why the loader sorts by id rather than trusting file order.
    """
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as error:
                raise BenchmarkSourceError(
                    f"{path.name} line {number} is not valid JSON ({error}).",
                    operation="load_beir_workload",
                    workload=workload,
                ) from None
            if not isinstance(record, dict):
                raise BenchmarkSourceError(
                    f"{path.name} line {number} is a JSON {type(record).__name__}, not an object.",
                    operation="load_beir_workload",
                    workload=workload,
                )
            yield cast("dict[str, object]", record)


def _required_text(record: dict[str, object], *, field_name: str, path: Path) -> str:
    value = record.get(field_name)
    if not isinstance(value, str):
        raise BenchmarkSourceError(
            f"{path.name} holds a record whose {field_name} is {value!r} rather than text.",
            operation="load_beir_workload",
        )
    return value


def _load_qrels(path: Path, *, workload: str) -> tuple[RetrievalQrel, ...]:
    """Read one BEIR qrels TSV, verifying its header and its score column."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise BenchmarkSourceError(
            f"{path.name} is empty.", operation="load_beir_workload", workload=workload
        )
    header = tuple(lines[0].split("\t"))
    if header != _QREL_HEADER:
        raise BenchmarkSourceError(
            f"{path.name} declares header {header}, not the BEIR qrels header "
            f"{list(_QREL_HEADER)}. Parsing by position would read judgments out of the wrong "
            "columns.",
            operation="load_beir_workload",
            workload=workload,
            expected=str(list(_QREL_HEADER)),
            observed=str(list(header)),
        )
    qrels: list[RetrievalQrel] = []
    for number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        columns = line.split("\t")
        if len(columns) != 3:
            raise BenchmarkSourceError(
                f"{path.name} line {number} has {len(columns)} columns, not 3.",
                operation="load_beir_workload",
                workload=workload,
                count=number,
            )
        raw_score = columns[2].strip()
        try:
            score = int(raw_score)
        except ValueError:
            raise BenchmarkSourceError(
                f"{path.name} line {number} has score {raw_score!r}, which is not an integer "
                "judgment level.",
                operation="load_beir_workload",
                workload=workload,
                count=number,
            ) from None
        qrels.append(
            RetrievalQrel(
                query_id=columns[0].strip(), document_id=columns[1].strip(), relevance=score
            )
        )
    return tuple(qrels)


def load_beir_workload(
    root: Path, spec: BeirSourceSpec
) -> tuple[RetrievalWorkload, BeirWorkloadReport]:
    """Load one extracted BEIR workload into canonical, validated form.

    ``root`` is the directory the verified archive was extracted to. The order is
    rebuilt by sorting, not inherited from file order: a corpus whose file order
    changed would produce different shard boundaries and different request
    sequences, and neither change would be visible in any artifact.

    Every declaration the loader makes — which documents had no embeddable text,
    how many queries carried no judgment — is returned in the report rather than
    applied silently.
    """
    corpus_path = root / "corpus.jsonl"
    queries_path = root / "queries.jsonl"
    qrels_path = root / "qrels" / f"{BEIR_QREL_SPLIT}.tsv"
    qrels = _load_qrels(qrels_path, workload=spec.workload)
    judged = {qrel.query_id for qrel in qrels}

    documents: list[RetrievalDocument] = []
    excluded: list[str] = []
    documents_in_archive = 0
    for record in _jsonl_records(corpus_path, workload=spec.workload):
        documents_in_archive += 1
        document_id = _required_text(record, field_name="_id", path=corpus_path)
        title = record.get("title")
        body = _required_text(record, field_name="text", path=corpus_path)
        text = beir_document_embedding_text(title if isinstance(title, str) else "", body)
        if not text.strip():
            # Declared, not silent: a document with no embeddable text has no
            # meaningful vector, and RES-137 refuses empty passage text for exactly
            # that reason. One such document exists in TREC-COVID.
            excluded.append(document_id)
            continue
        documents.append(
            RetrievalDocument(
                document_id=document_id,
                title=title.strip() if isinstance(title, str) else "",
                text=text,
                content_sha256=text_sha256(text),
            )
        )

    queries: list[RetrievalQuery] = []
    queries_in_archive = 0
    queries_without_text: list[str] = []
    for record in _jsonl_records(queries_path, workload=spec.workload):
        queries_in_archive += 1
        query_id = _required_text(record, field_name="_id", path=queries_path)
        if query_id not in judged:
            continue
        text = _required_text(record, field_name="text", path=queries_path)
        if not text.strip():
            # A judged query with no embeddable text has no vector and therefore no
            # metric row. Counted rather than dropped without record: dropping it
            # silently would change the metric denominator for that query.
            queries_without_text.append(query_id)
            continue
        queries.append(RetrievalQuery.from_beir(query_id=query_id, text=text))

    workload = RetrievalWorkload(
        name=spec.workload,
        documents=tuple(sorted(documents, key=lambda item: item.document_id)),
        queries=tuple(sorted(queries, key=lambda item: item.query_id)),
        qrels=tuple(sorted(qrels, key=lambda item: (item.query_id, item.document_id))),
    )
    relevances = [qrel.relevance for qrel in qrels]
    report = BeirWorkloadReport(
        spec=spec,
        workload_summary=dict(workload.summary()),
        queries_in_archive=queries_in_archive,
        documents_in_archive=documents_in_archive,
        documents_without_embedding_text=len(excluded),
        excluded_document_ids_sha256=ordered_ids_sha256(sorted(excluded)) if excluded else None,
        queries_without_judgement=queries_in_archive - len(workload.queries),
        queries_without_embedding_text=len(queries_without_text),
        qrel_rows=len(qrels),
        max_relevance=max(relevances) if relevances else 0,
        min_relevance=min(relevances) if relevances else 0,
    )
    return workload, report
