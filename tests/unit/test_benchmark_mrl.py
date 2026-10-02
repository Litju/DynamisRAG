"""MRL derivation, the calibration gate, the deterministic sample, and BEIR loading.

**The derivation is arithmetic, and arithmetic is asserted exactly.** A 1024
float32 unit vector's first 512 components, renormalised, must be a 512 float32
unit vector — checked against the norm, not against a previous run. Wrong input
dimension, non-finite components and a zero prefix norm are refused rather than
corrected, and no rounding is applied, because a rounded derived vector would
differ from the native one in the last place and the comparison below would then
be measuring the formatting.

**The gate decides, and the decision cannot disagree with its own evidence.** An
identical-path model passes and is allowed to derive; a model whose 512 output is
merely *correlated* with its prefix fails the cosine gate, and one whose ordering
moves fails the ordering gate, each for its own stated reason. A decision marked
allowed while recording failing numbers is refused, because the two are separate
fields precisely so they cannot.

**The calibration sample is a rule, not a choice.** Within-workload thirds, then
the smallest SHA-256 of the identity, two per cell — deterministic, reproducible
from the workload alone, spanning every workload on both sides. An empty cell is
an error, and so is a workload too small to have three bands.

**BEIR loading is verified end to end without touching the network.** Synthetic
ZIP fixtures with real digests: a correct archive loads into a canonically ordered
workload, a corrupted archive is refused before extraction, a cached archive is
re-verified on every use, a wrong qrels header is refused rather than parsed by
position, and the two declared loader policies — no embeddable text, and only
judged queries — are recorded in the report instead of applied silently.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Final, cast

import numpy as np
import pytest

from dynamisrag.benchmark.artifacts import ShardKind
from dynamisrag.benchmark.beir import (
    BeirSourceSpec,
    VerifiedSource,
    extract_verified_archive,
    load_beir_workload,
    verify_and_cache_beir_sources,
)
from dynamisrag.benchmark.calibration import (
    CalibrationItem,
    select_calibration_set,
)
from dynamisrag.benchmark.contracts import (
    RES138_BASE_DIMENSION,
    RES138_BEIR_SOURCES,
    RES138_CALIBRATION_BANDS,
    RES138_CALIBRATION_ITEMS_PER_CELL,
    RES138_MODEL_CANDIDATES,
    RES138_MRL_DERIVATION_REVISION,
    RetrievalDocument,
    RetrievalQrel,
    RetrievalQuery,
    RetrievalWorkload,
    text_sha256,
)
from dynamisrag.benchmark.errors import BenchmarkContractError, BenchmarkSourceError
from dynamisrag.benchmark.mrl import (
    MrlPathDecision,
    build_mrl_calibration_payload,
    derive_mrl_prefix,
    evaluate_mrl_equivalence,
)

_DIMENSION: Final[int] = RES138_BASE_DIMENSION
_HALF: Final[int] = _DIMENSION // 2
_CANDIDATE = RES138_MODEL_CANDIDATES[0]


def _unit_rows(count: int, seed: int = 7) -> np.ndarray:
    generator = np.random.default_rng(seed)
    raw = generator.normal(size=(count, _DIMENSION)).astype(np.float32)
    norms = np.linalg.norm(raw.astype(np.float64), axis=1, keepdims=True)
    return np.ascontiguousarray(raw / norms, dtype=np.float32)


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------


def test_the_derivation_takes_the_prefix_and_renormalises_it() -> None:
    source = _unit_rows(3, seed=11)

    derived = derive_mrl_prefix(source, operation="test")

    assert derived.shape == (3, _HALF)
    assert derived.dtype == np.float32
    for index in range(3):
        expected = source[index, :_HALF].astype(np.float64)
        expected /= np.linalg.norm(expected)
        assert np.allclose(derived[index], expected.astype(np.float32), atol=0, rtol=0)
    norms = np.linalg.norm(derived.astype(np.float64), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_the_derivation_is_not_idempotent_and_is_named_as_a_revision() -> None:
    """Deriving from a 512 matrix is refused, not silently re-derived.

    ``mrl-prefix-renorm-v1`` describes one operation on a 1024 vector. Applying it
    twice would produce a different set of vectors under the same revision.
    """
    derived = derive_mrl_prefix(_unit_rows(2), operation="test")
    with pytest.raises(BenchmarkContractError):
        derive_mrl_prefix(derived, operation="test")
    assert RES138_MRL_DERIVATION_REVISION == "mrl-prefix-renorm-v1"


def test_the_derivation_can_be_targeted_at_either_frozen_dimension() -> None:
    with pytest.raises(BenchmarkContractError):
        derive_mrl_prefix(_unit_rows(1), operation="test", dimension=256)


@pytest.mark.parametrize(
    "matrix",
    [
        pytest.param(np.zeros((1, _DIMENSION), dtype=np.float32), id="zero-prefix-norm"),
        pytest.param(
            np.concatenate(
                [np.zeros((1, _HALF), dtype=np.float32), np.ones((1, _HALF), dtype=np.float32)],
                axis=1,
            ),
            id="zero-first-half",
        ),
    ],
)
def test_a_prefix_with_no_direction_is_refused(matrix: np.ndarray) -> None:
    with pytest.raises(BenchmarkContractError) as caught:
        derive_mrl_prefix(matrix, operation="test")
    assert "direction" in str(caught.value)


def test_a_non_finite_or_wrong_typed_matrix_is_refused() -> None:
    broken = _unit_rows(2)
    broken[0, 5] = np.nan
    with pytest.raises(BenchmarkContractError):
        derive_mrl_prefix(broken, operation="test")
    with pytest.raises(BenchmarkContractError):
        derive_mrl_prefix(_unit_rows(2).astype(np.float64), operation="test")  # pyright: ignore[reportArgumentType]
    with pytest.raises(BenchmarkContractError):
        derive_mrl_prefix(_unit_rows(2)[0], operation="test")
    with pytest.raises(BenchmarkContractError):
        derive_mrl_prefix(_unit_rows(2)[:, :16], operation="test")


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def _decide(
    *,
    native_512: np.ndarray,
    native_1024: np.ndarray,
    item_ids: tuple[str, ...],
    kind: ShardKind = ShardKind.DOCUMENTS,
) -> MrlPathDecision:
    return evaluate_mrl_equivalence(
        candidate=_CANDIDATE,
        kind=kind,
        item_ids=item_ids,
        native_512=native_512,
        native_1024=native_1024,
        operation="test",
    )


def _exact_pair(count: int, *, seed: int = 21) -> tuple[np.ndarray, np.ndarray]:
    """A native-512 matrix that *is* the derived prefix, i.e. a perfect shortcut."""
    full = _unit_rows(count, seed=seed)
    derived = derive_mrl_prefix(full, operation="test")
    return derived, full


def test_a_model_whose_native_512_is_the_derived_prefix_passes_the_gate() -> None:
    native_512, native_1024 = _exact_pair(18)
    item_ids = tuple(f"d{index:03d}" for index in range(18))

    decision = _decide(native_512=native_512, native_1024=native_1024, item_ids=item_ids)

    assert decision.derived512_allowed is True
    assert decision.failures() == ()
    assert decision.minimum_cosine == pytest.approx(1.0, abs=1e-9)
    assert decision.maximum_absolute_difference == pytest.approx(0.0, abs=1e-9)
    assert decision.identical_top_k is True
    assert decision.derived_dimension == _HALF
    assert decision.vector_count == 18
    assert decision.derivation_revision == RES138_MRL_DERIVATION_REVISION


def test_a_native_512_that_is_only_correlated_with_its_prefix_fails_the_cosine_gate() -> None:
    native_1024 = _unit_rows(18, seed=33)
    unrelated = derive_mrl_prefix(_unit_rows(18, seed=99), operation="test")

    decision = _decide(native_512=unrelated, native_1024=native_1024, item_ids=_ids(18))

    assert decision.derived512_allowed is False
    assert "minimum_cosine" in decision.failures()
    assert decision.minimum_cosine < 0.999999


def test_a_near_miss_below_the_predeclared_cosine_is_refused() -> None:
    """1e-5 is the tolerance; a difference just above it must fail, not round away."""
    native_1024 = _unit_rows(18, seed=44)
    native_512 = derive_mrl_prefix(native_1024, operation="test")
    # Perturb one component by a relative amount above the gate, then renormalise
    # so the comparison measures the shortcut rather than a lost normalisation.
    native_512[0, 0] += np.float32(1e-3)
    norms = np.linalg.norm(native_512.astype(np.float64), axis=1, keepdims=True)
    native_512 = np.ascontiguousarray(native_512 / norms, dtype=np.float32)

    decision = _decide(native_512=native_512, native_1024=native_1024, item_ids=_ids(18))

    assert decision.derived512_allowed is False
    assert decision.failures()


def test_a_query_path_can_fail_while_its_document_path_passes() -> None:
    """The two paths are decided separately: pooling differs between models, and prompts differ."""
    exact_512, full = _exact_pair(18, seed=55)
    ids = _ids(18)

    documents = _decide(
        native_512=exact_512, native_1024=full, item_ids=ids, kind=ShardKind.DOCUMENTS
    )
    queries = _decide(
        native_512=derive_mrl_prefix(_unit_rows(18, seed=77), operation="test"),
        native_1024=full,
        item_ids=ids,
        kind=ShardKind.QUERIES,
    )

    assert documents.derived512_allowed is True
    assert queries.derived512_allowed is False
    assert documents.kind is ShardKind.DOCUMENTS
    assert queries.kind is ShardKind.QUERIES


def test_a_decision_that_disagrees_with_its_own_evidence_is_refused() -> None:
    native_512, native_1024 = _exact_pair(6, seed=88)
    decision = _decide(native_512=native_512, native_1024=native_1024, item_ids=_ids(6))

    with pytest.raises(BenchmarkContractError):
        replace(decision, minimum_cosine=0.5)
    with pytest.raises(BenchmarkContractError):
        replace(decision, identical_top_k=False)


def _ids(count: int) -> tuple[str, ...]:
    return tuple(f"d{index:03d}" for index in range(count))


def test_a_calibration_that_decides_only_some_pairs_is_refused() -> None:
    native_512, native_1024 = _exact_pair(6, seed=101)
    decision = _decide(native_512=native_512, native_1024=native_1024, item_ids=_ids(6))
    items: list[str] = []
    with pytest.raises(BenchmarkContractError) as caught:
        build_mrl_calibration_payload(
            decisions=[decision], calibration_items=items, operation="test"
        )
    assert "does not decide every model/path pair" in str(caught.value)
    with pytest.raises(BenchmarkContractError):
        build_mrl_calibration_payload(decisions=[], calibration_items=items, operation="test")


def test_a_complete_calibration_payload_states_the_inputs_and_every_decision() -> None:
    native_512, native_1024 = _exact_pair(18, seed=111)
    decisions = [
        evaluate_mrl_equivalence(
            candidate=candidate,
            kind=kind,
            item_ids=_ids(18),
            native_512=native_512,
            native_1024=native_1024,
            operation="test",
        )
        for candidate in RES138_MODEL_CANDIDATES
        for kind in ShardKind
    ]

    payload = build_mrl_calibration_payload(
        decisions=decisions,
        calibration_items=[{"workload": "scifact", "item_id": "d000"}],
        operation="test",
    )

    assert payload["artifact_revision"] == "res138-mrl-calibration-v1"
    assert payload["derivation_revision"] == RES138_MRL_DERIVATION_REVISION
    assert payload["derived512_allowed_everywhere"] is True
    recorded = cast("list[dict[str, str]]", payload["decisions"])
    assert len(recorded) == 4
    assert {entry["model_id"] for entry in recorded} == {
        candidate.model_id for candidate in RES138_MODEL_CANDIDATES
    }


# ---------------------------------------------------------------------------
# The calibration sample
# ---------------------------------------------------------------------------


def _workload(name: str, *, documents: int = 30, queries: int = 30) -> RetrievalWorkload:
    return RetrievalWorkload(
        name=name,
        documents=tuple(
            RetrievalDocument.from_beir(
                document_id=f"{name}-d{index:04d}",
                title=f"title {index}",
                body="x" * (10 * index + 5),
            )
            for index in range(documents)
        ),
        queries=tuple(
            RetrievalQuery.from_beir(query_id=f"{name}-q{index:04d}", text="y" * (index + 1))
            for index in range(queries)
        ),
        qrels=(RetrievalQrel(query_id=f"{name}-q0000", document_id=f"{name}-d0000", relevance=1),),
    )


def test_the_calibration_set_is_deterministic_and_spans_every_workload_and_band() -> None:
    workloads = [_workload("scifact"), _workload("nfcorpus"), _workload("trec-covid")]

    first = select_calibration_set(workloads)
    second = select_calibration_set(list(reversed(workloads)))

    assert first.items == second.items
    assert (
        len(first.items)
        == 3 * 2 * len(RES138_CALIBRATION_BANDS) * RES138_CALIBRATION_ITEMS_PER_CELL
    )
    assert {item.workload for item in first.items} == {"scifact", "nfcorpus", "trec-covid"}
    for workload in ("scifact", "nfcorpus", "trec-covid"):
        for kind in ("documents", "queries"):
            for band in RES138_CALIBRATION_BANDS:
                assert (
                    sum(
                        1
                        for item in first.items
                        if (item.workload, item.kind, item.band) == (workload, kind, band)
                    )
                    == RES138_CALIBRATION_ITEMS_PER_CELL
                )


def test_the_calibration_bands_are_within_workload_thirds_by_length() -> None:
    selected = select_calibration_set([_workload("scifact")])
    lengths = {
        (item.kind, item.band): item.length for item in selected.items if item.workload == "scifact"
    }
    for kind in ("documents", "queries"):
        assert lengths[(kind, "short")] < lengths[(kind, "typical")] < lengths[(kind, "long")]


def test_a_workload_too_small_to_fill_three_bands_is_refused() -> None:
    with pytest.raises(BenchmarkContractError) as caught:
        select_calibration_set([_workload("tiny", documents=2, queries=30)])
    assert "cannot be split into the three calibration bands" in str(caught.value)


def test_the_calibration_payload_carries_ids_and_digests_but_never_text() -> None:
    selected = select_calibration_set([_workload("scifact")])

    payload = selected.payload()
    rendered = json.dumps(payload, sort_keys=True)

    assert payload["selection_revision"] == "res138-calibration-selection-v1"
    assert payload["item_count"] == len(selected.items)
    for item in cast("list[dict[str, str]]", payload["items"]):
        assert set(item) == {
            "workload",
            "kind",
            "band",
            "item_id",
            "content_sha256",
            "length",
        }
        assert item["content_sha256"] == next(
            entry.content_sha256 for entry in selected.items if entry.item_id == item["item_id"]
        )
    # The corpus text itself ("body 12") never appears in the artifact.
    assert "body 12" not in rendered


def test_the_calibration_ids_are_paired_with_their_texts() -> None:
    selected = select_calibration_set([_workload("scifact")])
    ids = selected.ids(workload="scifact", kind="documents")
    texts = selected.texts(workload="scifact", kind="documents")
    assert len(ids) == len(texts) == 6
    for item_id, text in zip(ids, texts, strict=True):
        assert text_sha256(text) == next(
            item.content_sha256 for item in selected.items if item.item_id == item_id
        )


def test_a_calibration_item_never_republishes_its_text() -> None:
    item = CalibrationItem(
        workload="w",
        kind="queries",
        band="short",
        item_id="q1",
        content_sha256="0" * 64,
        length=5,
        text="secret query text",
    )
    assert "text" not in item.payload()
    assert "secret" not in json.dumps(item.payload())


# ---------------------------------------------------------------------------
# BEIR acquisition and loading
# ---------------------------------------------------------------------------


def _archive_bytes(
    *, workload: str = "unit", documents: int = 4, queries: int = 3, qrels: str | None = None
) -> bytes:
    """A BEIR-shaped archive whose digest the caller can compute first."""
    import io

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        corpus = "".join(
            json.dumps({"_id": f"d{index}", "title": f"Title {index}", "text": f"body {index}"})
            + "\n"
            for index in range(documents)
        )
        archive.writestr(f"{workload}/corpus.jsonl", corpus)
        archive.writestr(
            f"{workload}/queries.jsonl",
            "".join(
                json.dumps({"_id": f"q{index}", "text": f"question {index}"}) + "\n"
                for index in range(queries)
            ),
        )
        archive.writestr(
            f"{workload}/qrels/test.tsv",
            qrels
            if qrels is not None
            else (
                "query-id\tcorpus-id\tscore\n"
                + "".join(f"q{index}\td{index}\t1\n" for index in range(min(documents, queries)))
            ),
        )
    return buffer.getvalue()


def _spec(payload: bytes, *, workload: str = "unit") -> BeirSourceSpec:
    return BeirSourceSpec(
        workload=workload,
        archive_name=f"{workload}.zip",
        url=f"https://example.invalid/{workload}.zip",
        sha256=hashlib.sha256(payload).hexdigest(),
        byte_size=len(payload),
    )


def _seeded(tmp_path: Path, payload: bytes, spec: BeirSourceSpec) -> tuple[VerifiedSource, ...]:
    """Put a verified archive in the cache and fetch it the way a later session would.

    Every acquisition test starts from a cached archive rather than a download, so
    no test in this suite opens a socket: the network path is exercised by the
    refusal tests, not by the happy ones.
    """
    cache = tmp_path / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / spec.archive_name).write_bytes(payload)
    return verify_and_cache_beir_sources(
        scratch_dir=tmp_path / "scratch", cache_dir=cache, specs=(spec,)
    )


def test_a_verified_archive_is_cached_and_then_reused_after_re_verification(
    tmp_path: Path,
) -> None:
    payload = _archive_bytes()
    spec = _spec(payload)
    scratch = tmp_path / "scratch"
    cache = tmp_path / "cache"
    scratch.mkdir()
    cache.mkdir()
    # Seed the cache only, exactly as a previous session would have left it.
    (cache / spec.archive_name).write_bytes(payload)

    verified = _seeded(tmp_path, payload, spec)

    assert verified[0].sha256 == spec.sha256
    assert (scratch / spec.archive_name).read_bytes() == payload


def test_a_corrupted_cached_archive_is_refused_and_never_copied_into_scratch(
    tmp_path: Path,
) -> None:
    payload = _archive_bytes()
    spec = _spec(payload)
    scratch = tmp_path / "scratch"
    cache = tmp_path / "cache"
    scratch.mkdir()
    cache.mkdir()
    (cache / spec.archive_name).write_bytes(payload[:-4] + b"XXXX")

    with pytest.raises(BenchmarkSourceError) as caught:
        verify_and_cache_beir_sources(scratch_dir=scratch, cache_dir=cache, specs=(spec,))

    assert caught.value.expected == spec.sha256
    assert (cache / spec.archive_name).exists()


def test_an_archive_with_the_wrong_layout_is_refused_before_extraction(tmp_path: Path) -> None:
    import io

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("unit/README.txt", "not a BEIR distribution")
    payload = buffer.getvalue()
    spec = _spec(payload)

    verified = VerifiedSource(spec=spec, path=tmp_path / spec.archive_name, sha256=spec.sha256)
    (tmp_path / spec.archive_name).write_bytes(payload)

    with pytest.raises(BenchmarkSourceError) as caught:
        extract_verified_archive(verified, tmp_path / "extracted")
    assert "expected BEIR layout" in str(caught.value)
    assert not (tmp_path / "extracted" / "unit").exists()


def test_a_verified_archive_loads_into_a_canonically_ordered_workload(tmp_path: Path) -> None:
    payload = _archive_bytes(documents=4, queries=3)
    spec = _spec(payload)
    verified = _seeded(tmp_path, payload, spec)
    root = extract_verified_archive(verified[0], tmp_path / "extracted")

    workload, report = load_beir_workload(root, spec)

    assert workload.name == "unit"
    assert workload.document_ids == ("d0", "d1", "d2", "d3")
    assert workload.query_ids == ("q0", "q1", "q2")
    assert [qrel.payload() for qrel in workload.qrels] == [
        {"query_id": "q0", "document_id": "d0", "relevance": 1},
        {"query_id": "q1", "document_id": "d1", "relevance": 1},
        {"query_id": "q2", "document_id": "d2", "relevance": 1},
    ]
    assert report.workload_summary["document_count"] == 4
    assert report.qrel_rows == 3
    assert report.max_relevance == 1


def test_file_order_does_not_change_the_loaded_order(tmp_path: Path) -> None:
    """A corpus written in a different order must load into the same canonical order."""
    import io

    shuffled = io.BytesIO()
    with zipfile.ZipFile(shuffled, "w") as archive:
        archive.writestr(
            "unit/corpus.jsonl",
            "".join(
                json.dumps({"_id": name, "title": "t", "text": name}) + "\n"
                for name in ("d3", "d0", "d2", "d1")
            ),
        )
        archive.writestr(
            "unit/queries.jsonl",
            "".join(json.dumps({"_id": name, "text": name}) + "\n" for name in ("q2", "q0", "q1")),
        )
        archive.writestr("unit/qrels/test.tsv", "query-id\tcorpus-id\tscore\nq0\td0\t1\n")
    payload = shuffled.getvalue()
    spec = _spec(payload)
    verified = _seeded(tmp_path, payload, spec)
    root = extract_verified_archive(verified[0], tmp_path / "extracted")

    workload, _ = load_beir_workload(root, spec)

    assert workload.document_ids == ("d0", "d1", "d2", "d3")
    # Only q0 carries a judgment, so only q0 is part of the workload: the other two
    # query records are reported as unjudged rather than embedded and never scored.
    assert workload.query_ids == ("q0",)


def test_only_judged_queries_become_part_of_the_workload_and_the_rest_are_counted(
    tmp_path: Path,
) -> None:
    payload = _archive_bytes(
        documents=3, queries=6, qrels="query-id\tcorpus-id\tscore\nq0\td0\t1\n"
    )
    spec = _spec(payload)
    verified = _seeded(tmp_path, payload, spec)
    root = extract_verified_archive(verified[0], tmp_path / "extracted")

    workload, report = load_beir_workload(root, spec)

    assert workload.query_ids == ("q0",)
    assert report.queries_in_archive == 6
    assert report.queries_without_judgement == 5


def test_a_document_with_no_embeddable_text_is_excluded_and_recorded(tmp_path: Path) -> None:
    import io

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "unit/corpus.jsonl",
            json.dumps({"_id": "d0", "title": "kept", "text": "body"})
            + "\n"
            + json.dumps({"_id": "d1", "title": "", "text": "   "})
            + "\n"
            # An empty body with a title is still embeddable: it becomes the title.
            + json.dumps({"_id": "d2", "title": "title only", "text": ""})
            + "\n",
        )
        archive.writestr("unit/queries.jsonl", json.dumps({"_id": "q0", "text": "q"}) + "\n")
        archive.writestr("unit/qrels/test.tsv", "query-id\tcorpus-id\tscore\nq0\td0\t1\n")
    spec = _spec(buffer.getvalue())
    verified = _seeded(tmp_path, buffer.getvalue(), spec)
    root = extract_verified_archive(verified[0], tmp_path / "extracted")

    workload, report = load_beir_workload(root, spec)

    assert workload.document_ids == ("d0", "d2")
    assert report.documents_without_embedding_text == 1
    assert report.excluded_document_ids_sha256 is not None
    # A title with an empty body is embedded as "title\n" -- the policy joins, it does
    # not substitute -- and is therefore kept rather than dropped.
    assert workload.documents[1].text == "title only\n"


@pytest.mark.parametrize(
    "qrels",
    [
        pytest.param("q\td\ts\nq0\td0\t1\n", id="wrong-header"),
        pytest.param("query-id\tcorpus-id\tscore\nq0\td0\t1.5\n", id="non-integer-score"),
        pytest.param("query-id\tcorpus-id\tscore\nq0\td0\n", id="wrong-column-count"),
    ],
)
def test_a_qrels_file_that_is_not_beir_shaped_is_refused(tmp_path: Path, qrels: str) -> None:
    payload = _archive_bytes(qrels=qrels)
    spec = _spec(payload)
    verified = _seeded(tmp_path, payload, spec)
    root = extract_verified_archive(verified[0], tmp_path / "extracted")

    with pytest.raises(BenchmarkSourceError):
        load_beir_workload(root, spec)


def test_a_negative_judgment_is_carried_through_rather_than_refused(tmp_path: Path) -> None:
    """TREC-COVID's test.tsv carries two rows at -1; refusing them would refuse the corpus."""
    payload = _archive_bytes(
        documents=2, queries=1, qrels="query-id\tcorpus-id\tscore\nq0\td0\t-1\nq0\td1\t2\n"
    )
    spec = _spec(payload)
    verified = _seeded(tmp_path, payload, spec)
    root = extract_verified_archive(verified[0], tmp_path / "extracted")

    workload, report = load_beir_workload(root, spec)

    assert [qrel.relevance for qrel in workload.qrels] == [-1, 2]
    assert report.min_relevance == -1
    assert report.max_relevance == 2


def test_a_record_whose_fields_are_not_text_is_refused(tmp_path: Path) -> None:
    import io

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("unit/corpus.jsonl", json.dumps({"_id": 1, "text": "body"}) + "\n")
        archive.writestr("unit/queries.jsonl", json.dumps({"_id": "q0", "text": "q"}) + "\n")
        archive.writestr("unit/qrels/test.tsv", "query-id\tcorpus-id\tscore\nq0\td1\t1\n")
    spec = _spec(buffer.getvalue())
    verified = _seeded(tmp_path, buffer.getvalue(), spec)
    root = extract_verified_archive(verified[0], tmp_path / "extracted")

    with pytest.raises(BenchmarkSourceError):
        load_beir_workload(root, spec)


def test_a_corpus_line_that_is_not_json_is_refused(tmp_path: Path) -> None:
    import io

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("unit/corpus.jsonl", "{not json}\n")
        archive.writestr("unit/queries.jsonl", json.dumps({"_id": "q0", "text": "q"}) + "\n")
        archive.writestr("unit/qrels/test.tsv", "query-id\tcorpus-id\tscore\nq0\td1\t1\n")
    spec = _spec(buffer.getvalue())
    verified = _seeded(tmp_path, buffer.getvalue(), spec)
    root = extract_verified_archive(verified[0], tmp_path / "extracted")

    with pytest.raises(BenchmarkSourceError):
        load_beir_workload(root, spec)


def test_the_three_frozen_specs_are_the_ones_the_loader_would_fetch() -> None:
    assert [spec.workload for spec in RES138_BEIR_SOURCES] == ["scifact", "nfcorpus", "trec-covid"]
    for spec in RES138_BEIR_SOURCES:
        assert spec.archive_name == f"{spec.workload}.zip"
        assert spec.url.endswith(f"/{spec.workload}.zip")
