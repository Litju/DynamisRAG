"""The frozen RES-138 benchmark contracts.

Pure values: three source archives, two model candidates, a handful of integers
and the workload types. No network, no model, no GPU. What is asserted here is
the freeze itself, because a freeze that is only a comment is not a freeze:

* the three BEIR sources are exactly SciFact, NFCorpus and TREC-COVID, each an
  ``https`` URL with a 64-hex digest and a positive size, in a fixed order;
* the two candidates are exactly Voyage 4 Nano and Qwen3-Embedding-0.6B at
  immutable 40-hex revisions, with the pooling mode, truncation boundary and
  prompt contents transcribed from those revisions — including the trailing
  space on Voyage's two prompts, the missing space after Qwen's ``Query:`` and
  Qwen's *empty* document prompt, which is load-bearing rather than missing;
* the prompt identities hash into one ``prompt_sha256``, because a model whose
  query prompt changed is a different model for retrieval purposes;
* a mutable identity (``main``, ``latest``, a tag, an abbreviated SHA) is refused
  wherever an identity is stated, not only where it is finally recorded;
* the document text policy is one function, so a document's embedded string
  cannot be assembled two ways;
* content digests are *verified* against the text on construction, so an
  artifact cannot attest to text it never embedded;
* canonical order is enforced, not produced, and duplicates are refused — a
  repeated key makes a corpus disagree with itself;
* every dangling qrel reference, every invalid relevance type and every
  unsorted workload is refused at construction;
* a query or document with no embeddable text is refused, because RES-137
  refuses empty passage text and a meaningless vector would occupy a corpus slot
  and perturb Recall@100;
* the frozen numerics — dimensions, shard size, top-k, chunk size, bootstrap,
  calibration bands and gates, artifact revisions — are exactly the declared
  values, because each of them could otherwise be chosen after seeing results.
"""

from __future__ import annotations

import hashlib
from typing import Final

import pytest

from dynamisrag.benchmark import contracts
from dynamisrag.benchmark.contracts import (
    BEIR_QREL_SPLIT,
    RES138_ARTIFACT_REVISIONS,
    RES138_BASE_DIMENSION,
    RES138_BEIR_SOURCES,
    RES138_BOOTSTRAP_CONFIDENCE,
    RES138_BOOTSTRAP_SAMPLES,
    RES138_BOOTSTRAP_SEED,
    RES138_CALIBRATION_BANDS,
    RES138_CALIBRATION_ITEMS_PER_CELL,
    RES138_CALIBRATION_TOP_K,
    RES138_CANDIDATE_DIMENSIONS,
    RES138_CORPUS_CHUNK_SIZE,
    RES138_DOCUMENT_TEXT_POLICY,
    RES138_DRIVE_LOCATIONS,
    RES138_DRIVE_ROOT,
    RES138_LOCAL_SCRATCH_ROOT,
    RES138_MODEL_CANDIDATES,
    RES138_MRL_CALIBRATION_GATE,
    RES138_MRL_DERIVATION_REVISION,
    RES138_NDCG_CUTOFF,
    RES138_QUERY_SELECTION_POLICY,
    RES138_RECALL_CUTOFFS,
    RES138_RETRIEVAL_TOP_K,
    RES138_RUN_ID_PREFIX,
    RES138_SHARD_SIZE,
    RES138_TEI_EQUIVALENCE_GATE,
    RES138_WORKLOAD_NAMES,
    BeirSourceSpec,
    ModelCandidateSpec,
    RetrievalDocument,
    RetrievalPromptSpec,
    RetrievalQrel,
    RetrievalQuery,
    RetrievalWorkload,
    beir_document_embedding_text,
    ordered_ids_sha256,
    require_candidate_dimension,
    require_code_sha,
    require_shard_size,
    text_sha256,
)
from dynamisrag.benchmark.errors import BenchmarkContractError

_VOYAGE_ID: Final[str] = "voyageai/voyage-4-nano"
_VOYAGE_REVISION: Final[str] = "67fabc9bef010dabc5f6024aa1b1b6b93410426f"
_QWEN_ID: Final[str] = "Qwen/Qwen3-Embedding-0.6B"
_QWEN_REVISION: Final[str] = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"


def _document(
    document_id: str, *, title: str = "Title", text: str | None = None
) -> RetrievalDocument:
    return RetrievalDocument.from_beir(
        document_id=document_id,
        title=title,
        body=text if text is not None else f"body of {document_id}",
    )


def _query(query_id: str, *, text: str | None = None) -> RetrievalQuery:
    return RetrievalQuery.from_beir(
        query_id=query_id, text=text if text is not None else f"q {query_id}"
    )


def _workload(**overrides: object) -> RetrievalWorkload:
    defaults: dict[str, object] = {
        "name": "unit",
        "documents": (_document("a"), _document("b")),
        "queries": (_query("q1"), _query("q2")),
        "qrels": (
            RetrievalQrel(query_id="q1", document_id="a", relevance=1),
            RetrievalQrel(query_id="q2", document_id="b", relevance=2),
        ),
    }
    defaults.update(overrides)
    return RetrievalWorkload(**defaults)  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------------------
# Frozen sources
# ---------------------------------------------------------------------------


def test_exactly_three_workloads_are_frozen_in_a_fixed_order() -> None:
    assert RES138_WORKLOAD_NAMES == ("scifact", "nfcorpus", "trec-covid")
    assert [source.workload for source in RES138_BEIR_SOURCES] == list(RES138_WORKLOAD_NAMES)


@pytest.mark.parametrize(
    ("workload", "digest"),
    [
        ("scifact", "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165"),
        ("nfcorpus", "efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b"),
        ("trec-covid", "120f42a7864d2214234537733c0d2c6684e42fdfafff2c5eacf98afca6656aa0"),
    ],
)
def test_every_frozen_source_digest_is_the_digest_computed_from_the_archive(
    workload: str, digest: str
) -> None:
    """The digests are literals from the feasibility pass, asserted as such."""
    source = next(item for item in RES138_BEIR_SOURCES if item.workload == workload)
    assert source.sha256 == digest
    assert (
        source.url
        == f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{workload}.zip"
    )
    assert source.archive_name == f"{workload}.zip"
    assert source.byte_size > 0


def test_the_frozen_qrels_split_is_the_test_split() -> None:
    assert BEIR_QREL_SPLIT == "test"


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(
            lambda: BeirSourceSpec(
                workload="x",
                archive_name="x.zip",
                url="http://host/x.zip",
                sha256="a" * 64,
                byte_size=1,
            ),
            id="plaintext-transport",
        ),
        pytest.param(
            lambda: BeirSourceSpec(
                workload="x",
                archive_name="x.zip",
                url="https://host/x.zip",
                sha256="not-a-digest",
                byte_size=1,
            ),
            id="digest-is-not-a-digest",
        ),
        pytest.param(
            lambda: BeirSourceSpec(
                workload="",
                archive_name="x.zip",
                url="https://host/x.zip",
                sha256="a" * 64,
                byte_size=1,
            ),
            id="unnamed-workload",
        ),
        pytest.param(
            lambda: BeirSourceSpec(
                workload="x",
                archive_name="x.zip",
                url="https://host/x.zip",
                sha256="a" * 64,
                byte_size=0,
            ),
            id="empty-archive",
        ),
    ],
)
def test_an_unverifiable_source_spec_is_refused(mutation: object) -> None:
    with pytest.raises(BenchmarkContractError):
        mutation()  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Frozen candidates and prompts
# ---------------------------------------------------------------------------


def test_exactly_two_candidates_are_frozen_at_immutable_revisions() -> None:
    assert [candidate.model_id for candidate in RES138_MODEL_CANDIDATES] == [_VOYAGE_ID, _QWEN_ID]
    assert [candidate.revision for candidate in RES138_MODEL_CANDIDATES] == [
        _VOYAGE_REVISION,
        _QWEN_REVISION,
    ]
    assert all(len(candidate.revision) == 40 for candidate in RES138_MODEL_CANDIDATES)
    assert all(candidate.license == "Apache-2.0" for candidate in RES138_MODEL_CANDIDATES)


def test_the_frozen_prompts_are_the_pinned_repository_prompts_byte_for_byte() -> None:
    """Transcribed from ``config_sentence_transformers.json`` at the pinned revisions.

    The trailing space on Voyage's two prompts and the *absence* of one after
    Qwen's ``Query:`` are part of the frozen identity. So is Qwen's empty
    document prompt: it means "no instruction, because a document is not a
    question", and treating it as missing would substitute a different policy.
    """
    voyage = RES138_MODEL_CANDIDATES[0]
    qwen = RES138_MODEL_CANDIDATES[1]
    assert voyage.query_prompt.content == (
        "Represent the query for retrieving supporting documents: "
    )
    assert voyage.document_prompt.content == "Represent the document for retrieval: "
    assert qwen.query_prompt.content == (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query"
        "\nQuery:"
    )
    assert qwen.document_prompt.content == ""


def test_every_prompt_carries_the_digest_of_its_own_content() -> None:
    for candidate in RES138_MODEL_CANDIDATES:
        for prompt in (candidate.query_prompt, candidate.document_prompt):
            assert prompt.content_sha256 == text_sha256(prompt.content)
            assert prompt.content_sha256 == hashlib.sha256(prompt.content.encode()).hexdigest()


def test_the_prompt_pair_hashes_into_one_model_prompt_identity() -> None:
    candidate = RES138_MODEL_CANDIDATES[0]
    assert candidate.prompt_sha256 != RES138_MODEL_CANDIDATES[1].prompt_sha256
    assert candidate.prompt(kind="document").name == "document"
    assert candidate.prompt(kind="query") is candidate.query_prompt
    with pytest.raises(BenchmarkContractError):
        candidate.prompt(kind="passage")


def test_pooling_and_truncation_boundaries_are_declared_and_sourced() -> None:
    voyage, qwen = RES138_MODEL_CANDIDATES
    assert (voyage.pooling_mode, qwen.pooling_mode) == ("mean", "last_token")
    assert voyage.native_max_sequence_length == 32768
    assert qwen.native_max_sequence_length == 32768
    assert "sentence_bert_config.json" in voyage.sequence_length_source
    assert "max_position_embeddings" in qwen.sequence_length_source
    assert "131072" in qwen.sequence_length_source


@pytest.mark.parametrize("revision", ["main", "latest", "v1.0", "HEAD", "67fabc9"])
def test_a_mutable_or_abbreviated_candidate_revision_is_refused(revision: str) -> None:
    with pytest.raises(BenchmarkContractError):
        ModelCandidateSpec(
            model_id=_VOYAGE_ID,
            revision=revision,
            license="Apache-2.0",
            pooling_mode="mean",
            native_max_sequence_length=32768,
            sequence_length_source="test",
            query_prompt=RetrievalPromptSpec("query", "q", text_sha256("q")),
            document_prompt=RetrievalPromptSpec("document", "", text_sha256("")),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param({"pooling_mode": "cls"}, id="unsupported-pooling"),
        pytest.param({"native_max_sequence_length": 0}, id="non-positive-boundary"),
        pytest.param({"native_max_sequence_length": True}, id="boolean-boundary"),
        pytest.param({"sequence_length_source": ""}, id="unsourced-boundary"),
    ],
)
def test_a_candidate_that_cannot_state_itself_is_refused(mutation: dict[str, object]) -> None:
    fields: dict[str, object] = {
        "model_id": _QWEN_ID,
        "revision": _QWEN_REVISION,
        "license": "Apache-2.0",
        "pooling_mode": "last_token",
        "native_max_sequence_length": 32768,
        "sequence_length_source": "test",
        "query_prompt": RetrievalPromptSpec("query", "q", text_sha256("q")),
        "document_prompt": RetrievalPromptSpec("document", "", text_sha256("")),
    }
    fields.update(mutation)
    with pytest.raises(BenchmarkContractError):
        ModelCandidateSpec(**fields)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("name", ["passage", "QUERY", ""])
def test_a_prompt_outside_the_two_frozen_names_is_refused(name: str) -> None:
    with pytest.raises(BenchmarkContractError):
        RetrievalPromptSpec(name=name, content="x", content_sha256=text_sha256("x"))


# ---------------------------------------------------------------------------
# Frozen numerics
# ---------------------------------------------------------------------------


def test_the_frozen_numerics_are_exactly_the_declared_values() -> None:
    assert RES138_CANDIDATE_DIMENSIONS == (512, 1024)
    assert RES138_BASE_DIMENSION == 1024
    assert RES138_MRL_DERIVATION_REVISION == "mrl-prefix-renorm-v1"
    assert RES138_SHARD_SIZE == 4096
    assert RES138_RETRIEVAL_TOP_K == 100
    assert RES138_NDCG_CUTOFF == 10
    assert RES138_RECALL_CUTOFFS == (10, 100)
    assert RES138_CORPUS_CHUNK_SIZE == 8192
    assert RES138_RUN_ID_PREFIX == "colab"
    assert RES138_BOOTSTRAP_SEED == 138
    assert RES138_BOOTSTRAP_SAMPLES == 10_000
    assert RES138_BOOTSTRAP_CONFIDENCE == 0.95


def test_the_mrl_gate_is_frozen_before_any_calibration_is_seen() -> None:
    assert RES138_MRL_CALIBRATION_GATE.minimum_cosine == 0.999999
    assert RES138_MRL_CALIBRATION_GATE.maximum_absolute_difference == 1e-5
    assert RES138_MRL_CALIBRATION_GATE.require_identical_top_k is True


def test_the_tei_equivalence_gate_is_separate_and_looser_than_the_mrl_gate() -> None:
    assert RES138_TEI_EQUIVALENCE_GATE.minimum_cosine == 0.99999
    assert RES138_TEI_EQUIVALENCE_GATE.maximum_absolute_difference == 1e-4
    assert RES138_TEI_EQUIVALENCE_GATE.minimum_cosine < RES138_MRL_CALIBRATION_GATE.minimum_cosine


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param({"minimum_cosine": 1.5}, id="cosine-above-one"),
        pytest.param({"minimum_cosine": float("nan")}, id="nan-cosine"),
        pytest.param({"maximum_absolute_difference": -1e-5}, id="negative-difference"),
    ],
)
def test_a_gate_that_cannot_discriminate_is_refused(mutation: dict[str, float]) -> None:
    fields: dict[str, float | bool] = {
        "minimum_cosine": 0.999999,
        "maximum_absolute_difference": 1e-5,
        "require_identical_top_k": True,
    }
    fields.update(mutation)
    with pytest.raises(BenchmarkContractError):
        contracts.MrlCalibrationGate(**fields)  # pyright: ignore[reportArgumentType]


def test_the_calibration_set_shape_is_frozen() -> None:
    assert RES138_CALIBRATION_BANDS == ("short", "typical", "long")
    assert RES138_CALIBRATION_ITEMS_PER_CELL == 2
    assert RES138_CALIBRATION_TOP_K == 10


def test_exactly_nine_artifact_revisions_are_declared() -> None:
    assert RES138_ARTIFACT_REVISIONS == {
        "plan": "res138-plan-v1",
        "runtime": "res138-runtime-v1",
        "source_manifest": "res138-source-manifest-v1",
        "model_manifest": "res138-model-manifest-v1",
        "shard": "res138-shard-v1",
        "mrl_calibration": "res138-mrl-calibration-v1",
        "preflight": "res138-preflight-v1",
        "results": "res138-results-v1",
        "selection": "res138-selection-v1",
    }


# ---------------------------------------------------------------------------
# The Drive contract
# ---------------------------------------------------------------------------


def test_the_drive_contract_names_paths_and_documents_folder_ids_separately() -> None:
    assert RES138_DRIVE_ROOT == "/content/drive/MyDrive/DynamisRAG/RES-138"
    assert RES138_LOCAL_SCRATCH_ROOT == "/content/res138"
    assert contracts.drive_path("runs", "colab-x") == f"{RES138_DRIVE_ROOT}/runs/colab-x"
    paths = {location.label: location.path for location in RES138_DRIVE_LOCATIONS}
    assert paths["notebooks"] == f"{RES138_DRIVE_ROOT}/notebooks"
    assert paths["sources/beir"] == f"{RES138_DRIVE_ROOT}/sources/beir"
    assert paths["runs"] == f"{RES138_DRIVE_ROOT}/runs"
    # Ids are provenance for an operator opening Drive, never an input.
    assert all(location.folder_id is not None for location in RES138_DRIVE_LOCATIONS)


# ---------------------------------------------------------------------------
# Code identity
# ---------------------------------------------------------------------------


def test_an_exact_commit_sha_is_accepted() -> None:
    assert require_code_sha("a" * 40, operation="test") == "a" * 40


@pytest.mark.parametrize(
    "value",
    ["", "main", "latest", "julitocrztuga/res-138", "67fabc9", "A" * 40, "a" * 41, "main\n"],
)
def test_anything_but_an_exact_commit_is_refused(value: str) -> None:
    with pytest.raises(BenchmarkContractError) as caught:
        require_code_sha(value, operation="test")
    assert "40 lowercase hexadecimal" in str(caught.value)


def test_only_the_frozen_dimensions_and_shard_size_are_accepted() -> None:
    assert require_candidate_dimension(512, operation="test") == 512
    assert require_shard_size(4096, operation="test") == 4096
    for dimension in (256, 768, 2048, 0):
        with pytest.raises(BenchmarkContractError):
            require_candidate_dimension(dimension, operation="test")
    with pytest.raises(BenchmarkContractError):
        require_shard_size(8192, operation="test")


# ---------------------------------------------------------------------------
# The document text policy
# ---------------------------------------------------------------------------


def test_the_document_text_policy_joins_a_present_title_with_a_newline() -> None:
    assert beir_document_embedding_text("A title", "body") == "A title\nbody"
    assert beir_document_embedding_text("  padded  ", "body") == "padded\nbody"


def test_the_document_text_policy_uses_the_body_alone_when_there_is_no_title() -> None:
    assert beir_document_embedding_text("", "body") == "body"
    assert beir_document_embedding_text("   ", "body") == "body"


def test_the_document_text_policy_does_not_normalise_the_body() -> None:
    """No whitespace folding, no case folding: any of it would change the tokens."""
    raw = "line one\n\n  line two  "
    assert beir_document_embedding_text("t", raw) == f"t\n{raw}"
    assert beir_document_embedding_text("", raw) == raw


def test_a_document_digest_binds_the_embedded_string_not_the_raw_body() -> None:
    document = RetrievalDocument.from_beir(document_id="d1", title="T", body="B")
    assert document.title == "T"
    assert document.text == "T\nB"
    assert document.content_sha256 == text_sha256("T\nB")


def test_a_declaration_that_disagrees_with_its_text_is_refused() -> None:
    with pytest.raises(BenchmarkContractError) as caught:
        RetrievalDocument(document_id="d1", title="T", text="T\nB", content_sha256="0" * 64)
    assert caught.value.item_id == "d1"
    assert caught.value.expected == "0" * 64
    assert "never embedded" in str(caught.value)


def test_a_document_with_no_embeddable_text_is_refused() -> None:
    with pytest.raises(BenchmarkContractError) as caught:
        RetrievalDocument(document_id="d1", title="", text="   ", content_sha256=text_sha256("   "))
    assert "no embeddable text" in str(caught.value)


# ---------------------------------------------------------------------------
# Workload canonical order and integrity
# ---------------------------------------------------------------------------


def test_a_workload_exposes_its_ids_and_digests_in_canonical_order() -> None:
    workload = _workload()
    assert workload.document_ids == ("a", "b")
    assert workload.query_ids == ("q1", "q2")
    assert workload.summary() == {
        "name": "unit",
        "document_count": 2,
        "query_count": 2,
        "qrel_count": 2,
        "document_ids_sha256": ordered_ids_sha256(("a", "b")),
        "query_ids_sha256": ordered_ids_sha256(("q1", "q2")),
    }
    assert list(workload.qrels_by_query) == ["q1", "q2"]


def test_an_ordered_id_digest_binds_the_order_and_the_members() -> None:
    assert ordered_ids_sha256(["a", "b"]) == ordered_ids_sha256(["a", "b"])
    assert ordered_ids_sha256(["a", "b"]) != ordered_ids_sha256(["b", "a"])
    assert ordered_ids_sha256(["a", "b"]) != ordered_ids_sha256(["a"])


def test_a_query_digest_disagreement_is_refused() -> None:
    with pytest.raises(BenchmarkContractError) as caught:
        RetrievalQuery(query_id="q1", text="actual", content_sha256="1" * 64)
    assert caught.value.item_id == "q1"


def test_an_unnamed_workload_document_or_query_is_refused() -> None:
    with pytest.raises(BenchmarkContractError):
        RetrievalDocument.from_beir(document_id="", title="t", body="b")
    with pytest.raises(BenchmarkContractError):
        RetrievalQuery.from_beir(query_id="", text="q")
    with pytest.raises(BenchmarkContractError):
        _workload(name="")


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"documents": (_document("a"), _document("a"))}, id="duplicate-document-id"),
        pytest.param({"queries": (_query("q1"), _query("q1"))}, id="duplicate-query-id"),
        pytest.param(
            {
                "qrels": (
                    RetrievalQrel(query_id="q1", document_id="a", relevance=1),
                    RetrievalQrel(query_id="q1", document_id="a", relevance=1),
                )
            },
            id="duplicate-qrel",
        ),
        pytest.param(
            {"documents": (_document("b"), _document("a"))},
            id="documents-out-of-canonical-order",
        ),
        pytest.param({"queries": (_query("q2"), _query("q1"))}, id="queries-out-of-order"),
        pytest.param(
            {
                "qrels": (
                    RetrievalQrel(query_id="q2", document_id="b", relevance=1),
                    RetrievalQrel(query_id="q1", document_id="a", relevance=1),
                )
            },
            id="qrels-out-of-order",
        ),
        pytest.param(
            {"qrels": (RetrievalQrel(query_id="q9", document_id="a", relevance=1),)},
            id="qrel-names-an-absent-query",
        ),
        pytest.param(
            {"qrels": (RetrievalQrel(query_id="q1", document_id="zz", relevance=1),)},
            id="qrel-names-an-absent-document",
        ),
        pytest.param({"documents": ()}, id="empty-corpus"),
        pytest.param({"queries": ()}, id="empty-query-set"),
    ],
)
def test_a_workload_that_contradicts_itself_is_refused(overrides: dict[str, object]) -> None:
    with pytest.raises(BenchmarkContractError):
        _workload(**overrides)


@pytest.mark.parametrize("relevance", [1.0, "1", None, True, False])
def test_a_relevance_that_is_not_an_integer_judgment_is_refused(relevance: object) -> None:
    with pytest.raises(BenchmarkContractError) as caught:
        RetrievalQrel(query_id="q1", document_id="a", relevance=relevance)  # pyright: ignore[reportArgumentType]
    assert "integer judgment level" in str(caught.value)


@pytest.mark.parametrize("relevance", [0, 1, 2, -1])
def test_relevance_sign_carries_the_metric_semantics_not_the_contract(relevance: int) -> None:
    """TREC-COVID's ``test.tsv`` carries ``-1``; refusing it would refuse a real distribution."""
    qrel = RetrievalQrel(query_id="q1", document_id="a", relevance=relevance)
    assert qrel.is_relevant is (relevance > 0)
    assert qrel.payload() == {
        "query_id": "q1",
        "document_id": "a",
        "relevance": relevance,
    }


def test_a_qrel_naming_half_a_pair_is_refused() -> None:
    with pytest.raises(BenchmarkContractError):
        RetrievalQrel(query_id="", document_id="a", relevance=1)
    with pytest.raises(BenchmarkContractError):
        RetrievalQrel(query_id="q1", document_id="", relevance=1)


# ---------------------------------------------------------------------------
# The declared policies are named, versioned policies
# ---------------------------------------------------------------------------


def test_the_two_load_policies_carry_revisions() -> None:
    assert RES138_DOCUMENT_TEXT_POLICY == "beir-document-text-v1"
    assert RES138_QUERY_SELECTION_POLICY == "beir-judged-queries-v1"
