"""Local operator safeguards, without inference, downloads or live services."""

import time
from typing import cast
from unittest.mock import Mock, patch

import pytest
from scripts.real_scientific_retrieval import BudgetTei, CpuDeployment, verify_point
from sqlalchemy.orm import Session

from dynamisrag.embedding import (
    EmbeddingGenerationConfig,
    EmbeddingInput,
    EmbeddingRetryPolicy,
    EmbeddingRuntimeConfig,
    ExpectedTeiModel,
    TeiDeploymentSemantics,
    TruncationDirection,
)
from dynamisrag.embedding.contracts import passage_content_sha256
from dynamisrag.search.qdrant import Point


def test_cpu_deployment_changes_fingerprint_without_claiming_gpu_equivalence() -> None:
    cpu = CpuDeployment().payload()
    assert cpu != TeiDeploymentSemantics().payload()
    assert cpu["local_device"] == "cpu"
    digest = cpu["local_image_digest"]
    assert isinstance(digest, str) and digest.startswith("sha256:")
    assert cpu["tei_default_prompt_mode"] == "none"


def test_cpu_budget_stops_before_inference() -> None:
    provider = BudgetTei(
        base_url="http://127.0.0.1:1",
        expected_model=ExpectedTeiModel(
            "Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
        ),
        deployment_semantics=CpuDeployment(),
        generation_config=EmbeddingGenerationConfig(
            normalize=True,
            truncate=False,
            truncation_direction=TruncationDirection.RIGHT,
            dimensions=512,
        ),
        runtime_config=EmbeddingRuntimeConfig(
            batch_size=1, timeout_seconds=120, retry=EmbeddingRetryPolicy(max_attempts=1)
        ),
    )
    provider.deadline = time.monotonic() - 1
    try:
        with pytest.raises(TimeoutError, match="budget"):
            provider.embed(
                [
                    EmbeddingInput(
                        "a" * 64, passage_content_sha256("synthetic test"), "synthetic test"
                    )
                ]
            )
    finally:
        provider.close()


@pytest.mark.parametrize(
    "corruption", [None, "text", "span", "citation", "doi", "fragments", "gap"]
)
def test_independent_sql_verification_rejects_corruption(corruption: str | None) -> None:
    session = Mock(spec=Session)
    passage = Mock(
        id="p",
        text="canonical",
        content_sha256=passage_content_sha256("canonical"),
        document_version_id="v",
    )
    span = Mock(paragraph_id="para", source_order=0, start_char=0, end_char=9)
    paragraph = Mock(
        id="para",
        document_version_id="v",
        text="canonical",
        paragraph_key="pk",
        source_anchor="jats:#p",
    )
    version = Mock(document_id="d", source_artifact_id="a", version_key="vk", title="title")
    document = Mock(id="d", canonical_key="doi:10.1/test")
    artifact = Mock(source_external_id="PMC123")
    session.scalar.return_value = passage
    session.scalars.return_value = [span]
    session.get.side_effect = [paragraph, version, document, artifact]
    payload = {
        "passage_key": "passage",
        "text": "canonical",
        "content_sha256": passage_content_sha256("canonical"),
        "document_version_key": "vk",
        "document_canonical_key": "doi:10.1/test",
        "title": "title",
        "pmcid": "PMC123",
        "doi": "10.1/test",
        "pmid": None,
        "source_spans": [
            {
                "source_order": 0,
                "paragraph_key": "pk",
                "paragraph_source_anchor": "jats:#p",
                "start_char": 0,
                "end_char": 9,
            }
        ],
    }
    if corruption == "text":
        payload["text"] = "tampered"
    elif corruption == "span":
        span.end_char = 8
    elif corruption == "citation":
        payload["document_version_key"] = "wrong"
    elif corruption == "doi":
        payload["doi"] = "10.1/wrong"
    elif corruption in ("fragments", "gap"):
        span.end_char = 3
        second = Mock(
            paragraph_id="para",
            source_order=1,
            start_char=3 if corruption == "fragments" else 4,
            end_char=9,
        )
        session.scalars.return_value = [span, second]
        session.get.side_effect = [paragraph, paragraph, version, document, artifact]
        payload["source_spans"] = [
            {
                "source_order": 0,
                "paragraph_key": "pk",
                "paragraph_source_anchor": "jats:#p",
                "start_char": 0,
                "end_char": 3,
            },
            {
                "source_order": 1,
                "paragraph_key": "pk",
                "paragraph_source_anchor": "jats:#p",
                "start_char": second.start_char,
                "end_char": 9,
            },
        ]
    point = Point.model_validate({"id": "test", "payload": payload})
    identifiers = [
        Mock(namespace="doi", normalized_value="10.1/test"),
        Mock(namespace="pmcid", normalized_value="PMC123"),
    ]
    with patch(
        "scripts.real_scientific_retrieval.get_document_identifiers", return_value=identifiers
    ):
        if corruption in (None, "fragments"):
            assert (
                verify_point(cast(Session, session), point)["verified_against_postgresql"] is True
            )
        else:
            with pytest.raises(ValueError):
                verify_point(cast(Session, session), point)
