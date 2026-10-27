"""Verify retained real numerical evidence without loading an inference runtime."""

import hashlib
import json
from pathlib import Path

from dynamisrag.search.qdrant_artifact import load_embedding_manifest

EVIDENCE = Path(__file__).resolve().parents[2] / "docs/evidence/real-science-v1"


def test_original_receipt_binds_real_document_and_query_vectors() -> None:
    raw = (EVIDENCE / "receipt.json").read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "8ede85766f656acccf7fd44b9c3fde627bd18289b74fe375dbf895d02ff4d240"
    )
    receipt = json.loads(raw)
    docs = load_embedding_manifest(
        EVIDENCE / "embedding-manifest.json", expected_sha256=receipt["embedding_manifest_sha256"]
    )
    query = load_embedding_manifest(
        EVIDENCE / "query-manifest.json", expected_sha256=receipt["query_manifest_sha256"]
    )
    assert docs.document_count == 75 and query.document_count == 1
    assert docs.dimension == query.dimension == 512
    assert (
        docs.provider.model_sha
        == query.provider.model_sha
        == "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    )
    assert docs.provider.deployment.payload()["local_device"] == "cpu"
    assert docs.generation_config.prompt_name == "document"
    assert query.generation_config.prompt_name == "query"
    assert docs.embedding_config_sha256 != query.embedding_config_sha256
    assert not docs.generation_config.truncate and not query.generation_config.truncate
    result = json.loads((EVIDENCE / "retrieval.json").read_bytes())
    assert result["papers"] == 3
    assert result["passages"] == result["vectors"] == result["all_passages_verified"] == 75
    assert result["rrf"]["k"] == 60
    assert all(p["jats_reparse_matches_postgresql"] for p in result["source_checks"])
    for hit in result["rrf"]["hits"]:
        verified = result["independent_verification"][hit["passage_key"]]
        assert verified["verified_against_postgresql"]
        assert verified["pmcid"] == hit["provenance"]["pmcid"]
