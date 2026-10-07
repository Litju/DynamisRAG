"""The canonical Stage-B TEI server contract: identity, request body and endpoint locality.

Every load-bearing value here is a frozen contract the GPU operator must observe and
the local verifier must re-impose. The digest of a :class:`TeiServerInfo` is the
serving identity a GPU artifact binds, so these tests pin what changes it and what
does not, and that it is never a digest of the URL.
"""

from __future__ import annotations

import hashlib
from typing import Final

import pytest

from dynamisrag.benchmark.contracts import RES138_INPUT_MAX_TOKENS
from dynamisrag.benchmark.errors import BenchmarkContractError
from dynamisrag.benchmark.tei_server import (
    RES138_TEI_REQUEST_SEMANTICS,
    TEI_EMBED_REQUEST_FIELDS,
    TeiServerInfo,
    parse_tei_server_info,
    require_local_tei_endpoint,
    tei_embed_request_body,
)

_MODEL_ID: Final[str] = "Qwen/Qwen3-Embedding-0.6B"
_MODEL_SHA: Final[str] = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "version": "1.9.4",
        "model_id": _MODEL_ID,
        "model_sha": _MODEL_SHA,
        "model_dtype": "float16",
        "max_input_length": RES138_INPUT_MAX_TOKENS,
        "max_batch_tokens": RES138_INPUT_MAX_TOKENS,
        "auto_truncate": True,
        "max_client_batch_size": 32,
        "sha": "c" * 40,
        "docker_label": "ghcr.io/huggingface/text-embeddings-inference:1.9.4",
        "max_concurrent_requests": 512,
        "max_batch_requests": 4,
        "tokenization_workers": 8,
    }
    payload.update(overrides)
    return payload


def _parse(**overrides: object) -> TeiServerInfo:
    return parse_tei_server_info(
        _payload(**overrides),
        expected_model_id=_MODEL_ID,
        expected_model_revision=_MODEL_SHA,
        expected_precision="float16",
        min_max_client_batch_size=8,
        operation="test",
    )


def _record(**overrides: object) -> TeiServerInfo:
    """A well-formed record with arbitrary field values, for digest comparisons only."""
    fields: dict[str, object] = {
        "version": "1.9.4",
        "model_id": _MODEL_ID,
        "model_sha": _MODEL_SHA,
        "model_dtype": "float16",
        "max_input_length": RES138_INPUT_MAX_TOKENS,
        "max_batch_tokens": RES138_INPUT_MAX_TOKENS,
        "auto_truncate": True,
        "max_client_batch_size": 32,
        "sha": "c" * 40,
        "docker_label": None,
        "max_concurrent_requests": 512,
        "max_batch_requests": 4,
        "tokenization_workers": 8,
    }
    fields.update(overrides)
    return TeiServerInfo(**fields)  # pyright: ignore[reportArgumentType]


def test_the_frozen_server_info_parses_and_records_every_field() -> None:
    info = _parse()
    assert info.version == "1.9.4"
    assert info.model_id == _MODEL_ID
    assert info.model_sha == _MODEL_SHA
    assert info.model_dtype == "float16"
    assert info.max_input_length == 8192
    assert info.max_batch_tokens == 8192
    assert info.auto_truncate is True
    assert info.max_client_batch_size == 32
    assert info.sha == "c" * 40
    assert info.docker_label is not None
    assert info.max_concurrent_requests == 512
    assert info.max_batch_requests == 4
    assert info.tokenization_workers == 8


def test_the_server_info_digest_is_stable_and_is_not_the_url_digest() -> None:
    first = _parse()
    second = _parse()
    assert first.sha256 == second.sha256
    url_digest = hashlib.sha256(b"http://127.0.0.1:8080").hexdigest()
    assert first.sha256 != url_digest


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        pytest.param({"version": "1.8.0"}, "not the frozen", id="version"),
        pytest.param({"model_id": "someone/else"}, "not the plan's", id="model-id"),
        pytest.param({"model_sha": "0" * 40}, "not the pinned", id="model-sha"),
        pytest.param({"model_dtype": "float32"}, "declared production precision", id="dtype"),
        pytest.param({"max_input_length": 4096}, "max_input_length", id="max-input-length"),
        pytest.param({"max_batch_tokens": 16384}, "max_batch_tokens is 16384", id="max-batch"),
        pytest.param({"auto_truncate": False}, "auto_truncate", id="auto-truncate"),
        pytest.param(
            {"max_client_batch_size": 4}, "below the frozen client batch size", id="batch"
        ),
    ],
)
def test_the_server_info_contract_refuses_a_wrong_serving_identity(
    overrides: dict[str, object], match: str
) -> None:
    with pytest.raises(BenchmarkContractError, match=match):
        _parse(**overrides)


def test_the_server_info_digest_changes_when_load_bearing_info_changes() -> None:
    base = _record()
    for overrides in (
        {"model_dtype": "bfloat16"},
        {"sha": "d" * 40},
        {"max_concurrent_requests": 256},
        {"tokenization_workers": 4},
        {"docker_label": "ghcr.io/example:1.9.4"},
        {"max_batch_requests": None},
        {"max_client_batch_size": 64},
    ):
        changed = _record(**overrides)
        assert changed.sha256 != base.sha256, overrides


def test_a_server_that_cannot_state_its_build_is_refused() -> None:
    with pytest.raises(BenchmarkContractError, match="not the frozen"):
        _parse(version="")
    with pytest.raises(BenchmarkContractError, match="version"):
        parse_tei_server_info(
            {**_payload(), "version": None},
            expected_model_id=_MODEL_ID,
            expected_model_revision=_MODEL_SHA,
            expected_precision="float16",
            min_max_client_batch_size=8,
            operation="test",
        )


@pytest.mark.parametrize("dimension", [512, 1024])
def test_the_request_body_is_exactly_the_tei_1_9_4_schema(dimension: int) -> None:
    body = tei_embed_request_body(inputs=["a", "b"], prompt_name="document", dimension=dimension)
    assert tuple(body) == TEI_EMBED_REQUEST_FIELDS
    assert body["inputs"] == ["a", "b"]
    assert body["prompt_name"] == "document"
    assert body["truncate"] is True
    assert body["truncation_direction"] == "right"
    assert body["normalize"] is True
    assert body["dimensions"] == dimension
    assert "max_batch_tokens" not in body
    assert RES138_TEI_REQUEST_SEMANTICS == {
        "truncate": True,
        "truncation_direction": "right",
        "normalize": True,
    }


def test_the_request_body_refuses_anything_but_the_frozen_semantics() -> None:
    with pytest.raises(BenchmarkContractError, match="prompt_name"):
        tei_embed_request_body(inputs=["a"], prompt_name="passage", dimension=512)
    with pytest.raises(BenchmarkContractError, match="truncate=true"):
        tei_embed_request_body(inputs=["a"], prompt_name="query", dimension=512, truncate=False)
    with pytest.raises(BenchmarkContractError, match="truncation_direction"):
        tei_embed_request_body(
            inputs=["a"], prompt_name="query", dimension=512, truncation_direction="left"
        )
    with pytest.raises(BenchmarkContractError, match="normalize=true"):
        tei_embed_request_body(inputs=["a"], prompt_name="query", dimension=512, normalize=False)
    with pytest.raises(BenchmarkContractError, match="dimension"):
        tei_embed_request_body(inputs=["a"], prompt_name="query", dimension=0)
    with pytest.raises(BenchmarkContractError, match="at least one input"):
        tei_embed_request_body(inputs=[], prompt_name="query", dimension=512)


@pytest.mark.parametrize(
    "endpoint",
    ["http://127.0.0.1:8080", "http://localhost:8080", "http://[::1]:8080", "https://127.0.0.1"],
)
def test_a_local_endpoint_is_accepted(endpoint: str) -> None:
    assert require_local_tei_endpoint(endpoint, operation="test") == endpoint


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://tei.example.com:8080",
        "http://192.168.1.10:8080",
        "http://gpu-box:8080",
        "127.0.0.1:8080",
        "",
    ],
)
def test_a_non_local_endpoint_is_refused(endpoint: str) -> None:
    with pytest.raises(BenchmarkContractError):
        require_local_tei_endpoint(endpoint, operation="test")
