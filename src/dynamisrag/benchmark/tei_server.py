"""The RES-138 Stage-B TEI serving contract: what the production server must prove it is.

Stage B sends embedding requests to TEI 1.9.4 and imports the vectors back as
equivalence evidence. Two things therefore have to be exact, and neither is a
request field:

* **The request.** TEI 1.9.4's ``POST /embed`` accepts ``inputs``, ``truncate``,
  ``truncation_direction``, ``prompt_name``, ``normalize`` and ``dimensions``.
  ``max_batch_tokens`` is *server* configuration: it is not an ``/embed`` request
  field, and a request that carried it would be rejected by the schema rather
  than applied. :func:`tei_embed_request_body` is the one builder of that body, so
  "every request states all six semantics explicitly" is checkable in one place.
* **The server.** ``GET /health`` proves liveness only and carries no identity, so
  the serving identity authority is ``GET /info``. :class:`TeiServerInfo` is the
  canonical record of that response; :func:`parse_tei_server_info` validates the
  frozen version, the pinned Qwen revision, the declared production dtype, the
  8192/8192 boundaries and ``auto_truncate``, and its :attr:`TeiServerInfo.sha256`
  is the serving-identity digest a GPU artifact binds. ``sha256(url)`` is never a
  serving identity: a URL says where a process listened, not what it served.

The request contract is separate from the response contract on purpose. A request
cannot prove which weights produced a vector, which is why the vectors travel with
the artifact and the local verifier recomputes the gate from bytes; the server
info is what makes the artifact's *claim about its runtime* falsifiable against
the response the server actually sent.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, cast
from urllib.parse import urlsplit

from dynamisrag.benchmark.contracts import (
    RES138_INPUT_MAX_TOKENS,
    RES138_INPUT_TRUNCATION_DIRECTION,
    require_exact_int,
    require_exact_str,
)
from dynamisrag.benchmark.errors import BenchmarkContractError
from dynamisrag.benchmark.production import RES138_PRODUCTION_TEI_RUNTIME
from dynamisrag.embedding.contracts import canonical_json

__all__ = [
    "RES138_TEI_REQUEST_SEMANTICS",
    "RES138_TEI_SERVER_REVISION",
    "TEI_EMBED_REQUEST_FIELDS",
    "TeiServerInfo",
    "parse_tei_server_info",
    "require_local_tei_endpoint",
    "tei_embed_request_body",
]

RES138_TEI_SERVER_REVISION: Final[str] = "res138-tei-server-info-v1"
"""Revision of the canonical TEI server-info record.

One record for one ``GET /info`` response. The record's digest is what a GPU
artifact binds as its serving identity, and the revision is hashed into the
record so a later contract can be told apart from this one.
"""

TEI_EMBED_REQUEST_FIELDS: Final[tuple[str, ...]] = (
    "inputs",
    "prompt_name",
    "truncate",
    "truncation_direction",
    "normalize",
    "dimensions",
)
"""The exact keys of a TEI 1.9.4 ``POST /embed`` request, and nothing else.

Closed because the omission that matters is invisible: a request that inherited
``truncate`` or ``normalize`` from the server's default would still return
well-formed vectors, and the equivalence gate would be comparing a different
function. ``max_batch_tokens`` is deliberately absent — it is server
configuration, not a request field.
"""

RES138_TEI_REQUEST_SEMANTICS: Final[Mapping[str, object]] = {
    "truncate": True,
    "truncation_direction": RES138_INPUT_TRUNCATION_DIRECTION,
    "normalize": True,
}
"""The semantic flags every Stage-B request carries, from the frozen contracts.

The Stage A reference boundary is 8192 tokens with right truncation, and the
reference vectors are L2-normalised. Sending these explicitly is what makes a
request's function independent of TEI's defaults.
"""

_LOCAL_TEI_HOSTS: Final[frozenset[str]] = frozenset({"localhost", "127.0.0.1", "::1"})
"""The only TEI endpoint hosts this benchmark lane admits.

Stage B measures VRAM on the GPU operator host and compares the served dtype to
the production precision declared for the run; a remote endpoint would make the
VRAM provenance and the host identity two different machines. The lane is a
local one on purpose.
"""


def tei_embed_request_body(
    *,
    inputs: Sequence[str],
    prompt_name: str,
    dimension: int,
    truncate: bool = True,
    truncation_direction: str = RES138_INPUT_TRUNCATION_DIRECTION,
    normalize: bool = True,
) -> dict[str, object]:
    """The exact ``POST /embed`` request body for one Stage-B client batch.

    Every field is written explicitly and the key set is exactly
    :data:`TEI_EMBED_REQUEST_FIELDS`; there is no path by which an operator's
    request could carry ``max_batch_tokens`` or omit the truncation direction.
    """
    if isinstance(inputs, (str, bytes)):
        raise BenchmarkContractError(
            "a TEI /embed request carries a sequence of input texts, not a scalar.",
            operation="tei_embed_request_body",
        )
    texts = [
        require_exact_str(text, kind="TEI input text", operation="tei_embed_request_body")
        for text in inputs
    ]
    if not texts:
        raise BenchmarkContractError(
            "a TEI /embed request carries at least one input. An empty batch has no rows to "
            "attribute and no duration worth timing.",
            operation="tei_embed_request_body",
        )
    dimension = require_exact_int(
        dimension,
        kind="TEI request dimensions",
        operation="tei_embed_request_body",
        minimum=1,
        because="The requested dimension must be stated, never inherited from the server.",
    )
    prompt = require_exact_str(
        prompt_name, kind="TEI prompt_name", operation="tei_embed_request_body"
    )
    if prompt not in ("query", "document"):
        raise BenchmarkContractError(
            f"the TEI prompt_name {prompt!r} is neither 'query' nor 'document'. The frozen "
            "candidates declare exactly those two model-native prompts, and an ad-hoc name "
            "would resolve to a different template or none at all.",
            operation="tei_embed_request_body",
        )
    if truncate is not True:
        raise BenchmarkContractError(
            "Stage B requests truncate=true. The Stage A reference boundary truncates over-long "
            "inputs, and a request that refused them would measure a different function.",
            operation="tei_embed_request_body",
        )
    if truncation_direction != RES138_INPUT_TRUNCATION_DIRECTION:
        raise BenchmarkContractError(
            f"the TEI request truncation_direction is {truncation_direction!r}, not the frozen "
            f"{RES138_INPUT_TRUNCATION_DIRECTION!r}.",
            operation="tei_embed_request_body",
        )
    if normalize is not True:
        raise BenchmarkContractError(
            "Stage B requests normalize=true. Cosine similarity over unnormalised vectors "
            "measures length as well as direction, and the reference is normalised.",
            operation="tei_embed_request_body",
        )
    body: dict[str, object] = {
        "inputs": texts,
        "prompt_name": prompt,
        "truncate": True,
        "truncation_direction": RES138_INPUT_TRUNCATION_DIRECTION,
        "normalize": True,
        "dimensions": dimension,
    }
    if tuple(body) != TEI_EMBED_REQUEST_FIELDS:
        raise BenchmarkContractError(
            "the TEI request body no longer has exactly the frozen fields in the frozen order.",
            operation="tei_embed_request_body",
        )
    return body


def require_local_tei_endpoint(url: str, *, operation: str) -> str:
    """Require the TEI endpoint to be local to the GPU operator host.

    The equivalence evidence binds the *server-host* GPU identity and VRAM, so a
    URL pointing anywhere but this host would attach those observations to the
    wrong machine. ``localhost``, ``127.0.0.1`` and ``::1`` are the only hosts
    this lane admits; a hostname that merely resolves locally is not accepted,
    because resolution is an ambient fact and the lane's identity is not.
    """
    text = require_exact_str(url, kind="TEI endpoint", operation=operation)
    split = urlsplit(text)
    if split.scheme not in ("http", "https") or split.hostname not in _LOCAL_TEI_HOSTS:
        raise BenchmarkContractError(
            f"the TEI endpoint {text!r} is not local to the GPU operator host. This benchmark lane "
            "records the server host's GPU identity and VRAM beside the vectors, so it admits only "
            f"http(s) endpoints on {sorted(_LOCAL_TEI_HOSTS)}.",
            operation=operation,
        )
    return text


def _require_info_int(value: object, *, key: str, operation: str, minimum: int = 1) -> int:
    return require_exact_int(
        value,
        kind=f"TEI /info {key}",
        operation=operation,
        minimum=minimum,
        because="The serving contract records this value and refuses to record an absent one.",
    )


def _optional_info_int(value: object, *, key: str, operation: str) -> int | None:
    if value is None:
        return None
    return _require_info_int(value, key=key, operation=operation)


@dataclass(frozen=True)
class TeiServerInfo:
    """The canonical ``GET /info`` record of one production TEI server.

    Every recorded field is load-bearing for the digest, which is the point: a
    server whose dtype, revision, boundary or build changed is a different server,
    and :attr:`sha256` changes with it. ``docker_label``, ``max_batch_requests``
    and ``tokenization_workers`` are genuinely optional in TEI and are recorded as
    ``None`` when the build does not state them rather than fabricated.
    """

    version: str
    model_id: str
    model_sha: str
    model_dtype: str
    max_input_length: int
    max_batch_tokens: int
    auto_truncate: bool
    max_client_batch_size: int
    sha: str
    docker_label: str | None
    max_concurrent_requests: int
    max_batch_requests: int | None
    tokenization_workers: int | None

    def __post_init__(self) -> None:
        for name in ("version", "model_id", "model_sha", "model_dtype", "sha"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise BenchmarkContractError(
                    f"the TEI server info {name} is {value!r}, which is not a non-empty string. "
                    "The serving identity is made of observed values.",
                    operation="tei_server_info",
                )
        for name in (
            "max_input_length",
            "max_batch_tokens",
            "max_client_batch_size",
            "max_concurrent_requests",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise BenchmarkContractError(
                    f"the TEI server info {name} is {value!r}, which is not a positive integer.",
                    operation="tei_server_info",
                )
        if self.auto_truncate is not True and self.auto_truncate is not False:
            raise BenchmarkContractError(
                "the TEI server info auto_truncate must be a real boolean.",
                operation="tei_server_info",
            )
        if self.docker_label is not None and not self.docker_label:
            raise BenchmarkContractError(
                "the TEI server info docker_label must be a non-empty string when the build "
                "states one, or absent.",
                operation="tei_server_info",
            )
        for name in ("max_batch_requests", "tokenization_workers"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise BenchmarkContractError(
                    f"the TEI server info {name} is {value!r}, which is neither a positive integer "
                    "nor absent.",
                    operation="tei_server_info",
                )

    def payload(self) -> dict[str, object]:
        """The hashed canonical record of this server."""
        return {
            "artifact_revision": RES138_TEI_SERVER_REVISION,
            "version": self.version,
            "model_id": self.model_id,
            "model_sha": self.model_sha,
            "model_dtype": self.model_dtype,
            "max_input_length": self.max_input_length,
            "max_batch_tokens": self.max_batch_tokens,
            "auto_truncate": self.auto_truncate,
            "max_client_batch_size": self.max_client_batch_size,
            "sha": self.sha,
            "docker_label": self.docker_label,
            "max_concurrent_requests": self.max_concurrent_requests,
            "max_batch_requests": self.max_batch_requests,
            "tokenization_workers": self.tokenization_workers,
        }

    @property
    def sha256(self) -> str:
        """SHA-256 over the canonical server-info record.

        This is the serving identity a GPU artifact binds — never a digest of the
        URL, which would identify where the process listened rather than what it
        served.
        """
        return hashlib.sha256(canonical_json(self.payload()).encode("utf-8")).hexdigest()


def parse_tei_server_info(
    payload: Mapping[str, object],
    *,
    expected_model_id: str,
    expected_model_revision: str,
    expected_precision: str,
    min_max_client_batch_size: int,
    operation: str,
) -> TeiServerInfo:
    """Validate one ``GET /info`` response against the frozen Stage-B contract.

    The order is the order of trust: the build version first (a different TEI
    serves different vectors), then the model identity and immutable revision,
    then the declared dtype — which must be the precision this run deliberately
    chose, never one inferred from a benchmark — then the 8192/8192 boundary with
    ``auto_truncate``, and only then the client batch ceiling.
    """
    frozen_version = cast("str", RES138_PRODUCTION_TEI_RUNTIME["tei_version"])
    version = require_exact_str(
        payload.get("version"), kind="TEI /info version", operation=operation
    )
    if version != frozen_version:
        raise BenchmarkContractError(
            f"the TEI endpoint serves {version!r}, not the frozen {frozen_version!r}. A different "
            "serving build changes the vectors a request returns, so its vectors are not what this "
            "contract's equivalence gate compares.",
            operation=operation,
            expected=frozen_version,
            observed=version,
        )
    model_id = require_exact_str(
        payload.get("model_id"), kind="TEI /info model_id", operation=operation
    )
    if model_id != expected_model_id:
        raise BenchmarkContractError(
            f"the TEI endpoint serves model {model_id!r}, not the plan's {expected_model_id!r}.",
            operation=operation,
            expected=expected_model_id,
            observed=model_id,
        )
    model_sha = require_exact_str(
        payload.get("model_sha"), kind="TEI /info model_sha", operation=operation
    )
    if model_sha != expected_model_revision:
        raise BenchmarkContractError(
            f"the TEI endpoint serves {model_id!r} at revision {model_sha!r}, not the pinned "
            f"{expected_model_revision!r}. A mutable tag or another commit is a different model.",
            operation=operation,
            expected=expected_model_revision,
            observed=model_sha,
        )
    model_dtype = require_exact_str(
        payload.get("model_dtype"), kind="TEI /info model_dtype", operation=operation
    )
    if model_dtype != expected_precision:
        raise BenchmarkContractError(
            f"the TEI endpoint serves dtype {model_dtype!r}, not the declared production precision "
            f"{expected_precision!r}. The precision is chosen before launch and observed here; it "
            "is never inferred from a benchmark result.",
            operation=operation,
            expected=expected_precision,
            observed=model_dtype,
        )
    max_input_length = _require_info_int(
        payload.get("max_input_length"), key="max_input_length", operation=operation
    )
    if max_input_length != RES138_INPUT_MAX_TOKENS:
        raise BenchmarkContractError(
            f"the TEI endpoint max_input_length is {max_input_length}, not the frozen "
            f"{RES138_INPUT_MAX_TOKENS}.",
            operation=operation,
            expected=str(RES138_INPUT_MAX_TOKENS),
            observed=str(max_input_length),
        )
    max_batch_tokens = _require_info_int(
        payload.get("max_batch_tokens"), key="max_batch_tokens", operation=operation
    )
    if max_batch_tokens != RES138_INPUT_MAX_TOKENS:
        raise BenchmarkContractError(
            f"the TEI endpoint max_batch_tokens is {max_batch_tokens}, not the frozen "
            f"{RES138_INPUT_MAX_TOKENS}. TEI's default is 16384, which is not the reference "
            "boundary; the server must be launched with the frozen Stage-B configuration.",
            operation=operation,
            expected=str(RES138_INPUT_MAX_TOKENS),
            observed=str(max_batch_tokens),
        )
    auto_truncate = payload.get("auto_truncate")
    if auto_truncate is not True:
        raise BenchmarkContractError(
            f"the TEI endpoint auto_truncate is {auto_truncate!r}, not true. The Stage A reference "
            "truncates over-long inputs at the boundary; a server that refuses or passes them "
            "through is a different function.",
            operation=operation,
        )
    max_client_batch_size = _require_info_int(
        payload.get("max_client_batch_size"), key="max_client_batch_size", operation=operation
    )
    if max_client_batch_size < min_max_client_batch_size:
        raise BenchmarkContractError(
            f"the TEI endpoint max_client_batch_size is {max_client_batch_size}, below the frozen "
            f"client batch size {min_max_client_batch_size}. The measurement protocol cannot issue "
            "its declared batches against this server.",
            operation=operation,
            expected=str(min_max_client_batch_size),
            observed=str(max_client_batch_size),
        )
    sha = require_exact_str(payload.get("sha"), kind="TEI /info sha", operation=operation)
    docker_label = payload.get("docker_label")
    if docker_label is not None and not isinstance(docker_label, str):
        raise BenchmarkContractError(
            f"the TEI endpoint docker_label is {docker_label!r}, which is neither a string nor "
            "null.",
            operation=operation,
        )
    return TeiServerInfo(
        version=version,
        model_id=model_id,
        model_sha=model_sha,
        model_dtype=model_dtype,
        max_input_length=max_input_length,
        max_batch_tokens=max_batch_tokens,
        auto_truncate=auto_truncate,
        max_client_batch_size=max_client_batch_size,
        sha=sha,
        docker_label=docker_label,
        max_concurrent_requests=_require_info_int(
            payload.get("max_concurrent_requests"),
            key="max_concurrent_requests",
            operation=operation,
        ),
        max_batch_requests=_optional_info_int(
            payload.get("max_batch_requests"), key="max_batch_requests", operation=operation
        ),
        tokenization_workers=_optional_info_int(
            payload.get("tokenization_workers"), key="tokenization_workers", operation=operation
        ),
    )
