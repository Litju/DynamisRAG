"""Pinned loopback Ollama client. No downloads, retries, tools or GPU offload."""

from __future__ import annotations

import json
from typing import Literal
from urllib.parse import urlsplit

import httpx2
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError

MODEL = "qwen3:4b"
DIGEST = "359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7"
RUNTIME = "0.13.5"
CONTEXT_TOKENS = 8192
GENERATION_TOKENS = 1024
PROMPT_BYTES = 6000
OPTIONS: dict[str, JsonValue] = {
    "num_ctx": CONTEXT_TOKENS,
    "num_predict": GENERATION_TOKENS,
    "num_gpu": 0,
    "num_thread": 4,
    "temperature": 0,
    "seed": 42,
}
_CODES = {
    "generation_unavailable",
    "generation_timeout",
    "generation_identity_mismatch",
    "generation_invalid_output",
    "generation_busy",
    "answer_disabled",
    "retrieval_unavailable",
    "provenance_invalid",
}


class GenerationUnavailableError(Exception):
    def __init__(self, code: str) -> None:
        self.code = code if code in _CODES else "generation_unavailable"
        super().__init__(self.code)


class DraftClaim(BaseModel):
    """An extractive claim: the model may select findings, not invent wording or identifiers."""

    model_config = ConfigDict(extra="forbid", strict=True)
    passage_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    quote: str = Field(min_length=20, max_length=650)


class DraftAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    status: Literal["supported", "insufficient_evidence"]
    claims: list[DraftClaim] = Field(max_length=3)


class GenerationIdentity(BaseModel):
    model: str
    digest: str
    quantization: str
    parameter_size: str
    runtime_version: str
    device: str = "cpu"


class GenerationMetrics(BaseModel):
    model_config = ConfigDict(strict=True)
    load_duration: int = Field(ge=0)
    prompt_eval_count: int = Field(ge=0, le=CONTEXT_TOKENS - GENERATION_TOKENS)
    prompt_eval_duration: int = Field(ge=0)
    eval_count: int = Field(ge=0, le=GENERATION_TOKENS)
    eval_duration: int = Field(ge=0)
    total_duration: int = Field(ge=0)


class _Details(BaseModel):
    format: str
    family: str
    parameter_size: str
    quantization_level: str


class _Model(BaseModel):
    name: str
    digest: str
    details: _Details


class _Tags(BaseModel):
    models: list[_Model]


class _RunningModel(_Model):
    size_vram: int
    context_length: int


class _Running(BaseModel):
    models: list[_RunningModel]


class _Message(BaseModel):
    role: str
    content: str = Field(max_length=12000)


class _Chat(GenerationMetrics):
    model: str
    message: _Message
    done: bool
    done_reason: str


class OllamaClient:
    def __init__(self, url: str, *, transport: httpx2.BaseTransport | None = None) -> None:
        parsed = urlsplit(url)
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed.username
            or parsed.password
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Ollama requires loopback HTTP")
        self.http = httpx2.Client(
            base_url=url.rstrip("/"),
            trust_env=False,
            follow_redirects=False,
            timeout=httpx2.Timeout(300, connect=3, write=10, pool=3),
            transport=transport,
        )

    def close(self) -> None:
        self.http.close()

    def _request(
        self, method: str, path: str, body: dict[str, JsonValue] | None = None
    ) -> dict[str, JsonValue]:
        try:
            with self.http.stream(method, path, json=body) as response:
                if response.status_code != 200:
                    raise GenerationUnavailableError("generation_unavailable")
                raw = bytearray()
                for chunk in response.iter_bytes():
                    if len(raw) + len(chunk) > 65536:
                        raise GenerationUnavailableError("generation_invalid_output")
                    raw.extend(chunk)
                return TypeAdapter(dict[str, JsonValue]).validate_json(bytes(raw))
        except httpx2.TimeoutException:
            raise GenerationUnavailableError("generation_timeout") from None
        except httpx2.HTTPError:
            raise GenerationUnavailableError("generation_unavailable") from None
        except (ValueError, ValidationError):
            raise GenerationUnavailableError("generation_invalid_output") from None

    def describe(self) -> GenerationIdentity:
        try:
            version = self._request("GET", "/api/version").get("version")
            tags = _Tags.model_validate(self._request("GET", "/api/tags"))
            found = [m for m in tags.models if m.name == MODEL]
            show = self._request("POST", "/api/show", {"model": MODEL, "verbose": False})
            details = _Details.model_validate(show.get("details"))
            if (
                version != RUNTIME
                or len(found) != 1
                or found[0].digest != DIGEST
                or found[0].details != details
                or details.format != "gguf"
                or details.family != "qwen3"
                or details.parameter_size != "4.0B"
                or details.quantization_level != "Q4_K_M"
            ):
                raise GenerationUnavailableError("generation_identity_mismatch")
            return GenerationIdentity(
                model=MODEL,
                digest=found[0].digest,
                quantization=details.quantization_level,
                parameter_size=details.parameter_size,
                runtime_version=RUNTIME,
            )
        except (ValueError, ValidationError):
            raise GenerationUnavailableError("generation_identity_mismatch") from None

    def generate(
        self, system: str, context: str
    ) -> tuple[DraftAnswer, GenerationIdentity, GenerationMetrics]:
        if len((system + context).encode("utf-8")) > PROMPT_BYTES:
            raise GenerationUnavailableError("generation_invalid_output")
        before = self.describe()
        schema = TypeAdapter(dict[str, JsonValue]).validate_python(DraftAnswer.model_json_schema())
        body: dict[str, JsonValue] = {
            "model": MODEL,
            "stream": False,
            "think": False,
            "keep_alive": "10m",
            "options": OPTIONS,
            "format": schema,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": context},
            ],
        }
        try:
            chat = _Chat.model_validate(self._request("POST", "/api/chat", body))
            after = self.describe()
            running = _Running.model_validate(self._request("GET", "/api/ps"))
            active = [m for m in running.models if m.name == MODEL]
            if (
                before != after
                or chat.model != MODEL
                or len(active) != 1
                or active[0].digest != DIGEST
                or active[0].size_vram != 0
                or active[0].context_length != CONTEXT_TOKENS
            ):
                raise GenerationUnavailableError("generation_identity_mismatch")
            if not chat.done or chat.done_reason != "stop" or chat.message.role != "assistant":
                raise GenerationUnavailableError("generation_invalid_output")
            draft = DraftAnswer.model_validate_json(chat.message.content)
            if (draft.status == "supported") != bool(draft.claims):
                raise GenerationUnavailableError("generation_invalid_output")
            metrics = GenerationMetrics.model_validate(chat.model_dump())
            return draft, after, metrics
        except (ValueError, ValidationError, json.JSONDecodeError):
            raise GenerationUnavailableError("generation_invalid_output") from None
