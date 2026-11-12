"""Privacy-controlled LangSmith tracing for the RES-637 research agent.

Policy (all enforced in code and pinned by tests that inspect the *actual bytes* a LangSmith
client would transmit):

* **Off unless explicitly enabled.** ``TracingPolicy.off()`` wraps every run in
  ``tracing_context(enabled=False)``, which beats ambient ``LANGSMITH_TRACING`` environment
  variables. Enabling requires an explicit API key; a missing key is a configuration error, not a
  silent no-op.
* **No content by default.** The default mode, ``structure_only``, replaces every input and output
  value with a shape descriptor (type, character/element counts). Raw user questions, article
  text, answers, tool results, model reasoning and credentials therefore never leave the process.
* **Metadata is hidden** (``hide_metadata``): LangGraph/LangChain attach the thread id, run id,
  worker id and tags there, and those encode user and conversation identity.
* **Opt-in content mode.** ``redacted_content`` keeps text but scrubs credentials, DSN passwords and
  e-mail addresses, bounds string length, drops reasoning fields/``<think>`` blocks, and still hides
  metadata. It must be chosen explicitly and is never the default.
* The client is built per policy and passed to ``tracing_context``; secrets are never put in the
  process environment by this module.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final, cast

from langsmith import Client, tracing_context  # pyright: ignore[reportUnknownVariableType]
from pydantic import SecretStr

__all__ = [
    "REASONING_KEYS",
    "TracingMode",
    "TracingPolicy",
    "build_langsmith_client",
    "configure_privacy_safe_langsmith",
    "redact_trace_payload",
    "redact_trace_string",
    "structure_only",
]

REASONING_KEYS: Final[frozenset[str]] = frozenset(
    {"reasoning", "reasoning_content", "thinking", "thought", "thoughts", "chain_of_thought"}
)

_SECRET_KEY_PATTERN: Final = re.compile(
    r"(?:password|secret|token|api[_-]?key|authorization|dsn|database_url|credential)", re.I
)
_BEARER_VALUE_PATTERN: Final = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.I)
_PG_URL_PASSWORD_PATTERN: Final = re.compile(r"(postgresql(?:\+psycopg)?://[^:/\s]+:)[^@\s]+(@)")
_EMAIL_PATTERN: Final = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_API_KEY_LITERAL: Final = re.compile(r"\b(?:sk|lsv2|ls)[-_][A-Za-z0-9_-]{12,}\b")
_THINK_BLOCK: Final = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


class TracingMode(StrEnum):
    OFF = "off"
    STRUCTURE_ONLY = "structure_only"
    REDACTED_CONTENT = "redacted_content"


def redact_trace_string(value: str, *, max_len: int = 240) -> str:
    """Scrub credentials, DSN passwords and e-mail addresses; drop reasoning; bound length."""
    scrubbed = _THINK_BLOCK.sub("[REASONING_REMOVED]", value)
    scrubbed = _BEARER_VALUE_PATTERN.sub("Bearer [REDACTED]", scrubbed)
    scrubbed = _PG_URL_PASSWORD_PATTERN.sub(r"\1[REDACTED]\2", scrubbed)
    scrubbed = _API_KEY_LITERAL.sub("[REDACTED_KEY]", scrubbed)
    scrubbed = _EMAIL_PATTERN.sub("[REDACTED_EMAIL]", scrubbed)
    if len(scrubbed) > max_len:
        return scrubbed[:max_len] + "...[TRUNCATED]"
    return scrubbed


def redact_trace_payload(payload: object, *, max_string_chars: int = 240) -> object:
    """Recursively redact secrets/PII, reasoning fields and long text bodies."""
    if isinstance(payload, str):
        return redact_trace_string(payload, max_len=max_string_chars)
    if isinstance(payload, Mapping):
        out: dict[str, object] = {}
        for k, v in cast("Mapping[object, object]", payload).items():
            key_str = str(k)
            if _SECRET_KEY_PATTERN.search(key_str):
                out[key_str] = "[REDACTED]"
            elif key_str.lower() in REASONING_KEYS:
                out[key_str] = "[REASONING_REMOVED]"
            elif key_str in {"text", "verbatim_quote", "answer_markdown"} and isinstance(v, str):
                out[key_str] = redact_trace_string(v, max_len=min(max_string_chars, 160))
            else:
                out[key_str] = redact_trace_payload(v, max_string_chars=max_string_chars)
        return out
    if isinstance(payload, (list, tuple)):
        seq = cast("Sequence[object]", payload)
        return [redact_trace_payload(item, max_string_chars=max_string_chars) for item in seq[:25]]
    return payload


def structure_only(payload: object) -> object:
    """Describe the *shape* of a value without any of its content."""
    if isinstance(payload, str):
        return {"_type": "str", "chars": len(payload)}
    if isinstance(payload, bool) or payload is None:
        return payload
    if isinstance(payload, (int, float)):
        return {"_type": "number"}
    if isinstance(payload, Mapping):
        mapping = cast("Mapping[object, object]", payload)
        return {str(k): structure_only(v) for k, v in list(mapping.items())[:50]}
    if isinstance(payload, (list, tuple)):
        seq = cast("Sequence[object]", payload)
        return {
            "_type": "list",
            "items": len(seq),
            "first": structure_only(seq[0]) if seq else None,
        }
    return {"_type": type(payload).__name__}


def _as_mapping_fn(fn: Callable[[object], object]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def _apply(inputs: dict[str, Any]) -> dict[str, Any]:
        return cast("dict[str, Any]", fn(inputs))

    return _apply


@dataclass(frozen=True)
class TracingPolicy:
    mode: TracingMode = TracingMode.OFF
    project_name: str = "dynamisrag-res637"
    api_key: SecretStr | None = None
    api_url: str | None = None
    session: Any = None
    """Injectable ``requests.Session``-compatible transport (tests inspect the sent bytes)."""

    def __post_init__(self) -> None:
        if self.mode != TracingMode.OFF and self.api_key is None:
            raise ValueError("LangSmith tracing was enabled without an API key")

    @property
    def enabled(self) -> bool:
        return self.mode != TracingMode.OFF

    @classmethod
    def off(cls) -> TracingPolicy:
        return cls(mode=TracingMode.OFF)

    def client(self) -> Client | None:
        return build_langsmith_client(self) if self.enabled else None

    @contextmanager
    def scope(self, *, client: Client | None = None) -> Generator[None]:
        """Run the enclosed LangChain/LangGraph work under this policy."""
        if not self.enabled:
            with tracing_context(enabled=False):
                yield
            return
        with tracing_context(
            enabled=True, client=client or self.client(), project_name=self.project_name
        ):
            yield


def build_langsmith_client(policy: TracingPolicy) -> Client:
    """Build a client that redacts at the SDK boundary, before anything is serialised."""
    if not policy.enabled or policy.api_key is None:
        raise ValueError("A LangSmith client is only built for an enabled policy with a key")
    redactor: Callable[[object], object] = (
        redact_trace_payload if policy.mode == TracingMode.REDACTED_CONTENT else structure_only
    )
    kwargs: dict[str, Any] = {
        "api_key": policy.api_key.get_secret_value(),
        "hide_inputs": _as_mapping_fn(redactor),
        "hide_outputs": _as_mapping_fn(redactor),
        "hide_metadata": True,
        "omit_traced_runtime_info": True,
    }
    if policy.api_url:
        kwargs["api_url"] = policy.api_url
    if policy.session is not None:
        kwargs["session"] = policy.session
        kwargs["auto_batch_tracing"] = False
    return Client(**kwargs)


def configure_privacy_safe_langsmith(
    *,
    tracing_enabled: bool = False,
    project_name: str = "dynamisrag-res637-eval",
) -> dict[str, str]:
    """Set defence-in-depth environment defaults (hidden content, tracing flag, project).

    The authoritative control is :class:`TracingPolicy`; this only makes sure an ambient SDK
    default can never be *more* permissive than the policy. It never writes a secret.
    """
    flag = "true" if tracing_enabled else "false"
    values = {
        "LANGSMITH_TRACING": flag,
        "LANGCHAIN_TRACING_V2": flag,
        "LANGSMITH_HIDE_INPUTS": "true",
        "LANGSMITH_HIDE_OUTPUTS": "true",
        "LANGSMITH_HIDE_METADATA": "true",
        "LANGCHAIN_HIDE_INPUTS": "true",
        "LANGCHAIN_HIDE_OUTPUTS": "true",
        "LANGSMITH_PROJECT": project_name,
        "LANGCHAIN_PROJECT": project_name,
    }
    os.environ.update(values)
    return values
