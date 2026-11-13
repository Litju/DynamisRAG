"""Run RES-637 agent evaluation v2 against one model configuration and write a JSON receipt.

Example (local Ollama; CPU generation is slow, a full run takes tens of minutes)::

    uv run python -m scripts.res637_agent_eval --provider ollama --model qwen3:4b \
        --base-url http://127.0.0.1:11435 \
        --out docs/evidence/res637-agent-evaluation-v2/qwen3-4b.json

The receipt records per-case outcomes and per-metric ``passed / applicable`` denominators. A
``--provider deterministic`` run is a pipeline check only and is stamped ``generative: false``.
No credential is read from the environment or written to the receipt.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import httpx2
from tests.agent_eval_corpora import build_corpus
from tests.agent_stack import build_stack

from dynamisrag.agent.evaluation import CorpusKind, EvaluationReport, run_evaluation
from dynamisrag.agent.evaluation_cases import EVALUATION_CASES
from dynamisrag.agent.graph import ConversationalResearchOrchestrator
from dynamisrag.agent.providers import (
    AgentModelConfig,
    AgentProviderKind,
    DeterministicResearchChatModel,
    ProviderConfigurationError,
    build_agent_model_runtime,
)
from dynamisrag.agent.routing import (
    DEFAULT_ROSTER,
    ModelRuntime,
    OpenCodeGoRosterConfig,
    RoutedModelRuntime,
    build_opencode_go_runtime,
    verify_opencode_go_roster,
)
from dynamisrag.config import Settings


def _git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip()


def _model_digest(base_url: str, model: str) -> str | None:
    try:
        tags = httpx2.get(f"{base_url}/api/tags", timeout=10.0).json()
    except (httpx2.HTTPError, ValueError):
        return None
    for entry in tags.get("models", []):
        if entry.get("name") == model:
            return str(entry.get("digest"))
    return None


_BLOCKED_EXIT_CODE = 3


def _routed_deterministic() -> RoutedModelRuntime:
    """Three-role routing exercised with scripted doubles: a pipeline check, never a model run."""
    return RoutedModelRuntime(
        base_config=AgentModelConfig(
            provider=AgentProviderKind.DETERMINISTIC, model_name="gpt-6-luna", base_url=None
        ),
        specs=DEFAULT_ROSTER,
        models={
            role: DeterministicResearchChatModel(model_name=spec.model_name)
            for role, spec in DEFAULT_ROSTER.items()
        },
    )


_SECRET_HEADERS = frozenset({"authorization", "x-api-key", "api-key", "cookie", "set-cookie"})


def _wire_logger(path: Path, secret: str) -> httpx.Client:
    """HTTP client that appends every provider request/response (full bodies) to a JSONL file.

    Only credential headers are redacted; any occurrence of the key value itself is scrubbed from
    the serialised line as a second guard. Bodies are complete: prompts, tool schemas, tool calls,
    reasoning_content, usage.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    counter = itertools.count(1)
    pending: dict[int, int] = {}

    def write(entry: dict[str, object]) -> None:
        line = json.dumps(entry, ensure_ascii=False, default=str)
        if secret:
            line = line.replace(secret, "[REDACTED_KEY]")
        with lock, path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def clean(headers: httpx.Headers) -> dict[str, str]:
        return {
            k: ("[REDACTED]" if k.lower() in _SECRET_HEADERS else v) for k, v in headers.items()
        }

    def on_request(request: httpx.Request) -> None:
        seq = next(counter)
        pending[id(request)] = seq
        body = request.content.decode("utf-8", errors="replace")
        try:
            parsed: object = json.loads(body)
        except ValueError:
            parsed = body
        write(
            {
                "seq": seq,
                "direction": "request",
                "ts": time.time(),
                "method": request.method,
                "url": str(request.url),
                "headers": clean(request.headers),
                "body": parsed,
            }
        )

    def on_response(response: httpx.Response) -> None:
        response.read()
        try:
            parsed: object = response.json()
        except ValueError:
            parsed = response.text
        write(
            {
                "seq": pending.pop(id(response.request), -1),
                "direction": "response",
                "ts": time.time(),
                "status": response.status_code,
                "elapsed_ms": round(response.elapsed.total_seconds() * 1000, 1),
                "headers": clean(response.headers),
                "body": parsed,
            }
        )

    return httpx.Client(event_hooks={"request": [on_request], "response": [on_response]})


def _case_dumper(directory: Path) -> object:
    """Write each case's full runs (answers, citations, telemetry) and event stream as JSON."""
    directory.mkdir(parents=True, exist_ok=True)

    def dump(case: object, runs: object, events: object) -> None:
        case_id = getattr(case, "case_id", "case")
        payload = {
            "case": getattr(case, "model_dump", lambda **_: {})(mode="json"),
            "runs": [r.model_dump(mode="json") for r in runs],  # type: ignore[attr-defined]
            "events_per_run": [
                [e.model_dump(mode="json") for e in run_events]  # type: ignore[attr-defined]
                for run_events in events  # type: ignore[attr-defined]
            ],
        }
        (directory / f"{case_id}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
        )

    return dump


def _opencode_go(out: Path, log_dir: Path | None = None) -> ModelRuntime | None:
    """Build the OpenCode Go roster only when the owner has recorded provider authorization.

    Without it nothing is sent to the provider except the public catalogue read; a blocked
    receipt records exactly why. Technical API access is not permission to run this assistant.
    """
    settings = Settings()
    roster = OpenCodeGoRosterConfig(
        api_key=settings.agent_model_api_key,
        base_url=None
        if settings.agent_model_base_url == "http://127.0.0.1:11435"
        else settings.agent_model_base_url,
        usage_approved=settings.agent_opencode_go_usage_approved,
        fast_model=settings.agent_model_fast,
        coordinator_model=settings.agent_model_coordinator,
        pro_model=settings.agent_model_pro,
        max_pro_escalations=settings.agent_max_pro_escalations,
        max_run_cost_usd=settings.agent_run_cost_budget_usd,
        max_run_tokens=settings.agent_run_token_budget,
    )
    try:
        canonical: dict[str, str] = {
            r.value: c for r, c in verify_opencode_go_roster(roster).items()
        }
    except (ProviderConfigurationError, httpx2.HTTPError) as error:
        canonical = {"catalogue_check_failed": type(error).__name__}
    try:
        client = (
            _wire_logger(
                log_dir / "wire.jsonl",
                roster.api_key.get_secret_value() if roster.api_key else "",
            )
            if log_dir is not None
            else None
        )
        return build_opencode_go_runtime(roster, http_client=client)
    except ProviderConfigurationError as error:
        receipt = {
            "version": "res637-opencode-go-qualification-blocked",
            "status": "BLOCKED_PROVIDER_AUTHORIZATION",
            "reason": str(error),
            "catalogue_verified_models": canonical,
            "key_configured": roster.api_key is not None,
            "scientific_inference_performed": False,
            "code_revision": _git_revision(),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(f"BLOCKED: {error}")  # noqa: T201
        print(f"wrote {out}")  # noqa: T201
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=["ollama", "deterministic", "routed-deterministic", "opencode_go"],
        default="ollama",
    )
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11435")
    parser.add_argument("--cases", nargs="*", default=None, help="case ids (default: all)")
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--call-timeout", type=float, default=300.0)
    parser.add_argument("--run-timeout", type=float, default=3000.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, default=None, help="save full wire + per-case logs")
    args = parser.parse_args(argv)

    roster_label = "+".join(spec.model_name for spec in DEFAULT_ROSTER.values())
    runtime: ModelRuntime
    if args.provider == "routed-deterministic":
        provider = AgentProviderKind.DETERMINISTIC
        runtime = _routed_deterministic()
        model_name = f"routed-deterministic:{roster_label}"
    elif args.provider == "opencode_go":
        provider = AgentProviderKind.OPENCODE_GO
        built = _opencode_go(args.out, args.log_dir)
        if built is None:
            return _BLOCKED_EXIT_CODE
        runtime = built
        model_name = f"routed:{roster_label}"
    else:
        provider = AgentProviderKind(args.provider)
        model_name = args.model if provider != AgentProviderKind.DETERMINISTIC else "deterministic"
        runtime = build_agent_model_runtime(
            AgentModelConfig(
                provider=provider,
                model_name=model_name,
                base_url=args.base_url,
                max_tokens=args.max_tokens,
                timeout_seconds=args.call_timeout,
            )
        )
    cache: dict[CorpusKind, tuple[ConversationalResearchOrchestrator, dict[str, str]]] = {}

    def factory(kind: CorpusKind) -> tuple[ConversationalResearchOrchestrator, dict[str, str]]:
        if kind not in cache:
            engine, keys = build_corpus(kind)
            stack = build_stack(
                runtime=runtime, engine=engine, studies={}, run_timeout_seconds=args.run_timeout
            )
            cache[kind] = (stack.orchestrator, keys)
        return cache[kind]

    cases = [c for c in EVALUATION_CASES if args.cases is None or c.case_id in args.cases]
    report: EvaluationReport = run_evaluation(
        cases=cases,
        factory=factory,
        provider=provider.value,
        model_name=model_name[:200],
        generative=provider not in {AgentProviderKind.DETERMINISTIC},
        model_digest=_model_digest(args.base_url, args.model)
        if provider == AgentProviderKind.OLLAMA
        else None,
        progress=lambda line: print(line, flush=True),  # noqa: T201
        on_case=_case_dumper(args.log_dir / "cases") if args.log_dir is not None else None,  # type: ignore[arg-type]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump(mode="json")
    payload["code_revision"] = _git_revision()
    payload["corpus"] = "hermetic synthetic corpora (SQLite, canonical lexical retrieval)"
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
