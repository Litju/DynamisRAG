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
import json
import subprocess
import sys
from pathlib import Path

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


def _opencode_go(out: Path) -> ModelRuntime | None:
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
    canonical = {r.value: c for r, c in verify_opencode_go_roster(roster).items()}
    try:
        return build_opencode_go_runtime(roster)
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
    args = parser.parse_args(argv)

    roster_label = "+".join(spec.model_name for spec in DEFAULT_ROSTER.values())
    runtime: ModelRuntime
    if args.provider == "routed-deterministic":
        provider = AgentProviderKind.DETERMINISTIC
        runtime = _routed_deterministic()
        model_name = f"routed-deterministic:{roster_label}"
    elif args.provider == "opencode_go":
        provider = AgentProviderKind.OPENCODE_GO
        built = _opencode_go(args.out)
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
