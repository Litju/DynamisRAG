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
    build_agent_model_runtime,
)


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["ollama", "deterministic"], default="ollama")
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11435")
    parser.add_argument("--cases", nargs="*", default=None, help="case ids (default: all)")
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--call-timeout", type=float, default=300.0)
    parser.add_argument("--run-timeout", type=float, default=3000.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    provider = AgentProviderKind(args.provider)
    config = AgentModelConfig(
        provider=provider,
        model_name=args.model if provider != AgentProviderKind.DETERMINISTIC else "deterministic",
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        timeout_seconds=args.call_timeout,
    )
    runtime = build_agent_model_runtime(config)
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
        model_name=config.model_name,
        generative=provider != AgentProviderKind.DETERMINISTIC,
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
