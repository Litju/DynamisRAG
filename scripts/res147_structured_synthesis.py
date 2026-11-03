"""Run one real, bounded RES-147 synthesis and keep all source-bearing capture private."""

# ruff: noqa: T201 -- this operator CLI writes its safe receipt to stdout
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from dynamisrag.config import load_settings
from dynamisrag.db.engine import create_database_engine
from dynamisrag.domain.identity import evidence_identity
from dynamisrag.evidence.contracts import RetrievalIdentity, SelectionRequest
from dynamisrag.generation.ollama import OllamaClient
from dynamisrag.generation.ollama_provider import OllamaGenerationProvider
from dynamisrag.generation.provider import GenerationAttempt, GenerationContext, GenerationProvider
from dynamisrag.generation.structured import POLICY_SHA256, SYSTEM
from dynamisrag.generation.synthesis import (
    StructuredSynthesisResult,
    StructuredSynthesisService,
    SynthesisRequest,
)
from dynamisrag.search.scientific import ScientificRetrievalService

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = ROOT / ".dynamisrag" / "res147" / "private"
QUESTION = "What population, load and instruments were used in the knee-angle experiment?"


class CapturingProvider:
    def __init__(self, delegate: GenerationProvider) -> None:
        self.delegate = delegate
        self.capture_dir: Path | None = None

    def generate(self, context: GenerationContext) -> GenerationAttempt:
        private_root = PRIVATE_ROOT.resolve()
        if not private_root.is_relative_to(ROOT.resolve()):
            raise RuntimeError("private capture root resolved outside the repository")
        git = shutil.which("git")
        if git is None:
            raise RuntimeError("Git is required to verify the private capture directory")
        tracked = subprocess.run(  # noqa: S603 -- fixed Git command, executable resolved from PATH
            [git, "ls-files", "--", ".dynamisrag/res147"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if tracked:
            raise RuntimeError("private capture root is not ignored local evidence storage")
        attempt = self.delegate.generate(context)
        self.capture_dir = private_root / uuid4().hex
        self.capture_dir.mkdir(parents=True, exist_ok=False)
        (self.capture_dir / "system.txt").write_text(SYSTEM, encoding="utf-8")
        (self.capture_dir / "user.json").write_text(
            json.dumps(context.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        (self.capture_dir / "response.json").write_text(attempt.raw_output, encoding="utf-8")
        return attempt


def public_receipt(result: StructuredSynthesisResult) -> dict[str, object]:
    graph = result.graph
    return {
        "schema_revision": "res147-public-receipt-v1",
        "status": result.status,
        "question_identity": result.question_identity,
        "retrieval_identity": result.retrieval_identity,
        "retrieval_response_identity": result.retrieval_response_identity,
        "selection_identity": result.selection_identity,
        "selection_trace": [d.model_dump(mode="json") for d in result.selection_trace],
        "evidence_set_identity": result.evidence_set_identity,
        "members": [m.model_dump(mode="json") for m in result.evidence_set.members]
        if result.evidence_set
        else [],
        "generation_model": result.generation_semantics.model.model_dump(mode="json")
        if result.generation_semantics
        else None,
        "generation_policy_sha256": POLICY_SHA256,
        "generation_semantics": result.generation_semantics.model_dump(mode="json")
        if result.generation_semantics
        else None,
        "execution_conditions": result.execution_conditions.model_dump(mode="json")
        if result.execution_conditions
        else None,
        "execution_metrics": result.execution_metrics.model_dump(mode="json")
        if result.execution_metrics
        else None,
        "raw_output_sha256": result.raw_output_sha256,
        "graph_identity": result.graph_identity,
        "claim_identities": [claim.key for claim in graph.claims] if graph else [],
        "evidence_link_identities": [link.key for link in graph.links] if graph else [],
        "link_dispositions": [
            {
                "key": link.key,
                "relation": link.relation.value,
                "review_state": link.review_state.value,
            }
            for link in graph.links
        ]
        if graph
        else [],
        "missing_evidence": result.missing_evidence,
        "limitations": result.limitations,
        "context_bytes": result.context_bytes,
        "context_exclusions": result.context_exclusions,
        "generation_invoked": result.generation_invoked,
        "reference_database_identity_before_after": None,
        "source_files_identity_before_after": None,
        "qdrant_identity_before_after": None,
        "benchmark_identity_before_after": None,
    }


def main() -> None:
    sys.path.insert(0, str(ROOT))
    from scripts.res145_evidence_demo import database_receipt, protected_files, qdrant_receipt
    from scripts.res146_evidence_selection import protected_benchmark_files

    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default=QUESTION)
    args = parser.parse_args()
    settings = load_settings()
    engine = create_database_engine(settings)
    retrieval = ScientificRetrievalService(engine, settings)
    client = OllamaClient(str(settings.scientific_ollama_url))
    try:
        model = client.describe()
        retrieval.start()
        if retrieval.projection is None or retrieval.adapter is None:
            raise RuntimeError(f"scientific retrieval unavailable: {retrieval.unavailable_code}")
        before_database = evidence_identity("res147-canonical-db-v1", database_receipt(engine))
        before_sources = protected_files()
        before_qdrant = qdrant_receipt()
        before_benchmark = protected_benchmark_files()
        projection = retrieval.projection
        selection = SelectionRequest(
            question=args.question,
            corpus=RetrievalIdentity(
                embedding_manifest_sha256=projection.embeddings.manifest_sha256,
                projection_sha256=projection.passages.projection_sha256,
            ),
        )
        provider = CapturingProvider(OllamaGenerationProvider(client))
        service = StructuredSynthesisService(retrieval, provider)
        result = service.synthesize(SynthesisRequest(selection=selection))
        after_database = evidence_identity("res147-canonical-db-v1", database_receipt(engine))
        after_sources = protected_files()
        after_qdrant = qdrant_receipt()
        after_benchmark = protected_benchmark_files()
        if (before_database, before_sources, before_qdrant, before_benchmark) != (
            after_database,
            after_sources,
            after_qdrant,
            after_benchmark,
        ):
            raise RuntimeError("protected scientific data changed during synthesis")
        receipt = public_receipt(result)
        receipt.update(
            {
                "observed_generation_model": model.model_dump(mode="json"),
                "reference_database_identity_before_after": [before_database, after_database],
                "source_files_identity_before_after": [before_sources, after_sources],
                "qdrant_identity_before_after": [before_qdrant, after_qdrant],
                "benchmark_identity_before_after": [before_benchmark, after_benchmark],
                "private_capture_id": provider.capture_dir.name if provider.capture_dir else None,
            }
        )
        print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    finally:
        client.close()
        retrieval.close()
        engine.dispose()


if __name__ == "__main__":
    main()
