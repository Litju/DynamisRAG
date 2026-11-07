"""Run one real, bounded RES-147 synthesis and keep all source-bearing capture private."""

# ruff: noqa: T201 -- this operator CLI writes its safe receipt to stdout
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, cast
from urllib.parse import urlsplit
from uuid import uuid4

from psycopg import Connection as PsycopgConnection
from sqlalchemy import Engine, event

from dynamisrag.config import Settings, load_settings
from dynamisrag.db.engine import create_database_engine
from dynamisrag.db.test_cluster import (
    require_test_cluster_identity,
    require_test_cluster_system_identifier,
    verify_test_compose_resources,
)
from dynamisrag.domain.identity import evidence_identity
from dynamisrag.evidence.contracts import RetrievalIdentity, SelectionRequest
from dynamisrag.generation.ollama import PROMPT_BYTES, OllamaClient
from dynamisrag.generation.ollama_provider import OllamaGenerationProvider
from dynamisrag.generation.provider import (
    GenerationAttempt,
    GenerationContext,
    GenerationProvider,
    serialize_generation_context,
)
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


def verify_test_qdrant_resources() -> None:
    """Require the Qdrant API port and both volumes to belong to integration."""
    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError("Docker is required to verify the isolated Qdrant service")
    compose = [
        docker,
        "compose",
        "--project-name",
        "dynamisrag-integration",
        "--file",
        str(ROOT / "compose.yaml"),
        "--file",
        str(ROOT / "compose.integration.yaml"),
    ]
    try:
        container_ids = subprocess.run(  # noqa: S603 -- fixed Compose project and service
            [*compose, "ps", "--quiet", "qdrant"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.splitlines()
        if len(container_ids) != 1 or not container_ids[0]:
            raise RuntimeError("the isolated Qdrant container is missing or ambiguous")
        inspected = json.loads(
            subprocess.run(  # noqa: S603 -- inspect only the ID returned by fixed Compose
                [docker, "inspect", container_ids[0]],
                check=True,
                capture_output=True,
                text=True,
                timeout=20,
            ).stdout
        )
        if not isinstance(inspected, list):
            raise RuntimeError("Docker returned an ambiguous Qdrant identity")
        inspection_results = cast(list[object], inspected)
        if len(inspection_results) != 1 or not isinstance(inspection_results[0], dict):
            raise RuntimeError("Docker returned an ambiguous Qdrant identity")
        container = cast(dict[str, Any], inspection_results[0])
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        raise RuntimeError("could not verify the isolated Qdrant container") from error

    labels = cast(dict[str, str], container.get("Config", {}).get("Labels") or {})
    mounts = {
        mount.get("Destination"): mount.get("Name")
        for mount in cast(list[dict[str, Any]], container.get("Mounts", []))
    }
    bindings = cast(
        list[dict[str, str]],
        container.get("HostConfig", {}).get("PortBindings", {}).get("6333/tcp") or [],
    )
    if (
        not container.get("State", {}).get("Running")
        or labels.get("com.docker.compose.project") != "dynamisrag-integration"
        or labels.get("com.docker.compose.service") != "qdrant"
        or mounts.get("/qdrant/storage") != "dynamisrag-integration_qdrant-data"
        or mounts.get("/qdrant/snapshots") != "dynamisrag-integration_qdrant-snapshots"
        or {(binding.get("HostIp"), binding.get("HostPort")) for binding in bindings}
        != {("127.0.0.1", "16333")}
    ):
        raise RuntimeError("Qdrant is not the isolated integration service")


def require_isolated_targets(settings: Settings) -> None:
    """Reject retained/default service endpoints before constructing any client."""
    database = urlsplit(str(settings.database_url))
    if (
        database.scheme != "postgresql"
        or database.hostname != "127.0.0.1"
        or database.port != 55433
        or database.path != "/dynamisrag_test_res147_candidate"
        or database.query
        or database.fragment
        or database.username != "dynamisrag_test"
    ):
        raise RuntimeError(
            "RES-147 requires the read-only isolated 127.0.0.1:55433 "
            "dynamisrag_test_res147_candidate database"
        )

    endpoints = (
        ("OpenSearch", str(settings.opensearch_url), "https", 19200),
        ("Qdrant", str(settings.scientific_qdrant_url), "http", 16333),
        ("TEI", str(settings.scientific_tei_url), "http", 8081),
        ("Ollama", str(settings.scientific_ollama_url), "http", 11434),
    )
    for name, value, scheme, port in endpoints:
        endpoint = urlsplit(value)
        if (
            endpoint.scheme != scheme
            or endpoint.hostname != "127.0.0.1"
            or endpoint.port != port
            or endpoint.path not in {"", "/"}
            or endpoint.username
            or endpoint.password
            or endpoint.query
            or endpoint.fragment
        ):
            raise RuntimeError(f"RES-147 requires the isolated loopback {name} endpoint")

    require_test_cluster_system_identifier(settings.test_cluster_system_identifier)
    verify_test_compose_resources()
    verify_test_qdrant_resources()


def create_read_only_candidate_engine(settings: Settings) -> Engine:
    """Make every pooled candidate connection default to read-only transactions."""
    engine = create_database_engine(settings)

    @event.listens_for(engine, "connect")
    def set_read_only(dbapi_connection: PsycopgConnection[Any], _connection_record: object) -> None:
        with dbapi_connection.cursor() as cursor:
            cursor.execute("SET SESSION default_transaction_read_only = on")
        dbapi_connection.commit()

    return engine


def source_files_receipt(evidence_directory: Path) -> str:
    """Fingerprint the exact retained JATS set without copying or printing it."""
    source_root = evidence_directory / "sources" / "sha256"
    expected_receipt = cast(
        dict[str, object],
        json.loads(
            (ROOT / "docs/evidence/res147-canonical-reconstruction-receipt.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    expected_artifacts = expected_receipt.get("source_artifacts")
    if not source_root.is_dir() or not isinstance(expected_artifacts, list):
        raise RuntimeError("RES-147 requires the retained canonical source archive set")
    expected_hashes: set[str] = set()
    for artifact in cast(list[object], expected_artifacts):
        if not isinstance(artifact, dict):
            raise RuntimeError("canonical source receipt is malformed")
        digest = cast(dict[str, object], artifact).get("content_sha256")
        if not isinstance(digest, str):
            raise RuntimeError("canonical source receipt is malformed")
        expected_hashes.add(digest)
    files = sorted(path for path in source_root.rglob("*") if path.is_file())
    observed = [
        (
            path.relative_to(source_root).as_posix(),
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in files
    ]
    if len(files) != len(expected_hashes) or {digest for _, digest in observed} != expected_hashes:
        raise RuntimeError("retained canonical source archives differ from their receipt")
    return evidence_identity("res147-source-archives-v1", observed)


class CapturingProvider:
    def __init__(
        self, delegate: GenerationProvider, required_context_span: str | None = None
    ) -> None:
        self.delegate = delegate
        self.required_context_span = required_context_span
        self.capture_dir: Path | None = None
        self.required_span_verified = False
        self.prompt_sha256: str | None = None
        self.prompt_bytes: int | None = None

    def generate(self, context: GenerationContext) -> GenerationAttempt:
        serialized = serialize_generation_context(context)
        self.prompt_bytes = len((SYSTEM + serialized).encode("utf-8"))
        if self.prompt_bytes > PROMPT_BYTES:
            raise RuntimeError("serialized generation input exceeds the pinned prompt budget")
        messages = json.dumps(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": serialized},
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.prompt_sha256 = hashlib.sha256(messages.encode("utf-8")).hexdigest()
        if self.required_context_span is not None:
            matching_spans = [
                (evidence.passage_key, span)
                for evidence in context.evidence
                for span in evidence.spans
                if span.span_key == self.required_context_span
            ]
            wire_context = json.loads(serialized)
            wire_spans = {
                (evidence["passage_key"], span["span_key"]): span["text"]
                for evidence in wire_context["evidence"]
                for span in evidence["spans"]
            }
            if (
                len(matching_spans) != 1
                or wire_spans.get((matching_spans[0][0], self.required_context_span))
                != matching_spans[0][1].text
            ):
                raise RuntimeError("required canonical span is absent from serialized Ollama input")
            self.required_span_verified = True
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
        self.capture_dir = private_root / uuid4().hex
        self.capture_dir.mkdir(parents=True, exist_ok=False)
        (self.capture_dir / "system.txt").write_bytes(SYSTEM.encode("utf-8"))
        (self.capture_dir / "user.json").write_bytes(serialized.encode("utf-8"))
        attempt = self.delegate.generate(context)
        (self.capture_dir / "response.json").write_bytes(attempt.raw_output.encode("utf-8"))
        return attempt


def public_receipt(result: StructuredSynthesisResult, *, question: str) -> dict[str, object]:
    graph = result.graph
    return {
        "schema_revision": "res147-public-receipt-v2",
        "question": question,
        "status": result.status,
        "question_identity": result.question_identity,
        "retrieval_identity": result.retrieval_identity,
        "retrieval_response_identity": result.retrieval_response_identity,
        "selection_identity": result.selection_identity,
        "selection_trace": [d.model_dump(mode="json") for d in result.selection_trace],
        "evidence_set_identity": result.evidence_set_identity,
        "evidence_set_sufficiency": result.evidence_set.sufficiency
        if result.evidence_set
        else None,
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
        "claim_origins": [claim.origin for claim in graph.claims] if graph else [],
        "claim_review_states": [claim.review_state.value for claim in graph.claims]
        if graph
        else [],
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
        "evidence_link_review_states": [link.review_state.value for link in graph.links]
        if graph
        else [],
        "proposed_claim_count": len(result.proposed_claims),
        "missing_evidence_count": len(result.missing_evidence),
        "limitations": result.limitations,
        "context_bytes": result.context_bytes,
        "context_exclusions": result.context_exclusions,
        "context_coverage": result.context_coverage.model_dump(mode="json")
        if result.context_coverage
        else None,
        "generation_invoked": result.generation_invoked,
        "diagnostics": result.diagnostics,
        "automatic_persistence": False,
        "scientific_approval": False,
        "candidate_database_identity_before_after": None,
        "source_files_identity_before_after": None,
        "qdrant_identity_before_after": None,
        "benchmark_identity_before_after": None,
    }


def main() -> None:  # noqa: PLR0915 -- operator run brackets protected identities before output
    sys.path.insert(0, str(ROOT))
    from scripts.res145_evidence_demo import database_receipt, qdrant_receipt
    from scripts.res146_evidence_selection import protected_benchmark_files

    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default=QUESTION)
    parser.add_argument("--require-context-span")
    args = parser.parse_args()
    if args.require_context_span is not None and (
        len(args.require_context_span) != 64
        or any(character not in "0123456789abcdef" for character in args.require_context_span)
    ):
        parser.error("--require-context-span must be a lowercase SHA-256 key")
    settings = load_settings()
    require_isolated_targets(settings)
    before_sources = source_files_receipt(settings.scientific_evidence_directory)
    engine = create_read_only_candidate_engine(settings)
    retrieval = ScientificRetrievalService(engine, settings)
    client = OllamaClient(str(settings.scientific_ollama_url))
    try:
        model = client.describe()
        embedding_model = retrieval.provider.describe()
        database = urlsplit(str(settings.database_url))
        with engine.raw_connection() as raw_connection:
            require_test_cluster_identity(
                cast(PsycopgConnection[Any], raw_connection.driver_connection),
                settings.test_cluster_system_identifier,
                expected_database="dynamisrag_test_res147_candidate",
                expected_user=database.username,
            )
        before_database_receipt = database_receipt(engine)
        before_database = evidence_identity("res147-canonical-db-v1", before_database_receipt)
        before_qdrant = qdrant_receipt()
        before_benchmark = protected_benchmark_files()
        retrieval.start()
        if retrieval.projection is None or retrieval.adapter is None:
            raise RuntimeError(f"scientific retrieval unavailable: {retrieval.unavailable_code}")
        projection = retrieval.projection
        selection = SelectionRequest(
            question=args.question,
            corpus=RetrievalIdentity(
                embedding_manifest_sha256=projection.embeddings.manifest_sha256,
                projection_sha256=projection.passages.projection_sha256,
            ),
        )
        provider = CapturingProvider(
            OllamaGenerationProvider(client), required_context_span=args.require_context_span
        )
        service = StructuredSynthesisService(retrieval, provider)
        synthesis_started = perf_counter()
        result = service.synthesize(SynthesisRequest(selection=selection))
        synthesis_wall_clock_ms = (perf_counter() - synthesis_started) * 1000
        if args.require_context_span is not None and not provider.required_span_verified:
            raise RuntimeError("required source span was not confirmed before generation")
        after_database_receipt = database_receipt(engine)
        after_database = evidence_identity("res147-canonical-db-v1", after_database_receipt)
        after_sources = source_files_receipt(settings.scientific_evidence_directory)
        after_qdrant = qdrant_receipt()
        after_benchmark = protected_benchmark_files()
        if (before_database, before_sources, before_qdrant, before_benchmark) != (
            after_database,
            after_sources,
            after_qdrant,
            after_benchmark,
        ):
            raise RuntimeError("protected scientific data changed during synthesis")
        receipt = public_receipt(result, question=args.question)
        receipt.update(
            {
                "observed_generation_model": model.model_dump(mode="json"),
                "observed_embedding_model": {
                    "provider": embedding_model.provider,
                    "protocol_revision": embedding_model.protocol_revision,
                    "runtime_version": embedding_model.runtime_version,
                    "runtime_sha": embedding_model.runtime_sha,
                    "runtime_docker_label": embedding_model.runtime_docker_label,
                    "model_id": embedding_model.model_id,
                    "model_sha": embedding_model.model_sha,
                    "model_dtype": embedding_model.model_dtype,
                    "model_pooling": embedding_model.model_pooling,
                    "max_input_length": embedding_model.max_input_length,
                    "max_client_batch_size": embedding_model.max_client_batch_size,
                    "max_batch_tokens": embedding_model.max_batch_tokens,
                    "max_batch_requests": embedding_model.max_batch_requests,
                    "deployment": dict(embedding_model.deployment.payload()),
                },
                "candidate_database_identity_before_after": [before_database, after_database],
                "candidate_database": {
                    "database": "dynamisrag_test_res147_candidate",
                    "session_user": "dynamisrag_test",
                    "cluster_system_identifier": settings.test_cluster_system_identifier,
                },
                "candidate_database_receipt_before_after": [
                    before_database_receipt,
                    after_database_receipt,
                ],
                "source_files_identity_before_after": [before_sources, after_sources],
                "qdrant_identity_before_after": [before_qdrant, after_qdrant],
                "benchmark_identity_before_after": [before_benchmark, after_benchmark],
                "synthesis_wall_clock_ms": synthesis_wall_clock_ms,
                "private_capture_id": provider.capture_dir.name if provider.capture_dir else None,
                "model_prompt": {
                    "messages_sha256": provider.prompt_sha256,
                    "message_bytes": provider.prompt_bytes,
                    "required_context_span_key": args.require_context_span,
                    "required_context_span_verified_before_inference": (
                        provider.required_span_verified
                    ),
                    "private_capture_created": provider.capture_dir is not None,
                },
            }
        )
        print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    finally:
        client.close()
        retrieval.close()
        engine.dispose()


if __name__ == "__main__":
    main()
