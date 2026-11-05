from __future__ import annotations

import json
import subprocess

import pytest
import scripts.res147_structured_synthesis as res147

from dynamisrag.config import Settings
from dynamisrag.generation.synthesis import StructuredSynthesisResult

_SETTINGS = {
    "database_url": "postgresql://dynamisrag_test:pw@127.0.0.1:55433/dynamisrag_test_res147_candidate",
    "opensearch_url": "https://127.0.0.1:19200",
    "opensearch_password": "test-password",
    "scientific_qdrant_url": "http://127.0.0.1:16333",
    "scientific_tei_url": "http://127.0.0.1:8081",
    "scientific_ollama_url": "http://127.0.0.1:11434",
    "test_cluster_system_identifier": "765432109876543210",
}
_ALL_INTERFACES = ".".join(("0", "0", "0", "0"))


def test_res147_preflight_allows_only_attested_loopback_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(res147, "verify_test_compose_resources", lambda: calls.append(True))
    monkeypatch.setattr(res147, "verify_test_qdrant_resources", lambda: calls.append(True))

    res147.require_isolated_targets(Settings.from_mapping(_SETTINGS))

    assert calls == [True, True]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("database_url", "postgresql://candidate:pw@127.0.0.1:55432/dynamisrag"),
        ("database_url", "postgresql://candidate:pw@db.example:55433/dynamisrag"),
        ("database_url", "postgresql://dynamisrag:pw@127.0.0.1:55433/dynamisrag"),
        ("scientific_qdrant_url", "http://127.0.0.1:6333"),
        ("scientific_tei_url", "http://127.0.0.1:8080"),
        ("scientific_ollama_url", "https://127.0.0.1:11434"),
        ("opensearch_url", "https://127.0.0.1:9200"),
        ("test_cluster_system_identifier", None),
    ],
)
def test_res147_preflight_rejects_unisolated_targets_before_docker_or_connections(
    field: str, value: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(res147, "verify_test_compose_resources", lambda: calls.append(True))
    monkeypatch.setattr(res147, "verify_test_qdrant_resources", lambda: calls.append(True))
    settings = Settings.from_mapping({**_SETTINGS, field: value})

    with pytest.raises(RuntimeError):
        res147.require_isolated_targets(settings)

    assert calls == []


@pytest.mark.parametrize("host_ip", ["127.0.0.1", _ALL_INTERFACES])
def test_qdrant_container_must_be_bound_only_to_the_integration_loopback(
    host_ip: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def find_docker(command: str, mode: int = 0o777) -> str | None:
        return "docker" if command == "docker" and mode else None

    monkeypatch.setattr(res147.shutil, "which", find_docker)
    container = {
        "State": {"Running": True},
        "Config": {
            "Labels": {
                "com.docker.compose.project": "dynamisrag-integration",
                "com.docker.compose.service": "qdrant",
            }
        },
        "Mounts": [
            {"Destination": "/qdrant/storage", "Name": "dynamisrag-integration_qdrant-data"},
            {
                "Destination": "/qdrant/snapshots",
                "Name": "dynamisrag-integration_qdrant-snapshots",
            },
        ],
        "HostConfig": {"PortBindings": {"6333/tcp": [{"HostIp": host_ip, "HostPort": "16333"}]}},
    }

    def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        output = "container-id\n" if "ps" in command else json.dumps([container])
        return subprocess.CompletedProcess(command, 0, stdout=output)

    monkeypatch.setattr(res147.subprocess, "run", run)

    if host_ip == "127.0.0.1":
        res147.verify_test_qdrant_resources()
    else:
        with pytest.raises(RuntimeError, match="isolated integration"):
            res147.verify_test_qdrant_resources()


def test_public_receipt_keeps_model_missing_evidence_private() -> None:
    result = StructuredSynthesisResult(
        status="INSUFFICIENT_EVIDENCE",
        question_identity="a" * 64,
        retrieval_identity="b" * 64,
        retrieval_response_identity="c" * 64,
        selection_identity="d" * 64,
        selection_trace=(),
        evidence_set=None,
        evidence_set_identity=None,
        graph=None,
        missing_evidence=("PRIVATE_SOURCE_SENTENCE",),
        generation_invoked=False,
    )

    receipt = res147.public_receipt(result, question="a safe question")
    rendered = json.dumps(receipt)

    assert receipt["missing_evidence_count"] == 1
    assert receipt["automatic_persistence"] is False
    assert "PRIVATE_SOURCE_SENTENCE" not in rendered
