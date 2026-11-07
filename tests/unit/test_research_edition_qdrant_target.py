"""Fail-closed attestation of the only Qdrant stacks the operator may mutate."""

from __future__ import annotations

import json
import subprocess
from typing import Any
from unittest.mock import Mock

import pytest
import scripts.real_scientific_retrieval as operator

from dynamisrag.config import Settings
from tests._support import build_settings

_RETAINED_ENDPOINT = "http://127.0.0.1:6333"
_RESEARCH_ENDPOINT = "http://127.0.0.1:16334"
_RESTORE_ENDPOINT = "http://127.0.0.1:16335"
_INTEGRATION_ENDPOINT = "http://127.0.0.1:16333"
_ALL_INTERFACES = ".".join(("0", "0", "0", "0"))


def settings_for(url: str) -> Settings:
    return build_settings().model_copy(update={"scientific_qdrant_url": url})


def container_for(target_name: str = "research-edition") -> dict[str, Any]:
    target = operator.QDRANT_TARGETS[target_name]
    return {
        "Name": f"/{target.container_name}",
        "State": {"Running": True},
        "Config": {
            "Image": target.image,
            "Labels": {
                "com.docker.compose.project": target.compose_project,
                "com.docker.compose.service": "qdrant",
            },
        },
        "Mounts": [
            {"Destination": "/qdrant/storage", "Name": target.data_volume},
            {"Destination": "/qdrant/snapshots", "Name": target.snapshots_volume},
        ],
        "HostConfig": {
            "PortBindings": {
                "6333/tcp": [{"HostIp": "127.0.0.1", "HostPort": str(target.host_port)}]
            }
        },
    }


def find_docker(command: str, mode: int = 0o777) -> str | None:
    return "docker" if command == "docker" and mode else None


def docker_run(container: dict[str, Any]) -> Any:
    def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        output = json.dumps([container])
        return subprocess.CompletedProcess(command, 0, stdout=output)

    return run


def install_docker(monkeypatch: pytest.MonkeyPatch, container: dict[str, Any]) -> None:
    monkeypatch.setattr(operator.shutil, "which", find_docker)
    monkeypatch.setattr(operator.subprocess, "run", docker_run(container))


def mutate(container: dict[str, Any], mutation: str) -> None:
    if mutation == "project":
        container["Config"]["Labels"]["com.docker.compose.project"] = "dynamisrag"
    elif mutation == "service":
        container["Config"]["Labels"]["com.docker.compose.service"] = "opensearch"
    elif mutation == "name":
        container["Name"] = "/dynamisrag-qdrant"
    elif mutation == "image":
        container["Config"]["Image"] = "qdrant/qdrant:latest"
    elif mutation == "volume":
        container["Mounts"][0]["Name"] = "dynamisrag_qdrant-data"
    elif mutation == "retained-port":
        container["HostConfig"]["PortBindings"]["6333/tcp"] = [
            {"HostIp": "127.0.0.1", "HostPort": "6333"}
        ]
    elif mutation == "all-interfaces":
        container["HostConfig"]["PortBindings"]["6333/tcp"] = [
            {"HostIp": _ALL_INTERFACES, "HostPort": "16334"}
        ]
    elif mutation == "stopped":
        container["State"]["Running"] = False
    else:  # pragma: no cover - guards a typo in the parametrization
        raise AssertionError(f"unknown mutation: {mutation}")


@pytest.mark.parametrize(
    "url",
    [
        _RETAINED_ENDPOINT,
        "http://127.0.0.1:16334.evil.example",
        "http://127.0.0.1:16334/extra",
        "http://127.0.0.1",
        "https://127.0.0.1:16334",
    ],
)
def test_retained_or_ambiguous_endpoint_is_rejected_before_docker_or_client(
    url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    docker = Mock()
    monkeypatch.setattr(operator, "verify_qdrant_container", docker)

    with pytest.raises(RuntimeError, match="not the attested research-edition target"):
        operator.require_qdrant_target(settings_for(url), "research-edition")

    docker.assert_not_called()


def test_retained_endpoint_6333_rejects_retrieve_before_any_qdrant_client_or_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Mock(side_effect=AssertionError("a retained target must not get a client"))
    projector = Mock(side_effect=AssertionError("no projector may be constructed"))
    monkeypatch.setattr(
        operator, "load_settings", Mock(return_value=settings_for(_RETAINED_ENDPOINT))
    )
    monkeypatch.setattr(operator, "QdrantClient", client)
    monkeypatch.setattr(operator, "QdrantProjector", projector)

    with pytest.raises(RuntimeError, match="not the attested research-edition target"):
        operator.retrieve("0" * 64)

    client.assert_not_called()
    projector.assert_not_called()


@pytest.mark.parametrize(
    "mutation",
    ["project", "service", "name", "image", "volume", "retained-port", "all-interfaces", "stopped"],
)
def test_each_compose_identity_element_is_required(
    mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    container = container_for()
    mutate(container, mutation)
    install_docker(monkeypatch, container)

    with pytest.raises(RuntimeError, match="not the attested research-edition service"):
        operator.require_qdrant_target(settings_for(_RESEARCH_ENDPOINT), "research-edition")


def test_missing_or_ambiguous_docker_identity_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(operator.shutil, "which", Mock(return_value=None))
    with pytest.raises(RuntimeError, match="Docker is required"):
        operator.require_qdrant_target(settings_for(_RESEARCH_ENDPOINT), "research-edition")

    install_docker(monkeypatch, container_for())
    monkeypatch.setattr(
        operator.subprocess,
        "run",
        Mock(return_value=subprocess.CompletedProcess([], 0, stdout="[]")),
    )
    with pytest.raises(RuntimeError, match="ambiguous"):
        operator.require_qdrant_target(settings_for(_RESEARCH_ENDPOINT), "research-edition")


@pytest.mark.parametrize(
    ("target_name", "url"),
    [
        ("research-edition", _RESEARCH_ENDPOINT),
        ("research-edition-restore", _RESTORE_ENDPOINT),
        ("integration", _INTEGRATION_ENDPOINT),
    ],
)
def test_attested_isolated_targets_are_accepted(
    target_name: str, url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_docker(monkeypatch, container_for(target_name))

    target = operator.require_qdrant_target(settings_for(url), target_name)

    assert target.name == target_name
    assert target.endpoint == url


def test_unknown_target_profile_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    install_docker(monkeypatch, container_for())
    with pytest.raises(RuntimeError, match="unknown Qdrant target profile"):
        operator.require_qdrant_target(settings_for(_RESEARCH_ENDPOINT), "default")


def test_retained_stack_is_not_in_the_allowlist() -> None:
    endpoints = {target.endpoint for target in operator.QDRANT_TARGETS.values()}
    assert _RETAINED_ENDPOINT not in endpoints
    assert {target.compose_project for target in operator.QDRANT_TARGETS.values()} == {
        "dynamisrag-research-edition",
        "dynamisrag-research-restore",
        "dynamisrag-integration",
    }
