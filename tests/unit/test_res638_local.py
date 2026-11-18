"""Operator guards for a separate, locally paid research workspace."""

from pathlib import Path

import pytest
from scripts import res638_local


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / ".env.research-local"
    monkeypatch.setattr(res638_local, "ENV_FILE", path)
    secret_keys = (
        "RESEARCH_ADMIN_PASSWORD",
        "RESEARCH_OWNER_PASSWORD",
        "RESEARCH_AGENT_PASSWORD",
        "RESEARCH_READER_PASSWORD",
        "RESEARCH_QDRANT_KEY",
        "DYNAMISRAG_AGENT_IDENTITY_SECRET",
        "WORKSPACE_SESSION_SECRET",
    )
    path.write_text(
        "\n".join(f"{key}={'a' * 32}" for key in secret_keys)
        + f'\nWORKSPACE_ACCOUNTS=\'{{"owner":"{"a" * 32}:{"b" * 128}"}}\'\n'
        + "RESEARCH_TEI_CACHE_VOLUME=existing-cache\nRESEARCH_TEI_CACHE_EXTERNAL=true\n",
        encoding="utf-8",
    )
    return path


def test_paid_inference_is_opt_in_and_bounded(config: Path) -> None:
    assert res638_local.configuration() == (
        "http://127.0.0.1:3040",
        "Inference disabled",
        "existing-cache",
    )
    with config.open("a", encoding="utf-8") as file:
        file.write("DYNAMISRAG_AGENT_OPENCODE_GO_USAGE_APPROVED=true\n")
    with pytest.raises(ValueError, match="server-side API key"):
        res638_local.configuration()
    with config.open("a", encoding="utf-8") as file:
        file.write(
            "DYNAMISRAG_AGENT_MODEL_API_KEY=example\nDYNAMISRAG_AGENT_RUN_COST_BUDGET_USD=0.21\n"
        )
    with pytest.raises(ValueError, match="per-run inference budget"):
        res638_local.configuration()


def test_cannot_stop_while_research_is_active(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(res638_local, "active_runs", lambda: True)
    with pytest.raises(ValueError, match="Research is active"):
        res638_local.stop()
