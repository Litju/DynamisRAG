"""Attest and migrate the fresh RES-638 research-workflow stores. Imports NO papers.

    docker compose --env-file .env.research -f compose.research-workflow.yaml up -d --wait
    uv run python -m scripts.res638_research_bootstrap

The canonical corpus stays empty: the acceptance run must discover and ingest its own evidence.
The command refuses any container outside the new Compose project and any non-empty Qdrant.
"""

from __future__ import annotations

import json
import os
import subprocess
from argparse import Namespace

import httpx
import psycopg
from alembic import command
from alembic.config import Config
from dotenv import dotenv_values

PROJECT = "dynamisrag-res638-research"


def _attest(container: str) -> None:
    labels = json.loads(
        subprocess.check_output(  # noqa: S603 - fixed operator command, no shell
            ["docker", "inspect", container, "--format", "{{json .Config.Labels}}"],  # noqa: S607
            text=True,
        )
    )
    if labels.get("com.docker.compose.project") != PROJECT:
        raise RuntimeError(f"Refusing {container}: not in the new {PROJECT} project")


def main() -> None:
    for service in ("canonical", "agent-state", "qdrant", "tei"):
        _attest(f"{PROJECT}-{service}-1")
    values = dotenv_values(".env.research")
    owner = values.get("RESEARCH_OWNER_PASSWORD") or ""
    if len(owner) < 32 or not all(c in "0123456789abcdef" for c in owner):
        raise ValueError("Use a generated hex owner password in .env.research")
    url = f"postgresql://workspace_owner:{owner}@127.0.0.1:55448/workspace"
    os.environ["DYNAMISRAG_DATABASE_URL"] = url
    os.environ["DYNAMISRAG_AGENT_ENABLED"] = "false"
    os.environ.setdefault("DYNAMISRAG_OPENSEARCH_PASSWORD", "not-used-by-bootstrap")
    os.environ.setdefault("DYNAMISRAG_OPENSEARCH_URL", "https://127.0.0.1:1")
    with psycopg.connect(url) as connection:
        row = connection.execute(
            "SELECT oid FROM pg_database WHERE datname = current_database()"
        ).fetchone()
    if row is None:
        raise RuntimeError("canonical database missing")
    config = Config("alembic.ini")
    config.cmd_opts = Namespace(x=[f"maintenance_target=workspace:{row[0]}"])
    command.upgrade(config, "head")
    with psycopg.connect(url) as connection:
        documents = connection.execute("SELECT count(*) FROM document_version").fetchone()
    qdrant = httpx.get(
        "http://127.0.0.1:16346/collections",
        headers={"api-key": values.get("RESEARCH_QDRANT_KEY") or ""},
        timeout=10,
    ).json()["result"]["collections"]
    print(  # noqa: T201
        json.dumps(
            {
                "project": PROJECT,
                "canonical_document_versions": documents[0] if documents else None,
                "qdrant_collections": qdrant,
            }
        )
    )


if __name__ == "__main__":
    main()
