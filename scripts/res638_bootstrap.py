"""Initialize only the new RES-638 Compose corpus using the existing licensed importer."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
from argparse import Namespace
from pathlib import Path

import psycopg
from alembic import command
from alembic.config import Config
from dotenv import dotenv_values
from scripts.real_scientific_retrieval import ingest_sample


def main() -> None:
    labels = json.loads(
        subprocess.check_output(
            [  # noqa: S607 -- fixed operator command, no shell
                "docker",
                "inspect",
                "dynamisrag-res638-canonical-1",
                "--format",
                "{{json .Config.Labels}}",
            ],
            text=True,
        )
    )
    if labels.get("com.docker.compose.project") != "dynamisrag-res638":
        raise RuntimeError("Refusing any container outside the new RES-638 project")
    values = dotenv_values(".env.workspace")
    password = values.get("WORKSPACE_DB_PASSWORD")
    if not password or not all(c in "0123456789abcdef" for c in password):
        raise ValueError("Use a generated hex owner password in .env.workspace")
    for key, value in values.items():
        if key.startswith("DYNAMISRAG_") and value is not None:
            os.environ[key] = value
    owner = f"postgresql://workspace_owner:{password}@127.0.0.1:55438/workspace"
    os.environ["DYNAMISRAG_DATABASE_URL"] = owner
    os.environ["DYNAMISRAG_AGENT_ENABLED"] = "false"
    with psycopg.connect(owner) as connection:
        row = connection.execute(
            "SELECT oid FROM pg_database WHERE datname = current_database()"
        ).fetchone()
    if row is None:
        raise RuntimeError("Workspace database missing")
    config = Config("alembic.ini")
    config.cmd_opts = Namespace(x=[f"maintenance_target=workspace:{row[0]}"])
    command.upgrade(config, "head")
    root = Path(".dynamisrag/res638").resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (
        (root / "ingest.json").open("w", encoding="utf-8") as output,
        contextlib.redirect_stdout(output),
    ):
        ingest_sample(root=root)
    papers = json.loads((root / "papers.json").read_text(encoding="utf-8"))
    if len(papers) != 3 or sum(p["passages"] for p in papers) == 0:
        raise RuntimeError("The licensed sample import is incomplete")
    print(f"RES-638: imported {len(papers)} licensed papers into the new isolated corpus")  # noqa: T201


if __name__ == "__main__":
    main()
