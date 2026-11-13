# ruff: noqa: S101 -- explicit operator self-checks
"""RES-638 persistence checks against only the newly created workspace Compose project."""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable
from urllib.parse import urlsplit
from uuid import uuid4

from dotenv import dotenv_values
from tests.agent_store_contract import ALL_CHECKS, StoreHarness

from dynamisrag.agent.persistence import PostgresAgentConversationStore


def check_workspace_postgres_contract(check: Callable[[StoreHarness], None]) -> None:
    labels = json.loads(
        subprocess.check_output(
            [  # noqa: S607 -- fixed operator command, no shell
                "docker",
                "inspect",
                "dynamisrag-res638-agent-state-1",
                "--format",
                "{{json .Config.Labels}}",
            ],
            text=True,
        )
    )
    assert labels["com.docker.compose.project"] == "dynamisrag-res638"
    values = dotenv_values(".env.workspace")
    url = values["DYNAMISRAG_AGENT_DATABASE_URL"]
    canonical = values["DYNAMISRAG_DATABASE_URL"]
    assert url is not None and canonical is not None
    parsed = urlsplit(url)
    assert (parsed.hostname, parsed.port, parsed.path) == ("127.0.0.1", 55439, "/workspace")
    store = PostgresAgentConversationStore(
        url, schema_name=f"res638_test_{uuid4().hex}", protected_database_url=canonical
    )
    try:
        check(StoreHarness(store=store, lease_seconds=0.3, expire_leases=lambda: time.sleep(0.5)))
    finally:
        store.close()


if __name__ == "__main__":
    for contract_check in ALL_CHECKS:
        check_workspace_postgres_contract(contract_check)
    print(f"RES-638: {len(ALL_CHECKS)} PostgreSQL contract checks passed")  # noqa: T201
