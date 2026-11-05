#!/bin/sh
set -eu

# Run explicitly after PostgreSQL is ready. docker-entrypoint-initdb.d is only
# visited for new PGDATA and cannot repair an already initialized cluster.
# The provisioner verifies the dedicated Compose project, volume, port, and
# PostgreSQL system identifier before changing roles or database ACLs.
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"
exec uv run --locked python scripts/provision_test_role.py
