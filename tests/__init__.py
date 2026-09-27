"""Test suite for DynamisRAG.

Split into two halves:

* ``tests/unit`` -- no infrastructure. Runs under plain ``uv run pytest``.
* ``tests/integration`` -- requires the live PostgreSQL and OpenSearch services
  from ``docker compose up -d``. Runs under ``uv run pytest -m integration``.
"""
