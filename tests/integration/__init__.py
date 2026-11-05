"""Integration tests require isolated live services and a dedicated test DSN.

Provision a separate test cluster and role, then run ``uv run pytest -m integration``.
Each integration test creates, identity-checks, and removes its own database.
"""

from __future__ import annotations
