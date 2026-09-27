"""Integration tests. Every test here requires the live local stack:

Copy-Item .env.example .env
docker compose up -d --wait
uv run alembic upgrade head
uv run pytest -m integration
"""

from __future__ import annotations
