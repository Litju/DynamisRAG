"""The database migration state must match the migration code on disk.

Comparing against ``ScriptDirectory`` rather than a hard-coded revision id means
a new revision that is committed but not applied fails this test, without the
test itself needing editing.
"""

from __future__ import annotations

from typing import Final

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import Engine, text

from dynamisrag.config import Settings
from dynamisrag.db import create_database_engine
from tests._support import ALEMBIC_INI, REPO_ROOT

_SCRIPT_LOCATION: Final[str] = str(REPO_ROOT / "alembic")
_VERSION_TABLE: Final[str] = "alembic_version"
_EXPECTED_POSTGRES_MAJOR: Final[str] = "18"


def _engine(settings: Settings) -> Engine:
    return create_database_engine(settings)


def _script_heads() -> set[str]:
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", _SCRIPT_LOCATION)
    return set(ScriptDirectory.from_config(config).get_heads())


@pytest.mark.integration
def test_alembic_ini_points_at_the_repository_and_omits_the_dsn() -> None:
    """The DSN must come from dynamisrag.config, never from alembic.ini."""
    text_content = ALEMBIC_INI.read_text(encoding="utf-8")

    assert "sqlalchemy.url" not in text_content
    assert "path_separator = os" in text_content


@pytest.mark.integration
def test_database_is_at_the_migration_head(live_settings: Settings) -> None:
    engine = _engine(live_settings)
    try:
        with engine.connect() as connection:
            applied = set(
                connection.execute(text("SELECT version_num FROM alembic_version")).scalars()
            )
            server_version = connection.execute(text("SHOW server_version")).scalar_one()
    finally:
        engine.dispose()

    assert applied == _script_heads()
    assert str(server_version).startswith(_EXPECTED_POSTGRES_MAJOR)


@pytest.mark.integration
def test_baseline_revision_created_no_application_objects(live_settings: Settings) -> None:
    """RES-130 must not pre-empt RES-131: no application tables may exist yet."""
    engine = _engine(live_settings)
    try:
        with engine.connect() as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    text(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
                    )
                )
            }
    finally:
        engine.dispose()

    assert tables == {_VERSION_TABLE}
