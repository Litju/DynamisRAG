"""Pure identity checks for online migration target authorization."""

from __future__ import annotations

from dataclasses import replace

import pytest

from dynamisrag.db.migration_safety import (
    DatabaseTargetIdentity,
    migration_target_is_allowed,
)

_DISPOSABLE = DatabaseTargetIdentity(
    database="random_name_not_a_prefix",
    oid=16384,
    current_user="dynamisrag_test",
    session_user="dynamisrag_test",
    owner="dynamisrag_test",
    is_superuser=False,
    can_create_database=True,
    can_create_role=False,
    comment="dynamisrag-disposable:0123456789abcdef0123456789abcdef",
)


def test_disposable_identity_is_database_owned_and_marked() -> None:
    assert migration_target_is_allowed(_DISPOSABLE, None)
    assert not migration_target_is_allowed(replace(_DISPOSABLE, comment=None), None)
    assert not migration_target_is_allowed(replace(_DISPOSABLE, owner="another_role"), None)
    assert not migration_target_is_allowed(replace(_DISPOSABLE, is_superuser=True), None)
    assert not migration_target_is_allowed(replace(_DISPOSABLE, can_create_role=True), None)


@pytest.mark.parametrize("target", [None, "random_name_not_a_prefix:16385", "other:16384"])
def test_maintenance_requires_exact_database_name_and_oid(target: str | None) -> None:
    protected = replace(_DISPOSABLE, comment=None, is_superuser=True)

    assert not migration_target_is_allowed(protected, target)


def test_explicit_maintenance_target_matches_the_live_owner_and_oid() -> None:
    protected = replace(_DISPOSABLE, comment=None, is_superuser=True)

    assert migration_target_is_allowed(protected, "random_name_not_a_prefix:16384")
    assert not migration_target_is_allowed(
        replace(protected, session_user="different_session"),
        "random_name_not_a_prefix:16384",
    )
    assert not migration_target_is_allowed(
        replace(protected, owner="different_owner"),
        "random_name_not_a_prefix:16384",
    )
