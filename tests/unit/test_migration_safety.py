from __future__ import annotations

from dataclasses import replace

import pytest

from dynamisrag.db.migration_safety import (
    MAINTENANCE_ROLE,
    TEST_DATABASE_ROLE,
    DatabaseTargetIdentity,
    migration_target_is_allowed,
)
from dynamisrag.db.test_cluster import (
    TEST_CLUSTER_IDENTITY_PREFIX,
)
from dynamisrag.db.test_cluster import (
    test_cluster_identity_comment as cluster_identity_comment,
)


def _disposable() -> DatabaseTargetIdentity:
    return DatabaseTargetIdentity(
        database="dynamisrag_test_0123456789abcdef0123456789abcdef",
        oid=41001,
        current_user=TEST_DATABASE_ROLE,
        session_user=TEST_DATABASE_ROLE,
        owner=TEST_DATABASE_ROLE,
        is_superuser=False,
        can_create_database=True,
        can_create_role=False,
        can_replicate=False,
        can_bypass_rls=False,
        comment="dynamisrag-disposable:0123456789abcdef0123456789abcdef",
        direct_maintenance_member=False,
        test_cluster_identity_comment=cluster_identity_comment("765432109876543210"),
    )


def test_marked_test_role_owns_the_disposable_database() -> None:
    assert migration_target_is_allowed(_disposable(), None, "765432109876543210")


@pytest.mark.parametrize(
    "target",
    [
        replace(_disposable(), current_user="other", session_user="other", owner="other"),
        replace(_disposable(), session_user="other"),
        replace(_disposable(), owner="other"),
        replace(_disposable(), is_superuser=True),
        replace(_disposable(), can_create_database=False),
        replace(_disposable(), can_create_role=True),
        replace(_disposable(), can_replicate=True),
        replace(_disposable(), can_bypass_rls=True),
        replace(_disposable(), direct_maintenance_member=True),
        replace(_disposable(), comment=None),
        replace(_disposable(), comment="dynamisrag-disposable:not-a-random-marker"),
        replace(_disposable(), database="dynamisrag"),
        replace(_disposable(), test_cluster_identity_comment=None),
        replace(
            _disposable(),
            test_cluster_identity_comment=f"{TEST_CLUSTER_IDENTITY_PREFIX}123456789012345678",
        ),
    ],
    ids=(
        "wrong-test-role",
        "set-role-session",
        "wrong-owner",
        "superuser",
        "cannot-create-disposable-database",
        "can-create-roles",
        "replication-role",
        "bypass-rls",
        "maintenance-membership-on-test-role",
        "missing-marker",
        "malformed-marker",
        "not-a-disposable-name",
        "unmarked-cluster",
        "mismatched-cluster-identity",
    ),
)
def test_test_database_authorization_fails_closed(target: DatabaseTargetIdentity) -> None:
    assert not migration_target_is_allowed(target, None)


def test_maintenance_target_string_alone_is_not_authorization() -> None:
    target = replace(
        _disposable(),
        database="dynamisrag_reference_decoy",
        current_user="dynamisrag_decoy_owner",
        session_user="dynamisrag_decoy_owner",
        owner="dynamisrag_decoy_owner",
        can_create_database=False,
        comment=None,
    )

    assert not migration_target_is_allowed(target, "dynamisrag_reference_decoy:41001")


def test_maintenance_requires_direct_dba_granted_membership_and_exact_identity() -> None:
    target = replace(
        _disposable(),
        database="dynamisrag_reference_decoy",
        current_user="dynamisrag_decoy_owner",
        session_user="dynamisrag_decoy_owner",
        owner="dynamisrag_decoy_owner",
        can_create_database=False,
        comment=None,
        direct_maintenance_member=True,
    )

    assert migration_target_is_allowed(target, "dynamisrag_reference_decoy:41001")
    assert not migration_target_is_allowed(target, "dynamisrag_reference_decoy:41002")
    assert not migration_target_is_allowed(
        replace(target, is_superuser=True), "dynamisrag_reference_decoy:41001"
    )
    assert not migration_target_is_allowed(
        replace(target, can_create_role=True), "dynamisrag_reference_decoy:41001"
    )
    assert not migration_target_is_allowed(
        replace(target, can_create_database=True), "dynamisrag_reference_decoy:41001"
    )
    assert not migration_target_is_allowed(
        replace(target, direct_maintenance_member=False), "dynamisrag_reference_decoy:41001"
    )


def test_maintenance_group_is_a_distinct_named_authority() -> None:
    assert MAINTENANCE_ROLE != TEST_DATABASE_ROLE
