# Migration safety boundary

Online Alembic entrypoints check the live PostgreSQL database, owner, session
identity, role flags, marker, and direct maintenance-role membership before
`context.configure` or migration DDL. The check and migration run share one
transaction. Offline-generated SQL starts with the same database-side check, so
executing that file does not bypass the online authorization policy.

## Disposable test databases

Integration tests use `DYNAMISRAG_TEST_DATABASE_URL`, which must authenticate as
`dynamisrag_test` to the `postgres` database on a separate test cluster. The
role is `LOGIN CREATEDB NOINHERIT NOSUPERUSER NOCREATEROLE NOREPLICATION
NOBYPASSRLS`. It cannot connect to the application database. A helper creates a
randomly named database, records its OID and a random database-comment marker,
verifies the exact owner and marker after migration, and drops only that exact
database when no other sessions remain. If any identity check differs, cleanup
leaves the database in place and fails loudly.

The marker is not a security token. The trust boundary is that the test role is
not a superuser, cannot create roles or grant itself privileges, and has no
CONNECT privilege to the application database. It may mark only databases it
creates and owns on the configured test cluster.

`DYNAMISRAG_TEST_ADMIN_DATABASE_URL` is used only by the idempotent provisioning
command. It must connect as a PostgreSQL administrator to `postgres` on that
same isolated test cluster. Provisioning refuses a non-connectable application
database, an app/test DSN collision, an unsafe pre-existing role, or an unmanaged
role-name conflict. It never changes `ALLOW_CONNECTIONS`, runs migrations, or
connects to the application database. Run `scripts/init-test-role.sh` or
`uv run --locked python scripts/provision_test_role.py` after the PostgreSQL
cluster is ready; the command works on both freshly initialized and existing
test clusters.

## Explicit maintenance

The exact name/OID argument is an identity check, not authorization. An online
maintenance migration is allowed only when all of the following hold:

- `current_user`, `session_user`, and the database owner are the same role.
- The role is not a superuser, cannot create roles or databases, cannot
  replicate, and cannot bypass row security.
- The role is a direct member, without `ADMIN OPTION`, of the no-login
  `dynamisrag_migration_operators` group.
- The supplied `maintenance_target` exactly matches the connected database name
  and OID.

A PostgreSQL DBA creates the group and grants membership to a dedicated
non-superuser owner login out of band. The test-role provisioner creates the
empty group if needed and never grants it to the test role. A test runner with
the test DSN cannot obtain maintenance authority by supplying a CLI argument.

Example DBA provisioning SQL:

```sql
CREATE ROLE dynamisrag_migration_operators NOLOGIN NOSUPERUSER NOCREATEDB
    NOCREATEROLE NOREPLICATION NOBYPASSRLS NOINHERIT;
GRANT dynamisrag_migration_operators TO dynamisrag_owner_login;
```

The member login must own the target database and remain non-superuser without
`CREATEDB` or `CREATEROLE`. Superusers remain outside the migration path; a
human DBA can still run raw SQL directly, which is outside Alembic's enforcement
boundary.

Example after the DBA has provisioned the dedicated owner login:

```powershell
uv run alembic -x "maintenance_target=$databaseName`:$databaseOid" upgrade head
```

The operator must obtain the current name and OID from the intended target and
use the dedicated owner login. Passing a matching string from the test role
still fails before any DDL.
