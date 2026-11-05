# Migration safety boundary

Online Alembic entrypoints check the live PostgreSQL database, owner, session
identity, role flags, marker, and direct maintenance-role membership before
`context.configure` or migration DDL. The check and migration run share one
transaction. Offline-generated SQL starts with the same database-side check, so
executing that file does not bypass the online authorization policy.

## Positive test-cluster identity

Database names, OIDs, and the `dynamisrag_test` login do not identify a cluster.
Integration runs also require the exact PostgreSQL `system_identifier` generated
by `initdb`. A superuser reads it from `pg_control_system()` on the dedicated
Compose container. The provisioner checks that value before changing any role
or ACL, then records it in the comment of a cluster-wide `NOLOGIN` role named
`dynamisrag_test_cluster_identity`. The role has no elevated flags or
memberships. The test login can read its comment through `pg_roles`, but cannot
create or alter roles; it therefore cannot forge the identity marker. Online and
offline disposable-database migrations compare the marker with
`DYNAMISRAG_TEST_CLUSTER_SYSTEM_IDENTIFIER`.

The trusted boundary is the local operator and the PostgreSQL superuser that
provisions the marker. A superuser can change roles or execute raw SQL outside
Alembic. Keep the expected system identifier with the integration environment,
and do not copy it from the retained development or quarantined cluster.

## Isolated local Compose project

The default `compose.yaml` remains project `dynamisrag`; its implicit
`postgres-data` volume resolves to `dynamisrag_postgres-data`, and its PostgreSQL
port is 55432. Integration tests must use `compose.integration.yaml`, which
selects project `dynamisrag-integration`, volume
`dynamisrag-integration_postgres-data`, and loopback host port 55433. The other
service volumes are namespaced the same way; Qdrant and OpenSearch use ports
16333 and 19200. Integration preflight checks the running container's Compose
project label, mounted PostgreSQL volume, and port binding before opening any
database connection. The three PostgreSQL DSNs must all use `127.0.0.1:55433`;
the test and admin DSNs must connect to `postgres`. Set the OpenSearch URL to
`https://127.0.0.1:19200` and the scientific Qdrant URL to
`http://127.0.0.1:16333`. This rejects the default development endpoints before
a connection is attempted.

From PowerShell in the repository root, start only the isolated project:

```powershell
Copy-Item .env.example .env
$composeArgs = @(
    "--project-name", "dynamisrag-integration",
    "--file", "compose.yaml",
    "--file", "compose.integration.yaml"
)
docker compose @composeArgs up -d --wait
$systemIdentifier = docker compose @composeArgs exec -T postgres psql -U dynamisrag -d postgres -Atc "SELECT system_identifier::text FROM pg_control_system()"
$env:DYNAMISRAG_TEST_CLUSTER_SYSTEM_IDENTIFIER = $systemIdentifier
```

Set `DYNAMISRAG_DATABASE_URL` to `postgresql://dynamisrag:<admin-password>@127.0.0.1:55433/dynamisrag`,
`DYNAMISRAG_TEST_ADMIN_DATABASE_URL` to the same admin login on
`127.0.0.1:55433/postgres`, and `DYNAMISRAG_TEST_DATABASE_URL` to
`dynamisrag_test` on `127.0.0.1:55433/postgres`. The provisioner creates the
least-privilege test login using the test DSN's 64-character hex password. Keep
these settings, `DYNAMISRAG_OPENSEARCH_URL=https://127.0.0.1:19200`, and
`DYNAMISRAG_SCIENTIFIC_QDRANT_URL=http://127.0.0.1:16333` in the current
PowerShell session or ignored `.env` file. Then run:

```powershell
uv run --locked python scripts/provision_test_role.py
uv run pytest -m integration
docker compose @composeArgs stop
```

Do not use plain `docker compose` for the integration stack: it resolves to the
retained `dynamisrag` project. Do not pass `down -v` for either project.

## Disposable test databases

Integration tests use `DYNAMISRAG_TEST_DATABASE_URL`, which must authenticate as
`dynamisrag_test` to the `postgres` database on the attested test cluster. The
role is `LOGIN CREATEDB NOINHERIT NOSUPERUSER NOCREATEROLE NOREPLICATION
NOBYPASSRLS`. It cannot connect to the application database. A helper creates a
randomly named database, records its OID and a random database-comment marker,
verifies the exact owner and marker after migration, and drops only that exact
database when no other sessions remain. If any identity check differs, cleanup
leaves the database in place and fails loudly.

The per-database comment marker is not a security token. The independent cluster
identity is the DBA-provisioned role comment described above. The test role is
not a superuser, cannot create roles or grant itself privileges, and has no
CONNECT privilege to the application database. It may mark only databases it
creates and owns on the configured test cluster.

`DYNAMISRAG_TEST_ADMIN_DATABASE_URL` is used only by the idempotent provisioning
command. It must connect as a PostgreSQL superuser to `postgres` on the same
attested test cluster. Provisioning refuses a missing or mismatched system
identifier, wrong Compose project or volume, non-connectable application
database, an app/test DSN collision, an unsafe pre-existing role, or an unmanaged
role-name conflict. It never changes `ALLOW_CONNECTIONS`, runs migrations, or
connects to the application database. Run `scripts/init-test-role.sh` or
`uv run --locked python scripts/provision_test_role.py` only after the isolated
Compose project is ready and its system identifier is set.

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
