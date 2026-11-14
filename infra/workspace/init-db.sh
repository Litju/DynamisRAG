#!/bin/sh
set -eu
# Runs only on this new workspace's empty volumes.
psql --username "$POSTGRES_USER" --dbname postgres --set ON_ERROR_STOP=1 \
  --set app_password="$WORKSPACE_DB_PASSWORD" --set reader_password="$WORKSPACE_READER_PASSWORD" <<'SQL'
CREATE ROLE dynamisrag_migration_operators NOLOGIN;
CREATE ROLE workspace_owner LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD :'app_password';
GRANT dynamisrag_migration_operators TO workspace_owner;
CREATE ROLE workspace_reader LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD :'reader_password';
CREATE DATABASE workspace OWNER workspace_owner;
REVOKE ALL ON DATABASE workspace FROM PUBLIC;
GRANT CONNECT ON DATABASE workspace TO workspace_owner, workspace_reader;
SQL
psql --username "$POSTGRES_USER" --dbname workspace --set ON_ERROR_STOP=1 <<'SQL'
GRANT USAGE ON SCHEMA public TO workspace_reader;
ALTER DEFAULT PRIVILEGES FOR ROLE workspace_owner IN SCHEMA public GRANT SELECT ON TABLES TO workspace_reader;
SQL
