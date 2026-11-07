#!/bin/sh
set -eu

# This runs only for the fresh `dynamisrag-research-edition` PostgreSQL volume.
psql --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
    --set ON_ERROR_STOP=1 \
    --set owner_password="$DYNAMISRAG_RESEARCH_OWNER_PASSWORD" <<'SQL'
CREATE ROLE dynamisrag_migration_operators
    NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS NOINHERIT;
CREATE ROLE dynamisrag_owner
    LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS NOINHERIT
    PASSWORD :'owner_password';
GRANT dynamisrag_migration_operators TO dynamisrag_owner;
CREATE DATABASE dynamisrag_research OWNER dynamisrag_owner;
SQL
