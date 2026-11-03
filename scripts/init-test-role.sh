#!/bin/sh
set -eu

psql --no-psqlrc --set=ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<'SQL'
\getenv test_user DYNAMISRAG_TEST_POSTGRES_USER
\getenv test_password DYNAMISRAG_TEST_POSTGRES_PASSWORD
\getenv app_user POSTGRES_USER
\getenv app_database POSTGRES_DB

CREATE ROLE :"test_user" LOGIN PASSWORD :'test_password'
    NOSUPERUSER CREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
REVOKE CONNECT ON DATABASE :"app_database" FROM PUBLIC;
GRANT CONNECT ON DATABASE :"app_database" TO :"app_user";
SQL
