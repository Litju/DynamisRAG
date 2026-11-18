FROM ghcr.io/astral-sh/uv:0.12.4@sha256:d0a6eca6c669dc7e9c51218707b8438a3d30402733d739dcc00adb3e213e8f5c AS uv
FROM python:3.12.13-slim@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-default-groups --no-install-project
COPY src ./src
COPY alembic ./alembic
COPY alembic.ini ./
RUN uv sync --locked --no-default-groups && useradd --uid 10001 --create-home workspace
USER workspace
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["uvicorn", "dynamisrag.application:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
