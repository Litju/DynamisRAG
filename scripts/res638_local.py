"""Operate the isolated RES-638 owner workspace without recreating its stores."""

# ruff: noqa: T201 - operator output

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
DATA = Path.home() / ".dynamisrag" / "res638-local"
ENV_FILE = DATA / "settings.env"
PROJECT = "dynamisrag-res638-local"
VOLUMES = tuple(f"{PROJECT}-{name}" for name in ("canonical", "agent-state", "vectors"))
COMPOSE = [
    "docker",
    "compose",
    "--project-name",
    PROJECT,
    "--env-file",
    str(ENV_FILE),
    "-f",
    "compose.research-workflow.yaml",
    "-f",
    "compose.research-local.yaml",
]
MIN_DISK = 5 * 1024**3
MIN_RAM = 4 * 1024**3
INFRA_IMAGE = "postgres:18.6-alpine"
CACHE_MODEL = (
    "/data/models--Qwen--Qwen3-Embedding-0.6B/snapshots/"
    "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/model.safetensors"
)


def command(*args: str, capture: bool = False) -> str:
    # Interpolation comes from the explicit local file, never from a stale dev shell.
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith(
            ("COMPOSE_", "RESEARCH_", "LOCAL_WORKSPACE_", "DYNAMISRAG_", "WORKSPACE_")
        )
    }
    env["RES638_OWNER_DATA_DIR"] = DATA.as_posix()
    result = subprocess.run(  # noqa: S603 - fixed operator commands, no shell
        args,
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout.strip() if capture else ""


def compose(*args: str, capture: bool = False) -> str:
    return command(*COMPOSE, *args, capture=capture)


def configuration() -> tuple[str, str, str]:
    if not ENV_FILE.is_file():
        raise ValueError("Missing owner settings.env. See docs/res638-local-operations.md.")
    values = dotenv_values(ENV_FILE, interpolate=False)
    for key in (
        "RESEARCH_ADMIN_PASSWORD",
        "RESEARCH_OWNER_PASSWORD",
        "RESEARCH_AGENT_PASSWORD",
        "RESEARCH_READER_PASSWORD",
        "RESEARCH_QDRANT_KEY",
        "DYNAMISRAG_AGENT_IDENTITY_SECRET",
        "WORKSPACE_SESSION_SECRET",
    ):
        if not re.fullmatch(r"[0-9a-f]{32,128}", values.get(key) or ""):
            raise ValueError(f"{key} must be a generated hex secret of at least 32 characters.")
    accounts: object = json.loads(values.get("WORKSPACE_ACCOUNTS") or "{}")
    if not isinstance(accounts, dict) or len(accounts) != 1:
        raise ValueError("WORKSPACE_ACCOUNTS must contain one owner account.")
    user, credential = next(iter(accounts.items()))
    if not (
        isinstance(user, str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._@-]{0,127}", user)
        and isinstance(credential, str)
        and re.fullmatch(r"[0-9a-f]{32}:[0-9a-f]{128}", credential)
    ):
        raise ValueError(
            "Generate the owner password hash with apps/workspace/scripts/account.mjs."
        )
    cache = values.get("RESEARCH_TEI_CACHE_VOLUME")
    if not cache or values.get("RESEARCH_TEI_CACHE_EXTERNAL") != "true":
        raise ValueError(
            "Set RESEARCH_TEI_CACHE_VOLUME to the existing external Qwen3 model cache."
        )
    approved = values.get("DYNAMISRAG_AGENT_OPENCODE_GO_USAGE_APPROVED") or "false"
    if approved not in ("true", "false"):
        raise ValueError("Set DYNAMISRAG_AGENT_OPENCODE_GO_USAGE_APPROVED to true or false.")
    if approved == "true" and not values.get("DYNAMISRAG_AGENT_MODEL_API_KEY"):
        raise ValueError("The worker requires a server-side API key when inference is approved.")
    budget = values.get("DYNAMISRAG_AGENT_RUN_COST_BUDGET_USD") or "0.20"
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", budget) or not 0 < float(budget) <= 0.20:
        raise ValueError("The local per-run inference budget must be above $0 and at most $0.20.")
    tokens = values.get("DYNAMISRAG_AGENT_RUN_TOKEN_BUDGET") or "250000"
    if not tokens.isdecimal() or not 1_000 <= int(tokens) <= 250_000:
        raise ValueError("The local per-run token budget must be between 1,000 and 250,000.")
    port = values.get("LOCAL_WORKSPACE_PORT") or "3040"
    if not port.isdecimal() or not 1024 <= int(port) <= 65535:
        raise ValueError("LOCAL_WORKSPACE_PORT must be between 1024 and 65535.")
    cost = (
        f"Paid DeepSeek V4.1 Flash (estimated cap US${float(budget):.2f}/run)"
        if approved == "true"
        else "Inference disabled"
    )
    return f"http://127.0.0.1:{port}", cost, cache


def preflight(cache: str) -> None:
    if not (DATA / "seeded").is_file() or not all(
        (DATA / name).is_dir() for name in ("corpus/projections", "corpus/sources", "eve")
    ):
        raise ValueError("Owner research data are missing. See docs/res638-local-operations.md.")
    command("docker", "volume", "inspect", *VOLUMES, cache, capture=True)
    # Probe the Docker VM and the existing cache without mounting any research volume.
    output = command(
        "docker",
        "run",
        "--rm",
        "--pull",
        "never",
        "--network",
        "none",
        "--mount",
        f"type=volume,src={cache},dst=/data,readonly",
        "--entrypoint",
        "sh",
        INFRA_IMAGE,
        "-ec",
        f"test -r {CACHE_MODEL}; awk '/MemAvailable:/ {{print $2}}' /proc/meminfo; "
        "df -Pk / | awk 'NR == 2 {print $4}'",
        capture=True,
    )
    ram_kib, disk_kib = map(int, output.split())
    host_disk = shutil.disk_usage(DATA).free
    print(
        f"Preflight: Docker available RAM {ram_kib / 1024**2:.1f} GiB, "
        f"Docker free disk {disk_kib / 1024**2:.1f} GiB, host free disk "
        f"{host_disk / 1024**3:.1f} GiB."
    )
    if min(disk_kib * 1024, host_disk) < MIN_DISK:
        raise ValueError("Free at least 5 GiB on both the Docker disk and workspace drive.")
    if ram_kib * 1024 < MIN_RAM and not compose(
        "ps", "-q", "--status", "running", "tei", capture=True
    ):
        raise ValueError(
            "Free at least 4 GiB of available RAM in Docker before starting Qwen3 TEI."
        )


def start(*, build: bool) -> None:
    url, cost, cache = configuration()
    preflight(cache)
    if build:
        compose("build", "api", "workspace")
    compose("up", "-d", "--wait", "--wait-timeout", "600", "--no-build", "--pull", "never")
    print(f"DynamisRAG is ready at {url}\n{cost}.")


def active_runs() -> bool:
    if not compose("ps", "-q", "--status", "running", "agent-state", capture=True):
        return False
    count = compose(
        "exec",
        "-T",
        "agent-state",
        "psql",
        "-U",
        "workspace_admin",
        "-d",
        "workspace",
        "-v",
        "ON_ERROR_STOP=1",
        "-Atqc",
        "SELECT count(*) FROM agent_v2.agent_runs WHERE status IN ('queued', 'running')",
        capture=True,
    )
    return int(count) > 0


def stop() -> None:
    if active_runs():
        raise ValueError(
            "Research is active. Let it finish or cancel it in the browser, then stop."
        )
    frontend = compose("ps", "-q", "--status", "running", "workspace", capture=True)
    if frontend:
        compose("stop", "workspace")
    try:
        if active_runs():
            raise ValueError("Research just started. Let it finish or cancel it, then stop.")
    except (ValueError, subprocess.CalledProcessError):
        if frontend:
            compose("start", "workspace")
        raise
    compose("stop")
    print("DynamisRAG stopped. Papers, vectors and conversation state are on disk.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "stop", "restart", "status", "logs"))
    parser.add_argument(
        "--build", action="store_true", help="rebuild the app images before startup"
    )
    args = parser.parse_args()
    try:
        if args.action == "status":
            url, cost, _ = configuration()
            print(f"{url}\n{cost}.")
            compose("ps", "--all")
        elif args.action == "logs":
            compose("logs", "--tail", "50")
        else:
            if args.action in ("stop", "restart"):
                stop()
            if args.action in ("start", "restart"):
                start(build=args.build)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        if isinstance(error, subprocess.CalledProcessError):
            print(
                "Local stack command failed. Use status and logs to identify the service.",
                file=sys.stderr,
            )
        else:
            print(f"Local stack: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
