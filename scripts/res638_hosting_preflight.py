"""Read-only $0 preview decision gate. Never provisions cloud resources."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast


def validate_hosting(evidence: Mapping[str, object], *, now: datetime | None = None) -> None:
    for field in ("monthly_budget_usd", "estimated_monthly_cost_usd"):
        if type(evidence.get(field)) not in (int, float) or evidence[field] != 0:
            raise ValueError(f"{field} must be exactly zero")
    for field in (
        "paid_inference",
        "automatic_upgrade",
        "opencode_usage_approved",
        "public_launch",
    ):
        if evidence.get(field) is not False:
            raise ValueError(f"{field} must remain false")
    if evidence.get("paid_resources") != []:
        raise ValueError("No Oracle account, paid resource or subscription is authorized")
    for field in ("account_plan_verified", "no_overage_verified", "protection_verified"):
        if evidence.get(field) is not True:
            raise ValueError(f"{field} is unresolved")
    if evidence.get("inference_mode") != "blocked":
        raise ValueError("Hosted inference must remain blocked")
    checked = datetime.fromisoformat(str(evidence.get("checked_at_utc", "")))
    current = now or datetime.now(UTC)
    if checked.tzinfo is None or not timedelta(0) <= current - checked <= timedelta(hours=24):
        raise ValueError("Free-plan evidence must be at most 24 hours old")
    _validate_provider(evidence)


def _validate_provider(evidence: Mapping[str, object]) -> None:
    provider = evidence.get("provider")
    if provider == "cloudflare_free":
        if evidence.get("plan") != "free":
            raise ValueError("Only Cloudflare Free is allowed")
        if evidence.get("deployment_kind") == "next_frontend":
            if (
                evidence.get("next_adapter_qualified") is not True
                or evidence.get("eve_runtime_qualified") is not True
            ):
                raise ValueError("Qualify Next.js and Eve before choosing a Cloudflare adapter")
        elif evidence.get("deployment_kind") != "edge_gateway":
            raise ValueError("Cloudflare is not a persistent Python container host")
    elif provider == "vercel_hobby":
        if evidence.get("plan") != "hobby" or evidence.get("deployment_target") != "preview":
            raise ValueError("Only a protected Vercel Hobby preview is allowed")
    else:
        raise ValueError("No Oracle, Railway or unverified host is permitted")
    if evidence.get("always_on_worker") is not False:
        raise ValueError("PERSISTENT_PYTHON_HOST_UNRESOLVED_AT_ZERO_COST")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    if not isinstance(evidence, dict):
        parser.error("Evidence must be a JSON object")
    try:
        validate_hosting(cast("dict[str, object]", evidence))
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    print("$0 policy preflight passed; no cloud resource was created")  # noqa: T201


if __name__ == "__main__":
    main()
