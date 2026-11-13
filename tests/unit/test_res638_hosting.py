from datetime import UTC, datetime

import pytest
from scripts.res638_hosting_preflight import validate_hosting

NOW = datetime(2026, 11, 13, tzinfo=UTC)


def preview() -> dict[str, object]:
    return {
        "provider": "cloudflare_free",
        "plan": "free",
        "deployment_kind": "edge_gateway",
        "monthly_budget_usd": 0,
        "estimated_monthly_cost_usd": 0,
        "paid_inference": False,
        "automatic_upgrade": False,
        "opencode_usage_approved": False,
        "public_launch": False,
        "paid_resources": [],
        "inference_mode": "blocked",
        "account_plan_verified": True,
        "no_overage_verified": True,
        "protection_verified": True,
        "checked_at_utc": NOW.isoformat(),
        "always_on_worker": False,
    }


def test_only_verified_free_previews_are_accepted() -> None:
    validate_hosting(preview(), now=NOW)
    validate_hosting(
        preview() | {"provider": "vercel_hobby", "plan": "hobby", "deployment_target": "preview"},
        now=NOW,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("estimated_monthly_cost_usd", 0.01),
        ("monthly_budget_usd", False),
        ("paid_inference", True),
        ("automatic_upgrade", True),
        ("opencode_usage_approved", True),
        ("paid_resources", ["Oracle account"]),
        ("provider", "oracle_always_free"),
        ("provider", "railway"),
        ("plan", "paid"),
        ("protection_verified", False),
        ("account_plan_verified", False),
        ("no_overage_verified", False),
        ("public_launch", True),
        ("always_on_worker", True),
        ("checked_at_utc", "2026-11-10T00:00:00+00:00"),
        ("deployment_kind", "python_container"),
        ("inference_mode", "opencode_go"),
    ],
)
def test_cost_gate_refuses_paid_unknown_or_misleading_conditions(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        validate_hosting(preview() | {field: value}, now=NOW)


def test_cloudflare_next_adapter_must_be_qualified_before_selection() -> None:
    with pytest.raises(ValueError, match="Qualify"):
        validate_hosting(preview() | {"deployment_kind": "next_frontend"}, now=NOW)
