"""Identity-contract tests for the RES-637 `/api/v2` agent API.

Covers the signed-assertion verifier (forgery, algorithm confusion, expiry, audience, lifetime
bounds, rotation), the router's refusal to trust an unsigned user-id header, cross-user
isolation for threads/runs/evidence/papers, and fail-closed settings validation.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from dynamisrag.agent.auth import (
    MAX_ASSERTION_LIFETIME_SECONDS,
    AuthFailure,
    IdentityVerifier,
    mint_identity_assertion,
)
from dynamisrag.config import Environment, Settings
from tests._support import build_settings
from tests.agent_stack import AUDIENCE, IDENTITY_SECRET, ISSUER, auth_headers, build_stack

OTHER_SECRET = "a-completely-different-secret-0123456789"


def _stack_app(*, rate_limit_per_minute: int = 600) -> tuple[FastAPI, dict[str, Any]]:
    stack = build_stack(rate_limit_per_minute=rate_limit_per_minute)
    return stack.app, dict(stack.studies)


def _verifier(*secrets: str) -> IdentityVerifier:
    return IdentityVerifier(secrets=secrets or (IDENTITY_SECRET,), issuer=ISSUER, audience=AUDIENCE)


def _mint(**overrides: object) -> str:
    args: dict[str, object] = {
        "secret": IDENTITY_SECRET,
        "issuer": ISSUER,
        "audience": AUDIENCE,
        "subject": "scientist-1",
        "now": time.time(),
    }
    args.update(overrides)
    return mint_identity_assertion(**args)  # type: ignore[arg-type]


def _b64(data: dict[str, object]) -> str:
    return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()


def _failure_code(token: str, verifier: IdentityVerifier | None = None) -> str:
    with pytest.raises(AuthFailure) as info:
        (verifier or _verifier()).verify(token)
    return info.value.code


def test_valid_assertion_returns_verified_subject() -> None:
    assert _verifier().verify(_mint()) == "scientist-1"


def test_forged_signature_is_rejected() -> None:
    assert _failure_code(_mint(secret=OTHER_SECRET)) == "invalid_signature"
    header, body, _ = _mint().split(".")
    assert _failure_code(f"{header}.{body}.AAAA") == "invalid_signature"


def test_tampered_subject_is_rejected() -> None:
    header, _, signature = _mint().split(".")
    forged_body = _b64(
        {
            "iss": ISSUER,
            "aud": AUDIENCE,
            "sub": "victim",
            "iat": int(time.time()),
            "exp": int(time.time()) + 60,
        }
    )
    assert _failure_code(f"{header}.{forged_body}.{signature}") == "invalid_signature"


def test_alg_none_and_alg_confusion_are_rejected() -> None:
    body = _b64(
        {
            "iss": ISSUER,
            "aud": AUDIENCE,
            "sub": "scientist-1",
            "iat": int(time.time()),
            "exp": int(time.time()) + 60,
        }
    )
    assert (
        _failure_code(f"{_b64({'alg': 'none', 'typ': 'JWT'})}.{body}.") == "unsupported_algorithm"
    )
    assert _failure_code(f"{_b64({'alg': 'HS512'})}.{body}.AAAA") == "unsupported_algorithm"
    assert _failure_code(f"{_b64({'alg': 'RS256'})}.{body}.AAAA") == "unsupported_algorithm"


def test_expiry_audience_issuer_and_lifetime_bounds() -> None:
    now = time.time()
    assert _failure_code(_mint(now=now - 1000)) == "expired_credential"
    assert _failure_code(_mint(audience="someone-else")) == "invalid_audience"
    assert _failure_code(_mint(issuer="untrusted")) == "invalid_issuer"
    assert _failure_code(_mint(now=now + 3600)) == "invalid_claims"
    long_lived = _mint(lifetime_seconds=MAX_ASSERTION_LIFETIME_SECONDS + 1)
    assert _failure_code(long_lived) == "invalid_claims"


def test_invalid_subjects_are_rejected() -> None:
    for subject in ("", "has space", "../etc/passwd", "x" * 200, "-leading"):
        assert _failure_code(_mint(subject=subject)) == "invalid_subject"


def test_malformed_credentials_are_rejected_without_crashing() -> None:
    for token in ("", "a.b", "a.b.c.d", "!!!.???.***", "x" * 5000):
        with pytest.raises(AuthFailure):
            _verifier().verify(token)


def test_secret_rotation_accepts_previous_secret_only_when_configured() -> None:
    old = _mint(secret=OTHER_SECRET)
    assert _failure_code(old) == "invalid_signature"
    assert _verifier(IDENTITY_SECRET, OTHER_SECRET).verify(old) == "scientist-1"


def test_short_secrets_are_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="at least"):
        IdentityVerifier(secrets=("short",), issuer=ISSUER, audience=AUDIENCE)


def test_authorization_header_requires_bearer_scheme() -> None:
    with pytest.raises(AuthFailure) as info:
        _verifier().verify_authorization_header("Basic abc")
    assert info.value.code == "missing_credentials"
    with pytest.raises(AuthFailure):
        _verifier().verify_authorization_header(None)


# ---------------------------------------------------------------------------
# HTTP-level behaviour
# ---------------------------------------------------------------------------


def test_router_never_trusts_unsigned_user_id_header() -> None:
    app, _ = _stack_app()
    with TestClient(app) as client:
        resp = client.post(
            "/api/v2/threads",
            headers={"X-Dynamis-User-Id": "scientist-1"},
            json={"title": "spoof"},
        )
        assert resp.status_code == 401
        assert resp.json()["detail"]["code"] == "missing_credentials"


def test_header_cannot_override_a_valid_assertion() -> None:
    app, _ = _stack_app()
    headers = {**auth_headers("alice"), "X-Dynamis-User-Id": "bob"}
    with TestClient(app) as client:
        resp = client.post("/api/v2/threads", headers=headers, json={"title": "mine"})
        assert resp.status_code == 201
        assert resp.json()["owner_id"] == "alice"


def test_forged_and_expired_assertions_are_401() -> None:
    app, _ = _stack_app()
    forged = auth_headers("alice", secret=OTHER_SECRET)
    expired = {
        "Authorization": "Bearer "
        + mint_identity_assertion(
            secret=IDENTITY_SECRET,
            issuer=ISSUER,
            audience=AUDIENCE,
            subject="alice",
            now=time.time() - 4000,
        )
    }
    with TestClient(app) as client:
        for headers in (forged, expired, {}):
            assert client.post("/api/v2/threads", headers=headers, json={}).status_code == 401


def test_cross_user_access_is_forbidden_for_thread_run_evidence_and_papers() -> None:
    app, studies = _stack_app()
    alice, mallory = auth_headers("alice"), auth_headers("mallory")
    with TestClient(app) as client:
        thread_id = client.post("/api/v2/threads", headers=alice, json={"title": "t"}).json()[
            "thread_id"
        ]
        run = client.post(
            f"/api/v2/threads/{thread_id}/runs",
            headers=alice,
            json={
                "question": "Achilles tendon stiffness during depth jumping?",
                "idempotency_key": "k1",
            },
        ).json()
        pk = studies["PMC5263151"][1][1]

        assert client.get(f"/api/v2/threads/{thread_id}", headers=mallory).status_code == 403
        assert (
            client.post(
                f"/api/v2/threads/{thread_id}/runs",
                headers=mallory,
                json={"question": "steal", "idempotency_key": "k2"},
            ).status_code
            == 403
        )
        assert (
            client.get(f"/api/v2/runs/{run['run_id']}/events", headers=mallory).status_code == 403
        )
        assert (
            client.post(f"/api/v2/runs/{run['run_id']}/cancel", headers=mallory).status_code == 403
        )
        assert (
            client.post(
                f"/api/v2/runs/{run['run_id']}/resume", headers=mallory, json={"approve": True}
            ).status_code
            == 403
        )
        assert (
            client.get(f"/api/v2/evidence/{pk}?thread_id={thread_id}", headers=mallory).status_code
            == 403
        )
        assert (
            client.get(f"/api/v2/papers?thread_id={thread_id}", headers=mallory).status_code == 403
        )
        # The owner is unaffected.
        assert client.get(f"/api/v2/threads/{thread_id}", headers=alice).status_code == 200


def test_rate_limit_is_keyed_by_verified_identity() -> None:
    app, _ = _stack_app(rate_limit_per_minute=3)
    with TestClient(app) as client:
        codes = [
            client.post("/api/v2/threads", headers=auth_headers("alice"), json={}).status_code
            for _ in range(4)
        ]
        assert codes == [201, 201, 201, 429]
        assert (
            client.post("/api/v2/threads", headers=auth_headers("bob"), json={}).status_code == 201
        )


# ---------------------------------------------------------------------------
# Fail-closed settings
# ---------------------------------------------------------------------------


def _settings(**overrides: object) -> Settings:
    base = build_settings().model_dump()
    base.update(overrides)
    return Settings.from_mapping(base)


def test_agent_disabled_by_default() -> None:
    assert build_settings().agent_enabled is False


def test_enabling_agent_without_identity_or_store_fails_closed() -> None:
    with pytest.raises(ValidationError, match="DYNAMISRAG_AGENT_DATABASE_URL"):
        _settings(agent_enabled=True, agent_identity_secret="x" * 40)
    with pytest.raises(ValidationError, match="IDENTITY_SECRET"):
        _settings(agent_enabled=True, agent_allow_ephemeral_store=True)


def test_production_refuses_ephemeral_store_dev_header_and_deterministic_model() -> None:
    common: dict[str, object] = {
        "environment": Environment.PRODUCTION,
        "agent_enabled": True,
        "agent_identity_secret": "x" * 40,
        "agent_provider": "ollama",
    }
    with pytest.raises(ValidationError, match="durable"):
        _settings(**common, agent_allow_ephemeral_store=True)
    with pytest.raises(ValidationError, match="dev_identity_header"):
        _settings(
            **common,
            agent_database_url="postgresql://u:p@agent-db:5432/agent",
            agent_dev_identity_header_enabled=True,
        )
    with pytest.raises(ValidationError, match="deterministic"):
        _settings(
            **{**common, "agent_provider": "deterministic"},
            agent_database_url="postgresql://u:p@agent-db:5432/agent",
        )


def test_opencode_go_requires_explicit_usage_approval() -> None:
    kwargs: dict[str, object] = {
        "agent_enabled": True,
        "agent_identity_secret": "x" * 40,
        "agent_allow_ephemeral_store": True,
        "agent_provider": "opencode_go",
    }
    with pytest.raises(ValidationError, match="usage_approved"):
        _settings(**kwargs)
    assert (
        _settings(**kwargs, agent_opencode_go_usage_approved=True).agent_provider == "opencode_go"
    )
