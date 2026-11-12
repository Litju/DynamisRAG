"""Validated server-side identity contract for the RES-637 `/api/v2` agent API.

The previous contract trusted a client-supplied ``X-Dynamis-User-Id`` header, so any
caller who could reach the API could impersonate any user. This module replaces it with a
**signed identity assertion**: a short-lived HS256 JWT minted by a trusted server-side
caller (the Eve runtime in RES-638, or any other first-party backend) that shares a secret
with this service. The authenticated user id is the verified ``sub`` claim and nothing else.

Contract (see ``docs/res637-eve-python-contract.md``):

* ``Authorization: Bearer <jwt>`` is the only accepted credential.
* ``alg`` must be exactly ``HS256``; ``none`` and every other algorithm are rejected.
* Required claims: ``iss`` (configured issuer), ``aud`` (configured audience), ``sub``
  (user id, validated against :data:`USER_ID_PATTERN`), ``iat``, ``exp``.
* ``exp - iat`` may not exceed :data:`MAX_ASSERTION_LIFETIME_SECONDS`; the assertion must
  not be issued in the future beyond :data:`CLOCK_SKEW_SECONDS`.
* Signature comparison is constant time. A previous secret may be configured so the shared
  secret can be rotated without downtime.
* A development-only escape hatch (``X-Dynamis-User-Id`` without a signature) exists but is
  refused at settings-validation time in ``staging`` and ``production``.

Only the standard library is used: this is a ~100 line verifier whose behaviour is fully
pinned by tests, and it avoids adding a JWT dependency to the lockfile.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Final, cast

__all__ = [
    "CLOCK_SKEW_SECONDS",
    "MAX_ASSERTION_LIFETIME_SECONDS",
    "MIN_SECRET_LENGTH",
    "USER_ID_PATTERN",
    "AuthFailure",
    "IdentityVerifier",
    "mint_identity_assertion",
]

USER_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}$")
MAX_ASSERTION_LIFETIME_SECONDS: Final[int] = 300
CLOCK_SKEW_SECONDS: Final[int] = 30
MIN_SECRET_LENGTH: Final[int] = 32
_MAX_TOKEN_BYTES: Final[int] = 4096


class AuthFailure(Exception):  # noqa: N818 - domain term, not an error-suffixed class
    """A credential was missing or invalid. ``code`` is safe to return to the client."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _b64url_decode(segment: str) -> bytes:
    padded = segment + "=" * (-len(segment) % 4)
    try:
        return base64.urlsafe_b64decode(padded.encode("ascii"))
    except (binascii.Error, UnicodeEncodeError) as err:
        raise AuthFailure("malformed_credential", "Credential is not valid base64url") from err


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _sign(secret: str, signing_input: bytes) -> bytes:
    return hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()


def mint_identity_assertion(
    *,
    secret: str,
    issuer: str,
    audience: str,
    subject: str,
    now: float | None = None,
    lifetime_seconds: int = 120,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """Mint an HS256 identity assertion. Used by trusted callers and by the test-suite."""
    issued_at = int(now if now is not None else time.time())
    claims: dict[str, Any] = {
        **(extra_claims or {}),
        "iss": issuer,
        "aud": audience,
        "sub": subject,
        "iat": issued_at,
        "exp": issued_at + lifetime_seconds,
    }
    header = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode("utf-8"))
    body = _b64url_encode(json.dumps(claims, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header}.{body}".encode("ascii")
    return f"{header}.{body}.{_b64url_encode(_sign(secret, signing_input))}"


def _load_json_object(raw: bytes, what: str) -> dict[str, Any]:
    try:
        parsed: object = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise AuthFailure("malformed_credential", f"Credential {what} is not valid JSON") from err
    if not isinstance(parsed, dict):
        raise AuthFailure("malformed_credential", f"Credential {what} must be a JSON object")
    return cast("dict[str, Any]", parsed)


@dataclass(frozen=True)
class IdentityVerifier:
    """Verifies signed identity assertions and returns the authenticated user id."""

    secrets: tuple[str, ...]
    issuer: str
    audience: str

    def __post_init__(self) -> None:
        if not self.secrets:
            raise ValueError("IdentityVerifier requires at least one signing secret")
        for secret in self.secrets:
            if len(secret) < MIN_SECRET_LENGTH:
                raise ValueError(
                    f"Identity signing secrets must be at least {MIN_SECRET_LENGTH} characters"
                )

    def verify_authorization_header(
        self, authorization: str | None, *, now: float | None = None
    ) -> str:
        """Return the verified ``sub`` for a ``Bearer`` header or raise :class:`AuthFailure`."""
        if not authorization:
            raise AuthFailure("missing_credentials", "Authorization: Bearer <assertion> required")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise AuthFailure("missing_credentials", "Authorization: Bearer <assertion> required")
        return self.verify(token.strip(), now=now)

    def verify(self, token: str, *, now: float | None = None) -> str:
        if len(token) > _MAX_TOKEN_BYTES:
            raise AuthFailure("malformed_credential", "Credential is too large")
        parts = token.split(".")
        if len(parts) != 3:
            raise AuthFailure("malformed_credential", "Credential must have three segments")
        header_b64, body_b64, signature_b64 = parts

        header = _load_json_object(_b64url_decode(header_b64), "header")
        if header.get("alg") != "HS256":
            raise AuthFailure("unsupported_algorithm", "Only HS256 assertions are accepted")

        provided = _b64url_decode(signature_b64)
        signing_input = f"{header_b64}.{body_b64}".encode("ascii")
        signature_ok = False
        for secret in self.secrets:
            # Evaluate every secret so timing does not reveal which one matched.
            if hmac.compare_digest(_sign(secret, signing_input), provided):
                signature_ok = True
        if not signature_ok:
            raise AuthFailure("invalid_signature", "Credential signature is invalid")

        claims = _load_json_object(_b64url_decode(body_b64), "claims")
        return self._validate_claims(claims, now=now if now is not None else time.time())

    def _validate_claims(self, claims: dict[str, Any], *, now: float) -> str:
        if claims.get("iss") != self.issuer:
            raise AuthFailure("invalid_issuer", "Credential issuer is not trusted")
        audience = claims.get("aud")
        audiences: list[object] = (
            cast("list[object]", audience) if isinstance(audience, list) else [audience]
        )
        if self.audience not in audiences:
            raise AuthFailure("invalid_audience", "Credential audience does not match")

        iat = claims.get("iat")
        exp = claims.get("exp")
        if (
            isinstance(iat, bool)
            or isinstance(exp, bool)
            or not isinstance(iat, (int, float))
            or not isinstance(exp, (int, float))
        ):
            raise AuthFailure("invalid_claims", "Credential requires numeric iat and exp")
        if exp <= now - CLOCK_SKEW_SECONDS:
            raise AuthFailure("expired_credential", "Credential has expired")
        if iat > now + CLOCK_SKEW_SECONDS:
            raise AuthFailure("invalid_claims", "Credential was issued in the future")
        if exp - iat > MAX_ASSERTION_LIFETIME_SECONDS or exp <= iat:
            raise AuthFailure("invalid_claims", "Credential lifetime exceeds the permitted bound")

        subject = claims.get("sub")
        if not isinstance(subject, str) or not USER_ID_PATTERN.fullmatch(subject):
            raise AuthFailure("invalid_subject", "Credential subject is not a valid user id")
        return subject
