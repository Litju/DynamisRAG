"""The reference TEI deployment and the attestation that describes it (RES-137).

Three files make a claim about one container, and the claim is load-bearing:
``compose.embedding.yaml`` starts it, ``.env.example`` tells the application how
to reach it, and ``TeiDeploymentSemantics`` is what the application *asserts* about
its output-affecting startup configuration. Nothing in TEI's ``GET /info`` can
confirm the third, because ``/info`` reports the build, the model, the dtype, the
pooling and the limits, and reports nothing about ``--default-prompt`` or
``--dense-path``.

So the assertion is checked the only way an absence can be: by reading what the
compose file's command line does *not* contain. These tests need no Docker and no
network, because the contradiction they look for is in the repository rather than
in a running container -- which is why a contributor who adds a flag gets a red
build instead of a subtly different vector space discovered months later.

Pure file reading, deliberately. A test that shelled out to ``docker compose
config`` would pass on a developer machine and be skipped in CI; a test that
started TEI would download a model. The unit suite stays hermetic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import httpx2
import pytest
from pydantic import AnyHttpUrl

from dynamisrag.config import Settings
from dynamisrag.embedding import (
    REFERENCE_TEI_DEPLOYMENT_SEMANTICS,
    TEI_PROVIDER_NAME,
    EmbeddingGenerationConfig,
    EmbeddingRetryPolicy,
    ExpectedTeiModel,
    TeiDefaultPromptMode,
    TeiDeploymentSemantics,
    TruncationDirection,
    tei_provider_from_settings,
)
from tests._support import REPO_ROOT, TEI_MODEL_ID, TEI_MODEL_SHA, build_settings

_COMPOSE: Final[Path] = REPO_ROOT / "compose.embedding.yaml"
_ENV_EXAMPLE: Final[Path] = REPO_ROOT / ".env.example"

# Flags that change returned embeddings while leaving `/info` byte-identical. Each
# one is an absence the reference attestation claims, so each must be absent from
# the command line -- and a test cannot assert the absence of a string it cannot
# distinguish from the file's own explanation of why it is absent.
_OUTPUT_AFFECTING_FLAGS: Final[tuple[str, ...]] = (
    "--default-prompt",
    "--default-prompt-name",
    "--dense-path",
)

_EXPECTED_COMMAND_FLAGS: Final[tuple[str, ...]] = (
    "--model-id",
    "--revision",
    "--port",
    "--max-client-batch-size",
)


def _significant_lines(path: Path) -> list[str]:
    """Every non-blank, non-comment line of a repository file, stripped of indent.

    Comments are stripped the way YAML strips them -- from a ``#`` at the start of
    the line or after whitespace -- and quoted text is left alone. Necessary
    because the compose file explains in comments exactly which flags must not
    appear, so a naive substring search over the raw text would find the
    explanation and fail.
    """
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        cut = len(raw)
        for index, character in enumerate(raw):
            if character == "#" and (index == 0 or raw[index - 1].isspace()):
                cut = index
                break
        stripped = raw[:cut].strip()
        if stripped:
            lines.append(stripped)
    return lines


def _compose_command_flags() -> list[str]:
    """The long flags in the compose ``command`` list, in file order."""
    return [line[1:].strip() for line in _significant_lines(_COMPOSE) if line.startswith("- --")]


def _env_example_value(name: str) -> str:
    """The value a committed ``.env.example`` assigns, empty when it assigns none."""
    for line in _significant_lines(_ENV_EXAMPLE):
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1]
    return ""


def _refusing_transport() -> httpx2.MockTransport:
    """A transport that fails the test if it is ever used.

    Every assertion here is about what a provider is *built* with. Whether the
    container answers belongs to the manual real-TEI proof, which needs the
    container; a unit test that opened a socket would be testing the network.
    """

    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise AssertionError(f"no request is expected here, got {request.url.path}")

    return httpx2.MockTransport(refuse)


def _generation() -> EmbeddingGenerationConfig:
    return EmbeddingGenerationConfig(
        normalize=True,
        truncate=False,
        truncation_direction=TruncationDirection.RIGHT,
    )


def _settings(*, url: str, **overrides: object) -> Settings:
    base = build_settings()
    values: dict[str, object] = {
        "tei_url": AnyHttpUrl(url),
        "tei_expected_model_id": TEI_MODEL_ID,
        "tei_expected_model_sha": TEI_MODEL_SHA,
        **overrides,
    }
    return base.model_copy(update=values)


# ---------------------------------------------------------------------------
# The compose file cannot contradict the attestation
# ---------------------------------------------------------------------------


def test_the_command_line_is_read_at_all() -> None:
    """The control for the absence tests below.

    Without this, "no ``--default-prompt`` is passed" would also pass against a
    compose file whose ``command`` was empty, and the absence of a contradiction
    would be indistinguishable from the absence of a container.
    """
    assert tuple(_compose_command_flags()) == _EXPECTED_COMMAND_FLAGS


@pytest.mark.parametrize("flag", _OUTPUT_AFFECTING_FLAGS)
def test_the_reference_container_passes_no_output_affecting_startup_flag(flag: str) -> None:
    """The absence is the claim, so the claim is what gets asserted.

    ``REFERENCE_TEI_DEPLOYMENT_SEMANTICS`` says this deployment has no default
    prompt and no dense-path override. TEI's CLI cannot express "no default
    prompt" -- there is no ``--no-default-prompt`` -- so the policy is an absence,
    and the only way to hold an absence is to check that the command line does not
    contain the thing. A contributor who adds one of these flags gets a red build
    rather than a second vector space under an unchanged
    ``embedding_config_sha256``.
    """
    assert flag not in _compose_command_flags()


def test_the_attestation_applied_to_the_reference_container_is_the_reference_one() -> None:
    """The compose file and the constant are one source of truth, asserted equal.

    Read together with the absence tests: nothing in the command line contradicts
    the policy, and the policy being applied is this one. Together they say the
    container that starts is the container the fingerprint describes.
    """
    assert REFERENCE_TEI_DEPLOYMENT_SEMANTICS.default_prompt_mode is TeiDefaultPromptMode.NONE
    assert REFERENCE_TEI_DEPLOYMENT_SEMANTICS.default_prompt_name is None
    assert REFERENCE_TEI_DEPLOYMENT_SEMANTICS.default_prompt_sha256 is None
    assert REFERENCE_TEI_DEPLOYMENT_SEMANTICS.dense_path is None


def test_the_reference_container_pins_a_revision_and_states_its_client_batch_limit() -> None:
    """Both are stated rather than inherited from a default that could move.

    ``--revision`` is required to be a Hub commit id, which is what makes "TEI is
    serving something else" a local failure. ``--max-client-batch-size`` is
    operational rather than semantic -- it is in no fingerprint -- but stating it
    lets the adapter's own "a configured batch above the advertised limit is a
    configuration error" check be reasoned about without reading TEI's defaults.
    """
    flags = _compose_command_flags()
    lines = _significant_lines(_COMPOSE)

    assert "--revision" in flags
    assert "--max-client-batch-size" in flags
    # Required variables: a missing or blank value must fail loudly rather than
    # quietly start a model server nobody asked for.
    assert any("DYNAMISRAG_TEI_MODEL_SHA:?" in line for line in lines)
    assert any("DYNAMISRAG_TEI_MODEL_ID:?" in line for line in lines)
    # The image is an exact patch release, and the template names the same one, so a
    # teammate and CI agree on the build their fingerprints are taken against.
    assert any(
        line == "image: ghcr.io/huggingface/text-embeddings-inference:cpu-"
        "${DYNAMISRAG_TEI_VERSION:-1.9.4}"
        for line in lines
    )
    assert _env_example_value("DYNAMISRAG_TEI_VERSION") == "1.9.4"


# ---------------------------------------------------------------------------
# The Hub cache is actually persisted
# ---------------------------------------------------------------------------


def test_the_named_volume_is_mounted_where_the_image_puts_its_hub_cache() -> None:
    """The official 1.9.4 image sets ``HUGGINGFACE_HUB_CACHE=/data``.

    The file mounted ``tei-model-cache:/models``, which persisted nothing: the
    server downloaded to ``/data`` and every ``down``/``up`` re-downloaded the
    model while ``/models`` stayed empty. A named volume that does not hold the
    cache is not a cache, and the symptom is invisible until someone notices the
    download on every start.

    The environment variable is asserted as well as the mount, so the two cannot
    drift apart: a future image that moved its own default would be caught here
    rather than by a teammate wondering why the cache stopped persisting.
    """
    lines = _significant_lines(_COMPOSE)

    assert "HUGGINGFACE_HUB_CACHE: /data" in lines
    assert "- tei-model-cache:/data" in lines
    assert not any("tei-model-cache:/models" in line for line in lines)


# ---------------------------------------------------------------------------
# The application can reach the reference container
# ---------------------------------------------------------------------------


def test_the_reference_url_is_the_scheme_the_container_serves() -> None:
    """Plaintext HTTP on loopback, and the template says so.

    The container publishes host 8080 to container port 80 with no TLS, and the
    template said ``https://127.0.0.1:8080``. ``DYNAMISRAG_TEI_VERIFY_TLS=false``
    disables certificate *verification*; it does not make an HTTPS URL speak
    plaintext, so the documented configuration could not connect to the documented
    container. HTTPS remains fully supported for an external deployment -- nothing
    in the settings forbids it -- it is simply not what this container is.
    """
    url = _env_example_value("DYNAMISRAG_TEI_URL")

    assert url == "http://127.0.0.1:8080"
    assert any(line.strip() == '- "127.0.0.1:8080:80"' for line in _significant_lines(_COMPOSE))


def test_https_stays_supported_for_an_external_deployment() -> None:
    """The fix is the scheme, not the removal of transport security.

    A test that only asserted ``http://`` would be satisfied by an adapter that
    refused TLS outright, which is the wrong repair. ``Settings`` accepts any HTTP
    URL and ``verify_tls`` is a real knob, so a deployment behind a trusted
    certificate is one setting away.
    """
    provider = tei_provider_from_settings(
        _settings(url="https://tei.example.invalid:8443", tei_verify_tls=True),
        generation_config=_generation(),
        transport=_refusing_transport(),
    )

    assert provider.base_url == "https://tei.example.invalid:8443"
    provider.close()


def test_the_reference_template_builds_a_provider_against_the_reference_container() -> None:
    """The template, the container and the provider agree, with no container running.

    Construction only: settings are read and the identity is checked, and nothing
    is sent. That is enough to catch a template pointing somewhere the reference
    compose file does not publish -- which is exactly the bug that was here.
    """
    provider = tei_provider_from_settings(
        _settings(url=_env_example_value("DYNAMISRAG_TEI_URL")),
        generation_config=_generation(),
        transport=_refusing_transport(),
    )

    assert provider.base_url == "http://127.0.0.1:8080"
    assert provider.deployment_semantics == REFERENCE_TEI_DEPLOYMENT_SEMANTICS
    provider.close()


def test_the_reference_batch_size_fits_the_reference_client_batch_limit() -> None:
    """A template that cannot run is a template nobody notices until it is used.

    The adapter refuses a configured batch above the server's advertised
    ``max_client_batch_size`` rather than clamping it, so a template whose client
    batch exceeded the container's limit would fail every run. Both numbers are in
    the template; this is the cross-file check that keeps them honest.
    """
    client_batch = int(_env_example_value("DYNAMISRAG_TEI_BATCH_SIZE"))
    server_limit = int(_env_example_value("DYNAMISRAG_TEI_MAX_CLIENT_BATCH_SIZE"))

    assert client_batch <= server_limit


def test_the_retry_values_in_the_template_are_ones_the_contract_accepts() -> None:
    """A template that could not build a provider would fail at a late, confusing point.

    ``max_attempts=0`` or a negative backoff is refused at provider construction --
    correctly, and with a message about a configuration value a reader would have to
    go looking for. Reading them here is the cheap version of that.
    """
    policy = EmbeddingRetryPolicy(
        max_attempts=int(_env_example_value("DYNAMISRAG_TEI_MAX_ATTEMPTS")),
        base_backoff_seconds=float(_env_example_value("DYNAMISRAG_TEI_RETRY_BACKOFF_SECONDS")),
    )

    assert policy.max_attempts >= 1
    assert policy.backoff_seconds() == (0.5, 1.0)


def test_the_example_revision_in_the_template_is_a_real_commit_id() -> None:
    """The commented-out example is what a teammate copies.

    If it named a branch, the first thing they would hit is an
    :class:`~dynamisrag.embedding.errors.EmbeddingContractError` about a mutable
    alias -- at provider construction, before they had any reason to doubt it.
    """
    expected = ExpectedTeiModel(model_id=TEI_MODEL_ID, model_sha=TEI_MODEL_SHA)
    template = _ENV_EXAMPLE.read_text(encoding="utf-8")

    assert expected.model_sha == TEI_MODEL_SHA
    assert f"# DYNAMISRAG_TEI_MODEL_SHA={TEI_MODEL_SHA}" in template
    assert f"# DYNAMISRAG_TEI_MODEL_ID={TEI_MODEL_ID}" in template


# ---------------------------------------------------------------------------
# The attestation itself
# ---------------------------------------------------------------------------


def test_the_attestation_is_frozen_and_compares_by_content() -> None:
    """Two providers built with the same policy must agree; a different one must not.

    That only holds if the value is immutable and compares by content, which is
    what the run-identity comparison and the digest both depend on.
    """
    same = TeiDeploymentSemantics()
    other = TeiDeploymentSemantics(
        default_prompt_mode=TeiDefaultPromptMode.NAMED, default_prompt_name="query"
    )

    assert same == REFERENCE_TEI_DEPLOYMENT_SEMANTICS
    assert same != other
    assert len({same, REFERENCE_TEI_DEPLOYMENT_SEMANTICS, other}) == 2
    with pytest.raises(AttributeError):
        same.dense_path = "dense/2_Dense"  # type: ignore[misc]


def test_a_null_prompt_name_is_not_a_statement_about_prompts() -> None:
    """The sentence most likely to be misread, so it is asserted rather than prose.

    A null ``prompt_name`` is *not* "no prompt"; it is "whatever this deployment was
    started with, if anything". The request cannot express the difference, which is
    why the policy lives in the attestation and in the fingerprint rather than in
    the request body.
    """
    config = _generation()

    assert config.prompt_name is None
    assert REFERENCE_TEI_DEPLOYMENT_SEMANTICS.default_prompt_mode is TeiDefaultPromptMode.NONE
    assert TEI_PROVIDER_NAME == "tei"
