"""Stage C: the separate, optional long-context benchmark.

Stage A measures reference quality at one common boundary (8192). Stage B
qualifies a production configuration against that reference. Neither answers how
the candidates behave when the context grows, and neither is allowed to: folding
long context into Stage A would change the reference boundary, and folding it into
Stage B would change the deployment target.

Stage C is therefore defined as its own benchmark, in the style of LongEmbed and
LoCo, over the same frozen candidates and the same exact retrieval, at three
windows: 8k, 16k and 32k. It is **optional and non-blocking**:
:func:`require_long_context_promotion` refuses to let a Stage A or Stage B run
wait on it, and only an explicit promotion decision makes it a production
requirement.
"""

from __future__ import annotations

from typing import Final

from dynamisrag.benchmark.contracts import (
    RES138_ARTIFACT_REVISIONS,
    RES138_INPUT_MAX_TOKENS,
    RES138_LONG_CONTEXT_STAGE,
    require_exact_bool,
)
from dynamisrag.benchmark.errors import BenchmarkContractError

__all__ = [
    "LONG_CONTEXT_BENCHMARK_REVISION",
    "LONG_CONTEXT_STYLE",
    "LONG_CONTEXT_WINDOWS",
    "long_context_benchmark_payload",
    "require_long_context_promotion",
]

LONG_CONTEXT_BENCHMARK_REVISION: Final[str] = RES138_ARTIFACT_REVISIONS["long_context"]

LONG_CONTEXT_STYLE: Final[str] = "longembed-loco"
"""The workload style: LongEmbed/LoCo-style retrieval at extended context."""

LONG_CONTEXT_WINDOWS: Final[tuple[int, ...]] = (8192, 16384, 32768)
"""The three declared context windows.

The first equals the Stage A reference boundary, which is what makes the 8k
result comparable with Stage A at all; the others are separate measurements, not
extensions of the reference.
"""


def long_context_benchmark_payload() -> dict[str, object]:
    """The declaration of the optional Stage C benchmark.

    Written as data so a reviewer can see that Stage C is scoped, separate and
    not part of a Stage A/B result — and so a future promotion has to change this
    declaration rather than quietly reinterpret an existing boundary.
    """
    return {
        "artifact_revision": LONG_CONTEXT_BENCHMARK_REVISION,
        "stage": RES138_LONG_CONTEXT_STAGE,
        "style": LONG_CONTEXT_STYLE,
        "windows": list(LONG_CONTEXT_WINDOWS),
        "reference_boundary": RES138_INPUT_MAX_TOKENS,
        "optional": True,
        "promoted_to_production_requirement": False,
        "blocks_stage_a": False,
        "blocks_stage_b": False,
        "note": (
            "a separate optional benchmark; it must not block Stage A or Stage B unless an "
            "operator explicitly promotes it to a production requirement"
        ),
    }


def require_long_context_promotion(*, promoted: bool) -> None:
    """Refuse a promotion that is not explicit.

    Called by any Stage A or Stage B orchestration that is asked to wait on Stage
    C; a truthy default, a missing flag or an inferred decision all fail here,
    because promoting long context changes the production requirement and must be
    a visible declaration rather than an accident of ordering.
    """
    require_exact_bool(
        promoted,
        kind="long-context promotion flag",
        operation="require_long_context_promotion",
        because="Promoting Stage C to a production requirement is a deliberate decision, not a "
        "truthy value that arrived from a config default.",
    )
    if not promoted:
        raise BenchmarkContractError(
            "the long-context benchmark is optional and has not been promoted to a production "
            "requirement. Stage A and Stage B must not block on it.",
            operation="require_long_context_promotion",
        )
