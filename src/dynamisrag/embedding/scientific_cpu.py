"""Pinned CPU deployment used by the real scientific corpus, distinct from RES-142."""

from dynamisrag.embedding.contracts import EmbeddingGenerationConfig, TruncationDirection
from dynamisrag.embedding.tei import TeiDeploymentSemantics

CPU_IMAGE_DIGEST = "sha256:2538ea1c9640d3763b15af668039d24172d063b42337b0c27796fc2be180c78d"
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
DOCUMENT_FINGERPRINT = "8438d37d8e8c8f975f62cbc6f6c644abc27304b18b7a330d9157a70f21c9c1f7"
QUERY_FINGERPRINT = "94245fa64ff582ab1428c7aea38387e9091a23e538c96d7fbf7b48e1755da8fd"
DOCUMENT_CONFIG = EmbeddingGenerationConfig(
    normalize=True,
    truncate=False,
    truncation_direction=TruncationDirection.RIGHT,
    prompt_name="document",
    dimensions=512,
)
QUERY_CONFIG = EmbeddingGenerationConfig(
    normalize=True,
    truncate=False,
    truncation_direction=TruncationDirection.RIGHT,
    prompt_name="query",
    dimensions=512,
)


class CpuDeployment(TeiDeploymentSemantics):
    """Attestation matching the demonstrated image, CPU backend and MKL/OMP setting."""

    def payload(self) -> dict[str, object]:
        return {
            **super().payload(),
            "local_device": "cpu",
            "local_backend": "candle-mkl",
            "local_image_digest": CPU_IMAGE_DIGEST,
            "local_threads": 4,
            "local_platform": "linux-amd64",
        }
