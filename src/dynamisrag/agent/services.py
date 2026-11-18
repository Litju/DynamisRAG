"""Assembly of the RES-637 agent runtime (store, tools, model, orchestrator, workers).

Shared by the FastAPI application factory and the standalone worker process so both build the
agent identically from validated :class:`~dynamisrag.config.Settings`.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sqlalchemy.engine import Engine

from dynamisrag.agent.graph import ConversationalResearchOrchestrator
from dynamisrag.agent.observability import TracingMode, TracingPolicy
from dynamisrag.agent.persistence import (
    AgentConversationStore,
    InMemoryAgentConversationStore,
    PostgresAgentConversationStore,
)
from dynamisrag.agent.providers import (
    AgentModelConfig,
    AgentProviderKind,
    build_agent_model_runtime,
)
from dynamisrag.agent.routing import (
    ModelRuntime,
    OpenCodeGoRosterConfig,
    ProviderBlockedRuntime,
    build_opencode_go_runtime,
)
from dynamisrag.agent.tools import ScientificToolRegistry
from dynamisrag.agent.worker import AgentWorkerPool
from dynamisrag.config import Settings
from dynamisrag.db.engine import create_database_engine
from dynamisrag.embedding.errors import EmbeddingProviderError
from dynamisrag.search.bm25 import Bm25SearchService
from dynamisrag.search.client import OpenSearchClient
from dynamisrag.search.retrieval import (
    HybridRetrievalService,
    QueryEmbeddingService,
    create_query_embedding_provider,
)
from dynamisrag.search.scientific import CHUNKER, ScientificRetrievalService

if TYPE_CHECKING:
    from dynamisrag.corpus.research import LiteratureAcquisition

__all__ = [
    "AgentRuntime",
    "build_agent_services",
    "build_model_runtime",
    "build_research_acquisition",
    "build_research_retrieval",
    "build_standalone_agent_runtime",
    "model_config_from_settings",
    "tracing_policy_from_settings",
]


def _empty_callbacks() -> list[Callable[[], None]]:
    return []


@dataclass
class AgentRuntime:
    """Everything the API and workers need; ``close`` releases it in reverse order."""

    store: AgentConversationStore
    registry: ScientificToolRegistry
    orchestrator: ConversationalResearchOrchestrator
    pool: AgentWorkerPool | None = None
    _cleanups: list[Callable[[], None]] = field(default_factory=_empty_callbacks)

    def start_workers(self) -> None:
        if self.pool is not None:
            self.pool.start()

    def close(self) -> None:
        if self.pool is not None:
            self.pool.stop()
        for cleanup in reversed(self._cleanups):
            cleanup()
        self.store.close()


_OLLAMA_DEFAULT_URL = "http://127.0.0.1:11435"


def model_config_from_settings(settings: Settings) -> AgentModelConfig:
    base_url = settings.agent_model_base_url
    if settings.agent_provider != "ollama" and base_url == _OLLAMA_DEFAULT_URL:
        base_url = None  # the Ollama default must never be used as another provider's endpoint
    return AgentModelConfig(
        provider=AgentProviderKind(settings.agent_provider),
        model_name=settings.agent_model_name,
        model_revision=settings.agent_model_revision,
        base_url=base_url,
        api_key=settings.agent_model_api_key,
        num_ctx=settings.agent_model_num_ctx,
        usage_approved=settings.agent_opencode_go_usage_approved,
    )


def build_model_runtime(settings: Settings) -> ModelRuntime:
    """OpenCode Go builds the three-model routed roster; other providers stay single-model."""
    if settings.agent_provider == AgentProviderKind.OPENCODE_GO.value:
        if not settings.agent_opencode_go_usage_approved:
            # Research-use authorization is unverified: no client, no key, no request.
            return ProviderBlockedRuntime(model_name=settings.agent_model_coordinator)
        return build_opencode_go_runtime(
            OpenCodeGoRosterConfig(
                api_key=settings.agent_model_api_key,
                base_url=None
                if settings.agent_model_base_url == _OLLAMA_DEFAULT_URL
                else settings.agent_model_base_url,
                usage_approved=settings.agent_opencode_go_usage_approved,
                fast_model=settings.agent_model_fast,
                coordinator_model=settings.agent_model_coordinator,
                pro_model=settings.agent_model_pro,
                max_pro_escalations=settings.agent_max_pro_escalations,
                max_run_cost_usd=settings.agent_run_cost_budget_usd,
                max_run_tokens=settings.agent_run_token_budget,
            )
        )
    return build_agent_model_runtime(model_config_from_settings(settings))


def tracing_policy_from_settings(settings: Settings) -> TracingPolicy:
    if not settings.agent_langsmith_tracing_enabled:
        return TracingPolicy.off()
    return TracingPolicy(
        mode=TracingMode(settings.agent_langsmith_mode),
        project_name=settings.agent_langsmith_project,
        api_key=settings.agent_langsmith_api_key,
    )


def build_research_retrieval(settings: Settings, engine: Engine) -> ScientificRetrievalService:
    """Dense retrieval that follows the research workspace's published Qdrant alias."""
    from dynamisrag.search.qdrant import QdrantClient
    from dynamisrag.search.research_projection import ResearchCorpusRetrieval

    return ResearchCorpusRetrieval(
        engine,
        qdrant=QdrantClient(
            str(settings.research_qdrant_url).rstrip("/"),
            api_key=settings.research_qdrant_api_key.get_secret_value()
            if settings.research_qdrant_api_key
            else None,
        ),
        tei_url=str(settings.research_tei_url),
        alias=settings.research_qdrant_alias,
        store_root=settings.research_store_directory,
        chunker_revision=CHUNKER,
    )


def build_research_acquisition(settings: Settings) -> tuple[LiteratureAcquisition, Engine]:
    """The single canonical writer: owner engine, pinned document TEI, versioned Qdrant."""
    from sqlalchemy import create_engine

    from dynamisrag.config import to_sqlalchemy_url
    from dynamisrag.corpus.research import AcquisitionLimits, LiteratureAcquisition
    from dynamisrag.embedding import (
        EmbeddingRetryPolicy,
        EmbeddingRuntimeConfig,
        ExpectedTeiModel,
        TeiEmbeddingProvider,
    )
    from dynamisrag.embedding.scientific_cpu import (
        DOCUMENT_CONFIG,
        MODEL_ID,
        MODEL_REVISION,
        CpuDeployment,
    )
    from dynamisrag.search.qdrant import QdrantClient

    if settings.research_ingest_database_url is None:
        raise RuntimeError("research_ingest_database_url is required")
    ingest_engine = create_engine(
        to_sqlalchemy_url(settings.research_ingest_database_url), pool_pre_ping=True
    )

    def document_provider() -> TeiEmbeddingProvider:
        return TeiEmbeddingProvider(
            base_url=str(settings.research_tei_url).rstrip("/"),
            expected_model=ExpectedTeiModel(MODEL_ID, MODEL_REVISION),
            deployment_semantics=CpuDeployment(),
            generation_config=DOCUMENT_CONFIG,
            runtime_config=EmbeddingRuntimeConfig(
                batch_size=1, timeout_seconds=180, retry=EmbeddingRetryPolicy(max_attempts=2)
            ),
            verify_tls=False,
        )

    acquisition = LiteratureAcquisition(
        ingest_engine=ingest_engine,
        store_root=settings.research_store_directory,
        qdrant=QdrantClient(
            str(settings.research_qdrant_url).rstrip("/"),
            api_key=settings.research_qdrant_api_key.get_secret_value()
            if settings.research_qdrant_api_key
            else None,
        ),
        alias=settings.research_qdrant_alias,
        document_provider=document_provider,
        chunker_revision=CHUNKER,
        limits=AcquisitionLimits(
            max_papers=settings.research_max_papers,
            deadline_seconds=settings.research_acquisition_deadline_seconds,
        ),
    )
    return acquisition, ingest_engine


def build_agent_services(
    *,
    settings: Settings,
    engine: Engine,
    scientific: ScientificRetrievalService | None,
    search: Bm25SearchService,
    retrieval: HybridRetrievalService,
    acquisition: LiteratureAcquisition | None = None,
) -> AgentRuntime | None:
    """Build the agent runtime, or ``None`` when the agent is disabled.

    Raises ``AgentDatabaseIsolationError`` (at startup) if the agent database is the protected
    canonical database.
    """
    if not settings.agent_enabled:
        return None
    store: AgentConversationStore = (
        PostgresAgentConversationStore(
            str(settings.agent_database_url),
            schema_name=settings.agent_postgres_schema,
            protected_database_url=str(settings.database_url),
        )
        if settings.agent_database_url is not None
        else InMemoryAgentConversationStore()
    )
    registry = ScientificToolRegistry(
        engine,
        scientific_retrieval=scientific,
        opensearch_bm25=search,
        opensearch_hybrid=retrieval,
    )
    orchestrator = ConversationalResearchOrchestrator(
        tool_registry=registry,
        store=store,
        model_runtime=build_model_runtime(settings),
        run_timeout_seconds=settings.agent_run_timeout_seconds,
        lease_seconds=settings.agent_lease_seconds,
        max_attempts=settings.agent_max_run_attempts,
        tracing=tracing_policy_from_settings(settings),
        literature_acquisition=acquisition,
    )
    pool = (
        AgentWorkerPool(orchestrator, size=settings.agent_inline_workers)
        if settings.agent_inline_workers > 0
        else None
    )
    return AgentRuntime(store=store, registry=registry, orchestrator=orchestrator, pool=pool)


def build_standalone_agent_runtime(settings: Settings) -> AgentRuntime:
    """Build an agent runtime with its own infrastructure clients for a worker process."""
    stack = ExitStack()
    engine = create_database_engine(settings)
    stack.callback(engine.dispose)
    client = OpenSearchClient(settings)
    stack.callback(client.close)
    search = Bm25SearchService(client, alias=settings.opensearch_index_alias)
    embedder = None
    try:
        provider = create_query_embedding_provider(settings)
        if provider is not None:
            stack.callback(provider.close)
            embedder = QueryEmbeddingService(provider)
    except EmbeddingProviderError:
        embedder = None
    retrieval = HybridRetrievalService(
        client, alias=settings.opensearch_index_alias, query_embedder=embedder
    )
    acquisition: LiteratureAcquisition | None = None
    if settings.research_acquisition_enabled:
        scientific: ScientificRetrievalService | None = build_research_retrieval(settings, engine)
        acquisition, ingest_engine = build_research_acquisition(settings)
        stack.callback(ingest_engine.dispose)
        stack.callback(acquisition.qdrant.close)
    else:
        scientific = (
            ScientificRetrievalService(engine, settings)
            if settings.scientific_qdrant_enabled
            else None
        )
    if scientific is not None:
        stack.callback(scientific.close)
        scientific.start()
    runtime = build_agent_services(
        settings=settings,
        engine=engine,
        scientific=scientific,
        search=search,
        retrieval=retrieval,
        acquisition=acquisition,
    )
    if runtime is None:
        stack.close()
        raise RuntimeError("agent_enabled is false")
    runtime.pool = None
    runtime._cleanups.append(stack.close)  # pyright: ignore[reportPrivateUsage]
    return runtime
