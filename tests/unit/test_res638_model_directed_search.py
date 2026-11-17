"""RES-638: the research model chooses the literature search; the graph runs it once."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from sqlalchemy.engine import Engine

from dynamisrag.agent.graph import ConversationalResearchOrchestrator
from dynamisrag.agent.persistence import InMemoryAgentConversationStore
from dynamisrag.agent.tools import ScientificToolRegistry
from tests.agent_stack import scripted_runtime, tool_call_message
from tests.unit.test_agent_graph import (
    graph_engine_and_studies,  # noqa: F401  # pyright: ignore[reportUnusedImport]
)
from tests.unit.test_agent_routing_graph import (
    _SeedingAcquisition,  # pyright: ignore[reportPrivateUsage]
)

_USER = "researcher-alpha"
SEARCHES = [{"all_of": [["linear position transducer", "LPT"], ["force plate", "criterion"]]}]


class _RecordingAcquisition(_SeedingAcquisition):
    def __init__(self, engine: Engine) -> None:
        super().__init__(engine)
        self.searches: list[Any] = []

    def acquire(
        self,
        question: str,
        *,
        context: str = "",
        should_stop: Any = None,
        progress: Any = None,
        model_searches: Any = (),
    ) -> dict[str, Any]:
        self.searches.append(model_searches)
        return super().acquire(
            question, context=context, should_stop=should_stop, progress=progress
        )


def _script(messages: list[BaseMessage], bound: list[str], index: int) -> AIMessage:
    del index
    text = "\n".join(str(m.content) for m in messages)
    marker = "CANDIDATE_EVIDENCE_JSON:"
    if marker in text:
        item = json.loads(text[text.index(marker) + len(marker) :].strip())[0]
        claim = {k: item[k] for k in ("passage_key", "span_key", "verbatim_quote")}
        claim |= {"claim_text": item["verbatim_quote"], "aspect": "validity"}
        claim["support"] = "study_finding"
        return AIMessage(content=json.dumps({"claims": [claim], "evidence_gaps": ["no RCT"]}))
    if isinstance(messages[-1], ToolMessage):
        return AIMessage(content="done")
    system = str(messages[0].content) if isinstance(messages[0], SystemMessage) else ""
    if "LITERATURE_SEARCH_DONE" in system:
        assert "discover_external_literature" not in bound  # once per run
        return tool_call_message(
            ("hybrid_search", {"query": "linear position transducer validity", "limit": 5})
        )
    assert "discover_external_literature" in bound
    return tool_call_message(
        ("discover_external_literature", {"searches": SEARCHES, "rationale": "no LPT study"})
    )


def test_model_chooses_the_literature_search_then_searches_the_new_corpus(
    graph_engine_and_studies: tuple[Engine, Any],  # noqa: F811
) -> None:
    engine, _ = graph_engine_and_studies
    acquisition = _RecordingAcquisition(engine)
    runtime, _ = scripted_runtime(_script)
    store = InMemoryAgentConversationStore()
    orch = ConversationalResearchOrchestrator(
        tool_registry=ScientificToolRegistry(engine),
        store=store,
        model_runtime=runtime,
        literature_acquisition=acquisition,
    )
    thread = store.create_thread(owner_id=_USER, title="t")
    run = orch.execute_run(
        thread_id=thread.thread_id,
        user_id=_USER,
        question=(
            "What is the validity of a linear position transducer against a force plate reference?"
        ),
        idempotency_key="model-acq",
    )
    assert acquisition.searches == [
        [[["linear position transducer", "LPT"], ["force plate", "criterion"]]]
    ]
    acq = run.telemetry["literature_acquisition"]
    assert acq["origin"] == "model_directed"
    assert acq["model_rationale"] == "no LPT study"
    called = [
        e.payload
        for e in store.list_events(run.run_id, user_id=_USER)
        if e.event_type == "tool.called"
    ]
    names = [c["tool_name"] for c in called]
    assert names[:2] == ["discover_external_literature", "discover_external_literature"]
    assert {c["origin"] for c in called[:2]} == {"model_directed"}
    assert any(n == "hybrid_search" for n in names)
    assert run.telemetry["tool_origins"]["graph_fallback"] == 0
    assert run.telemetry["synthesis_source"] == "model", (run.status, run.abstention_reason)
    assert "Aspect `validity`" in (run.answer_markdown or "")
    assert "Gap: no RCT" in (run.answer_markdown or "")
