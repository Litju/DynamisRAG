"""Loopback browser test backend: synthetic corpus and deterministic inference only."""

import time

import uvicorn
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.outputs import ChatResult

from dynamisrag.agent.providers import DeterministicResearchChatModel
from tests.agent_stack import build_stack, runtime_for


class BrowserTestModel(DeterministicResearchChatModel):
    """Explicit test-only fault markers exercise cancellation and recovery through the UI."""

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        question = next(
            (str(m.content) for m in reversed(messages) if isinstance(m, HumanMessage)), ""
        )
        if "[browser-slow]" in question:
            time.sleep(3)
        if "[browser-failure]" in question:
            raise RuntimeError("test-only worker failure")
        return super()._generate(messages, stop, run_manager, **kwargs)


def main() -> None:
    stack = build_stack(rate_limit_per_minute=10000, runtime=runtime_for(BrowserTestModel()))
    pool = stack.pool()
    pool.start()
    try:
        uvicorn.run(stack.app, host="127.0.0.1", port=8018, log_level="warning")
    finally:
        pool.stop()
        stack.engine.dispose()


if __name__ == "__main__":
    main()
