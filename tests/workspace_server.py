"""Loopback browser test backend: synthetic corpus and deterministic inference only."""

import uvicorn

from tests.agent_stack import build_stack


def main() -> None:
    stack = build_stack(rate_limit_per_minute=10000)
    pool = stack.pool()
    pool.start()
    try:
        uvicorn.run(stack.app, host="127.0.0.1", port=8018, log_level="warning")
    finally:
        pool.stop()
        stack.engine.dispose()


if __name__ == "__main__":
    main()
