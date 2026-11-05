"""Bounded restart readiness polling, including connection resets and refusals."""

import argparse
import math
import time

import httpx2


def wait_ready(*, timeout_seconds: float = 60, port: int = 6333) -> None:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("readiness timeout must be positive and finite")
    if not 1 <= port <= 65535:
        raise ValueError("Qdrant port must be between 1 and 65535")
    deadline = time.monotonic() + timeout_seconds
    with httpx2.Client(trust_env=False, follow_redirects=False) as client:
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                response = client.get(f"http://127.0.0.1:{port}/readyz", timeout=min(2, remaining))
                if response.status_code == 200:
                    return
            except httpx2.HTTPError:
                # Startup may accept TCP and then reset it before HTTP is ready.
                pass
            time.sleep(min(0.5, max(0, deadline - time.monotonic())))
    raise RuntimeError("Qdrant did not become ready before the restart deadline")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=float, default=60)
    parser.add_argument("--port", type=int, default=6333)
    args = parser.parse_args()
    wait_ready(timeout_seconds=args.timeout_seconds, port=args.port)
