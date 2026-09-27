"""Console entry point: ``python -m dynamisrag`` / ``dynamisrag``.

Runs the development server bound to the configured host and port. For a
production process prefer the ASGI factory directly so no reload or
auto-discovery behaviour is implied::

    uvicorn --factory dynamisrag.application:create_app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import uvicorn

from dynamisrag.application import create_app
from dynamisrag.config import load_settings
from dynamisrag.logging_config import configure_logging

__all__ = ["main"]


def main() -> None:
    """Load configuration, install logging and serve the application."""
    settings = load_settings()
    configure_logging(settings.log_level)
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
