"""Deterministic logging setup for the local runtime.

Deliberately built on the standard library only: no logging framework is added
to the dependency set for the foundation slice, and the format is a plain
single-line record so that local PowerShell output and container log scraping
agree.
"""

from __future__ import annotations

import logging
import sys
from typing import Final

__all__ = ["APP_LOGGER_NAME", "LOG_FORMAT", "configure_logging"]

APP_LOGGER_NAME: Final[str] = "dynamisrag"
"""Root logger name for every module in the package."""

LOG_FORMAT: Final[str] = "%(asctime)s %(levelname)-8s %(name)s %(message)s"


def configure_logging(level: str) -> None:
    """Install a single stderr handler and apply ``level`` to the root logger.

    ``force=True`` replaces any handler installed by an importing host (pytest,
    uvicorn, alembic) so repeated calls cannot accumulate duplicate records.
    """
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        stream=sys.stderr,
        force=True,
    )
