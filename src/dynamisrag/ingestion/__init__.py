"""Source acquisition: Europe PMC full-text ingestion (RES-132)."""

from __future__ import annotations

from dynamisrag.ingestion.acquisition import AcquisitionResult, EuropePmcAcquisition
from dynamisrag.ingestion.europe_pmc import (
    AcquiredFulltext,
    EuropePmcClient,
    EuropePmcError,
    EuropePmcInvalidPmcid,
    EuropePmcNotAvailable,
    EuropePmcUnexpectedResponse,
)

__all__ = [
    "AcquiredFulltext",
    "AcquisitionResult",
    "EuropePmcAcquisition",
    "EuropePmcClient",
    "EuropePmcError",
    "EuropePmcInvalidPmcid",
    "EuropePmcNotAvailable",
    "EuropePmcUnexpectedResponse",
]
