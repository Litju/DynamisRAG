"""Reusable information-retrieval evaluation contracts (RES-140).

Independent of the frozen RES-138 model-selection benchmark and of production
search. Importing this package does not contact PostgreSQL, OpenSearch or TEI.
"""

from dynamisrag.ir.contracts import (
    IR_CONTRACT_REVISION,
    IrContractError,
    IrDataset,
    IrExperimentConfig,
    IrHit,
    IrQrel,
    IrQuery,
    IrRun,
    canonical_ir_json,
    trec_qrels,
    trec_run,
)

__all__ = [
    "IR_CONTRACT_REVISION",
    "IrContractError",
    "IrDataset",
    "IrExperimentConfig",
    "IrHit",
    "IrQrel",
    "IrQuery",
    "IrRun",
    "canonical_ir_json",
    "trec_qrels",
    "trec_run",
]
