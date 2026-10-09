"""Reusable information-retrieval evaluation contracts (RES-140).

Independent of the frozen RES-138 model-selection benchmark and of production
search. Importing this package does not contact PostgreSQL, OpenSearch or TEI.
"""

from dynamisrag.ir.artifacts import (
    read_ir_inputs,
    read_verified_ir_evaluation,
    verify_ir_bundle,
    write_ir_bundle,
)
from dynamisrag.ir.contracts import (
    IR_CONTRACT_REVISION,
    IR_METRIC_POLICY_REVISION,
    IrContractError,
    IrDataset,
    IrExperimentConfig,
    IrHit,
    IrMetricPolicy,
    IrPassageHit,
    IrPassageMapEntry,
    IrPassageMapping,
    IrQrel,
    IrQuery,
    IrRun,
    canonical_ir_json,
    document_run_from_passages,
    trec_qrels,
    trec_run,
)

__all__ = [
    "IR_CONTRACT_REVISION",
    "IR_METRIC_POLICY_REVISION",
    "IrContractError",
    "IrDataset",
    "IrExperimentConfig",
    "IrHit",
    "IrMetricPolicy",
    "IrPassageHit",
    "IrPassageMapEntry",
    "IrPassageMapping",
    "IrQrel",
    "IrQuery",
    "IrRun",
    "canonical_ir_json",
    "document_run_from_passages",
    "read_ir_inputs",
    "read_verified_ir_evaluation",
    "trec_qrels",
    "trec_run",
    "verify_ir_bundle",
    "write_ir_bundle",
]
