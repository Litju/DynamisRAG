"""Read-only HTTP/SQL evidence for LIVE-RAG-001; no ingestion or document inference."""
# ruff: noqa: T201 -- execution receipt

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import httpx2
from real_scientific_retrieval import verify_point
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from dynamisrag.config import load_settings
from dynamisrag.db import create_database_engine
from dynamisrag.db.models import DocumentRecord, DocumentVersionRecord, PassageRecord
from dynamisrag.search.qdrant import Point, QdrantClient

ROOT = Path(__file__).resolve().parents[1] / ".dynamisrag/live-rag-001/readiness-update"
QUESTION = (
    "How does starting knee flexion angle influence muscle activation during plyometric exercises?"
)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    engine = create_database_engine(load_settings())
    qdrant = QdrantClient()

    def snapshot() -> dict[str, object]:
        with Session(engine) as session:
            counts = [
                session.scalar(select(func.count()).select_from(model))
                for model in (DocumentRecord, DocumentVersionRecord, PassageRecord)
            ]
        return {
            "canonical_counts": counts,
            "aliases": qdrant.aliases(),
            "collections": {
                name: hashlib.sha256(
                    json.dumps(
                        [p.model_dump(mode="json") for p in qdrant.scroll(name)],
                        sort_keys=True,
                    ).encode()
                ).hexdigest()
                for name in sorted(qdrant.collections())
            },
        }

    try:
        before = snapshot()
        executions = []
        requests = [
            {"question": QUESTION, "limit": 3},
            {
                "question": (
                    "How do vastus medialis and vastus lateralis responses vary "
                    "between dominant and nondominant legs?"
                ),
                "limit": 3,
            },
            {"question": QUESTION, "limit": 3},
            {"question": QUESTION, "limit": 3, "filters": {"source_external_id": "PMC10780924"}},
            {"question": QUESTION, "limit": 3, "filters": {"language": "zz"}},
        ]
        with httpx2.Client(
            base_url="http://127.0.0.1:8011", timeout=130, trust_env=False
        ) as client:
            readiness_before = client.get("/readyz/qdrant")
            if readiness_before.status_code != 200:
                raise RuntimeError("scientific service not ready")
            for body in requests:
                start = perf_counter()
                response = client.post("/retrieve/qdrant", json=body)
                wall = (perf_counter() - start) * 1000
                if response.status_code != 200:
                    raise RuntimeError(
                        f"API rejected probe: {response.status_code} {response.text}"
                    )
                result = response.json()
                checked = set()
                with Session(engine) as session:
                    for lane in ("bm25", "dense"):
                        for candidate in result[lane]["candidates"]:
                            verify_point(
                                session,
                                Point(id=candidate["point_id"], payload=candidate["provenance"]),
                            )
                            checked.add(candidate["provenance"]["passage_key"])
                    for fused in result["fusion"]["hits"]:
                        if fused["passage_key"] not in checked:
                            raise ValueError(
                                "fused citation absent from independently verified lanes"
                            )
                executions.append(
                    {
                        "request": body,
                        "http_status": response.status_code,
                        "client_wall_ms": wall,
                        "independently_verified_passages": len(checked),
                        "response": result,
                    }
                )
                print(
                    json.dumps(
                        {
                            "question": body["question"],
                            "filters": body.get("filters", {}),
                            "timings": result["timings"],
                            "client_wall_ms": wall,
                            "verified_passages": len(checked),
                            "top_fused": [
                                {
                                    "passage_key": h["passage_key"],
                                    "pmcid": h["provenance"]["pmcid"],
                                    "score": h["rrf_score"],
                                }
                                for h in result["fusion"]["hits"]
                            ],
                        },
                        ensure_ascii=True,
                    ),
                    flush=True,
                )
            invalid = client.post("/retrieve/qdrant", json={"question": " "})
            if invalid.status_code != 422:
                raise ValueError("invalid query did not fail closed")
            readiness_after = client.get("/readyz/qdrant")
            if readiness_after.status_code != 200:
                raise RuntimeError("scientific service lost readiness")
        after = snapshot()
        if before != after:
            raise ValueError("corpus, aliases or stored vectors changed")
        receipt = {
            "observed_at_utc": datetime.now(UTC).isoformat(),
            "executions": executions,
            "before": before,
            "after": after,
            "state_unchanged": True,
            "invalid_query_response": invalid.json(),
            "readiness_before": readiness_before.json(),
            "readiness_after": readiness_after.json(),
        }
        (ROOT / "observed-responses.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
        )
        print("five fresh query requests; independent citations verified; stored corpus unchanged")
    finally:
        qdrant.close()
        engine.dispose()


if __name__ == "__main__":
    main()
