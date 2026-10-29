"""Actual local RAG answer receipts; independently verify citations and immutable corpus."""
# ruff: noqa: T201 -- operator execution evidence

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

ROOT = Path(__file__).resolve().parents[1] / ".dynamisrag/live-rag-002"
QUESTION = (
    "How does starting knee flexion angle influence vastus medialis and vastus lateralis "
    "activation during resisted plyometric exercise?"
)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    engine = create_database_engine(load_settings())
    qdrant = QdrantClient()

    def snapshot() -> dict[str, object]:
        with Session(engine) as s:
            counts = [
                s.scalar(select(func.count()).select_from(m))
                for m in (DocumentRecord, DocumentVersionRecord, PassageRecord)
            ]
        return {
            "canonical_counts": counts,
            "aliases": qdrant.aliases(),
            "collections": {
                n: hashlib.sha256(
                    json.dumps(
                        [p.model_dump(mode="json") for p in qdrant.scroll(n)], sort_keys=True
                    ).encode()
                ).hexdigest()
                for n in sorted(qdrant.collections())
            },
        }

    try:
        before = snapshot()
        with httpx2.Client(timeout=430, trust_env=False) as client:
            running = client.get("http://127.0.0.1:11434/api/ps").json()
            first_label = "warm-first" if running["models"] else "cold"
            (ROOT / "model-before-first-request.json").write_text(
                json.dumps(running, indent=2), encoding="utf-8", newline="\n"
            )
            for label, question in [
                (first_label, QUESTION),
                ("warm", QUESTION),
                (
                    "out-of-corpus",
                    "What determines neutrino mass ordering and leptonic CP violation?",
                ),
            ]:
                started = perf_counter()
                response = client.post(
                    "http://127.0.0.1:8021/answer/scientific", json={"question": question}
                )
                wall = (perf_counter() - started) * 1000
                receipt = {
                    "observed_at_utc": datetime.now(UTC).isoformat(),
                    "request": {"question": question},
                    "http_status": response.status_code,
                    "client_wall_ms": wall,
                    "response": response.json(),
                }
                (ROOT / f"{label}.json").write_text(
                    json.dumps(receipt, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                    newline="\n",
                )
                if response.status_code != 200:
                    raise RuntimeError(
                        f"answer API rejected {label}: {response.status_code} {response.text}"
                    )
                answer = receipt["response"]
                if label == "out-of-corpus":
                    if (
                        answer["status"] != "insufficient_evidence"
                        or answer["claims"]
                        or answer["answer"] is not None
                    ):
                        raise ValueError("out-of-corpus answer did not abstain")
                else:
                    if (
                        answer["status"] != "supported"
                        or not answer["generation_invoked"]
                        or not answer["claims"]
                    ):
                        raise ValueError(f"no supported generated answer: {answer['reason']}")
                    excerpts = {p["passage_key"]: p["text"] for p in answer["context_excerpts"]}
                    with Session(engine) as session:
                        for claim in answer["claims"]:
                            source = claim["source"]
                            verify_point(
                                session, Point(id="independent-answer-verifier", payload=source)
                            )
                            quote = claim["supporting_excerpt"]
                            if (
                                quote != claim["text"]
                                or quote not in excerpts[source["passage_key"]]
                                or source["text"][claim["excerpt_start"] : claim["excerpt_end"]]
                                != quote
                            ):
                                raise ValueError(
                                    "generated quotation does not match actual evidence"
                                )
                print(
                    json.dumps(
                        {
                            "request_type": label,
                            "status": answer["status"],
                            "answer": answer["answer"],
                            "claims": len(answer["claims"]),
                            "selected_evidence": len(answer["evidence"]),
                            "generation_invoked": answer["generation_invoked"],
                            "timings": answer["timings"],
                            "model_metrics": answer["generation_metrics"],
                            "client_wall_ms": wall,
                        },
                        ensure_ascii=True,
                    ),
                    flush=True,
                )
        after = snapshot()
        if before != after:
            raise ValueError("canonical corpus or collection contents changed")
        (ROOT / "state-unchanged.json").write_text(
            json.dumps({"before": before, "after": after, "unchanged": True}, indent=2),
            encoding="utf-8",
            newline="\n",
        )
        print(
            "grounded answers and out-of-corpus abstention verified; "
            "residency labels retained; corpus unchanged"
        )
    finally:
        qdrant.close()
        engine.dispose()


if __name__ == "__main__":
    main()
