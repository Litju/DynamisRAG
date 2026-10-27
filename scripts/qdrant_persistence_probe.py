"""Read-only synthetic DB qualification before/after restart. Never inserts points."""

import argparse
import json
from pathlib import Path

from tests.integration.test_qdrant_live import assert_source_spans
from tests.unit.test_qdrant import SYNTHETIC_LABEL, synthetic_projection

from dynamisrag.search.qdrant import (
    QdrantClient,
    QdrantProjector,
    QdrantRetrieval,
    digest,
)


def observe() -> dict[str, object]:
    client = QdrantClient()
    try:
        projection = synthetic_projection()
        projector = QdrantProjector(client)
        projector.verify(projection)
        projector.verify(synthetic_projection(alternate=True))
        service = QdrantRetrieval(
            client,
            alias="dynamisrag-synthetic-database-tests",
            document_identity=projection.embeddings.embedding_model_identity,
            query_identity=projection.embeddings.embedding_model_identity,
        )
        lexical, dense, fused = service.hybrid(
            "probiotic",
            projection.embeddings.entries[0].values,
            embedding_config_sha256=projection.expected_embedding_config_sha256,
        )
        assert_source_spans(dense.points)
        collections = {}
        for name in sorted(client.collections()):
            points = client.scroll(name)
            collections[name] = {
                "points": len(points),
                "stored_points_sha256": digest([p.model_dump(mode="json") for p in points]),
            }
        return {
            "label": SYNTHETIC_LABEL,
            "version": client.request("GET", "/")["version"],
            "collections": collections,
            "aliases": client.aliases(),
            "dense": [
                {"passage_key": p.payload["passage_key"], "score": p.score} for p in dense.points
            ],
            "bm25": [
                {"passage_key": p.payload["passage_key"], "score": p.score} for p in lexical.points
            ],
            "rrf": fused.model_dump(mode="json"),
            "source_span_reconstruction": "passed",
        }
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("snapshot", "verify"))
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    observed = observe()
    if args.action == "snapshot":
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(observed, sort_keys=True, indent=2), encoding="utf-8")
    elif json.loads(args.receipt.read_text(encoding="utf-8")) != observed:
        raise RuntimeError("post-restart collection, vectors, aliases or retrieval changed")
    print(json.dumps({"action": args.action, "passed": True, **observed}, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
