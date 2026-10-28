"""PostgreSQL snapshot + retained vectors -> Qdrant; never invokes inference."""

import argparse
import json
from pathlib import Path

from sqlalchemy.orm import Session

from dynamisrag.config import load_settings
from dynamisrag.db import create_database_engine
from dynamisrag.db.canonical import list_passage_projection_records
from dynamisrag.search.projection import build_projection_manifest
from dynamisrag.search.qdrant import QdrantClient, QdrantProjection, QdrantProjector
from dynamisrag.search.qdrant_artifact import load_embedding_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("populate", "verify", "publish"))
    parser.add_argument("--embedding-manifest", type=Path, required=True)
    parser.add_argument("--artifact-sha256", required=True)
    parser.add_argument("--embedding-config-sha256", required=True)
    parser.add_argument("--chunker-revision", required=True)
    parser.add_argument("--alias", default="dynamisrag-passages-qdrant")
    parser.add_argument(
        "--expected-previous", help="Required for replacement; physical target name"
    )
    args = parser.parse_args()
    embeddings = load_embedding_manifest(
        args.embedding_manifest, expected_sha256=args.artifact_sha256
    )
    engine = create_database_engine(load_settings())
    try:
        with Session(engine) as session:
            session.connection(execution_options={"isolation_level": "REPEATABLE READ"})
            records = list_passage_projection_records(
                session, chunker_revision=args.chunker_revision
            )
            passages = build_projection_manifest(records, chunker_revision=args.chunker_revision)
            projection = QdrantProjection(passages, embeddings, args.embedding_config_sha256)
        client = QdrantClient()
        try:
            projector = QdrantProjector(client)
            if args.action == "populate":
                projector.create(projection)
                projector.populate(projection)
            projector.verify(projection)
            if args.action == "publish":
                projector.publish(
                    projection, alias=args.alias, expected_previous=args.expected_previous
                )
            print(  # noqa: T201 - operator CLI output
                json.dumps(
                    {
                        "collection": projection.collection_name,
                        "points": passages.document_count,
                        "identity": projection.identity,
                        "aliases": client.aliases(),
                    },
                    indent=2,
                )
            )
        finally:
            client.close()
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
