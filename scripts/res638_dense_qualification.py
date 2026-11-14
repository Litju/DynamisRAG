"""RES-638 dense-retrieval qualification against a fresh, isolated Qdrant projection.

This operator script does not read or write the retained Qdrant instance, its
aliases, historical receipts or snapshots. It refuses a non-empty target and
creates exactly one new collection plus one qualification alias.

    uv run python -m scripts.res638_dense_qualification \
        --qdrant-url http://127.0.0.1:16335 --tei-url http://127.0.0.1:8082

A failed check writes ``DENSE_RETRIEVAL_NOT_QUALIFIED`` and exits non-zero. It
never substitutes another embedding model or claims success from a partial run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from dynamisrag.config import Settings, to_sqlalchemy_url
from dynamisrag.db.canonical import list_passage_projection_records
from dynamisrag.db.models import DocumentVersionRecord, PassageRecord
from dynamisrag.embedding import (
    EmbeddingInput,
    EmbeddingModelIdentity,
    EmbeddingRetryPolicy,
    EmbeddingRuntimeConfig,
    ExpectedTeiModel,
    TeiEmbeddingProvider,
    embed_passages,
    passage_content_sha256,
)
from dynamisrag.embedding.scientific_cpu import (
    DOCUMENT_CONFIG,
    DOCUMENT_FINGERPRINT,
    MODEL_ID,
    MODEL_REVISION,
    QUERY_CONFIG,
    QUERY_FINGERPRINT,
    CpuDeployment,
)
from dynamisrag.search.projection import build_projection_manifest
from dynamisrag.search.qdrant import (
    QdrantClient,
    QdrantProjection,
    QdrantProjector,
    QdrantRetrieval,
    validate_vector,
)
from dynamisrag.search.scientific import (
    CHUNKER,
    read_canonical,
    require_cpu_identity,
    verify_canonical,
)

QUALIFICATION_ALIAS = "dynamisrag-res638-dense-qualification-v1"

_QUESTIONS = (
    (
        "positive_achilles",
        "What happened to Achilles tendon stiffness during depth jumping?",
        ("PMC5263151",),
    ),
    (
        "positive_knee_flexion",
        "How did knee flexion angle affect muscle activity?",
        ("PMC10780924",),
    ),
    (
        "positive_comparison",
        "Compare the study designs and populations of the plyometric studies.",
        ("PMC5263151", "PMC10780924", "PMC10680285"),
    ),
    (
        "negative_kepler",
        "What is the orbital period of the exoplanet Kepler-22b?",
        (),
    ),
    (
        "negative_sourdough",
        "What hydration ratio do the studies recommend for sourdough bread?",
        (),
    ),
)


def _engine() -> Engine:
    load_dotenv(".env.workspace")
    load_dotenv()
    settings = Settings()
    url = to_sqlalchemy_url(settings.database_url)
    if "workspace_reader" not in url:
        raise SystemExit("Refusing a non-read-only licensed-corpus URL")
    return create_engine(url)


def _corpus(engine: Engine) -> tuple[str, dict[str, str], dict[str, object]]:
    with Session(engine) as session:
        versions = tuple(
            session.execute(
                select(DocumentVersionRecord.version_key, DocumentVersionRecord.title).order_by(
                    DocumentVersionRecord.version_key
                )
            ).all()
        )
        passages = session.scalar(select(func.count()).select_from(PassageRecord)) or 0
    titles = " ".join(title for _, title in versions)
    if len(versions) != 3 or passages != 75:
        raise SystemExit(
            f"Expected the licensed 3-paper / 75-passage corpus, got {len(versions)} / {passages}"
        )
    # Map PMIDs to version keys through the canonical titles used by this corpus.
    pmcid_to_version: dict[str, str] = {}
    for key, title in versions:
        lowered = title.casefold()
        if "knee flexion" in lowered:
            pmcid_to_version["PMC10780924"] = key
        elif "cricketers" in lowered or "complex training" in lowered:
            pmcid_to_version["PMC10680285"] = key
        elif "depth jumping" in lowered:
            pmcid_to_version["PMC5263151"] = key
    if len(pmcid_to_version) != 3:
        raise SystemExit(f"Could not map all three PMCIDs from titles: {titles[:200]}")
    receipt = {
        "document_versions": [
            {"version_key": key, "title": title[:300]} for key, title in versions
        ],
        "passage_count": passages,
        "document_count": len(versions),
        "chunker_revision": CHUNKER,
    }
    return hashlib.sha256(titles.encode()).hexdigest(), pmcid_to_version, receipt


def _provider(base_url: str, *, document: bool) -> TeiEmbeddingProvider:
    return TeiEmbeddingProvider(
        base_url=base_url.rstrip("/"),
        expected_model=ExpectedTeiModel(MODEL_ID, MODEL_REVISION),
        deployment_semantics=CpuDeployment(),
        generation_config=DOCUMENT_CONFIG if document else QUERY_CONFIG,
        runtime_config=EmbeddingRuntimeConfig(
            batch_size=1,
            timeout_seconds=180,
            retry=EmbeddingRetryPolicy(max_attempts=1),
        ),
        verify_tls=False,
    )


def _write(path: Path, receipt: dict[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:  # noqa: PLR0912, PLR0915 - explicit operator qualification stages
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:16335")
    parser.add_argument("--tei-url", default="http://127.0.0.1:8082")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/evidence/res638-dense-qualification/receipt.json"),
    )
    args = parser.parse_args()
    base_receipt: dict[str, object] = {
        "schema": "res638-dense-retrieval-qualification-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "code_revision": subprocess.check_output(  # noqa: S603 - fixed local command
            [shutil.which("git") or "git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "pinned": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "document_config_sha256": DOCUMENT_FINGERPRINT,
            "query_config_sha256": QUERY_FINGERPRINT,
            "chunker_revision": CHUNKER,
            "dimension": 512,
            "tei_image": "ghcr.io/huggingface/text-embeddings-inference:cpu-1.9.4",
        },
        "isolation": {
            "qdrant_url": args.qdrant_url,
            "tei_url": args.tei_url,
            "retained_qdrant_reused": False,
            "retained_receipts_reused": False,
        },
    }
    engine = _engine()
    corpus_sha, pmcid_to_version, corpus = _corpus(engine)
    base_receipt["corpus"] = corpus
    base_receipt["corpus_sha256"] = corpus_sha

    client = QdrantClient(args.qdrant_url)
    provider = _provider(args.tei_url, document=True)
    query_provider = _provider(args.tei_url, document=False)
    try:
        if client.request("GET", "/").get("version") != "1.20.0":
            raise RuntimeError("qdrant_version_mismatch")
        if client.collections() or client.aliases():
            raise RuntimeError("qdrant_target_not_fresh")
        identity = provider.describe()
        require_cpu_identity(identity)
        base_receipt["observed_embedding_identity"] = {
            "provider": identity.provider,
            "protocol_revision": identity.protocol_revision,
            "runtime_version": identity.runtime_version,
            "runtime_sha": identity.runtime_sha,
            "model_id": identity.model_id,
            "model_sha": identity.model_sha,
            "model_dtype": identity.model_dtype,
            "model_pooling": identity.model_pooling,
            "max_input_length": identity.max_input_length,
            "max_client_batch_size": identity.max_client_batch_size,
            "deployment": identity.deployment.payload(),
        }

        with Session(engine) as session:
            records = tuple(list_passage_projection_records(session, chunker_revision=CHUNKER))
        passages = build_projection_manifest(records, chunker_revision=CHUNKER)
        if passages.document_count != 75:
            raise RuntimeError("projection_passage_count_mismatch")
        inputs = tuple(
            EmbeddingInput(
                passage_key=document.passage_key,
                content_sha256=document.content_sha256,
                text=document.text,
            )
            for document in passages.documents
        )
        embeddings = embed_passages(provider, inputs)
        if (
            embeddings.embedding_config_sha256 != DOCUMENT_FINGERPRINT
            or embeddings.dimension != 512
            or embeddings.document_count != 75
        ):
            raise RuntimeError("document_embedding_fingerprint_mismatch")
        projection = QdrantProjection(passages, embeddings, DOCUMENT_FINGERPRINT)
        projector = QdrantProjector(client)
        projector.create(projection)
        projector.populate(projection)
        projector.verify(projection)
        projector.publish(
            projection,
            alias=QUALIFICATION_ALIAS,
            expected_previous=None,
        )
        observed = client.scroll(projection.collection_name)
        if len(observed) != 75:
            raise RuntimeError("projection_membership_mismatch")
        expected_keys = {document.passage_key for document in passages.documents}
        observed_keys = {str(point.payload["passage_key"]) for point in observed if point.payload}
        if observed_keys != expected_keys:
            raise RuntimeError("canonical_passage_membership_mismatch")
        norms = [
            math.sqrt(sum(value * value for value in point.vector["dense"])) for point in observed
        ]
        if not all(math.isclose(value, 1.0, abs_tol=2e-3) for value in norms):
            raise RuntimeError("vector_norm_not_normalized")
        if any(len(point.vector["dense"]) != 512 for point in observed):
            raise RuntimeError("vector_dimension_mismatch")

        adapter = QdrantRetrieval(
            client,
            alias=QUALIFICATION_ALIAS,
            projection=projection,
            query_identity=EmbeddingModelIdentity(MODEL_ID, MODEL_REVISION, QUERY_FINGERPRINT),
        )
        adapter.resolve()
        retrieval_results: list[dict[str, object]] = []
        positives: list[str] = []
        for case_id, question, expected_pmcids in _QUESTIONS:
            query_digest = passage_content_sha256(question)
            query_manifest = embed_passages(
                query_provider,
                (EmbeddingInput(query_digest, query_digest, question),),
            )
            if query_manifest.embedding_config_sha256 != QUERY_FINGERPRINT:
                raise RuntimeError("query_embedding_fingerprint_mismatch")
            vector = validate_vector(query_manifest.entries[0].values)
            if not math.isclose(math.hypot(*vector), 1.0, abs_tol=1e-5):
                raise RuntimeError("query_vector_not_normalized")
            lexical, dense, fusion = adapter.hybrid(
                question,
                vector,
                embedding_config_sha256=QUERY_FINGERPRINT,
                limit=5,
            )
            points = tuple({point.id: point for point in (*lexical.points, *dense.points)}.values())
            verify_canonical(
                read_canonical(
                    engine, tuple(str(point.payload["passage_key"]) for point in points)
                ),
                points,
            )
            top = [
                {
                    "rank": hit.final_rank,
                    "score": hit.rrf_score,
                    "passage_key": hit.passage_key,
                    "title": hit.provenance.title[:200],
                    "document_version_key": hit.provenance.document_version_key,
                }
                for hit in fusion.hits
            ]
            expected_versions = {pmcid_to_version[pmcid] for pmcid in expected_pmcids}
            recalled = bool(expected_versions) and any(
                item["document_version_key"] in expected_versions for item in top
            )
            if expected_pmcids and not recalled:
                raise RuntimeError(f"expected_document_not_recalled:{case_id}")
            if expected_pmcids:
                positives.append(case_id)
            retrieval_results.append(
                {
                    "case_id": case_id,
                    "question": question,
                    "expected_pmcids": list(expected_pmcids),
                    "expected_document_recalled": recalled,
                    "dense_hits": len(dense.points),
                    "lexical_hits": len(lexical.points),
                    "top_fused": top,
                }
            )

        isolation_after = {
            "collections": sorted(client.collections()),
            "aliases": client.aliases(),
        }
        receipt = {
            **base_receipt,
            "status": "QUALIFIED",
            "collection": projection.collection_name,
            "alias": QUALIFICATION_ALIAS,
            "embedding_manifest_sha256": projection.embeddings.manifest_sha256,
            "projection_sha256": projection.passages.projection_sha256,
            "collection_identity": projection.identity,
            "point_count": len(observed),
            "vector_dimension": 512,
            "vector_norm_min": min(norms),
            "vector_norm_max": max(norms),
            "canonical_passage_membership": "exact",
            "retrieval": retrieval_results,
            "positive_cases_recalled": positives,
            "isolation_after": isolation_after,
            "limitations": [
                "Dense retrieval scores are not semantic entailment or scientific validity.",
                "This qualifies the pinned CPU TEI model and a fresh local projection only.",
                "Off-domain queries still return nearest neighbours; abstention is the "
                "answerability gate's job.",
                "No hosted Qdrant/TEI, load test, or paid inference was used.",
            ],
        }
        digest = _write(args.output, receipt)
        sys.stdout.write(
            json.dumps(
                {
                    "status": receipt["status"],
                    "output": str(args.output),
                    "sha256": digest,
                    "collection": projection.collection_name,
                    "points": len(observed),
                    "positive_cases": len(positives),
                }
            )
            + "\n"
        )
    except Exception as error:
        receipt = {
            **base_receipt,
            "status": "DENSE_RETRIEVAL_NOT_QUALIFIED",
            "error_type": type(error).__name__,
            "error_code": str(error)[:240],
            "limitations": [
                "No success is claimed. No alternative embedding model was substituted.",
                "The qualification target was fresh and isolated; no retained Qdrant "
                "data was read.",
            ],
        }
        digest = _write(args.output, receipt)
        sys.stdout.write(
            json.dumps({"status": receipt["status"], "output": str(args.output), "sha256": digest})
            + "\n"
        )
        raise SystemExit(1) from error
    finally:
        provider.close()
        query_provider.close()
        client.close()


if __name__ == "__main__":
    main()
