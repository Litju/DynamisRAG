"""Explicit bounded corpus operator: acquire/resume or offline verify; never index."""
# ruff: noqa: T201 -- explicit operator output

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx2

from dynamisrag.config import load_settings
from dynamisrag.corpus.discovery import BoundedHttp, DiscoveryPlan
from dynamisrag.corpus.pipeline import run_snapshot
from dynamisrag.corpus.snapshot import publish_json, read_json, verify_run
from dynamisrag.db import create_database_engine
from dynamisrag.ingestion.europe_pmc import EuropePmcClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("acquire", "verify"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--store-root", type=Path, default=Path(".dynamisrag/real-science-v1/sources")
    )
    parser.add_argument("--topic", action="append", dest="topics")
    parser.add_argument("--pages", type=int, default=2)
    parser.add_argument("--page-size", type=int, default=4)
    parser.add_argument("--max-members", type=int, default=25)
    args = parser.parse_args()
    if args.operation == "verify":
        manifest = verify_run(args.run_root, args.store_root.resolve())
        discovery = read_json(args.run_root / "discovery.json")
        print(
            json.dumps(
                {
                    "verified": manifest["snapshot_id"],
                    "members": len(manifest["members"]),
                    "discovery_status": "partial" if discovery["failures"] else "complete",
                    "failure_receipt_provenance": (
                        "none"
                        if not discovery["failures"]
                        else (
                            "legacy-unbound"
                            if discovery["schema_revision"] == "discovery-1"
                            else "request-bound"
                        )
                    ),
                }
            )
        )
        return
    plan = DiscoveryPlan(
        topics=tuple(
            args.topics
            or ("plyometric training", "sprint biomechanics", "resistance training athletes")
        ),
        pages_per_topic=args.pages,
        page_size=args.page_size,
    )
    engine = create_database_engine(load_settings())
    fulltext = EuropePmcClient(timeout_seconds=40)
    try:
        with httpx2.Client(
            trust_env=False,
            follow_redirects=True,
            headers={
                "User-Agent": "DynamisRAG/0.1 (+https://github.com/Litju/DynamisRAG)",
            },
        ) as client:
            result = run_snapshot(
                engine=engine,
                http=BoundedHttp(client),
                fulltext=fulltext,
                run_root=args.run_root,
                store_root=args.store_root,
                plan=plan,
                max_members=args.max_members,
            )
        # Volatile execution facts are deliberately outside the snapshot identity.
        publish_json(
            args.run_root / "executions" / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}.json",
            {"completed_at": datetime.now(UTC).isoformat(), "result": result},
        )
        print(
            json.dumps(
                {
                    **result,
                    "discovery_status": "partial" if result["provider_failures"] else "complete",
                },
                ensure_ascii=True,
                indent=2,
            )
        )
    finally:
        fulltext.close()
        engine.dispose()


if __name__ == "__main__":
    main()
