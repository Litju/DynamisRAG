"""RES-140: sealed IR inputs, identities and TREC rank preservation."""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, cast

import pytest

from dynamisrag.ir import (
    IR_CONTRACT_REVISION,
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

_CORPUS_SHA = "a" * 64
_PROJECTION_SHA = "b" * 64
_CODE_SHA = "c" * 40


def _dataset() -> IrDataset:
    return IrDataset(
        source_id="scifact/test",
        source_revision="2026-01",
        corpus_sha256=_CORPUS_SHA,
        queries=(IrQuery("q1", "velocity in CMJ?"), IrQuery("q2", "no hits")),
        qrels=(
            IrQrel("q1", "d1", 2),
            IrQrel("q1", "d2", -1),
            IrQrel("q2", "d3", 0),
        ),
    )


def _config(dataset: IrDataset) -> IrExperimentConfig:
    return IrExperimentConfig(
        dataset_sha256=dataset.sha256,
        code_sha=_CODE_SHA,
        retrieval_revision="hybrid-rrf-v1",
        projection_sha256=_PROJECTION_SHA,
        parameters_json='{"candidate_window":50,"rrf_k":60}',
    )


def _run(dataset: IrDataset, config: IrExperimentConfig) -> IrRun:
    return IrRun(
        config_sha256=config.sha256,
        dataset_sha256=dataset.sha256,
        query_ids=("q1", "q2"),
        hits=(IrHit("q1", "d1", 1, 0.1), IrHit("q1", "d2", 2, 800.0)),
        evaluation_depth=50,
    )


def test_canonical_ir_json_is_byte_stable_and_lf_terminated() -> None:
    assert canonical_ir_json({"z": 1, "a": "ñ"}) == b'{"a":"\xc3\xb1","z":1}\n'
    with pytest.raises(ValueError):
        canonical_ir_json({"nan": float("nan")})


def test_complete_contract_binds_dataset_config_and_run() -> None:
    dataset = _dataset()
    config = _config(dataset)
    run = _run(dataset, config)
    run.validate_against(dataset, config)
    assert set(dataset.payload()) == {
        "revision",
        "source_id",
        "source_revision",
        "corpus_sha256",
        "queries",
        "qrels",
    }
    assert dataset.payload()["revision"] == IR_CONTRACT_REVISION
    for fingerprint in (dataset.sha256, config.sha256, run.sha256):
        assert len(fingerprint) == 64
        assert int(fingerprint, 16) >= 0
    assert run.sha256 == hashlib.sha256(canonical_ir_json(run.payload())).hexdigest()
    assert run.query_ids == ("q1", "q2")  # q2 has zero hits but remains in denominator.


def test_trec_qrels_keep_original_signed_judgments() -> None:
    assert trec_qrels(_dataset()) == "q1 0 d1 2\nq1 0 d2 -1\nq2 0 d3 0\n"


def test_trec_run_ranks_are_not_reordered_by_native_scores() -> None:
    dataset = _dataset()
    run = _run(dataset, _config(dataset))
    assert trec_run(run, run_tag="baseline") == ("q1 Q0 d1 1 -1 baseline\nq1 Q0 d2 2 -2 baseline\n")
    with pytest.raises(IrContractError, match="run tag"):
        trec_run(run, run_tag="has space")


@pytest.mark.parametrize("bad", ["", "bad id", "bad\nid", "bad\tid", "bad\x00id"])
def test_trec_identifiers_reject_whitespace_or_control_characters(bad: str) -> None:
    with pytest.raises(IrContractError):
        IrQuery(bad, "text")
    with pytest.raises(IrContractError):
        IrQrel("q", bad, 1)
    with pytest.raises(IrContractError):
        IrHit("q", bad, 1, 1.0)


def test_dataset_requires_canonical_order_unique_qrels_and_known_queries() -> None:
    with pytest.raises(IrContractError, match="queries"):
        IrDataset("s", "r", _CORPUS_SHA, (IrQuery("b", "B"), IrQuery("a", "A")), ())
    with pytest.raises(IrContractError, match="qrels"):
        IrDataset(
            "s",
            "r",
            _CORPUS_SHA,
            (IrQuery("q", "Q"),),
            (IrQrel("q", "d", 1), IrQrel("q", "d", 0)),
        )
    with pytest.raises(IrContractError, match="declared query"):
        IrDataset("s", "r", _CORPUS_SHA, (IrQuery("q", "Q"),), (IrQrel("other", "d", 1),))
    with pytest.raises(IrContractError, match="corpus_sha256"):
        IrDataset("s", "r", "moving-tag", (IrQuery("q", "Q"),), ())


def test_run_refuses_wrong_rank_duplicates_and_unknown_query() -> None:
    ds = _dataset()
    cfg = _config(ds)
    base = _run(ds, cfg)
    for hits in (
        (IrHit("q1", "d1", 2, 0.2),),
        (IrHit("q1", "d1", 1, 0.2), IrHit("q1", "d1", 2, 0.1)),
        (IrHit("qX", "d1", 1, 0.2),),
        (IrHit("q2", "d1", 1, 0.2), IrHit("q1", "d1", 1, 0.2)),
    ):
        with pytest.raises(IrContractError):
            dataclasses.replace(base, hits=hits)


def test_config_identity_is_sensitive_to_every_semantic_setting() -> None:
    ds = _dataset()
    cfg = _config(ds)
    assert (
        cfg.sha256
        != dataclasses.replace(cfg, parameters_json='{"candidate_window":50,"rrf_k":61}').sha256
    )
    assert cfg.sha256 != dataclasses.replace(cfg, code_sha="d" * 40).sha256
    assert cfg.sha256 != dataclasses.replace(cfg, projection_sha256="e" * 64).sha256
    assert cfg.sha256 != dataclasses.replace(cfg, retrieval_revision="dense-knn-v1").sha256
    for invalid in ('{"z":1, "a":2}', '{"z":NaN}', "[]", '{"x":1,"x":2}'):
        with pytest.raises(IrContractError):
            dataclasses.replace(cfg, parameters_json=invalid)


def test_config_identity_rejects_environment_and_runtime_fields() -> None:
    ds = _dataset()
    cfg = _config(ds)
    for invalid in (
        '{"cache_path":"C:/tmp/index"}',
        '{"api_token":"private"}',
        '{"started_at":"2026-10-08T00:00:00Z"}',
        '{"node":"https://private.example"}',
        '{"backend_error":"index unavailable"}',
    ):
        with pytest.raises(IrContractError):
            dataclasses.replace(cfg, parameters_json=invalid)


def test_run_cannot_be_attributed_to_different_dataset_or_config() -> None:
    ds = _dataset()
    cfg = _config(ds)
    run = _run(ds, cfg)
    with pytest.raises(IrContractError):
        run.validate_against(dataclasses.replace(ds, source_revision="another"), cfg)
    with pytest.raises(IrContractError):
        run.validate_against(ds, dataclasses.replace(cfg, parameters_json='{"x":1}'))
    with pytest.raises(IrContractError):
        dataclasses.replace(run, query_ids=("q1",)).validate_against(ds, cfg)


def test_signed_zero_has_a_single_identity_and_non_finite_scores_are_refused() -> None:
    assert IrHit("q1", "d1", 1, -0.0).payload() == IrHit("q1", "d1", 1, 0.0).payload()
    for invalid in (float("nan"), float("inf"), float("-inf"), True):
        with pytest.raises(IrContractError):
            IrHit("q1", "d1", 1, cast(Any, invalid))


def test_qrels_preserve_negative_labels_but_refuse_boolean_relevance() -> None:
    assert IrQrel("q1", "d1", -1).relevance == -1
    invalid_relevance: Any = True
    with pytest.raises(IrContractError):
        IrQrel("q1", "d1", invalid_relevance)


def test_metric_policy_is_explicit_and_versioned() -> None:
    policy = IrMetricPolicy().payload()
    assert policy["revision"] == "ir-metric-policy-v1"
    assert policy["measures"] == ["nDCG@10", "Recall@10", "MAP", "MRR"]
    assert policy["ndcg_gain"] == "linear_relevance"
    assert policy["aggregation"] == "macro_mean_over_all_declared_queries"
    assert policy["negative_relevance_for_scoring"] == 0


def test_passage_hits_map_to_documents_with_stable_tie_break_and_dedup() -> None:
    dataset = _dataset()
    config = _config(dataset)
    mapping = IrPassageMapping(
        (
            IrPassageMapEntry("p1", "d1", "v1"),
            IrPassageMapEntry("p2", "d1", "v1"),
            IrPassageMapEntry("p3", "d2", "v1"),
            IrPassageMapEntry("p4", "d3", "v1"),
        )
    )
    hits = (
        IrPassageHit("q1", "p3", 3, 100.0),
        IrPassageHit("q1", "p2", 1, 900.0),
        IrPassageHit("q1", "p1", 1, 0.1),
        IrPassageHit("q1", "p4", 4, 0.05),
    )

    run = document_run_from_passages(
        dataset=dataset,
        config=config,
        passage_mapping=mapping,
        hits=hits,
        evaluation_depth=50,
        source_exhausted_query_ids=("q1",),
    )
    reordered = document_run_from_passages(
        dataset=dataset,
        config=config,
        passage_mapping=mapping,
        hits=tuple(reversed(hits)),
        evaluation_depth=50,
        source_exhausted_query_ids=("q1",),
    )

    assert run == reordered
    assert [
        (hit.document_id, hit.rank, hit.raw_score, hit.source_passage_id) for hit in run.hits
    ] == [
        ("d1", 1, 0.1, "p1"),
        ("d2", 2, 100.0, "p3"),
        ("d3", 3, 0.05, "p4"),
    ]
    mapping.validate_run(run)
    assert run.source_exhausted_query_ids == ("q1",)


def test_passage_conversion_requires_a_complete_source_rank_prefix() -> None:
    dataset = _dataset()
    config = _config(dataset)
    mapping = IrPassageMapping(
        (
            IrPassageMapEntry("p1", "d1", "v1"),
            IrPassageMapEntry("p3", "d2", "v1"),
            IrPassageMapEntry("p50", "d3", "v1"),
        )
    )

    for hits in (
        (IrPassageHit("q1", "p50", 50, 1.0),),
        (IrPassageHit("q1", "p1", 1, 1.0), IrPassageHit("q1", "p3", 3, 0.5)),
    ):
        with pytest.raises(IrContractError, match="complete one-based prefix"):
            document_run_from_passages(
                dataset=dataset,
                config=config,
                passage_mapping=mapping,
                hits=hits,
                evaluation_depth=50,
                source_exhausted_query_ids=("q1",),
            )


def test_passage_conversion_requires_exhaustion_for_censored_top_ten() -> None:
    dataset = _dataset()
    config = _config(dataset)
    mapping = IrPassageMapping(
        tuple(IrPassageMapEntry(f"p{rank:02}", f"d{rank % 9}", "v1") for rank in range(1, 51))
    )
    hits = tuple(IrPassageHit("q1", f"p{rank:02}", rank, float(51 - rank)) for rank in range(1, 51))

    with pytest.raises(IrContractError, match="complete document top-10"):
        document_run_from_passages(
            dataset=dataset,
            config=config,
            passage_mapping=mapping,
            hits=hits,
            evaluation_depth=50,
        )

    run = document_run_from_passages(
        dataset=dataset,
        config=config,
        passage_mapping=mapping,
        hits=hits,
        evaluation_depth=50,
        source_exhausted_query_ids=("q1",),
    )
    assert len(run.hits) == 9
    assert run.source_exhausted_query_ids == ("q1",)
    assert run.query_ids == ("q1", "q2")


def test_passage_mapping_rejects_missing_duplicate_and_conflicting_sources() -> None:
    with pytest.raises(IrContractError, match="conflicting document versions"):
        IrPassageMapping(
            (
                IrPassageMapEntry("p1", "d1", "v1"),
                IrPassageMapEntry("p2", "d1", "v2"),
            )
        )

    dataset = _dataset()
    config = _config(dataset)
    mapping = IrPassageMapping((IrPassageMapEntry("p1", "d1", "v1"),))
    duplicate = (IrPassageHit("q1", "p1", 1, 1.0),) * 2
    with pytest.raises(IrContractError, match="duplicate passage hit"):
        document_run_from_passages(
            dataset=dataset,
            config=config,
            passage_mapping=mapping,
            hits=duplicate,
            evaluation_depth=50,
        )
    with pytest.raises(IrContractError, match="no document mapping"):
        document_run_from_passages(
            dataset=dataset,
            config=config,
            passage_mapping=mapping,
            hits=(IrPassageHit("q1", "missing", 1, 1.0),),
            evaluation_depth=50,
        )


def test_run_depth_cannot_overclaim_recall_or_contain_deeper_hits() -> None:
    dataset = _dataset()
    config = _config(dataset)
    with pytest.raises(IrContractError, match="cannot support Recall@10"):
        IrRun(config.sha256, dataset.sha256, ("q1", "q2"), (), evaluation_depth=9)
    with pytest.raises(IrContractError, match="exceeds the declared evaluation depth"):
        IrRun(
            config.sha256,
            dataset.sha256,
            ("q1", "q2"),
            (IrHit("q1", "d1", 11, 1.0),),
            evaluation_depth=10,
        )
