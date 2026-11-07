"""The bundled Research Edition corpus must keep its pinned source bytes."""

import shutil
from pathlib import Path

import pytest
from scripts.real_scientific_retrieval import SAMPLE_CORPUS, load_sample_payloads


def test_sample_corpus_matches_its_pinned_license_and_source_receipts() -> None:
    payloads = load_sample_payloads()

    assert [payload.pmcid for _, payload in payloads] == [
        "PMC10780924",
        "PMC10680285",
        "PMC5263151",
    ]
    assert all(
        payload.license_uri == "https://creativecommons.org/licenses/by/4.0/"
        for _, payload in payloads
    )


def test_sample_corpus_rejects_changed_source_bytes(tmp_path: Path) -> None:
    sample = tmp_path / "sample"
    shutil.copytree(SAMPLE_CORPUS, sample)
    source = sample / "sources" / "PMC5263151.xml"
    source.write_bytes(source.read_bytes() + b" ")

    with pytest.raises(ValueError, match="sample article bytes differ"):
        load_sample_payloads(sample)
