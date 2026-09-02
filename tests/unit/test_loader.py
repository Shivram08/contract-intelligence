"""Unit tests for the CUAD loader.

The loader has exactly one hard requirement: **do not corrupt offsets.** CUAD's
``answer_start`` values are byte-exact against the raw file (``docs/DATA_AUDIT.md``
check 4), so anything the loader does to the text on the way in -- newline
translation, stripping, re-normalizing Unicode -- silently invalidates every
annotation in that document. Nothing raises; the grounding verifier just starts
rejecting correct extractions.

These tests use synthetic files rather than the CUAD download, so they run in CI.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pytest

from docintel.ingest.loader import (
    iter_documents,
    load_document,
    load_gold_spans,
    load_split,
    read_contract_text,
)


def write_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


class TestReadContractText:
    def test_crlf_is_preserved_not_collapsed(self, tmp_path: Path) -> None:
        """The whole reason ``newline=""`` is passed.

        Python's default universal-newline mode turns CRLF into LF, losing one
        character per line. Every offset after the first line break then points
        one place too far left, per line.
        """
        path = write_bytes(tmp_path / "c.txt", b"line one\r\nline two\r\nDelaware")
        text = read_contract_text(path)
        assert "\r\n" in text
        assert text.count("\r") == 2
        assert text.index("Delaware") == len("line one\r\nline two\r\n")

    def test_offsets_match_the_bytes_on_disk(self, tmp_path: Path) -> None:
        raw = "1. TERM.\r\n\r\n2. GOVERNING LAW. Delaware."
        path = write_bytes(tmp_path / "c.txt", raw.encode("utf-8"))
        assert read_contract_text(path) == raw

    def test_lone_cr_is_preserved(self, tmp_path: Path) -> None:
        """Old Mac line endings appear in a few EDGAR filings."""
        path = write_bytes(tmp_path / "c.txt", b"alpha\rbravo")
        assert read_contract_text(path) == "alpha\rbravo"

    def test_leading_and_trailing_whitespace_is_kept(self, tmp_path: Path) -> None:
        """Stripping here would shift every offset in the document by the
        length of the leading whitespace."""
        path = write_bytes(tmp_path / "c.txt", b"\n\n  body  \n\n")
        assert read_contract_text(path) == "\n\n  body  \n\n"

    def test_utf8_multibyte_characters_survive(self, tmp_path: Path) -> None:
        raw = "LECLANCHÉ S.A. and a café clause"
        path = write_bytes(tmp_path / "c.txt", raw.encode("utf-8"))
        assert read_contract_text(path) == raw

    def test_unicode_is_not_renormalized(self, tmp_path: Path) -> None:
        """NFC folding is for *ids*, never for text.

        Composing "E + combining acute" into a single character would shorten
        the string and shift every subsequent offset.
        """
        decomposed = "LECLANCHÉ S.A."
        path = write_bytes(tmp_path / "c.txt", decomposed.encode("utf-8"))
        text = read_contract_text(path)
        assert text == decomposed
        assert len(text) == len(decomposed)
        assert text != unicodedata.normalize("NFC", decomposed)

    def test_empty_file(self, tmp_path: Path) -> None:
        assert read_contract_text(write_bytes(tmp_path / "c.txt", b"")) == ""


class TestLoadDocument:
    def test_document_id_is_the_nfc_folded_stem(self, tmp_path: Path) -> None:
        path = write_bytes(tmp_path / "LECLANCHÉ S.A..txt", b"body")
        document = load_document(path)
        assert document.document_id == unicodedata.normalize("NFC", path.stem)

    def test_ascii_stem_is_unchanged(self, tmp_path: Path) -> None:
        path = write_bytes(tmp_path / "PLAIN_AGREEMENT.txt", b"body")
        assert load_document(path).document_id == "PLAIN_AGREEMENT"

    def test_records_the_source_path(self, tmp_path: Path) -> None:
        path = write_bytes(tmp_path / "c.txt", b"body")
        assert load_document(path).source_path == str(path)

    def test_metadata_defaults_to_empty(self, tmp_path: Path) -> None:
        assert load_document(write_bytes(tmp_path / "c.txt", b"x")).metadata == {}

    def test_metadata_is_carried_through(self, tmp_path: Path) -> None:
        path = write_bytes(tmp_path / "c.txt", b"x")
        assert load_document(path, {"split": "dev"}).metadata == {"split": "dev"}


class TestIterDocuments:
    @pytest.fixture
    def corpus(self, tmp_path: Path) -> Path:
        for name in ("A_AGREEMENT", "B_AGREEMENT", "C_AGREEMENT"):
            write_bytes(tmp_path / "full_contract_txt" / f"{name}.txt", f"1. {name}".encode())
        # Must be ignored: only .txt files are contracts.
        write_bytes(tmp_path / "full_contract_txt" / "notes.md", b"ignore me")
        return tmp_path

    def test_yields_every_contract_when_unfiltered(self, corpus: Path) -> None:
        assert [d.document_id for d in iter_documents(corpus)] == [
            "A_AGREEMENT",
            "B_AGREEMENT",
            "C_AGREEMENT",
        ]

    def test_ignores_non_txt_files(self, corpus: Path) -> None:
        assert all(not d.document_id.endswith(".md") for d in iter_documents(corpus))

    def test_order_is_deterministic(self, corpus: Path) -> None:
        assert [d.document_id for d in iter_documents(corpus)] == [
            d.document_id for d in iter_documents(corpus)
        ]

    def test_filters_to_the_requested_ids(self, corpus: Path) -> None:
        got = [d.document_id for d in iter_documents(corpus, {"B_AGREEMENT"})]
        assert got == ["B_AGREEMENT"]

    def test_empty_filter_set_yields_nothing(self, corpus: Path) -> None:
        """An empty set means "no documents", not "no filter"."""
        assert list(iter_documents(corpus, set())) == []

    def test_unknown_id_in_the_filter_is_ignored(self, corpus: Path) -> None:
        got = [d.document_id for d in iter_documents(corpus, {"A_AGREEMENT", "NOPE"})]
        assert got == ["A_AGREEMENT"]

    def test_filter_matches_across_unicode_normal_forms(self, tmp_path: Path) -> None:
        """A caller passing ids from split.json must not have to know which
        normal form the filesystem chose."""
        stem = "LECLANCHÉ S.A."  # decomposed on disk
        write_bytes(tmp_path / "full_contract_txt" / f"{stem}.txt", b"body")
        precomposed = unicodedata.normalize("NFC", stem)
        assert [d.document_id for d in iter_documents(tmp_path, {precomposed})] == [precomposed]


class TestLoadSplit:
    @pytest.fixture
    def split_file(self, tmp_path: Path) -> Path:
        path = tmp_path / "split.json"
        path.write_text(
            json.dumps(
                {
                    "seed": 42,
                    "splits": {
                        "dev": {"n": 2, "contract_ids": ["A", "B"]},
                        "golden": {"n": 1, "contract_ids": ["C"]},
                    },
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_reads_the_named_split(self, split_file: Path) -> None:
        assert load_split(split_file, "dev") == {"A", "B"}

    def test_splits_are_separate(self, split_file: Path) -> None:
        assert load_split(split_file, "golden") == {"C"}

    def test_unknown_split_raises_with_the_available_names(self, split_file: Path) -> None:
        with pytest.raises(KeyError, match="unknown split"):
            load_split(split_file, "reserve")

    def test_ids_are_nfc_folded(self, tmp_path: Path) -> None:
        path = tmp_path / "split.json"
        path.write_text(
            json.dumps({"splits": {"dev": {"contract_ids": ["LECLANCHÉ S.A."]}}}),
            encoding="utf-8",
        )
        assert load_split(path, "dev") == {unicodedata.normalize("NFC", "LECLANCHÉ S.A.")}


class TestLoadGoldSpans:
    @pytest.fixture
    def cuad_dir(self, tmp_path: Path) -> Path:
        payload = {
            "version": "test",
            "data": [
                {
                    "title": "DOC_A",
                    "paragraphs": [
                        {
                            "context": "1. GOVERNING LAW. Delaware.",
                            "qas": [
                                {
                                    "id": "DOC_A__Governing Law",
                                    "answers": [{"text": "Delaware", "answer_start": 18}],
                                    "is_impossible": False,
                                },
                                {
                                    "id": "DOC_A__Non-Compete",
                                    "answers": [],
                                    "is_impossible": True,
                                },
                            ],
                        }
                    ],
                },
                {
                    "title": "DOC_B",
                    "paragraphs": [
                        {
                            "context": "1. PARTIES. Acme and Beta.",
                            "qas": [
                                {
                                    "id": "DOC_B__Parties",
                                    "answers": [
                                        {"text": "Acme", "answer_start": 12},
                                        {"text": "Beta", "answer_start": 21},
                                    ],
                                    "is_impossible": False,
                                }
                            ],
                        }
                    ],
                },
            ],
        }
        (tmp_path / "CUAD_v1.json").write_text(json.dumps(payload), encoding="utf-8")
        return tmp_path

    def test_loads_spans_with_category_and_offsets(self, cuad_dir: Path) -> None:
        spans = load_gold_spans(cuad_dir)
        governing = [s for s in spans if s.category == "Governing Law"]
        assert len(governing) == 1
        assert governing[0].text == "Delaware"
        assert governing[0].char_start == 18

    def test_char_end_is_derived_from_the_text_length(self, cuad_dir: Path) -> None:
        span = next(s for s in load_gold_spans(cuad_dir) if s.category == "Governing Law")
        assert span.char_end == 18 + len("Delaware")

    def test_negative_categories_contribute_no_spans(self, cuad_dir: Path) -> None:
        """A category with no answers is a labelled negative, not a missing row."""
        assert not [s for s in load_gold_spans(cuad_dir) if s.category == "Non-Compete"]

    def test_multi_span_annotations_are_all_kept(self, cuad_dir: Path) -> None:
        """61% of positive pairs are single-span; the rest must not be truncated."""
        parties = [s for s in load_gold_spans(cuad_dir) if s.category == "Parties"]
        assert {s.text for s in parties} == {"Acme", "Beta"}

    def test_filters_by_document_id(self, cuad_dir: Path) -> None:
        spans = load_gold_spans(cuad_dir, {"DOC_B"})
        assert {s.document_id for s in spans} == {"DOC_B"}

    def test_empty_filter_set_yields_nothing(self, cuad_dir: Path) -> None:
        assert load_gold_spans(cuad_dir, set()) == []

    def test_category_is_parsed_from_the_id_suffix(self, cuad_dir: Path) -> None:
        """Ids are "<title>__<Category>" and titles contain underscores, so the
        split must be on the first "__" only."""
        assert {s.category for s in load_gold_spans(cuad_dir)} == {"Governing Law", "Parties"}
