"""Unit tests for docintel.text -- span offset verification and title folding.

This logic decides whether a model's quoted evidence is grounded in the source
document. A false negative here rejects a correct extraction as a hallucination;
a false positive lets a fabricated quote through. Both failure modes are silent
in aggregate metrics, so the boundaries are pinned down explicitly.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from audit_data import build_text_index
from docintel.text import (
    SpanCheck,
    SpanStatus,
    contract_key,
    normalize_whitespace,
    verify_span,
)

SOURCE = "This Agreement is governed by the laws of the State of Delaware."


class TestNormalizeWhitespace:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("a  b", "a b"),
            ("a\tb", "a b"),
            ("a\nb", "a b"),
            ("a\r\nb", "a b"),
            ("a \n\t  b", "a b"),
            ("  leading and trailing  ", "leading and trailing"),
            ("", ""),
            ("   ", ""),
            ("no-change", "no-change"),
        ],
    )
    def test_collapses_whitespace_runs(self, raw: str, expected: str) -> None:
        assert normalize_whitespace(raw) == expected

    def test_is_idempotent(self) -> None:
        once = normalize_whitespace("The  Company\n\tshall  pay")
        assert normalize_whitespace(once) == once

    def test_shortens_the_string_which_is_why_offsets_must_not_be_normalized(self) -> None:
        """The reason the pipeline stores raw text as the coordinate system.

        Normalizing a source document shifts every character after the first
        collapsed run, silently invalidating every stored offset.
        """
        raw = "Section 1.\n\n   Governing Law.  Delaware."
        normalized = normalize_whitespace(raw)
        assert len(normalized) < len(raw)

        start = raw.index("Delaware")
        assert raw[start : start + 8] == "Delaware"
        # Same offset against the normalized text points somewhere else entirely.
        assert normalized[start : start + 8] != "Delaware"


class TestVerifySpanExactMatches:
    def test_exact_slice_is_ok(self) -> None:
        start = SOURCE.index("State of Delaware")
        check = verify_span(SOURCE, start, "State of Delaware")
        assert check.status is SpanStatus.EXACT
        assert check.ok
        assert check.actual_offset is None

    def test_span_at_position_zero(self) -> None:
        check = verify_span(SOURCE, 0, "This Agreement")
        assert check.status is SpanStatus.EXACT

    def test_span_ending_exactly_at_end_of_source(self) -> None:
        check = verify_span(SOURCE, len(SOURCE) - 9, "Delaware.")
        assert check.status is SpanStatus.EXACT

    def test_whole_document_as_a_span(self) -> None:
        assert verify_span(SOURCE, 0, SOURCE).status is SpanStatus.EXACT

    def test_empty_gold_text_is_vacuously_exact(self) -> None:
        # Degenerate but reachable: a model may emit an empty quote. It must not
        # crash, and it must not be reported as a mismatch.
        check = verify_span(SOURCE, 5, "")
        assert check.status is SpanStatus.EXACT


class TestVerifySpanFailures:
    def test_off_by_one_offset_is_a_mismatch(self) -> None:
        """The drift mode the audit exists to catch."""
        start = SOURCE.index("State of Delaware")
        check = verify_span(SOURCE, start + 1, "State of Delaware")
        assert check.status is SpanStatus.MISMATCH
        assert not check.ok

    def test_mismatch_reports_where_the_text_actually_is(self) -> None:
        """A consistent shift across failures means a repairable systematic offset."""
        true_start = SOURCE.index("State of")
        check = verify_span(SOURCE, true_start + 7, "State of")
        assert check.status is SpanStatus.MISMATCH
        assert check.actual_offset == true_start

    def test_text_absent_from_source_has_no_actual_offset(self) -> None:
        """Distinguishes 'offsets drifted' from 'the model made it up'."""
        check = verify_span(SOURCE, 0, "State of New York")
        assert check.status is SpanStatus.MISMATCH
        assert check.actual_offset is None

    def test_span_running_past_end_is_out_of_bounds(self) -> None:
        check = verify_span(SOURCE, len(SOURCE) - 3, "Delaware and more text")
        assert check.status is SpanStatus.OUT_OF_BOUNDS
        assert not check.ok

    def test_start_beyond_end_of_source_is_out_of_bounds(self) -> None:
        check = verify_span(SOURCE, len(SOURCE) + 50, "anything")
        assert check.status is SpanStatus.OUT_OF_BOUNDS

    def test_negative_start_is_out_of_bounds_not_python_slicing(self) -> None:
        """Guards a real bug: Python would silently index from the end."""
        check = verify_span(SOURCE, -5, "Delaware.")
        assert check.status is SpanStatus.OUT_OF_BOUNDS


class TestVerifySpanWhitespaceHandling:
    def test_whitespace_only_difference_is_its_own_status(self) -> None:
        source = "the laws of the State  of\nDelaware"
        gold = "the laws of the State of Delaware"
        check = verify_span(source, 0, gold)
        assert check.status is SpanStatus.WHITESPACE_ONLY

    def test_whitespace_only_is_not_ok(self) -> None:
        """Deliberate: it is recoverable, but it is not usable as a raw offset.

        Grading it as a pass would hide exactly the drift the audit measures.
        """
        source = "the laws of  the State of Delaware"
        check = verify_span(source, 0, "the laws of the State of Delaware")
        assert check.status is SpanStatus.WHITESPACE_ONLY
        assert not check.ok

    def test_differing_content_is_a_mismatch_even_with_equal_whitespace(self) -> None:
        check = verify_span(
            "the laws of the State of Nevada!", 0, "the laws of the State of Nevada?"
        )
        assert check.status is SpanStatus.MISMATCH

    def test_leading_whitespace_shift_is_not_forgiven_by_stripping(self) -> None:
        """`normalize_whitespace` strips ends, so a shifted-but-padded slice
        could compare equal. It must still be flagged, not silently accepted."""
        source = "   Delaware"
        check = verify_span(source, 0, "Delaware")
        assert check.status is SpanStatus.WHITESPACE_ONLY
        assert not check.ok


class TestSpanCheckContract:
    def test_only_exact_is_ok(self) -> None:
        for status in SpanStatus:
            check = SpanCheck(status=status, char_start=0, gold_text="x", found_text="x")
            assert check.ok is (status is SpanStatus.EXACT)

    def test_is_frozen(self) -> None:
        check = verify_span(SOURCE, 0, "This")
        with pytest.raises((AttributeError, TypeError)):
            check.char_start = 99  # type: ignore[misc]


class TestContractKey:
    def test_nfc_and_nfd_forms_of_the_same_title_collide(self) -> None:
        """The LECLANCHE case: CUAD's JSON title and the filename on disk use
        different Unicode normal forms, so plain equality drops the contract."""
        # These two literals are visually identical but are different byte
        # sequences (U+00C9 vs. E + U+0301). The inequality assert below is
        # load-bearing: if an editor ever normalizes this file, it fails loudly
        # instead of leaving a test that passes for the wrong reason.
        precomposed = "LECLANCHÉ S.A. - AGREEMENT"
        decomposed = "LECLANCHÉ S.A. - AGREEMENT"
        assert precomposed != decomposed
        assert contract_key(precomposed) == contract_key(decomposed)

    def test_is_idempotent(self) -> None:
        title = unicodedata.normalize("NFD", "LECLANCHÉ S.A.")
        assert contract_key(contract_key(title)) == contract_key(title)

    def test_leaves_ascii_titles_untouched(self) -> None:
        title = "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT"
        assert contract_key(title) == title


class TestBuildTextIndex:
    def test_keys_on_normalized_stem(self, tmp_path: Path) -> None:
        (tmp_path / "LECLANCHÉ S.A..txt").write_text("body", encoding="utf-8")
        index = build_text_index(tmp_path)
        assert contract_key("LECLANCHÉ S.A.") in index

    def test_ignores_non_txt_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("x", encoding="utf-8")
        (tmp_path / "b.pdf").write_bytes(b"x")
        assert set(build_text_index(tmp_path)) == {"a"}


class TestRoundTripAgainstSyntheticCorpus:
    """End-to-end shape of the audit's inner loop, without touching real data."""

    def test_offsets_survive_a_read_write_round_trip(self, tmp_path: Path) -> None:
        text = "Section 1.\r\n\r\nGoverning Law. This Agreement is governed by Delaware law.\n"
        path = tmp_path / "contract.txt"
        path.write_text(text, encoding="utf-8", newline="")

        reloaded = path.read_text(encoding="utf-8")
        start = reloaded.index("Delaware law")
        assert verify_span(reloaded, start, "Delaware law").status is SpanStatus.EXACT

    def test_crlf_translation_on_read_breaks_offsets(self, tmp_path: Path) -> None:
        """Why the loader pins encoding and must not rely on universal newlines.

        Offsets computed against CRLF bytes do not hold once Python collapses
        them to LF -- one character lost per line before the span.
        """
        raw = "line one\r\nline two\r\nDelaware"
        path = tmp_path / "crlf.txt"
        path.write_bytes(raw.encode("utf-8"))

        start_in_crlf = raw.index("Delaware")
        translated = path.read_text(encoding="utf-8")  # newline=None collapses CRLF
        assert verify_span(translated, start_in_crlf, "Delaware").status is SpanStatus.OUT_OF_BOUNDS
