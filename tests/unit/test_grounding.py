"""Unit tests for the verbatim grounding verifier.

This is the hallucination gate, so both directions of error matter and are
tested separately:

- A **false negative** rejects a correct extraction. That inflates the reported
  grounding violation rate with something that is not hallucination, and the
  damage is invisible because the number still looks like a measurement.
- A **false positive** lets a fabricated quote through, which is the whole
  failure mode the gate exists to stop.

The tests also pin what the gate deliberately does *not* catch, so nobody later
mistakes it for a correctness check.
"""

from __future__ import annotations

import pytest

from docintel.schemas import ClauseExtraction, ClauseType, Document, Evidence
from docintel.validation.grounding import (
    GroundingStatus,
    check_clause,
    check_evidence,
    check_extractions,
)

CONTRACT = (
    "MASTER SERVICES AGREEMENT\n\n"
    "1. TERM. This Agreement commences on the Effective Date.\n\n"
    "2. GOVERNING LAW. This Agreement is governed by the laws of the State of\n"
    "Delaware, without regard to its conflict of laws principles.\n\n"
    "3. LIABILITY. Each party's aggregate liability is capped at fees paid.\n"
)


def doc(text: str = CONTRACT) -> Document:
    return Document(document_id="DOC", text=text)


def evidence(quote: str, start: int | None = None, end: int | None = None) -> Evidence:
    """Build evidence, defaulting to the quote's real position."""
    if start is None:
        start = CONTRACT.index(quote)
    if end is None:
        end = start + len(quote)
    return Evidence(quote=quote, char_start=start, char_end=end)


class TestExactMatch:
    def test_correct_quote_and_offsets_is_exact(self) -> None:
        check = check_evidence(doc(), evidence("governed by the laws"))
        assert check.status is GroundingStatus.EXACT
        assert check.is_grounded

    def test_exact_match_needs_no_repair(self) -> None:
        item = evidence("conflict of laws principles")
        assert check_evidence(doc(), item).repaired == item

    def test_quote_at_position_zero(self) -> None:
        check = check_evidence(doc(), evidence("MASTER SERVICES AGREEMENT"))
        assert check.status is GroundingStatus.EXACT

    def test_quote_spanning_a_newline(self) -> None:
        """The source wraps mid-sentence; a quote copied with the newline is exact."""
        check = check_evidence(doc(), evidence("State of\nDelaware"))
        assert check.status is GroundingStatus.EXACT


class TestWhitespaceTolerance:
    def test_reflowed_quote_at_right_offsets_is_normalized(self) -> None:
        """A model that un-wraps a quote is quoting correctly, not fabricating."""
        start = CONTRACT.index("State of\nDelaware")
        item = Evidence(
            quote="State of Delaware",
            char_start=start,
            char_end=start + len("State of\nDelaware"),
        )
        check = check_evidence(doc(), item)
        assert check.status is GroundingStatus.NORMALIZED
        assert check.is_grounded

    def test_reflowed_quote_at_wrong_offsets_is_relocated(self) -> None:
        item = Evidence(quote="State of Delaware", char_start=0, char_end=17)
        check = check_evidence(doc(), item)
        assert check.status is GroundingStatus.RELOCATED
        assert check.is_grounded

    def test_relocation_finds_the_newline_spanning_original(self) -> None:
        item = Evidence(quote="State of Delaware", char_start=0, char_end=17)
        repaired = check_evidence(doc(), item).repaired
        assert repaired is not None
        assert repaired.quote == "State of\nDelaware"
        assert doc().slice(repaired.char_start, repaired.char_end) == repaired.quote

    def test_extra_internal_whitespace_is_tolerated(self) -> None:
        item = Evidence(quote="governed   by    the  laws", char_start=0, char_end=26)
        assert check_evidence(doc(), item).status is GroundingStatus.RELOCATED


class TestOffsetRepair:
    def test_correct_quote_wrong_offsets_is_relocated_not_rejected(self) -> None:
        """Models copy text well and count characters badly.

        Rejecting a real quote over arithmetic would report hallucination where
        there is none.
        """
        item = Evidence(quote="conflict of laws principles", char_start=3, char_end=30)
        check = check_evidence(doc(), item)
        assert check.status is GroundingStatus.RELOCATED
        assert check.is_grounded

    def test_repaired_span_actually_contains_the_quote(self) -> None:
        item = Evidence(quote="aggregate liability is capped", char_start=0, char_end=29)
        repaired = check_evidence(doc(), item).repaired
        assert repaired is not None
        assert (
            doc().slice(repaired.char_start, repaired.char_end) == "aggregate liability is capped"
        )

    def test_ambiguous_quote_reports_other_occurrences(self) -> None:
        """A repair that had several candidates may have picked the wrong one,
        and the caller deserves to know that."""
        source = doc("alpha CLAUSE beta CLAUSE gamma CLAUSE")
        item = Evidence(quote="CLAUSE", char_start=30, char_end=36)
        check = check_evidence(source, item)
        assert check.status is GroundingStatus.RELOCATED
        assert check.other_occurrences == 2

    def test_unambiguous_quote_reports_no_other_occurrences(self) -> None:
        item = Evidence(quote="conflict of laws", char_start=0, char_end=16)
        assert check_evidence(doc(), item).other_occurrences == 0


class TestFabrication:
    def test_invented_sentence_is_not_found(self) -> None:
        item = Evidence(
            quote="This Agreement is governed by the laws of the State of New York.",
            char_start=100,
            char_end=164,
        )
        check = check_evidence(doc(), item)
        assert check.status is GroundingStatus.NOT_FOUND
        assert not check.is_grounded
        assert check.repaired is None

    def test_plausible_but_absent_legalese_is_caught(self) -> None:
        """The realistic failure: fluent text that is not in this document."""
        item = Evidence(
            quote="Neither party shall be liable for consequential damages.",
            char_start=10,
            char_end=66,
        )
        assert check_evidence(doc(), item).status is GroundingStatus.NOT_FOUND

    def test_single_changed_word_is_caught(self) -> None:
        """'Delaware' -> 'Nevada' is the kind of error that changes the answer."""
        item = Evidence(quote="the laws of the State of Nevada", char_start=0, char_end=31)
        assert not check_evidence(doc(), item).is_grounded

    def test_out_of_bounds_offsets_with_absent_text(self) -> None:
        item = Evidence(quote="not in the document", char_start=99_999, char_end=100_018)
        assert check_evidence(doc(), item).status is GroundingStatus.OUT_OF_BOUNDS

    def test_empty_quote_is_ungrounded(self) -> None:
        item = Evidence(quote="   ", char_start=0, char_end=3)
        check = check_evidence(doc(), item)
        assert check.status is GroundingStatus.EMPTY
        assert not check.is_grounded


class TestWhatItCannotCatch:
    """Pins the documented limits, so the gate is not mistaken for more."""

    def test_verbatim_quote_supporting_a_negated_conclusion_still_passes(self) -> None:
        """The most important limitation.

        A quote saying liability is *not* capped grounds a `cap_on_liability`
        extraction perfectly well. Grounding checks provenance, not inference --
        that is what the rules and the reranker are for.
        """
        source = doc("5. LIABILITY. Liability shall not be capped in any circumstance.")
        item = Evidence(
            quote="Liability shall not be capped",
            char_start=source.text.index("Liability shall not be capped"),
            char_end=source.text.index("Liability shall not be capped")
            + len("Liability shall not be capped"),
        )
        assert check_evidence(source, item).is_grounded

    def test_quote_lifted_from_a_definition_still_passes(self) -> None:
        source = doc('1. DEFINITIONS. "Territory" means the exclusive area of operation.')
        quote = "the exclusive area of operation"
        item = Evidence(
            quote=quote,
            char_start=source.text.index(quote),
            char_end=source.text.index(quote) + len(quote),
        )
        assert check_evidence(source, item).is_grounded


class TestClauseLevel:
    def test_ungrounded_evidence_is_dropped_from_the_clause(self) -> None:
        """Dropping rather than keeping is deliberate: the clause then has
        present=True with no evidence, which trips a rule and fails validation.
        Keeping it would let a fabrication through with a shortened list."""
        clause = ClauseExtraction(
            clause_type=ClauseType.GOVERNING_LAW,
            present=True,
            value="US-DE",
            evidence=[
                evidence("governed by the laws"),
                Evidence(quote="invented text here", char_start=0, char_end=18),
            ],
            confidence=0.9,
        )
        _, repaired = check_clause(doc(), clause)
        assert len(repaired.evidence) == 1
        assert repaired.evidence[0].quote == "governed by the laws"

    def test_fully_fabricated_clause_ends_with_no_evidence(self) -> None:
        clause = ClauseExtraction(
            clause_type=ClauseType.NON_COMPETE,
            present=True,
            evidence=[Evidence(quote="a total fabrication", char_start=0, char_end=19)],
            confidence=0.8,
        )
        _, repaired = check_clause(doc(), clause)
        assert repaired.evidence == []
        assert repaired.present is True  # left inconsistent on purpose, for the rules

    def test_clean_clause_is_returned_unchanged(self) -> None:
        clause = ClauseExtraction(
            clause_type=ClauseType.GOVERNING_LAW,
            present=True,
            value="US-DE",
            evidence=[evidence("conflict of laws principles")],
            confidence=0.9,
        )
        _, repaired = check_clause(doc(), clause)
        assert repaired is clause

    def test_absent_clause_with_no_evidence_produces_no_checks(self) -> None:
        clause = ClauseExtraction(clause_type=ClauseType.NON_COMPETE, present=False, confidence=0.9)
        checks, _ = check_clause(doc(), clause)
        assert checks == []


class TestReport:
    def _clauses(self) -> list[ClauseExtraction]:
        return [
            ClauseExtraction(
                clause_type=ClauseType.GOVERNING_LAW,
                present=True,
                value="US-DE",
                evidence=[evidence("governed by the laws"), evidence("conflict of laws")],
                confidence=0.9,
            ),
            ClauseExtraction(
                clause_type=ClauseType.CAP_ON_LIABILITY,
                present=True,
                evidence=[Evidence(quote="fabricated clause text", char_start=0, char_end=22)],
                confidence=0.7,
            ),
        ]

    def test_counts_every_span(self) -> None:
        assert check_extractions(doc(), self._clauses()).total == 3

    def test_violation_rate_is_per_span_not_per_document(self) -> None:
        """One bad quote among three is 1/3, not a failed document."""
        report = check_extractions(doc(), self._clauses())
        assert report.violation_rate == pytest.approx(1 / 3)

    def test_ungrounded_lists_only_the_failures(self) -> None:
        report = check_extractions(doc(), self._clauses())
        assert len(report.ungrounded) == 1
        assert report.ungrounded[0].status is GroundingStatus.NOT_FOUND

    def test_status_counts(self) -> None:
        report = check_extractions(doc(), self._clauses())
        assert report.count(GroundingStatus.EXACT) == 2
        assert report.count(GroundingStatus.NOT_FOUND) == 1

    def test_empty_extraction_has_zero_rate_not_a_crash(self) -> None:
        report = check_extractions(doc(), [])
        assert report.total == 0
        assert report.violation_rate == 0.0

    def test_all_absent_clauses_have_zero_rate(self) -> None:
        clauses = [
            ClauseExtraction(clause_type=ct, present=False, confidence=0.9)
            for ct in list(ClauseType)[:3]
        ]
        assert check_extractions(doc(), clauses).violation_rate == 0.0
