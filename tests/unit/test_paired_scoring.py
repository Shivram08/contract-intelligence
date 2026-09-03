"""Tests for paired scoring and the offsets-unknown sentinel.

Two defects motivated these, and both produced numbers that looked fine.

**Paired scoring.** Batch 1 scored each arm on its own completed subset. Those
subsets were different -- 7/10 and 6/10, overlapping on 5 -- and both arms
reported presence F1 = 1.000 on about five cases each. Two confident-looking
numbers computed over different documents are not a comparison.

**The 0/0 sentinel.** `Evidence` rejected `char_start == char_end == 0`, which
killed four of ten single-call runs outright. The extraction prompt explicitly
tells the model the quote is what matters and that offsets can be recovered, so
the schema was contradicting the prompt and failing submissions for something
the grounding layer exists to repair.

The danger in fixing that is turning grounding into an escape hatch, so the
second half of this file pins the property the hallucination gate depends on: a
quote that is not in the source is still a violation, offsets or no offsets.
"""

from __future__ import annotations

import pytest
from evals.cases import GoldenCase
from evals.metrics import score_cases

from docintel.schemas import ClauseExtraction, ClauseType, Document, Evidence
from docintel.validation.grounding import GroundingStatus, check_extractions

SOURCE = (
    "1. GOVERNING LAW. This Agreement is governed by the laws of Delaware. "
    "2. ASSIGNMENT. Neither party may assign this Agreement without consent."
)


def document() -> Document:
    return Document(document_id="DOC_A", text=SOURCE)


def case(document_id: str, clause: ClauseType, present: bool) -> GoldenCase:
    return GoldenCase(
        case_id=f"{document_id}::{clause.value}",
        document_id=document_id,
        clause_type=clause,
        present=present,
        gold_spans=((18, 68),) if present else (),
    )


def extraction(clause: ClauseType, present: bool, quote: str = "") -> ClauseExtraction:
    return ClauseExtraction(
        clause_type=clause,
        present=present,
        evidence=[Evidence(quote=quote, char_start=0, char_end=0)] if present else [],
        confidence=0.9,
    )


class TestPairedScoringExcludesUnpairedDocuments:
    """A case counts only where every arm completed."""

    def test_a_document_missing_from_scoreable_is_excluded(self) -> None:
        cases = [
            case("DOC_A", ClauseType.GOVERNING_LAW, True),
            case("DOC_B", ClauseType.GOVERNING_LAW, True),
        ]
        predictions = {
            "DOC_A": [extraction(ClauseType.GOVERNING_LAW, True, "governed by the laws")],
            "DOC_B": [extraction(ClauseType.GOVERNING_LAW, True, "governed by the laws")],
        }
        both = score_cases("arm", cases, predictions, scoreable={"DOC_A", "DOC_B"})
        one = score_cases("arm", cases, predictions, scoreable={"DOC_A"})
        assert both.cases_scored == 2
        assert one.cases_scored == 1

    def test_an_incomplete_arm_removes_the_case_from_the_other_arm_too(self) -> None:
        """The property that matters. If arm B failed on DOC_B, arm A must not
        be scored on DOC_B either -- otherwise the arms are compared on
        different documents, which is what produced two F1 = 1.000 readings."""
        cases = [
            case("DOC_A", ClauseType.GOVERNING_LAW, True),
            case("DOC_B", ClauseType.GOVERNING_LAW, True),
        ]
        arm_a = {
            "DOC_A": [extraction(ClauseType.GOVERNING_LAW, True, "governed by the laws")],
            "DOC_B": [extraction(ClauseType.GOVERNING_LAW, True, "governed by the laws")],
        }
        arm_b = {"DOC_A": [extraction(ClauseType.GOVERNING_LAW, True, "governed by the laws")]}

        # DOC_B completed in A but not in B, so neither arm scores it.
        scoreable = {"DOC_A"}
        a = score_cases("a", cases, arm_a, scoreable=scoreable)
        b = score_cases("b", cases, arm_b, scoreable=scoreable)
        assert a.cases_scored == b.cases_scored == 1
        assert a.presence.overall.total == b.presence.overall.total

    def test_scoreable_none_scores_everything(self) -> None:
        """Backwards compatible: a single-arm run has nothing to pair against."""
        cases = [case("DOC_A", ClauseType.GOVERNING_LAW, True)]
        summary = score_cases("arm", cases, {"DOC_A": []}, scoreable=None)
        assert summary.cases_scored == 1

    def test_empty_scoreable_set_scores_nothing(self) -> None:
        cases = [case("DOC_A", ClauseType.GOVERNING_LAW, True)]
        summary = score_cases("arm", cases, {"DOC_A": []}, scoreable=set())
        assert summary.cases_scored == 0
        assert summary.presence.overall.total == 0


class TestOffsetsUnknownSentinel:
    def test_zero_zero_is_accepted(self) -> None:
        """0/0 means 'quoting accurately, offsets not computed', which the
        prompt invites."""
        evidence = Evidence(quote="governed by the laws", char_start=0, char_end=0)
        assert evidence.offsets_unknown

    def test_a_real_range_is_not_flagged_unknown(self) -> None:
        assert not Evidence(quote="x", char_start=5, char_end=9).offsets_unknown

    def test_an_inverted_range_is_still_rejected(self) -> None:
        """The sentinel is 0/0 exactly. Genuinely malformed offsets still fail."""
        with pytest.raises(ValueError, match="must exceed"):
            Evidence(quote="x", char_start=90, char_end=12)

    def test_a_zero_width_range_that_is_not_at_zero_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must exceed"):
            Evidence(quote="x", char_start=40, char_end=40)


class TestGroundingStillBitesWithUnknownOffsets:
    """The hallucination gate must not become an escape hatch.

    CLAUDE.md §11 claims fabricated output is mechanically rejected. That claim
    now rests on grounding catching a bad quote even when offsets carry no
    information at all.
    """

    def test_a_real_quote_with_unknown_offsets_is_relocated_and_grounded(self) -> None:
        clause = extraction(ClauseType.GOVERNING_LAW, True, "governed by the laws of Delaware")
        report = check_extractions(document(), [clause])
        assert report.checks[0].status is GroundingStatus.RELOCATED
        assert report.checks[0].is_grounded
        assert not report.ungrounded

    def test_relocation_recovers_the_true_offsets(self) -> None:
        quote = "governed by the laws of Delaware"
        clause = extraction(ClauseType.GOVERNING_LAW, True, quote)
        repaired = check_extractions(document(), [clause]).repaired_clauses[0]
        evidence = repaired.evidence[0]
        assert SOURCE[evidence.char_start : evidence.char_end] == quote

    def test_a_fabricated_quote_with_unknown_offsets_is_still_a_violation(self) -> None:
        """The one that matters. 0/0 must not launder a hallucination."""
        clause = extraction(
            ClauseType.NON_COMPETE, True, "Seller shall not compete in any territory"
        )
        report = check_extractions(document(), [clause])
        assert not report.checks[0].is_grounded
        assert len(report.ungrounded) == 1
        assert report.violation_rate == pytest.approx(1.0)

    def test_a_paraphrase_with_unknown_offsets_is_still_a_violation(self) -> None:
        """Near-misses are the realistic failure, not invented sentences."""
        clause = extraction(
            ClauseType.GOVERNING_LAW, True, "this agreement is governed by Delaware state law"
        )
        assert not check_extractions(document(), [clause]).checks[0].is_grounded

    def test_mixed_evidence_counts_only_the_bad_span(self) -> None:
        clause = ClauseExtraction(
            clause_type=ClauseType.GOVERNING_LAW,
            present=True,
            evidence=[
                Evidence(quote="governed by the laws of Delaware", char_start=0, char_end=0),
                Evidence(quote="a sentence that is not in the contract", char_start=0, char_end=0),
            ],
            confidence=0.9,
        )
        report = check_extractions(document(), [clause])
        assert report.total == 2
        assert len(report.ungrounded) == 1
        assert report.violation_rate == pytest.approx(0.5)
