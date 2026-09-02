"""Unit tests for the deterministic validation rules.

The section that matters most is `TestAuditCorrectedRules`. Three of the rules
CLAUDE.md section 5 proposes are wrong against the gold data, and these tests
pin the corrected behaviour so nobody "fixes" them back to the intuitive version:

- `cap_on_liability` and `uncapped_liability` are **not** mutually exclusive.
  Every one of the 111 CUAD contracts with uncapped liability also has a cap.
- `renewal_term => notice_period` holds 109/176 (62%), so it ships as INFO.
- `expiration_date => effective_date` holds 342/413 (83%), so it ships as INFO.

Severity is load-bearing throughout: only ERROR forces review, so a rule with
the wrong severity either floods the queue or silently stops gating.
"""

from __future__ import annotations

import pytest

from docintel.schemas import (
    PERPETUAL,
    ClauseExtraction,
    ClauseType,
    Document,
    Evidence,
    ExtractionResult,
    RuleViolation,
    Severity,
)
from docintel.validation.rules import (
    GLOBAL_CONFIDENCE_FLOOR,
    RULES,
    TIER3_CONFIDENCE_FLOOR,
    apply_rules,
    load_jurisdictions,
    needs_review,
)

JURISDICTIONS = load_jurisdictions()

SOURCE = (
    "2. GOVERNING LAW. This Agreement is governed by the laws of the State of "
    "Delaware. 3. LIABILITY. Liability is capped at fees paid, except that "
    "liability for breach of confidentiality is unlimited."
)


def doc() -> Document:
    return Document(document_id="DOC", text=SOURCE)


def good_evidence(quote: str = "governed by the laws of the State of Delaware") -> Evidence:
    start = SOURCE.index(quote)
    return Evidence(quote=quote, char_start=start, char_end=start + len(quote))


def clause(
    clause_type: ClauseType,
    present: bool = True,
    value: str | None = None,
    raw_text: str | None = None,
    evidence: list[Evidence] | None = None,
    confidence: float = 0.9,
) -> ClauseExtraction:
    if evidence is None:
        evidence = [good_evidence()] if present else []
    return ClauseExtraction(
        clause_type=clause_type,
        present=present,
        value=value,
        raw_text=raw_text,
        evidence=evidence,
        confidence=confidence,
    )


def ids(violations: list[RuleViolation], severity: Severity | None = None) -> set[str]:
    return {v.rule_id for v in violations if severity is None or v.severity is severity}


def run(clauses: list[ClauseExtraction], with_doc: bool = True) -> list[RuleViolation]:
    return apply_rules(clauses, document=doc() if with_doc else None, jurisdictions=JURISDICTIONS)


class TestRegistry:
    def test_ships_at_least_twenty_five_rules(self) -> None:
        """CLAUDE.md section 5 asks for 25 or more."""
        assert len(RULES) >= 25

    def test_rule_ids_are_unique(self) -> None:
        rule_ids = [rule.rule_id for rule in RULES]
        assert len(rule_ids) == len(set(rule_ids))

    def test_every_rule_has_a_description(self) -> None:
        assert all(rule.description.strip() for rule in RULES)

    def test_all_three_severities_are_used(self) -> None:
        """A ruleset that is all errors floods the review queue."""
        assert {rule.severity for rule in RULES} == {
            Severity.ERROR,
            Severity.WARNING,
            Severity.INFO,
        }

    def test_violations_are_sorted_most_severe_first(self) -> None:
        violations = run(
            [
                clause(ClauseType.GOVERNING_LAW, value="NOT_A_JURISDICTION"),
                clause(ClauseType.NON_COMPETE, confidence=0.5),
            ]
        )
        severities = [v.severity for v in violations]
        assert severities == sorted(
            severities, key=lambda s: {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}[s]
        )


class TestAuditCorrectedRules:
    """The three rules the data contradicted. Do not "fix" these back."""

    def test_cap_and_uncapped_together_is_not_an_error(self) -> None:
        """CLAUDE.md proposes these are mutually exclusive. All 111 CUAD
        contracts with uncapped liability also carry a cap, so the exclusion
        rule would fire on correct data every single time."""
        violations = run(
            [
                clause(ClauseType.CAP_ON_LIABILITY),
                clause(ClauseType.UNCAPPED_LIABILITY),
            ]
        )
        assert "cap_and_uncapped_mutually_exclusive" not in ids(violations)
        assert "uncapped_without_cap" not in ids(violations)

    def test_uncapped_without_cap_is_an_error(self) -> None:
        """The inverted form, which holds 111/111 on gold data."""
        violations = run([clause(ClauseType.UNCAPPED_LIABILITY)])
        assert "uncapped_without_cap" in ids(violations, Severity.ERROR)

    def test_cap_without_uncapped_is_fine(self) -> None:
        violations = run([clause(ClauseType.CAP_ON_LIABILITY)])
        assert "uncapped_without_cap" not in ids(violations)

    def test_renewal_without_notice_period_is_only_info(self) -> None:
        """Holds 109/176. As an ERROR it would be wrong 67 times."""
        violations = run([clause(ClauseType.RENEWAL_TERM, value="365 recurring")])
        assert "renewal_term_without_notice_period" in ids(violations, Severity.INFO)
        assert "renewal_term_without_notice_period" not in ids(violations, Severity.ERROR)

    def test_notice_period_without_renewal_is_a_warning(self) -> None:
        """The inverse holds 109/111, strong enough to warn on."""
        violations = run([clause(ClauseType.NOTICE_PERIOD_TO_TERMINATE_RENEWAL, value="30")])
        assert "notice_period_without_renewal_term" in ids(violations, Severity.WARNING)

    def test_expiration_without_effective_is_only_info(self) -> None:
        """Holds 342/413; the agreement date often serves as the start."""
        violations = run([clause(ClauseType.EXPIRATION_DATE, value="2024-12-31")])
        assert "expiration_without_effective" in ids(violations, Severity.INFO)


class TestPresenceConsistency:
    def test_present_without_evidence_is_an_error(self) -> None:
        """The rule that makes the grounding verifier bite.

        Grounding drops fabricated spans, leaving present=True with no evidence;
        this rule is what turns that into a rejection.
        """
        violations = run([clause(ClauseType.GOVERNING_LAW, value="US-DE", evidence=[])])
        assert "presence_requires_evidence" in ids(violations, Severity.ERROR)

    def test_absent_with_evidence_is_an_error(self) -> None:
        violations = run(
            [clause(ClauseType.NON_COMPETE, present=False, evidence=[good_evidence()])]
        )
        assert "absence_forbids_evidence" in ids(violations, Severity.ERROR)

    def test_absent_with_value_is_an_error(self) -> None:
        violations = run([clause(ClauseType.GOVERNING_LAW, present=False, value="US-DE")])
        assert "absence_forbids_value" in ids(violations, Severity.ERROR)

    def test_absent_with_raw_text_is_an_error(self) -> None:
        violations = run([clause(ClauseType.NON_COMPETE, present=False, raw_text="some clause")])
        assert "absence_forbids_raw_text" in ids(violations, Severity.ERROR)

    def test_clean_absent_clause_is_silent(self) -> None:
        violations = run([clause(ClauseType.NON_COMPETE, present=False)])
        for rule_id in (
            "absence_forbids_evidence",
            "absence_forbids_value",
            "absence_forbids_raw_text",
        ):
            assert rule_id not in ids(violations)


class TestValueFormats:
    def test_unknown_jurisdiction_is_an_error(self) -> None:
        violations = run([clause(ClauseType.GOVERNING_LAW, value="Atlantis")])
        assert "governing_law_unknown_jurisdiction" in ids(violations, Severity.ERROR)

    def test_known_jurisdiction_id_passes(self) -> None:
        violations = run([clause(ClauseType.GOVERNING_LAW, value="US-DE")])
        assert "governing_law_unknown_jurisdiction" not in ids(violations)

    def test_multi_jurisdiction_value_is_split_on_semicolons(self) -> None:
        """CUAD genuinely has contracts naming two forums."""
        violations = run([clause(ClauseType.GOVERNING_LAW, value="US-IL; US-NY")])
        assert "governing_law_unknown_jurisdiction" not in ids(violations)

    def test_one_bad_jurisdiction_among_several_is_caught(self) -> None:
        violations = run([clause(ClauseType.GOVERNING_LAW, value="US-IL; Atlantis")])
        assert "governing_law_unknown_jurisdiction" in ids(violations, Severity.ERROR)

    def test_sentinel_jurisdiction_is_accepted(self) -> None:
        violations = run([clause(ClauseType.GOVERNING_LAW, value="DEFERRED")])
        assert "governing_law_unknown_jurisdiction" not in ids(violations)

    @pytest.mark.parametrize("value", ["2019", "2019-03", "2019-03-15"])
    def test_iso_date_prefixes_are_accepted(self, value: str) -> None:
        """Partial dates are required, not optional: `[]/[]/2020` is CUAD's most
        common agreement-date value."""
        violations = run([clause(ClauseType.EFFECTIVE_DATE, value=value)])
        assert "date_unparseable" not in ids(violations)

    @pytest.mark.parametrize("value", ["March 15, 2019", "15/03/2019", "3/15/19", "soon"])
    def test_non_iso_dates_are_errors(self, value: str) -> None:
        violations = run([clause(ClauseType.EFFECTIVE_DATE, value=value)])
        assert "date_unparseable" in ids(violations, Severity.ERROR)

    def test_perpetual_is_valid_for_expiration(self) -> None:
        violations = run([clause(ClauseType.EXPIRATION_DATE, value=PERPETUAL)])
        assert "date_unparseable" not in ids(violations)

    def test_perpetual_is_invalid_for_effective_date(self) -> None:
        violations = run([clause(ClauseType.EFFECTIVE_DATE, value=PERPETUAL)])
        assert "perpetual_effective_date" in ids(violations, Severity.ERROR)

    def test_implausible_year_is_an_error(self) -> None:
        violations = run([clause(ClauseType.EFFECTIVE_DATE, value="0201")])
        assert "date_implausible_year" in ids(violations, Severity.ERROR)

    def test_notice_period_must_be_an_integer(self) -> None:
        violations = run(
            [clause(ClauseType.NOTICE_PERIOD_TO_TERMINATE_RENEWAL, value="thirty days")]
        )
        assert "notice_period_not_integer" in ids(violations, Severity.ERROR)

    def test_notice_period_integer_passes(self) -> None:
        violations = run([clause(ClauseType.NOTICE_PERIOD_TO_TERMINATE_RENEWAL, value="30")])
        assert "notice_period_not_integer" not in ids(violations)

    def test_negative_notice_period_is_an_error(self) -> None:
        violations = run([clause(ClauseType.NOTICE_PERIOD_TO_TERMINATE_RENEWAL, value="-5")])
        assert "notice_period_non_positive" in ids(violations, Severity.ERROR)

    def test_absurd_notice_period_warns_about_units(self) -> None:
        """'3 years' normalized as 3 -> then re-read as years gives 1095+."""
        violations = run([clause(ClauseType.NOTICE_PERIOD_TO_TERMINATE_RENEWAL, value="99999")])
        assert "notice_period_implausible" in ids(violations, Severity.WARNING)

    def test_boolean_clause_with_a_value_warns(self) -> None:
        """A summary in `value` is a paraphrase, which grounding cannot verify."""
        violations = run([clause(ClauseType.CAP_ON_LIABILITY, value="capped at fees")])
        assert "boolean_clause_carries_no_value" in ids(violations, Severity.WARNING)

    def test_valued_clause_without_a_value_warns(self) -> None:
        violations = run([clause(ClauseType.GOVERNING_LAW, value=None)])
        assert "valued_clause_missing_value" in ids(violations, Severity.WARNING)


class TestDateOrdering:
    def test_effective_after_expiration_is_an_error(self) -> None:
        violations = run(
            [
                clause(ClauseType.EFFECTIVE_DATE, value="2024-06-01"),
                clause(ClauseType.EXPIRATION_DATE, value="2023-01-01"),
            ]
        )
        assert "effective_after_expiration" in ids(violations, Severity.ERROR)

    def test_correct_order_passes(self) -> None:
        violations = run(
            [
                clause(ClauseType.EFFECTIVE_DATE, value="2019-03-15"),
                clause(ClauseType.EXPIRATION_DATE, value="2024-12-31"),
            ]
        )
        assert "effective_after_expiration" not in ids(violations)

    def test_year_only_comparison_still_catches_inversion(self) -> None:
        violations = run(
            [
                clause(ClauseType.EFFECTIVE_DATE, value="2024"),
                clause(ClauseType.EXPIRATION_DATE, value="2019"),
            ]
        )
        assert "effective_after_expiration" in ids(violations, Severity.ERROR)

    def test_same_year_at_year_precision_is_skipped_not_failed(self) -> None:
        """A rule that cannot evaluate must not report a violation.

        Both sides are 2019 at year precision; there is no way to know the order,
        so silence is the correct output.
        """
        violations = run(
            [
                clause(ClauseType.EFFECTIVE_DATE, value="2019"),
                clause(ClauseType.EXPIRATION_DATE, value="2019"),
            ]
        )
        assert "effective_after_expiration" not in ids(violations)

    def test_perpetual_expiration_is_never_before_the_start(self) -> None:
        violations = run(
            [
                clause(ClauseType.EFFECTIVE_DATE, value="2019-01-01"),
                clause(ClauseType.EXPIRATION_DATE, value=PERPETUAL),
            ]
        )
        assert "effective_after_expiration" not in ids(violations)

    def test_mixed_precision_within_a_year_is_compared(self) -> None:
        violations = run(
            [
                clause(ClauseType.EFFECTIVE_DATE, value="2019-06"),
                clause(ClauseType.EXPIRATION_DATE, value="2019-03-01"),
            ]
        )
        assert "effective_after_expiration" in ids(violations, Severity.ERROR)


class TestEvidenceQuality:
    def test_very_short_quote_warns(self) -> None:
        violations = run(
            [
                clause(
                    ClauseType.GOVERNING_LAW,
                    value="US-DE",
                    evidence=[Evidence(quote="laws", char_start=0, char_end=4)],
                )
            ]
        )
        assert "evidence_too_short" in ids(violations, Severity.WARNING)

    def test_evidence_beyond_the_document_is_an_error(self) -> None:
        violations = run(
            [
                clause(
                    ClauseType.GOVERNING_LAW,
                    value="US-DE",
                    evidence=[Evidence(quote="a" * 40, char_start=99_000, char_end=99_040)],
                )
            ]
        )
        assert "evidence_outside_document" in ids(violations, Severity.ERROR)

    def test_duplicate_spans_warn(self) -> None:
        item = good_evidence()
        violations = run([clause(ClauseType.GOVERNING_LAW, value="US-DE", evidence=[item, item])])
        assert "evidence_duplicated" in ids(violations, Severity.WARNING)

    def test_paraphrased_raw_text_is_an_error(self) -> None:
        violations = run(
            [
                clause(
                    ClauseType.GOVERNING_LAW,
                    value="US-DE",
                    raw_text="The contract says Delaware law applies.",
                )
            ]
        )
        assert "raw_text_not_verbatim" in ids(violations, Severity.ERROR)

    def test_verbatim_raw_text_passes(self) -> None:
        violations = run(
            [
                clause(
                    ClauseType.GOVERNING_LAW,
                    value="US-DE",
                    raw_text="governed by the laws of the State of Delaware",
                )
            ]
        )
        assert "raw_text_not_verbatim" not in ids(violations)

    def test_raw_text_reflowed_across_lines_still_passes(self) -> None:
        """Comparison is whitespace-normalized, so re-wrapping is not paraphrase."""
        violations = run(
            [
                clause(
                    ClauseType.GOVERNING_LAW,
                    value="US-DE",
                    raw_text="governed by the laws\nof the   State of Delaware",
                )
            ]
        )
        assert "raw_text_not_verbatim" not in ids(violations)

    def test_raw_text_is_not_checked_without_a_document(self) -> None:
        violations = run(
            [clause(ClauseType.GOVERNING_LAW, value="US-DE", raw_text="anything")],
            with_doc=False,
        )
        assert "raw_text_not_verbatim" not in ids(violations)


class TestConfidenceGates:
    def test_low_confidence_tier3_warns(self) -> None:
        violations = run([clause(ClauseType.NON_COMPETE, confidence=0.5)])
        assert "tier3_low_confidence" in ids(violations, Severity.WARNING)

    def test_same_confidence_on_tier1_does_not_trip_the_tier3_gate(self) -> None:
        violations = run([clause(ClauseType.GOVERNING_LAW, value="US-DE", confidence=0.5)])
        assert "tier3_low_confidence" not in ids(violations)

    def test_very_low_confidence_warns_on_any_tier(self) -> None:
        violations = run([clause(ClauseType.GOVERNING_LAW, value="US-DE", confidence=0.1)])
        assert "very_low_confidence" in ids(violations, Severity.WARNING)

    def test_floors_are_ordered(self) -> None:
        assert GLOBAL_CONFIDENCE_FLOOR < TIER3_CONFIDENCE_FLOOR

    def test_constant_confidence_across_clauses_warns(self) -> None:
        """A model answering 1.0 for everything has produced no signal, and
        routing on a constant is routing on nothing."""
        violations = run(
            [
                clause(ClauseType.GOVERNING_LAW, value="US-DE", confidence=1.0),
                clause(ClauseType.PARTIES, value="A (X); B (Y)", confidence=1.0),
                clause(ClauseType.CAP_ON_LIABILITY, confidence=1.0),
            ]
        )
        assert "confidence_is_constant" in ids(violations, Severity.WARNING)

    def test_varied_confidence_does_not_warn(self) -> None:
        violations = run(
            [
                clause(ClauseType.GOVERNING_LAW, value="US-DE", confidence=0.95),
                clause(ClauseType.PARTIES, value="A (X); B (Y)", confidence=0.88),
                clause(ClauseType.CAP_ON_LIABILITY, confidence=0.72),
            ]
        )
        assert "confidence_is_constant" not in ids(violations)


class TestDocumentLevel:
    def test_all_absent_is_an_error(self) -> None:
        violations = run([clause(ct, present=False) for ct in list(ClauseType)[:5]])
        assert "nothing_extracted" in ids(violations, Severity.ERROR)

    def test_all_twelve_present_warns(self) -> None:
        """The rarest of the twelve appears in 21.8% of contracts."""
        violations = run(
            [
                clause(ct, value=None if ct.name in {"CAP_ON_LIABILITY"} else "x")
                for ct in ClauseType
            ]
        )
        assert "everything_present" in ids(violations, Severity.WARNING)

    def test_missing_clause_types_warn(self) -> None:
        violations = run([clause(ClauseType.GOVERNING_LAW, value="US-DE")])
        assert "incomplete_clause_coverage" in ids(violations, Severity.WARNING)

    def test_duplicate_clause_type_is_an_error(self) -> None:
        """Keyed lookups collapse duplicates silently, which is the bug."""
        violations = run(
            [
                clause(ClauseType.GOVERNING_LAW, value="US-DE"),
                clause(ClauseType.GOVERNING_LAW, value="US-NY"),
            ]
        )
        assert "duplicate_clause_types" in ids(violations, Severity.ERROR)

    def test_parties_absent_warns(self) -> None:
        violations = run(
            [
                clause(ClauseType.PARTIES, present=False),
                clause(ClauseType.GOVERNING_LAW, value="US-DE"),
            ]
        )
        assert "parties_absent" in ids(violations, Severity.WARNING)

    def test_empty_clause_list_does_not_crash(self) -> None:
        assert isinstance(run([]), list)


class TestNeedsReview:
    def _result(self, **kwargs: object) -> ExtractionResult:
        defaults: dict[str, object] = {"document_id": "DOC"}
        defaults.update(kwargs)
        return ExtractionResult(**defaults)  # type: ignore[arg-type]

    def test_clean_result_needs_no_review(self) -> None:
        result = self._result(
            clauses=[clause(ClauseType.GOVERNING_LAW, value="US-DE", confidence=0.95)]
        )
        flagged, reasons = needs_review(result)
        assert not flagged
        assert reasons == []

    def test_error_violation_forces_review(self) -> None:
        result = self._result(
            violations=run([clause(ClauseType.UNCAPPED_LIABILITY)]),
        )
        flagged, reasons = needs_review(result)
        assert flagged
        assert any("error-severity" in r for r in reasons)

    def test_warning_alone_does_not_force_review(self) -> None:
        """Otherwise the queue fills with things nobody needs to look at."""
        result = self._result(
            violations=[
                RuleViolation(
                    rule_id="evidence_too_short",
                    severity=Severity.WARNING,
                    message="short quote",
                )
            ]
        )
        flagged, _ = needs_review(result)
        assert not flagged

    def test_low_confidence_tier3_forces_review(self) -> None:
        result = self._result(clauses=[clause(ClauseType.NON_COMPETE, confidence=0.4)])
        flagged, reasons = needs_review(result)
        assert flagged
        assert any("Tier-3" in r for r in reasons)

    def test_budget_stop_forces_review(self) -> None:
        """An extraction cut off mid-run is incomplete by construction."""
        result = self._result(stopped_on_budget=True)
        flagged, reasons = needs_review(result)
        assert flagged
        assert any("budget" in r for r in reasons)

    def test_reasons_accumulate(self) -> None:
        result = self._result(
            clauses=[clause(ClauseType.NON_COMPETE, confidence=0.3)],
            violations=run([clause(ClauseType.UNCAPPED_LIABILITY)]),
            stopped_on_budget=True,
        )
        _, reasons = needs_review(result)
        assert len(reasons) == 3


class TestNoOrphanedRules:
    """Guards a bug that already happened once.

    `_confidence_is_meaningful` was written, tested, and never added to `RULES`.
    Everything passed except the one test that expected it to fire -- a rule that
    exists but is not registered is silently dead code, and the ruleset looks
    complete from the outside.
    """

    def test_every_check_function_is_registered(self) -> None:
        import inspect

        from docintel.validation import rules as module

        registered = {rule.check for rule in RULES}
        check_functions = [
            function
            for name, function in vars(module).items()
            if name.startswith("_")
            and inspect.isfunction(function)
            and str(inspect.signature(function).return_annotation).endswith("list[RuleViolation]")
        ]
        orphans = sorted(f.__name__ for f in check_functions if f not in registered)
        assert not orphans, f"rule functions never added to RULES: {orphans}"

    def test_registry_severity_matches_what_the_rule_emits(self) -> None:
        """A registry entry claiming ERROR while the function emits WARNING makes
        the rule catalogue lie about what gates."""
        emitted = {
            "renewal_term_without_notice_period": Severity.INFO,
            "expiration_without_effective": Severity.INFO,
            "non_compete_without_exclusivity": Severity.INFO,
        }
        for rule in RULES:
            if rule.rule_id in emitted:
                assert rule.severity is emitted[rule.rule_id], rule.rule_id
