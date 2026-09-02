"""Deterministic validation rules.

Each rule is a pure function from an extraction result to zero or more
``RuleViolation``. No model calls, no network, no I/O -- which is what makes them
cheap enough to run on every output and trustworthy enough to gate on.

**Three of the rules CLAUDE.md section 5 proposes are wrong, and the audit
measured it rather than guessing.** A rule that fires on correct data is worse
than no rule: it teaches whoever reads the review queue to ignore violations.

| Proposed | Holds | Violated | Shipped as |
|---|---|---|---|
| ``cap_on_liability`` and ``uncapped_liability`` mutually exclusive | 399 | 111 | **inverted** |
| ``uncapped_liability`` implies ``cap_on_liability`` | 111 | 0 | ERROR |
| ``renewal_term`` implies ``notice_period`` | 109 | 67 | WARNING, inverted |
| ``notice_period`` implies ``renewal_term`` | 109 | 2 | WARNING |
| ``expiration_date`` implies ``effective_date`` | 342 | 71 | INFO |

Every contract in CUAD carrying an uncapped-liability clause also carries a cap
on liability, and there are zero uncapped-only contracts. The relation is subset,
not exclusion: a general cap with specific carve-outs. You cannot carve an
exception out of a cap that does not exist. See ``docs/DATA_AUDIT.md``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final

import yaml

from docintel.schemas import (
    BOOLEAN_CLAUSES,
    PERPETUAL,
    UNSPECIFIED,
    ClauseExtraction,
    ClauseType,
    Document,
    ExtractionResult,
    RuleViolation,
    Severity,
    Tier,
)

__all__ = [
    "RULES",
    "JurisdictionIndex",
    "Rule",
    "apply_rules",
    "load_jurisdictions",
    "needs_review",
]

REFERENCE_DIR: Final = Path(__file__).resolve().parents[3] / "data" / "reference"

#: Confidence below which a Tier-3 clause routes to review. Tier 3 is where
#: annotators themselves disagree, so a hedged answer there is worth a human
#: glance in a way the same number on Tier 1 is not.
TIER3_CONFIDENCE_FLOOR: Final = 0.70

#: Confidence below which any clause routes to review.
GLOBAL_CONFIDENCE_FLOOR: Final = 0.40

#: A quote this short cannot identify a clause; it is usually a stray fragment.
MIN_EVIDENCE_CHARS: Final = 12

#: CUAD contracts are SEC filings from roughly 1996 onward. A date outside this
#: window is a parse error, not a contract term.
MIN_PLAUSIBLE_YEAR: Final = 1900
MAX_PLAUSIBLE_YEAR: Final = 2100

#: Notice periods beyond this are almost certainly a unit confusion
#: (for example "3" read as years and normalized to days).
MAX_PLAUSIBLE_NOTICE_DAYS: Final = 1825  # five years

_ISO_DATE: Final = re.compile(r"^(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?$")


# --------------------------------------------------------------------------
# Reference data
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class JurisdictionIndex:
    """Closed vocabulary of governing-law values, from jurisdictions.yaml."""

    #: Canonical id -> display name.
    ids: dict[str, str]
    #: Sentinel ids such as UNSPECIFIED and DEFERRED.
    sentinels: frozenset[str]

    def is_known(self, value: str) -> bool:
        return value in self.ids or value in self.sentinels

    @property
    def all_ids(self) -> frozenset[str]:
        return frozenset(self.ids) | self.sentinels


def load_jurisdictions(path: Path | None = None) -> JurisdictionIndex:
    """Load the jurisdiction vocabulary."""
    source = path or REFERENCE_DIR / "jurisdictions.yaml"
    payload: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8"))
    return JurisdictionIndex(
        ids={entry["id"]: entry["canonical"] for entry in payload["jurisdictions"]},
        sentinels=frozenset(entry["id"] for entry in payload["sentinels"]),
    )


# --------------------------------------------------------------------------
# Rule plumbing
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RuleContext:
    """What a rule is allowed to look at."""

    #: Keyed by clause type. Duplicates have already collapsed here, which is
    #: why `all_clauses` exists.
    clauses: dict[ClauseType, ClauseExtraction]
    #: Every extraction as returned, order preserved and duplicates intact.
    all_clauses: tuple[ClauseExtraction, ...]
    document: Document | None
    jurisdictions: JurisdictionIndex

    def get(self, clause_type: ClauseType) -> ClauseExtraction | None:
        return self.clauses.get(clause_type)

    def is_present(self, clause_type: ClauseType) -> bool:
        clause = self.clauses.get(clause_type)
        return bool(clause and clause.present)


@dataclass(frozen=True, slots=True)
class Rule:
    """One deterministic check."""

    rule_id: str
    severity: Severity
    description: str
    check: Callable[[RuleContext], list[RuleViolation]]


def _violation(rule: str, severity: Severity, message: str, *types: ClauseType) -> RuleViolation:
    return RuleViolation(rule_id=rule, severity=severity, message=message, clause_types=list(types))


def _parse_partial_date(value: str) -> tuple[int, int | None, int | None] | None:
    """Parse an ISO 8601 prefix: ``2019``, ``2019-03``, or ``2019-03-15``.

    Partial dates are not optional here. ``[]/[]/2020`` is the most common gold
    value for ``Agreement Date`` in CUAD, so ``value`` cannot be a
    ``datetime.date`` -- see data/reference/normalization.md section 1.
    """
    match = _ISO_DATE.match(value.strip())
    if not match:
        return None
    year, month, day = match.groups()
    return int(year), int(month) if month else None, int(day) if day else None


def _comparable(parsed: tuple[int, int | None, int | None]) -> date | None:
    """Lower bound of a partial date, for ordering. None when only a year."""
    year, month, day = parsed
    if month is None:
        return None
    try:
        return date(year, month, day or 1)
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Field rules
# --------------------------------------------------------------------------


def _presence_requires_evidence(ctx: RuleContext) -> list[RuleViolation]:
    return [
        _violation(
            "presence_requires_evidence",
            Severity.ERROR,
            f"{clause.clause_type} is present but cites no evidence",
            clause.clause_type,
        )
        for clause in ctx.clauses.values()
        if clause.present and not clause.evidence
    ]


def _absence_forbids_evidence(ctx: RuleContext) -> list[RuleViolation]:
    return [
        _violation(
            "absence_forbids_evidence",
            Severity.ERROR,
            f"{clause.clause_type} is absent but cites {len(clause.evidence)} evidence span(s)",
            clause.clause_type,
        )
        for clause in ctx.clauses.values()
        if not clause.present and clause.evidence
    ]


def _absence_forbids_value(ctx: RuleContext) -> list[RuleViolation]:
    return [
        _violation(
            "absence_forbids_value",
            Severity.ERROR,
            f"{clause.clause_type} is absent but carries value {clause.value!r}",
            clause.clause_type,
        )
        for clause in ctx.clauses.values()
        if not clause.present and clause.value is not None
    ]


def _absence_forbids_raw_text(ctx: RuleContext) -> list[RuleViolation]:
    return [
        _violation(
            "absence_forbids_raw_text",
            Severity.ERROR,
            f"{clause.clause_type} is absent but carries raw_text",
            clause.clause_type,
        )
        for clause in ctx.clauses.values()
        if not clause.present and clause.raw_text
    ]


def _boolean_clauses_carry_no_value(ctx: RuleContext) -> list[RuleViolation]:
    """The six boolean clause types must not invent a summary string.

    A summary is a paraphrase, and a paraphrase cannot be verified by the
    grounding check -- which is the whole reason these fields are null.
    """
    return [
        _violation(
            "boolean_clause_carries_no_value",
            Severity.WARNING,
            f"{clause.clause_type} is a presence-only clause but returned "
            f"value {clause.value!r}; use raw_text for the clause text",
            clause.clause_type,
        )
        for clause in ctx.clauses.values()
        if clause.clause_type in BOOLEAN_CLAUSES and clause.value is not None
    ]


def _valued_clause_has_value(ctx: RuleContext) -> list[RuleViolation]:
    return [
        _violation(
            "valued_clause_missing_value",
            Severity.WARNING,
            f"{clause.clause_type} is present but returned no normalized value",
            clause.clause_type,
        )
        for clause in ctx.clauses.values()
        if clause.present and clause.clause_type not in BOOLEAN_CLAUSES and clause.value is None
    ]


def _governing_law_is_known(ctx: RuleContext) -> list[RuleViolation]:
    clause = ctx.get(ClauseType.GOVERNING_LAW)
    if not clause or not clause.present or clause.value is None:
        return []
    # The field is a list: CUAD separates multiple forums with ";" and a few
    # contracts genuinely name two. See normalization.md section 3.
    unknown = [
        part.strip()
        for part in clause.value.split(";")
        if part.strip() and not ctx.jurisdictions.is_known(part.strip())
    ]
    if not unknown:
        return []
    return [
        _violation(
            "governing_law_unknown_jurisdiction",
            Severity.ERROR,
            f"governing_law value(s) {unknown} do not resolve to jurisdictions.yaml",
            ClauseType.GOVERNING_LAW,
        )
    ]


def _dates_parse(ctx: RuleContext) -> list[RuleViolation]:
    violations: list[RuleViolation] = []
    for clause_type in (ClauseType.EFFECTIVE_DATE, ClauseType.EXPIRATION_DATE):
        clause = ctx.get(clause_type)
        if not clause or not clause.present or clause.value is None:
            continue
        if clause.value in {PERPETUAL, UNSPECIFIED}:
            continue
        if _parse_partial_date(clause.value) is None:
            violations.append(
                _violation(
                    "date_unparseable",
                    Severity.ERROR,
                    f"{clause_type} value {clause.value!r} is not an ISO 8601 "
                    f"date prefix (YYYY, YYYY-MM, or YYYY-MM-DD)",
                    clause_type,
                )
            )
    return violations


def _dates_are_plausible(ctx: RuleContext) -> list[RuleViolation]:
    violations: list[RuleViolation] = []
    for clause_type in (ClauseType.EFFECTIVE_DATE, ClauseType.EXPIRATION_DATE):
        clause = ctx.get(clause_type)
        if not clause or not clause.present or clause.value is None:
            continue
        parsed = _parse_partial_date(clause.value)
        if parsed is None:
            continue
        year = parsed[0]
        if not MIN_PLAUSIBLE_YEAR <= year <= MAX_PLAUSIBLE_YEAR:
            violations.append(
                _violation(
                    "date_implausible_year",
                    Severity.ERROR,
                    f"{clause_type} year {year} is outside "
                    f"{MIN_PLAUSIBLE_YEAR}-{MAX_PLAUSIBLE_YEAR}",
                    clause_type,
                )
            )
    return violations


def _perpetual_only_on_expiration(ctx: RuleContext) -> list[RuleViolation]:
    clause = ctx.get(ClauseType.EFFECTIVE_DATE)
    if clause and clause.value == PERPETUAL:
        return [
            _violation(
                "perpetual_effective_date",
                Severity.ERROR,
                "effective_date cannot be PERPETUAL; a contract has a start",
                ClauseType.EFFECTIVE_DATE,
            )
        ]
    return []


def _notice_period_is_integer_days(ctx: RuleContext) -> list[RuleViolation]:
    clause = ctx.get(ClauseType.NOTICE_PERIOD_TO_TERMINATE_RENEWAL)
    if not clause or not clause.present or clause.value is None:
        return []
    if clause.value == UNSPECIFIED:
        return []
    try:
        days = int(clause.value)
    except ValueError:
        return [
            _violation(
                "notice_period_not_integer",
                Severity.ERROR,
                f"notice_period value {clause.value!r} is not an integer day count",
                ClauseType.NOTICE_PERIOD_TO_TERMINATE_RENEWAL,
            )
        ]
    if days <= 0:
        return [
            _violation(
                "notice_period_non_positive",
                Severity.ERROR,
                f"notice_period {days} days must be positive",
                ClauseType.NOTICE_PERIOD_TO_TERMINATE_RENEWAL,
            )
        ]
    if days > MAX_PLAUSIBLE_NOTICE_DAYS:
        return [
            _violation(
                "notice_period_implausible",
                Severity.WARNING,
                f"notice_period {days} days exceeds {MAX_PLAUSIBLE_NOTICE_DAYS}; "
                "likely a unit confusion",
                ClauseType.NOTICE_PERIOD_TO_TERMINATE_RENEWAL,
            )
        ]
    return []


def _confidence_is_meaningful(ctx: RuleContext) -> list[RuleViolation]:
    """Flag uniform confidence, which means the field carries no signal.

    A model that answers 1.0 for everything, or 0.5 for everything, has produced
    a constant, and routing on a constant is routing on nothing.
    """
    if len(ctx.clauses) < 3:
        return []
    values = {clause.confidence for clause in ctx.clauses.values()}
    if len(values) == 1:
        return [
            _violation(
                "confidence_is_constant",
                Severity.WARNING,
                f"every clause reported confidence {values.pop()}; the field is carrying no signal",
            )
        ]
    return []


def _evidence_is_substantive(ctx: RuleContext) -> list[RuleViolation]:
    return [
        _violation(
            "evidence_too_short",
            Severity.WARNING,
            f"{clause.clause_type} cites a {len(item.quote)}-character quote "
            f"({item.quote!r}); too short to identify a clause",
            clause.clause_type,
        )
        for clause in ctx.clauses.values()
        for item in clause.evidence
        if len(item.quote.strip()) < MIN_EVIDENCE_CHARS
    ]


def _evidence_offsets_are_ordered(ctx: RuleContext) -> list[RuleViolation]:
    return [
        _violation(
            "evidence_offsets_inverted",
            Severity.ERROR,
            f"{clause.clause_type} evidence has char_end <= char_start",
            clause.clause_type,
        )
        for clause in ctx.clauses.values()
        for item in clause.evidence
        if item.char_end <= item.char_start
    ]


def _evidence_within_document(ctx: RuleContext) -> list[RuleViolation]:
    if ctx.document is None:
        return []
    length = len(ctx.document.text)
    return [
        _violation(
            "evidence_outside_document",
            Severity.ERROR,
            f"{clause.clause_type} evidence span [{item.char_start}, {item.char_end}) "
            f"falls outside a {length}-character document",
            clause.clause_type,
        )
        for clause in ctx.clauses.values()
        for item in clause.evidence
        if item.char_end > length
    ]


def _evidence_spans_do_not_duplicate(ctx: RuleContext) -> list[RuleViolation]:
    violations: list[RuleViolation] = []
    for clause in ctx.clauses.values():
        seen = {(item.char_start, item.char_end) for item in clause.evidence}
        if len(seen) != len(clause.evidence):
            violations.append(
                _violation(
                    "evidence_duplicated",
                    Severity.WARNING,
                    f"{clause.clause_type} cites the same span more than once",
                    clause.clause_type,
                )
            )
    return violations


def _raw_text_is_not_paraphrased(ctx: RuleContext) -> list[RuleViolation]:
    """``raw_text`` must be a verbatim substring, never a summary."""
    if ctx.document is None:
        return []
    from docintel.text import normalize_whitespace

    haystack = normalize_whitespace(ctx.document.text)
    return [
        _violation(
            "raw_text_not_verbatim",
            Severity.ERROR,
            f"{clause.clause_type} raw_text does not appear in the document; it looks paraphrased",
            clause.clause_type,
        )
        for clause in ctx.clauses.values()
        if clause.raw_text and normalize_whitespace(clause.raw_text) not in haystack
    ]


def _no_duplicate_clause_types(ctx: RuleContext) -> list[RuleViolation]:
    """Reject the same clause type appearing twice in one result.

    Reads `all_clauses` rather than `clauses`, because the dict keyed by type has
    already silently collapsed any duplicate by the time rules run -- which is
    exactly the bug this rule is meant to catch.
    """
    counts: dict[ClauseType, int] = {}
    for clause in ctx.all_clauses:
        counts[clause.clause_type] = counts.get(clause.clause_type, 0) + 1
    return [
        _violation(
            "duplicate_clause_types",
            Severity.ERROR,
            f"{clause_type} appears {count} times in one result",
            clause_type,
        )
        for clause_type, count in sorted(counts.items())
        if count > 1
    ]


# --------------------------------------------------------------------------
# Cross-field rules
# --------------------------------------------------------------------------


def _uncapped_implies_cap(ctx: RuleContext) -> list[RuleViolation]:
    """The inverted form of CLAUDE.md's mutual-exclusion rule.

    Holds 111/111 on gold data. An uncapped-liability carve-out presupposes a
    general cap to carve out of.
    """
    if ctx.is_present(ClauseType.UNCAPPED_LIABILITY) and not ctx.is_present(
        ClauseType.CAP_ON_LIABILITY
    ):
        return [
            _violation(
                "uncapped_without_cap",
                Severity.ERROR,
                "uncapped_liability is present without cap_on_liability; "
                "every such contract in CUAD carries both",
                ClauseType.UNCAPPED_LIABILITY,
                ClauseType.CAP_ON_LIABILITY,
            )
        ]
    return []


def _notice_period_implies_renewal(ctx: RuleContext) -> list[RuleViolation]:
    """Holds 109/111 on gold data, so WARNING rather than ERROR."""
    if ctx.is_present(ClauseType.NOTICE_PERIOD_TO_TERMINATE_RENEWAL) and not ctx.is_present(
        ClauseType.RENEWAL_TERM
    ):
        return [
            _violation(
                "notice_period_without_renewal_term",
                Severity.WARNING,
                "a notice period to terminate renewal implies a renewal term",
                ClauseType.NOTICE_PERIOD_TO_TERMINATE_RENEWAL,
                ClauseType.RENEWAL_TERM,
            )
        ]
    return []


def _renewal_implies_notice_period(ctx: RuleContext) -> list[RuleViolation]:
    """CLAUDE.md proposes this as a hard dependency. It holds 109/176 -- 62%.

    Shipped as INFO, not WARNING: at a 38% miss rate it is a weak signal, and
    warning on it would bury the warnings that mean something. Plenty of
    contracts renew automatically with no notice requirement at all.
    """
    if ctx.is_present(ClauseType.RENEWAL_TERM) and not ctx.is_present(
        ClauseType.NOTICE_PERIOD_TO_TERMINATE_RENEWAL
    ):
        return [
            _violation(
                "renewal_term_without_notice_period",
                Severity.INFO,
                "renewal_term without a notice period; common (38% of gold) but worth recording",
                ClauseType.RENEWAL_TERM,
                ClauseType.NOTICE_PERIOD_TO_TERMINATE_RENEWAL,
            )
        ]
    return []


def _effective_before_expiration(ctx: RuleContext) -> list[RuleViolation]:
    """``effective_date <= expiration_date``, skipped when not comparable.

    Skipping matters: CUAD's dates are frequently year-only, and PERPETUAL is not
    a date at all. A rule that cannot evaluate must not report a violation.
    """
    effective = ctx.get(ClauseType.EFFECTIVE_DATE)
    expiration = ctx.get(ClauseType.EXPIRATION_DATE)
    if not (effective and expiration and effective.present and expiration.present):
        return []
    if effective.value is None or expiration.value is None:
        return []
    if expiration.value in {PERPETUAL, UNSPECIFIED} or effective.value == UNSPECIFIED:
        return []

    start = _parse_partial_date(effective.value)
    end = _parse_partial_date(expiration.value)
    if start is None or end is None:
        return []

    # Year-level comparison first; it is always available.
    if start[0] > end[0]:
        return [
            _violation(
                "effective_after_expiration",
                Severity.ERROR,
                f"effective_date {effective.value} is after expiration_date {expiration.value}",
                ClauseType.EFFECTIVE_DATE,
                ClauseType.EXPIRATION_DATE,
            )
        ]
    if start[0] < end[0]:
        return []

    # Same year: compare only if both sides resolve to a real date.
    lower, upper = _comparable(start), _comparable(end)
    if lower is None or upper is None:
        return []
    if lower > upper:
        return [
            _violation(
                "effective_after_expiration",
                Severity.ERROR,
                f"effective_date {effective.value} is after expiration_date {expiration.value}",
                ClauseType.EFFECTIVE_DATE,
                ClauseType.EXPIRATION_DATE,
            )
        ]
    return []


def _expiration_implies_effective(ctx: RuleContext) -> list[RuleViolation]:
    """Holds 342/413 on gold data -- INFO only."""
    if ctx.is_present(ClauseType.EXPIRATION_DATE) and not ctx.is_present(ClauseType.EFFECTIVE_DATE):
        return [
            _violation(
                "expiration_without_effective",
                Severity.INFO,
                "expiration_date without effective_date; often the agreement "
                "date serves as the start",
                ClauseType.EXPIRATION_DATE,
                ClauseType.EFFECTIVE_DATE,
            )
        ]
    return []


def _parties_are_present(ctx: RuleContext) -> list[RuleViolation]:
    """Parties appear in 509 of 510 contracts. Absence is almost always an error."""
    clause = ctx.get(ClauseType.PARTIES)
    if clause is not None and not clause.present:
        return [
            _violation(
                "parties_absent",
                Severity.WARNING,
                "no parties found; 509 of 510 CUAD contracts name their parties",
                ClauseType.PARTIES,
            )
        ]
    return []


def _non_compete_and_exclusivity_coherent(ctx: RuleContext) -> list[RuleViolation]:
    """A non-compete with no exclusivity is unusual enough to record.

    INFO rather than WARNING: the two clause types are related but genuinely
    independent, and this has not been measured against gold yet.
    """
    if ctx.is_present(ClauseType.NON_COMPETE) and not ctx.is_present(ClauseType.EXCLUSIVITY):
        return [
            _violation(
                "non_compete_without_exclusivity",
                Severity.INFO,
                "non_compete without exclusivity; related restraints usually co-occur",
                ClauseType.NON_COMPETE,
                ClauseType.EXCLUSIVITY,
            )
        ]
    return []


def _tier3_confidence_gate(ctx: RuleContext) -> list[RuleViolation]:
    return [
        _violation(
            "tier3_low_confidence",
            Severity.WARNING,
            f"{clause.clause_type} (Tier 3) confidence {clause.confidence:.2f} "
            f"is below {TIER3_CONFIDENCE_FLOOR}",
            clause.clause_type,
        )
        for clause in ctx.clauses.values()
        if clause.tier is Tier.REQUIRES_JUDGEMENT
        and clause.present
        and clause.confidence < TIER3_CONFIDENCE_FLOOR
    ]


def _global_confidence_gate(ctx: RuleContext) -> list[RuleViolation]:
    return [
        _violation(
            "very_low_confidence",
            Severity.WARNING,
            f"{clause.clause_type} confidence {clause.confidence:.2f} is below "
            f"{GLOBAL_CONFIDENCE_FLOOR}",
            clause.clause_type,
        )
        for clause in ctx.clauses.values()
        if clause.present and clause.confidence < GLOBAL_CONFIDENCE_FLOOR
    ]


def _all_clauses_absent(ctx: RuleContext) -> list[RuleViolation]:
    """A contract with nothing found at all is a pipeline failure, not a finding."""
    if ctx.clauses and not any(clause.present for clause in ctx.clauses.values()):
        return [
            _violation(
                "nothing_extracted",
                Severity.ERROR,
                f"all {len(ctx.clauses)} clause types reported absent; "
                "retrieval or the agent loop probably failed",
            )
        ]
    return []


def _all_clauses_present(ctx: RuleContext) -> list[RuleViolation]:
    """Every clause present is implausible given the measured base rates.

    The rarest of the twelve appears in 21.8% of contracts; all twelve at once
    is the signature of a model answering "yes" to everything.
    """
    if len(ctx.clauses) >= 12 and all(clause.present for clause in ctx.clauses.values()):
        return [
            _violation(
                "everything_present",
                Severity.WARNING,
                "all 12 clause types reported present; base rates make this very unlikely",
            )
        ]
    return []


def _incomplete_coverage(ctx: RuleContext) -> list[RuleViolation]:
    missing = sorted(set(ClauseType) - set(ctx.clauses))
    if missing:
        return [
            _violation(
                "incomplete_clause_coverage",
                Severity.WARNING,
                f"{len(missing)} clause type(s) missing from the result: "
                f"{[str(m) for m in missing]}",
                *missing,
            )
        ]
    return []


# --------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------

RULES: Final[tuple[Rule, ...]] = (
    # --- field rules: internal consistency of one clause ---
    Rule(
        "presence_requires_evidence",
        Severity.ERROR,
        "A clause marked present must cite at least one evidence span.",
        _presence_requires_evidence,
    ),
    Rule(
        "absence_forbids_evidence",
        Severity.ERROR,
        "A clause marked absent must cite no evidence.",
        _absence_forbids_evidence,
    ),
    Rule(
        "absence_forbids_value",
        Severity.ERROR,
        "A clause marked absent must carry no normalized value.",
        _absence_forbids_value,
    ),
    Rule(
        "absence_forbids_raw_text",
        Severity.ERROR,
        "A clause marked absent must carry no raw text.",
        _absence_forbids_raw_text,
    ),
    Rule(
        "boolean_clause_carries_no_value",
        Severity.WARNING,
        "Presence-only clause types must not return a summary value.",
        _boolean_clauses_carry_no_value,
    ),
    Rule(
        "valued_clause_missing_value",
        Severity.WARNING,
        "A value-bearing clause marked present should return a value.",
        _valued_clause_has_value,
    ),
    # --- field rules: value formats ---
    Rule(
        "governing_law_unknown_jurisdiction",
        Severity.ERROR,
        "Governing law must resolve to the closed jurisdiction vocabulary.",
        _governing_law_is_known,
    ),
    Rule(
        "date_unparseable",
        Severity.ERROR,
        "Dates must be an ISO 8601 prefix or a documented sentinel.",
        _dates_parse,
    ),
    Rule(
        "date_implausible_year",
        Severity.ERROR,
        "Date years must fall in a plausible range.",
        _dates_are_plausible,
    ),
    Rule(
        "perpetual_effective_date",
        Severity.ERROR,
        "PERPETUAL is only meaningful for an expiration date.",
        _perpetual_only_on_expiration,
    ),
    Rule(
        "notice_period_format",
        Severity.ERROR,
        "Notice periods must normalize to a positive integer day count.",
        _notice_period_is_integer_days,
    ),
    # --- field rules: evidence quality ---
    Rule(
        "evidence_offsets_inverted",
        Severity.ERROR,
        "Evidence offsets must describe a non-empty forward span.",
        _evidence_offsets_are_ordered,
    ),
    Rule(
        "evidence_outside_document",
        Severity.ERROR,
        "Evidence offsets must fall inside the document.",
        _evidence_within_document,
    ),
    Rule(
        "evidence_too_short",
        Severity.WARNING,
        "An evidence quote must be long enough to identify a clause.",
        _evidence_is_substantive,
    ),
    Rule(
        "evidence_duplicated",
        Severity.WARNING,
        "A clause should not cite the same span twice.",
        _evidence_spans_do_not_duplicate,
    ),
    Rule(
        "raw_text_not_verbatim",
        Severity.ERROR,
        "Raw clause text must be a verbatim substring of the document.",
        _raw_text_is_not_paraphrased,
    ),
    # --- cross-field rules ---
    Rule(
        "uncapped_without_cap",
        Severity.ERROR,
        "Uncapped liability implies a general cap to carve out of.",
        _uncapped_implies_cap,
    ),
    Rule(
        "notice_period_without_renewal_term",
        Severity.WARNING,
        "A renewal notice period implies a renewal term.",
        _notice_period_implies_renewal,
    ),
    Rule(
        "renewal_term_without_notice_period",
        Severity.INFO,
        "A renewal term often but not always implies a notice period.",
        _renewal_implies_notice_period,
    ),
    Rule(
        "effective_after_expiration",
        Severity.ERROR,
        "Effective date must not follow expiration date.",
        _effective_before_expiration,
    ),
    Rule(
        "expiration_without_effective",
        Severity.INFO,
        "An expiration date usually comes with an effective date.",
        _expiration_implies_effective,
    ),
    Rule(
        "parties_absent",
        Severity.WARNING,
        "Nearly every contract names its parties.",
        _parties_are_present,
    ),
    Rule(
        "non_compete_without_exclusivity",
        Severity.INFO,
        "Related restraint clauses usually co-occur.",
        _non_compete_and_exclusivity_coherent,
    ),
    # --- confidence gates ---
    Rule(
        "tier3_low_confidence",
        Severity.WARNING,
        "Low confidence on a judgement-tier clause routes to review.",
        _tier3_confidence_gate,
    ),
    Rule(
        "very_low_confidence",
        Severity.WARNING,
        "Very low confidence on any clause routes to review.",
        _global_confidence_gate,
    ),
    Rule(
        "confidence_is_constant",
        Severity.WARNING,
        "Identical confidence across every clause means the field carries no signal.",
        _confidence_is_meaningful,
    ),
    # --- document-level sanity ---
    Rule(
        "nothing_extracted",
        Severity.ERROR,
        "Finding no clauses at all indicates a pipeline failure.",
        _all_clauses_absent,
    ),
    Rule(
        "everything_present",
        Severity.WARNING,
        "Finding every clause contradicts the measured base rates.",
        _all_clauses_present,
    ),
    Rule(
        "incomplete_clause_coverage",
        Severity.WARNING,
        "A result should cover all 12 clause types.",
        _incomplete_coverage,
    ),
    Rule(
        "duplicate_clause_types",
        Severity.ERROR,
        "A result must not report the same clause type twice.",
        _no_duplicate_clause_types,
    ),
)


def apply_rules(
    clauses: Sequence[ClauseExtraction],
    document: Document | None = None,
    jurisdictions: JurisdictionIndex | None = None,
    rules: Sequence[Rule] = RULES,
) -> list[RuleViolation]:
    """Run every rule and collect the violations, most severe first."""
    index = jurisdictions if jurisdictions is not None else load_jurisdictions()

    ctx = RuleContext(
        clauses={clause.clause_type: clause for clause in clauses},
        all_clauses=tuple(clauses),
        document=document,
        jurisdictions=index,
    )

    violations: list[RuleViolation] = []
    for rule in rules:
        violations.extend(rule.check(ctx))

    order = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
    return sorted(violations, key=lambda v: (order[v.severity], v.rule_id))


def needs_review(result: ExtractionResult) -> tuple[bool, list[str]]:
    """Whether a result must go to a human, and why.

    Returns the reasons rather than a bare bool, because a review queue that
    cannot say why an item is in it wastes the reviewer's first minute.
    """
    reasons: list[str] = []

    errors = [v for v in result.violations if v.severity is Severity.ERROR]
    if errors:
        reasons.append(f"{len(errors)} error-severity rule violation(s)")

    low_tier3 = [
        c
        for c in result.clauses
        if c.tier is Tier.REQUIRES_JUDGEMENT and c.present and c.confidence < TIER3_CONFIDENCE_FLOOR
    ]
    if low_tier3:
        reasons.append(
            f"{len(low_tier3)} Tier-3 clause(s) below confidence {TIER3_CONFIDENCE_FLOOR}"
        )

    if result.stopped_on_budget:
        reasons.append("the agent loop stopped on a budget limit rather than finishing")

    return bool(reasons), reasons
