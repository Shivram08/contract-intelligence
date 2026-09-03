"""Metrics for the golden-set evaluation.

Pure functions over gold labels and predictions. No network, no model, no I/O,
so every number here is unit-testable and the eval harness cannot quietly change
what a metric means.

Three decisions worth stating, because each was a choice and not a default:

**Presence is scored any-span-hit.** A hit needs ``present=True`` plus at least
one returned evidence span overlapping *any* gold span. Finding 1 of 3 gold spans
is a hit, not 0.33. CUAD's multiple spans are usually one obligation restated
across a definitions section, an operative clause, and a schedule, so treating
them as independent targets would penalise a correct extraction for not being
exhaustive. The cost of that choice is that it cannot tell "found the clause"
from "found every mention", which is why span token F1 is reported beside it and
is the metric that penalises partial coverage. Recorded in
``docs/DATA_AUDIT.md`` check 5 before any results were seen.

**Every proportion carries a Wilson interval.** At the affordable golden-set
size the rarest clause type has a handful of positives, and a bare per-clause F1
on four positives is precision theatre. Wilson rather than the normal
approximation because the normal interval is badly wrong at small n and at
proportions near 0 or 1 -- exactly this regime.

**Base rate travels with every F1.** ``CLAUDE.md`` section 2.2 requires it, and
the reason is concrete: ``parties`` appears in 99.8% of contracts, so an
always-yes classifier scores 0.999 presence F1 there. An F1 without its base
rate beside it is not interpretable.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from docintel.schemas import (
    CLAUSE_TIERS,
    ClauseExtraction,
    ClauseType,
    Tier,
)
from evals.cases import GoldenCase

__all__ = [
    "ClassificationCounts",
    "Interval",
    "MetricSummary",
    "PresenceMetrics",
    "SpanMetrics",
    "precision_at_recall",
    "score_cases",
    "span_token_f1",
    "wilson_interval",
]


# --------------------------------------------------------------------------
# Intervals
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Interval:
    """A confidence interval on a proportion."""

    low: float
    high: float

    @property
    def width(self) -> float:
        return self.high - self.low

    def __str__(self) -> str:
        return f"[{self.low:.2f}, {self.high:.2f}]"


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> Interval:
    """Wilson score interval for a binomial proportion.

    Used instead of the normal approximation because the latter is badly wrong
    at small ``trials`` and at proportions near 0 or 1 -- it can produce bounds
    below 0 or above 1, and it is over-confident exactly where this evaluation
    operates. Wilson stays inside [0, 1] and behaves at n=4.

    ``trials == 0`` yields the whole interval: no evidence, no claim.
    """
    if trials <= 0:
        return Interval(0.0, 1.0)
    phat = successes / trials
    denominator = 1 + z**2 / trials
    centre = (phat + z**2 / (2 * trials)) / denominator
    margin = z * math.sqrt(phat * (1 - phat) / trials + z**2 / (4 * trials**2)) / denominator
    return Interval(max(0.0, centre - margin), min(1.0, centre + margin))


# --------------------------------------------------------------------------
# Presence classification
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ClassificationCounts:
    """Confusion-matrix counts for a binary presence decision."""

    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    def observe(self, predicted: bool, actual: bool) -> None:
        if predicted and actual:
            self.true_positives += 1
        elif predicted and not actual:
            self.false_positives += 1
        elif not predicted and actual:
            self.false_negatives += 1
        else:
            self.true_negatives += 1

    def __add__(self, other: ClassificationCounts) -> ClassificationCounts:
        return ClassificationCounts(
            self.true_positives + other.true_positives,
            self.false_positives + other.false_positives,
            self.true_negatives + other.true_negatives,
            self.false_negatives + other.false_negatives,
        )

    @property
    def total(self) -> int:
        return (
            self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        )

    @property
    def gold_positives(self) -> int:
        return self.true_positives + self.false_negatives

    @property
    def base_rate(self) -> float:
        """Share of cases that are gold-positive.

        Reported beside every F1. `parties` at 99.8% makes an always-yes
        classifier look excellent; without this number that is invisible.
        """
        return self.gold_positives / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        if not (self.precision + self.recall):
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)

    @property
    def accuracy(self) -> float:
        return (self.true_positives + self.true_negatives) / self.total if self.total else 0.0

    @property
    def trivial_f1(self) -> float:
        """F1 of an always-yes classifier: precision = base rate, recall = 1.

        The number an F1 has to beat to mean anything on a high-prevalence
        clause.
        """
        rate = self.base_rate
        return 2 * rate / (1 + rate) if rate else 0.0

    @property
    def precision_interval(self) -> Interval:
        return wilson_interval(self.true_positives, self.true_positives + self.false_positives)

    @property
    def recall_interval(self) -> Interval:
        return wilson_interval(self.true_positives, self.gold_positives)

    @property
    def has_enough_positives(self) -> bool:
        """Whether per-clause figures are worth quoting at all.

        Ten is a judgement call, not a theorem: below it the Wilson interval on
        recall is wider than roughly +/-0.3, which is wider than any difference
        between baselines worth reporting.
        """
        return self.gold_positives >= 10


# --------------------------------------------------------------------------
# Span overlap
# --------------------------------------------------------------------------


def _merge(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping spans, so a union is counted once."""
    ordered = sorted(span for span in spans if span[1] > span[0])
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _covered(spans: Sequence[tuple[int, int]]) -> int:
    return sum(end - start for start, end in _merge(spans))


def _intersection(left: Sequence[tuple[int, int]], right: Sequence[tuple[int, int]]) -> int:
    """Total characters covered by both span sets."""
    total = 0
    for a_start, a_end in _merge(left):
        for b_start, b_end in _merge(right):
            overlap = min(a_end, b_end) - max(a_start, b_start)
            if overlap > 0:
                total += overlap
    return total


def span_token_f1(predicted: Sequence[tuple[int, int]], gold: Sequence[tuple[int, int]]) -> float:
    """Character-level F1 between predicted and gold span sets.

    Scored against the **union** of gold spans, which is the counterpart to the
    any-span-hit presence rule: presence forgives finding only one of several
    mentions, and this is the metric that does not. Together they distinguish
    "found the clause" from "found all of it".

    Characters rather than word tokens, because the gold offsets are character
    offsets and tokenising would introduce a boundary convention that has to be
    defended. Both are monotonic in overlap; characters need no extra argument.
    """
    if not gold and not predicted:
        return 1.0
    if not gold or not predicted:
        return 0.0

    overlap = _intersection(predicted, gold)
    if not overlap:
        return 0.0
    precision = overlap / _covered(predicted)
    recall = overlap / _covered(gold)
    return 2 * precision * recall / (precision + recall)


def spans_overlap(predicted: Sequence[tuple[int, int]], gold: Sequence[tuple[int, int]]) -> bool:
    """Whether any predicted span touches any gold span. The any-span-hit rule."""
    return _intersection(predicted, gold) > 0


# --------------------------------------------------------------------------
# Precision at recall
# --------------------------------------------------------------------------


def precision_at_recall(
    scored: Sequence[tuple[float, bool]], target_recall: float = 0.80
) -> tuple[float, float] | None:
    """Highest precision achievable at or above ``target_recall``.

    CUAD's own reporting convention, which is why it is here -- it makes these
    numbers comparable to the published figures.

    ``scored`` is ``(confidence, is_gold_positive)`` for every case the system
    called present or could have. Sweeping the confidence threshold downward
    produces a precision-recall curve; this returns the best precision at a
    point on it that reaches ``target_recall``, with the threshold that achieved
    it. Returns None when the target recall is unreachable at any threshold,
    which is itself a finding and must not be reported as precision 0.
    """
    positives = sum(1 for _, actual in scored if actual)
    if not positives:
        return None

    ordered = sorted(scored, key=lambda pair: -pair[0])
    true_positives = 0
    false_positives = 0
    best: tuple[float, float] | None = None

    for index, (confidence, actual) in enumerate(ordered):
        if actual:
            true_positives += 1
        else:
            false_positives += 1
        recall = true_positives / positives
        precision = true_positives / (index + 1)
        if recall >= target_recall and (best is None or precision > best[0]):
            best = (precision, confidence)
    return best


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


@dataclass(slots=True)
class PresenceMetrics:
    """Presence classification, sliced the ways CLAUDE.md section 6 requires."""

    overall: ClassificationCounts = field(default_factory=ClassificationCounts)
    by_clause: dict[ClauseType, ClassificationCounts] = field(default_factory=dict)
    by_tier: dict[Tier, ClassificationCounts] = field(default_factory=dict)
    by_quartile: dict[int, ClassificationCounts] = field(default_factory=dict)


@dataclass(slots=True)
class SpanMetrics:
    """Span overlap, averaged over gold-positive cases only.

    Averaging over negatives too would reward a system for correctly returning
    nothing, which presence F1 already measures. This is about extraction
    precision where there is something to extract.
    """

    scores: list[float] = field(default_factory=list)
    by_tier: dict[Tier, list[float]] = field(default_factory=dict)

    @property
    def mean(self) -> float:
        return statistics.mean(self.scores) if self.scores else 0.0

    def tier_mean(self, tier: Tier) -> float:
        values = self.by_tier.get(tier, [])
        return statistics.mean(values) if values else 0.0


@dataclass(slots=True)
class MetricSummary:
    """Everything one baseline's run produced."""

    baseline: str
    presence: PresenceMetrics = field(default_factory=PresenceMetrics)
    spans: SpanMetrics = field(default_factory=SpanMetrics)
    #: (confidence, is_gold_positive) for the precision-at-recall sweep.
    confidence_pairs: list[tuple[float, bool]] = field(default_factory=list)
    #: Evidence spans checked, and how many were not grounded.
    grounding_checked: int = 0
    grounding_violations: int = 0
    #: Documents whose first submission parsed into Pydantic without a retry.
    schema_first_try: int = 0
    schema_attempts: int = 0
    #: Documents tripping at least one ERROR-severity rule.
    documents_with_errors: int = 0
    documents: int = 0
    #: Runs attempted and runs that finished. Accuracy metrics are computed
    #: over the completed ones only -- scoring an incomplete run as all-absent
    #: measures the turn ceiling, not the model.
    attempted: int = 0
    completed: int = 0
    #: document_id -> terminal state, for the runs that were excluded.
    excluded: dict[str, str] = field(default_factory=dict)
    #: Documents scored, after intersecting completions across every arm.
    paired_documents: int = 0
    #: Evidence spans whose offsets were wrong but whose quote was real. Counted
    #: so the grounding recovery path cannot become a silent escape hatch.
    relocated: int = 0
    #: Submissions rejected before one was accepted, summed over scored runs.
    retries_used: int = 0
    #: Per-document cost and latency, for mean and p95.
    costs_usd: list[float] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    #: Per-stage, so section 6's "latency per stage" is answerable.
    retrieval_ms: list[float] = field(default_factory=list)
    validation_ms: list[float] = field(default_factory=list)
    cases_scored: int = 0

    @property
    def completion_rate(self) -> float:
        """Share of attempted runs that finished and can be scored.

        Reported beside F1, with its denominator, because an arm that completes
        60% of the time and scores well on those is not comparable to one that
        completes always -- and averaging the failures in as zeros hides which
        of the two you have.
        """
        return self.completed / self.attempted if self.attempted else 0.0

    @property
    def grounding_violation_rate(self) -> float:
        """Share of evidence spans not found in the source.

        Per span, not per document: one fabricated quote among ten real ones is
        10%, not a wholly failed document.
        """
        return self.grounding_violations / self.grounding_checked if self.grounding_checked else 0.0

    @property
    def schema_validity_rate(self) -> float:
        return self.schema_first_try / self.schema_attempts if self.schema_attempts else 0.0

    @property
    def rule_violation_rate(self) -> float:
        return self.documents_with_errors / self.documents if self.documents else 0.0

    @property
    def mean_cost_usd(self) -> float:
        return statistics.mean(self.costs_usd) if self.costs_usd else 0.0

    @property
    def total_cost_usd(self) -> float:
        return sum(self.costs_usd)

    def percentile(self, values: Sequence[float], q: float) -> float:
        """Nearest-rank percentile. No interpolation at these sample sizes."""
        if not values:
            return 0.0
        ordered = sorted(values)
        rank = max(1, min(len(ordered), math.ceil(q / 100 * len(ordered))))
        return ordered[rank - 1]

    @property
    def p95_cost_usd(self) -> float:
        return self.percentile(self.costs_usd, 95)

    @property
    def p50_latency_ms(self) -> float:
        return self.percentile(self.latencies_ms, 50)

    @property
    def p95_latency_ms(self) -> float:
        return self.percentile(self.latencies_ms, 95)

    def to_json(self) -> dict[str, Any]:
        """Frozen-baseline form, for evals/baselines/ and the CI gate."""
        overall = self.presence.overall
        return {
            "baseline": self.baseline,
            "documents": self.documents,
            "cases_scored": self.cases_scored,
            "completion": {
                "rate": round(self.completion_rate, 4),
                "completed": self.completed,
                "attempted": self.attempted,
                "paired_documents": self.paired_documents,
                "excluded": self.excluded,
            },
            "retries_used": self.retries_used,
            "presence": {
                "f1": round(overall.f1, 4),
                "precision": round(overall.precision, 4),
                "recall": round(overall.recall, 4),
                "base_rate": round(overall.base_rate, 4),
                "trivial_f1": round(overall.trivial_f1, 4),
                "counts": {
                    "tp": overall.true_positives,
                    "fp": overall.false_positives,
                    "tn": overall.true_negatives,
                    "fn": overall.false_negatives,
                },
                "by_tier": {
                    str(int(tier)): {
                        "f1": round(counts.f1, 4),
                        "base_rate": round(counts.base_rate, 4),
                        "n": counts.total,
                    }
                    for tier, counts in sorted(self.presence.by_tier.items())
                },
                "by_clause_type": {
                    clause.value: {
                        "f1": round(counts.f1, 4),
                        "base_rate": round(counts.base_rate, 4),
                        "n": counts.total,
                        "gold_positives": counts.gold_positives,
                        "recall_interval": [
                            round(counts.recall_interval.low, 3),
                            round(counts.recall_interval.high, 3),
                        ],
                        "interpretable": counts.has_enough_positives,
                    }
                    for clause, counts in sorted(
                        self.presence.by_clause.items(), key=lambda kv: kv[0].value
                    )
                },
            },
            "span_token_f1": {
                "mean": round(self.spans.mean, 4),
                "by_tier": {
                    str(int(tier)): round(self.spans.tier_mean(tier), 4)
                    for tier in sorted(self.spans.by_tier)
                },
            },
            "grounding": {
                "violation_rate": round(self.grounding_violation_rate, 4),
                "spans_checked": self.grounding_checked,
                "violations": self.grounding_violations,
                "relocated": self.relocated,
            },
            "schema_validity_rate": round(self.schema_validity_rate, 4),
            "rule_violation_rate": round(self.rule_violation_rate, 4),
            "cost_usd": {
                "mean": round(self.mean_cost_usd, 4),
                "p95": round(self.p95_cost_usd, 4),
                "total": round(self.total_cost_usd, 4),
            },
            "latency_ms": {
                "p50": round(self.p50_latency_ms, 1),
                "p95": round(self.p95_latency_ms, 1),
                "retrieval_p50": round(self.percentile(self.retrieval_ms, 50), 1),
                "validation_p50": round(self.percentile(self.validation_ms, 50), 1),
            },
        }


def score_cases(
    baseline: str,
    cases: Sequence[GoldenCase],
    predictions: dict[str, list[ClauseExtraction]],
    scoreable: set[str] | None = None,
) -> MetricSummary:
    """Score one baseline's predictions against the gold cases.

    ``predictions`` maps ``document_id`` to that document's clause extractions.
    ``scoreable`` names the documents whose run actually completed; cases from
    any other document are **excluded**, not scored.

    That exclusion is the point. An earlier version scored a missing document as
    all-absent, reasoning that a crashed run should surface as recall loss. On
    real data that was wrong: three of five smoke-test contracts hit the turn
    ceiling and returned nothing, and scoring them as zeros would have reported
    a presence F1 that was mostly a measurement of ``max_turns``. Incomplete
    runs belong in the completion rate, not in the accuracy numerator.
    """
    summary = MetricSummary(baseline=baseline)
    summary.documents = len({case.document_id for case in cases})

    for case in cases:
        if scoreable is not None and case.document_id not in scoreable:
            continue
        by_type = {clause.clause_type: clause for clause in predictions.get(case.document_id, [])}
        predicted = by_type.get(case.clause_type)

        predicted_present = bool(predicted and predicted.present)
        predicted_spans = (
            [(e.char_start, e.char_end) for e in predicted.evidence] if predicted else []
        )

        # Presence needs both the flag and an overlapping span: claiming a clause
        # is present while citing the wrong part of the contract is not a hit.
        hit = predicted_present and (
            spans_overlap(predicted_spans, case.gold_spans) if case.gold_spans else True
        )
        effective = hit if case.present else predicted_present

        tier = CLAUSE_TIERS[case.clause_type]
        summary.presence.overall.observe(effective, case.present)
        summary.presence.by_clause.setdefault(case.clause_type, ClassificationCounts()).observe(
            effective, case.present
        )
        summary.presence.by_tier.setdefault(tier, ClassificationCounts()).observe(
            effective, case.present
        )
        summary.presence.by_quartile.setdefault(
            case.length_quartile, ClassificationCounts()
        ).observe(effective, case.present)

        if predicted is not None:
            summary.confidence_pairs.append((predicted.confidence, case.present))

        if case.present:
            score = span_token_f1(predicted_spans, list(case.gold_spans))
            summary.spans.scores.append(score)
            summary.spans.by_tier.setdefault(tier, []).append(score)

        summary.cases_scored += 1

    return summary
