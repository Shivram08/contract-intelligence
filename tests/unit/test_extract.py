"""Unit tests for the extraction orchestrator.

The test that matters most is `TestGroundingRunsBeforeRules`. Grounding drops
fabricated evidence, which leaves `present=True` with an empty evidence list,
and that is what `presence_requires_evidence` rejects. Run the rules first and
a fabricating extraction sails through, because at that moment it still has
evidence attached. The ordering *is* the hallucination gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from docintel.agent.loop import AgentBudget, StopReason
from docintel.extract import Extractor, extract_document
from docintel.schemas import ClauseType, Document, Severity

CONTRACT = (
    "2. GOVERNING LAW. This Agreement is governed by the laws of the State of "
    "Delaware, without regard to conflict of laws principles. "
    "3. LIABILITY. Aggregate liability is capped at the fees paid."
)
QUOTE = "governed by the laws of the State of Delaware"


def document() -> Document:
    return Document(document_id="DOC", text=CONTRACT)


# --- minimal fake client, same shape as test_agent_loop's ---------------------


@dataclass
class FakeUsage:
    input_tokens: int = 500
    output_tokens: int = 100
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class FakeToolUse:
    name: str
    input: dict[str, Any]
    id: str = "toolu_1"
    type: str = "tool_use"


@dataclass
class FakeResponse:
    content: list[Any]
    stop_reason: str = "tool_use"
    usage: FakeUsage = field(default_factory=FakeUsage)
    stop_details: Any = None


class FakeClient:
    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.messages = self

    def create(self, **kwargs: Any) -> Any:
        if not self._script:
            return FakeResponse(content=[], stop_reason="end_turn")
        step = self._script.pop(0)
        if isinstance(step, Exception):
            # Raise, do not return. Returning it made the loop treat the
            # exception object as a response with no content, so the API-error
            # path was never actually exercised.
            raise step
        return step


def clauses_payload(
    *,
    governing_law_quote: str = QUOTE,
    governing_law_value: str | None = "US-DE",
    start: int | None = None,
) -> list[dict[str, Any]]:
    """All 12 clause types, only governing_law present."""
    if start is None:
        start = CONTRACT.find(governing_law_quote)
        if start < 0:
            start = 0
    payload: list[dict[str, Any]] = []
    for index, clause_type in enumerate(ClauseType):
        present = clause_type is ClauseType.GOVERNING_LAW
        payload.append(
            {
                "clause_type": clause_type.value,
                "present": present,
                "value": governing_law_value if present else None,
                "raw_text": governing_law_quote if present else None,
                "evidence": (
                    [
                        {
                            "quote": governing_law_quote,
                            "char_start": start,
                            "char_end": start + len(governing_law_quote),
                        }
                    ]
                    if present
                    else []
                ),
                "confidence": round(0.95 - index * 0.01, 2),
            }
        )
    return payload


def submit(payload: list[dict[str, Any]]) -> FakeResponse:
    return FakeResponse(content=[FakeToolUse("submit_extraction", {"clauses": payload}, id="s1")])


def run(payload: list[dict[str, Any]]) -> Any:
    return extract_document(FakeClient([submit(payload)]), document())


class TestHappyPath:
    def test_returns_all_twelve_clauses(self) -> None:
        assert len(run(clauses_payload()).result.clauses) == 12

    def test_records_the_document_id(self) -> None:
        assert run(clauses_payload()).result.document_id == "DOC"

    def test_records_per_stage_latency(self) -> None:
        latency = run(clauses_payload()).result.latency_ms
        assert set(latency) == {"agent", "validation", "total"}
        assert latency["total"] >= latency["validation"]

    def test_records_usage_and_prompt_version(self) -> None:
        result = run(clauses_payload()).result
        assert result.usage.input_tokens == 500
        assert len(result.prompt_version) == 12
        assert result.model == "claude-sonnet-5"

    def test_grounded_extraction_needs_no_review(self) -> None:
        outcome = run(clauses_payload())
        assert not outcome.result.needs_review
        assert outcome.review is None

    def test_grounding_report_is_returned(self) -> None:
        outcome = run(clauses_payload())
        assert outcome.grounding.total == 1
        assert outcome.grounding.violation_rate == 0.0

    def test_agent_outcome_is_returned(self) -> None:
        assert run(clauses_payload()).agent.stop_reason is StopReason.SUBMITTED


class TestGroundingRunsBeforeRules:
    """The ordering that makes the hallucination gate actually bite."""

    def test_fabricated_quote_is_dropped_from_evidence(self) -> None:
        outcome = run(clauses_payload(governing_law_quote="the laws of the State of Nevada"))
        clause = outcome.result.clause(ClauseType.GOVERNING_LAW)
        assert clause is not None
        assert clause.evidence == []

    def test_fabrication_produces_an_error_violation(self) -> None:
        """Grounding empties the evidence list; the rule turns that into a
        rejection. Neither step alone catches it."""
        outcome = run(clauses_payload(governing_law_quote="the laws of the State of Nevada"))
        assert "presence_requires_evidence" in {v.rule_id for v in outcome.result.errors}

    def test_fabrication_routes_to_review(self) -> None:
        outcome = run(clauses_payload(governing_law_quote="the laws of the State of Nevada"))
        assert outcome.result.needs_review
        assert outcome.review is not None

    def test_review_reason_names_the_ungrounded_span(self) -> None:
        outcome = run(clauses_payload(governing_law_quote="the laws of the State of Nevada"))
        assert outcome.review is not None
        assert any("ungrounded" in reason for reason in outcome.review.reasons)

    def test_grounding_violation_rate_is_reported(self) -> None:
        outcome = run(clauses_payload(governing_law_quote="wholly invented clause text"))
        assert outcome.grounding.violation_rate == 1.0


class TestOffsetRepair:
    def test_wrong_offsets_with_a_real_quote_are_corrected(self) -> None:
        """Not a hallucination -- the quote is real, the arithmetic was not."""
        outcome = run(clauses_payload(start=3))
        clause = outcome.result.clause(ClauseType.GOVERNING_LAW)
        assert clause is not None
        assert len(clause.evidence) == 1
        assert document().slice(clause.evidence[0].char_start, clause.evidence[0].char_end) == QUOTE

    def test_repaired_extraction_does_not_route_to_review(self) -> None:
        assert not run(clauses_payload(start=3)).result.needs_review

    def test_returned_clauses_are_the_repaired_ones(self) -> None:
        outcome = run(clauses_payload(start=3))
        clause = outcome.result.clause(ClauseType.GOVERNING_LAW)
        assert clause is not None
        assert clause.evidence[0].char_start == CONTRACT.index(QUOTE)


class TestRuleViolationsSurface:
    def test_unknown_jurisdiction_is_surfaced_as_an_error(self) -> None:
        outcome = run(clauses_payload(governing_law_value="Atlantis"))
        assert "governing_law_unknown_jurisdiction" in {v.rule_id for v in outcome.result.errors}

    def test_violations_are_attached_to_the_result(self) -> None:
        outcome = run(clauses_payload(governing_law_value="Atlantis"))
        assert outcome.result.violations
        assert any(v.severity is Severity.ERROR for v in outcome.result.violations)

    def test_error_violation_forces_review(self) -> None:
        assert run(clauses_payload(governing_law_value="Atlantis")).result.needs_review


class TestAgentFailurePropagates:
    def test_budget_stop_is_recorded_and_reviewed(self) -> None:
        forever = [FakeResponse(content=[FakeToolUse("get_schema", {})]) for _ in range(20)]
        outcome = extract_document(FakeClient(forever), document(), budget=AgentBudget(max_turns=2))
        assert outcome.result.stopped_on_budget
        assert outcome.result.needs_review
        assert outcome.review is not None
        assert any("budget" in reason for reason in outcome.review.reasons)

    def test_api_error_routes_to_review_with_the_message(self) -> None:
        outcome = extract_document(FakeClient([RuntimeError("boom")]), document())
        assert outcome.result.needs_review
        assert outcome.review is not None
        assert any("boom" in reason for reason in outcome.review.reasons)

    def test_no_submission_yields_no_clauses_but_a_review_item(self) -> None:
        outcome = extract_document(
            FakeClient([FakeResponse(content=[], stop_reason="end_turn")]), document()
        )
        assert outcome.result.clauses == []
        assert outcome.result.needs_review


class TestExtractorReuse:
    def test_extractor_shares_the_jurisdiction_index(self) -> None:
        """Loaded once rather than per document; a 150-case run would otherwise
        re-parse the YAML 150 times."""
        extractor = Extractor(client=FakeClient([submit(clauses_payload())]))
        first = extractor.jurisdictions
        extractor.extract(document())
        assert extractor.jurisdictions is first

    def test_extractor_produces_the_same_result_as_the_function(self) -> None:
        extractor = Extractor(client=FakeClient([submit(clauses_payload())]))
        outcome = extractor.extract(document())
        assert len(outcome.result.clauses) == 12
        assert not outcome.result.needs_review

    def test_search_is_passed_through_to_the_tools(self) -> None:
        seen: list[str] = []

        def search(query: str, top_k: int) -> list[Any]:
            seen.append(query)
            return []

        client = FakeClient(
            [
                FakeResponse(content=[FakeToolUse("search_contract", {"query": "governing law"})]),
                submit(clauses_payload()),
            ]
        )
        extract_document(client, document(), search=search)
        assert seen == ["governing law"]
