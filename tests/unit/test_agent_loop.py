"""Unit tests for the hand-rolled tool-calling loop.

Driven by a scripted fake client, so every budget and failure path is tested
without an API key and without spending money. The fake mimics only what the
loop touches: `client.messages.create(...)` returning an object with
`content`, `stop_reason`, and `usage`.

What is being tested is not "does it call the API" but the four things a loop
gets wrong in production: it runs forever, it costs more than expected, it hangs,
or it accepts output that will not validate.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import pytest

from docintel.agent.loop import (
    MODEL_PRICING,
    AgentBudget,
    AgentRun,
    StopReason,
    compute_cost,
    load_prompt,
    prompt_version,
    run_extraction,
    summarize_tool_use,
)
from docintel.agent.tools import ToolContext
from docintel.schemas import ClauseType, Document, TokenUsage

CONTRACT = (
    "2. GOVERNING LAW. This Agreement is governed by the laws of the State of "
    "Delaware, without regard to conflict of laws principles."
)


# --------------------------------------------------------------------------
# Fake client
# --------------------------------------------------------------------------


@dataclass
class FakeUsage:
    input_tokens: int = 1_000
    output_tokens: int = 200
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class FakeToolUse:
    name: str
    input: dict[str, Any]
    id: str = "toolu_1"
    type: str = "tool_use"


@dataclass
class FakeText:
    text: str = "Working on it."
    type: str = "text"


@dataclass
class FakeResponse:
    content: list[Any]
    stop_reason: str = "tool_use"
    usage: FakeUsage = field(default_factory=FakeUsage)
    stop_details: Any = None


class FakeMessages:
    """Records each request, deep-copying `messages`.

    The loop mutates a single `messages` list in place, so recording the
    reference would make every stored call show the *final* conversation rather
    than what was sent at the time -- and every assertion about request shape
    silently meaningless.
    """

    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append({**kwargs, "messages": copy.deepcopy(kwargs["messages"])})
        if not self._script:
            # Ran off the end of the script: keep returning a no-tool response so
            # the loop terminates rather than hanging the test.
            return FakeResponse(content=[FakeText()], stop_reason="end_turn")
        step = self._script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


class FakeClient:
    def __init__(self, script: list[Any]) -> None:
        self.messages = FakeMessages(script)


def all_twelve_clauses(confidence: float = 0.9) -> list[dict[str, Any]]:
    """A submission covering all 12 clause types, varied confidence."""
    quote = "governed by the laws of the State of Delaware"
    start = CONTRACT.index(quote)
    payload: list[dict[str, Any]] = []
    for index, clause_type in enumerate(ClauseType):
        present = clause_type is ClauseType.GOVERNING_LAW
        payload.append(
            {
                "clause_type": clause_type.value,
                "present": present,
                "value": "US-DE" if present else None,
                "raw_text": quote if present else None,
                "evidence": (
                    [{"quote": quote, "char_start": start, "char_end": start + len(quote)}]
                    if present
                    else []
                ),
                # Vary it so the constant-confidence rule stays quiet.
                "confidence": round(confidence - index * 0.01, 2),
            }
        )
    return payload


def submit(clauses: list[dict[str, Any]] | None = None, call_id: str = "toolu_sub") -> FakeResponse:
    return FakeResponse(
        content=[
            FakeToolUse(
                name="submit_extraction",
                input={"clauses": clauses if clauses is not None else all_twelve_clauses()},
                id=call_id,
            )
        ]
    )


def context(search: Any = None) -> ToolContext:
    return ToolContext(document=Document(document_id="DOC", text=CONTRACT), search=search)


def fake_search(query: str, top_k: int) -> list[Any]:
    return []


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


class TestHappyPath:
    def test_submission_ends_the_loop(self) -> None:
        outcome = run_extraction(FakeClient([submit()]), context())
        assert outcome.stop_reason is StopReason.SUBMITTED
        assert outcome.submitted
        assert len(outcome.clauses) == 12

    def test_records_turns_used(self) -> None:
        client = FakeClient(
            [
                FakeResponse(content=[FakeToolUse("search_contract", {"query": "governing law"})]),
                submit(),
            ]
        )
        outcome = run_extraction(client, context(search=fake_search))
        assert outcome.turns == 2

    def test_accumulates_usage_across_turns(self) -> None:
        client = FakeClient(
            [
                FakeResponse(content=[FakeToolUse("get_schema", {})]),
                submit(),
            ]
        )
        outcome = run_extraction(client, context())
        assert outcome.usage.input_tokens == 2_000
        assert outcome.usage.output_tokens == 400
        assert outcome.usage.cost_usd > 0

    def test_records_the_prompt_version(self) -> None:
        outcome = run_extraction(FakeClient([submit()]), context())
        assert len(outcome.prompt_version) == 12

    def test_tool_calls_are_traced(self) -> None:
        client = FakeClient(
            [
                FakeResponse(content=[FakeToolUse("get_schema", {})]),
                submit(),
            ]
        )
        outcome = run_extraction(client, context())
        assert summarize_tool_use(outcome.tool_calls)["get_schema"] == 1

    def test_not_flagged_as_budget_stopped(self) -> None:
        outcome = run_extraction(FakeClient([submit()]), context())
        assert not outcome.stopped_on_budget


class TestParallelToolUse:
    def test_multiple_tool_uses_in_one_turn_all_get_results(self) -> None:
        """All results must go back in ONE user message; splitting them teaches
        the model to stop making parallel calls."""
        client = FakeClient(
            [
                FakeResponse(
                    content=[
                        FakeToolUse("get_schema", {}, id="a"),
                        FakeToolUse("read_span", {"char_start": 0, "char_end": 40}, id="b"),
                    ]
                ),
                submit(),
            ]
        )
        run_extraction(client, context())

        # The second create() call carries the tool results.
        messages = client.messages.calls[1]["messages"]
        user_results = [
            m for m in messages if m["role"] == "user" and isinstance(m["content"], list)
        ]
        results = user_results[-1]["content"]
        assert len(results) == 2
        assert {r["tool_use_id"] for r in results} == {"a", "b"}

    def test_results_are_in_a_single_message(self) -> None:
        client = FakeClient(
            [
                FakeResponse(
                    content=[
                        FakeToolUse("get_schema", {}, id="a"),
                        FakeToolUse("get_schema", {}, id="b"),
                    ]
                ),
                submit(),
            ]
        )
        run_extraction(client, context())
        messages = client.messages.calls[1]["messages"]
        tool_result_messages = [
            m
            for m in messages
            if m["role"] == "user"
            and isinstance(m["content"], list)
            and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in m["content"])
        ]
        assert len(tool_result_messages) == 1


class TestBudgets:
    def test_turn_limit_stops_the_loop(self) -> None:
        """A model that searches forever must be cut off."""
        forever = [
            FakeResponse(content=[FakeToolUse("search_contract", {"query": "x"})])
            for _ in range(50)
        ]
        outcome = run_extraction(
            FakeClient(forever), context(search=fake_search), AgentBudget(max_turns=4)
        )
        assert outcome.stop_reason is StopReason.MAX_TURNS
        assert outcome.turns == 4
        assert outcome.stopped_on_budget

    def test_cost_limit_stops_the_loop_independently_of_turns(self) -> None:
        """The reason a turn limit alone is not enough: ten turns over a 90k-token
        contract costs very differently from ten over 3k."""
        expensive = [
            FakeResponse(
                content=[FakeToolUse("get_schema", {})],
                usage=FakeUsage(input_tokens=1_000_000, output_tokens=0),
            )
            for _ in range(50)
        ]
        outcome = run_extraction(
            FakeClient(expensive),
            context(),
            AgentBudget(max_turns=100, max_cost_usd=3.0),
        )
        assert outcome.stop_reason is StopReason.MAX_COST
        assert outcome.turns < 100

    def test_timeout_stops_the_loop(self) -> None:
        outcome = run_extraction(
            FakeClient([FakeResponse(content=[FakeToolUse("get_schema", {})])] * 5),
            context(),
            AgentBudget(timeout_s=0.0),
        )
        assert outcome.stop_reason is StopReason.TIMEOUT

    def test_budget_stop_records_an_error(self) -> None:
        outcome = run_extraction(
            FakeClient([FakeResponse(content=[FakeToolUse("get_schema", {})])] * 5),
            context(),
            AgentBudget(max_turns=1),
        )
        assert outcome.error is not None
        assert "budget" in outcome.error

    def test_budget_stop_returns_no_clauses(self) -> None:
        outcome = run_extraction(
            FakeClient([FakeResponse(content=[FakeToolUse("get_schema", {})])] * 5),
            context(),
            AgentBudget(max_turns=1),
        )
        assert outcome.clauses == []


class TestAgentRunBudgetChecks:
    def test_fresh_run_has_not_exceeded_anything(self) -> None:
        assert AgentRun().exceeded(AgentBudget()) is None

    def test_reports_the_first_budget_blown(self) -> None:
        run = AgentRun(turns=99)
        assert run.exceeded(AgentBudget(max_turns=12)) is StopReason.MAX_TURNS

    def test_cost_is_checked_against_accumulated_usage(self) -> None:
        run = AgentRun(usage=TokenUsage(cost_usd=1.0))
        assert run.exceeded(AgentBudget(max_cost_usd=0.5)) is StopReason.MAX_COST

    def test_retries_are_a_stopping_condition(self) -> None:
        run = AgentRun(retries=3)
        assert run.exceeded(AgentBudget(max_retries=3)) is StopReason.MAX_RETRIES


class TestRetryOnParseFailure:
    def test_invalid_submission_is_rejected_and_retried(self) -> None:
        """The model gets the specific complaint, not a generic failure."""
        client = FakeClient(
            [
                submit([{"clause_type": "governing_law"}]),  # missing required fields
                submit(),
            ]
        )
        outcome = run_extraction(client, context())
        assert outcome.stop_reason is StopReason.SUBMITTED
        assert len(outcome.clauses) == 12

    def test_rejection_message_reaches_the_model(self) -> None:
        client = FakeClient([submit([{"clause_type": "governing_law"}]), submit()])
        run_extraction(client, context())
        messages = client.messages.calls[1]["messages"]
        results = [
            block
            for message in messages
            if isinstance(message["content"], list)
            for block in message["content"]
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        assert any(r.get("is_error") and "Rejected" in str(r["content"]) for r in results)

    def test_incomplete_submission_is_rejected(self) -> None:
        """Fewer than 12 clause types is not a valid answer."""
        partial = all_twelve_clauses()[:5]
        outcome = run_extraction(
            FakeClient([submit(partial)] * 10), context(), AgentBudget(max_retries=2)
        )
        # Rejected by validate_submission, the shared definition of a valid
        # submission. Before that existed the loop accepted this silently.
        assert outcome.stop_reason is StopReason.MAX_RETRIES
        assert outcome.clauses == []

    def test_repeated_parse_failures_stop_on_max_retries(self) -> None:
        bad = submit([{"clause_type": "governing_law"}])
        outcome = run_extraction(
            FakeClient([bad] * 20), context(), AgentBudget(max_retries=2, max_turns=50)
        )
        assert outcome.stop_reason is StopReason.MAX_RETRIES


class TestFailureModes:
    def test_api_error_is_captured_not_raised(self) -> None:
        outcome = run_extraction(FakeClient([RuntimeError("connection reset")]), context())
        assert outcome.stop_reason is StopReason.API_ERROR
        assert outcome.error is not None
        assert "connection reset" in outcome.error

    def test_prose_answer_without_submission_is_an_error(self) -> None:
        """Answering in text instead of calling submit_extraction is not valid."""
        outcome = run_extraction(
            FakeClient([FakeResponse(content=[FakeText()], stop_reason="end_turn")]), context()
        )
        assert outcome.stop_reason is StopReason.END_TURN
        assert outcome.error is not None
        assert "submit_extraction" in outcome.error

    def test_refusal_is_handled(self) -> None:
        @dataclass
        class Details:
            category: str = "cyber"

        outcome = run_extraction(
            FakeClient(
                [FakeResponse(content=[FakeText()], stop_reason="refusal", stop_details=Details())]
            ),
            context(),
        )
        assert outcome.stop_reason is StopReason.REFUSAL

    def test_tool_error_does_not_kill_the_run(self) -> None:
        """A bad argument should cost one turn, not the whole extraction."""
        client = FakeClient(
            [
                FakeResponse(content=[FakeToolUse("read_span", {"char_start": 99, "char_end": 5})]),
                submit(),
            ]
        )
        outcome = run_extraction(client, context())
        assert outcome.stop_reason is StopReason.SUBMITTED

    def test_unknown_tool_is_reported_to_the_model(self) -> None:
        client = FakeClient(
            [FakeResponse(content=[FakeToolUse("delete_everything", {})]), submit()]
        )
        outcome = run_extraction(client, context())
        assert outcome.stop_reason is StopReason.SUBMITTED


class TestRequestShape:
    def test_system_prompt_is_cached(self) -> None:
        """Caching the stable prefix is what makes turn 2 onward cheap."""
        client = FakeClient([submit()])
        run_extraction(client, context())
        system = client.messages.calls[0]["system"]
        assert system[0]["cache_control"] == {"type": "ephemeral"}

    def test_tools_are_sent_every_turn_and_are_identical(self) -> None:
        """A tool list that varies between calls invalidates the cached prefix."""
        client = FakeClient([FakeResponse(content=[FakeToolUse("get_schema", {})]), submit()])
        run_extraction(client, context())
        first, second = client.messages.calls[0]["tools"], client.messages.calls[1]["tools"]
        assert first == second

    def test_model_defaults_to_sonnet(self) -> None:
        """CLAUDE.md section 3 specifies Claude Sonnet for this project."""
        client = FakeClient([submit()])
        run_extraction(client, context())
        assert client.messages.calls[0]["model"] == "claude-sonnet-5"

    def test_assistant_content_is_echoed_back_whole(self) -> None:
        client = FakeClient([FakeResponse(content=[FakeToolUse("get_schema", {})]), submit()])
        run_extraction(client, context())
        messages = client.messages.calls[1]["messages"]
        assistant = [m for m in messages if m["role"] == "assistant"]
        assert assistant, "the assistant turn must be echoed back"


class TestCost:
    def test_sonnet_pricing_is_two_and_ten(self) -> None:
        assert MODEL_PRICING["claude-sonnet-5"][:2] == (2.00, 10.00)

    def test_cost_matches_the_published_rate(self) -> None:
        usage = FakeUsage(input_tokens=1_000_000, output_tokens=0)
        assert compute_cost("claude-sonnet-5", usage) == pytest.approx(2.00)

    def test_cached_reads_are_ten_times_cheaper(self) -> None:
        """A loop that counts raw input tokens overstates spend once caching
        works, because cache reads bill at ~0.1x."""
        cached = FakeUsage(input_tokens=0, output_tokens=0, cache_read_input_tokens=1_000_000)
        uncached = FakeUsage(input_tokens=1_000_000, output_tokens=0)
        assert compute_cost("claude-sonnet-5", cached) == pytest.approx(
            compute_cost("claude-sonnet-5", uncached) / 10
        )

    def test_unknown_model_costs_zero_rather_than_raising(self) -> None:
        assert compute_cost("claude-not-a-model", FakeUsage()) == 0.0

    def test_usage_addition_accumulates_every_field(self) -> None:
        total = TokenUsage(input_tokens=1, output_tokens=2, cost_usd=0.5) + TokenUsage(
            input_tokens=10, output_tokens=20, cost_usd=1.5
        )
        assert (total.input_tokens, total.output_tokens, total.cost_usd) == (11, 22, 2.0)


class TestPromptVersioning:
    def test_prompt_loads(self) -> None:
        assert "verbatim" in load_prompt("extract_v1")

    def test_missing_prompt_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_prompt("does_not_exist")

    def test_version_is_stable_for_the_same_input(self) -> None:
        assert prompt_version("abc", "tools") == prompt_version("abc", "tools")

    def test_prompt_change_changes_the_version(self) -> None:
        assert prompt_version("abc", "t") != prompt_version("abd", "t")

    def test_tool_change_also_changes_the_version(self) -> None:
        """A reworded tool description changes behaviour as much as a reworded
        prompt; hashing only the prompt would make that invisible."""
        assert prompt_version("abc", "tools_v1") != prompt_version("abc", "tools_v2")
