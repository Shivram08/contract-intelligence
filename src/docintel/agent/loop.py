"""The tool-calling loop, written by hand.

CLAUDE.md section 3 is explicit that this is not a framework wrapper, and the
reason holds up: the interesting parts of an agent loop are the parts a
quickstart hides. Everything below that matters is a budget or a failure mode.

**Four independent stopping conditions**, because each catches a different way a
run goes wrong:

- ``max_turns`` -- the model is looping, searching the same thing repeatedly.
- ``max_cost_usd`` -- the run is expensive regardless of turn count, usually
  because the document is long and every turn resends it.
- ``timeout_s`` -- wall clock, which is what a caller with an SLA cares about.
- ``max_retries`` -- the model keeps submitting output that will not validate.

A loop with only a turn limit looks safe and is not: ten turns over a
90k-token contract is a different amount of money from ten turns over 3k.

**Cost is computed, not estimated.** ``response.usage`` is authoritative, and
cache reads are priced separately -- at a tenth of input rate -- so a loop that
counts raw input tokens overstates spend by an order of magnitude once caching
is working.

**Retry is targeted.** A submission that fails Pydantic validation gets sent
back with the specific complaint, not a generic "try again". Parse failures are
counted separately from tool errors, because a model that cannot produce valid
JSON after three tries has a prompt problem, not a transient one.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from docintel.agent.tools import (
    ToolContext,
    accept_submission,
    build_tool_schemas,
    execute_tool,
)
from docintel.schemas import ClauseExtraction, ClauseType, TokenUsage
from docintel.telemetry import stage_span

__all__ = [
    "MODEL_PRICING",
    "AgentBudget",
    "AgentOutcome",
    "AgentRun",
    "StopReason",
    "compute_cost",
    "load_prompt",
    "prompt_version",
    "run_extraction",
]

PROMPTS_DIR: Final = Path(__file__).resolve().parent / "prompts"

#: CLAUDE.md section 3 specifies Claude Sonnet for this project, so the default
#: is Sonnet 5 rather than the Opus default. The choice is a cost one: section
#: 2.6 sets a hard nightly budget, and Sonnet is 2.5x cheaper per token than
#: Opus on both input and output.
DEFAULT_MODEL: Final = "claude-sonnet-5"

#: USD per million tokens: (input, output, cache_read, cache_write).
#: Cache reads are ~0.1x input and writes ~1.25x, per the pricing docs.
MODEL_PRICING: Final[dict[str, tuple[float, float, float, float]]] = {
    "claude-sonnet-5": (2.00, 10.00, 0.20, 2.50),
    "claude-opus-5": (5.00, 25.00, 0.50, 6.25),
    "claude-haiku-4-5": (1.00, 5.00, 0.10, 1.25),
}


def compute_cost(model: str, usage: Any) -> float:
    """Dollar cost of one API response, from its reported usage.

    Unknown models return 0.0 rather than raising: a missing price should not
    abort an extraction, and a zero in the results table is visibly wrong in a
    way a silently plausible estimate would not be.
    """
    prices = MODEL_PRICING.get(model)
    if prices is None:
        return 0.0
    input_price, output_price, cache_read_price, cache_write_price = prices
    return (
        getattr(usage, "input_tokens", 0) * input_price
        + getattr(usage, "output_tokens", 0) * output_price
        + (getattr(usage, "cache_read_input_tokens", 0) or 0) * cache_read_price
        + (getattr(usage, "cache_creation_input_tokens", 0) or 0) * cache_write_price
    ) / 1_000_000


def load_prompt(name: str = "extract_v1") -> str:
    """Read a versioned prompt from ``agent/prompts/``."""
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


def prompt_version(prompt: str, tools_json: str = "") -> str:
    """Short content hash of the prompt and tool definitions.

    Recorded on every result so a regression can be attributed to a prompt
    change rather than a code change. The tool schemas are included because a
    reworded tool description changes behaviour just as much as a reworded
    system prompt, and only hashing the prompt would make that invisible.
    """
    digest = hashlib.sha256()
    digest.update(prompt.encode("utf-8"))
    digest.update(tools_json.encode("utf-8"))
    return digest.hexdigest()[:12]


class StopReason(StrEnum):
    """Why the loop ended."""

    SUBMITTED = "submitted"
    END_TURN = "end_turn"
    MAX_TURNS = "max_turns"
    MAX_COST = "max_cost"
    TIMEOUT = "timeout"
    MAX_RETRIES = "max_retries"
    API_ERROR = "api_error"
    REFUSAL = "refusal"


@dataclass(frozen=True, slots=True)
class AgentBudget:
    """Hard limits on one extraction.

    Defaults are sized from the corpus profile: the median contract is 6,751
    tokens, so a typical run resends roughly that plus retrieved passages each
    turn. Twelve turns at that size lands near $0.05 per document on Sonnet.
    """

    max_turns: int = 12
    #: Measured on 10 dev contracts: mean $0.23, max $0.40. $0.50 was too tight
    #: -- it cut off a run mid-extraction before the agent could submit.
    max_cost_usd: float = 1.50
    #: Also measured: median 73s, but one 2,958-token contract took 190s and was
    #: killed by the previous 180s default. It was looping, not large, so the
    #: budget was doing its job -- but losing 1 document in 10 to a default is
    #: too aggressive when the turn limit already bounds the loop.
    timeout_s: float = 300.0
    #: Retries for a submission that fails validation, not for API errors --
    #: the SDK already retries those.
    max_retries: int = 3
    max_tokens_per_turn: int = 8_000


@dataclass(slots=True)
class AgentRun:
    """Accumulated state of one loop. Mutable by design."""

    turns: int = 0
    retries: int = 0
    usage: TokenUsage = field(default_factory=TokenUsage)
    #: One entry per turn. Without this, a run that costs 20x the estimate is a
    #: single aggregate number with no way to see which turn or which token
    #: class caused it.
    turn_usage: list[TokenUsage] = field(default_factory=list)
    #: Model time per turn, taken from capture when a response is replayed.
    #: None entries mean "unavailable" -- a replayed entry recorded before
    #: latency capture existed. Never silently zero.
    turn_model_ms: list[float | None] = field(default_factory=list)
    started_at: float = field(default_factory=time.monotonic)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    def exceeded(self, budget: AgentBudget) -> StopReason | None:
        """Which budget, if any, this run has blown."""
        if self.turns >= budget.max_turns:
            return StopReason.MAX_TURNS
        if self.usage.cost_usd >= budget.max_cost_usd:
            return StopReason.MAX_COST
        if self.elapsed_s >= budget.timeout_s:
            return StopReason.TIMEOUT
        if self.retries >= budget.max_retries:
            return StopReason.MAX_RETRIES
        return None


@dataclass(frozen=True, slots=True)
class AgentOutcome:
    """What one extraction produced, and how it ended."""

    clauses: list[ClauseExtraction]
    stop_reason: StopReason
    turns: int
    #: Submissions rejected before one was accepted. Zero means the first
    #: attempt parsed, which is the schema validity rate in CLAUDE.md section 6.
    retries: int
    usage: TokenUsage
    latency_ms: float
    model: str
    prompt_version: str
    tool_calls: list[dict[str, Any]]
    #: Sum of per-turn model time, or None if any turn's latency was
    #: unavailable. None propagates deliberately: a partial sum would look like
    #: a fast run rather than a missing measurement.
    model_ms: float | None = None
    #: Per-turn usage, for diagnosing where the cost went.
    turn_usage: list[TokenUsage] = field(default_factory=list)
    #: Model time per turn, taken from capture when a response is replayed.
    #: None entries mean "unavailable" -- a replayed entry recorded before
    #: latency capture existed. Never silently zero.
    turn_model_ms: list[float | None] = field(default_factory=list)
    #: Set when the loop ended without a valid submission.
    error: str | None = None

    @property
    def submitted(self) -> bool:
        return self.stop_reason == StopReason.SUBMITTED

    @property
    def stopped_on_budget(self) -> bool:
        return self.stop_reason in {
            StopReason.MAX_TURNS,
            StopReason.MAX_COST,
            StopReason.TIMEOUT,
            StopReason.MAX_RETRIES,
        }


def _usage_from_response(model: str, response: Any) -> TokenUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        cost_usd=compute_cost(model, usage),
    )


def _user_prompt(document_id: str, clause_count: int) -> str:
    return (
        f"Extract all {clause_count} clause types from the contract "
        f"`{document_id}`.\n\n"
        "Search for each clause type before concluding it is absent. Call "
        "validate_extraction on your draft, fix anything it reports, then call "
        "submit_extraction once."
    )


def run_extraction(
    client: Any,
    ctx: ToolContext,
    budget: AgentBudget | None = None,
    model: str = DEFAULT_MODEL,
    prompt_name: str = "extract_v1",
) -> AgentOutcome:
    """Run the tool-calling loop until the agent submits or a budget stops it.

    ``client`` is duck-typed rather than annotated as ``anthropic.Anthropic`` so
    the loop can be driven by a scripted fake in tests. Everything it needs is
    ``client.messages.create(...)``.
    """
    budget = budget or AgentBudget()
    system = load_prompt(prompt_name)
    tools = build_tool_schemas()
    version = prompt_version(system, str(sorted(t["name"] for t in tools)))
    run = AgentRun()

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": _user_prompt(ctx.document.document_id, len(ClauseType))}
    ]

    clauses: list[ClauseExtraction] = []

    def finish(reason: StopReason, err: str | None = None) -> AgentOutcome:
        return AgentOutcome(
            clauses=clauses,
            stop_reason=reason,
            turns=run.turns,
            retries=run.retries,
            model_ms=(
                None
                if any(v is None for v in run.turn_model_ms)
                else sum(v for v in run.turn_model_ms if v is not None)
            ),
            usage=run.usage,
            latency_ms=run.elapsed_s * 1000,
            model=model,
            prompt_version=version,
            tool_calls=ctx.calls,
            turn_usage=list(run.turn_usage),
            error=err,
        )

    while True:
        blown = run.exceeded(budget)
        if blown is not None:
            return finish(blown, f"budget exhausted: {blown}")

        try:
            call_started = time.perf_counter()
            # One span per model call. This is where 97%+ of wall clock goes, so
            # a trace without it shows almost nothing.
            with stage_span("model", **{"docintel.turn": run.turns + 1}):
                response = client.messages.create(
                    model=model,
                    max_tokens=budget.max_tokens_per_turn,
                    # Auto-caches the last cacheable block, which is the end of the
                    # growing conversation. Without this, every turn re-prices the
                    # whole history at full input rate and total input cost is
                    # quadratic in turn count -- the system breakpoint below only
                    # covers the fixed prefix.
                    cache_control={"type": "ephemeral"},
                    system=[
                        {
                            "type": "text",
                            "text": system,
                            # The system prompt and tool list are the stable
                            # prefix; caching them makes turn 2 onward ~10x
                            # cheaper on input.
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    tools=tools,
                    messages=messages,
                )
            call_elapsed_ms = (time.perf_counter() - call_started) * 1000
        except Exception as exc:
            # The SDK already retried transient failures per max_retries, so
            # reaching here means it is not going to succeed.
            return finish(StopReason.API_ERROR, f"{type(exc).__name__}: {exc}")

        run.turns += 1
        ctx.turn = run.turns
        # Latency provenance, and the fallback is deliberately narrow.
        #
        # A replayed response always carries the `capture_latency_ms` attribute,
        # even when its value is None (an entry recorded before latency capture
        # existed). Presence of the attribute is therefore what identifies a
        # replay -- not the value. Falling back to the wall clock whenever the
        # value was None re-introduced the exact bug this fixes: cache-read time
        # reported as model latency, 1.2s standing in for 44s.
        #
        # Live call  -> wall clock, which is the real thing.
        # Replay     -> stored value, or None meaning unavailable. Never a
        #               substitute measured now.
        if hasattr(response, "capture_latency_ms"):
            run.turn_model_ms.append(response.capture_latency_ms)
        else:
            run.turn_model_ms.append(call_elapsed_ms)
        turn = _usage_from_response(model, response)
        run.turn_usage.append(turn)
        run.usage = run.usage + turn

        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            return finish(
                StopReason.REFUSAL,
                f"model refused: {getattr(details, 'category', 'unknown')}",
            )

        content = list(getattr(response, "content", []) or [])
        tool_uses = [block for block in content if getattr(block, "type", None) == "tool_use"]

        if not tool_uses:
            # No tools requested. Either the agent is finished (and should have
            # submitted) or it answered in prose, which is not a valid answer.
            return finish(
                StopReason.END_TURN,
                None if clauses else "ended without calling submit_extraction",
            )

        messages.append({"role": "assistant", "content": content})

        # All tool results go back in ONE user message. Splitting them across
        # several silently teaches the model to stop making parallel calls.
        results: list[dict[str, Any]] = []
        submitted_this_turn = False

        for block in tool_uses:
            name = getattr(block, "name", "")
            args = getattr(block, "input", {}) or {}

            if name == "submit_extraction":
                try:
                    clauses = accept_submission(ctx, args)
                    submitted_this_turn = True
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Accepted {len(clauses)} clause extractions.",
                        }
                    )
                except Exception as exc:
                    # Retry-on-parse-failure: hand back the specific complaint.
                    run.retries += 1
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": (
                                f"Rejected: {exc}\nFix this and call "
                                f"submit_extraction again "
                                f"(attempt {run.retries} of {budget.max_retries})."
                            ),
                            "is_error": True,
                        }
                    )
                continue

            text, is_error = execute_tool(ctx, name, args)
            result: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": text,
            }
            if is_error:
                result["is_error"] = True
            results.append(result)

        messages.append({"role": "user", "content": results})

        if submitted_this_turn:
            return finish(StopReason.SUBMITTED)


def summarize_tool_use(calls: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Count calls per tool, for the eval harness and for traces."""
    counts: dict[str, int] = {}
    for call in calls:
        name = str(call.get("tool", "unknown"))
        counts[name] = counts.get(name, 0) + 1
    return counts
