"""The five baselines from CLAUDE.md section 6.

Each runner takes a ``Document`` and returns the same ``BaselineOutcome``, so
``run_eval`` scores them identically and the results table is genuinely
comparable. Every one of them ends with the same grounding check and the same
30 deterministic rules -- a baseline that skipped validation would post a
flattering grounding rate for the wrong reason.

The ladder, and what each rung is for:

1. **regex** -- the floor. No model at all. Establishes what pattern matching
   alone buys on the two clause types where it is even plausible, so the LLM
   numbers have something to be better than.
2. **zero-shot truncated** -- the naive approach. One call, no retrieval, the
   document cut to fit. This is what "just put it in the prompt" looks like.
3. **RAG top-k, no rerank** -- the standard approach.
4. **RAG + rerank + agent** -- the system.
5. **full long-context** -- the expensive ceiling. One call, whole document.
   Affordable on this corpus precisely because ``docs/DATA_AUDIT.md`` found
   nothing over 128k tokens.

Baselines 2 and 5 differ only in truncation, and 3 and 4 only in reranking. That
is deliberate: each adjacent pair isolates one variable, so the table reads as an
ablation rather than five unrelated systems.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final, Protocol

from docintel.agent.loop import (
    DEFAULT_MODEL,
    AgentBudget,
    StopReason,
    compute_cost,
    load_prompt,
    prompt_version,
    run_extraction,
)
from docintel.agent.tools import ToolContext, build_tool_schemas, validate_submission
from docintel.schemas import (
    PERPETUAL,
    ClauseExtraction,
    ClauseType,
    Document,
    Evidence,
    RetrievalHit,
    Severity,
    TokenUsage,
)
from docintel.validation.grounding import GroundingReport, check_extractions
from docintel.validation.rules import JurisdictionIndex, apply_rules, load_jurisdictions

__all__ = [
    "BASELINE_NAMES",
    "AgentBaseline",
    "BaselineOutcome",
    "RegexBaseline",
    "RunStatus",
    "SingleCallBaseline",
]

BASELINE_NAMES: Final[tuple[str, ...]] = (
    "1_regex",
    "2_zeroshot_truncated",
    "3_rag_no_rerank",
    "4_rag_rerank_agent",
    "5_full_context",
)

#: Baseline 2's truncation. 8k is the smallest context window worth calling
#: "naive but plausible", and `docs/DATA_AUDIT.md` found 42.5% of contracts
#: exceed it -- so this baseline is genuinely handicapped on nearly half the
#: corpus, which is the point.
TRUNCATE_TOKENS: Final = 8_192
#: Characters per token, rough. Only used to cut text for baseline 2, where
#: being a few hundred characters off changes nothing about the finding.
CHARS_PER_TOKEN: Final = 4


class RunStatus(StrEnum):
    """How a run ended, and therefore whether it may be scored.

    A budget-exhausted run used to be scored as 0/12 present -- total recall
    loss -- which is a harness defect: it silently contaminates every downstream
    metric with a number that measures the turn ceiling rather than the model.
    Only COMPLETED runs enter accuracy scoring; the rest are excluded and
    reported as a completion rate instead.
    """

    COMPLETED = "completed"
    INCOMPLETE_MAX_TURNS = "incomplete_max_turns"
    INCOMPLETE_MAX_COST = "incomplete_max_cost"
    INCOMPLETE_TIMEOUT = "incomplete_timeout"
    INCOMPLETE_MAX_RETRIES = "incomplete_max_retries"
    ERROR = "error"

    @property
    def is_scoreable(self) -> bool:
        return self is RunStatus.COMPLETED


_STOP_TO_STATUS: Final[dict[str, RunStatus]] = {
    StopReason.SUBMITTED.value: RunStatus.COMPLETED,
    StopReason.MAX_TURNS.value: RunStatus.INCOMPLETE_MAX_TURNS,
    StopReason.MAX_COST.value: RunStatus.INCOMPLETE_MAX_COST,
    StopReason.TIMEOUT.value: RunStatus.INCOMPLETE_TIMEOUT,
    StopReason.MAX_RETRIES.value: RunStatus.INCOMPLETE_MAX_RETRIES,
    StopReason.API_ERROR.value: RunStatus.ERROR,
    StopReason.REFUSAL.value: RunStatus.ERROR,
    StopReason.END_TURN.value: RunStatus.ERROR,
}


def status_for(stop_reason: str, clauses: Sequence[Any]) -> RunStatus:
    """Terminal state from a stop reason.

    END_TURN maps to ERROR unless clauses were actually submitted: ending the
    turn without calling submit_extraction is a failure, not a completion.
    """
    status = _STOP_TO_STATUS.get(stop_reason, RunStatus.ERROR)
    if status is RunStatus.ERROR and clauses:
        return RunStatus.COMPLETED
    return status


@dataclass(slots=True)
class BaselineOutcome:
    """One baseline's result for one document, ready to score."""

    document_id: str
    clauses: list[ClauseExtraction]
    grounding: GroundingReport | None = None
    violations: list[Any] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    #: Per-stage wall clock, required by CLAUDE.md section 6. retrieval is time
    #: inside search calls, validation is grounding plus the rules, and model is
    #: what is left -- which is where nearly all of it goes.
    retrieval_ms: float = 0.0
    validation_ms: float = 0.0
    search_calls: int = 0
    turns: int = 0
    #: Whether the first submission parsed without a retry. The schema validity
    #: rate in CLAUDE.md section 6.
    schema_ok: bool = True
    #: Submissions rejected before one was accepted. Reported per arm because
    #: the agent gets 3 and the single-call arm gets 1, and that asymmetry has
    #: to be visible rather than smoothed over.
    retries: int = 0
    stop_reason: str = "submitted"
    status: RunStatus = RunStatus.COMPLETED
    prompt_version: str = ""
    error: str | None = None
    #: Turn-by-turn tool calls, for diagnosing a loop without re-running it.
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_scoreable(self) -> bool:
        return self.status.is_scoreable

    @property
    def has_errors(self) -> bool:
        return any(v.severity is Severity.ERROR for v in self.violations)


class Baseline(Protocol):
    """A scoreable extraction strategy."""

    name: str

    def run(self, document: Document) -> BaselineOutcome: ...


def _finalize(
    document: Document,
    clauses: list[ClauseExtraction],
    jurisdictions: JurisdictionIndex,
    **kwargs: Any,
) -> BaselineOutcome:
    """Apply grounding then the rules, identically for every baseline.

    Order matters and is the same as ``extract.py``: grounding drops fabricated
    spans, which leaves ``present=True`` with no evidence, and that is what the
    ``presence_requires_evidence`` rule rejects.
    """
    started = time.perf_counter()
    grounding = check_extractions(document, clauses)
    violations = apply_rules(
        grounding.repaired_clauses, document=document, jurisdictions=jurisdictions
    )
    kwargs["validation_ms"] = (time.perf_counter() - started) * 1000
    return BaselineOutcome(
        document_id=document.document_id,
        clauses=grounding.repaired_clauses,
        grounding=grounding,
        violations=violations,
        **kwargs,
    )


def _absent(clause_type: ClauseType, confidence: float = 0.5) -> ClauseExtraction:
    return ClauseExtraction(clause_type=clause_type, present=False, confidence=confidence)


# --------------------------------------------------------------------------
# 1. Regex / keyword -- the floor
# --------------------------------------------------------------------------

#: "governed by the laws of the State of Delaware", "construed in accordance
#: with the laws of New York". Deliberately narrow: this baseline exists to
#: establish a floor, and a regex that tries to be clever stops being a floor.
_GOVERNING_LAW: Final = re.compile(
    r"(?:governed\s+by|construed\s+in\s+accordance\s+with|subject\s+to)\s+"
    r"(?:and\s+construed\s+in\s+accordance\s+with\s+)?"
    r"the\s+(?:internal\s+|substantive\s+)?laws?\s+of\s+"
    r"(?:the\s+)?(?:State\s+of\s+|Commonwealth\s+of\s+|State\s+or\s+Commonwealth\s+of\s+)?"
    r"([A-Z][A-Za-z\s]{2,30}?)(?=[,\.\;\)]|\s+without|\s+and\s|\s+excluding)",
    re.IGNORECASE,
)

_MONTHS: Final = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
)

#: "7th day of September, 1999" and "September 7, 1999".
_DATE_PATTERNS: Final = (
    re.compile(rf"(\d{{1,2}})(?:st|nd|rd|th)?\s+day\s+of\s+({_MONTHS}),?\s+(\d{{4}})", re.I),
    re.compile(rf"({_MONTHS})\s+(\d{{1,2}}),?\s+(\d{{4}})", re.I),
)

_MONTH_NUMBERS: Final = {
    month.lower(): index for index, month in enumerate(_MONTHS.split("|"), start=1)
}

_PERPETUAL_HINTS: Final = re.compile(
    r"\b(?:in\s+perpetuity|perpetual|until\s+terminated|remain\s+in\s+(?:full\s+)?"
    r"force\s+and\s+effect\s+until\s+terminated)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class RegexBaseline:
    """Pattern matching only, on governing law and dates. No model.

    Per CLAUDE.md section 6 this covers "Governing Law and dates only". The other
    nine clause types are reported absent, which is the honest floor: a regex has
    no way to decide whether a liability cap is present, and pretending otherwise
    with a keyword search for "cap" would be a worse baseline, not a better one.
    """

    name: str = "1_regex"
    jurisdictions: JurisdictionIndex = field(default_factory=load_jurisdictions)
    _alias_index: dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        # canonical name -> id, plus every alias, all casefolded.
        import yaml

        from docintel.validation.rules import REFERENCE_DIR

        payload = yaml.safe_load((REFERENCE_DIR / "jurisdictions.yaml").read_text(encoding="utf-8"))
        for entry in payload["jurisdictions"]:
            self._alias_index[str(entry["canonical"]).casefold()] = entry["id"]
            for alias in entry.get("aliases") or []:
                self._alias_index[str(alias).casefold()] = entry["id"]

    def _governing_law(self, text: str) -> ClauseExtraction:
        for match in _GOVERNING_LAW.finditer(text):
            candidate = " ".join(match.group(1).split()).casefold()
            jurisdiction = self._alias_index.get(candidate)
            if jurisdiction is None:
                continue
            start, end = match.start(), match.end()
            return ClauseExtraction(
                clause_type=ClauseType.GOVERNING_LAW,
                present=True,
                value=jurisdiction,
                raw_text=text[start:end],
                evidence=[Evidence(quote=text[start:end], char_start=start, char_end=end)],
                # A fixed confidence, because a regex has no calibrated belief.
                # Stated plainly rather than dressed up as a probability.
                confidence=0.60,
            )
        return _absent(ClauseType.GOVERNING_LAW)

    def _first_date(self, text: str, clause_type: ClauseType) -> ClauseExtraction:
        for pattern in _DATE_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            groups = match.groups()
            if groups[0].isdigit():
                day, month_name, year = groups
            else:
                month_name, day, year = groups
            month = _MONTH_NUMBERS.get(month_name.lower())
            if month is None:
                continue
            start, end = match.start(), match.end()
            return ClauseExtraction(
                clause_type=clause_type,
                present=True,
                value=f"{int(year):04d}-{month:02d}-{int(day):02d}",
                raw_text=text[start:end],
                evidence=[Evidence(quote=text[start:end], char_start=start, char_end=end)],
                confidence=0.55,
            )
        return _absent(clause_type)

    def _expiration(self, text: str) -> ClauseExtraction:
        match = _PERPETUAL_HINTS.search(text)
        if match:
            start, end = match.start(), match.end()
            return ClauseExtraction(
                clause_type=ClauseType.EXPIRATION_DATE,
                present=True,
                value=PERPETUAL,
                raw_text=text[start:end],
                evidence=[Evidence(quote=text[start:end], char_start=start, char_end=end)],
                confidence=0.50,
            )
        return _absent(ClauseType.EXPIRATION_DATE)

    def run(self, document: Document) -> BaselineOutcome:
        started = time.perf_counter()
        text = document.text
        handled = {
            ClauseType.GOVERNING_LAW: self._governing_law(text),
            ClauseType.EFFECTIVE_DATE: self._first_date(text, ClauseType.EFFECTIVE_DATE),
            ClauseType.EXPIRATION_DATE: self._expiration(text),
        }
        clauses = [handled.get(clause_type) or _absent(clause_type) for clause_type in ClauseType]
        return _finalize(
            document,
            clauses,
            self.jurisdictions,
            latency_ms=(time.perf_counter() - started) * 1000,
            usage=TokenUsage(),
            turns=0,
        )


# --------------------------------------------------------------------------
# 2 and 5. Single-call baselines
# --------------------------------------------------------------------------


def _submit_only_tool() -> dict[str, Any]:
    """Just ``submit_extraction``, so the model has no way to search or read."""
    return next(tool for tool in build_tool_schemas() if tool["name"] == "submit_extraction")


@dataclass(slots=True)
class SingleCallBaseline:
    """One API call, no tools beyond a forced submission, no retrieval.

    ``truncate_tokens=None`` is baseline 5 (whole document); a value is baseline
    2 (the naive truncated approach). The two share every other detail so the
    difference in the results table is attributable to truncation alone.

    ``tool_choice`` forces the submission tool, which is how structured output is
    obtained without an agent loop: the model cannot answer in prose.
    """

    name: str
    client: Any
    truncate_tokens: int | None = None
    model: str = DEFAULT_MODEL
    max_tokens: int = 8_000
    #: One retry, against the agent's three.
    #:
    #: A DESIGN CHOICE, not a bug fix, and it happens to help the arm expected
    #: to win -- so it is recorded here and in LIMITATIONS.md rather than left
    #: implicit. Zero retries against three was an unfair handicap; matching all
    #: three would make "single call" a misnomer. One is the compromise, and the
    #: retry-used counter shows how often it actually mattered.
    max_retries: int = 1
    jurisdictions: JurisdictionIndex = field(default_factory=load_jurisdictions)

    def _prompt(self, document: Document) -> str:
        text = document.text
        note = ""
        if self.truncate_tokens is not None:
            limit = self.truncate_tokens * CHARS_PER_TOKEN
            if len(text) > limit:
                text = text[:limit]
                note = (
                    f"\n\n[The contract was truncated to the first {limit:,} "
                    "characters and continues beyond this point.]"
                )
        return (
            f"Contract `{document.document_id}`:\n\n"
            f"<contract>\n{text}\n</contract>{note}\n\n"
            "Extract all 12 clause types and call submit_extraction once. "
            "Character offsets are counted from the start of the contract text "
            "shown above."
        )

    def run(self, document: Document) -> BaselineOutcome:
        started = time.perf_counter()
        system = load_prompt("extract_v1")
        tools = [_submit_only_tool()]
        version = prompt_version(system, str([t["name"] for t in tools]))

        messages: list[dict[str, Any]] = [{"role": "user", "content": self._prompt(document)}]
        token_usage = TokenUsage()
        clauses: list[ClauseExtraction] = []
        schema_ok = True
        error: str | None = None
        retries = 0

        # One retry, against the agent's three. Recorded per run so the
        # asymmetry is visible in the results rather than assumed away.
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    cache_control={"type": "ephemeral"},
                    system=[{"type": "text", "text": system}],
                    tools=tools,
                    tool_choice={"type": "tool", "name": "submit_extraction"},
                    messages=messages,
                )
            except Exception as exc:
                return BaselineOutcome(
                    document_id=document.document_id,
                    clauses=[],
                    usage=token_usage,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    schema_ok=False,
                    retries=retries,
                    stop_reason=StopReason.API_ERROR,
                    status=RunStatus.ERROR,
                    error=f"{type(exc).__name__}: {exc}",
                )

            usage = getattr(response, "usage", None)
            token_usage = token_usage + TokenUsage(
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
                cost_usd=compute_cost(self.model, usage) if usage else 0.0,
            )

            content = list(getattr(response, "content", []) or [])
            blocks = [b for b in content if getattr(b, "type", None) == "tool_use"]
            if not blocks:
                schema_ok = False
                error = "no submit_extraction call in the response"
                break

            try:
                clauses = validate_submission(getattr(blocks[0], "input", {}) or {})
                error = None
                break
            except Exception as exc:
                error = str(exc)
                schema_ok = False
                if attempt >= self.max_retries:
                    break
                retries += 1
                # Hand back the specific complaint, the same way the agent loop
                # does, rather than failing the whole run on a fixable mistake.
                messages = [
                    *messages,
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": getattr(blocks[0], "id", ""),
                                "is_error": True,
                                "content": (
                                    f"Rejected: {exc}\n"
                                    "Fix only what is named and call "
                                    "submit_extraction again."
                                ),
                            }
                        ],
                    },
                ]

        outcome = _finalize(
            document,
            clauses,
            self.jurisdictions,
            usage=token_usage,
            latency_ms=(time.perf_counter() - started) * 1000,
            turns=1,
            schema_ok=schema_ok and retries == 0,
            retries=retries,
            prompt_version=version,
            error=error,
        )
        accepted = error is None and bool(clauses)
        outcome.stop_reason = "submitted" if accepted else "schema_error"
        outcome.status = RunStatus.COMPLETED if accepted else RunStatus.ERROR
        return outcome


# --------------------------------------------------------------------------
# 3 and 4. Agent baselines
# --------------------------------------------------------------------------


@dataclass(slots=True)
class AgentBaseline:
    """The tool-calling loop over retrieval. ``rerank`` is the only difference
    between baselines 3 and 4."""

    name: str
    client: Any
    search: Callable[[str, str, int], Sequence[RetrievalHit]]
    budget: AgentBudget = field(default_factory=AgentBudget)
    model: str = DEFAULT_MODEL
    prompt_name: str = "extract_v1"
    jurisdictions: JurisdictionIndex = field(default_factory=load_jurisdictions)

    def run(self, document: Document) -> BaselineOutcome:
        started = time.perf_counter()
        retrieval_ms = 0.0
        search_calls = 0

        def search(query: str, top_k: int) -> Sequence[RetrievalHit]:
            nonlocal retrieval_ms, search_calls
            call_started = time.perf_counter()
            try:
                return self.search(document.document_id, query, top_k)
            finally:
                retrieval_ms += (time.perf_counter() - call_started) * 1000
                search_calls += 1

        ctx = ToolContext(document=document, search=search, jurisdictions=self.jurisdictions)
        agent = run_extraction(
            self.client,
            ctx,
            budget=self.budget,
            model=self.model,
            prompt_name=self.prompt_name,
        )
        return _finalize(
            document,
            agent.clauses,
            self.jurisdictions,
            usage=agent.usage,
            latency_ms=(time.perf_counter() - started) * 1000,
            retrieval_ms=retrieval_ms,
            search_calls=search_calls,
            turns=agent.turns,
            schema_ok=agent.retries == 0,
            retries=agent.retries,
            stop_reason=str(agent.stop_reason),
            status=status_for(str(agent.stop_reason), agent.clauses),
            prompt_version=agent.prompt_version,
            error=agent.error,
            tool_calls=list(ctx.calls),
        )
