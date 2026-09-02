"""The extraction orchestrator: retrieval, agent, validation, review routing.

This is the seam where the pieces meet, and the ordering is the whole point.

**Grounding runs before the rules, not after.** Grounding drops fabricated
evidence spans, which leaves a clause with ``present=True`` and an empty
evidence list -- and *that* is what the ``presence_requires_evidence`` rule
rejects. Run the rules first and a fabricating extraction passes them, because
at that moment it still has evidence attached. The gate only bites in this
order.

**Validation runs on the repaired clauses, and those are what get returned.**
An extraction whose quote was real but whose offsets were three characters off
is corrected rather than discarded: models copy text reliably and count
characters badly, and rejecting the whole clause over arithmetic would report
hallucination where none happened.

**The agent's own tool sees the same code.** ``validate_extraction`` calls
``check_extractions`` and ``apply_rules`` directly, so what the agent is told
during the loop is exactly what happens to it afterwards. A separate, more
permissive shadow copy of the rules is the obvious way for this to rot.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from docintel.agent.loop import AgentBudget, AgentOutcome, run_extraction
from docintel.agent.tools import ToolContext
from docintel.schemas import (
    Document,
    ExtractionResult,
    RetrievalHit,
    ReviewItem,
)
from docintel.validation.grounding import GroundingReport, check_extractions
from docintel.validation.rules import (
    JurisdictionIndex,
    apply_rules,
    load_jurisdictions,
    needs_review,
)

__all__ = ["ExtractionOutcome", "Extractor", "extract_document"]

#: A search function: (query, top_k) -> hits. Injected rather than a retriever
#: object so an ablation can pass a single-arm or no-rerank retriever, and so the
#: orchestrator is testable without a database.
SearchFn = Callable[[str, int], Sequence[RetrievalHit]]


@dataclass(frozen=True, slots=True)
class ExtractionOutcome:
    """The result plus the diagnostics that produced it.

    ``ExtractionResult`` is the API-facing object; the grounding report and the
    agent outcome are kept beside it because the eval harness needs the
    grounding violation rate and the tool-call trace, and recomputing either
    afterwards is impossible.
    """

    result: ExtractionResult
    grounding: GroundingReport
    agent: AgentOutcome
    review: ReviewItem | None = None


@dataclass(slots=True)
class Extractor:
    """Runs extraction over documents with a shared client and retriever.

    Holds the jurisdiction index so it is loaded once rather than per document --
    it is a 100-entry YAML parse, which is cheap but not free across a 150-case
    eval run.
    """

    client: Any
    budget: AgentBudget = field(default_factory=AgentBudget)
    model: str | None = None
    prompt_name: str = "extract_v1"
    jurisdictions: JurisdictionIndex = field(default_factory=load_jurisdictions)

    def extract(
        self,
        document: Document,
        search: SearchFn | None = None,
    ) -> ExtractionOutcome:
        return extract_document(
            client=self.client,
            document=document,
            search=search,
            budget=self.budget,
            model=self.model,
            prompt_name=self.prompt_name,
            jurisdictions=self.jurisdictions,
        )


def extract_document(
    client: Any,
    document: Document,
    search: SearchFn | None = None,
    budget: AgentBudget | None = None,
    model: str | None = None,
    prompt_name: str = "extract_v1",
    jurisdictions: JurisdictionIndex | None = None,
) -> ExtractionOutcome:
    """Extract all clause types from one document."""
    index = jurisdictions if jurisdictions is not None else load_jurisdictions()
    ctx = ToolContext(document=document, search=search, jurisdictions=index)

    agent_started = time.perf_counter()
    kwargs: dict[str, Any] = {"budget": budget, "prompt_name": prompt_name}
    if model is not None:
        kwargs["model"] = model
    agent = run_extraction(client, ctx, **kwargs)
    agent_ms = (time.perf_counter() - agent_started) * 1000

    validation_started = time.perf_counter()

    # Order matters. Grounding first, so fabricated spans are gone before the
    # rules look for missing evidence.
    grounding = check_extractions(document, agent.clauses)
    violations = apply_rules(grounding.repaired_clauses, document=document, jurisdictions=index)
    validation_ms = (time.perf_counter() - validation_started) * 1000

    result = ExtractionResult(
        document_id=document.document_id,
        clauses=grounding.repaired_clauses,
        violations=violations,
        prompt_version=agent.prompt_version,
        model=agent.model,
        usage=agent.usage,
        latency_ms={
            "agent": round(agent_ms, 1),
            "validation": round(validation_ms, 1),
            "total": round(agent_ms + validation_ms, 1),
        },
        turns_used=agent.turns,
        stopped_on_budget=agent.stopped_on_budget,
    )

    flagged, reasons = needs_review(result)

    # An ungrounded quote is a fabrication, so it routes to review even when no
    # rule fired -- a clause can be dropped to zero evidence and still look
    # internally consistent if the model also flipped `present` to false.
    if grounding.ungrounded:
        flagged = True
        reasons = [
            *reasons,
            f"{len(grounding.ungrounded)} ungrounded evidence span(s)",
        ]

    if agent.error:
        flagged = True
        reasons = [*reasons, f"agent: {agent.error}"]

    result = result.model_copy(update={"needs_review": flagged})

    return ExtractionOutcome(
        result=result,
        grounding=grounding,
        agent=agent,
        review=(
            ReviewItem(document_id=document.document_id, reasons=reasons, result=result)
            if flagged
            else None
        ),
    )
